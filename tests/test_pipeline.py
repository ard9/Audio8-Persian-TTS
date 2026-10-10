"""End-to-end test of the notebook pipeline on CPU with a tiny stand-in model.

The fake model (tests/fake_audio8.py) mirrors the Audio8 interfaces the
notebook relies on, so this covers data preparation, Drive restore, manifests,
training, per-checkpoint samples, rotation, resume after a broken checkpoint,
inference and export. It does not test audio quality.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import fa_tools as fa
from fake_audio8 import FakeConfig, FakeModel, FakeProcessor


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    base, work, drive = tmp_path / "base", tmp_path / "work", tmp_path / "drive"
    for path in (base, work, drive):
        path.mkdir()
    torch.manual_seed(0)
    FakeModel(FakeConfig()).save_pretrained(base)
    monkeypatch.setattr(fa, "load_model", lambda base_dir, dtype=None: FakeModel.from_pretrained(str(base)).to(dtype or torch.float32))
    monkeypatch.setattr(fa, "load_processor", lambda base_dir: FakeProcessor())
    monkeypatch.setattr(fa, "load_config", lambda base_dir: FakeConfig())

    def fake_download(repo, filename, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(sum(map(ord, filename)))
        names, texts, audios, quality = [], [], [], []
        for i in range(24):
            seconds = float(rng.uniform(0.5, 12))
            names.append(f"{i}-{i % 3}.wav")
            texts.append(f"جمله شماره {i} برای آزمایش" if i % 7 else "")
            audios.append((rng.standard_normal(int(seconds * 44100)) * 0.1).astype(np.float32))
            quality.append("HIGH" if i % 5 else "MIDDLE")
        table = pa.table({
            "file name": names,
            "transcript": texts,
            "duration": [len(a) / 44100 for a in audios],
            "match quality": quality,
            "audio": pa.array([a.tolist() for a in audios], type=pa.list_(pa.float32())),
            "samplerate": [44100.0] * len(names),
        })
        path = target_dir / Path(filename).name
        pq.write_table(table, path)
        return path

    monkeypatch.setattr(fa, "_download", fake_download)
    return base, work, drive


def test_full_pipeline(fake_env, audio8_repo):
    base, work, drive = fake_env
    plan = {"mana": 2, "gptinformal": 1}
    options = fa.PrepOptions(max_seconds=10)

    fa.prepare_all(base, work, drive, plan, options)
    assert len(list((drive / "shards").glob("*.tar"))) == 3
    refs = json.loads((drive / "refs.json").read_text("utf-8"))
    assert {r["dataset"] for r in refs} == {"mana", "gptinformal"}

    # New Colab session: local disk wiped, Drive kept -> restore without re-encoding.
    shutil.rmtree(work)
    work.mkdir()
    fa.prepare_all(base, work, drive, plan, options)
    train_jsonl, eval_jsonl = fa.build_manifests(base, audio8_repo, work, drive, plan, max_length=600, eval_size=4)
    rows = [json.loads(line) for line in train_jsonl.read_text("utf-8").splitlines()]
    assert rows and any("reference_codes" in r for r in rows)
    first = train_jsonl.read_text("utf-8")
    fa.build_manifests(base, audio8_repo, work, drive, plan, max_length=600, eval_size=4)
    assert train_jsonl.read_text("utf-8") == first  # reused, same order

    clone = next(r for r in refs if r["dataset"] == "gptinformal")
    cfg = fa.TrainConfig(
        base_dir=str(base), repo_dir=str(audio8_repo), train_jsonl=str(train_jsonl), eval_jsonl=str(eval_jsonl),
        output_dir=str(drive / "checkpoints"), samples_dir=str(drive / "samples"), drive_folder_for_trash=None,
        batch_size=2, grad_accum=2, epochs=1, save_steps=3, keep_checkpoints=2, max_length=600,
        test_sentences=("سلام دنیا", "این یک آزمایش است"), clone_reference=clone, logging_steps=1,
        num_workers=0, learning_rate=1e-3,
    )
    fa.train(cfg)
    steps = [s for s, _ in fa.list_checkpoints(cfg.output_dir)]
    assert len(steps) == 2

    # Simulate a disconnect during a save: an incomplete newer checkpoint must be skipped.
    broken = Path(cfg.output_dir) / "checkpoint-99999"
    broken.mkdir()
    (broken / "trainer_state.json").write_text("{")
    cfg.epochs = 2
    fa.train(cfg)
    steps_after = [s for s, _ in fa.list_checkpoints(cfg.output_dir)]
    assert 99999 not in steps_after and max(steps_after) == 2 * max(steps)

    index = (drive / "samples" / "index.jsonl").read_text("utf-8").splitlines()
    assert len(index) >= 4
    last = json.loads(index[-1])
    assert any(s["clone"] for s in last["samples"])

    ckpt = fa.resolve_checkpoint(cfg.output_dir, "latest")
    model, processor = fa.load_for_inference(base, ckpt)
    audio, rate, finished = fa.synthesize(model, processor, "سلام. " * 40)
    assert rate == 44100 and audio.size > 0 and finished

    exported = fa.export_model(base, ckpt, drive / "export")
    reloaded = FakeModel.from_pretrained(str(exported))
    from safetensors.torch import load_file

    tuned = load_file(str(ckpt / "model.safetensors"))
    key = "layers.0.mlp.weight"
    assert torch.allclose(reloaded.state_dict()[key].float(), tuned[key].to(torch.bfloat16).float())
