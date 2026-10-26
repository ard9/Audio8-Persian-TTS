"""Central Kurdish (Sorani) path: csv+wav dataset, Sorani text rules, speaker-aware references,
private data repo, and training with Sorani test sentences."""

import json
from pathlib import Path

import fa_tools as fa
from fake_hub import FakeHfApi


def _speaker_of_codes(work: Path) -> dict[str, str]:
    mapping = {}
    for manifest in (work / "shards").glob("*.jsonl"):
        for line in manifest.read_text("utf-8").splitlines():
            row = json.loads(line)
            mapping[row["target_codes"]] = row["speaker"]
    return mapping


def test_kurdish_pipeline(fake_env, audio8_repo, tmp_path):
    base, work, cache = fake_env
    api = FakeHfApi(tmp_path / "hub")
    api.create_repo("me/ckb-codes", repo_type="dataset", private=False)  # existed as public before
    hub = fa.HubStore("me/ckb-codes", "me/ckb-training", api=api, data_private=True, data_card=fa.DATA_CARD_CKB)
    assert api.private[("dataset", "me/ckb-codes")] is True
    plan = {"ckb_tts4all": 3}

    fa.prepare_all(base, work, cache, plan, fa.PrepOptions(max_seconds=10, only_high_quality=False), hub=hub)
    assert "cc-by-nc-nd-4.0" in (tmp_path / "hub" / "dataset" / "me/ckb-codes" / "README.md").read_text("utf-8")
    assert len(list((cache / "shards").glob("ckb_tts4all__part_*.tar"))) == 3

    rows = [json.loads(l) for m in sorted((work / "shards").glob("*.jsonl")) for l in m.read_text("utf-8").splitlines()]
    assert {r["speaker"] for r in rows} == {"fatih", "shahen", "giganet"}
    assert all(not any(ch.isdigit() for ch in r["text"]) for r in rows)  # numbers spelled out in Sorani
    assert any("ژمارە" in r["text"] for r in rows)
    refs = fa.load_refs(cache)
    assert {r.get("speaker") for r in refs} >= {"fatih"}

    manifest_dir = tmp_path / "run" / "manifests"
    train_jsonl, eval_jsonl = fa.build_manifests(
        base, audio8_repo, work, cache, plan, max_length=600, eval_size=4, default_voice="auto",
        manifest_dir=manifest_dir, hub=hub, run="ckb1")
    voices = json.loads((manifest_dir / "voices.json").read_text("utf-8"))
    assert voices["default"] == "ckb_tts4all:fatih"  # the speaker with the most audio

    speaker_of = _speaker_of_codes(work)
    items = [json.loads(l) for p in (train_jsonl, eval_jsonl) for l in p.read_text("utf-8").splitlines()]
    with_ref = [i for i in items if "reference_codes" in i]
    assert with_ref
    for item in with_ref:  # references always come from the same speaker
        assert speaker_of[item["reference_codes"]] == speaker_of[item["target_codes"]]
    for item in items:  # only the default voice is trained without a reference
        if "reference_codes" not in item:
            assert speaker_of[item["target_codes"]] == "fatih"

    clone = fa.pick_clone_reference(refs, manifest_dir)
    if clone is not None:
        assert clone.get("speaker") != "fatih"

    cfg = fa.TrainConfig(
        base_dir=str(base), repo_dir=str(audio8_repo), train_jsonl=str(train_jsonl), eval_jsonl=str(eval_jsonl),
        output_dir=str(tmp_path / "run" / "checkpoints"), samples_dir=str(tmp_path / "run" / "samples"),
        batch_size=2, grad_accum=2, epochs=1, save_steps=3, keep_checkpoints=1, max_length=600,
        test_sentences=("سڵاو، ئەمڕۆ ٣ کتێبم خوێندەوە.",), lang="ckb", clone_reference=clone,
        logging_steps=1, num_workers=0, learning_rate=1e-3, hub=hub, hub_run="ckb1",
    )
    assert fa.train(cfg) is not None
    last = json.loads((tmp_path / "run" / "samples" / "index.jsonl").read_text("utf-8").splitlines()[-1])
    assert last["samples"] and last["samples"][0]["finished"] is not None
