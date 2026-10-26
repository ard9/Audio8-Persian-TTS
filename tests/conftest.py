import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def audio8_repo() -> Path:
    """Path to a clone of Edge0-AI/Audio8_TTS (set AUDIO8_REPO)."""
    path = Path(os.environ.get("AUDIO8_REPO", ROOT / ".cache" / "Audio8_TTS"))
    if not (path / "audio8_tts_sft.py").is_file():
        pytest.skip(f"Audio8_TTS repo not found at {path}; set AUDIO8_REPO")
    return path


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    import fa_tools as fa
    from fake_audio8 import FakeConfig, FakeModel, FakeProcessor

    base, work, cache = tmp_path / "base", tmp_path / "work", tmp_path / "cache"
    for path in (base, work, cache):
        path.mkdir()
    torch.manual_seed(0)
    FakeModel(FakeConfig()).save_pretrained(base)
    monkeypatch.setattr(fa, "load_model", lambda base_dir, dtype=None: FakeModel.from_pretrained(str(base)).to(dtype or torch.float32))
    monkeypatch.setattr(fa, "load_processor", lambda base_dir: FakeProcessor())
    monkeypatch.setattr(fa, "load_config", lambda base_dir: FakeConfig())

    def fake_download(repo, filename, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(sum(map(ord, filename)))
        if filename.endswith("metadata.csv"):  # central-kurdish-tts4all layout: metadata.csv + wavs/<speaker>/
            lines = ["file_name,text,speaker"]
            for i in range(240):
                speaker = ("fatih", "fatih", "shahen", "giganet")[i % 4]
                text = f"ئەمە ڕستەی ژمارە {i} ە بۆ تاقیکردنەوە" if i % 9 else ""
                lines.append(f'wavs/{speaker}/{speaker}_{i:03d}.wav,"{text}",{speaker}')
            path = target_dir / "metadata.csv"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return path
        if filename.endswith(".wav"):
            import soundfile as sf

            seconds = float(rng.uniform(0.5, 12))
            path = target_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, (rng.standard_normal(int(seconds * 24000)) * 0.1).astype(np.float32), 24000)
            return path
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
    return base, work, cache


