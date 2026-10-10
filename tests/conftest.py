import os
import sys
from pathlib import Path

import pytest

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
