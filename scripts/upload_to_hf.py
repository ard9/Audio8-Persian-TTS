#!/usr/bin/env python3
"""Upload an exported model folder to the Hugging Face Hub with its model card.

1. In the notebook, run the "Export the final model" cell. It writes a
   standalone folder such as  MyDrive/audio8_fa/runs/<run>/export/checkpoint-N
2. Fill in the TODOs in docs/MODEL_CARD.md (results, training details).
3. Log in once:  huggingface-cli login   (or set HF_TOKEN)
4. Run:
    python scripts/upload_to_hf.py \
        --export-dir /content/drive/MyDrive/audio8_fa/runs/audio8_fa_v1/export/checkpoint-N \
        --repo-id ArminRahimi/Audio8-TTS-Persian \
        --samples-dir samples/
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("config.json", "model.safetensors", "codec.pth", "tokenizer.json", "modeling_arktts.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True, help="e.g. ArminRahimi/Audio8-TTS-Persian")
    parser.add_argument("--model-card", type=Path, default=ROOT / "docs" / "MODEL_CARD.md")
    parser.add_argument("--samples-dir", type=Path, help="Optional folder of demo .wav files.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--allow-todo", action="store_true", help="Upload even if the card still has TODOs.")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (args.export_dir / name).is_file()]
    if missing:
        parser.error(f"{args.export_dir} is not an exported model folder; missing {missing}")
    card = args.model_card.read_text(encoding="utf-8")
    todos = re.findall(r"TODO[^\n]*", card)
    if todos and not args.allow_todo:
        preview = "\n  ".join(todos[:8])
        parser.error(f"model card still has {len(todos)} TODO(s):\n  {preview}\nFill them in or pass --allow-todo.")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(card, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=args.repo_id)
        notice = ROOT / "NOTICE"
        if notice.is_file():
            staged = Path(tmp) / "NOTICE"
            shutil.copyfile(notice, staged)
            api.upload_file(path_or_fileobj=str(staged), path_in_repo="NOTICE", repo_id=args.repo_id)
    api.upload_folder(
        folder_path=str(args.export_dir),
        repo_id=args.repo_id,
        ignore_patterns=["*.partial", "README.md", ".*", "*.jpeg", "*.jpg"],
        commit_message="Upload Persian fine-tuned Audio8 TTS weights",
    )
    if args.samples_dir and args.samples_dir.is_dir():
        api.upload_folder(
            folder_path=str(args.samples_dir),
            path_in_repo="samples",
            repo_id=args.repo_id,
            allow_patterns=["*.wav", "*.mp3", "*.md"],
            commit_message="Add audio samples",
        )
    print(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
