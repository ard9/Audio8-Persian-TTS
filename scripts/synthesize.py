#!/usr/bin/env python3
"""Synthesize Persian or Central Kurdish (Sorani) speech with a fine-tuned Audio8 model.

Examples:
    # exported folder (from the notebook's export cell) or a Hugging Face model ID
    python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Persian --text "سلام دنیا" --output hello.wav

    # base model + a raw Trainer checkpoint (e.g. downloaded from your private training repo)
    python scripts/synthesize.py --model /path/to/base_model --checkpoint /path/to/checkpoint-1200 \
        --text "سلام دنیا" --output hello.wav

    # voice cloning from a 5-10 s reference clip and its exact transcript
    python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Persian --text "سلام" \
        --reference-audio me.wav --reference-text "متن دقیق صدای مرجع" --output clone.wav

    # Central Kurdish model: add --lang ckb so the text is normalized with the Sorani rules
    python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Kurdish-Sorani --lang ckb \
        --text "سڵاو، ئەمڕۆ کەشوهەوا زۆر خۆشە." --output silav.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fa_tools as fa  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local model folder or Hugging Face model ID.")
    parser.add_argument("--checkpoint", help="Optional Trainer checkpoint folder with model.safetensors.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-text")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-frames", type=int, default=1200, help="~21.5 frames per second of audio.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lang", choices=sorted(fa.NORMALIZERS), default="fa",
                        help="Text normalization: fa = Persian (default), ckb = Central Kurdish (Sorani).")
    args = parser.parse_args()

    if bool(args.reference_audio) != bool(args.reference_text):
        parser.error("--reference-audio and --reference-text must be given together")

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(args.model, token=fa.hf_token()))

    import soundfile as sf

    model, processor = fa.load_for_inference(model_dir, Path(args.checkpoint) if args.checkpoint else None)
    audio, rate, finished = fa.synthesize(
        model,
        processor,
        args.text,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        max_frames=args.max_frames,
        temperature=args.temperature,
        seed=args.seed,
        lang=args.lang,
    )
    sf.write(args.output, audio, rate)
    status = "" if finished else " (warning: generation hit --max-frames before the end of speech)"
    print(f"wrote {args.output}: {len(audio) / rate:.1f}s at {rate} Hz{status}")
    print(f"normalized input: {fa.normalize_text(args.text, args.lang)}")


if __name__ == "__main__":
    main()
