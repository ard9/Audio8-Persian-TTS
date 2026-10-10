"""Persian (Farsi) fine-tuning helpers for Audio8 TTS on Google Colab.

Everything that needs to survive a Colab disconnect lives on Google Drive:
prepared codec data (one .tar per dataset shard), the train/eval manifests,
Trainer checkpoints and the audio samples written after every checkpoint.
Each step checks Drive first, so re-running the notebook resumes instead of
starting over.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import unicodedata
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

CODEC_SAMPLE_RATE = 44100
FRAMES_PER_SECOND = CODEC_SAMPLE_RATE / 2048  # ~21.5 codec frames per second
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

DATASETS: dict[str, dict[str, Any]] = {
    "mana": {
        "repo": "MahtaFetrat/Mana-TTS",
        "files": [f"dataset/dataset_part_{i:03d}.parquet" for i in range(1, 78)],
    },
    "gptinformal": {
        "repo": "MahtaFetrat/GPTInformal-Persian",
        "files": [f"dataset/dataset_part_{i}.parquet" for i in range(1, 6)],
    },
}


def log(message: str) -> None:
    print(f"[fa] {message}", flush=True)


# ---------------------------------------------------------------------------
# Persian text normalization (applied identically at training and test time)
# ---------------------------------------------------------------------------

_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "«": '"',
        "»": '"',
        "٬": ",",
        "٫": ".",
        "–": "-",
        "—": "-",
    }
)
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS_RE = re.compile("[ً-ٰٟۖ-ۭ]")
# The repo's clean_text() drops Unicode "C*" characters, which includes ZWNJ
# (U+200C). We remove it explicitly so training and inference always agree.
_INVISIBLE_RE = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _int_to_words(value: str) -> str:
    try:
        from num2fawords import words
    except ImportError:  # keep digits if the package is missing
        return value
    text = words(int(value))
    if text.startswith("یک هزار"):
        text = text[len("یک ") :]
    return text


def _number_to_words(match: re.Match[str]) -> str:
    token = match.group()
    if "," in token:
        whole, _, rest = token.partition(",")
        if len(rest) == 3:  # thousands separator: 1,234
            return _int_to_words(whole + rest)
        return f"{_int_to_words(whole)} ، {_int_to_words(rest)}"
    if "." in token:
        whole, _, frac = token.partition(".")
        return f"{_int_to_words(whole)} ممیز {_int_to_words(frac)}"
    return _int_to_words(token)


def normalize_fa(text: Any) -> str:
    """Light, deterministic Persian normalization for TTS input."""
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.translate(_CHAR_MAP).translate(_DIGITS)
    text = _DIACRITICS_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace("ـ", "")  # tatweel
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}\b)", "", text)
    text = text.replace("%", " درصد ").replace("٪", " درصد ")
    text = _NUMBER_RE.sub(lambda m: f" {_number_to_words(m)} ", text)
    text = re.sub(r"\s+([،,.!?؟:;؛])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, max_chars: int = 150) -> list[str]:
    """Split long input into chunks the model handles well (README: <=150 chars)."""
    text = normalize_fa(text)
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = re.split(r"(?<=[.!?؟؛])\s+", text)
    chunks: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = max(piece.rfind("،", 0, max_chars), piece.rfind(",", 0, max_chars))
            if cut < max_chars // 3:
                cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(piece[: cut + 1].strip())
            piece = piece[cut + 1 :].strip()
        if piece:
            if chunks and len(chunks[-1]) + len(piece) + 1 <= max_chars:
                chunks[-1] = f"{chunks[-1]} {piece}"
            else:
                chunks.append(piece)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def gpu_profile() -> dict[str, Any]:
    """Pick precision and batch settings from the attached GPU."""
    import torch

    if not torch.cuda.is_available():
        return {"name": "CPU", "memory_gb": 0, "precision": "fp32", "batch_size": 1, "grad_accum": 16}
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1024**3
    native_bf16 = props.major >= 8  # T4/V100 only emulate bf16
    if memory_gb >= 38:
        batch_size, grad_accum = 8, 2
    elif memory_gb >= 22:
        batch_size, grad_accum = 4, 4
    else:
        batch_size, grad_accum = 2, 8
    return {
        "name": props.name,
        "memory_gb": round(memory_gb, 1),
        "precision": "bf16" if native_bf16 else "fp16",
        "batch_size": batch_size,
        "grad_accum": grad_accum,
    }


def inference_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        return torch.bfloat16
    return torch.float32


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def add_repo_to_path(repo_dir: str | Path) -> None:
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def load_model(base_dir: str | Path, dtype=None):
    """Load the Audio8 model (remote code) from the downloaded base checkpoint."""
    import torch
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        str(base_dir), trust_remote_code=True, dtype=dtype or torch.float32
    )


def load_processor(base_dir: str | Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(base_dir), trust_remote_code=True)


def load_config(base_dir: str | Path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(base_dir), trust_remote_code=True)


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:  # Colab secrets panel (key icon) -> HF_TOKEN
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reading the Mana-TTS / GPTInformal parquet files
# ---------------------------------------------------------------------------


def _norm_col(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def iter_parquet_rows(path: Path, batch_rows: int = 32) -> Iterator[dict[str, Any]]:
    """Yield rows with keys: file_name, transcript, duration, match_quality, audio, samplerate."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    columns = {_norm_col(field.name): field.name for field in pf.schema_arrow}
    aliases = {
        "file_name": ("file_name", "filename", "id", "path"),
        "transcript": ("transcript", "text", "sentence", "normalized_text"),
        "duration": ("duration",),
        "match_quality": ("match_quality",),
        "audio": ("audio",),
        "samplerate": ("samplerate", "sample_rate", "sampling_rate", "sr"),
    }
    picked: dict[str, str] = {}
    for key, options in aliases.items():
        for option in options:
            if option in columns:
                picked[key] = columns[option]
                break
    for required in ("transcript", "audio"):
        if required not in picked:
            raise ValueError(f"{path.name}: missing column {required!r}; found {list(columns)}")

    for batch in pf.iter_batches(batch_size=batch_rows, columns=list(picked.values())):
        cols = {key: batch.column(batch.schema.get_field_index(name)) for key, name in picked.items()}
        audio_col = cols["audio"]
        audio_type = audio_col.type
        is_list = pa.types.is_list(audio_type) or pa.types.is_large_list(audio_type)
        if is_list:
            offsets = audio_col.offsets.to_numpy()
            values = audio_col.values.to_numpy(zero_copy_only=False)
        for i in range(batch.num_rows):
            row: dict[str, Any] = {}
            for key, col in cols.items():
                if key != "audio":
                    row[key] = col[i].as_py()
            if not audio_col[i].is_valid:
                continue
            if is_list:
                row["audio"] = np.array(values[offsets[i] : offsets[i + 1]], dtype=np.float32)
            else:
                row["audio"], row["samplerate"] = _decode_audio_cell(audio_col[i].as_py())
            yield row


def _decode_audio_cell(cell: Any) -> tuple[np.ndarray, int]:
    import io

    import soundfile as sf

    data = cell.get("bytes") if isinstance(cell, dict) else cell
    if not data:
        raise ValueError("audio cell has no bytes")
    array, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return array.mean(axis=1), int(rate)


def _to_codec_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if int(rate) != CODEC_SAMPLE_RATE:
        import torch
        from torchaudio.functional import resample

        audio = resample(torch.from_numpy(audio), int(rate), CODEC_SAMPLE_RATE).numpy()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio * (0.95 / peak)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _safe_id(*parts: str) -> str:
    raw = "_".join(str(p) for p in parts if p)
    raw = re.sub(r"\.(wav|mp3|flac)$", "", raw, flags=re.IGNORECASE)
    return re.sub(r"[^\w.-]+", "_", raw).strip("._") or "item"


# ---------------------------------------------------------------------------
# Codec encoding (data preparation)
# ---------------------------------------------------------------------------


class CodecEncoder:
    """Turns 44.1 kHz waveforms into [10, T] codec indices, like audio8_tts_prepare.py."""

    def __init__(self, base_dir: str | Path, device: str | None = None):
        import torch

        self.device = torch.device(device or device_name())
        self.dtype = inference_dtype() if self.device.type == "cuda" else torch.float32
        self.model = load_model(base_dir, dtype=self.dtype).eval()
        self.model.load_codec(device=self.device, dtype=self.dtype)
        self.num_codebooks = int(self.model.config.num_codebooks)

    def _encode(self, waveforms: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        lengths = torch.tensor([w.size for w in waveforms], dtype=torch.long)
        values = torch.zeros((len(waveforms), 1, int(lengths.max())), dtype=torch.float32)
        for index, wave in enumerate(waveforms):
            values[index, 0, : wave.size] = torch.from_numpy(wave)
        with torch.inference_mode():
            codes, code_lengths = self.model.encode_audio(
                values.to(self.device), lengths.to(self.device)
            )
        result = []
        for index in range(len(waveforms)):
            length = int(code_lengths[index])
            array = codes[index, :, :length].cpu().numpy()
            if array.shape[0] != self.num_codebooks or array.shape[1] == 0:
                raise RuntimeError(f"codec returned shape {array.shape}")
            result.append(array.astype(np.int16))
        return result

    def encode(self, waveforms: list[np.ndarray]) -> list[np.ndarray | None]:
        import torch

        try:
            return list(self._encode(waveforms))
        except Exception as exc:  # fall back to one clip at a time
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if len(waveforms) == 1:
                log(f"encode failed: {type(exc).__name__}: {exc}")
                return [None]
            out: list[np.ndarray | None] = []
            for wave in waveforms:
                try:
                    out.extend(self._encode([wave]))
                except Exception as single_exc:
                    log(f"encode failed: {type(single_exc).__name__}: {single_exc}")
                    out.append(None)
            return out

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class PrepOptions:
    min_seconds: float = 1.0
    max_seconds: float = 20.0
    only_high_quality: bool = True
    batch_seconds: float = 160.0
    refs_per_dataset: int = 2


def _shard_name(dataset: str, filename: str) -> str:
    return f"{dataset}__{Path(filename).stem}"


def _download(repo: str, filename: str, target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=hf_token(),
    )
    return Path(path)


def _copy_to_drive(local: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    partial_path = final.with_name(final.name + ".partial")
    shutil.copyfile(local, partial_path)
    os.replace(partial_path, final)


def _extract_shard(tar_path: Path, work_dir: Path) -> None:
    with tarfile.open(tar_path, "r") as handle:
        members = handle.getmembers()
        for member in members:
            target = (work_dir / member.name).resolve()
            if not str(target).startswith(str(work_dir.resolve())) or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe path in {tar_path.name}: {member.name}")
        try:
            handle.extractall(work_dir, members=members, filter="data")
        except TypeError:  # Python without tarfile extraction filters
            handle.extractall(work_dir, members=members)


def prepare_shard(
    encoder: Callable[[], CodecEncoder],
    dataset: str,
    filename: str,
    work_dir: Path,
    drive_data_dir: Path,
    options: PrepOptions,
    refs: list[dict[str, Any]],
) -> dict[str, float]:
    """Download one parquet part, encode it, and store a tar on Drive."""
    import soundfile as sf

    shard = _shard_name(dataset, filename)
    tar_on_drive = drive_data_dir / "shards" / f"{shard}.tar"
    local_manifest = work_dir / "shards" / f"{shard}.jsonl"
    if local_manifest.is_file():
        return {"status": "local"}
    if tar_on_drive.is_file():
        _extract_shard(tar_on_drive, work_dir)
        return {"status": "restored"}

    started = time.time()
    spec = DATASETS[dataset]
    parquet = _download(spec["repo"], filename, work_dir / "downloads")
    codes_rel = Path("codes") / shard
    (work_dir / codes_rel).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped = {"quality": 0, "length": 0, "text": 0, "encode": 0}
    pending: list[tuple[str, str, np.ndarray]] = []
    stem_tag = re.sub(r"\D", "", Path(filename).stem) or Path(filename).stem
    ref_count = sum(1 for r in refs if r["dataset"] == dataset)
    seen_ids: set[str] = set()

    def flush() -> None:
        nonlocal ref_count
        if not pending:
            return
        pending.sort(key=lambda item: item[2].size)
        batches: list[list[tuple[str, str, np.ndarray]]] = [[]]
        total = 0
        for item in pending:
            if batches[-1] and total + item[2].size > options.batch_seconds * CODEC_SAMPLE_RATE:
                batches.append([])
                total = 0
            batches[-1].append(item)
            total += item[2].size
        for batch in batches:
            encoded = encoder().encode([wave for _, _, wave in batch])
            for (sample_id, text, wave), codes in zip(batch, encoded):
                if codes is None:
                    skipped["encode"] += 1
                    continue
                rel = codes_rel / f"{sample_id}.npy"
                np.save(work_dir / rel, codes)
                seconds = wave.size / CODEC_SAMPLE_RATE
                rows.append(
                    {
                        "id": sample_id,
                        "dataset": dataset,
                        "text": text,
                        "target_codes": rel.as_posix(),
                        "frames": int(codes.shape[1]),
                        "seconds": round(seconds, 3),
                    }
                )
                if ref_count < options.refs_per_dataset and 4.0 <= seconds <= 9.0:
                    ref_dir = drive_data_dir / "refs"
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    ref_path = ref_dir / f"{sample_id}.wav"
                    sf.write(ref_path, wave, CODEC_SAMPLE_RATE)
                    refs.append({"dataset": dataset, "id": sample_id, "text": text, "audio": str(ref_path)})
                    ref_count += 1
        pending.clear()

    for row in iter_parquet_rows(parquet):
        quality = str(row.get("match_quality") or "").upper()
        if options.only_high_quality and quality and quality != "HIGH":
            skipped["quality"] += 1
            continue
        text = normalize_fa(row.get("transcript"))
        if len(text) < 2:
            skipped["text"] += 1
            continue
        rate = int(float(row.get("samplerate") or CODEC_SAMPLE_RATE))
        seconds = row["audio"].size / max(rate, 1)
        if not options.min_seconds <= seconds <= options.max_seconds:
            skipped["length"] += 1
            continue
        sample_id = _safe_id(dataset, stem_tag, str(row.get("file_name") or len(seen_ids)))
        if sample_id in seen_ids:
            sample_id = f"{sample_id}_{len(seen_ids)}"
        seen_ids.add(sample_id)
        pending.append((sample_id, text, _to_codec_rate(row["audio"], rate)))
        if len(pending) >= 96:
            flush()
    flush()

    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    with local_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    local_tar = work_dir / "tmp" / f"{shard}.tar"
    local_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "w") as handle:
        handle.add(work_dir / codes_rel, arcname=codes_rel.as_posix())
        handle.add(local_manifest, arcname=f"shards/{shard}.jsonl")
    _copy_to_drive(local_tar, tar_on_drive)
    local_tar.unlink(missing_ok=True)
    parquet.unlink(missing_ok=True)
    hours = sum(r["seconds"] for r in rows) / 3600
    return {
        "status": "prepared",
        "clips": len(rows),
        "hours": round(hours, 2),
        "minutes_taken": round((time.time() - started) / 60, 1),
        **{f"skipped_{k}": v for k, v in skipped.items()},
    }


def prepare_all(
    base_dir: str | Path,
    work_dir: str | Path,
    drive_data_dir: str | Path,
    plan: dict[str, int],
    options: PrepOptions | None = None,
) -> None:
    """plan maps dataset name -> number of parquet parts to use (from the start)."""
    options = options or PrepOptions()
    work_dir, drive_data_dir = Path(work_dir), Path(drive_data_dir)
    refs_path = drive_data_dir / "refs.json"
    refs: list[dict[str, Any]] = json.loads(refs_path.read_text("utf-8")) if refs_path.is_file() else []
    holder: dict[str, CodecEncoder] = {}

    def get_encoder() -> CodecEncoder:
        if "encoder" not in holder:
            log("loading codec for encoding ...")
            holder["encoder"] = CodecEncoder(base_dir)
        return holder["encoder"]

    jobs = [(name, f) for name, count in plan.items() for f in DATASETS[name]["files"][: max(0, count)]]
    try:
        for index, (dataset, filename) in enumerate(jobs, 1):
            result = prepare_shard(get_encoder, dataset, filename, work_dir, drive_data_dir, options, refs)
            log(f"[{index}/{len(jobs)}] {_shard_name(dataset, filename)}: {result}")
            if result.get("status") == "prepared":
                refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=1), "utf-8")
    finally:
        if "encoder" in holder:
            holder["encoder"].close()
    total = 0.0
    for manifest in sorted((work_dir / "shards").glob("*.jsonl")):
        with manifest.open(encoding="utf-8") as handle:
            total += sum(json.loads(line)["seconds"] for line in handle if line.strip())
    log(f"data ready: {total / 3600:.1f} hours of audio")


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def build_manifests(
    base_dir: str | Path,
    repo_dir: str | Path,
    work_dir: str | Path,
    drive_data_dir: str | Path,
    plan: dict[str, int],
    max_length: int = 1024,
    eval_size: int = 200,
    reference_prob: float = 0.3,
    always_reference: tuple[str, ...] = ("gptinformal",),
    seed: int = 1234,
    rebuild: bool = False,
    manifest_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build train/eval JSONL files in the repo's prepared format.

    Rows from datasets in ``always_reference`` are trained with a reference clip
    of the same speaker, so the default (no-reference) voice stays Mana's while
    the model also learns Persian voice cloning. The manifests are stored on
    Drive and reused, keeping the sample order identical across sessions.
    """
    import torch

    add_repo_to_path(repo_dir)
    from audio8_tts_data import build_sft_example, load_codes

    work_dir, drive_data_dir = Path(work_dir), Path(drive_data_dir)
    drive_manifests = Path(manifest_dir) if manifest_dir else drive_data_dir / "manifests"
    train_path, eval_path = work_dir / "train.jsonl", work_dir / "eval.jsonl"
    plan_key = json.dumps({"plan": plan, "max_length": max_length, "ref": reference_prob}, sort_keys=True)
    plan_file = drive_manifests / "plan.json"
    if (
        not rebuild
        and plan_file.is_file()
        and plan_file.read_text("utf-8") == plan_key
        and (drive_manifests / "train.jsonl").is_file()
    ):
        shutil.copyfile(drive_manifests / "train.jsonl", train_path)
        shutil.copyfile(drive_manifests / "eval.jsonl", eval_path)
        log("reused manifests from Drive (same data plan)")
        return train_path, eval_path

    wanted = {_shard_name(n, f) for n, c in plan.items() for f in DATASETS[n]["files"][:c]}
    rows: list[dict[str, Any]] = []
    for manifest in sorted((work_dir / "shards").glob("*.jsonl")):
        if manifest.stem not in wanted:
            continue
        with manifest.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise RuntimeError("no prepared data found; run the data preparation cell first")

    tokenizer = load_processor(base_dir).tokenizer
    config = load_config(base_dir)
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    candidates = {
        name: [r for r in items if 3 * FRAMES_PER_SECOND <= r["frames"] <= 9 * FRAMES_PER_SECOND]
        for name, items in by_dataset.items()
    }

    def fits(row: dict[str, Any], ref: dict[str, Any] | None) -> bool:
        try:
            build_sft_example(
                tokenizer=tokenizer,
                config=config,
                text=row["text"],
                target_codes=load_codes(work_dir / row["target_codes"]),
                reference_codes=load_codes(work_dir / ref["target_codes"]) if ref else None,
                reference_text=ref["text"] if ref else None,
                max_length=max_length,
            )
            return True
        except ValueError:
            return False

    final_rows: list[dict[str, Any]] = []
    dropped = 0
    for name in sorted(by_dataset):
        prob = 1.0 if name in always_reference else reference_prob
        pool = candidates.get(name) or []
        for row in by_dataset[name]:
            ref = None
            if pool and rng.random() < prob:
                for _ in range(5):
                    pick = rng.choice(pool)
                    if pick["id"] != row["id"]:
                        ref = pick
                        break
            if ref is not None and not fits(row, ref):
                ref = None
            if ref is None and not fits(row, None):
                dropped += 1
                continue
            item = {"id": row["id"], "text": row["text"], "target_codes": row["target_codes"]}
            if ref is not None:
                item.update(reference_codes=ref["target_codes"], reference_text=ref["text"])
            final_rows.append(item)

    rng.shuffle(final_rows)
    eval_size = min(eval_size, max(1, len(final_rows) // 20))
    eval_rows, train_rows = final_rows[:eval_size], final_rows[eval_size:]
    for path, items in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    drive_manifests.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_path, drive_manifests / "train.jsonl")
    shutil.copyfile(eval_path, drive_manifests / "eval.jsonl")
    plan_file.write_text(plan_key, "utf-8")
    with_ref = sum(1 for r in final_rows if "reference_codes" in r)
    log(
        f"manifests: train={len(train_rows)} eval={len(eval_rows)} "
        f"with_reference={with_ref} dropped_too_long={dropped}"
    )
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return train_path, eval_path


# ---------------------------------------------------------------------------
# Speech generation (used by the training callback and the test cells)
# ---------------------------------------------------------------------------


def _generation_layers(core) -> list[Any]:
    layers = list(getattr(core, "layers", [])) + list(getattr(core, "fast_layers", []))
    return [getattr(layer, "attention", None) for layer in layers if getattr(layer, "attention", None) is not None]


def _snapshot_caches(core) -> list[tuple[Any, bool, Any]]:
    snapshot = []
    for attention in _generation_layers(core):
        existed = "kv_cache" in attention.__dict__ or "kv_cache" in attention._modules or "kv_cache" in attention._buffers
        snapshot.append((attention, existed, getattr(attention, "kv_cache", None)))
    return snapshot


def _restore_caches(snapshot: list[tuple[Any, bool, Any]]) -> None:
    """generate() leaves static KV caches on every layer; put things back as they were."""
    for attention, existed, value in snapshot:
        if existed:
            setattr(attention, "kv_cache", value)
        else:
            for store in (attention._modules, attention._buffers, attention.__dict__):
                store.pop("kv_cache", None)


def synthesize(
    model,
    processor,
    text: str,
    reference_audio: str | None = None,
    reference_text: str | None = None,
    max_frames: int = 600,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int | None = 1234,
) -> tuple[np.ndarray, int, bool]:
    """Generate one waveform. Long text is split into chunks and joined."""
    import torch

    chunks = split_sentences(text)
    if not chunks:
        raise ValueError("text is empty after normalization")
    device = next(model.parameters()).device
    sample_rate = int(model.config.codec_sample_rate)
    pieces: list[np.ndarray] = []
    all_finished = True
    snapshot = _snapshot_caches(model)
    try:
        for index, chunk in enumerate(chunks):
            kwargs: dict[str, Any] = {"text": [chunk], "return_tensors": "pt"}
            if reference_audio:
                kwargs.update(
                    reference_audio=[str(reference_audio)],
                    reference_text=[normalize_fa(reference_text or "")],
                )
            inputs = processor(**kwargs)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(int(seed) + index)
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_frames),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                return_dict_in_generate=True,
                generator=generator,
            )
            finished = getattr(output, "finished", None)
            if finished is not None:
                all_finished = all_finished and bool(finished[0])
            waveforms, lengths = model.decode_audio(output.codes)
            pieces.append(waveforms[0, : int(lengths[0])].float().cpu().numpy())
            if index < len(chunks) - 1:
                pieces.append(np.zeros(int(0.18 * sample_rate), dtype=np.float32))
    finally:
        _restore_caches(snapshot)
    return np.concatenate(pieces), sample_rate, all_finished


def _show_audio(path: Path, title: str) -> None:
    try:
        from IPython.display import Audio, Markdown, display

        display(Markdown(f"**{title}**"))
        display(Audio(str(path)))
    except Exception:
        log(f"saved {path}")


def write_samples(
    model,
    processor,
    out_dir: Path,
    sentences: list[str],
    reference: dict[str, Any] | None,
    max_frames: int,
    title: str,
) -> list[dict[str, Any]]:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f"s{i + 1:02d}", text, None) for i, text in enumerate(sentences)]
    if reference:
        jobs.append(("clone", sentences[0], reference))
    records = []
    for name, text, ref in jobs:
        started = time.time()
        audio, rate, finished = synthesize(
            model,
            processor,
            text,
            reference_audio=ref["audio"] if ref else None,
            reference_text=ref["text"] if ref else None,
            max_frames=max_frames,
        )
        path = out_dir / f"{name}.wav"
        sf.write(path, audio, rate)
        note = "" if finished else " (did not reach the end of the sentence - normal early on)"
        label = "voice clone" if ref else f"sentence {name[1:]}"
        _show_audio(path, f"{title} | {label} | {len(audio) / rate:.1f}s{note}")
        records.append(
            {"file": path.name, "text": text, "clone": bool(ref), "finished": finished,
             "seconds": round(len(audio) / rate, 2), "gen_seconds": round(time.time() - started, 1)}
        )
    return records


# ---------------------------------------------------------------------------
# Checkpoints on Drive
# ---------------------------------------------------------------------------


def checkpoint_is_complete(path: Path) -> bool:
    try:
        state = path / "trainer_state.json"
        weights = path / "model.safetensors"
        if not (state.is_file() and weights.is_file() and weights.stat().st_size > 0):
            return False
        json.loads(state.read_text("utf-8"))
        if not any((path / name).is_file() for name in ("optimizer.pt", "optimizer.bin")):
            return False
        from safetensors import safe_open

        with safe_open(str(weights), framework="pt") as handle:
            return len(list(handle.keys())) > 0
    except Exception:
        return False


def list_checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    found = []
    for path in output_dir.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def find_resume_checkpoint(output_dir: str | Path) -> Path | None:
    """Newest complete checkpoint. Incomplete ones (cut off by a disconnect) are set aside."""
    for step, path in reversed(list_checkpoints(output_dir)):
        if checkpoint_is_complete(path):
            return path
        broken = path.with_name(f"broken-{path.name}-{int(time.time())}")
        log(f"checkpoint {path.name} is incomplete (likely a disconnect); moving it aside")
        try:
            path.rename(broken)
            shutil.rmtree(broken, ignore_errors=True)
        except OSError as exc:
            log(f"could not move {path.name}: {exc}")
    return None


def rotate_checkpoints(output_dir: str | Path, keep: int) -> list[str]:
    """Delete older checkpoints by step number (Trainer's mtime ordering is unsafe on Drive)."""
    complete = [(s, p) for s, p in list_checkpoints(output_dir) if checkpoint_is_complete(p)]
    removed = []
    for _step, path in complete[: max(0, len(complete) - max(1, keep))]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed


class DriveTrashPurger:
    """Permanently delete checkpoint folders that Colab's Drive mount moved to the trash.

    Files deleted through /content/drive go to Google Drive's trash and keep
    using storage for 30 days. This only touches trashed folders named
    checkpoint-N / broken-checkpoint-N whose parent is this run's checkpoint folder.
    """

    NAME_RE = re.compile(r"^(broken-)?checkpoint-\d+(-\d+)?$")

    def __init__(self, folder_under_my_drive: str):
        self.relative = folder_under_my_drive.strip("/")
        self.service = None
        self.folder_id: str | None = None
        self.disabled = False

    def _setup(self) -> bool:
        if self.disabled:
            return False
        if self.service is not None and self.folder_id:
            return True
        try:
            from googleapiclient.discovery import build  # type: ignore

            self.service = build("drive", "v3", cache_discovery=False)
            parent = "root"
            for name in [p for p in self.relative.split("/") if p]:
                escaped = name.replace("\\", "\\\\").replace("'", "\\'")
                result = self.service.files().list(
                    q=(
                        f"name = '{escaped}' and '{parent}' in parents and trashed = false "
                        "and mimeType = 'application/vnd.google-apps.folder'"
                    ),
                    fields="files(id, name)",
                    pageSize=10,
                ).execute()
                files = result.get("files", [])
                if len(files) != 1:
                    log(f"trash cleanup disabled: could not uniquely find folder {name!r}")
                    self.disabled = True
                    return False
                parent = files[0]["id"]
            self.folder_id = parent
            return True
        except Exception as exc:
            log(f"trash cleanup disabled ({type(exc).__name__}: {exc})")
            self.disabled = True
            return False

    def purge(self) -> int:
        if not self._setup():
            return 0
        removed = 0
        try:
            result = self.service.files().list(
                q=f"'{self.folder_id}' in parents and trashed = true",
                fields="files(id, name, mimeType)",
                pageSize=100,
            ).execute()
            for item in result.get("files", []):
                if item.get("mimeType") != "application/vnd.google-apps.folder":
                    continue
                if not self.NAME_RE.match(item.get("name", "")):
                    continue
                self.service.files().delete(fileId=item["id"]).execute()
                removed += 1
        except Exception as exc:
            log(f"trash cleanup failed: {type(exc).__name__}: {exc}")
        return removed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def enable_slow_checkpointing(core) -> None:
    """Activation checkpointing for the slow AR layers.

    ArkttsModel does not implement Transformers' gradient_checkpointing_enable(),
    so the repo's default GRADIENT_CHECKPOINTING=true would fail. We wrap each
    slow layer ourselves; the fast branch uses the repo's own switch.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    for layer in core.layers:
        if getattr(layer, "_fa_checkpointed", False):
            continue
        original = layer.forward

        def forward(*args, _original=original, **kwargs):
            if core.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        layer.forward = forward
        layer._fa_checkpointed = True
    core.gradient_checkpointing = True  # read by the repo's fast_codebook_logits()


@dataclass
class TrainConfig:
    base_dir: str
    repo_dir: str
    train_jsonl: str
    eval_jsonl: str | None
    output_dir: str  # on Drive
    samples_dir: str  # on Drive
    drive_folder_for_trash: str | None  # checkpoints folder relative to MyDrive, or None
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    batch_size: int = 0  # 0 = auto
    grad_accum: int = 0  # 0 = auto
    learning_rate: float = 3e-5
    epochs: float = 3.0
    warmup_ratio: float = 0.03
    save_steps: int = 300
    keep_checkpoints: int = 2
    max_length: int = 1024
    freeze_fast_ar: bool = False
    freeze_slow_ar: bool = False
    optim: str = "adamw_8bit"
    test_sentences: tuple[str, ...] = ()
    clone_reference: dict[str, Any] | None = None
    sample_max_frames: int = 500
    logging_steps: int = 20
    num_workers: int = 2
    seed: int = 1234


def _resolve_training_settings(cfg: TrainConfig) -> dict[str, Any]:
    profile = gpu_profile()
    precision = profile["precision"] if cfg.precision == "auto" else cfg.precision
    if precision == "bf16" and profile["name"] != "CPU":
        import torch

        if torch.cuda.get_device_properties(0).major < 8:
            log("this GPU has no native bf16; switching to fp16")
            precision = "fp16"
    if profile["name"] == "CPU":
        precision = "fp32"
    optim = cfg.optim
    if optim.endswith("8bit"):
        try:
            import bitsandbytes  # noqa: F401
            import torch

            if not torch.cuda.is_available():
                raise ImportError("8-bit Adam needs a GPU")
        except Exception as exc:
            log(f"8-bit Adam unavailable ({exc}); using adamw_torch")
            optim = "adamw_torch"
    return {
        "gpu": profile["name"],
        "precision": precision,
        "batch_size": cfg.batch_size or profile["batch_size"],
        "grad_accum": cfg.grad_accum or profile["grad_accum"],
        "optim": optim,
    }


def train(cfg: TrainConfig) -> Path | None:
    """Fine-tune with automatic resume, per-checkpoint audio samples and safe rotation."""
    import torch
    from transformers import TrainerCallback, TrainingArguments

    add_repo_to_path(cfg.repo_dir)
    import audio8_tts_sft as sft

    settings = _resolve_training_settings(cfg)
    log(
        f"GPU={settings['gpu']} precision={settings['precision']} batch={settings['batch_size']} "
        f"x accum={settings['grad_accum']} optim={settings['optim']}"
    )
    output_dir, samples_dir = Path(cfg.output_dir), Path(cfg.samples_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(cfg.base_dir)
    # fp32 master weights + autocast (bf16/fp16). The repo default trains pure
    # bf16 weights, where small learning-rate updates are often rounded away.
    model = load_model(cfg.base_dir, dtype=torch.float32)
    if cfg.max_length > int(model.config.max_seq_len):
        raise ValueError(f"max_length {cfg.max_length} > model limit {model.config.max_seq_len}")
    sft.set_trainable_modules(model, freeze_slow=cfg.freeze_slow_ar, freeze_fast=cfg.freeze_fast_ar)
    enable_slow_checkpointing(model)

    train_dataset = sft.Audio8TTSDataset(
        Path(cfg.train_jsonl), tokenizer=processor.tokenizer, config=model.config,
        max_length=cfg.max_length, max_samples=None,
    )
    eval_dataset = None
    if cfg.eval_jsonl and Path(cfg.eval_jsonl).is_file():
        eval_dataset = sft.Audio8TTSDataset(
            Path(cfg.eval_jsonl), tokenizer=processor.tokenizer, config=model.config,
            max_length=cfg.max_length, max_samples=None,
        )

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["grad_accum"],
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=None,  # rotation is done by step number in the callback
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.save_steps if eval_dataset is not None else None,
        prediction_loss_only=True,  # never gather 155k-vocab logits during eval
        bf16=settings["precision"] == "bf16",
        fp16=settings["precision"] == "fp16",
        gradient_checkpointing=False,  # handled by enable_slow_checkpointing()
        optim=settings["optim"],
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=["tensorboard"] if _has_tensorboard() else "none",
        logging_dir=str(output_dir / "logs"),
        seed=cfg.seed,
        save_safetensors=True,
    )

    trash = DriveTrashPurger(cfg.drive_folder_for_trash) if cfg.drive_folder_for_trash else None
    reference = cfg.clone_reference if cfg.clone_reference and Path(cfg.clone_reference["audio"]).is_file() else None

    class SampleAndRotate(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):
            step = int(state.global_step)
            core = sft.actual_model(model)
            eval_loss = next(
                (entry["eval_loss"] for entry in reversed(state.log_history) if "eval_loss" in entry), None
            )
            header = f"step {step}" + (f" | eval_loss={eval_loss:.3f}" if eval_loss is not None else "")
            records: list[dict[str, Any]] = []
            if cfg.test_sentences:
                was_training = core.training
                core.eval()
                try:
                    records = write_samples(
                        core, processor, samples_dir / f"step_{step:07d}", list(cfg.test_sentences),
                        reference, cfg.sample_max_frames, header,
                    )
                except Exception as exc:  # samples must never stop training
                    log(f"sample generation failed at step {step}: {type(exc).__name__}: {exc}")
                finally:
                    core.train(was_training)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            with (samples_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "eval_loss": eval_loss, "samples": records,
                                         "time": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
            removed = rotate_checkpoints(args.output_dir, cfg.keep_checkpoints)
            purged = trash.purge() if trash else 0
            log(f"checkpoint-{step} saved to Drive" + (f"; removed {removed}" if removed else "")
                + (f"; emptied {purged} from Drive trash" if purged else ""))
            return control

    trainer = sft.Audio8TTSTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(sft.collate_sft_examples, pad_token_id=int(model.config.pad_token_id)),
        slow_loss_weight=1.0,
        fast_loss_weight=1.0,
        callbacks=[SampleAndRotate()],
    )
    # forward() takes **kwargs, so Trainer assumes the loss is already normalized
    # for gradient accumulation; the repo's loss is a plain mean, so turn that off.
    trainer.model_accepts_loss_kwargs = False

    steps_per_epoch = math.ceil(len(train_dataset) / (settings["batch_size"] * settings["grad_accum"]))
    total_steps = math.ceil(steps_per_epoch * cfg.epochs)
    resume = find_resume_checkpoint(output_dir)
    log(
        f"train clips={len(train_dataset)} | steps/epoch={steps_per_epoch} | total steps={total_steps} | "
        f"checkpoint every {cfg.save_steps} steps"
    )
    log(f"resuming from {resume.name}" if resume else "no checkpoint on Drive yet; starting fresh")

    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    last = list_checkpoints(output_dir)
    if not last or last[-1][0] < trainer.state.global_step:
        log("saving the final step as a checkpoint ...")
        trainer._save_checkpoint(trainer.model, trial=None)
        trainer.callback_handler.on_save(trainer.args, trainer.state, trainer.control)
    final = find_resume_checkpoint(output_dir)
    log(f"training finished; final checkpoint: {final}")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Testing and export
# ---------------------------------------------------------------------------


def resolve_checkpoint(output_dir: str | Path, which: str | int | None) -> Path | None:
    """'latest' -> newest complete checkpoint, a number -> that step, 'base' -> None."""
    value = str(which if which is not None else "latest").strip().lower()
    if value in {"base", "none", ""}:
        return None
    if value == "latest":
        found = [p for _s, p in reversed(list_checkpoints(output_dir)) if checkpoint_is_complete(p)]
        if not found:
            raise FileNotFoundError(f"no complete checkpoint in {output_dir}")
        return found[0]
    path = Path(output_dir) / f"checkpoint-{int(value)}"
    if not path.is_dir():
        options = [s for s, _ in list_checkpoints(output_dir)]
        raise FileNotFoundError(f"{path.name} not found; available steps: {options}")
    return path


def load_for_inference(base_dir: str | Path, checkpoint: Path | None):
    """Base model (with its codec file) + fine-tuned weights from a Trainer checkpoint."""
    import torch
    from safetensors.torch import load_file

    dtype = inference_dtype()
    model = load_model(base_dir, dtype=dtype)
    if checkpoint is not None:
        state = load_file(str(Path(checkpoint) / "model.safetensors"))
        state = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}
        result = model.load_state_dict(state, strict=False)
        param_names = {name for name, _ in model.named_parameters()}
        missing = [k for k in result.missing_keys if k in param_names]
        if missing:
            raise RuntimeError(f"checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        log(f"loaded weights from {Path(checkpoint).name}")
    model = model.eval().to(device_name())
    return model, load_processor(base_dir)


def export_model(base_dir: str | Path, checkpoint: str | Path, export_dir: str | Path) -> Path:
    """Standalone bf16 model folder (code, tokenizer, codec, weights) loadable with AutoModel."""
    import torch
    from safetensors.torch import load_file, save_file

    base_dir, export_dir = Path(base_dir), Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for item in base_dir.iterdir():
        if item.is_file() and item.name != "model.safetensors" and not item.name.startswith("."):
            target = export_dir / item.name
            if not target.is_file() or target.stat().st_size != item.stat().st_size:
                shutil.copyfile(item, target)
    merged = load_file(str(base_dir / "model.safetensors"))
    tuned = load_file(str(Path(checkpoint) / "model.safetensors"))
    unknown = [k for k in tuned if k not in merged]
    if unknown:
        log(f"note: {len(unknown)} extra tensors in checkpoint ignored, e.g. {unknown[:2]}")
    for key in merged:
        if key in tuned:
            merged[key] = tuned[key]
    merged = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in merged.items()}
    temporary = export_dir / "model.safetensors.partial"
    save_file(merged, str(temporary), metadata={"format": "pt"})
    os.replace(temporary, export_dir / "model.safetensors")
    log(f"exported to {export_dir}")
    return export_dir
