"""Builds notebooks/Audio8_Persian_Finetune.ipynb from src/fa_tools.py.

The notebook embeds src/fa_tools.py with %%writefile so it runs standalone in
Colab. Edit src/fa_tools.py (or this file), then run:

    python scripts/build_notebook.py

CI fails if the committed notebook is out of sync.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = (ROOT / "src" / "fa_tools.py").read_text(encoding="utf-8")
REPO_URL = "https://github.com/ard9/Audio8-Persian-TTS"
REPO_COMMIT = "07e40f5d0b03fc473635ef378654bfb581027ac3"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, title=None):
    meta = {"cellView": "form"} if title else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(f"""
# 🎙️ Fine-tuning Audio8 TTS for Persian (Farsi)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ard9/Audio8-Persian-TTS/blob/main/notebooks/Audio8_Persian_Finetune.ipynb) · Project: {REPO_URL}

This notebook fine-tunes **Audio8-TTS-Preview-0.6B** on the **Mana-TTS** dataset (plus, optionally, GPTInformal for a conversational style).

**What happens automatically:**
- 💾 Everything (prepared data, checkpoints, audio samples) is saved to your **Google Drive**.
- 🔁 If Colab disconnects, just run **Runtime → Run all** again; training resumes from the last checkpoint.
- 🔊 After **every checkpoint**, a few test sentences are generated and played right here, so you can hear quality improve.
- 🧪 After training, use the **Test** cell to try any text with any checkpoint (or your own voice).

**Before you start:**
1. **Runtime → Change runtime type** → pick a GPU. **L4 or A100 (Colab Pro) is recommended**; the free T4 works but is much slower.
2. Drive space: each checkpoint is about **5 GB** and the last 2 are kept by default (~10 GB). Prepared data is under 1 GB.
3. Review the settings in the next cell, then **Runtime → Run all**.
""")

code('''
#@title ⚙️ Main settings (review before running)
#@markdown **Run name** — keep it unchanged to continue the same training. Use a new name for a new experiment.
RUN_NAME = "audio8_fa_v1"  #@param {type:"string"}
DRIVE_FOLDER = "audio8_fa"  #@param {type:"string"}
#@markdown ---
#@markdown **Data** — Mana-TTS has 77 parts (each ≈ 1.5 hours of audio). Use 10 parts (≈ 15 h) for a first test, 40–77 parts for good results.
MANA_PARTS = 10  #@param {type:"slider", min:1, max:77, step:1}
USE_GPTINFORMAL = True  #@param {type:"boolean"}
ONLY_HIGH_QUALITY = True  #@param {type:"boolean"}
MAX_AUDIO_SECONDS = 20  #@param {type:"number"}
#@markdown ---
#@markdown **Training** — 0 means automatic, based on the GPU.
EPOCHS = 3  #@param {type:"number"}
LEARNING_RATE = 3e-5  #@param {type:"number"}
SAVE_EVERY_STEPS = 300  #@param {type:"integer"}
KEEP_CHECKPOINTS = 2  #@param {type:"integer"}
PRECISION = "auto"  #@param ["auto", "bf16", "fp16", "fp32"]
BATCH_SIZE = 0  #@param {type:"integer"}
GRAD_ACCUM = 0  #@param {type:"integer"}
FREEZE_FAST_AR = False  #@param {type:"boolean"}
#@markdown ---
#@markdown **Auto-empty Drive trash** — only for this run's old checkpoints (otherwise Drive fills up).
EMPTY_DRIVE_TRASH = True  #@param {type:"boolean"}

# Sentences read aloud after every checkpoint. They stay in Persian because
# they are the model's input text; edit them freely.
TEST_SENTENCES = [
    "سلام، امروز هوا خیلی خوب است و می‌خواهم کمی در پارک قدم بزنم.",
    "این یک آزمایش برای سنجش کیفیت مدل تبدیل متن به گفتار فارسی است.",
    "دیروز با دوستام رفتیم سینما، فیلمش واقعاً عالی بود!",
]

from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive") / DRIVE_FOLDER
DATA_DIR = DRIVE_ROOT / "data"                # prepared data (shared across runs)
RUN_DIR = DRIVE_ROOT / "runs" / RUN_NAME
CKPT_DIR = RUN_DIR / "checkpoints"
SAMPLES_DIR = RUN_DIR / "samples"
EXPORT_DIR = RUN_DIR / "export"
WORK_DIR = Path("/content/work")              # Colab's temporary local disk
BASE_DIR = Path("/content/base_model")
REPO_DIR = Path("/content/Audio8_TTS")
MAX_LENGTH = 1024
print("✅ Settings saved. This run's folder on Drive:", RUN_DIR)
''', title=True)

code('''
#@title 📁 Connect Google Drive and check the GPU
from google.colab import drive
drive.mount("/content/drive")
for p in (DATA_DIR, CKPT_DIR, SAMPLES_DIR, WORK_DIR):
    p.mkdir(parents=True, exist_ok=True)

import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
if not gpu:
    raise SystemExit("❌ No GPU attached. Go to Runtime → Change runtime type → GPU.")
print("GPU:", gpu)
if "T4" in gpu:
    print("⚠️ T4: training works but is slow (fp16). If you see NaN / FloatingPointError, set PRECISION to fp32.")

if EMPTY_DRIVE_TRASH:
    from google.colab import auth
    auth.authenticate_user()  # permission to permanently delete old checkpoints from Drive trash
''', title=True)

code(f'''
#@title 📦 Install dependencies (~2 minutes)
import os
if not (REPO_DIR / ".git").is_dir():
    !git clone -q https://github.com/Edge0-AI/Audio8_TTS.git {{REPO_DIR}}
!cd {{REPO_DIR}} && git fetch -q origin && git checkout -q {REPO_COMMIT}
!pip install -q "transformers>=4.57.0,<5" "accelerate>=1.10" bitsandbytes num2fawords soundfile tensorboard
import transformers
print("transformers", transformers.__version__)
if int(transformers.__version__.split(".")[0]) >= 5:
    print("⚠️ An older transformers is required: Runtime → Restart session, then Run all again.")
''', title=True)

code("%%writefile /content/fa_tools.py\n" + TOOLS)

code('''
#@title ⬇️ Download the Audio8 base model (≈ 2.5 GB)
import sys, importlib
sys.path.insert(0, "/content")
import fa_tools as fa
importlib.reload(fa)
fa.add_repo_to_path(REPO_DIR)

from huggingface_hub import snapshot_download
if not (BASE_DIR / "codec.pth").is_file():
    for repo_id in ("Edge0/Audio8-TTS-Preview-0.6b", "Audio8/Audio8-TTS-Preview-0.6b"):
        try:
            snapshot_download(repo_id, local_dir=str(BASE_DIR), token=fa.hf_token())
            break
        except Exception as exc:
            print(f"{repo_id}: {exc}")
    else:
        raise SystemExit("❌ Could not download the base model.")
print("✅ Base model ready:", sorted(p.name for p in BASE_DIR.iterdir()))
''', title=True)

md("""
## 1️⃣ Data preparation

Each dataset part is downloaded, its audio is converted to Audio8 codec codes, and the result is stored as one small file on Drive (`data/shards`). If you get disconnected midway, finished parts are not processed again.

> ⏱️ Roughly a few minutes per part (download + encoding). The first time takes a while; later sessions just restore from Drive (seconds).
>
> ℹ️ The data options (`ONLY_HIGH_QUALITY`, `MAX_AUDIO_SECONDS`) only apply the first time each part is processed.
""")

code('''
#@title 🧹 Prepare data (resumes after a disconnect)
PLAN = {"mana": int(MANA_PARTS)}
if USE_GPTINFORMAL:
    PLAN["gptinformal"] = 5
fa.prepare_all(BASE_DIR, WORK_DIR, DATA_DIR, PLAN,
               fa.PrepOptions(max_seconds=float(MAX_AUDIO_SECONDS), only_high_quality=ONLY_HIGH_QUALITY))
TRAIN_JSONL, EVAL_JSONL = fa.build_manifests(
    BASE_DIR, REPO_DIR, WORK_DIR, DATA_DIR, PLAN, max_length=MAX_LENGTH,
    manifest_dir=RUN_DIR / "manifests",
)
''', title=True)

code('''
#@title 🔈 (Optional) Base model's voice before training — for comparison
RUN_BASELINE = True  #@param {type:"boolean"}
if RUN_BASELINE and not (SAMPLES_DIR / "base").is_dir():
    import gc, torch
    model, processor = fa.load_for_inference(BASE_DIR, None)
    fa.write_samples(model, processor, SAMPLES_DIR / "base", TEST_SENTENCES[:2], None, 500, "Base model (before training)")
    del model, processor; gc.collect(); torch.cuda.empty_cache()
elif RUN_BASELINE:
    print("Base model samples already exist:", SAMPLES_DIR / "base")
''', title=True)

md("""
## 2️⃣ Training

- Every `SAVE_EVERY_STEPS` steps a checkpoint is saved to `runs/<run name>/checkpoints` on Drive, and audio samples are played **right below** (and saved under `samples/`).
- If Colab disconnects: **Runtime → Run all** (approve Drive access again). It finds the newest complete checkpoint and continues. A checkpoint that was cut off mid-save is set aside automatically.
- `eval_loss` should go down over time. Early on the audio is unintelligible; that is normal.
""")

code('''
#@title 🚀 Start / resume training
import json
refs_file = DATA_DIR / "refs.json"
refs = json.loads(refs_file.read_text("utf-8")) if refs_file.is_file() else []
clone_ref = next((r for r in refs if r["dataset"] == "gptinformal"), None)  # voice-clone test with the second speaker

cfg = fa.TrainConfig(
    base_dir=str(BASE_DIR), repo_dir=str(REPO_DIR),
    train_jsonl=str(TRAIN_JSONL), eval_jsonl=str(EVAL_JSONL),
    output_dir=str(CKPT_DIR), samples_dir=str(SAMPLES_DIR),
    drive_folder_for_trash=f"{DRIVE_FOLDER}/runs/{RUN_NAME}/checkpoints" if EMPTY_DRIVE_TRASH else None,
    precision=PRECISION, batch_size=int(BATCH_SIZE), grad_accum=int(GRAD_ACCUM),
    learning_rate=float(LEARNING_RATE), epochs=float(EPOCHS),
    save_steps=int(SAVE_EVERY_STEPS), keep_checkpoints=int(KEEP_CHECKPOINTS),
    max_length=MAX_LENGTH, freeze_fast_ar=FREEZE_FAST_AR,
    test_sentences=tuple(TEST_SENTENCES), clone_reference=clone_ref,
)
FINAL_CKPT = fa.train(cfg)
''', title=True)

code('''
#@title 📈 (Optional) Loss curves in TensorBoard
%load_ext tensorboard
%tensorboard --logdir {CKPT_DIR}/logs
''', title=True)

md("""
## 3️⃣ Listen and test
""")

code('''
#@title 🎧 All samples saved after each checkpoint
SHOW_LAST = 6  #@param {type:"integer"}
import json
from IPython.display import Audio, Markdown, display
index = SAMPLES_DIR / "index.jsonl"
entries = [json.loads(l) for l in index.read_text("utf-8").splitlines() if l.strip()] if index.is_file() else []
seen = {}
for e in entries:
    seen[e["step"]] = e  # latest record for each step
if (SAMPLES_DIR / "base").is_dir():
    display(Markdown("### Base model (before training)"))
    for f in sorted((SAMPLES_DIR / "base").glob("*.wav")):
        display(Audio(str(f)))
for step in sorted(seen)[-int(SHOW_LAST):]:
    e = seen[step]
    loss = f" — eval_loss = {e['eval_loss']:.3f}" if e.get("eval_loss") is not None else ""
    display(Markdown(f"### Step {step}{loss} ({e.get('time', '')})"))
    folder = SAMPLES_DIR / f"step_{step:07d}"
    for f in sorted(folder.glob("*.wav")):
        display(Markdown(f"`{f.name}`")); display(Audio(str(f)))
if not seen:
    print("No checkpoint saved yet.")
''', title=True)

code('''
#@title 🧪 Test the model with your own text
#@markdown `latest` = newest checkpoint, or a step number (e.g. `1200`), or `base` for the original model.
CHECKPOINT = "latest"  #@param {type:"string"}
TEXT = "سلام! این صدای مدل فارسی من است. امیدوارم کیفیتش خوب باشد."  #@param {type:"string"}
#@markdown **Clone your own voice** (optional): upload a clean 5–10 second audio clip and type its **exact** transcript.
USE_MY_VOICE = False  #@param {type:"boolean"}
UPLOAD_NEW_VOICE = False  #@param {type:"boolean"}
REFERENCE_TEXT = ""  #@param {type:"string"}
TEMPERATURE = 0.8  #@param {type:"slider", min:0.3, max:1.2, step:0.05}
SEED = 1234  #@param {type:"integer"}

import gc, time, subprocess, torch, soundfile as sf
from IPython.display import Audio, display

ckpt = fa.resolve_checkpoint(CKPT_DIR, CHECKPOINT)
key = str(ckpt)
if globals().get("_TEST_KEY") != key:
    for name in ("_TEST_MODEL", "_TEST_PROC"):
        globals().pop(name, None)
    gc.collect(); torch.cuda.empty_cache()
    _TEST_MODEL, _TEST_PROC = fa.load_for_inference(BASE_DIR, ckpt)
    _TEST_KEY = key

ref_audio = None
if USE_MY_VOICE:
    if not REFERENCE_TEXT.strip():
        raise SystemExit("❌ Type the exact transcript of your reference clip in REFERENCE_TEXT.")
    if globals().get("_MY_VOICE") is None or UPLOAD_NEW_VOICE:
        from google.colab import files
        up = files.upload()
        src = next(iter(up))
        _MY_VOICE = "/content/my_voice.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "44100", _MY_VOICE], check=True)
    ref_audio = _MY_VOICE

t0 = time.time()
audio, sr, finished = fa.synthesize(_TEST_MODEL, _TEST_PROC, TEXT, reference_audio=ref_audio,
                                    reference_text=REFERENCE_TEXT if ref_audio else None,
                                    max_frames=1200, temperature=float(TEMPERATURE), seed=int(SEED))
out_dir = RUN_DIR / "tests"; out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{ckpt.name if ckpt else 'base'}.wav"
sf.write(out, audio, sr)
print(f"Checkpoint: {ckpt.name if ckpt else 'base'} | audio {len(audio)/sr:.1f}s | generated in {time.time()-t0:.1f}s"
      + ("" if finished else " | ⚠️ the model did not finish the sentence"))
print("Saved:", out)
display(Audio(audio, rate=sr))
''', title=True)

code('''
#@title 📤 Export the final model (standalone bf16 folder ≈ 2.6 GB)
#@markdown By default the export goes to Colab's local disk (it does not use Drive space). Tick `SAVE_TO_DRIVE` to keep a copy on Drive.
EXPORT_CHECKPOINT = "latest"  #@param {type:"string"}
SAVE_TO_DRIVE = False  #@param {type:"boolean"}
ckpt = fa.resolve_checkpoint(CKPT_DIR, EXPORT_CHECKPOINT)
EXPORTED = fa.export_model(BASE_DIR, ckpt, (EXPORT_DIR if SAVE_TO_DRIVE else Path("/content/export")) / ckpt.name)
print(f"""
Exported {ckpt.name} to {EXPORTED}
To use it later (e.g. with the repo's scripts/synthesize.py):
  python scripts/synthesize.py --model "{EXPORTED}" --text "سلام" --output out.wav
""")
''', title=True)

code('''
#@title 🤗 Upload the exported model to Hugging Face
#@markdown Needs a Hugging Face **write** token: add it in Colab's 🔑 Secrets panel as `HF_TOKEN` (or you will be asked to log in).
HF_REPO_ID = "ArminRahimi/Audio8-TTS-Persian"  #@param {type:"string"}
PRIVATE = False  #@param {type:"boolean"}
UPLOAD_SAMPLES = True  #@param {type:"boolean"}
MODEL_CARD_URL = "https://raw.githubusercontent.com/ard9/Audio8-Persian-TTS/main/docs/MODEL_CARD.md"  #@param {type:"string"}

import urllib.request
from huggingface_hub import HfApi, login
token = fa.hf_token()
if not token:
    login()
api = HfApi(token=token)
if "EXPORTED" not in globals():
    raise SystemExit("❌ Run the Export cell first.")
api.create_repo(HF_REPO_ID, private=PRIVATE, exist_ok=True)

try:
    card = urllib.request.urlopen(MODEL_CARD_URL, timeout=20).read().decode("utf-8")
except Exception as exc:
    print(f"⚠️ Could not fetch the model card ({exc}); uploading a minimal one. Replace it later with docs/MODEL_CARD.md.")
    card = ("---\nlicense: apache-2.0\nlanguage: [fa]\nbase_model: Audio8/Audio8-TTS-Preview-0.6b\n"
            "pipeline_tag: text-to-speech\ntags: [custom_code, persian, tts]\n---\n\n# Audio8-TTS-Persian\n\n"
            "Persian fine-tune of Audio8 TTS. Code: https://github.com/ard9/Audio8-Persian-TTS\n")
api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=HF_REPO_ID)
api.upload_folder(folder_path=str(EXPORTED), repo_id=HF_REPO_ID,
                  ignore_patterns=["*.partial", "README.md", ".*", "*.jpeg", "*.jpg"],
                  commit_message=f"Upload {Path(str(EXPORTED)).name}")
if UPLOAD_SAMPLES:
    last = [s for s, _ in fa.list_checkpoints(CKPT_DIR)][-1:]
    folder = SAMPLES_DIR / f"step_{last[0]:07d}" if last else None
    if folder and folder.is_dir():
        names = {"s01.wav": "ft_s01.wav", "s02.wav": "ft_s02.wav", "s03.wav": "ft_s03.wav", "clone.wav": "clone_s01.wav"}
        for src, dst in names.items():
            if (folder / src).is_file():
                api.upload_file(path_or_fileobj=str(folder / src), path_in_repo=f"samples/{dst}", repo_id=HF_REPO_ID)
print(f"✅ Done: https://huggingface.co/{HF_REPO_ID}")
''', title=True)

md("""
## 💡 Tips and troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` (e.g. 2 or 1) and raise `GRAD_ACCUM` (keep their product around 16). |
| `FloatingPointError` / `NaN` on T4 | `PRECISION = "fp32"` (slower but stable). |
| Drive is full | `KEEP_CHECKPOINTS = 1`, make sure `EMPTY_DRIVE_TRASH` is on, empty the Drive trash manually, and delete old `export/` folders on Drive (the export cell writes to local disk by default). |
| Audio runs on / repeats | Normal in early checkpoints. Keep training or add more data. |
| Some words are mispronounced | More data (higher `MANA_PARTS`) helps the most. |

**How to judge quality as training goes on:**
1. Early: mumbling or non-Persian sounds → normal.
2. Next: Persian-like syllables, but words are unclear.
3. Next: understandable words, intonation still robotic.
4. Finally: fluent, understandable sentences → a good point to add more data or stop.

**Scaling up after the first test:** raise `MANA_PARTS` and use a new `RUN_NAME` (parts already prepared are not processed again).
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "L4", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
for cell in notebook["cells"]:
    lines = cell["source"].splitlines(keepends=True)
    cell["source"] = lines


def render() -> str:
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    out = ROOT / "notebooks" / "Audio8_Persian_Finetune.ipynb"
    out.write_text(render(), encoding="utf-8")
    print("wrote", out, len(cells), "cells")
