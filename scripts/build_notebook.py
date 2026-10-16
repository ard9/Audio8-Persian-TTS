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
- 💾 Prepared data, checkpoints and audio samples are stored on the **Hugging Face Hub**, so you can continue on any machine.
- 🔁 If the session ends, run **Run all** again — on the same or a different machine. Training resumes from the last checkpoint.
- 🔊 After **every checkpoint**, a few test sentences are generated and played right here.
- 🧪 After training, use the **Test** cell to try any text with any checkpoint (or your own voice), then **Upload** the model.

**Before you start:**
1. **Runtime → Change runtime type** → pick a GPU. **L4 or A100 (Colab Pro) is recommended**; the free T4 works but is slower.
2. Create a Hugging Face **write** token at https://huggingface.co/settings/tokens and add it in Colab's 🔑 **Secrets** panel as `HF_TOKEN` (enable *Notebook access*).
   Data goes to a public dataset repo (Mana-TTS and GPTInformal are CC0); checkpoints go to a **private** repo where only the newest one is kept (~4.4 GB).
3. Review the settings in the next cell, then **Runtime → Run all**.

Works on Kaggle too (enable Internet and a GPU, add `HF_TOKEN` under *Add-ons → Secrets*).
""")

code('''
#@title ⚙️ Main settings (review before running)
#@markdown **Run name** — keep it unchanged to continue the same training. Use a new name for a new experiment.
RUN_NAME = "audio8_fa_v2"  #@param {type:"string"}
#@markdown ---
#@markdown **Hugging Face storage** — data, checkpoints and samples are kept here (needs `HF_TOKEN`).
HF_USERNAME = "ArminRahimi"  #@param {type:"string"}
HF_DATA_REPO = f"{HF_USERNAME}/audio8-fa-codes"        # public dataset: prepared codec codes
HF_TRAIN_REPO = f"{HF_USERNAME}/audio8-fa-training"    # private: checkpoints, samples, manifests
HF_MODEL_REPO = f"{HF_USERNAME}/Audio8-TTS-Persian"    # public: the final model
#@markdown ---
#@markdown **Data** — Mana-TTS has 77 parts (each ≈ 1.5 hours of audio). 10 parts ≈ 15 h for a quick test; 77 parts ≈ 114 h for the best quality.
MANA_PARTS = 77  #@param {type:"slider", min:1, max:77, step:1}
USE_GPTINFORMAL = True  #@param {type:"boolean"}
ONLY_HIGH_QUALITY = True  #@param {type:"boolean"}
MAX_AUDIO_SECONDS = 20  #@param {type:"number"}
#@markdown ---
#@markdown **Training** — 0 means automatic, based on the GPU.
EPOCHS = 2  #@param {type:"number"}
LEARNING_RATE = 3e-5  #@param {type:"number"}
SAVE_EVERY_STEPS = 1000  #@param {type:"integer"}
KEEP_CHECKPOINTS = 1  #@param {type:"integer"}
PRECISION = "auto"  #@param ["auto", "bf16", "fp16", "fp32"]
BATCH_SIZE = 0  #@param {type:"integer"}
GRAD_ACCUM = 0  #@param {type:"integer"}
FREEZE_FAST_AR = False  #@param {type:"boolean"}

# Sentences read aloud after every checkpoint. They stay in Persian because
# they are the model's input text; edit them freely.
TEST_SENTENCES = [
    "سلام، امروز هوا خیلی خوب است و می‌خواهم کمی در پارک قدم بزنم.",
    "این یک آزمایش برای سنجش کیفیت مدل تبدیل متن به گفتار فارسی است.",
    "دیروز با دوستام رفتیم سینما، فیلمش واقعاً عالی بود!",
]

import os
from pathlib import Path
ROOT = Path("/content") if Path("/content").is_dir() else Path.cwd()   # Colab or Kaggle
WORK_DIR = ROOT / "work"                      # temporary local disk
BASE_DIR = ROOT / "base_model"
REPO_DIR = ROOT / "Audio8_TTS"
DATA_DIR = WORK_DIR / "data_cache"            # local cache; the Hub holds the real copy
RUN_DIR = WORK_DIR / "runs" / RUN_NAME          # local cache; the Hub holds the real copy
CKPT_DIR = RUN_DIR / "checkpoints"
SAMPLES_DIR = RUN_DIR / "samples"
EXPORT_DIR = ROOT / "export"
MAX_LENGTH = 1024
print(f"✅ Settings saved. Run: {RUN_NAME} → https://huggingface.co/{HF_TRAIN_REPO}")
''', title=True)

code('''
#@title 🔑 Check the GPU and the Hugging Face token
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
if not gpu:
    raise SystemExit("❌ No GPU attached. Go to Runtime → Change runtime type → GPU.")
print("GPU:", gpu)
if "T4" in gpu:
    print("⚠️ T4: training works but is slow (fp16). If you see NaN / FloatingPointError, set PRECISION to fp32.")

token = os.environ.get("HF_TOKEN")
for getter in ("google.colab", "kaggle_secrets"):
    if token:
        break
    try:
        if getter == "google.colab":
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        else:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if not token:
    raise SystemExit("❌ HF_TOKEN not found. Add a Hugging Face *write* token in the 🔑 Secrets panel as HF_TOKEN.")
os.environ["HF_TOKEN"] = token
print("✅ Hugging Face token found")
for p in (DATA_DIR, CKPT_DIR, SAMPLES_DIR, WORK_DIR):
    p.mkdir(parents=True, exist_ok=True)
''', title=True)

code(f'''
#@title 📦 Install dependencies (~2 minutes)
if not (REPO_DIR / ".git").is_dir():
    !git clone -q https://github.com/Edge0-AI/Audio8_TTS.git {{REPO_DIR}}
!cd {{REPO_DIR}} && git fetch -q origin && git checkout -q {REPO_COMMIT}
!pip install -q "transformers>=4.57.0,<5" "accelerate>=1.10" bitsandbytes num2fawords soundfile tensorboard
import transformers
print("transformers", transformers.__version__)
if int(transformers.__version__.split(".")[0]) >= 5:
    print("⚠️ An older transformers is required: Runtime → Restart session, then Run all again.")
''', title=True)

code("%%writefile fa_tools.py\n" + TOOLS)

code('''
#@title ⬇️ Download the Audio8 base model (≈ 2.5 GB) and connect the Hub
import sys, importlib
sys.path.insert(0, os.getcwd())
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

HUB = fa.HubStore(HF_DATA_REPO, HF_TRAIN_REPO)
print(f"✅ Hub: data → https://huggingface.co/datasets/{HF_DATA_REPO} | checkpoints → https://huggingface.co/{HF_TRAIN_REPO} (private)")
''', title=True)

md("""
## 1️⃣ Data preparation

Each dataset part is downloaded, its audio is converted to Audio8 codec codes, and the result is stored as one small `.tar` in your Hugging Face dataset repo. Parts that already exist there are just downloaded (seconds) instead of being re-encoded. If you get disconnected midway, finished parts are not processed again.

> ⏱️ Encoding takes a few minutes per part the first time.
>
> ℹ️ The data options (`ONLY_HIGH_QUALITY`, `MAX_AUDIO_SECONDS`) only apply the first time each part is processed.
""")

code('''
#@title 🧹 Prepare data (resumes after a disconnect)
PLAN = {"mana": int(MANA_PARTS)}
if USE_GPTINFORMAL:
    PLAN["gptinformal"] = 5
fa.prepare_all(BASE_DIR, WORK_DIR, DATA_DIR, PLAN,
               fa.PrepOptions(max_seconds=float(MAX_AUDIO_SECONDS), only_high_quality=ONLY_HIGH_QUALITY),
               hub=HUB)
TRAIN_JSONL, EVAL_JSONL = fa.build_manifests(
    BASE_DIR, REPO_DIR, WORK_DIR, DATA_DIR, PLAN, max_length=MAX_LENGTH,
    manifest_dir=RUN_DIR / "manifests", hub=HUB, run=RUN_NAME,
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

- Every `SAVE_EVERY_STEPS` steps a checkpoint is saved, audio samples are played **right below**, and the checkpoint and samples are uploaded to your private training repo. Only the newest checkpoint is kept there.
- If the session ends: **Run all** again, on any machine. It finds the newest complete checkpoint (locally or on the Hub) and continues. A checkpoint that was cut off mid-save is set aside automatically.
- `eval_loss` should go down over time. Early on the audio is unintelligible; that is normal.
""")

code('''
#@title 🚀 Start / resume training
refs = fa.load_refs(DATA_DIR)
clone_ref = next((r for r in refs if r["dataset"] == "gptinformal"), None)  # voice-clone test with the second speaker

cfg = fa.TrainConfig(
    base_dir=str(BASE_DIR), repo_dir=str(REPO_DIR),
    train_jsonl=str(TRAIN_JSONL), eval_jsonl=str(EVAL_JSONL),
    output_dir=str(CKPT_DIR), samples_dir=str(SAMPLES_DIR),
    precision=PRECISION, batch_size=int(BATCH_SIZE), grad_accum=int(GRAD_ACCUM),
    learning_rate=float(LEARNING_RATE), epochs=float(EPOCHS),
    save_steps=int(SAVE_EVERY_STEPS), keep_checkpoints=int(KEEP_CHECKPOINTS),
    max_length=MAX_LENGTH, freeze_fast_ar=FREEZE_FAST_AR,
    test_sentences=tuple(TEST_SENTENCES), clone_reference=clone_ref,
    hub=HUB, hub_run=RUN_NAME,
)
FINAL_CKPT = fa.train(cfg)
''', title=True)

code('''
#@title 📈 (Optional) Loss curves in TensorBoard
%load_ext tensorboard
%tensorboard --logdir {CKPT_DIR}/logs
''', title=True)

md("""
## 3️⃣ Listen, test and publish
""")

code('''
#@title 🎧 All samples saved after each checkpoint
SHOW_LAST = 6  #@param {type:"integer"}
import json
from IPython.display import Audio, Markdown, display
index = SAMPLES_DIR / "index.jsonl"
if not index.is_file():
    HUB.pull_run_files(RUN_NAME, "samples", SAMPLES_DIR)
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
#@markdown **Clone your own voice** (optional): a clean 5–10 second clip and its **exact** transcript. In Colab you are asked to upload it; elsewhere set `MY_VOICE_PATH`.
USE_MY_VOICE = False  #@param {type:"boolean"}
UPLOAD_NEW_VOICE = False  #@param {type:"boolean"}
MY_VOICE_PATH = ""  #@param {type:"string"}
REFERENCE_TEXT = ""  #@param {type:"string"}
TEMPERATURE = 0.8  #@param {type:"slider", min:0.3, max:1.2, step:0.05}
SEED = 1234  #@param {type:"integer"}

import gc, time, torch, soundfile as sf
from IPython.display import Audio, display

if CHECKPOINT.strip().lower() != "base" and not fa.list_checkpoints(CKPT_DIR):
    HUB.download_latest_checkpoint(RUN_NAME, CKPT_DIR)
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
    if MY_VOICE_PATH.strip():
        src = MY_VOICE_PATH.strip()
    elif globals().get("_MY_VOICE") is None or UPLOAD_NEW_VOICE:
        from google.colab import files
        src = next(iter(files.upload()))
    else:
        src = None
    if src:
        _MY_VOICE = str(ROOT / "my_voice.wav")
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
#@title 📤 Export the final model (standalone bf16 folder ≈ 2.6 GB, on local disk)
EXPORT_CHECKPOINT = "latest"  #@param {type:"string"}
if not fa.list_checkpoints(CKPT_DIR):
    HUB.download_latest_checkpoint(RUN_NAME, CKPT_DIR)
ckpt = fa.resolve_checkpoint(CKPT_DIR, EXPORT_CHECKPOINT)
EXPORTED = fa.export_model(BASE_DIR, ckpt, EXPORT_DIR / ckpt.name)
print(f"Exported {ckpt.name} to {EXPORTED}")
''', title=True)

code('''
#@title 🤗 Upload the exported model to Hugging Face (public model repo)
UPLOAD_SAMPLES = True  #@param {type:"boolean"}
MODEL_CARD_URL = "https://raw.githubusercontent.com/ard9/Audio8-Persian-TTS/main/docs/MODEL_CARD.md"  #@param {type:"string"}
import urllib.request
from huggingface_hub import HfApi
if "EXPORTED" not in globals():
    raise SystemExit("❌ Run the Export cell first.")
api = HfApi(token=fa.hf_token())
api.create_repo(HF_MODEL_REPO, exist_ok=True)
try:
    card = urllib.request.urlopen(MODEL_CARD_URL, timeout=20).read()
except Exception as exc:
    print(f"⚠️ Could not fetch the model card ({exc}); uploading a minimal one.")
    card = (f"---\\nlicense: apache-2.0\\nlanguage: [fa]\\nbase_model: Audio8/Audio8-TTS-Preview-0.6b\\n"
            f"pipeline_tag: text-to-speech\\n---\\n\\n# Audio8-TTS-Persian\\n\\nCode: {REPO_URL}\\n").encode()
api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=HF_MODEL_REPO)
api.upload_folder(folder_path=str(EXPORTED), repo_id=HF_MODEL_REPO,
                  ignore_patterns=["*.partial", "README.md", ".*", "*.jpeg", "*.jpg"],
                  commit_message=f"{RUN_NAME}: {EXPORTED.name}")
if UPLOAD_SAMPLES:
    steps = [s for s, _ in fa.list_checkpoints(CKPT_DIR)]
    folder = SAMPLES_DIR / f"step_{steps[-1]:07d}" if steps else None
    names = {"s01.wav": "ft_s01.wav", "s02.wav": "ft_s02.wav", "s03.wav": "ft_s03.wav", "clone.wav": "clone_s01.wav"}
    for src, dst in names.items():
        if folder and (folder / src).is_file():
            api.upload_file(path_or_fileobj=str(folder / src), path_in_repo=f"samples/{dst}", repo_id=HF_MODEL_REPO)
print(f"✅ Done: https://huggingface.co/{HF_MODEL_REPO}")
''', title=True)

md("""
## 💡 Tips and troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` (e.g. 2 or 1) and raise `GRAD_ACCUM` (keep their product around 16). |
| `FloatingPointError` / `NaN` on T4 | `PRECISION = "fp32"` (slower but stable). |
| `HF_TOKEN not found` / 401 / 403 | Use a **write** token of the account in `HF_USERNAME`, saved as the `HF_TOKEN` secret with notebook access on. |
| Local disk is full | Lower `KEEP_CHECKPOINTS` to 1 (only the newest local checkpoint is kept; the Hub keeps its own copy). |
| Audio runs on / repeats | Normal in early checkpoints. Keep training or add more data. |
| Some words are mispronounced | More data (higher `MANA_PARTS`) helps the most. |

**How to judge quality as training goes on:**
1. Early: mumbling or non-Persian sounds → normal.
2. Next: Persian-like syllables, but words are unclear.
3. Next: understandable words, intonation still robotic.
4. Finally: fluent, understandable sentences → a good point to add more data or stop.
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
