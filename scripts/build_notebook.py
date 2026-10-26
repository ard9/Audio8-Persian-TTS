"""Builds the Colab notebooks from src/fa_tools.py:

    notebooks/Audio8_Persian_Finetune.ipynb          (Persian, lang "fa")
    notebooks/Audio8_Kurdish_Sorani_Finetune.ipynb   (Central Kurdish / Sorani, lang "ckb")

Both notebooks embed src/fa_tools.py with %%writefile so they run standalone in
Colab, and share every cell except the language-specific ones in LANGS below.
Edit src/fa_tools.py (or this file), then run:

    python scripts/build_notebook.py

CI fails if a committed notebook is out of sync.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = (ROOT / "src" / "fa_tools.py").read_text(encoding="utf-8")
REPO_URL = "https://github.com/ard9/Audio8-Persian-TTS"
REPO_COMMIT = "07e40f5d0b03fc473635ef378654bfb581027ac3"
NOTEBOOKS = {
    "fa": "Audio8_Persian_Finetune.ipynb",
    "ckb": "Audio8_Kurdish_Sorani_Finetune.ipynb",
}

SECTION = "# ---------------------------------------------------------------------------\n# "

# src/fa_tools.py is written to disk in several cells, one per section, each with
# a short explanation. The first cell creates the file (%%writefile) and the
# others append to it (%%writefile -a), so the result is identical to the source.
# Each entry: (text that starts the chunk, explanation shown before the cell).
TOOLS_PARTS = [
    ("", """
**Part 1 — Imports and dataset list.** Starts `fa_tools.py`: the Python imports, audio constants (44.1 kHz, ≈ 21.5 codec frames per second) and the Hugging Face repos/files of the datasets (Mana-TTS and GPTInformal for Persian, central-kurdish-tts4all for Central Kurdish).
"""),
    (SECTION + "Text normalization", """
**Part 2 — Text normalization (Persian and Central Kurdish).** Unifies Arabic letter variants, removes diacritics and invisible characters (incl. ZWNJ), spells out numbers as words and splits long text into sentences. Persian and Sorani have separate rules (Sorani keeps its own letters such as ە ڕ ڵ ۆ ێ), chosen by `LANG`. The same function is used for training and testing, so the model always sees text in the same form.
"""),
    (SECTION + "Environment helpers", """
**Part 3 — Environment helpers.** Detects the GPU and picks precision, batch size and gradient accumulation for it; loads the Audio8 model, processor and config; reads the `HF_TOKEN` secret.
"""),
    (SECTION + "Hugging Face Hub storage", """
**Part 4 — Hugging Face Hub storage.** The `HubStore` class uploads/downloads prepared data parts, manifests, checkpoints and audio samples, so training can resume on any machine. It also deletes old checkpoints on the Hub.
"""),
    (SECTION + "Reading the source datasets", """
**Part 5 — Reading the dataset files.** Reads parquet parts row by row (low memory), or WAV files listed in a `metadata.csv`, decodes the audio and resamples it to 44.1 kHz mono.
"""),
    (SECTION + "Codec encoding", """
**Part 6 — Codec encoder.** `CodecEncoder` turns waveforms into Audio8 codec codes (the "audio tokens" the model learns to generate). `PrepOptions` holds the data filters (min/max length, high-quality only).
"""),
    ("def prepare_shard(", """
**Part 7 — Data preparation per part.** `prepare_shard` encodes one dataset part and packs it into a `.tar` on the Hub (or just downloads it if it already exists). `prepare_all` does this for every part and skips finished ones after a disconnect.
"""),
    (SECTION + "Manifests", """
**Part 8 — Manifests.** Builds the train/eval `.jsonl` lists (text + codec file per clip). Some clips get a short reference clip of the same speaker, so the model also learns voice cloning. Clips that don't fit in `MAX_LENGTH` tokens are dropped, and a seeded shuffle sets aside up to 200 clips for evaluation.
"""),
    (SECTION + "Speech generation", """
**Part 9 — Speech generation.** `synthesize` turns text (and optionally a reference voice) into audio; `write_samples` saves and plays the test sentences after each checkpoint.
"""),
    (SECTION + "Local checkpoints", """
**Part 10 — Local checkpoints.** Lists checkpoints, checks that a checkpoint was saved completely, finds the one to resume from, and removes older ones.
"""),
    (SECTION + "Training", """
**Part 11 — Training: memory saving.** Activation (gradient) checkpointing for the slow AR layers: it uses less GPU memory at the cost of some extra compute.
"""),
    ("@dataclass\nclass TrainConfig:", """
**Part 12 — Training settings.** `TrainConfig` holds all training options. **Warmup:** `warmup_ratio = 0.03` — the learning rate rises from 0 to `LEARNING_RATE` during the first 3% of steps, then decreases along a cosine curve. On resume the scheduler state is restored, so warmup is not repeated. `gpu_memory_stats` reads the current, peak and reserved GPU memory. `_resolve_training_settings` fills in the automatic values (precision, batch size, 8-bit Adam).
"""),
    ("def train(cfg: TrainConfig)", """
**Part 13 — The training loop.** `train()` loads the model with fp32 master weights, builds the datasets and the Hugging Face `Trainer` (cosine schedule with warmup, AdamW, gradient clipping at 1.0), and adds two callbacks: one logs GPU memory (VRAM) with every loss line and in TensorBoard and prints the peak at each checkpoint; the other, after every checkpoint, generates test audio, rotates old checkpoints and backs everything up to the Hub. It resumes from the newest checkpoint (local or Hub) automatically.
"""),
    (SECTION + "Testing and export", """
**Part 14 — Testing and export.** Picks a checkpoint (`latest`, a step number or `base`), loads it for inference, and exports a standalone bf16 model folder for upload.
"""),
]


def split_tools(text):
    """Split src/fa_tools.py at the markers in TOOLS_PARTS (each must occur once, in order)."""
    starts = []
    for marker, _ in TOOLS_PARTS:
        if not marker:
            starts.append(0)
            continue
        if text.count(marker) != 1:
            raise SystemExit(f"marker not found exactly once in src/fa_tools.py: {marker!r}")
        starts.append(text.index(marker))
    if starts != sorted(starts):
        raise SystemExit("TOOLS_PARTS markers are out of order")
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


# ---------------------------------------------------------------------------
# Language-specific cells
# ---------------------------------------------------------------------------

COMMON_PATHS = '''
import os
from pathlib import Path
ROOT = Path("/content") if Path("/content").is_dir() else Path.cwd()   # Colab or Kaggle
WORK_DIR = ROOT / "work"                      # temporary local disk
BASE_DIR = ROOT / __BASE_DIR__
REPO_DIR = ROOT / "Audio8_TTS"
DATA_DIR = WORK_DIR / "data_cache"            # local cache; the Hub holds the real copy
RUN_DIR = WORK_DIR / "runs" / RUN_NAME          # local cache; the Hub holds the real copy
CKPT_DIR = RUN_DIR / "checkpoints"
SAMPLES_DIR = RUN_DIR / "samples"
EXPORT_DIR = ROOT / "export"
MAX_LENGTH = 1024
print(f"✅ Settings saved. Run: {RUN_NAME} → https://huggingface.co/{HF_TRAIN_REPO}")
'''

TRAINING_SETTINGS = '''#@markdown ---
#@markdown **Training** — 0 means automatic, based on the GPU.
EPOCHS = 2  #@param {type:"number"}
LEARNING_RATE = 3e-5  #@param {type:"number"}
SAVE_EVERY_STEPS = 1000  #@param {type:"integer"}
KEEP_CHECKPOINTS = 1  #@param {type:"integer"}
PRECISION = "auto"  #@param ["auto", "bf16", "fp16", "fp32"]
BATCH_SIZE = 0  #@param {type:"integer"}
GRAD_ACCUM = 0  #@param {type:"integer"}
FREEZE_FAST_AR = False  #@param {type:"boolean"}
'''

LANGS = {
    "fa": {
        "intro": f"""
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
""",
        "settings": '''
#@title ⚙️ Main settings (review before running)
#@markdown **Run name** — keep it unchanged to continue the same training. Use a new name for a new experiment.
RUN_NAME = "audio8_fa_v2"  #@param {type:"string"}
LANG = "fa"  # text normalization rules: Persian
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
''' + TRAINING_SETTINGS + '''
# Sentences read aloud after every checkpoint. They stay in Persian because
# they are the model's input text; edit them freely.
TEST_SENTENCES = [
    "سلام، امروز هوا خیلی خوب است و می‌خواهم کمی در پارک قدم بزنم.",
    "این یک آزمایش برای سنجش کیفیت مدل تبدیل متن به گفتار فارسی است.",
    "دیروز با دوستام رفتیم سینما، فیلمش واقعاً عالی بود!",
]
''' + COMMON_PATHS.replace("__BASE_DIR__", '"base_model"'),
        "base_cell": '''
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
''',
        "data_md": """
## 1️⃣ Data preparation

Each dataset part is downloaded, its audio is converted to Audio8 codec codes, and the result is stored as one small `.tar` in your Hugging Face dataset repo. Parts that already exist there are just downloaded (seconds) instead of being re-encoded. If you get disconnected midway, finished parts are not processed again.

> ⏱️ Encoding takes a few minutes per part the first time.
>
> ℹ️ The data options (`ONLY_HIGH_QUALITY`, `MAX_AUDIO_SECONDS`) only apply the first time each part is processed.
""",
        "prep_cell": '''
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
''',
        "start_label": "Base model",
        "clone_ref": 'clone_ref = next((r for r in refs if r["dataset"] == "gptinformal"), None)  # voice-clone test with the second speaker',
        "test_text": "سلام! این صدای مدل فارسی من است. امیدوارم کیفیتش خوب باشد.",
        "card_url": "https://raw.githubusercontent.com/ard9/Audio8-Persian-TTS/main/docs/MODEL_CARD.md",
        "card_fallback": '''card = (f"---\\nlicense: apache-2.0\\nlanguage: [fa]\\nbase_model: Audio8/Audio8-TTS-Preview-0.6b\\n"
            f"pipeline_tag: text-to-speech\\n---\\n\\n# Audio8-TTS-Persian\\n\\nCode: {REPO_URL}\\n").encode()''',
        "tips": """
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
""",
    },
    "ckb": {
        "intro": f"""
# 🎙️ Fine-tuning Audio8 TTS for Central Kurdish (Sorani)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ard9/Audio8-Persian-TTS/blob/main/notebooks/Audio8_Kurdish_Sorani_Finetune.ipynb) · Project: {REPO_URL}

This notebook fine-tunes Audio8 TTS on **[central-kurdish-tts4all](https://huggingface.co/datasets/aranemini/central-kurdish-tts4all)**: about 35 hours of read Central Kurdish (Sorani) speech from 3 speakers (a male and a female audiobook narrator and a male studio voice).

You can start from the original **Audio8-TTS-Preview-0.6B** or from the **Persian fine-tune** ([ArminRahimi/Audio8-TTS-Persian](https://huggingface.co/ArminRahimi/Audio8-TTS-Persian)). Persian and Sorani share most of their script, so the Persian model already links these letters to speech sounds. Set `INIT_MODEL` and use a different `RUN_NAME` for each, to compare them.

**⚠️ Non-commercial data.** The dataset is licensed **CC BY-NC-ND 4.0**. Here that means:
- the prepared data goes to a **private** dataset repo (sharing modified versions of the data is not allowed);
- the trained model may be shared **for non-commercial use only** (research, education, demos); its model card uses CC BY-NC 4.0 and credits the dataset.

**What happens automatically:**
- 💾 Prepared data, checkpoints and audio samples are stored on the **Hugging Face Hub**, so you can continue on any machine.
- 🔁 If the session ends, run **Run all** again — on the same or a different machine. Training resumes from the last checkpoint.
- 🔊 After **every checkpoint**, a few test sentences are generated and played right here.
- 🧪 After training, use the **Test** cell to try any text with any checkpoint (or your own voice), then **Upload** the model.

**Before you start:**
1. **Runtime → Change runtime type** → pick a GPU. **L4 or A100 (Colab Pro) is recommended**; the free T4 works but is slower.
2. Create a Hugging Face **write** token at https://huggingface.co/settings/tokens and add it in Colab's 🔑 **Secrets** panel as `HF_TOKEN` (enable *Notebook access*).
   Data and checkpoints go to **private** repos; only the newest checkpoint is kept (~4.4 GB).
3. Review the settings in the next cell, then **Runtime → Run all**.

Works on Kaggle too (enable Internet and a GPU, add `HF_TOKEN` under *Add-ons → Secrets*).
""",
        "settings": '''
#@title ⚙️ Main settings (review before running)
#@markdown **Run name** — keep it unchanged to continue the same training. Use a new name for a new experiment (e.g. one per `INIT_MODEL`).
RUN_NAME = "audio8_ckb_v1"  #@param {type:"string"}
LANG = "ckb"  # text normalization rules: Central Kurdish (Sorani)
#@markdown **Starting point** — `persian`: the Persian fine-tune (shares the script); `base`: the original Audio8.
INIT_MODEL = "persian"  #@param ["persian", "base"]
#@markdown ---
#@markdown **Hugging Face storage** — data, checkpoints and samples are kept here (needs `HF_TOKEN`).
HF_USERNAME = "ArminRahimi"  #@param {type:"string"}
HF_DATA_REPO = f"{HF_USERNAME}/audio8-ckb-codes"            # PRIVATE dataset (CC BY-NC-ND source)
HF_TRAIN_REPO = f"{HF_USERNAME}/audio8-ckb-training"        # private: checkpoints, samples, manifests
HF_MODEL_REPO = f"{HF_USERNAME}/Audio8-TTS-Kurdish-Sorani"  # the final model (non-commercial license)
#@markdown ---
#@markdown **Data** — the dataset is split into 12 parts (each ≈ 3 hours, all 3 speakers). 4 parts ≈ 12 h for a quick test; 12 parts ≈ 35 h for the best quality.
TTS4ALL_PARTS = 12  #@param {type:"slider", min:1, max:12, step:1}
MAX_AUDIO_SECONDS = 20  #@param {type:"number"}
#@markdown **Default voice** — the speaker used when no reference clip is given. `auto` = the speaker with the most audio; or a name such as `ckb_tts4all:fatih`. Other speakers are trained with a reference clip, so they can be cloned.
DEFAULT_VOICE = "auto"  #@param {type:"string"}
''' + TRAINING_SETTINGS + '''
# Sentences read aloud after every checkpoint. They stay in Sorani because
# they are the model's input text; edit them freely.
TEST_SENTENCES = [
    "سڵاو، ئەمڕۆ کەشوهەوا زۆر خۆشە و دەمەوێت کەمێک لە پارک پیاسە بکەم.",
    "ئەمە تاقیکردنەوەیەکە بۆ هەڵسەنگاندنی کوالیتی مۆدێلی گۆڕینی دەق بۆ دەنگی کوردی.",
    "دوێنێ لەگەڵ هاوڕێکانم چووین بۆ سینەما، فیلمەکە بەڕاستی زۆر جوان بوو!",
]
''' + COMMON_PATHS.replace("__BASE_DIR__", 'f"init_{INIT_MODEL}"'),
        "base_cell": '''
#@title ⬇️ Download the starting model (≈ 2.5 GB) and connect the Hub
import sys, importlib
sys.path.insert(0, os.getcwd())
import fa_tools as fa
importlib.reload(fa)
fa.add_repo_to_path(REPO_DIR)

from huggingface_hub import snapshot_download
INIT_REPOS = {
    "persian": ("ArminRahimi/Audio8-TTS-Persian",),
    "base": ("Edge0/Audio8-TTS-Preview-0.6b", "Audio8/Audio8-TTS-Preview-0.6b"),
}[INIT_MODEL]
BASE_MODEL_ID = INIT_REPOS[0]
if not (BASE_DIR / "codec.pth").is_file():
    for repo_id in INIT_REPOS:
        try:
            snapshot_download(repo_id, local_dir=str(BASE_DIR), token=fa.hf_token(), ignore_patterns=["samples/*"])
            BASE_MODEL_ID = repo_id
            break
        except Exception as exc:
            print(f"{repo_id}: {exc}")
    else:
        raise SystemExit("❌ Could not download the starting model.")
print(f"✅ Starting model ({INIT_MODEL}) ready:", sorted(p.name for p in BASE_DIR.iterdir()))

# The source data is CC BY-NC-ND 4.0, so the prepared data repo is kept private.
HUB = fa.HubStore(HF_DATA_REPO, HF_TRAIN_REPO, data_private=True, data_card=fa.DATA_CARD_CKB)
print(f"✅ Hub: data → https://huggingface.co/datasets/{HF_DATA_REPO} (private) | checkpoints → https://huggingface.co/{HF_TRAIN_REPO} (private)")
''',
        "data_md": """
## 1️⃣ Data preparation

Each part of the dataset (≈ 3 hours, WAV files listed in `metadata.csv`) is downloaded, its audio is converted to Audio8 codec codes, and the result is stored as one small `.tar` in your **private** Hugging Face dataset repo. Parts that already exist there are just downloaded (seconds) instead of being re-encoded. If you get disconnected midway, finished parts are not processed again.

The text is normalized with the Sorani rules (`LANG = "ckb"`): Arabic letter variants are unified, numbers are spelled out in Kurdish, and the Kurdish letters are kept as they are.

> ⏱️ Downloading and encoding take a few minutes per part the first time.
>
> ℹ️ `MAX_AUDIO_SECONDS` only applies the first time each part is processed.
""",
        "prep_cell": '''
#@title 🧹 Prepare data (resumes after a disconnect)
PLAN = {"ckb_tts4all": int(TTS4ALL_PARTS)}
fa.prepare_all(BASE_DIR, WORK_DIR, DATA_DIR, PLAN,
               fa.PrepOptions(max_seconds=float(MAX_AUDIO_SECONDS), only_high_quality=False),
               hub=HUB)
TRAIN_JSONL, EVAL_JSONL = fa.build_manifests(
    BASE_DIR, REPO_DIR, WORK_DIR, DATA_DIR, PLAN, max_length=MAX_LENGTH,
    default_voice=DEFAULT_VOICE.strip() or "auto",
    manifest_dir=RUN_DIR / "manifests", hub=HUB, run=RUN_NAME,
)
''',
        "start_label": "Starting model",
        "clone_ref": 'clone_ref = fa.pick_clone_reference(refs, RUN_DIR / "manifests")  # voice-clone test with another speaker',
        "test_text": "سڵاو! ئەمە دەنگی مۆدێلی کوردیی منە. هیوادارم کوالیتییەکەی باش بێت.",
        "card_url": "https://raw.githubusercontent.com/ard9/Audio8-Persian-TTS/main/docs/MODEL_CARD_ckb.md",
        "card_fallback": '''card = (f"---\\nlicense: cc-by-nc-4.0\\nlanguage: [ckb]\\nbase_model: {BASE_MODEL_ID}\\n"
            f"datasets: [aranemini/central-kurdish-tts4all]\\npipeline_tag: text-to-speech\\n---\\n\\n"
            f"# Audio8-TTS-Kurdish-Sorani\\n\\nNon-commercial use only (training data: CC BY-NC-ND 4.0). Code: {REPO_URL}\\n").encode()''',
        "tips": """
## 💡 Tips and troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` (e.g. 2 or 1) and raise `GRAD_ACCUM` (keep their product around 16). |
| `FloatingPointError` / `NaN` on T4 | `PRECISION = "fp32"` (slower but stable). |
| `HF_TOKEN not found` / 401 / 403 | Use a **write** token of the account in `HF_USERNAME`, saved as the `HF_TOKEN` secret with notebook access on. |
| Local disk is full | Lower `KEEP_CHECKPOINTS` to 1 (only the newest local checkpoint is kept; the Hub keeps its own copy). |
| Audio runs on / repeats | Normal in early checkpoints. Keep training or add more data. |
| A Persian accent remains (`INIT_MODEL = "persian"`) | Train longer, or compare with a run that starts from `base` (use a new `RUN_NAME`). |
| Some words are mispronounced | More data (higher `TTS4ALL_PARTS`) helps the most. Check how the text was written: the model learns from the dataset's spelling. |

**How to judge quality as training goes on:**
1. Early: mumbling, or Persian-sounding speech when starting from the Persian model → normal.
2. Next: Kurdish-like syllables, but words are unclear.
3. Next: understandable words, intonation still robotic.
4. Finally: fluent, understandable sentences → a good point to stop.

**License reminder:** share the model only for non-commercial use, credit [central-kurdish-tts4all](https://huggingface.co/datasets/aranemini/central-kurdish-tts4all), and keep the data repo private.
""",
    },
}


# ---------------------------------------------------------------------------
# Notebook assembly (shared by all languages)
# ---------------------------------------------------------------------------


def build(lang):
    L = LANGS[lang]
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})

    def code(text, title=None):
        meta = {"cellView": "form"} if title else {}
        cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                      "source": text.strip("\n")})

    md(L["intro"])
    code(L["settings"], title=True)

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

    md("""
## 🧰 Helper library (`fa_tools.py`)

The next cells write the project's helper library, `fa_tools.py`, to disk, one section per cell. They only save code (nothing runs yet), so they finish instantly. Run them all, in order: the first cell creates the file and the rest add to it.
""")
    for i, ((_, explanation), chunk) in enumerate(zip(TOOLS_PARTS, split_tools(TOOLS))):
        md(explanation)
        # Not via code(): the chunk keeps its trailing newlines so the joined file matches the source exactly.
        magic = "%%writefile fa_tools.py\n" if i == 0 else "%%writefile -a fa_tools.py\n"
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": magic + chunk})

    code(L["base_cell"], title=True)
    md(L["data_md"])
    code(L["prep_cell"], title=True)

    start = L["start_label"]
    code(f'''
#@title 🔈 (Optional) {start}'s voice before training — for comparison
RUN_BASELINE = True  #@param {{type:"boolean"}}
if RUN_BASELINE and not (SAMPLES_DIR / "base").is_dir():
    import gc, torch
    model, processor = fa.load_for_inference(BASE_DIR, None)
    fa.write_samples(model, processor, SAMPLES_DIR / "base", TEST_SENTENCES[:2], None, 500, "{start} (before training)", lang=LANG)
    del model, processor; gc.collect(); torch.cuda.empty_cache()
elif RUN_BASELINE:
    print("{start} samples already exist:", SAMPLES_DIR / "base")
''', title=True)

    md("""
## 2️⃣ Training

- Every `SAVE_EVERY_STEPS` steps a checkpoint is saved, audio samples are played **right below**, and the checkpoint and samples are uploaded to your private training repo. Only the newest checkpoint is kept there.
- If the session ends: **Run all** again, on any machine. It finds the newest complete checkpoint (locally or on the Hub) and continues. A checkpoint that was cut off mid-save is set aside automatically.
- `eval_loss` should go down over time. Early on the audio is unintelligible; that is normal.
""")

    code(f'''
#@title 🚀 Start / resume training
refs = fa.load_refs(DATA_DIR)
{L["clone_ref"]}

cfg = fa.TrainConfig(
    base_dir=str(BASE_DIR), repo_dir=str(REPO_DIR),
    train_jsonl=str(TRAIN_JSONL), eval_jsonl=str(EVAL_JSONL),
    output_dir=str(CKPT_DIR), samples_dir=str(SAMPLES_DIR),
    precision=PRECISION, batch_size=int(BATCH_SIZE), grad_accum=int(GRAD_ACCUM),
    learning_rate=float(LEARNING_RATE), epochs=float(EPOCHS),
    save_steps=int(SAVE_EVERY_STEPS), keep_checkpoints=int(KEEP_CHECKPOINTS),
    max_length=MAX_LENGTH, freeze_fast_ar=FREEZE_FAST_AR,
    test_sentences=tuple(TEST_SENTENCES), lang=LANG, clone_reference=clone_ref,
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

    code(f'''
#@title 🎧 All samples saved after each checkpoint
SHOW_LAST = 6  #@param {{type:"integer"}}
import json
from IPython.display import Audio, Markdown, display
index = SAMPLES_DIR / "index.jsonl"
if not index.is_file():
    HUB.pull_run_files(RUN_NAME, "samples", SAMPLES_DIR)
entries = [json.loads(l) for l in index.read_text("utf-8").splitlines() if l.strip()] if index.is_file() else []
seen = {{}}
for e in entries:
    seen[e["step"]] = e  # latest record for each step
if (SAMPLES_DIR / "base").is_dir():
    display(Markdown("### {start} (before training)"))
    for f in sorted((SAMPLES_DIR / "base").glob("*.wav")):
        display(Audio(str(f)))
for step in sorted(seen)[-int(SHOW_LAST):]:
    e = seen[step]
    loss = f" — eval_loss = {{e['eval_loss']:.3f}}" if e.get("eval_loss") is not None else ""
    display(Markdown(f"### Step {{step}}{{loss}} ({{e.get('time', '')}})"))
    folder = SAMPLES_DIR / f"step_{{step:07d}}"
    for f in sorted(folder.glob("*.wav")):
        display(Markdown(f"`{{f.name}}`")); display(Audio(str(f)))
if not seen:
    print("No checkpoint saved yet.")
''', title=True)

    code(f'''
#@title 🧪 Test the model with your own text
#@markdown `latest` = newest checkpoint, or a step number (e.g. `1200`), or `base` for the starting model.
CHECKPOINT = "latest"  #@param {{type:"string"}}
TEXT = "{L["test_text"]}"  #@param {{type:"string"}}
#@markdown **Clone your own voice** (optional): a clean 5–10 second clip and its **exact** transcript. In Colab you are asked to upload it; elsewhere set `MY_VOICE_PATH`.
USE_MY_VOICE = False  #@param {{type:"boolean"}}
UPLOAD_NEW_VOICE = False  #@param {{type:"boolean"}}
MY_VOICE_PATH = ""  #@param {{type:"string"}}
REFERENCE_TEXT = ""  #@param {{type:"string"}}
TEMPERATURE = 0.8  #@param {{type:"slider", min:0.3, max:1.2, step:0.05}}
SEED = 1234  #@param {{type:"integer"}}

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
                                    max_frames=1200, temperature=float(TEMPERATURE), seed=int(SEED), lang=LANG)
out_dir = RUN_DIR / "tests"; out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{{time.strftime('%Y%m%d_%H%M%S')}}_{{ckpt.name if ckpt else 'base'}}.wav"
sf.write(out, audio, sr)
print(f"Checkpoint: {{ckpt.name if ckpt else 'base'}} | audio {{len(audio)/sr:.1f}}s | generated in {{time.time()-t0:.1f}}s"
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

    code(f'''
#@title 🤗 Upload the exported model to Hugging Face (model repo)
UPLOAD_SAMPLES = True  #@param {{type:"boolean"}}
MODEL_CARD_URL = "{L["card_url"]}"  #@param {{type:"string"}}
import urllib.request
from huggingface_hub import HfApi
if "EXPORTED" not in globals():
    raise SystemExit("❌ Run the Export cell first.")
api = HfApi(token=fa.hf_token())
api.create_repo(HF_MODEL_REPO, exist_ok=True)
try:
    card = urllib.request.urlopen(MODEL_CARD_URL, timeout=20).read()
except Exception as exc:
    print(f"⚠️ Could not fetch the model card ({{exc}}); uploading a minimal one.")
    {L["card_fallback"]}
api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=HF_MODEL_REPO)
api.upload_folder(folder_path=str(EXPORTED), repo_id=HF_MODEL_REPO,
                  ignore_patterns=["*.partial", "README.md", ".*", "*.jpeg", "*.jpg"],
                  commit_message=f"{{RUN_NAME}}: {{EXPORTED.name}}")
if UPLOAD_SAMPLES:
    steps = [s for s, _ in fa.list_checkpoints(CKPT_DIR)]
    folder = SAMPLES_DIR / f"step_{{steps[-1]:07d}}" if steps else None
    names = {{"s01.wav": "ft_s01.wav", "s02.wav": "ft_s02.wav", "s03.wav": "ft_s03.wav", "clone.wav": "clone_s01.wav"}}
    for src, dst in names.items():
        if folder and (folder / src).is_file():
            api.upload_file(path_or_fileobj=str(folder / src), path_in_repo=f"samples/{{dst}}", repo_id=HF_MODEL_REPO)
print(f"✅ Done: https://huggingface.co/{{HF_MODEL_REPO}}")
''', title=True)

    md(L["tips"])

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
        cell["source"] = cell["source"].splitlines(keepends=True)
    return notebook


def render(lang: str = "fa") -> str:
    return json.dumps(build(lang), ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    for lang, name in NOTEBOOKS.items():
        out = ROOT / "notebooks" / name
        out.write_text(render(lang), encoding="utf-8")
        print("wrote", out, len(build(lang)["cells"]), "cells")
