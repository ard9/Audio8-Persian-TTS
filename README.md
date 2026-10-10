# Audio8-Persian-TTS 🎙️🇮🇷

**Persian (Farsi) fine-tuning of [Audio8 TTS](https://github.com/Edge0-AI/Audio8_TTS)** — a 0.6B-parameter multilingual text-to-speech model with zero-shot voice cloning that does **not** officially support Persian.

To our knowledge (checked October 2026), this is the **first public Persian adaptation of Audio8 TTS**, and it was trained on free or low-cost Google Colab hardware.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ard9/Audio8-Persian-TTS/blob/main/notebooks/Audio8_Persian_Finetune.ipynb)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Audio8--TTS--Persian-yellow)](https://huggingface.co/ArminRahimi/Audio8-TTS-Persian)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/ard9/Audio8-Persian-TTS/actions/workflows/tests.yml/badge.svg)](https://github.com/ard9/Audio8-Persian-TTS/actions/workflows/tests.yml)

[فارسی](README_fa.md)

---

## 🔊 Samples

All samples come from the v0.1 run (step 1257); none are cherry-picked: each is the single fixed-seed sample generated automatically after the checkpoint was saved.

| Text | Base model (before) | Fine-tuned (step 1257) | Voice clone (step 1257) |
|---|---|---|---|
| سلام، امروز هوا خیلی خوب است و می‌خواهم کمی در پارک قدم بزنم. | [base_s01.wav](samples/base_s01.wav) | [ft_s01.wav](samples/ft_s01.wav) | [clone_s01.wav](samples/clone_s01.wav) |
| این یک آزمایش برای سنجش کیفیت مدل تبدیل متن به گفتار فارسی است. | [base_s02.wav](samples/base_s02.wav) | [ft_s02.wav](samples/ft_s02.wav) | — |
| دیروز با دوستام رفتیم سینما، فیلمش واقعاً عالی بود! | — | [ft_s03.wav](samples/ft_s03.wav) | — |

**Training progression** (sentence 1 at each checkpoint): [step 300](samples/progress_step0300.wav) · [step 600](samples/progress_step0600.wav) · [step 900](samples/progress_step0900.wav) · [step 1200](samples/progress_step1200.wav) · [step 1257 (final)](samples/ft_s01.wav)

The voice-clone sample uses a reference clip from GPTInformal-Persian (a second speaker). Full per-checkpoint log: [`docs/training_samples_index.jsonl`](docs/training_samples_index.jsonl).

## 📊 Results (v0.1)

| | Value |
|---|---|
| Training data | 10 of 77 Mana-TTS parts (≈ 15 h before filtering; only HIGH-alignment clips of 1–20 s kept) + all 5 parts of GPTInformal-Persian (≈ 6 h) |
| Training steps / epochs | 1257 optimizer steps / 3 epochs |
| Hardware and training time | 1× TODO GPU on Google Colab, ≈ 2.2 h of training (≈ 6 s per step, including evaluation and sample generation) |
| Eval loss (slow + fast AR, held-out) | 9.00 (step 300) → 8.82 (600) → 8.82 (900) → **8.80** (1200) |
| Generation | All test sentences reached end-of-speech at every checkpoint (no run-on or truncated outputs) |
| CER (Whisper ASR on generated speech) | not measured yet |
| MOS (native listeners) | not measured yet |

<!-- TODO: replace "TODO GPU" with the GPU name printed by the notebook (e.g. L4 or T4), and add one honest sentence on how the samples sound. If you changed any default setting (epochs, learning rate), update the table. -->


## ✨ What's in this repo

| Path | What it is |
|---|---|
| [`notebooks/Audio8_Persian_Finetune.ipynb`](notebooks/Audio8_Persian_Finetune.ipynb) | One-click Colab notebook: data preparation, training, per-checkpoint audio samples, testing, export. |
| [`src/fa_tools.py`](src/fa_tools.py) | All the logic the notebook uses (embedded into it with `%%writefile`). |
| [`scripts/synthesize.py`](scripts/synthesize.py) | Command-line inference from an exported model, a Hugging Face ID, or a raw checkpoint. |
| [`scripts/upload_to_hf.py`](scripts/upload_to_hf.py) | Publishes an exported model and its model card to the Hugging Face Hub. |
| [`scripts/build_notebook.py`](scripts/build_notebook.py) | Rebuilds the notebook from `src/fa_tools.py`. |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Hugging Face model card template. |
| [`docs/PUBLISHING.md`](docs/PUBLISHING.md) | Step-by-step checklist for GitHub and Hugging Face releases. |
| [`tests/`](tests) | CPU end-to-end test of the whole pipeline with a tiny stand-in model. |

## 🚀 Quick start (Colab)

1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → GPU.** An L4 or A100 (Colab Pro) is recommended; the free T4 works but is slower.
3. Review the settings cell, then **Runtime → Run all**.
4. If Colab disconnects, run **Run all** again: data, checkpoints and samples live on your Google Drive and training resumes from the last complete checkpoint.

Default settings train on 10 parts of Mana-TTS (≈ 15 h). Raise `MANA_PARTS` (up to 77, ≈ 114 h) for better quality.

## 🔧 How it works

```
Mana-TTS / GPTInformal parquet  ──►  44.1 kHz audio  ──►  Audio8 codec codes (10 × T)
            (downloaded once)                               saved to Google Drive as one .tar per part
                                                                       │
                     train/eval manifests (some rows with a same-speaker reference clip)
                                                                       │
  fine-tune both DualAR branches ──► checkpoint on Drive ──► auto-generate test sentences ──► listen
                                                                       │
                                     export standalone bf16 model ──► Hugging Face Hub
```

- **Data is prepared once.** Each dataset part is downloaded, encoded to codec codes and stored on Drive (codes for all 114 h of Mana-TTS take < 500 MB). Later sessions restore it in seconds.
- **Persian voice cloning.** Mana-TTS rows are trained mostly without a reference (so the default voice is Mana's speaker). GPTInformal rows always use a reference clip of the same speaker, teaching the model to clone voices in Persian.
- **Robust on Colab.** Incomplete checkpoints from a disconnect are detected and skipped; old checkpoints are rotated by step number and removed from Drive's trash so storage does not fill up.

### Fixes to the upstream SFT pipeline

The stock `audio8_tts_sft.sh` (repo commit `07e40f5`) does not run as-is on a single Colab GPU. This project trains through the upstream dataset, loss and trainer classes, with these changes:

| Upstream behavior | Change here |
|---|---|
| Passes `--deepspeed configs/deepspeed_zero2.json`, but that file is not in the repo. | No DeepSpeed (unnecessary on one GPU). |
| Enables `--gradient_checkpointing`, but `ArkttsModel` does not implement Transformers' checkpointing API, so `Trainer` raises an error. | Activation checkpointing implemented directly on the slow-AR layers. |
| `forward(**kwargs)` makes `Trainer` assume the loss is already normalized for gradient accumulation; the repo's loss is a plain mean, so gradients are scaled by the accumulation factor. | `trainer.model_accepts_loss_kwargs = False`. |
| Loads and trains weights in pure bf16, where small learning-rate updates are rounded away. | fp32 master weights + bf16/fp16 autocast, 8-bit AdamW. |
| Evaluation gathers full 155k-vocabulary logits. | `prediction_loss_only=True`. |

### Text normalization

`fa_tools.normalize_fa()` is applied identically at training and inference time: Arabic → Persian letters (ي→ی, ك→ک), Persian/Arabic digits → spoken words (`۱۴۰۲` → «هزار و چهارصد و دو»), percent, decimals, thousands separators, diacritics removal, and whitespace/punctuation cleanup. Zero-width non-joiners are removed because the upstream `clean_text()` strips them anyway.

## 💻 Local inference

```bash
git clone https://github.com/ard9/Audio8-Persian-TTS.git && cd Audio8-Persian-TTS
pip install -r requirements.txt   # plus a CUDA build of PyTorch for GPU inference
python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Persian --text "سلام! حال شما چطور است؟" --output hello.wav

# voice cloning (5–10 s clean clip + its exact transcript)
python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Persian --text "سلام" \
    --reference-audio me.wav --reference-text "متن دقیق صدای مرجع" --output clone.wav
```

## ⚠️ Limitations

- **Training data:** v0.1 uses only 10 of 77 Mana-TTS parts (≈ 15 h) from essentially **one speaker**, plus ≈ 6 h of a second speaker (GPTInformal). Prosody and voice variety are limited, and cloning of unseen voices is weaker than in Audio8's officially supported languages.
- **Compute:** a single Colab GPU for ≈ 2.2 hours (1257 steps). Training on all 114 h of Mana-TTS, longer schedules and hyperparameter search were not possible within this budget. The eval loss flattened after ~600 steps, so v0.1 is likely limited more by data and schedule than by convergence.
- **No phonemizer (G2P):** Persian script omits short vowels, so the model infers pronunciation from data. Homographs (e.g. کُرد / کَرد), the ezafe (-e) and rare words can be mispronounced.
- **Text coverage:** numbers are expanded to words, but dates, abbreviations, Latin-script words and code-switching are not handled specially.
- **Base model:** Audio8 TTS is a *Preview* checkpoint; Persian was never part of its pre-training.
- **Evaluation:** no human MOS study and no ASR-based intelligibility (CER) measurement yet; quality claims are based on listening only.
- Inputs longer than ~150 characters are split into sentences and synthesized separately.

## 🛡️ Responsible use

This model can imitate voices. Only clone a voice with the speaker's explicit consent, and clearly disclose synthetic audio. Do not use it for impersonation, fraud or misinformation. The Mana-TTS and GPTInformal authors explicitly prohibit using their data to imitate the speakers' voices for malicious purposes.

## 🗺️ Roadmap

- [ ] Automatic intelligibility (CER with a Persian Whisper model) and speaker-similarity evaluation per checkpoint
- [ ] Human MOS evaluation with native speakers
- [ ] Data-size ablation (15 h vs. 50 h vs. 114 h)
- [ ] Multi-speaker training (e.g. [ParsVoice](https://arxiv.org/abs/2510.10774))
- [ ] Persian G2P / ezafe handling
- [ ] Hugging Face Space demo

## 🙏 Acknowledgements & references

- **Audio8 TTS** — base model, codec and SFT code (Apache-2.0): [GitHub](https://github.com/Edge0-AI/Audio8_TTS) · [Hugging Face](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
- **Fish Audio S2 Pro / Fish Speech** — the DualAR architecture Audio8 builds on: [GitHub](https://github.com/fishaudio/fish-speech)
- **Mana-TTS** — 114 h single-speaker Persian corpus (CC0), from the *Nasl-e-Mana* magazine of Iran's blind community: [dataset](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) · [paper](https://aclanthology.org/2025.naacl-long.464/)
- **GPTInformal-Persian** — 6 h informal Persian speech (CC0): [dataset](https://huggingface.co/datasets/MahtaFetrat/GPTInformal-Persian)
- **Hugging Face Transformers**, **bitsandbytes**, **num2fawords**

Related Persian TTS work: [ParsVoice](https://arxiv.org/abs/2510.10774), [Persian Tacotron2 on ManaTTS](https://huggingface.co/MahtaFetrat/Persian-Tacotron2-on-ManaTTS), [ParsVoice-XTTS](https://huggingface.co/MohammadJRanjbar/ParsVoice-XTTS), [Chatterbox-TTS-Persian](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi), [Persian-tts-coqui](https://github.com/karim23657/Persian-tts-coqui).

## 📝 Citation

If you use this work, please cite it together with Audio8 TTS and the datasets:

```bibtex
@misc{audio8_persian_tts_2026,
  title        = {Audio8-Persian-TTS: Adapting a Compact Multilingual TTS Model to Persian},
  author       = {Rahimi Darehbagh, Armin},
  year         = {2026},
  howpublished = {\url{https://github.com/ard9/Audio8-Persian-TTS}}
}

@misc{audio8_tts,
  title        = {Audio8 TTS Preview: SOTA-Class TTS at Compact Scale},
  author       = {{Audio8 / Edge0-AI}},
  year         = {2026},
  howpublished = {\url{https://github.com/Edge0-AI/Audio8_TTS}}
}

@inproceedings{qharabagh-etal-2025-manatts,
  title     = {{M}ana{TTS} {P}ersian: A Recipe for Creating {TTS} Datasets for Lower-Resource Languages},
  author    = {Qharabagh, Mahta Fetrat and Dehghanian, Zahra and Rabiee, Hamid R.},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year      = {2025},
  address   = {Albuquerque, New Mexico},
  publisher = {Association for Computational Linguistics},
  pages     = {9177--9206},
  url       = {https://aclanthology.org/2025.naacl-long.464/}
}
```

## 📄 License

Code in this repository: [Apache License 2.0](LICENSE). The fine-tuned weights are a derivative of Audio8 TTS (Apache-2.0); see [NOTICE](NOTICE). Training data: Mana-TTS and GPTInformal-Persian (CC0-1.0).
