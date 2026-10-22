---
license: apache-2.0
language:
  - fa
base_model: Audio8/Audio8-TTS-Preview-0.6b
datasets:
  - MahtaFetrat/Mana-TTS
  - MahtaFetrat/GPTInformal-Persian
pipeline_tag: text-to-speech
library_name: transformers
tags:
  - text-to-speech
  - tts
  - persian
  - farsi
  - voice-cloning
  - audio8
  - custom_code
---

# Audio8-TTS-Persian (v0.2)

A **Persian (Farsi)** fine-tune of [Audio8-TTS-Preview-0.6B](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b), a 0.6B-parameter DualAR text-to-speech model with zero-shot voice cloning. Persian is not among the base model's supported languages; to our knowledge this is the first public Persian adaptation of Audio8.

Training code, Colab notebook and documentation: **https://github.com/ard9/Audio8-Persian-TTS**

## 🔊 Samples

| Text | Audio |
|---|---|
| سلام، امروز هوا خیلی خوب است و می‌خواهم کمی در پارک قدم بزنم. | [ft_s01.wav](samples/ft_s01.wav) |
| دیروز با دوستام رفتیم سینما، فیلمش واقعاً عالی بود! | [ft_s03.wav](samples/ft_s03.wav) |
| Voice clone (reference from a second speaker) | [clone_s01.wav](samples/clone_s01.wav) |

## Usage

The model uses Audio8's custom Transformers code (`trust_remote_code=True`). Normalize Persian text the same way as in training (digits → words, Arabic → Persian letters) with `normalize_fa` from the GitHub repo.

```python
# pip install "transformers>=4.57,<5" soundfile num2fawords
# git clone https://github.com/ard9/Audio8-Persian-TTS  (for fa_tools.normalize_fa)
import sys, torch, soundfile as sf
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, "Audio8-Persian-TTS/src")
from fa_tools import normalize_fa

model_id = "ArminRahimi/Audio8-TTS-Persian"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).eval().to(device)

text = normalize_fa("سلام! حال شما چطور است؟")
inputs = processor(text=[text], return_tensors="pt")
# Voice cloning: add reference_audio=["ref.wav"], reference_text=[normalize_fa("exact transcript")]
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                         do_sample=True, return_dict_in_generate=True)
    wav, lengths = model.decode_audio(out.codes)
sf.write("output.wav", wav[0, : int(lengths[0])].float().cpu().numpy(), model.config.codec_sample_rate)
```

Or from the command line: `python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Persian --text "سلام" --output out.wav`.

Keep each input under ~150 characters; `scripts/synthesize.py` splits longer text into sentences automatically.

## Training details

| | |
|---|---|
| Base model | Audio8-TTS-Preview-0.6B (repo commit `07e40f5`) |
| Data | All 77 Mana-TTS parts (≈ 114 h before filtering; HIGH alignment quality only, 1–20 s clips) + GPTInformal-Persian (≈ 6 h) |
| Reference conditioning | ~30% of Mana-TTS rows and 100% of GPTInformal rows trained with a same-speaker reference clip |
| Trainable parameters | Both DualAR branches (slow + fast AR), all 601M parameters |
| Optimizer | 8-bit AdamW, lr 3e-5, cosine schedule, 3% warmup, grad-norm clip 1.0 |
| Precision | fp32 master weights + fp16 autocast (T4) |
| Effective batch size | 16 (batch 2 × gradient accumulation 8) |
| Steps / epochs | 7000 / 2 |
| Hardware | 1× T4 on Google Colab, several sessions resumed from Hub checkpoints |
| Exported weights | bf16 |

## Evaluation

| Metric | Value |
|---|---|
| Eval loss (held-out, slow + fast AR) | 8.00 at step 7000 (v0.1, trained on 10 parts: 8.80; different eval split) |
| Listening impression | Intelligible Persian speech; prosody still somewhat robotic |
| CER, Whisper transcription of generated speech | not measured yet |
| MOS, native listeners | not measured yet |
| Speaker similarity (voice cloning) | not measured yet |

## Limitations

- **Mostly single-speaker data** (≈ 114 h Mana-TTS from one speaker + ≈ 6 h GPTInformal). Voice variety and prosody are limited; cloning of unseen voices is weaker than in Audio8's supported languages.
- **Limited compute** (a single Colab T4); no hyperparameter search.
- **No G2P**: unwritten short vowels, homographs and the ezafe can be mispronounced.
- Dates, abbreviations and Latin-script words are not specially normalized.
- The base model is a Preview checkpoint without Persian pre-training.
- Prosody still sounds somewhat robotic.

## Responsible use

This model can imitate voices. Clone a voice only with the speaker's explicit consent and disclose synthetic audio. Do not use it for impersonation, fraud or misinformation. The training data's authors prohibit using it to imitate their speakers' voices for malicious purposes.

## License and attribution

Apache-2.0, as a derivative of [Audio8 TTS](https://github.com/Edge0-AI/Audio8_TTS) (Apache-2.0; architecture inspired by Fish Audio S2 Pro's DualAR). Training data: [Mana-TTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) and [GPTInformal-Persian](https://huggingface.co/datasets/MahtaFetrat/GPTInformal-Persian) (CC0-1.0).

## Citation

```bibtex
@misc{audio8_persian_tts_2026,
  title        = {Audio8-Persian-TTS: Adapting a Compact Multilingual TTS Model to Persian},
  author       = {Rahimi Darehbagh, Armin},
  year         = {2026},
  howpublished = {\url{https://github.com/ard9/Audio8-Persian-TTS}}
}

@inproceedings{qharabagh-etal-2025-manatts,
  title     = {{M}ana{TTS} {P}ersian: A Recipe for Creating {TTS} Datasets for Lower-Resource Languages},
  author    = {Qharabagh, Mahta Fetrat and Dehghanian, Zahra and Rabiee, Hamid R.},
  booktitle = {Proceedings of NAACL 2025 (Volume 1: Long Papers)},
  year      = {2025},
  pages     = {9177--9206},
  url       = {https://aclanthology.org/2025.naacl-long.464/}
}
```
