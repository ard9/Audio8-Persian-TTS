---
license: cc-by-nc-4.0
language:
  - ckb
base_model: ArminRahimi/Audio8-TTS-Persian
datasets:
  - aranemini/central-kurdish-tts4all
pipeline_tag: text-to-speech
library_name: transformers
tags:
  - text-to-speech
  - tts
  - kurdish
  - central-kurdish
  - sorani
  - voice-cloning
  - audio8
  - custom_code
---

# Audio8-TTS-Kurdish-Sorani

A **Central Kurdish (Sorani)** fine-tune of Audio8 TTS, a 0.6B-parameter DualAR text-to-speech model with zero-shot voice cloning. Neither Central Kurdish nor Persian is among the base model's supported languages. Training started from the Persian fine-tune [ArminRahimi/Audio8-TTS-Persian](https://huggingface.co/ArminRahimi/Audio8-TTS-Persian), whose script Sorani largely shares. <!-- If you trained from the original Audio8 (INIT_MODEL = "base"), change this sentence and base_model above to Edge0/Audio8-TTS-Preview-0.6b. -->

Training code, Colab notebook and documentation: **https://github.com/ard9/Audio8-Persian-TTS**

> **Non-commercial use only.** The training data, [central-kurdish-tts4all](https://huggingface.co/datasets/aranemini/central-kurdish-tts4all), is licensed CC BY-NC-ND 4.0, so this model is released under **CC BY-NC 4.0**: free for research, education and non-commercial demos; not for paid products or services.

## 🔊 Samples

| Text | Audio |
|---|---|
| سڵاو، ئەمڕۆ کەشوهەوا زۆر خۆشە و دەمەوێت کەمێک لە پارک پیاسە بکەم. | [ft_s01.wav](samples/ft_s01.wav) |
| دوێنێ لەگەڵ هاوڕێکانم چووین بۆ سینەما، فیلمەکە بەڕاستی زۆر جوان بوو! | [ft_s03.wav](samples/ft_s03.wav) |
| Voice clone (reference from another speaker of the dataset) | [clone_s01.wav](samples/clone_s01.wav) |

## Usage

The model uses Audio8's custom Transformers code (`trust_remote_code=True`). Normalize the text with the same Sorani rules as in training (`normalize_ckb` in the GitHub repo: Arabic letter variants unified, numbers spelled out in Kurdish).

```python
# pip install "transformers>=4.57,<5" soundfile
# git clone https://github.com/ard9/Audio8-Persian-TTS  (for fa_tools.normalize_ckb)
import sys, torch, soundfile as sf
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, "Audio8-Persian-TTS/src")
from fa_tools import normalize_ckb

model_id = "ArminRahimi/Audio8-TTS-Kurdish-Sorani"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).eval().to(device)

text = normalize_ckb("سڵاو! چۆنی؟")
inputs = processor(text=[text], return_tensors="pt")
# Voice cloning: add reference_audio=["ref.wav"], reference_text=[normalize_ckb("exact transcript")]
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                         do_sample=True, return_dict_in_generate=True)
    wav, lengths = model.decode_audio(out.codes)
sf.write("output.wav", wav[0, : int(lengths[0])].float().cpu().numpy(), model.config.codec_sample_rate)
```

Or from the command line: `python scripts/synthesize.py --model ArminRahimi/Audio8-TTS-Kurdish-Sorani --lang ckb --text "سڵاو" --output out.wav`.

## Training details

| | |
|---|---|
| Starting model | Audio8-TTS-Persian (itself fine-tuned from Audio8-TTS-Preview-0.6B, repo commit `07e40f5`) |
| Data | central-kurdish-tts4all: ≈ 35 h of read Central Kurdish, 3 speakers (male and female audiobook narrators, male studio voice), 24 kHz, clips of 1–20 s |
| Default voice | The speaker with the most audio is trained without a reference clip; the other speakers always with a same-speaker reference, which teaches voice cloning |
| Text normalization | Sorani rules: Arabic yeh/kaf → Kurdish forms, ھ → ه, heh + ZWNJ → ە, diacritics and ZWNJ removed, numbers spelled out in Kurdish |
| Trainable parameters | Both DualAR branches (slow + fast AR), all 601M parameters |
| Optimizer | 8-bit AdamW, lr 3e-5, cosine schedule, 3% warmup, grad-norm clip 1.0 |
| Steps / epochs | TODO |
| Hardware | TODO GPU on Google Colab |
| Exported weights | bf16 |

## Evaluation

| Metric | Value |
|---|---|
| Eval loss (held-out, slow + fast AR) | TODO |
| CER, ASR transcription of generated speech | not measured yet |
| MOS, native listeners | not measured yet |

## Limitations

- **Three speakers, read speech.** Voice variety and prosody are limited; conversational speech and unseen voices are weaker.
- **Possible Persian accent** when starting from the Persian model, especially in early checkpoints.
- **Spelling follows the dataset**: the model learns pronunciation from how the transcripts are written; non-standard spellings and Latin-script words may be mispronounced.
- The base model is a Preview checkpoint without Kurdish or Persian pre-training.

## Responsible use

This model can imitate voices. Clone a voice only with the speaker's explicit consent and disclose synthetic audio. Do not use it for impersonation, fraud or misinformation, or to imitate the dataset's speakers.

## License and attribution

**CC BY-NC 4.0** (non-commercial). Model code and architecture: [Audio8 TTS](https://github.com/Edge0-AI/Audio8_TTS) (Apache-2.0). Training data: [central-kurdish-tts4all](https://huggingface.co/datasets/aranemini/central-kurdish-tts4all) by Aran Emini et al. (CC BY-NC-ND 4.0), including audiobook content used with the permission of Malaka Mustafa Soltani, read by Shahen Shahlai and Fatih.

## Citation

```bibtex
@misc{audio8_persian_tts_2026,
  title        = {Audio8-Persian-TTS: Adapting a Compact Multilingual TTS Model to Persian},
  author       = {Rahimi Darehbagh, Armin},
  year         = {2026},
  howpublished = {\url{https://github.com/ard9/Audio8-Persian-TTS}}
}
```

Please also cite the central-kurdish-tts4all dataset as described on [its dataset page](https://huggingface.co/datasets/aranemini/central-kurdish-tts4all).
