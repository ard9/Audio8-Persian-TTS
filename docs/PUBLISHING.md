# Publishing checklist

## 1. Before the first push

- [ ] `grep -rn "TODO" .` — everything left should be results you will fill in after training.

```bash
git init -b main
git add .
git commit -m "Initial release: Persian fine-tuning pipeline for Audio8 TTS"
git remote add origin https://github.com/ard9/Audio8-Persian-TTS.git
git push -u origin main
```

Create the empty repository `Audio8-Persian-TTS` on GitHub first (no README, no license; they are already here).

## 2. After training

1. Fill the **Results** and **Limitations** sections in `README.md` / `README_fa.md` and the **Training details** and **Evaluation** sections in `docs/MODEL_CARD.md`.
2. Copy a few demo files into `samples/` (see `samples/README.md`).
3. In the notebook, run the **Export** cell and then the **Upload the exported model to Hugging Face** cell.
   (Alternatively, from any machine: `python scripts/upload_to_hf.py --export-dir <export>/checkpoint-N --repo-id ArminRahimi/Audio8-TTS-Persian --samples-dir samples/`.)

4. Commit and push the updated READMEs and samples.

## 3. Editing the notebook

The notebook is generated. Edit `src/fa_tools.py` or `scripts/build_notebook.py`, then:

```bash
python scripts/build_notebook.py
AUDIO8_REPO=/path/to/Audio8_TTS pytest -q
```

CI runs the same tests on every push and fails if the notebook is out of sync.

## 4. Never commit

Model weights, `codec.pth`, checkpoints, prepared data, Hugging Face tokens or local paths with personal information. `.gitignore` already excludes the common ones.
