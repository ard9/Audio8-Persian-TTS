# Samples

`base_*` are from the base model, `ft_*` and `clone_s01.wav` from the v0.2 run (step 7000), and `progress_step*` from the earlier v0.1 run (10 Mana-TTS parts). To refresh them after a new run:

Put a few short demo files here (they are linked from the main README):

| File | Where to get it |
|---|---|
| `base_s01.wav`, `base_s02.wav` | training repo: `runs/<run>/samples/base/s01.wav`, `s02.wav` |
| `ft_s01.wav` … `ft_s03.wav` | training repo: `runs/<run>/samples/step_<last>/s01.wav` … `s03.wav` |
| `clone_s01.wav` | training repo: `runs/<run>/samples/step_<last>/clone.wav` |

Keep each file small (a few seconds, < 1 MB). Converting to MP3 is fine:
`ffmpeg -i ft_s01.wav -b:a 128k ft_s01.mp3` (then update the links in README.md).

Do not upload voice clones of real people without their consent.
