# Original Mode Sound Matching — Iteration Guide

## Goal

Match the sound of Polynome's "original" mode to the reference video
(stretta's monome Press Café, starting at 2:55):
https://www.youtube.com/watch?v=Su0i0kkfe5E&t=175

## Setup

1. Place the reference audio as `analysis/reference.wav` (mono or stereo, any sample rate).
   Extract from the video with:
   ```
   ffmpeg -ss 175 -t 180 -i <video_file> -vn -ar 44100 -ac 1 analysis/reference.wav
   ```

2. Install dependencies:
   ```
   pip install numpy scipy librosa soundfile
   ```

## File Layout

```
build/
  render_original.py   — Offline sequencer: renders all 26 patterns to WAV
  compare_audio.py     — Computes distance metrics between reference and rendered
  iterate.sh           — Run one render+compare cycle
  gen_original.py      — Tone synthesis (edit this to change timbre)
  common.py            — DSP library
js/modes/original.js   — Patterns, COL_SEQS, STEP_VELS, tempo, effects config
analysis/
  reference.wav        — Ground truth audio from the video (you provide this)
  rendered.wav         — Latest render from the sequencer
  distance.json        — Latest comparison report
  history.csv          — Append-only log: iteration, composite_distance, description
```

## Iteration Loop

Each iteration follows this cycle:

### 1. Render
```bash
python build/render_original.py
```
This produces `analysis/rendered.wav` by simulating the 26-pattern sequence
using the current tone synthesis (`gen_original.py`) and sequencer config
(`COL_SEQS`, `STEP_VELS`, `defaultStepMs` from `original.js`).

### 2. Compare
```bash
python build/compare_audio.py --iteration N --description "what you changed"
```
This outputs `analysis/distance.json` with sub-metrics and a composite score,
and appends a row to `analysis/history.csv`.

### 3. Or run both at once
```bash
bash build/iterate.sh N "description of change"
```

### 4. Read the report
The distance report contains:

| Metric | What it means | What to tweak |
|--------|--------------|---------------|
| `spectral_centroid` | Brightness | Harmonics in `gen_original.py` |
| `spectral_rolloff` | High-freq energy | Saturation, comb filter |
| `mfcc` | Timbral envelope shape | Envelope params, harmonic structure |
| `onset_density` | Notes per second | `defaultStepMs` in `original.js` |
| `ioi_histogram` | Rhythmic pattern | `COL_SEQS` in `original.js` |
| `rms_correlation` | Volume shape over time | Pattern order, `STEP_VELS` |
| `pitch_class` | Scale/tuning match | `mainFreqs` in `original.js` |
| `composite` | Weighted sum of all above | — |

The `diagnosis` array suggests specific actions.

### 5. Make a targeted change

Based on the diagnosis, edit **one thing at a time**:

- **Timbre too bright** → In `gen_original.py`: reduce number of harmonics,
  lower `harmonic_amplitudes_db` for upper partials, remove comb filter,
  reduce saturation `drive`.
- **Envelope wrong** → In `gen_original.py`: change `decay_time` (shorter =
  more percussive), `attack_ms`, sample `dur`.
- **Tempo off** → In `render_original.py`: change `--step-ms` flag. Once
  correct, update `defaultStepMs` in `original.js`.
- **Rhythm wrong** → In `render_original.py` and `original.js`: edit `COL_SEQS`.
- **Dynamics off** → In `render_original.py` and `original.js`: edit `STEP_VELS`.
- **Pitches wrong** → In `gen_original.py` `DEFAULT_FREQS` and `original.js`
  `mainFreqs`.

### 6. Re-render, re-compare, check if composite decreased

If composite decreased → commit with:
```
git add -A && git commit -m "iteration N: <description> (composite: X.XXX → Y.YYY)"
```

If composite increased → revert the change and try something else.

## Metric Weights

```
spectral_centroid:  0.15
mfcc:               0.25
onset_density:      0.10
ioi_histogram:      0.10
rms_correlation:    0.15
pitch_class:        0.10
spectral_rolloff:   0.15
```

Timbre (centroid + mfcc + rolloff = 0.55) is weighted highest because it's
likely the biggest difference — the current synthesis is a complex marimba-like
tone, while the Press Café original may use a simpler waveform.

## Key Files to Edit

**For timbre changes** — `build/gen_original.py`:
- `DEFAULT_ENVELOPE` dict: `harmonic_ratios`, `harmonic_amplitudes_db`,
  `inharmonicity_cents`, `attack_ms`, `decay_time`
- `gen_tone()` function: comb filter params, saturation params, sample duration

**For sequence/rhythm/tempo changes** — `js/modes/original.js` AND
`build/render_original.py` (keep them in sync!):
- `COL_SEQS`: 16×16 binary array of column rhythms
- `STEP_VELS`: 16-step velocity accent pattern
- `defaultStepMs`: tempo

**For pitch changes** — `build/gen_original.py` `DEFAULT_FREQS` AND
`js/modes/original.js` `mainFreqs` (keep in sync!)

## Notes

- The renderer mirrors `engine.js` lines 408–444: for each step, check
  `COL_SEQS[col][step % 16]`, apply `STEP_VELS[step % 16]`, mix sample.
- Effects in the renderer (delay + reverb) approximate the browser's Web Audio
  chain. They don't need to be perfect — focus on the dry signal first.
- The comparison trims both files to the shorter duration before computing
  metrics, so different lengths are fine.
- Change one parameter at a time so you can attribute metric changes to
  specific edits.
