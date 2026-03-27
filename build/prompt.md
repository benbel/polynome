# Sound Matching: Original Mode → Press Café Video

## Context

The Polynome project has an "original" mode that replicates stretta's monome
Press Café performance. The sound doesn't match the reference video yet.

Reference video: https://www.youtube.com/watch?v=Su0i0kkfe5E (starting at 2:55)

The reference audio has already been extracted to `analysis/reference.wav`.

## What exists

- `build/render_original.py` — Offline renderer that produces `analysis/rendered.wav`
  by simulating the 26-pattern sequencer using tone synthesis from `gen_original.py`
  and sequence config (`COL_SEQS`, `STEP_VELS`, step timing).
- `build/compare_audio.py` — Computes 7 distance sub-metrics + weighted composite
  between `analysis/reference.wav` and `analysis/rendered.wav`. Outputs
  `analysis/distance.json` and appends to `analysis/history.csv`.
- `build/iterate.sh N "description"` — Runs render then compare in one step.
- `build/gen_original.py` — Tone synthesis (harmonics, envelope, comb filter, saturation).
- `build/common.py` — DSP library.
- `js/modes/original.js` — Runtime config: `COL_SEQS`, `STEP_VELS`, `mainFreqs`,
  `defaultStepMs`, effects, patterns.

## Your task

Iteratively reduce the composite distance between the rendered output and the
reference audio. Work in a loop:

### For each iteration:

1. **Render**: `python build/render_original.py`
2. **Compare**: `python build/compare_audio.py --iteration N --description "what you changed"`
3. **Read the JSON report** (`analysis/distance.json`). It contains:
   - `spectral_centroid` (0–1): brightness mismatch → adjust harmonics in `gen_original.py`
   - `spectral_rolloff` (0–1): high-freq energy → adjust saturation, comb filter
   - `mfcc` (0–1): timbral shape → adjust envelope, harmonic structure
   - `onset_density` (0–1): notes/sec ratio → adjust `--step-ms` in renderer
   - `ioi_histogram` (0–1): rhythmic structure → adjust `COL_SEQS`
   - `rms_correlation` (0–1): volume shape → adjust pattern order, `STEP_VELS`
   - `pitch_class` (0–1): scale/tuning → adjust `DEFAULT_FREQS`/`mainFreqs`
   - `composite`: weighted sum (lower = better)
   - `diagnosis`: array of actionable suggestions
4. **Make ONE targeted change** based on the diagnosis. Edit the relevant file:
   - Timbre → `build/gen_original.py` (`DEFAULT_ENVELOPE`, `gen_tone()` function)
   - Tempo → `--step-ms` flag, then `defaultStepMs` in `original.js`
   - Rhythm → `COL_SEQS` in both `render_original.py` and `original.js` (keep in sync!)
   - Velocity → `STEP_VELS` in both files (keep in sync!)
   - Pitch → `DEFAULT_FREQS` in `gen_original.py` and `mainFreqs` in `original.js`
5. **Re-render and re-compare**. If composite decreased, keep the change. If it
   increased, revert and try something else.
6. **Commit** after each successful iteration:
   ```
   git commit -am "iteration N: <description> (composite: X.XXX → Y.YYY)"
   ```

### Important rules:

- Change ONE parameter at a time so you can attribute metric changes.
- Always keep `render_original.py` and `original.js` in sync for shared
  constants (`COL_SEQS`, `STEP_VELS`, `DEFAULT_FREQS`/`mainFreqs`, step timing).
- Start with iteration 0 as baseline (no changes, just render+compare).
- Focus on the highest sub-metric first — that's the biggest gap.
- The `diagnosis` array in the JSON output tells you what to do. Follow it.
- Do NOT listen to the audio. Only use the numeric metrics to guide changes.
- Stop when composite is below 0.15 or when 3 consecutive iterations fail to
  improve it.

### Start now:

Run iteration 0 as baseline, read the report, then begin improving.
