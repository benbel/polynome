#!/usr/bin/env python3
"""Infer button press patterns from video + audio analysis.

Combines video LED detection (which rows are lit) with audio onset/pitch
detection (when notes play) to reconstruct the pattern sequence used in
the Press Café performance.

Constraints from user analysis:
- Max 5 rows simultaneously active
- Uses COL_SEQS model (columns = different rhythmic sequences)
- Patterns are cumulative (cells stay on once activated)

Usage:
    python build/infer_patterns.py [--step-ms 61.5] [--output analysis/process/inferred_patterns.json]
"""

import json, os, sys
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_full import COL_SEQS, STEP_VELS, SEQ_LEN, STEPS_PER_PATTERN, PARAM_DEFAULTS, PARAM_SPEC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Video analysis ────────────────────────────────────────────────────────

def load_video_timeline():
    """Load the video grid timeline and extract row activity over time."""
    path = os.path.join(ROOT, 'analysis', 'process', 'video_grid_timeline.json')
    if not os.path.exists(path):
        print(f"  [warn] {path} not found, skipping video analysis")
        return None

    with open(path) as f:
        data = json.load(f)

    # Build row activity timeline: for each time point, which rows have LEDs
    timeline = []
    for entry in data['timeline']:
        rows = set(r for r, c in entry['lit_cells'])
        timeline.append({
            'time': entry['ref_time'],
            'rows': rows,
            'n_lit': entry['n_lit'],
            'cells': [(r, c) for r, c in entry['lit_cells']],
        })

    return timeline


def filter_video_rows(timeline, max_rows=5):
    """Filter video timeline to enforce max_rows constraint.

    Strategy: use persistence — rows that appear in more consecutive frames
    are more likely to be real. Track row confidence and keep top max_rows.
    """
    if timeline is None:
        return None

    # Build row confidence: how many of the last N frames had this row lit
    window = 6  # 3 seconds at 0.5s intervals
    filtered = []

    for i, entry in enumerate(timeline):
        # Compute confidence for each row based on recent history
        row_confidence = {}
        for row in range(8):
            count = 0
            for j in range(max(0, i - window), i + 1):
                if row in timeline[j]['rows']:
                    count += 1
            if count > 0:
                row_confidence[row] = count

        # Keep top max_rows by confidence
        if len(row_confidence) > max_rows:
            sorted_rows = sorted(row_confidence.keys(),
                                key=lambda r: row_confidence[r], reverse=True)
            active_rows = set(sorted_rows[:max_rows])
        else:
            active_rows = set(row_confidence.keys())

        # Intersect with actually detected rows
        active_rows = active_rows & entry['rows']

        filtered.append({
            'time': entry['time'],
            'rows': active_rows,
            'confidence': {r: row_confidence.get(r, 0) for r in active_rows},
        })

    return filtered


def get_video_row_activation_timeline(filtered_video):
    """From filtered video, determine when each row first becomes active
    and when it's deactivated."""
    if filtered_video is None:
        return {}

    # For each row, find sustained activation periods
    row_periods = {}  # row -> list of (start_time, end_time)

    for row in range(8):
        active = False
        start = None
        periods = []

        for entry in filtered_video:
            if row in entry['rows']:
                if not active:
                    start = entry['time']
                    active = True
            else:
                if active:
                    periods.append((start, entry['time']))
                    active = False

        if active:
            periods.append((start, filtered_video[-1]['time']))

        # Merge short gaps (< 3s) as the performer's hand may briefly occlude
        merged = []
        for period in periods:
            if merged and period[0] - merged[-1][1] < 3.0:
                merged[-1] = (merged[-1][0], period[1])
            else:
                merged.append(period)

        # Only keep periods longer than 2 seconds
        row_periods[row] = [(s, e) for s, e in merged if e - s > 2.0]

    return row_periods


# ── Audio analysis ────────────────────────────────────────────────────────

def detect_audio_events(ref_path, freqs, sr=22050):
    """Detect note onsets and pitches from reference audio.

    Returns list of {time, row, freq, confidence} events.
    """
    y, sr = librosa.load(ref_path, sr=sr, mono=True)
    print(f"  Audio: {len(y)/sr:.1f}s at {sr}Hz")

    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
    print(f"  Detected {len(onsets)} onsets")

    events = []
    for onset_s in onsets:
        end = min(onset_s + int(sr * 0.2), len(y))
        segment = y[onset_s:end]
        if len(segment) < sr * 0.02:
            continue

        try:
            f0, _, vp = librosa.pyin(segment, fmin=80, fmax=2000, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) == 0:
                continue
            freq = float(np.median(f0_valid))
        except Exception:
            continue

        # Map to closest row frequency (allow octave matching)
        best_row, best_dist = -1, float('inf')
        for row, ref_freq in enumerate(freqs):
            for oct in [-1, 0, 1]:
                shifted = ref_freq * (2 ** oct)
                if shifted < 40:
                    continue
                dist = abs(1200 * np.log2(max(freq, 1) / shifted))
                if dist < best_dist:
                    best_dist = dist
                    best_row = row

        if best_dist > 200:  # too far from any known frequency
            continue

        events.append({
            'time': float(onset_s / sr),
            'row': best_row,
            'freq': freq,
            'dist_cents': best_dist,
        })

    print(f"  Matched {len(events)} events to rows")
    return events


def compute_audio_row_density(events, duration, window=4.0):
    """Compute how active each row is over time windows."""
    n_windows = int(duration / window)
    row_density = np.zeros((8, n_windows))

    for event in events:
        w = int(event['time'] / window)
        if 0 <= w < n_windows:
            row_density[event['row'], w] += 1

    return row_density


# ── Pattern inference ─────────────────────────────────────────────────────

def infer_col_for_row(events_in_window, step_ms):
    """Given audio events for a specific row within a time window,
    determine which column (COL_SEQ) best explains the trigger pattern."""
    if not events_in_window:
        return None, 0

    # Convert event times to step indices
    step_indices = set()
    for event in events_in_window:
        global_step = int(round(event['time'] / (step_ms / 1000)))
        seq_idx = global_step % SEQ_LEN
        step_indices.add(seq_idx)

    # Score each column
    best_col, best_score = -1, -float('inf')
    for col in range(16):
        expected = set(s for s in range(SEQ_LEN) if COL_SEQS[col][s] == 1)
        hits = len(step_indices & expected)
        misses = len(expected - step_indices)
        false_alarms = len(step_indices - expected)
        density = sum(COL_SEQS[col])  # how many triggers per cycle

        # Score: reward hits, penalize false alarms more than misses
        # (we might miss events due to polyphony masking)
        score = hits - 0.3 * misses - 1.0 * false_alarms

        if score > best_score:
            best_score = score
            best_col = col

    return best_col, best_score


def infer_patterns(video_timeline, audio_events, step_ms, duration, max_rows=5):
    """Combine video and audio to infer the pattern sequence.

    Returns list of pattern dicts (same format as render_original.py PATTERNS).
    """
    n_patterns = int(duration / (STEPS_PER_PATTERN * step_ms / 1000))
    pattern_duration = STEPS_PER_PATTERN * step_ms / 1000  # seconds per pattern

    print(f"\n  Pattern duration: {pattern_duration:.1f}s, total patterns: {n_patterns}")

    # Get video row activation periods
    video_row_periods = {}
    if video_timeline is not None:
        video_row_periods = get_video_row_activation_timeline(video_timeline)
        print("  Video row activation periods:")
        for row in range(8):
            if video_row_periods.get(row):
                periods_str = ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in video_row_periods[row])
                print(f"    Row {row}: {periods_str}")

    # Get audio row density
    audio_row_density = compute_audio_row_density(audio_events, duration, window=pattern_duration)
    print(f"  Audio row density shape: {audio_row_density.shape}")

    # For each pattern, determine which rows are active
    patterns = []
    cumulative_cells = {}  # (row, col) -> velocity, accumulates across patterns
    prev_active_rows = set()

    for pi in range(n_patterns):
        t_start = pi * pattern_duration
        t_end = (pi + 1) * pattern_duration
        t_mid = (t_start + t_end) / 2

        # Determine active rows from video
        video_rows = set()
        if video_timeline is not None:
            for row, periods in video_row_periods.items():
                for start, end in periods:
                    if start <= t_mid <= end:
                        video_rows.add(row)

        # Determine active rows from audio
        audio_rows = set()
        pattern_events = [e for e in audio_events if t_start <= e['time'] < t_end]
        for event in pattern_events:
            audio_rows.add(event['row'])

        # Combine: union of video and audio evidence
        candidate_rows = video_rows | audio_rows

        # Enforce max_rows constraint using confidence scoring
        if len(candidate_rows) > max_rows:
            # Score each row by combined evidence
            row_scores = {}
            for row in candidate_rows:
                score = 0
                # Video confidence: was this row seen in video?
                if row in video_rows:
                    score += 2.0
                # Audio confidence: how many events for this row?
                n_events = sum(1 for e in pattern_events if e['row'] == row)
                score += n_events * 0.5
                # Persistence: was this row active in previous pattern?
                if row in prev_active_rows:
                    score += 1.0
                row_scores[row] = score

            sorted_rows = sorted(row_scores.keys(),
                                key=lambda r: row_scores[r], reverse=True)
            candidate_rows = set(sorted_rows[:max_rows])

        # For each active row, determine its column assignment
        pattern_cells = {}
        for row in candidate_rows:
            # Get audio events for this row in this time window
            row_events = [e for e in pattern_events if e['row'] == row]

            if row_events:
                col, score = infer_col_for_row(row_events, step_ms)
                if col is not None and score > -2:
                    pattern_cells[(row, col)] = 1

            # If row was in cumulative_cells (persistent), keep it
            for (r, c), v in cumulative_cells.items():
                if r == row and (r, c) not in pattern_cells:
                    # Check if this row-col combo should persist
                    if row in video_rows or row in prev_active_rows:
                        pattern_cells[(r, c)] = v

        # Update cumulative cells (Press Café is additive — cells stay on)
        # But respect max_rows: only keep cells for active rows
        active_rows_in_pattern = set(r for r, c in pattern_cells.keys())
        new_cumulative = {}
        for (r, c), v in cumulative_cells.items():
            if r in candidate_rows:
                new_cumulative[(r, c)] = v
        new_cumulative.update(pattern_cells)
        cumulative_cells = new_cumulative

        patterns.append(dict(pattern_cells))
        prev_active_rows = candidate_rows

        if pi % 5 == 0 or pi < 3:
            rows_str = sorted(set(r for r, c in pattern_cells.keys()))
            print(f"    Pattern {pi:2d} (t={t_start:.0f}-{t_end:.0f}s): "
                  f"rows={rows_str}, cells={len(pattern_cells)}")

    return patterns


def evaluate_patterns(patterns, ref_path, step_ms):
    """Evaluate inferred patterns by rendering and comparing."""
    from optimize_full import (render_with_params, compute_composite,
                                PARAM_DEFAULTS, COMPARE_SR)

    y_ref, _ = librosa.load(ref_path, sr=COMPARE_SR, mono=True)
    y_ren = render_with_params(PARAM_DEFAULTS.copy(), patterns, fast=True)
    composite, metrics = compute_composite(y_ren, y_ref)

    return composite, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Infer patterns from video + audio')
    parser.add_argument('--step-ms', type=float, default=61.5)
    parser.add_argument('--max-rows', type=int, default=5)
    parser.add_argument('--output', default=None)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    ref_path = os.path.join(ROOT, 'analysis', 'reference.wav')
    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    # Get current frequencies from params
    freqs = [PARAM_SPEC[i][1] for i, (name, _, _, _) in enumerate(PARAM_SPEC)
             if name.startswith('freq_')]

    # Step 1: Load video timeline
    print("[1/4] Loading video timeline...")
    video_timeline = load_video_timeline()
    if video_timeline:
        filtered_video = filter_video_rows(video_timeline, max_rows=args.max_rows)
        print(f"  Filtered to max {args.max_rows} rows per frame")
    else:
        filtered_video = None

    # Step 2: Audio onset/pitch detection
    print("\n[2/4] Detecting audio events...")
    audio_events = detect_audio_events(ref_path, freqs)

    # Duration from reference
    y_ref, sr = librosa.load(ref_path, sr=22050, mono=True)
    duration = len(y_ref) / sr

    # Step 3: Infer patterns
    print("\n[3/4] Inferring patterns...")
    patterns = infer_patterns(filtered_video, audio_events, args.step_ms,
                              duration, max_rows=args.max_rows)

    # Stats
    print(f"\n  Total patterns: {len(patterns)}")
    max_r = max(len(set(r for r, c in p.keys())) for p in patterns if p) if patterns else 0
    print(f"  Max rows in any pattern: {max_r}")
    all_rows = set()
    for p in patterns:
        all_rows.update(r for r, c in p.keys())
    print(f"  All rows used: {sorted(all_rows)}")

    # Step 4: Evaluate
    if args.evaluate:
        print("\n[4/4] Evaluating...")

        # Compare with current AMXD patterns
        from render_original import PATTERNS as amxd_patterns
        print("  AMXD patterns:")
        c_amxd, m_amxd = evaluate_patterns(amxd_patterns, ref_path, args.step_ms)
        print(f"    Composite: {c_amxd:.4f}")
        print(f"    RMS corr:  {m_amxd['rms_correlation']:.4f}")

        print("  Inferred patterns:")
        c_inf, m_inf = evaluate_patterns(patterns, ref_path, args.step_ms)
        print(f"    Composite: {c_inf:.4f}")
        print(f"    RMS corr:  {m_inf['rms_correlation']:.4f}")

        winner = "INFERRED" if c_inf < c_amxd else "AMXD"
        print(f"\n  Winner: {winner}")
        for k in sorted(m_amxd.keys()):
            diff = m_inf[k] - m_amxd[k]
            mark = "+" if diff > 0 else "-" if diff < 0 else "="
            print(f"    {k:25s}: AMXD={m_amxd[k]:.4f}  INF={m_inf[k]:.4f}  {mark}{abs(diff):.4f}")

    # Save
    out_path = args.output or os.path.join(ROOT, 'analysis', 'process', 'inferred_patterns.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert tuple keys to string for JSON
    json_patterns = []
    for p in patterns:
        jp = {}
        for (r, c), v in p.items():
            jp[f"{r}-{c}"] = v
        json_patterns.append(jp)

    output = {
        'step_ms': args.step_ms,
        'max_rows': args.max_rows,
        'n_patterns': len(patterns),
        'patterns': json_patterns,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
