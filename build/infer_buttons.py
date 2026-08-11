#!/usr/bin/env python3
"""Bayesian inference of button presses from reference audio.

For each detected onset in the reference audio:
1. Detect the pitch (pYIN)
2. Map to the nearest row (pitch → row index)
3. Map the timing to a step index (onset_time → step within sequence)
4. Infer which column sequence could produce that step trigger

Given the known COL_SEQS, STEP_VELS, and DEFAULT_FREQS, reconstruct
which (row, col) buttons are active at each moment.

Output: Inferred pattern sequence that can replace PATTERNS in render_original.py

Usage:
    python build/infer_buttons.py [--ref analysis/reference.wav]
"""

import argparse, json, os, sys
import numpy as np
import librosa

SR = 44100

# Known parameters (from render_original.py)
DEFAULT_FREQS = [587, 494, 415, 349, 294, 247, 196, 147]

COL_SEQS = [
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
]

STEP_VELS = [127, 36, 64, 36, 127, 36, 64, 36,
             127, 36, 72, 36, 127, 36, 90, 36]
STEP_VELS = [v / 127 for v in STEP_VELS]

SEQ_LEN = 16
STEPS_PER_PATTERN = 64
STEP_MS = 74


def detect_note_events(y, sr):
    """Detect onset times and pitches for each note event."""
    print("  Detecting onsets...")
    onsets_samples = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
    onsets_time = onsets_samples / sr

    print(f"  {len(onsets_time)} onsets found")

    events = []
    for i, (onset_s, onset_t) in enumerate(zip(onsets_samples, onsets_time)):
        # Extract segment for pitch detection
        end = min(onset_s + int(sr * 0.15), len(y))
        segment = y[onset_s:end]
        if len(segment) < sr * 0.02:
            continue

        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                segment, fmin=80, fmax=1500, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                median_f0 = float(np.median(f0_valid))
                confidence = float(np.mean(voiced_prob[~np.isnan(f0)]))
                events.append({
                    'time_s': float(onset_t),
                    'freq_hz': median_f0,
                    'confidence': confidence,
                    'amplitude': float(np.max(np.abs(segment[:int(sr * 0.05)]))),
                })
        except Exception:
            continue

    print(f"  {len(events)} note events with pitch")
    return events


def map_freq_to_row(freq, freqs=DEFAULT_FREQS, tolerance_cents=100):
    """Map a detected frequency to the nearest row index.

    Uses log-frequency distance (cents) to handle octave differences.
    Allows matching within tolerance_cents of any octave of each freq.
    """
    best_row = -1
    best_dist = float('inf')

    for row, ref_freq in enumerate(freqs):
        # Check multiple octaves
        for octave_shift in [-2, -1, 0, 1, 2]:
            shifted = ref_freq * (2 ** octave_shift)
            if shifted < 40 or shifted > 2000:
                continue
            dist_cents = abs(1200 * np.log2(freq / shifted))
            if dist_cents < best_dist:
                best_dist = dist_cents
                best_row = row

    if best_dist > tolerance_cents:
        return -1, best_dist
    return best_row, best_dist


def infer_step_timing(events, step_ms=STEP_MS):
    """Map onset times to step indices."""
    step_s = step_ms / 1000
    for event in events:
        # Global step index
        global_step = int(round(event['time_s'] / step_s))
        event['global_step'] = global_step
        event['seq_idx'] = global_step % SEQ_LEN
        event['pattern_idx'] = global_step // STEPS_PER_PATTERN
        event['local_step'] = global_step % STEPS_PER_PATTERN


def compute_col_likelihood(row, seq_idx, pattern_active_cells):
    """For a given row and seq_idx, compute likelihood for each column.

    A column is likely if:
    1. COL_SEQS[col][seq_idx] == 1 (the column sequence triggers on this step)
    2. The (row, col) pair would be a plausible button press
    """
    likelihoods = []
    for col in range(16):
        if COL_SEQS[col][seq_idx] == 1:
            # This column triggers on this step
            # Higher likelihood for columns already active
            prior = 0.1  # base prior
            if (row, col) in pattern_active_cells:
                prior = 0.9  # already known to be active
            likelihoods.append((col, prior))
        else:
            likelihoods.append((col, 0.0))  # can't trigger
    return likelihoods


def infer_patterns(events, step_ms=STEP_MS):
    """Infer the pattern sequence from note events.

    Strategy:
    1. Group events by pattern (each pattern = STEPS_PER_PATTERN steps)
    2. For each pattern, determine which (row, col) cells are active
    3. Use the constraint that patterns build up (cells are added, rarely removed)
    """
    # Map events to rows
    for event in events:
        row, dist = map_freq_to_row(event['freq_hz'])
        event['row'] = row
        event['row_dist_cents'] = dist

    # Group by pattern
    n_patterns = max(e['pattern_idx'] for e in events if e.get('pattern_idx') is not None) + 1
    patterns_events = [[] for _ in range(n_patterns)]
    for event in events:
        if event['row'] >= 0 and 0 <= event['pattern_idx'] < n_patterns:
            patterns_events[event['pattern_idx']].append(event)

    print(f"\n  Events per pattern:")
    for i, pe in enumerate(patterns_events):
        rows_seen = set(e['row'] for e in pe)
        print(f"    Pattern {i:2d}: {len(pe):3d} events, rows: {sorted(rows_seen)}")

    # Infer active cells per pattern
    inferred_patterns = []
    cumulative_cells = set()  # cells accumulate over patterns

    for pi, pe in enumerate(patterns_events):
        # Count how many times each row fires at each seq_idx
        row_seq_counts = {}
        for event in pe:
            key = (event['row'], event['seq_idx'])
            row_seq_counts[key] = row_seq_counts.get(key, 0) + 1

        # For each row that appears, find the best column assignment
        rows_in_pattern = set(e['row'] for e in pe)
        pattern_cells = {}

        for row in rows_in_pattern:
            # Get all seq indices where this row fires
            row_triggers = []
            for (r, s), count in row_seq_counts.items():
                if r == row:
                    row_triggers.append(s)

            # Find the column whose sequence best matches these triggers
            best_col = -1
            best_score = -1

            for col in range(16):
                # How well does COL_SEQS[col] match the observed triggers?
                col_seq = COL_SEQS[col]

                # Count matches (expected triggers that actually occurred)
                # and misses (expected triggers that didn't occur)
                expected_triggers = [s for s in range(SEQ_LEN) if col_seq[s] == 1]
                expected_trigger_set = set(expected_triggers)

                # How many of our 64 steps' worth of triggers match?
                # Within each pattern we cycle through SEQ_LEN 4 times
                observed_set = set(row_triggers)

                hits = len(observed_set & expected_trigger_set)
                # Penalty for expected triggers that weren't observed
                misses = len(expected_trigger_set - observed_set)
                # Penalty for observed triggers that weren't expected
                false_alarms = len(observed_set - expected_trigger_set)

                score = hits - 0.5 * misses - 1.0 * false_alarms

                # Bonus if this (row, col) was already active in previous patterns
                if (row, col) in cumulative_cells:
                    score += 2.0

                if score > best_score:
                    best_score = score
                    best_col = col

            if best_col >= 0 and best_score > 0:
                pattern_cells[(row, best_col)] = 1

        # Add cells from previous patterns (buttons stay pressed)
        for cell in cumulative_cells:
            if cell not in pattern_cells:
                # Check if this cell's row still appears
                # (if a row fires, its column assignment stays)
                pattern_cells[cell] = 1

        cumulative_cells.update(pattern_cells.keys())
        inferred_patterns.append(dict(pattern_cells))

    return inferred_patterns


def format_patterns_python(patterns):
    """Format inferred patterns as Python code for render_original.py."""
    lines = ["PATTERNS = ["]
    for i, pat in enumerate(patterns):
        items = sorted(pat.keys())
        if not items:
            lines.append(f"    {{}},  # pattern {i}")
        else:
            cell_strs = [f"({r}, {c}): {pat[(r,c)]}" for r, c in items]
            lines.append(f"    {{{', '.join(cell_strs)}}},  # pattern {i}")
    lines.append("]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None)
    parser.add_argument('--step-ms', type=float, default=STEP_MS)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = args.ref or os.path.join(root, 'analysis', 'reference.wav')

    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    print(f"Loading: {ref_path}")
    y, sr = librosa.load(ref_path, sr=SR, mono=True)
    print(f"  Duration: {len(y)/sr:.1f}s")

    print("\n[1] Detecting note events...")
    events = detect_note_events(y, sr)

    print("\n[2] Mapping to step grid...")
    infer_step_timing(events, args.step_ms)

    print("\n[3] Inferring patterns...")
    patterns = infer_patterns(events, args.step_ms)

    # Output
    process_dir = os.path.join(root, 'analysis', 'process')
    os.makedirs(process_dir, exist_ok=True)

    # Save raw events
    events_path = os.path.join(process_dir, 'note_events.json')
    with open(events_path, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"\nNote events: {events_path}")

    # Save inferred patterns
    patterns_path = os.path.join(process_dir, 'inferred_patterns.json')
    # Convert tuple keys to strings for JSON
    json_patterns = []
    for pat in patterns:
        json_pat = {}
        for (r, c), v in pat.items():
            json_pat[f"{r}-{c}"] = v
        json_patterns.append(json_pat)
    with open(patterns_path, 'w') as f:
        json.dump(json_patterns, f, indent=2)
    print(f"Inferred patterns: {patterns_path}")

    # Print Python code for patterns
    print("\n" + "=" * 60)
    print("INFERRED PATTERNS (Python code):")
    print("=" * 60)
    print(format_patterns_python(patterns))

    # Summary stats
    print(f"\n{len(patterns)} patterns inferred")
    print(f"Unique cells across all patterns: {len(set().union(*(p.keys() for p in patterns)))}")
    for i, pat in enumerate(patterns):
        cells = sorted(pat.keys())
        print(f"  P{i:2d}: {len(cells)} cells - {cells}")


if __name__ == '__main__':
    main()
