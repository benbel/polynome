#!/usr/bin/env python3
"""Analyze the Monome demo video to detect grid button presses.

Extracts frames, detects lit LEDs on the grid, and builds a timeline
of which buttons are pressed when.
"""

import subprocess, json, os, sys
import numpy as np

VIDEO_PATH = 'analysis/original/Monome video demo.mp4'
VIDEO_START = 178.0
VIDEO_END = 290.0
FRAME_INTERVAL = 0.5

GRID_COLS = 16
GRID_ROWS = 8


def extract_frames(video_path, start_s, end_s, interval_s, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    times = np.arange(start_s, end_s, interval_s)
    for i, t in enumerate(times):
        out_path = os.path.join(out_dir, f'frame_{i:04d}_{t:.1f}s.png')
        if not os.path.exists(out_path):
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(t), '-i', video_path,
                '-frames:v', '1', '-q:v', '2', out_path
            ], capture_output=True)
    return times


def detect_leds(frame_path):
    from PIL import Image
    from scipy.ndimage import label, center_of_mass
    img = Image.open(frame_path)
    arr = np.array(img)
    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    brightness = r + g + b
    orange_mask = (
        (r > 150) & (g > 50) & (g < 220) & (b < 100) &
        (r > g * 1.2) & (brightness > 250)
    )
    if not np.any(orange_mask):
        return []
    labeled, n_features = label(orange_mask)
    if n_features == 0:
        return []
    led_positions = []
    for idx in range(1, n_features + 1):
        size = np.sum(labeled == idx)
        if 3 < size < 200:
            cy, cx = center_of_mass(orange_mask, labeled, idx)
            led_positions.append((float(cx), float(cy)))
    return led_positions


def find_grid_bounds(all_led_positions):
    if not all_led_positions:
        return None
    all_pts = np.array(all_led_positions)
    return (np.percentile(all_pts[:,0], 5), np.percentile(all_pts[:,1], 5),
            np.percentile(all_pts[:,0], 95), np.percentile(all_pts[:,1], 95))


def map_leds_to_grid(led_positions, grid_bounds):
    if not led_positions or grid_bounds is None:
        return set()
    x_min, y_min, x_max, y_max = grid_bounds
    cell_w = (x_max - x_min) / GRID_COLS
    cell_h = (y_max - y_min) / GRID_ROWS
    lit_cells = set()
    for x, y in led_positions:
        col = int((x - x_min) / cell_w)
        row = int((y - y_min) / cell_h)
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            lit_cells.add((row, col))
    return lit_cells


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(root, VIDEO_PATH)
    frame_dir = os.path.join(root, 'analysis', 'process', 'video_frames')

    if not os.path.exists(video_path):
        print(f"ERROR: {video_path} not found"); sys.exit(1)

    print(f"Extracting frames from {VIDEO_START}s to {VIDEO_END}s...")
    times = extract_frames(video_path, VIDEO_START, VIDEO_END, FRAME_INTERVAL, frame_dir)
    print(f"  {len(times)} frames")

    print("Detecting LEDs...")
    all_led_positions = []
    frame_leds = []
    for i, t in enumerate(times):
        frame_path = os.path.join(frame_dir, f'frame_{i:04d}_{t:.1f}s.png')
        leds = detect_leds(frame_path)
        frame_leds.append(leds)
        all_led_positions.extend(leds)
        if len(leds) > 0 and i % 20 == 0:
            print(f"  t={t:.1f}s: {len(leds)} LEDs")

    total = sum(len(fl) for fl in frame_leds)
    print(f"  Total: {total} LED detections")
    if total == 0:
        print("ERROR: No LEDs detected"); sys.exit(1)

    grid_bounds = find_grid_bounds(all_led_positions)
    print(f"  Grid: x=[{grid_bounds[0]:.0f},{grid_bounds[2]:.0f}] y=[{grid_bounds[1]:.0f},{grid_bounds[3]:.0f}]")

    print("Building timeline...")
    timeline = []
    for i, t in enumerate(times):
        lit_cells = map_leds_to_grid(frame_leds[i], grid_bounds)
        ref_time = t - VIDEO_START
        timeline.append({
            'video_time': round(t, 1),
            'ref_time': round(ref_time, 1),
            'lit_cells': sorted([list(c) for c in lit_cells]),
            'n_lit': len(lit_cells),
        })

    # Detect press/release events
    print("\nButton events:")
    prev_cells = set()
    events = []
    for entry in timeline:
        current_cells = set(tuple(c) for c in entry['lit_cells'])
        pressed = current_cells - prev_cells
        released = prev_cells - current_cells
        for r, c in sorted(pressed):
            events.append({'time': entry['ref_time'], 'action': 'press', 'row': r, 'col': c})
            print(f"  {entry['ref_time']:6.1f}s: PRESS   row={r} col={c}")
        for r, c in sorted(released):
            events.append({'time': entry['ref_time'], 'action': 'release', 'row': r, 'col': c})
            print(f"  {entry['ref_time']:6.1f}s: RELEASE row={r} col={c}")
        prev_cells = current_cells

    n_press = sum(1 for e in events if e['action'] == 'press')
    n_release = sum(1 for e in events if e['action'] == 'release')
    print(f"\nTotal: {n_press} presses, {n_release} releases")

    # Row activation progression
    print("\nRow activation:")
    seen_rows = set()
    for entry in timeline:
        rows_now = set(r for r, c in entry['lit_cells'])
        new_rows = rows_now - seen_rows
        if new_rows:
            seen_rows.update(new_rows)
            print(f"  t={entry['ref_time']:.1f}s: active rows = {sorted(seen_rows)}")

    out_path = os.path.join(root, 'analysis', 'process', 'video_grid_timeline.json')
    output = {
        'grid_bounds': list(grid_bounds),
        'frame_interval': FRAME_INTERVAL,
        'video_start': VIDEO_START,
        'timeline': timeline,
        'events': events,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
