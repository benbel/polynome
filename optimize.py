#!/usr/bin/env python3
"""Polynome optimizer — run E/M iterations until stopped.

Usage:
    python optimize.py                   # default: block coordinate descent
    python optimize.py --joint           # joint CMA-ES instead
    python optimize.py --budget 100      # evals per block per M-step
    python optimize.py --skip-estep      # skip pattern inference (M-step only)
    python optimize.py --blocks A_synthesis B_harmonics  # specific blocks

Press Ctrl+C to stop gracefully (saves current best before exit).
"""

import argparse, json, os, signal, sys, time, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build'))

from optimize_full import (
    PARAM_SPEC, PARAM_NAMES, PARAM_DEFAULTS, PARAM_BOUNDS, BLOCKS,
    SR, COMPARE_SR,
    unpack_params, render_with_params, compute_composite,
    infer_buttons, optimize_block_cmaes, optimize_joint_cmaes,
    get_block_indices,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, 'analysis', 'process', 'optimizer_state.json')
LOG_PATH = os.path.join(ROOT, 'analysis', 'process', 'optimization_log.csv')

# Graceful shutdown
_stop = False
def _handle_signal(sig, frame):
    global _stop
    _stop = True
    print("\n[!] Ctrl+C received — finishing current evaluation, then saving & exiting...")
signal.signal(signal.SIGINT, _handle_signal)


def load_state():
    """Load saved optimizer state (params + patterns), or return defaults."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        # Reconstruct param vector from saved params dict
        x = PARAM_DEFAULTS.copy()
        saved_params = state.get('params', {})
        for i, (name, _, _, _) in enumerate(PARAM_SPEC):
            if name in saved_params:
                x[i] = saved_params[name]
        # Reconstruct patterns
        patterns = []
        for pat_dict in state.get('patterns', []):
            pat = {}
            for key, vel in pat_dict.items():
                r, c = key.split('-')
                pat[(int(r), int(c))] = vel
            patterns.append(pat)
        iteration = state.get('iteration', 0)
        composite = state.get('composite', float('inf'))
        print(f"[resume] Loaded state from iteration {iteration}, composite={composite:.4f}")
        return x, patterns, iteration, composite
    else:
        # Fall back to defaults from code
        from render_original import PATTERNS
        return PARAM_DEFAULTS.copy(), PATTERNS, 0, float('inf')


def save_state(x, patterns, iteration, composite, metrics=None):
    """Save optimizer state to disk."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    state = {
        'iteration': iteration,
        'composite': composite,
        'metrics': {k: round(v, 6) for k, v in metrics.items()} if metrics else {},
        'params': {name: round(float(x[i]), 6)
                   for i, (name, _, _, _) in enumerate(PARAM_SPEC)},
        'patterns': [
            {f"{r}-{c}": v for (r, c), v in pat.items()}
            for pat in patterns
        ],
    }
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def log_iteration(iteration, composite, metrics):
    """Append iteration result to CSV log."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['iteration', 'composite', 'timestamp'] +
                            sorted(metrics.keys()))
        writer.writerow([iteration, f'{composite:.6f}', time.strftime('%Y-%m-%d %H:%M:%S')] +
                        [f'{metrics[k]:.6f}' for k in sorted(metrics.keys())])


def main():
    parser = argparse.ArgumentParser(description='Polynome E/M optimizer (runs until Ctrl+C)')
    parser.add_argument('--budget', type=int, default=150,
                        help='Max evaluations per block per M-step (default: 150)')
    parser.add_argument('--blocks', nargs='*', default=None,
                        help='Specific blocks to optimize (default: all)')
    parser.add_argument('--joint', action='store_true',
                        help='Optimize all params jointly instead of block-by-block')
    parser.add_argument('--skip-estep', action='store_true',
                        help='Skip E-step (pattern inference)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset state (start from PARAM_DEFAULTS)')
    args = parser.parse_args()

    import librosa

    ref_path = os.path.join(ROOT, 'analysis', 'reference.wav')
    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    print(f"Loading reference: {ref_path}")
    y_ref_compare, _ = librosa.load(ref_path, sr=COMPARE_SR, mono=True)
    y_ref_full, _ = librosa.load(ref_path, sr=SR, mono=True)
    print(f"  Duration: {len(y_ref_compare)/COMPARE_SR:.1f}s")

    # Load or reset state
    if args.reset or not os.path.exists(STATE_PATH):
        from render_original import PATTERNS
        x = PARAM_DEFAULTS.copy()
        patterns = PATTERNS
        iteration = 0
        best_composite = float('inf')
        print("[init] Starting from PARAM_DEFAULTS")
    else:
        x, patterns, iteration, best_composite = load_state()

    # Select blocks
    block_order = args.blocks or ['A_synthesis', 'B_harmonics', 'C_body',
                                   'C_eq', 'D_effects', 'E_frequencies', 'F_timing']
    for b in block_order:
        if b not in BLOCKS:
            print(f"ERROR: unknown block '{b}'. Available: {list(BLOCKS.keys())}")
            sys.exit(1)

    # Initial evaluation
    print(f"\n{'='*70}")
    print(f"Initial evaluation...")
    t0 = time.time()
    y_init = render_with_params(x, patterns, fast=False)
    c_init, m_init = compute_composite(y_init, y_ref_compare)
    t_eval = time.time() - t0
    print(f"  Composite: {c_init:.4f} ({t_eval:.1f}s per eval)")
    print(f"  Metrics: {', '.join(f'{k}={v:.3f}' for k, v in sorted(m_init.items()))}")
    best_composite = c_init
    best_x = x.copy()
    best_patterns = patterns

    # Log initial state
    log_iteration(iteration, c_init, m_init)

    print(f"\nStarting E/M iterations (Ctrl+C to stop)...")
    print(f"{'='*70}\n")

    while not _stop:
        iteration += 1
        iter_start = time.time()

        print(f"{'='*70}")
        print(f"  ITERATION {iteration}")
        print(f"{'='*70}")

        # ── E-step ──────────────────────────────────────────────────────
        if not args.skip_estep:
            print(f"\n  [E-step] Inferring button presses...")
            p = unpack_params(x)
            inferred = infer_buttons(y_ref_full, SR, p['freqs'], p['step_ms'])
            if inferred:
                n_inf = len(inferred)
                total_cells = sum(len(pat) for pat in inferred)
                print(f"    Inferred {n_inf} patterns, {total_cells} total cells")

                y_inferred = render_with_params(x, inferred, fast=False)
                c_inferred, _ = compute_composite(y_inferred, y_ref_compare)

                y_current = render_with_params(x, patterns, fast=False)
                c_current, _ = compute_composite(y_current, y_ref_compare)

                print(f"    Current:  {c_current:.4f}")
                print(f"    Inferred: {c_inferred:.4f}")

                if c_inferred < c_current:
                    print(f"    -> Using inferred patterns")
                    patterns = inferred
                else:
                    print(f"    -> Keeping current patterns")
            else:
                print(f"    No patterns inferred, keeping current")

        if _stop:
            break

        # ── M-step ──────────────────────────────────────────────────────
        if args.joint:
            print(f"\n  [M-step] Joint CMA-ES ({len(PARAM_SPEC)} params, budget={args.budget})")
            x_improved, c_improved, n_evals = optimize_joint_cmaes(
                x, patterns, y_ref_compare, budget=args.budget)
            if c_improved < best_composite:
                x = x_improved
                best_composite = c_improved
                best_x = x.copy()
                best_patterns = patterns
                print(f"    -> Improved to {c_improved:.4f} ({n_evals} evals)")
            else:
                print(f"    -> No improvement ({n_evals} evals)")
        else:
            print(f"\n  [M-step] Block coordinate descent ({len(block_order)} blocks, "
                  f"budget={args.budget}/block)")

            for block_name in block_order:
                if _stop:
                    break
                indices = get_block_indices(block_name)
                n_params = len(indices)
                param_names = [PARAM_NAMES[i] for i in indices]
                print(f"\n    Block {block_name} ({n_params} params: "
                      f"{', '.join(param_names[:4])}{'...' if n_params > 4 else ''})")

                x_before = x.copy()
                c_before = best_composite

                x_improved, c_improved, n_evals = optimize_block_cmaes(
                    block_name, x, patterns, y_ref_compare, budget=args.budget)

                if c_improved < c_before:
                    improvement = c_before - c_improved
                    x = x_improved
                    best_composite = c_improved
                    best_x = x.copy()
                    best_patterns = patterns
                    print(f"      -> {c_before:.4f} -> {c_improved:.4f} "
                          f"(delta={improvement:.4f}, {n_evals} evals)")
                else:
                    x = x_before
                    print(f"      -> No improvement ({n_evals} evals)")

        # ── End of iteration ────────────────────────────────────────────
        elapsed = time.time() - iter_start

        # Full evaluation for logging
        y_eval = render_with_params(best_x, best_patterns, fast=False)
        c_eval, m_eval = compute_composite(y_eval, y_ref_compare)

        print(f"\n  >>> ITERATION {iteration} DONE | "
              f"composite={c_eval:.4f} | "
              f"time={elapsed:.0f}s")
        print(f"      Metrics: {', '.join(f'{k}={v:.3f}' for k, v in sorted(m_eval.items()))}")

        # Save state and log
        save_state(best_x, best_patterns, iteration, c_eval, m_eval)
        log_iteration(iteration, c_eval, m_eval)
        print(f"      Saved state to {STATE_PATH}")

    # ── Final save on exit ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STOPPED after iteration {iteration}")
    print(f"Best composite: {best_composite:.4f}")
    print(f"State saved to: {STATE_PATH}")
    print(f"Log saved to:   {LOG_PATH}")
    print(f"{'='*70}")

    # Save final state
    y_final = render_with_params(best_x, best_patterns, fast=False)
    c_final, m_final = compute_composite(y_final, y_ref_compare)
    save_state(best_x, best_patterns, iteration, c_final, m_final)

    # Print current params for reference
    p = unpack_params(best_x)
    print(f"\nFrequencies: {[round(f, 1) for f in p['freqs']]}")
    print(f"Step ms: {p['step_ms']:.1f}")


if __name__ == '__main__':
    main()
