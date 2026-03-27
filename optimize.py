#!/usr/bin/env python3
"""Polynome optimizer — run E/M iterations until stopped.

Fully autonomous: adapts budget, strategy, and block ordering based on
which parameters are improving. No manual tuning needed.

Usage:
    python optimize.py           # run until Ctrl+C
    python optimize.py --reset   # start fresh from PARAM_DEFAULTS

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
    """Load saved optimizer state, or return defaults."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        x = PARAM_DEFAULTS.copy()
        saved_params = state.get('params', {})
        for i, (name, _, _, _) in enumerate(PARAM_SPEC):
            if name in saved_params:
                x[i] = saved_params[name]
        patterns = []
        for pat_dict in state.get('patterns', []):
            pat = {}
            for key, vel in pat_dict.items():
                r, c = key.split('-')
                pat[(int(r), int(c))] = vel
            patterns.append(pat)
        iteration = state.get('iteration', 0)
        composite = state.get('composite', float('inf'))
        block_stats = state.get('block_stats', {})
        print(f"[resume] Loaded state from iteration {iteration}, composite={composite:.4f}")
        return x, patterns, iteration, composite, block_stats
    else:
        from render_original import PATTERNS
        return PARAM_DEFAULTS.copy(), PATTERNS, 0, float('inf'), {}


def save_state(x, patterns, iteration, composite, metrics=None, block_stats=None):
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
        'block_stats': block_stats or {},
    }
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def log_iteration(iteration, composite, metrics, strategy=''):
    """Append iteration result to CSV log."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    metric_keys = sorted(metrics.keys())
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['iteration', 'composite', 'timestamp', 'strategy'] + metric_keys)
        writer.writerow([iteration, f'{composite:.6f}', time.strftime('%Y-%m-%d %H:%M:%S'),
                         strategy] + [f'{metrics[k]:.6f}' for k in metric_keys])


ALL_BLOCKS = ['A_synthesis', 'B_harmonics', 'C_body', 'C_eq',
              'D_effects', 'E_frequencies', 'F_timing']


def adaptive_strategy(iteration, block_stats, best_composite):
    """Decide what to do this iteration based on history.

    Returns (strategy, block_order, budget_per_block).

    Strategies:
    - 'blocks_focused': optimize only blocks that have been improving
    - 'blocks_full': optimize all blocks (exploration)
    - 'joint': optimize all params jointly (escape local minima)
    - 'perturb': random perturbation + local search (shake things up)
    """
    # Every 5th iteration: joint optimization to escape local minima
    if iteration % 5 == 0 and iteration > 0:
        return 'joint', ALL_BLOCKS, 300

    # Every 10th iteration: perturbation to explore new regions
    if iteration % 10 == 0 and iteration > 0:
        return 'perturb', ALL_BLOCKS, 200

    # Rank blocks by recent improvement
    block_improvements = {}
    for block in ALL_BLOCKS:
        stats = block_stats.get(block, {})
        recent_deltas = stats.get('recent_deltas', [])
        # Average improvement over last 3 attempts
        if recent_deltas:
            avg = sum(recent_deltas[-3:]) / len(recent_deltas[-3:])
        else:
            avg = 0.01  # assume worth trying if no data
        block_improvements[block] = avg

    # Sort: most productive blocks first
    sorted_blocks = sorted(ALL_BLOCKS, key=lambda b: block_improvements[b], reverse=True)

    # Count how many blocks improved last round
    n_improved = sum(1 for b in ALL_BLOCKS
                     if block_stats.get(b, {}).get('recent_deltas', [0])[-1:] != [0])

    if n_improved == 0 and iteration > 2:
        # Nothing improved last round — do a full sweep with more budget
        return 'blocks_full', sorted_blocks, 250
    else:
        # Focus on productive blocks, skip stagnant ones (but include all
        # every 3rd iteration to re-check)
        if iteration % 3 == 0:
            block_order = sorted_blocks
        else:
            # Only blocks that improved at least once recently
            block_order = [b for b in sorted_blocks
                           if any(d > 0 for d in block_stats.get(b, {}).get('recent_deltas', [0.01])[-3:])]
            if not block_order:
                block_order = sorted_blocks  # fallback

        # Adaptive budget: more budget for blocks that improve more
        base_budget = 150
        return 'blocks_focused', block_order, base_budget


def perturb_params(x, scale=0.05):
    """Random perturbation within bounds to escape local minima."""
    x_new = x.copy()
    lowers = np.array([b[0] for b in PARAM_BOUNDS])
    uppers = np.array([b[1] for b in PARAM_BOUNDS])
    ranges = uppers - lowers
    noise = np.random.randn(len(x)) * scale * ranges
    x_new = np.clip(x_new + noise, lowers, uppers)
    return x_new


def main():
    parser = argparse.ArgumentParser(description='Polynome autonomous E/M optimizer')
    parser.add_argument('--reset', action='store_true', help='Start fresh from PARAM_DEFAULTS')
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
        block_stats = {}
        print("[init] Starting from PARAM_DEFAULTS")
    else:
        x, patterns, iteration, best_composite, block_stats = load_state()

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

    # Stagnation tracking
    stagnant_count = 0  # consecutive iterations with no improvement
    prev_composite = best_composite

    print(f"\nStarting autonomous E/M optimization (Ctrl+C to stop)...")
    print(f"{'='*70}\n")

    while not _stop:
        iteration += 1
        iter_start = time.time()

        # ── Decide strategy ─────────────────────────────────────────────
        strategy, block_order, budget = adaptive_strategy(
            iteration, block_stats, best_composite)

        print(f"{'='*70}")
        print(f"  ITERATION {iteration} | strategy={strategy} | "
              f"current={best_composite:.4f} | stagnant={stagnant_count}")
        print(f"{'='*70}")

        # ── E-step (every 3rd iteration, or after perturbation) ─────────
        do_estep = (iteration % 3 == 1) or strategy == 'perturb'
        if do_estep:
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
        if strategy == 'joint':
            print(f"\n  [M-step] Joint CMA-ES ({len(PARAM_SPEC)} params, budget={budget})")
            x_improved, c_improved, n_evals = optimize_joint_cmaes(
                x, patterns, y_ref_compare, budget=budget)
            if c_improved < best_composite:
                x = x_improved
                best_composite = c_improved
                best_x = x.copy()
                best_patterns = patterns
                print(f"    -> Improved to {c_improved:.4f} ({n_evals} evals)")
            else:
                print(f"    -> No improvement ({n_evals} evals)")

        elif strategy == 'perturb':
            print(f"\n  [M-step] Perturbation + local search")
            # Try a few random perturbations, keep best
            scale = 0.03 + 0.02 * min(stagnant_count, 10)  # larger perturbation if more stagnant
            print(f"    Perturbation scale: {scale:.3f}")
            best_perturbed = x.copy()
            best_perturbed_c = best_composite
            for attempt in range(5):
                if _stop:
                    break
                x_pert = perturb_params(x, scale=scale)
                y_pert = render_with_params(x_pert, patterns, fast=False)
                c_pert, _ = compute_composite(y_pert, y_ref_compare)
                if c_pert < best_perturbed_c:
                    best_perturbed = x_pert.copy()
                    best_perturbed_c = c_pert
                    print(f"    Perturbation {attempt+1}: {c_pert:.4f} (new best)")
                else:
                    print(f"    Perturbation {attempt+1}: {c_pert:.4f}")

            # Local search from best perturbation
            if best_perturbed_c < best_composite * 1.05:  # within 5% — worth refining
                print(f"    Refining from {best_perturbed_c:.4f}...")
                x_refined, c_refined, n_evals = optimize_joint_cmaes(
                    best_perturbed, patterns, y_ref_compare, budget=budget)
                if c_refined < best_composite:
                    x = x_refined
                    best_composite = c_refined
                    best_x = x.copy()
                    best_patterns = patterns
                    print(f"    -> Improved to {c_refined:.4f} after refinement")
                else:
                    print(f"    -> No improvement after refinement ({c_refined:.4f})")
            else:
                print(f"    -> Perturbations too far off, skipping refinement")

        else:
            # Block coordinate descent
            print(f"\n  [M-step] Blocks: {', '.join(block_order)} (budget={budget}/block)")

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

                # Adaptive budget per block: give more to productive blocks
                stats = block_stats.get(block_name, {})
                recent = stats.get('recent_deltas', [])
                if recent and max(recent[-3:]) > 0.001:
                    block_budget = int(budget * 1.5)  # productive — invest more
                elif recent and max(recent[-3:]) == 0:
                    block_budget = int(budget * 0.5)  # stagnant — spend less
                else:
                    block_budget = budget

                x_improved, c_improved, n_evals = optimize_block_cmaes(
                    block_name, x, patterns, y_ref_compare, budget=block_budget)

                delta = max(0, c_before - c_improved)
                if c_improved < c_before:
                    x = x_improved
                    best_composite = c_improved
                    best_x = x.copy()
                    best_patterns = patterns
                    print(f"      -> {c_before:.4f} -> {c_improved:.4f} "
                          f"(delta={delta:.4f}, {n_evals} evals)")
                else:
                    x = x_before
                    delta = 0
                    print(f"      -> No improvement ({n_evals} evals)")

                # Update block stats
                if block_name not in block_stats:
                    block_stats[block_name] = {'recent_deltas': [], 'total_evals': 0}
                block_stats[block_name]['recent_deltas'].append(round(delta, 6))
                # Keep last 10
                block_stats[block_name]['recent_deltas'] = \
                    block_stats[block_name]['recent_deltas'][-10:]
                block_stats[block_name]['total_evals'] = \
                    block_stats[block_name].get('total_evals', 0) + n_evals

        # ── End of iteration ────────────────────────────────────────────
        elapsed = time.time() - iter_start

        # Full evaluation for logging
        y_eval = render_with_params(best_x, best_patterns, fast=False)
        c_eval, m_eval = compute_composite(y_eval, y_ref_compare)

        # Stagnation tracking
        if c_eval >= prev_composite - 1e-5:
            stagnant_count += 1
        else:
            stagnant_count = 0
        prev_composite = c_eval

        print(f"\n  >>> ITERATION {iteration} DONE | "
              f"composite={c_eval:.4f} | "
              f"strategy={strategy} | "
              f"time={elapsed:.0f}s")
        print(f"      Metrics: {', '.join(f'{k}={v:.3f}' for k, v in sorted(m_eval.items()))}")

        # Save state and log
        save_state(best_x, best_patterns, iteration, c_eval, m_eval, block_stats)
        log_iteration(iteration, c_eval, m_eval, strategy)
        print(f"      Saved.")

    # ── Final save on exit ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STOPPED after iteration {iteration}")
    print(f"Best composite: {best_composite:.4f}")
    print(f"State saved to: {STATE_PATH}")
    print(f"Log saved to:   {LOG_PATH}")
    print(f"{'='*70}")

    y_final = render_with_params(best_x, best_patterns, fast=False)
    c_final, m_final = compute_composite(y_final, y_ref_compare)
    save_state(best_x, best_patterns, iteration, c_final, m_final, block_stats)

    # Block performance summary
    print(f"\nBlock performance summary:")
    for block in ALL_BLOCKS:
        stats = block_stats.get(block, {})
        deltas = stats.get('recent_deltas', [])
        total = stats.get('total_evals', 0)
        avg = sum(deltas) / len(deltas) if deltas else 0
        print(f"  {block:20s}: avg_delta={avg:.5f}, total_evals={total}")


if __name__ == '__main__':
    main()
