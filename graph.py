#!/usr/bin/env python3
"""Plot optimization progress: composite distance vs iteration number.

Usage:
    python graph.py                # show plot
    python graph.py --save         # save to analysis/process/optimization_progress.png
"""

import argparse, csv, os, sys

def main():
    parser = argparse.ArgumentParser(description='Plot optimization progress')
    parser.add_argument('--save', action='store_true', help='Save to PNG instead of showing')
    parser.add_argument('--log', default=None, help='Path to optimization_log.csv')
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    log_path = args.log or os.path.join(root, 'analysis', 'process', 'optimization_log.csv')

    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found. Run optimize.py first.")
        sys.exit(1)

    iterations = []
    composites = []
    descriptions = []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            composites.append(float(row['composite']))
            descriptions.append(row.get('description', ''))

    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iterations, composites, 'b-o', markersize=3, linewidth=1.5, label='Composite Distance')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Composite Distance')
    ax.set_title('Polynome Optimization Progress')
    ax.grid(True, alpha=0.3)

    if composites:
        ax.set_ylim(bottom=0, top=max(composites) * 1.1)

    # Annotate key milestones
    best_so_far = float('inf')
    milestones = []
    for i, (it, comp, desc) in enumerate(zip(iterations, composites, descriptions)):
        if comp < best_so_far:
            best_so_far = comp
            milestones.append((it, comp, desc))

    # Show a few key milestone annotations (not all, to avoid clutter)
    step = max(1, len(milestones) // 8)
    for idx in range(0, len(milestones), step):
        it, comp, desc = milestones[idx]
        short = desc[:30] + '...' if len(desc) > 30 else desc
        if short:
            ax.annotate(f'{comp:.3f}', (it, comp),
                        textcoords="offset points", xytext=(5, 8),
                        fontsize=7, color='darkblue', alpha=0.8)
    # Always annotate the last point
    if milestones:
        it, comp, desc = milestones[-1]
        ax.annotate(f'{comp:.4f}', (it, comp),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=8, color='red', fontweight='bold')

    ax.legend()
    plt.tight_layout()

    if args.save:
        out_path = os.path.join(root, 'analysis', 'process', 'optimization_progress.png')
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
