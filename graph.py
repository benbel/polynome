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
    metrics_data = {}

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            composites.append(float(row['composite']))
            for k, v in row.items():
                if k not in ('iteration', 'composite', 'timestamp'):
                    if k not in metrics_data:
                        metrics_data[k] = []
                    metrics_data[k].append(float(v))

    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: composite distance
    ax1.plot(iterations, composites, 'b-o', markersize=4, linewidth=1.5, label='Composite')
    ax1.set_ylabel('Composite Distance')
    ax1.set_title('Optimization Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if composites:
        ax1.set_ylim(bottom=0, top=max(composites) * 1.1)

    # Bottom: per-metric breakdown
    colors = plt.cm.tab10(range(len(metrics_data)))
    for (name, values), color in zip(sorted(metrics_data.items()), colors):
        ax2.plot(iterations, values, '-', linewidth=1, label=name, color=color, alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Per-Metric Breakdown')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)

    plt.tight_layout()

    if args.save:
        out_path = os.path.join(root, 'analysis', 'process', 'optimization_progress.png')
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
