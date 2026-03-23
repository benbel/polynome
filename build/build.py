#!/usr/bin/env python3
"""Build entry point: generate audio assets for polynome modes.

Usage:
    python build/build.py                   # all modes
    python build/build.py texture clear     # specific modes
    python build/build.py --format wav      # WAV instead of OGG
"""

import sys, os, importlib

MODES = ['original', 'texture', 'clear', 'voices', 'speech']


def main():
    args = sys.argv[1:]
    fmt = 'ogg'
    modes = []

    for arg in args:
        if arg == '--format':
            continue
        if args[args.index(arg) - 1] == '--format' if args.index(arg) > 0 else False:
            fmt = arg
            continue
        if arg.startswith('--format='):
            fmt = arg.split('=', 1)[1]
            continue
        if arg in MODES:
            modes.append(arg)

    if not modes:
        modes = MODES

    # Parse --format properly
    for i, arg in enumerate(args):
        if arg == '--format' and i + 1 < len(args):
            fmt = args[i + 1]

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for mode in modes:
        out_dir = os.path.join(root, 'assets', mode)
        os.makedirs(out_dir, exist_ok=True)

        try:
            gen_module = importlib.import_module(f'gen_{mode}')
        except ModuleNotFoundError:
            print(f'[skip] gen_{mode}.py not found')
            continue

        print(f'[build] {mode} -> {out_dir} ({fmt})')
        gen_module.generate(out_dir, fmt=fmt)
        print(f'[done] {mode}')


if __name__ == '__main__':
    main()
