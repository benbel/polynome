#!/usr/bin/env python3
import argparse, importlib, json, os, sys

MODES = ['original', 'texture', 'clear', 'voices', 'speech']


def main():
    parser = argparse.ArgumentParser(description='generate audio assets for polynome modes')
    parser.add_argument('modes', nargs='*', metavar='mode',
                        help=f'modes to build, any of: {", ".join(MODES)} (default: all)')
    parser.add_argument('--format', default='ogg', choices=['ogg', 'wav'],
                        help='audio file format (default: ogg)')
    args = parser.parse_args()

    unknown = [m for m in args.modes if m not in MODES]
    if unknown:
        parser.error(f'unknown mode(s): {", ".join(unknown)}')

    modes = args.modes or MODES
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(root, 'assets')
    os.makedirs(assets, exist_ok=True)

    built = []
    for mode in modes:
        out_dir = os.path.join(assets, mode)
        os.makedirs(out_dir, exist_ok=True)

        try:
            gen_module = importlib.import_module(f'gen_{mode}')
        except ModuleNotFoundError:
            print(f'[skip] gen_{mode}.py not found')
            continue

        print(f'[build] {mode} -> {out_dir} ({args.format})')
        gen_module.generate(out_dir, fmt=args.format)
        built.append(mode)
        print(f'[done] {mode}')

    index = [m for m in MODES
             if m in built or os.path.exists(os.path.join(assets, m, 'manifest.json'))]
    with open(os.path.join(assets, 'modes.json'), 'w') as f:
        json.dump(index, f)

    if not built:
        sys.exit('nothing built')


if __name__ == '__main__':
    main()
