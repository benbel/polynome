#!/usr/bin/env python3
import json, os, sys
import numpy as np
import soundfile as sf

MIN_PEAK = 0.3

# Every browser we target has to be able to decode the sprites. Safari (and so
# every iPhone) has no Ogg Vorbis decoder, which makes an ogg build load
# without error and then play nothing at all.
PLAYABLE = {'.mp3', '.wav', '.m4a'}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets = os.path.join(root, 'assets')
index = os.path.join(assets, 'modes.json')

if not os.path.exists(index):
    sys.exit('missing assets/modes.json')

modes = json.load(open(index))
if not modes:
    sys.exit('assets/modes.json is empty')

for mode in modes:
    path = os.path.join(assets, mode, 'manifest.json')
    if not os.path.exists(path):
        sys.exit(f'missing assets/{mode}/manifest.json')

    manifest = json.load(open(path))
    instruments = manifest.get('instruments') or []
    if not instruments:
        sys.exit(f'{mode}: no instruments')

    for inst in instruments:
        sprite = os.path.join(assets, mode, inst['sprite'])
        ext = os.path.splitext(sprite)[1].lower()
        if ext not in PLAYABLE:
            sys.exit(f'{mode}/' + inst['id'] + f': {ext} sprites are silent in Safari '
                     'and on iOS -- rebuild with: python build/build.py --format mp3')
        if not os.path.exists(sprite) or os.path.getsize(sprite) == 0:
            sys.exit(f'{mode}: missing or empty ' + inst['sprite'])
        if len(inst['offsets']) != inst['pitchCount']:
            sys.exit(f'{mode}/' + inst['id'] + ': offset count does not match pitch count')

        audio, rate = sf.read(sprite)
        if abs(len(audio) / rate - inst['spriteDuration']) > 0.05:
            sys.exit(f'{mode}/' + inst['id'] + ': sprite length does not match manifest')

        for i, (start, dur) in enumerate(inst['offsets']):
            segment = audio[int(start * rate):int((start + dur) * rate)]
            if len(segment) == 0 or np.max(np.abs(segment)) < MIN_PEAK:
                sys.exit(f'{mode}/' + inst['id'] + f': pitch {i} is silent')

    pitches = sum(i['pitchCount'] for i in instruments)
    print(f'{mode}: {len(instruments)} sprites, {pitches} pitches')
