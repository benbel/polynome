"""Pitch construction for polynome.

Every frequency in the app is derived here rather than picked by ear:
12-tone equal temperament referenced to A4 = 440 Hz, laid out as degrees
of a named scale.

Why pentatonic by default
-------------------------
The main grid is not a step sequencer.  A cell in column `c` fires every
`cols - c` steps, so active rows are free-running against each other and
any combination of pitches can sound at once -- there is no bar in which
to resolve a dissonance.  A pentatonic collection has no semitones and no
tritone between any two of its members, so every simultaneity the machine
can produce is consonant.  That is what makes an arbitrary polyrhythm
sound intentional instead of accidental.

Grid rows are numbered top-down (row 0 is the highest pitch), so the
tables these helpers build descend.
"""

A4_HZ = 440.0
A4_MIDI = 69

SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

SCALES = {
    'major': (0, 2, 4, 5, 7, 9, 11),
    'natural_minor': (0, 2, 3, 5, 7, 8, 10),
    'dorian': (0, 2, 3, 5, 7, 9, 10),
    'minor_pentatonic': (0, 3, 5, 7, 10),
    'major_pentatonic': (0, 2, 4, 7, 9),
}

_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def midi(note):
    """Note name to MIDI number: 'A4' -> 69.  Accepts 'F#3' and 'Eb5'."""
    letter = note[0].upper()
    if letter not in SEMITONE:
        raise ValueError(f'bad note name: {note!r}')

    semitone = SEMITONE[letter]
    i = 1
    while i < len(note) and note[i] in '#b':
        semitone += 1 if note[i] == '#' else -1
        i += 1
    if i >= len(note):
        raise ValueError(f'note name needs an octave: {note!r}')

    return 12 * (int(note[i:]) + 1) + semitone


def hz(note):
    """Frequency of a note name or MIDI number in 12-TET."""
    n = note if isinstance(note, (int, float)) else midi(note)
    return A4_HZ * 2 ** ((n - A4_MIDI) / 12)


def note_name(n):
    """MIDI number to note name: 69 -> 'A4'."""
    return f'{_NAMES[n % 12]}{n // 12 - 1}'


def descending_midi(root, scale, top, count):
    """`count` MIDI numbers of `scale` on `root`, stepping down from `top`.

    `top` may be a note name or a MIDI number, and is snapped down to the
    nearest degree of the scale if it is not itself one.
    """
    if scale not in SCALES:
        raise ValueError(f'unknown scale: {scale!r} (have {sorted(SCALES)})')

    degrees = SCALES[scale]
    root_pc = midi(root + '0') % 12
    n = midi(top) if isinstance(top, str) else int(top)

    pitches = []
    while len(pitches) < count:
        if n < 0:
            raise ValueError(f'{scale} on {root} runs out of pitches below {top}')
        if (n - root_pc) % 12 in degrees:
            pitches.append(n)
        n -= 1
    return pitches


def descending_hz(root, scale, top, count, decimals=2):
    """`count` frequencies of `scale` on `root`, stepping down from `top`."""
    return [round(hz(n), decimals) for n in descending_midi(root, scale, top, count)]


def spell(root, scale, top, count):
    """The same run as note names, for build logs: 'A5 G5 E5 D5 ...'."""
    return ' '.join(note_name(n) for n in descending_midi(root, scale, top, count))
