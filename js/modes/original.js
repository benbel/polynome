// original mode — Replication of stretta's monome Press Cafe video.
// Grid data, velocities, pitches, and tempo extracted from the Gridlab
// Connect Press Cafe.amxd file.  Eight patterns build from sparse to
// the full dense state, simulating the performer adding cells over time.

// Velocity per column from amxd: 127 36 64 36 127 36 64 36 127 36 72 36 127 36 90 36
// Cols 0,4,8,12 → vol 1 (strong); all others → vol 0.5 (soft)
const V = 0.5;

// Helper: build a pattern object from a cells dict
function pat(cells) {
  return { grids: [{ cells, instrument: 'tone' }], melody: { cells: {}, instrument: 'tone' } };
}

// Pattern 1 — opening: 3 slow pulses
const p1 = pat({
  '0-0': 1,
  '3-0': 1,
  '7-0': 1,
});

// Pattern 2 — add faster repeats, introduce row 1
const p2 = pat({
  '0-0': 1,  '0-8': 1,
  '1-0': 1,
  '3-0': 1,  '3-8': 1,
  '7-0': 1,  '7-5': V,
});

// Pattern 3 — diagonal emerging, add rows 5
const p3 = pat({
  '0-0': 1,  '0-4': 1,  '0-8': 1,
  '1-0': 1,  '1-8': 1,
  '3-0': 1,  '3-4': 1,  '3-8': 1,
  '5-0': 1,  '5-4': 1,
  '7-0': 1,  '7-5': V,  '7-11': V,
});

// Pattern 4 — all rows active, denser columns
const p4 = pat({
  '0-0': 1,  '0-4': 1,  '0-8': 1,  '0-12': 1,
  '1-0': 1,  '1-4': 1,  '1-8': 1,
  '3-0': 1,  '3-4': 1,  '3-8': 1,  '3-12': 1,
  '4-0': 1,  '4-4': 1,
  '5-0': 1,  '5-4': 1,  '5-9': V,
  '6-0': 1,  '6-4': 1,
  '7-0': 1,  '7-2': V,  '7-5': V,  '7-11': V,
});

// Pattern 5 — half-density, row 2 enters
const p5 = pat({
  '0-0': 1,  '0-2': V,  '0-4': 1,  '0-8': 1,  '0-12': 1,
  '1-0': 1,  '1-4': 1,  '1-8': 1,  '1-12': 1,
  '2-0': 1,  '2-4': 1,  '2-8': 1,  '2-12': 1,
  '3-0': 1,  '3-3': V,  '3-4': 1,  '3-7': V,  '3-8': 1,  '3-12': 1,
  '4-0': 1,  '4-4': 1,  '4-8': 1,
  '5-0': 1,  '5-2': V,  '5-4': 1,  '5-7': V,  '5-9': V,
  '6-0': 1,  '6-4': 1,  '6-10': V,
  '7-0': 1,  '7-2': V,  '7-3': V,  '7-5': V,  '7-11': V,
});

// Pattern 6 — approaching full, odd-numbered cols filling in
const p6 = pat({
  '0-0': 1,  '0-2': V,  '0-4': 1,  '0-6': V,  '0-8': 1,  '0-12': 1,  '0-14': V,
  '1-0': 1,  '1-4': 1,  '1-8': 1,  '1-12': 1,
  '2-0': 1,  '2-1': V,  '2-4': 1,  '2-5': V,  '2-8': 1,  '2-9': V,  '2-12': 1,  '2-13': V,
  '3-0': 1,  '3-3': V,  '3-4': 1,  '3-7': V,  '3-8': 1,  '3-11': V,  '3-12': 1,
  '4-0': 1,  '4-1': V,  '4-4': 1,  '4-6': V,  '4-8': 1,  '4-12': 1,  '4-14': V,
  '5-0': 1,  '5-2': V,  '5-4': 1,  '5-5': V,  '5-7': V,  '5-11': V,  '5-13': V,
  '6-0': 1,  '6-1': V,  '6-4': 1,  '6-6': V,  '6-12': 1,
  '7-0': 1,  '7-2': V,  '7-3': V,  '7-5': V,  '7-7': V,  '7-11': V,  '7-13': V,
});

// Pattern 7 — nearly full
const p7 = pat({
  '0-0': 1,  '0-2': V,  '0-4': 1,  '0-6': V,  '0-8': 1,  '0-10': V,  '0-12': 1,  '0-14': V,
  '1-0': 1,  '1-4': 1,  '1-8': 1,  '1-12': 1,
  '2-0': 1,  '2-1': V,  '2-3': V,  '2-4': 1,  '2-5': V,  '2-8': 1,  '2-9': V,  '2-12': 1,  '2-13': V,
  '3-0': 1,  '3-3': V,  '3-4': 1,  '3-7': V,  '3-8': 1,  '3-11': V,  '3-12': 1,  '3-15': V,
  '4-0': 1,  '4-1': V,  '4-3': V,  '4-4': 1,  '4-6': V,  '4-8': 1,  '4-12': 1,  '4-14': V,
  '5-0': 1,  '5-2': V,  '5-4': 1,  '5-5': V,  '5-7': V,  '5-9': V,  '5-13': V,
  '6-0': 1,  '6-1': V,  '6-4': 1,  '6-6': V,  '6-10': V,  '6-12': 1,  '6-15': V,
  '7-0': 1,  '7-2': V,  '7-3': V,  '7-5': V,  '7-7': V,  '7-11': V,  '7-13': V,
});

// Pattern 8 — full amxd pattern
const p8 = pat({
  '0-0': 1,  '0-2': V,  '0-4': 1,  '0-6': V,  '0-8': 1,  '0-10': V,  '0-12': 1,  '0-14': V,
  '1-0': 1,  '1-4': 1,  '1-8': 1,  '1-12': 1,
  '2-0': 1,  '2-1': V,  '2-3': V,  '2-4': 1,  '2-5': V,  '2-7': V,  '2-8': 1,  '2-9': V,  '2-11': V,  '2-12': 1,  '2-13': V,  '2-15': V,
  '3-0': 1,  '3-3': V,  '3-4': 1,  '3-7': V,  '3-8': 1,  '3-11': V,  '3-12': 1,  '3-15': V,
  '4-0': 1,  '4-1': V,  '4-3': V,  '4-4': 1,  '4-6': V,  '4-8': 1,  '4-10': V,  '4-12': 1,  '4-14': V,
  '5-0': 1,  '5-2': V,  '5-4': 1,  '5-5': V,  '5-7': V,  '5-9': V,  '5-11': V,  '5-13': V,  '5-15': V,
  '6-0': 1,  '6-1': V,  '6-4': 1,  '6-6': V,  '6-10': V,  '6-12': 1,  '6-15': V,
  '7-0': 1,  '7-2': V,  '7-3': V,  '7-5': V,  '7-7': V,  '7-11': V,  '7-13': V,
});

export default {
  id: 'original',
  label: 'original',

  mainGrids: [
    { rows: 8, cols: 16, defaultInstrument: 'tone' },
  ],
  mainGridLayout: { cols: 1 },

  melodyGrids: null,

  mainInstruments: [
    { id: 'tone', label: 'tone' },
  ],
  melodyInstruments: [],

  fx: {
    tone: { delay: 0.15, reverb: 0.25, gain: 0.35 },
  },

  effects: {
    reverbWet: 0.20,
    reverbDark: 0.35,
    reverbLength: 2.0,
    delayL: 0.33,
    delayR: 0.22,
    delayFeedback: 0.25,
    delayDarkLP: 3000,
    delayWet: 0.15,
    compThreshold: -18,
    compRatio: 4,
  },

  // MIDI 60,59,57,55,53,52,50,48 → C4 down to C3
  mainFreqs: [261.6, 246.9, 220.0, 196.0, 174.6, 164.8, 146.8, 130.8],
  defaultStepMs: 450,
  numPatterns: 8,
  cellSize: 28,
  hideControls: ['melody', 'cycle'],
  defaultPatterns: [p1, p2, p3, p4, p5, p6, p7, p8],
};
