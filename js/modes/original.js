// original mode — Replication of stretta's monome Press Cafe video.
// The performer only had ~4-5 cells active at a time.  Each cell's
// column sets its repeat period, so a handful of cells creates the
// characteristic cascading polyrhythmic texture.

const V = 0.5;

function pat(cells) {
  return { grids: [{ cells, instrument: 'tone' }], melody: { cells: {}, instrument: 'tone' } };
}

// Pattern 1 — 2 cells
const p1 = pat({
  '3-12': 1,   // period 4
  '7-10': V,   // period 6
});

// Pattern 2 — 3 cells
const p2 = pat({
  '0-14': V,   // period 2
  '3-12': 1,   // period 4
  '7-10': V,   // period 6
});

// Pattern 3 — 4 cells
const p3 = pat({
  '0-14': V,   // period 2
  '1-12': 1,   // period 4
  '5-10': V,   // period 6
  '7-7': V,    // period 9
});

// Pattern 4 — 5 cells
const p4 = pat({
  '0-14': V,   // period 2
  '2-12': 1,   // period 4
  '4-9': V,    // period 7
  '6-7': V,    // period 9
  '7-5': V,    // period 11
});

// Pattern 5 — 4 cells, different positions
const p5 = pat({
  '1-12': 1,   // period 4
  '3-8': 1,    // period 8
  '5-10': V,   // period 6
  '7-6': V,    // period 10
});

// Pattern 6 — 5 cells, final variation
const p6 = pat({
  '0-14': V,   // period 2
  '2-12': 1,   // period 4
  '4-8': 1,    // period 8
  '6-10': V,   // period 6
  '7-5': V,    // period 11
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
  numPatterns: 6,
  cellSize: 28,
  hideControls: ['melody', 'cycle'],
  defaultPatterns: [p1, p2, p3, p4, p5, p6],
};
