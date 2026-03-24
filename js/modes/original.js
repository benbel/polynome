// original mode — Replication of stretta's monome Press Cafe video.
// The amxd stores step-sequencer output patterns (which steps fire).
// Reverse-engineering those into polynome's column=period system shows
// the performer pressed only ~25 cells total.  Six patterns simulate
// the performer gradually adding cells during the video.

// Velocity per column from amxd: 127 36 64 36 127 36 64 36 127 36 72 36 127 36 90 36
// Cols 0,4,8,12 → vol 1 (strong beat); all others → vol 0.5
const V = 0.5;

function pat(cells) {
  return { grids: [{ cells, instrument: 'tone' }], melody: { cells: {}, instrument: 'tone' } };
}

// Pattern 1 — opening: 2 cells, 2 rows
const p1 = pat({
  '3-12': 1,   // period 4
  '7-10': V,   // period 6
});

// Pattern 2 — 5 cells, 4 rows
const p2 = pat({
  '0-14': V,   // period 2
  '3-12': 1,
  '6-10': V,
  '7-10': V,  '7-7': V,
});

// Pattern 3 — 10 cells, 6 rows
const p3 = pat({
  '0-14': V,
  '1-12': 1,
  '3-12': 1,  '3-8': 1,
  '5-10': V,
  '6-10': V,  '6-7': V,
  '7-10': V,  '7-7': V,  '7-5': V,
});

// Pattern 4 — 16 cells, all 8 rows
const p4 = pat({
  '0-14': V,
  '1-12': 1,
  '2-14': V,
  '3-12': 1,  '3-8': 1,  '3-7': V,
  '4-12': 1,  '4-8': 1,
  '5-10': V,  '5-8': 1,  '5-7': V,
  '6-10': V,  '6-7': V,
  '7-10': V,  '7-7': V,  '7-5': V,
});

// Pattern 5 — 21 cells
const p5 = pat({
  '0-14': V,
  '1-12': 1,
  '2-14': V,  '2-12': 1,
  '3-12': 1,  '3-8': 1,  '3-7': V,
  '4-12': 1,  '4-9': V,  '4-8': 1,
  '5-10': V,  '5-9': V,  '5-8': 1,  '5-7': V,
  '6-10': V,  '6-7': V,  '6-5': V,
  '7-10': V,  '7-7': V,  '7-5': V,  '7-6': V,
});

// Pattern 6 — 25 cells, full amxd equivalent
const p6 = pat({
  '0-14': V,
  '1-12': 1,
  '2-14': V,  '2-12': 1,
  '3-12': 1,  '3-8': 1,  '3-7': V,  '3-5': V,
  '4-12': 1,  '4-9': V,  '4-8': 1,  '4-7': V,
  '5-10': V,  '5-9': V,  '5-8': 1,  '5-7': V,  '5-5': V,
  '6-10': V,  '6-7': V,  '6-5': V,
  '7-10': V,  '7-6': V,  '7-7': V,  '7-5': V,  '7-4': 1,
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
