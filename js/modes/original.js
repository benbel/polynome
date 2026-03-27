// original mode — Replication of stretta's monome Press Cafe video.
// Each column holds a predefined 16-step rhythmic sequence (from the
// Gridlab Connect Press Cafe amxd).  Pressing a cell activates that
// column's sequence for the corresponding row's pitch.

// Column sequences extracted from the amxd autopattr restore data.
// Each array is 16 steps long; 1 = trigger, 0 = silent.
// Velocities per step: 127 36 64 36 127 36 64 36 127 36 72 36 127 36 90 36
export const COL_SEQS = [
  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  // col  0: alternating
  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  // col  1: quarter
  [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  // col  2: syncopated
  [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  // col  3
  [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  // col  4
  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  // col  5
  [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],  // col  6
  [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  // col  7
  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  // col  8: sparse dotted
  [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  // col  9: triplet-ish
  [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  // col 10: syncopated
  [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  // col 11
  [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  // col 12: offbeat
  [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  // col 13: irregular
  [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  // col 14: sparse offset
  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  // col 15: very sparse
];

// Step velocities from the amxd (normalized to 0–1)
export const STEP_VELS = [
  127, 64, 90, 64, 127, 64, 90, 64,
  127, 64, 90, 64, 127, 64, 90, 64,
].map(v => v / 127);

const SEQ_LEN = 16;

const V = 0.5;

function pat(cells) {
  return { grids: [{ cells, instrument: 'tone' }], melody: { cells: {}, instrument: 'tone' } };
}

// Default patterns — 26 patterns building up layered polyrhythms
const p1  = pat({ '0-1': 1 });
const p2  = pat({ '0-1': 1, '1-1': 1 });
const p3  = pat({ '0-1': 1 });
const p4  = pat({ '0-1': 1, '1-2': 1 });
const p5  = pat({ '0-1': 1 });
const p6  = pat({ '0-1': 1, '1-3': 1 });
const p7  = pat({ '0-1': 1, '2-3': 1 });
const p8  = pat({ '0-1': 1, '3-3': 1 });
const p9  = pat({ '0-1': 1, '4-3': 1 });
const p10 = pat({ '0-1': 1, '4-3': 1, '2-5': 1 });
const p11 = pat({ '0-1': 1, '4-3': 1, '1-5': 1 });
const p12 = pat({ '0-1': 1, '4-3': 1, '3-5': 1 });
const p13 = pat({ '0-1': 1, '4-3': 1, '2-5': 1 });
const p14 = pat({ '0-1': 1, '4-3': 1, '3-5': 1 });
const p15 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '6-5': 1 });
const p16 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '7-5': 1 });
const p17 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '6-5': 1 });
const p18 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '7-5': 1 });
const p19 = pat({ '0-1': 1, '4-3': 1, '1-7': 1, '7-5': 1 });
const p20 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '1-7': 1, '7-5': 1 });
const p21 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '1-7': 1, '7-5': 1, '2-5': 1 });
const p22 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '1-7': 1, '7-5': 1 });
const p23 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '1-7': 1, '7-5': 1, '2-5': 1 });
const p24 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '1-7': 1, '7-5': 1 });
const p25 = pat({ '0-1': 1, '4-3': 1, '3-5': 1, '1-7': 1, '7-5': 1, '2-5': 1 });
const p26 = pat({ '0-1': 1, '1-7': 1, '2-10': 1, '4-3': 1, '7-5': 1 });

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
    tone: { delay: 0.15, reverb: 0.25, gain: 0.65 },
  },

  effects: {
    reverbWet: 0.08,
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

  // Wide range: C5 down to C3 (~2 octaves)
  mainFreqs: [587, 494, 415, 349, 294, 247, 196, 147],
  defaultStepMs: 74,
  numPatterns: 26,
  cellSize: 28,
  seqLen: SEQ_LEN,
  colSeqs: COL_SEQS,
  stepVels: STEP_VELS,
  hideControls: ['melody'],
  defaultPatterns: [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26],
};
