// original mode — Exact replication of monome grid video
// Single grid, no melody, minimal controls. Sound from pre-rendered assets only.

// Later pattern from the original monome grid video — dense polyrhythmic
// texture with multiple cells per row at various column positions, creating
// the characteristic cascading waterfall of interlocking rhythmic cycles.
const defaultPattern = {
  grids: [{
    cells: {
      // Row 0: sparse anchor hits
      '0-0': 1,  '0-6': 0.5, '0-12': 1,
      // Row 1: offset syncopation
      '1-1': 1,  '1-5': 0.5, '1-9': 1,  '1-13': 0.5,
      // Row 2: wider spacing
      '2-2': 1,  '2-8': 1,   '2-14': 0.5,
      // Row 3: dense cluster
      '3-0': 0.5, '3-3': 1,  '3-7': 1,  '3-11': 0.5,
      // Row 4: off-grid feel
      '4-1': 0.5, '4-4': 1,  '4-10': 1,
      // Row 5: dotted rhythm
      '5-2': 1,  '5-5': 0.5, '5-9': 1,  '5-13': 1,
      // Row 6: syncopated pair
      '6-3': 1,  '6-7': 0.5, '6-11': 1,
      // Row 7: bass anchor
      '7-0': 1,  '7-6': 1,   '7-10': 0.5, '7-14': 1,
    },
    instrument: 'tone',
  }],
  melody: { cells: {}, instrument: 'tone' },
};

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

  mainFreqs: [440, 392, 330, 294, 262, 220, 196, 165],
  numPatterns: 1,
  cellSize: 28,
  hideControls: ['melody', 'cycle', 'pattern'],
  defaultPatterns: [defaultPattern],
};
