// original mode — Exact replication of monome grid video
// Single grid, no melody, minimal controls. Sound from pre-rendered assets only.

// Pre-loaded pattern inspired by the end of the monome grid video —
// cascading interlocking phrases across the pitch rows
const defaultPattern = {
  grids: [{
    cells: {
      '0-0': 1, '0-4': 1, '0-7': 1, '0-10': 0.5, '0-13': 1,
      '1-2': 1, '1-5': 0.5, '1-9': 1, '1-14': 1,
      '2-1': 1, '2-6': 1, '2-8': 0.5, '2-11': 1, '2-15': 1,
      '3-3': 1, '3-7': 0.5, '3-12': 1,
      '4-0': 0.5, '4-5': 1, '4-9': 1, '4-13': 0.5,
      '5-2': 1, '5-6': 0.5, '5-10': 1, '5-14': 1,
      '6-1': 0.5, '6-4': 1, '6-8': 1, '6-11': 0.5, '6-15': 1,
      '7-3': 1, '7-7': 1, '7-12': 0.5,
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
    tone: { delay: 0.15, reverb: 0.25, gain: 0.3 },
  },

  effects: {
    reverbWet: 0.20,
    reverbDark: 0.5,
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
  numPatterns: 8,
  cellSize: 28,
  hideControls: ['melody', 'cycle'],
  defaultPatterns: [defaultPattern],
};
