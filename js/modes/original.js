// original mode — Exact replication of monome grid video
// Single grid, no melody, minimal controls. Sound from pre-rendered assets only.

// Default pattern: cascading diagonal — notes stagger by row,
// creating the interlocking polyrhythmic waterfall from the monome grid video.
// Each row has a note at a different column offset, producing phase-shifted
// repeating patterns that drift in and out of alignment.
const defaultPattern = {
  grids: [{
    cells: {
      // Diagonal cascade — each row offset by 2 columns
      '0-0': 1,
      '1-2': 1,
      '2-4': 1,
      '3-6': 1,
      '4-8': 1,
      '5-10': 1,
      '6-12': 1,
      '7-14': 1,
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
  numPatterns: 8,
  cellSize: 28,
  hideControls: ['melody', 'cycle'],
  defaultPatterns: [defaultPattern],
};
