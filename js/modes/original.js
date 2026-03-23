// original mode — Exact replication of monome grid video
// Single grid, no melody, minimal controls. Sound from pre-rendered assets only.

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
  numPatterns: 1,
  cellSize: 28,
  hideControls: ['melody', 'cycle', 'pattern'],
};
