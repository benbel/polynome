// clear mode — Early electronic / Kraftwerk-to-Warp
// Precise, crystalline, mechanical. Sharp envelopes, clean synthesis.
// One warm element (strings) against a cold backdrop.

export default {
  id: 'clear',
  label: 'clear',

  mainGrids: [
    { rows: 16, cols: 32, defaultInstrument: 'moog_bass' },
    { rows: 16, cols: 32, defaultInstrument: 'pluck' },
    { rows: 16, cols: 32, defaultInstrument: 'string_machine' },
    { rows: 16, cols: 32, defaultInstrument: 'bell' },
  ],
  mainGridLayout: { cols: 2 },

  melodyGrids: [
    { rows: 8, cols: 32 },
    { rows: 8, cols: 32 },
  ],
  melodyDefaultInstrument: 'wurli',

  mainInstruments: [
    { id: 'moog_bass', label: 'moog bass' },
    { id: 'pluck', label: 'pluck' },
    { id: 'string_machine', label: 'strings' },
    { id: 'sync', label: 'sync' },
    { id: 'bell', label: 'bell' },
  ],
  melodyInstruments: [
    { id: 'wurli', label: 'wurli' },
    { id: 'clav', label: 'clav' },
    { id: 'celesta', label: 'celesta' },
  ],

  fx: {
    moog_bass: { delay: 0.06, reverb: 0.06, gain: 0.28 },
    pluck: { delay: 0.10, reverb: 0.10, gain: 0.26 },
    string_machine: { delay: 0.12, reverb: 0.15, gain: 0.18 },
    sync: { delay: 0.08, reverb: 0.10, gain: 0.22 },
    bell: { delay: 0.15, reverb: 0.18, gain: 0.16 },
    wurli: { delay: 0.10, reverb: 0.12, gain: 0.20 },
    clav: { delay: 0.06, reverb: 0.08, gain: 0.24 },
    celesta: { delay: 0.12, reverb: 0.15, gain: 0.18 },
  },

  effects: {
    reverbWet: 0.08,
    reverbDark: 0.3,
    reverbLength: 1.8,
    delayL: 0.38,
    delayR: 0.25,
    delayFeedback: 0.18,
    delayDarkLP: 5000,
    delayWet: 0.08,
    compThreshold: -20,
    compRatio: 2.5,
  },

  mainFreqs: [523, 440, 370, 311, 262, 220, 175, 147, 123, 104, 82, 65, 55, 44, 33, 25],
  melodyFreqs: [659, 523, 440, 349, 262, 220, 175, 131],
  numPatterns: 8,
  defaultMelN: 2,
  cellSize: 16,
};
