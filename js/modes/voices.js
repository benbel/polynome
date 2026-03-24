// voices mode — Extended vocal technique (Roomful of Teeth, Meredith Monk)
// Glottal pulse synthesis + formant filters, all synthesized

export default {
  id: 'voices',
  label: 'voices',

  mainGrids: [
    { rows: 16, cols: 32, defaultInstrument: 'throat' },
    { rows: 16, cols: 32, defaultInstrument: 'overtone' },
    { rows: 16, cols: 32, defaultInstrument: 'breath' },
    { rows: 16, cols: 32, defaultInstrument: 'belt' },
  ],
  mainGridLayout: { cols: 2 },

  melodyGrids: [
    { rows: 8, cols: 32 },
    { rows: 8, cols: 32 },
  ],
  melodyDefaultInstrument: 'aah',

  mainInstruments: [
    { id: 'throat', label: 'throat' },
    { id: 'overtone', label: 'overtone' },
    { id: 'breath', label: 'breath' },
    { id: 'belt', label: 'belt' },
    { id: 'hum', label: 'hum' },
  ],
  melodyInstruments: [
    { id: 'aah', label: 'aah' },
    { id: 'ooh', label: 'ooh' },
    { id: 'mmm', label: 'mmm' },
  ],

  fx: {
    throat: { delay: 0.08, reverb: 0.10, gain: 0.26 },
    overtone: { delay: 0.10, reverb: 0.15, gain: 0.22 },
    breath: { delay: 0.12, reverb: 0.18, gain: 0.18 },
    belt: { delay: 0.06, reverb: 0.08, gain: 0.28 },
    hum: { delay: 0.08, reverb: 0.12, gain: 0.24 },
    aah: { delay: 0.06, reverb: 0.08, gain: 0.24 },
    ooh: { delay: 0.06, reverb: 0.08, gain: 0.22 },
    mmm: { delay: 0.05, reverb: 0.06, gain: 0.24 },
  },

  effects: {
    reverbWet: 0.15,
    reverbDark: 0.45,
    reverbLength: 2.5,
    delayL: 0.40,
    delayR: 0.28,
    delayFeedback: 0.20,
    delayDarkLP: 3000,
    delayWet: 0.12,
    compThreshold: -18,
    compRatio: 3,
  },

  mainFreqs: [523, 440, 370, 311, 262, 220, 175, 147, 123, 104, 82, 65, 55, 44, 33, 25],
  melodyFreqs: [659, 523, 440, 349, 262, 220, 175, 131],
  numPatterns: 8,
  defaultMelN: 2,
  cellSize: 16,
};
