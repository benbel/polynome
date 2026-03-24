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
    throat: { delay: 0.10, reverb: 0.20, gain: 0.28 },
    overtone: { delay: 0.15, reverb: 0.30, gain: 0.22 },
    breath: { delay: 0.20, reverb: 0.40, gain: 0.18 },
    belt: { delay: 0.08, reverb: 0.15, gain: 0.30 },
    hum: { delay: 0.12, reverb: 0.25, gain: 0.26 },
    aah: { delay: 0.08, reverb: 0.15, gain: 0.24 },
    ooh: { delay: 0.08, reverb: 0.15, gain: 0.22 },
    mmm: { delay: 0.06, reverb: 0.12, gain: 0.24 },
  },

  effects: {
    reverbWet: 0.35,
    reverbDark: 0.65,
    reverbLength: 3.5,
    delayL: 0.40,
    delayR: 0.28,
    delayFeedback: 0.30,
    delayDarkLP: 2500,
    delayWet: 0.18,
    compThreshold: -12,
    compRatio: 4,
  },

  mainFreqs: [262, 220, 196, 165, 131, 110, 98, 82, 73, 65, 55, 49, 41, 33, 27, 21],
  melodyFreqs: [523, 440, 392, 330, 262, 220, 196, 165],
  numPatterns: 8,
  defaultMelN: 2,
  cellSize: 16,
};
