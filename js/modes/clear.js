// clear mode — Crystalline analog synthesis (Wendy Carlos)
// Moog ladder filters, precise clean synthesis, microtonal options

export default {
  id: 'clear',
  label: 'clear',

  mainGrids: [
    { rows: 16, cols: 32, defaultInstrument: 'moog_bass' },
    { rows: 16, cols: 32, defaultInstrument: 'moog_lead' },
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
    { id: 'moog_lead', label: 'moog lead' },
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
    moog_bass: { delay: 0.08, reverb: 0.12, gain: 0.30 },
    moog_lead: { delay: 0.15, reverb: 0.20, gain: 0.24 },
    string_machine: { delay: 0.20, reverb: 0.35, gain: 0.18 },
    sync: { delay: 0.12, reverb: 0.18, gain: 0.22 },
    bell: { delay: 0.30, reverb: 0.45, gain: 0.16 },
    wurli: { delay: 0.18, reverb: 0.25, gain: 0.20 },
    clav: { delay: 0.10, reverb: 0.15, gain: 0.24 },
    celesta: { delay: 0.25, reverb: 0.40, gain: 0.18 },
  },

  effects: {
    reverbWet: 0.15,
    reverbDark: 0.4,
    reverbLength: 2.5,
    delayL: 0.38,
    delayR: 0.25,
    delayFeedback: 0.25,
    delayDarkLP: 4000,
    delayWet: 0.12,
    compThreshold: -18,
    compRatio: 3,
  },

  mainFreqs: [262, 220, 196, 165, 131, 110, 98, 82, 73, 65, 55, 49, 41, 33, 27, 21],
  melodyFreqs: [523, 440, 392, 330, 262, 220, 196, 165],
  numPatterns: 8,
  defaultMelN: 2,
  cellSize: 16,
};
