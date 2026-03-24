// speech mode — Music emerging from talk (Laurie Anderson, Holly Herndon)
// Each instrument IS a phoneme class with consistent identity.
// Drone provides harmonic foundation; articulations layer on top.

export default {
  id: 'speech',
  label: 'speech',

  mainGrids: [
    { rows: 8, cols: 16, defaultInstrument: 'clicks' },
    { rows: 8, cols: 16, defaultInstrument: 'vowels' },
    { rows: 8, cols: 16, defaultInstrument: 'hiss' },
    { rows: 8, cols: 16, defaultInstrument: 'drone' },
  ],
  mainGridLayout: { cols: 2 },

  melodyGrids: [
    { rows: 8, cols: 32 },
  ],
  melodyDefaultInstrument: 'syllables',

  mainInstruments: [
    { id: 'vowels', label: 'vowels' },
    { id: 'clicks', label: 'clicks' },
    { id: 'hiss', label: 'hiss' },
    { id: 'nasal', label: 'nasal' },
    { id: 'drone', label: 'drone' },
  ],
  melodyInstruments: [
    { id: 'syllables', label: 'syllables' },
  ],

  fx: {
    vowels: { delay: 0.15, reverb: 0.25, gain: 0.24 },
    clicks: { delay: 0.06, reverb: 0.08, gain: 0.30 },
    hiss: { delay: 0.18, reverb: 0.25, gain: 0.20 },
    nasal: { delay: 0.12, reverb: 0.20, gain: 0.26 },
    drone: { delay: 0.10, reverb: 0.15, gain: 0.22 },
    syllables: { delay: 0.15, reverb: 0.25, gain: 0.22 },
  },

  effects: {
    reverbWet: 0.18,
    reverbDark: 0.5,
    reverbLength: 2.0,
    delayL: 0.35,
    delayR: 0.23,
    delayFeedback: 0.30,
    delayDarkLP: 2500,
    delayWet: 0.18,
    compThreshold: -14,
    compRatio: 5,
  },

  mainFreqs: [523, 440, 349, 294, 220, 175, 131, 110],
  melodyFreqs: [659, 523, 440, 349, 262, 220, 175, 131],
  numPatterns: 8,
  defaultMelN: 2,
  cellSize: 20,
};
