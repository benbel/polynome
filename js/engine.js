// engine.js — Core sequencer, effects chain, playback. Mode-agnostic.

export const STYLES = {
  OFF:  { bg: 'transparent', border: '#ebebeb', shadow: 'none' },
  LIT:  { bg: 'rgba(255,228,188,.55)', border: 'rgba(248,215,170,.65)', shadow: '0 0 8px 2px rgba(255,222,175,.4)' },
  HALF: { bg: 'rgba(255,220,170,.7)', border: 'rgba(242,200,148,.8)', shadow: '0 0 10px 3px rgba(255,218,165,.5)' },
  FULL: { bg: 'rgb(255,215,160)', border: 'rgb(240,195,135)', shadow: '0 0 14px 4px rgba(255,218,165,.7), 0 0 28px 8px rgba(255,212,155,.25)' },
  TRIG: { bg: 'rgb(250,205,145)', border: 'rgb(235,185,120)', shadow: '0 0 18px 5px rgba(255,210,150,.85), 0 0 36px 10px rgba(250,205,140,.35)' },
};

export function applyStyle(el, s) {
  el.style.background = s.bg;
  el.style.borderColor = s.border;
  el.style.boxShadow = s.shadow;
}

// ======================== STATE ========================

export const state = {
  ctx: null,
  step: 0,
  timer: null,
  playing: false,
  stepMs: 200,
  grids: [],
  buffers: {},
  melCells: [],
  melActive: {},
  melInstrument: 'pad',
  melN: 2,
  delayInput: null,
  reverbInput: null,
  masterComp: null,
  patterns: [],
  currentPattern: 0,
  autoCycle: false,
  cycleSteps: 64,
  patButtons: [],
  melButtons: [],
  canonEnabled: false,
  canonMode: 'simple',   // 'simple' | 'interval' | 'crab' | 'mirror' | 'table'
  canonInterval: 4,      // rows to shift for interval canon (default: alla quarta)
  canonOffset: 8,        // columns to shift for simple/interval canon
  canonCells: [],       // DOM elements for canon grid
  canonActive: {},      // computed from melActive
  trSteps: 16,
  trRem: 0,
  oldActives: [],
  oldInsts: [],
  oldMelActive: {},
  oldMelInst: 'pad',
  balance: 50,       // 0 = all melody, 100 = all rhythm, 50 = equal
  simpleGrid: true,  // true = 8x16, false = full complex grids
  mode: null,        // current mode config
  modePatterns: {},   // per-mode pattern storage
};

// ======================== EFFECTIVE GRIDS ========================
// In simple mode, cap grids to smaller dimensions
function melGrids() {
  const mode = state.mode;
  if (!mode || !mode.melodyGrids) return null;
  if (state.simpleGrid && mode.id !== 'original') return [{ rows: 8, cols: 32 }];
  return mode.melodyGrids;
}
function mainGridsDef() {
  const mode = state.mode;
  if (!mode) return [];
  if (state.simpleGrid && mode.id !== 'original')
    return [{ rows: 8, cols: 16, defaultInstrument: mode.mainGrids[0].defaultInstrument }];
  return mode.mainGrids;
}

// ======================== CANON ========================

export function computeCanon() {
  const s = state;
  const mode = s.mode;
  if (!mode || !melGrids() || !s.canonEnabled) { s.canonActive = {}; return; }

  const totalCols = melGrids().reduce((sum, g) => sum + g.cols, 0);
  const melRows = melGrids()[0].rows;
  const result = {};

  for (const k of Object.keys(s.melActive)) {
    const [rs, cs] = k.split('-');
    const r = parseInt(rs), c = parseInt(cs);
    let nr = r, nc = c;

    if (s.canonMode === 'simple') {
      nc = (c + s.canonOffset) % totalCols;
    } else if (s.canonMode === 'interval') {
      nr = r + s.canonInterval;
      nc = (c + s.canonOffset) % totalCols;
      if (nr >= melRows) continue;
    } else if (s.canonMode === 'crab') {
      nc = totalCols - 1 - c;
    } else if (s.canonMode === 'mirror') {
      nr = melRows - 1 - r;
    } else if (s.canonMode === 'table') {
      nr = melRows - 1 - r;
      nc = totalCols - 1 - c;
    }

    result[nr + '-' + nc] = { vol: s.melActive[k].vol };
  }

  s.canonActive = result;
}

// ======================== PATTERNS ========================

export function initPatterns(mode) {
  const n = mode.numPatterns || 8;
  state.patterns = [];
  const defaults = mode.defaultPatterns || [];
  for (let i = 0; i < n; i++) {
    if (i < defaults.length) {
      state.patterns[i] = JSON.parse(JSON.stringify(defaults[i]));
    } else {
      const p = { grids: [], melody: { cells: {}, instrument: mode.melodyDefaultInstrument || 'pad' } };
      for (let gi = 0; gi < mainGridsDef().length; gi++) {
        p.grids[gi] = { cells: {}, instrument: mainGridsDef()[gi].defaultInstrument };
      }
      // Default melody: single non-repeating line filling every column.
      // No canon-like structure — canon is a separate user-enabled feature.
      if (i === 0 && melGrids() && melGrids().length > 0) {
        const totalCols = melGrids().reduce((s, g) => s + g.cols, 0);
        const rows = melGrids()[0].rows;
        const mc = {};
        // Walk through every column with gentle stepwise motion + occasional leaps.
        // Start in the middle, wander without repeating a pattern.
        let pitch = Math.floor(rows / 2);
        const steps = [-1, 0, 1, -1, 1, 0, -2, 1, 1, 0, -1, 2, 0, -1, 1, -1,
                       0, 1, -1, 0, 2, -1, 0, 1, -2, 1, 0, -1, 1, 0, -1, 1];
        for (let c = 0; c < totalCols; c++) {
          mc[pitch + '-' + c] = 1;
          const step = steps[c % steps.length];
          pitch = Math.max(0, Math.min(rows - 1, pitch + step));
        }
        p.melody.cells = mc;
      }
      state.patterns[i] = p;
    }
  }
}

export function savePat(idx) {
  const s = state;
  for (let gi = 0; gi < s.grids.length; gi++) {
    const g = s.grids[gi];
    const c = {};
    for (const k of Object.keys(g.active)) c[k] = g.active[k].vol;
    s.patterns[idx].grids[gi] = { cells: c, instrument: g.instrument };
  }
  const mc = {};
  for (const k of Object.keys(s.melActive)) mc[k] = s.melActive[k].vol;
  s.patterns[idx].melody = { cells: mc, instrument: s.melInstrument };
}

export function loadPat(idx) {
  const s = state;
  s.oldActives = [];
  s.oldInsts = [];
  for (let gi = 0; gi < s.grids.length; gi++) {
    const g = s.grids[gi];
    s.oldActives[gi] = {};
    s.oldInsts[gi] = g.instrument;
    for (const k of Object.keys(g.active)) {
      s.oldActives[gi][k] = { offset: g.active[k].offset, vol: g.active[k].vol };
    }
  }
  s.oldMelActive = {};
  s.oldMelInst = s.melInstrument;
  for (const k of Object.keys(s.melActive)) s.oldMelActive[k] = { vol: s.melActive[k].vol };
  s.trRem = s.trSteps;

  for (let gi = 0; gi < s.grids.length; gi++) {
    const g = s.grids[gi];
    const pat = s.patterns[idx].grids[gi];
    g.active = {};
    g.instrument = pat.instrument;
    for (const b of g.buttons) b.className = b.dataset.inst === g.instrument ? 'sel' : '';
    for (const k of Object.keys(pat.cells)) {
      const [, cs] = k.split('-');
      const c = parseInt(cs);
      if (mode.colSeqs) {
        // Sequence mode: offset not used; trigger is purely sequence-driven
        g.active[k] = { offset: 0, vol: pat.cells[k] };
      } else {
        const period = g.cols - c;
        g.active[k] = { offset: s.step - (g.cols - 1) - Math.floor(Math.random() * period), vol: pat.cells[k] };
      }
    }
    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const a = g.active[r + '-' + c];
        applyStyle(g.cells[r][c], a ? (a.vol >= 1 ? STYLES.FULL : STYLES.HALF) : STYLES.OFF);
      }
    }
  }

  const mp = s.patterns[idx].melody;
  s.melActive = {};
  s.melInstrument = mp.instrument;
  for (const b of s.melButtons) b.className = b.dataset.inst === s.melInstrument ? 'sel' : '';
  for (const k of Object.keys(mp.cells)) s.melActive[k] = { vol: mp.cells[k] };

  const mode = s.mode;
  if (melGrids()) {
    const totalMelCols = melGrids().reduce((s, g) => s + g.cols, 0);
    const melRows = melGrids()[0].rows;
    for (let r = 0; r < melRows; r++) {
      for (let c = 0; c < totalMelCols; c++) {
        if (s.melCells[r] && s.melCells[r][c]) {
          const a = s.melActive[r + '-' + c];
          applyStyle(s.melCells[r][c], a ? (a.vol >= 1 ? STYLES.FULL : STYLES.HALF) : STYLES.OFF);
        }
      }
    }
  }

  s.currentPattern = idx;
  for (let i = 0; i < s.patButtons.length; i++) {
    s.patButtons[i].className = i === s.currentPattern ? 'pat-btn sel' : 'pat-btn';
  }

  computeCanon();
}

export function selectPat(idx) {
  savePat(state.currentPattern);
  loadPat(idx);
}

// ======================== EFFECTS CHAIN ========================

export function buildEffects(config) {
  const ctx = state.ctx;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = config.compThreshold || -12;
  comp.knee.value = 4;
  comp.ratio.value = config.compRatio || 8;
  comp.attack.value = 0.002;
  comp.release.value = 0.12;
  comp.connect(ctx.destination);
  state.masterComp = comp;

  // Convolution reverb
  const irLen = Math.round(ctx.sampleRate * (config.reverbLength || 3.5));
  const irBuf = ctx.createBuffer(2, irLen, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = irBuf.getChannelData(ch);
    for (let i = 0; i < irLen; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / irLen, 2.5);
    const dark = config.reverbDark || 0.8;
    for (let i = 1; i < irLen; i++) d[i] = d[i] * (1 - dark) + d[i - 1] * dark;
  }
  const conv = ctx.createConvolver();
  conv.buffer = irBuf;
  const rWet = ctx.createGain();
  rWet.gain.value = config.reverbWet || 0.22;
  state.reverbInput = ctx.createGain();
  state.reverbInput.connect(conv);
  conv.connect(rWet).connect(comp);

  // Ping-pong delay
  const dL = ctx.createDelay(2);
  dL.delayTime.value = config.delayL || 0.45;
  const dR = ctx.createDelay(2);
  dR.delayTime.value = config.delayR || 0.30;
  const fbL = ctx.createGain();
  fbL.gain.value = config.delayFeedback || 0.4;
  const fbR = ctx.createGain();
  fbR.gain.value = config.delayFeedback || 0.4;
  const lpFreq = config.delayDarkLP || 1200;
  const fL1 = ctx.createBiquadFilter();
  fL1.type = 'lowpass'; fL1.frequency.value = lpFreq;
  const fL2 = ctx.createBiquadFilter();
  fL2.type = 'lowpass'; fL2.frequency.value = lpFreq;
  const pL = ctx.createStereoPanner();
  pL.pan.value = -0.75;
  const pR = ctx.createStereoPanner();
  pR.pan.value = 0.75;
  const dWet = ctx.createGain();
  dWet.gain.value = config.delayWet || 0.24;

  state.delayInput = ctx.createGain();
  state.delayInput.connect(dL);
  dL.connect(fbL).connect(fL1).connect(dR);
  dR.connect(fbR).connect(fL2).connect(dL);
  dL.connect(pL).connect(dWet);
  dR.connect(pR).connect(dWet);
  dWet.connect(comp);
  dWet.connect(state.reverbInput);
}

// ======================== PLAYBACK ========================

export function playNote(inst, fi, vol, time) {
  const buf = state.buffers[inst];
  if (!buf || !buf[fi]) return;
  const fx = state.mode.fx[inst];
  if (!fx) return;
  const src = state.ctx.createBufferSource();
  src.buffer = buf[fi];
  const v = fx.gain * vol;

  const dry = state.ctx.createGain();
  dry.gain.value = v;
  src.connect(dry).connect(state.masterComp);

  const ds = state.ctx.createGain();
  ds.gain.value = fx.delay * v;
  src.connect(ds).connect(state.delayInput);

  const rs = state.ctx.createGain();
  rs.gain.value = fx.reverb * v;
  src.connect(rs).connect(state.reverbInput);

  src.start(time);
}

// ======================== SEQUENCER ========================

function mod(a, n) { return ((a % n) + n) % n; }

// Sequence-based trigger check for original mode (Press Cafe).
// Returns true if the cell at column `col` should fire on step `step`.
function seqTrigger(mode, col, step) {
  const seqs = mode.colSeqs;
  if (!seqs) return false;
  const len = mode.seqLen || seqs[0].length;
  return seqs[col][mod(step, len)] === 1;
}

export function tick() {
  const s = state;
  const mode = s.mode;
  if (!mode) return;

  const hasMelody = melGrids() && melGrids().length > 0;
  const totalMelCols = hasMelody ? melGrids().reduce((sum, g) => sum + g.cols, 0) : 0;
  const melRows = hasMelody ? melGrids()[0].rows : 0;

  // Auto-cycle
  if (s.autoCycle && s.step > 0 && s.step % s.cycleSteps === 0) {
    savePat(s.currentPattern);
    let next = (s.currentPattern + 1) % s.patterns.length;
    let ch = 0;
    while (ch < s.patterns.length) {
      let any = false;
      for (let gi = 0; gi < s.grids.length; gi++) {
        if (Object.keys(s.patterns[next].grids[gi].cells).length > 0) any = true;
      }
      if (Object.keys(s.patterns[next].melody.cells).length > 0) any = true;
      if (any) break;
      next = (next + 1) % s.patterns.length;
      ch++;
    }
    if (ch < s.patterns.length) loadPat(next);
  }

  const t = s.ctx.currentTime + 0.02;

  // Balance: 0 = all melody, 100 = all rhythm, 50 = equal
  const rhythmBal = Math.min(1, s.balance / 50);
  const melBal = Math.min(1, (100 - s.balance) / 50);

  // Crossfade
  let newV = 1, oldV = 0;
  if (s.trRem > 0) {
    const pr = 1 - (s.trRem / s.trSteps);
    newV = pr * pr;
    oldV = (1 - pr) * (1 - pr);
    s.trRem--;
    if (oldV > 0.01) {
      for (let gi = 0; gi < s.grids.length; gi++) {
        const g = s.grids[gi];
        const oA = s.oldActives[gi];
        if (!oA) continue;
        const oI = s.oldInsts[gi];
        for (const k of Object.keys(oA)) {
          const [rs, cs] = k.split('-');
          const c = parseInt(cs);
          const a = oA[k];
          const fires = mode.colSeqs
            ? seqTrigger(mode, c, s.step)
            : mod(s.step - a.offset - (g.cols - 1), g.cols - c) === 0;
          if (fires) {
            const vol = mode.stepVels ? a.vol * mode.stepVels[mod(s.step, mode.seqLen)] : a.vol;
            playNote(oI, parseInt(rs), vol * oldV * rhythmBal, t);
          }
        }
      }
      if (hasMelody) {
        const melStep = Math.floor(s.step / s.melN);
        const mc = mod(melStep, totalMelCols);
        for (const k of Object.keys(s.oldMelActive)) {
          const [, cs] = k.split('-');
          if (parseInt(cs) === mc) {
            playNote(s.oldMelInst, parseInt(k.split('-')[0]), s.oldMelActive[k].vol * oldV * melBal, t);
          }
        }
      }
    }
  }

  // Main grids
  for (let gi = 0; gi < s.grids.length; gi++) {
    const g = s.grids[gi];
    const lit = [];
    const trig = [];
    for (let r = 0; r < g.rows; r++) { lit[r] = {}; trig[r] = false; }

    const useSeq = !!mode.colSeqs;
    const seqIdx = useSeq ? mod(s.step, mode.seqLen) : -1;

    for (const k of Object.keys(g.active)) {
      const [rs, cs] = k.split('-');
      const r = parseInt(rs), c = parseInt(cs);
      const a = g.active[k];
      if (useSeq) {
        if (mode.colSeqs[c][seqIdx]) {
          const vol = mode.stepVels ? a.vol * mode.stepVels[seqIdx] : a.vol;
          playNote(g.instrument, r, vol * newV * rhythmBal, t);
          trig[r] = true;
        }
        // Lit: show upcoming triggers in the sequence
        for (let j = 0; j < g.cols; j++) {
          if (mode.colSeqs[j] && mode.colSeqs[j][seqIdx] && g.active[r + '-' + j]) lit[r][j] = true;
        }
      } else {
        const period = g.cols - c;
        if (mod(s.step - a.offset - (g.cols - 1), period) === 0) {
          playNote(g.instrument, r, a.vol * newV * rhythmBal, t);
          trig[r] = true;
        }
        for (let j = 0; j < g.cols; j++) {
          if (mod(s.step - a.offset - j, period) === 0) lit[r][j] = true;
        }
      }
    }

    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const a = g.active[r + '-' + c];
        const isL = !!lit[r][c];
        const isT = trig[r] && isL;
        if (isT) applyStyle(g.cells[r][c], STYLES.TRIG);
        else if (a && a.vol >= 1) applyStyle(g.cells[r][c], STYLES.FULL);
        else if (a) applyStyle(g.cells[r][c], STYLES.HALF);
        else if (isL) applyStyle(g.cells[r][c], STYLES.LIT);
        else applyStyle(g.cells[r][c], STYLES.OFF);
      }
    }
  }

  // Melody grids
  if (hasMelody) {
    const melStep = Math.floor(s.step / s.melN);
    const mc = mod(melStep, totalMelCols);
    for (let r = 0; r < melRows; r++) {
      for (let c = 0; c < totalMelCols; c++) {
        if (!s.melCells[r] || !s.melCells[r][c]) continue;
        const a = s.melActive[r + '-' + c];
        const isCursor = (c === mc);
        if (a && isCursor) {
          applyStyle(s.melCells[r][c], STYLES.TRIG);
          if (s.step % s.melN === 0) playNote(s.melInstrument, r, a.vol * newV * melBal, t);
        } else if (a && a.vol >= 1) applyStyle(s.melCells[r][c], STYLES.FULL);
        else if (a) applyStyle(s.melCells[r][c], STYLES.HALF);
        else if (isCursor) applyStyle(s.melCells[r][c], STYLES.LIT);
        else applyStyle(s.melCells[r][c], STYLES.OFF);
      }
    }

    // Canon grid
    if (s.canonEnabled) {
      for (let r = 0; r < melRows; r++) {
        for (let c = 0; c < totalMelCols; c++) {
          if (!s.canonCells[r] || !s.canonCells[r][c]) continue;
          const a = s.canonActive[r + '-' + c];
          const isCursor = (c === mc);
          if (a && isCursor) {
            applyStyle(s.canonCells[r][c], STYLES.TRIG);
            if (s.step % s.melN === 0) playNote(s.melInstrument, r, a.vol * newV * melBal, t);
          } else if (a && a.vol >= 1) applyStyle(s.canonCells[r][c], STYLES.FULL);
          else if (a) applyStyle(s.canonCells[r][c], STYLES.HALF);
          else if (isCursor) applyStyle(s.canonCells[r][c], STYLES.LIT);
          else applyStyle(s.canonCells[r][c], STYLES.OFF);
        }
      }
    }
  }

  s.step++;
}

export function start() {
  if (!state.ctx) return;
  if (state.ctx.state === 'suspended') state.ctx.resume();
  state.playing = true;
  tick();
  state.timer = setInterval(tick, state.stepMs);
}

export function stop() {
  const s = state;
  s.playing = false;
  if (s.timer) clearInterval(s.timer);
  s.timer = null;
  s.step = 0;

  for (const g of s.grids) {
    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const a = g.active[r + '-' + c];
        applyStyle(g.cells[r][c], a ? (a.vol >= 1 ? STYLES.FULL : STYLES.HALF) : STYLES.OFF);
      }
    }
  }

  const mode = s.mode;
  if (mode && melGrids()) {
    const totalMelCols = melGrids().reduce((sum, g) => sum + g.cols, 0);
    const melRows = melGrids()[0].rows;
    for (let r = 0; r < melRows; r++) {
      for (let c = 0; c < totalMelCols; c++) {
        if (s.melCells[r] && s.melCells[r][c]) {
          const a = s.melActive[r + '-' + c];
          applyStyle(s.melCells[r][c], a ? (a.vol >= 1 ? STYLES.FULL : STYLES.HALF) : STYLES.OFF);
        }
      }
    }
  }
}

export function restartTimer() {
  if (state.timer) clearInterval(state.timer);
  if (state.playing) state.timer = setInterval(tick, state.stepMs);
}

// ======================== RESET / RANDOMIZE ========================

export function resetAll() {
  if (state.playing) stop();
  state.step = 0;
  for (const g of state.grids) {
    g.active = {};
    for (let r = 0; r < g.rows; r++)
      for (let c = 0; c < g.cols; c++)
        applyStyle(g.cells[r][c], STYLES.OFF);
  }
  state.melActive = {};
  const mode = state.mode;
  if (mode && melGrids()) {
    const totalMelCols = melGrids().reduce((s, g) => s + g.cols, 0);
    const melRows = melGrids()[0].rows;
    for (let r = 0; r < melRows; r++)
      for (let c = 0; c < totalMelCols; c++)
        if (state.melCells[r] && state.melCells[r][c])
          applyStyle(state.melCells[r][c], STYLES.OFF);
  }
  initPatterns(mode);
  state.currentPattern = 0;
  for (let i = 0; i < state.patButtons.length; i++)
    state.patButtons[i].className = i === 0 ? 'pat-btn sel' : 'pat-btn';
  computeCanon();
}

// ======================== SAVE / LOAD ========================

export function saveSong() {
  savePat(state.currentPattern);
  const song = {
    mode: state.mode.id,
    patterns: state.patterns,
    currentPattern: state.currentPattern,
    bpm: Math.round(60000 / state.stepMs),
    cycleSteps: state.cycleSteps,
    autoCycle: state.autoCycle,
    melN: state.melN,
  };
  const blob = new Blob([JSON.stringify(song)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `polynome-${state.mode.id}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function loadSong(onLoaded) {
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = '.json';
  inp.addEventListener('change', () => {
    if (!inp.files[0]) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const s = JSON.parse(reader.result);
        if (!s.patterns) return;
        if (s.mode && s.mode !== state.mode.id) {
          console.warn(`Song is for mode "${s.mode}", current mode is "${state.mode.id}"`);
          return;
        }
        if (state.playing) stop();
        state.step = 0;
        const n = state.mode.numPatterns || 8;
        for (let pi = 0; pi < n; pi++) {
          if (s.patterns[pi]) state.patterns[pi] = s.patterns[pi];
        }
        state.trRem = 0;
        state.currentPattern = s.currentPattern || 0;
        loadPat(state.currentPattern);
        if (onLoaded) onLoaded(s);
      } catch (e) { console.error(e); }
    };
    reader.readAsText(inp.files[0]);
  });
  inp.click();
}

// ======================== RANDOM HELPERS ========================

export function pick(a) { return a[Math.floor(Math.random() * a.length)]; }
export function shuffle(a) {
  const r = a.slice();
  for (let i = r.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [r[i], r[j]] = [r[j], r[i]];
  }
  return r;
}
export function randInt(lo, hi) { return lo + Math.floor(Math.random() * (hi - lo + 1)); }

export function randomize() {
  if (state.playing) stop();
  state.step = 0;
  const mode = state.mode;

  // maxCol: exclude rightmost column (period=1, fires every step)
  const maxCol = (cols) => cols - 2;

  // Coprime period pools — pairs that create long composite cycles
  const coprimeSets = [
    [3, 4], [3, 5], [3, 7], [4, 5], [4, 7], [5, 7], [5, 8],
    [3, 4, 7], [3, 5, 8], [4, 7, 11], [5, 7, 13], [3, 8, 11],
  ];

  // ---- Rhythm generation ----
  for (let gi = 0; gi < state.grids.length; gi++) {
    const g = state.grids[gi];
    const rows = g.rows, cols = g.cols;
    const cells = {};

    if (mode.colSeqs) {
      // Sequence-based (original mode): pick 3-5 cells from interesting columns
      const allRows = shuffle([...Array(rows).keys()]);
      const interestingCols = [];
      for (let c = 0; c < cols; c++) {
        if (mode.colSeqs[c] && mode.colSeqs[c].some(v => v === 0)) interestingCols.push(c);
      }
      const useCols = shuffle(interestingCols.length > 0 ? interestingCols : [...Array(cols).keys()]);
      const n = randInt(3, 5);
      for (let i = 0; i < n && i < rows && i < useCols.length; i++) {
        cells[allRows[i] + '-' + useCols[i]] = i === 0 ? 1 : 0.5;
      }
    } else {
      const mc = maxCol(cols);

      // Pick coprime periods for this grid
      const periods = pick(coprimeSets).filter(p => p <= mc);
      if (periods.length === 0) continue;

      const allRows = shuffle([...Array(rows).keys()]);
      let ri = 0;

      // --- Anchor rows: one per period, with pattern variety ---
      const anchorInfo = []; // store for response rows
      for (const period of periods) {
        if (ri >= rows) break;
        const row = allRows[ri++];
        const col = mc - period; // column that gives this period
        const patType = pick(['pulse', 'skip', 'cluster']);

        if (patType === 'pulse') {
          // Single cell — clean periodic pulse
          cells[row + '-' + col] = 1;
        } else if (patType === 'skip') {
          // Two cells with different periods — polyrhythmic
          cells[row + '-' + col] = 1;
          const col2 = Math.max(0, Math.min(mc, col - randInt(1, 3)));
          if (col2 !== col) cells[row + '-' + col2] = 0.5;
        } else {
          // Cluster: 2-3 adjacent cells creating burst patterns
          cells[row + '-' + col] = 1;
          if (col > 0) cells[row + '-' + (col - 1)] = 0.5;
          if (col > 1 && Math.random() > 0.5) cells[row + '-' + (col - 2)] = 0.5;
        }
        anchorInfo.push({ row, period, col });
      }

      // --- Response rows: play on OFF-beats of an anchor ---
      const nResponse = randInt(1, 2);
      for (let rr = 0; rr < nResponse && ri < rows; rr++) {
        const row = allRows[ri++];
        const anchor = pick(anchorInfo);
        // Offset by half the anchor's period
        const offset = Math.floor(anchor.period / 2);
        const responseCol = Math.max(0, Math.min(mc, anchor.col + offset));
        if (responseCol !== anchor.col) {
          cells[row + '-' + responseCol] = 0.5;
        }
      }

      // --- Accent row: single strong hit at phrase boundary ---
      if (ri < rows && Math.random() > 0.3) {
        const row = allRows[ri++];
        // Phrase-start accent at column 0 (longest period) or mid-point
        const accentCol = pick([0, Math.floor(mc / 2)]);
        cells[row + '-' + accentCol] = 1;
      }
    }

    state.patterns[0].grids[gi] = { cells, instrument: pick(mode.mainInstruments).id };
  }

  // ---- Melody generation: motif-based ----
  if (melGrids()) {
    const totalCols = melGrids().reduce((s, g) => s + g.cols, 0);
    const melRows = melGrids()[0].rows;
    const mc = {};

    // Generate a 3-5 note motif (interval sequence)
    const motifLen = randInt(3, 5);
    const motif = [];
    const intervalPool = [-3, -2, -1, 1, 2, 3];
    for (let i = 0; i < motifLen; i++) {
      motif.push(pick(intervalPool));
    }

    // Place motif + variations across the grid
    const clamp = (v) => Math.max(0, Math.min(melRows - 1, v));
    let startPitch = randInt(1, Math.max(1, melRows - 3));
    const spacing = randInt(2, 4); // columns between notes

    // Motif at phrase start
    let col = 0;
    let pitch = startPitch;
    mc[pitch + '-' + col] = 1;
    for (let i = 0; i < motif.length && col + spacing < totalCols; i++) {
      col += spacing;
      pitch = clamp(pitch + motif[i]);
      mc[pitch + '-' + col] = (i === 0) ? 1 : 0.5;
    }

    // Gap (breathing space)
    col += spacing * randInt(2, 4);

    // Repeat motif transposed (up or down 1-2 rows)
    if (col < totalCols - motifLen * spacing) {
      const transpose = pick([-2, -1, 1, 2]);
      pitch = clamp(startPitch + transpose);
      mc[pitch + '-' + col] = 1;
      for (let i = 0; i < motif.length && col + spacing < totalCols; i++) {
        col += spacing;
        pitch = clamp(pitch + motif[i]);
        mc[pitch + '-' + col] = (i === 0) ? 1 : 0.5;
      }
      col += spacing * randInt(2, 3);
    }

    // Variation: retrograde or truncated motif
    if (col < totalCols - 3 * spacing) {
      const varType = pick(['retrograde', 'truncated', 'augmented']);
      let varMotif;
      if (varType === 'retrograde') {
        varMotif = motif.slice().reverse().map(x => -x);
      } else if (varType === 'truncated') {
        varMotif = motif.slice(0, Math.max(2, motif.length - 1));
      } else {
        // Augmented: double the intervals
        varMotif = motif.map(x => x * 2);
      }
      const varTranspose = pick([-1, 0, 1]);
      pitch = clamp(startPitch + varTranspose);
      mc[pitch + '-' + col] = 1;
      for (let i = 0; i < varMotif.length && col + spacing < totalCols; i++) {
        col += spacing;
        pitch = clamp(pitch + varMotif[i]);
        mc[pitch + '-' + col] = 0.5;
      }
    }

    const melInst = (mode.melodyInstruments || []).length > 0
      ? pick(mode.melodyInstruments).id
      : mode.melodyDefaultInstrument;
    state.patterns[0].melody = { cells: mc, instrument: melInst };
  }

  // ---- Pattern evolution: narrative arc (sparse → dense) ----
  const nPat = state.patterns.length;
  for (let pi = 1; pi < nPat; pi++) {
    const progress = pi / (nPat - 1); // 0→1

    for (let gi = 0; gi < state.grids.length; gi++) {
      const base = state.patterns[0].grids[gi];
      const g = state.grids[gi];
      const mc = maxCol(g.cols);
      const cells = { ...base.cells };

      // Early patterns: add specific rows (intentional additions)
      // Late patterns: densify existing rows + add ghost notes
      const nAdd = Math.floor(1 + progress * 4);

      for (let a = 0; a < nAdd; a++) {
        if (progress < 0.5) {
          // Early: add a new cell in a new row (expand coverage)
          const r = randInt(0, g.rows - 1);
          const period = pick([3, 4, 5, 7, 8, 11]);
          const c = Math.max(0, Math.min(mc, mc - period));
          cells[r + '-' + c] = Math.random() > 0.3 ? 1 : 0.5;
        } else {
          // Late: add cells near existing ones (densify)
          const keys = Object.keys(cells);
          if (keys.length > 0) {
            const k = pick(keys);
            const [rs, cs] = k.split('-');
            const r = parseInt(rs);
            const c = parseInt(cs);
            // Adjacent cell (±1 row or col)
            const nr = Math.max(0, Math.min(g.rows - 1, r + pick([-1, 0, 1])));
            const nc = Math.max(0, Math.min(mc, c + pick([-2, -1, 1, 2])));
            cells[nr + '-' + nc] = 0.5;
          }
        }
      }

      // Occasional removal for variety (more likely in middle patterns)
      if (progress > 0.2 && progress < 0.7 && Math.random() > 0.5) {
        const keys = Object.keys(cells);
        if (keys.length > 2) delete cells[pick(keys)];
      }

      state.patterns[pi].grids[gi] = { cells, instrument: base.instrument };
    }

    // Melody evolution: shift notes, add ornaments
    if (melGrids()) {
      const baseMel = state.patterns[Math.max(0, pi - 1)].melody;
      const melCells = { ...baseMel.cells };
      const totalCols = melGrids().reduce((s, g) => s + g.cols, 0);
      const melRows = melGrids()[0].rows;

      const nMut = Math.floor(1 + progress * 3);
      for (let m = 0; m < nMut; m++) {
        const keys = Object.keys(melCells);
        if (keys.length === 0) break;

        const action = Math.random();
        if (action < 0.4) {
          // Shift a note by 1-2 rows (pitch variation)
          const k = pick(keys);
          const [rs, cs] = k.split('-');
          const nr = Math.max(0, Math.min(melRows - 1, parseInt(rs) + pick([-2, -1, 1, 2])));
          const nk = nr + '-' + cs;
          if (!melCells[nk]) {
            const vol = melCells[k];
            delete melCells[k];
            melCells[nk] = vol;
          }
        } else if (action < 0.7) {
          // Add passing note between two existing notes
          const sorted = keys.map(k => {
            const [r, c] = k.split('-');
            return { r: parseInt(r), c: parseInt(c) };
          }).sort((a, b) => a.c - b.c);
          if (sorted.length >= 2) {
            const idx = randInt(0, sorted.length - 2);
            const a = sorted[idx], b = sorted[idx + 1];
            const midC = Math.floor((a.c + b.c) / 2);
            const midR = Math.floor((a.r + b.r) / 2);
            if (midC !== a.c && midC !== b.c) {
              melCells[midR + '-' + midC] = 0.5;
            }
          }
        } else {
          // Remove a note (creates space)
          if (keys.length > 3) delete melCells[pick(keys)];
        }
      }

      state.patterns[pi].melody = { cells: melCells, instrument: baseMel.instrument };
    }
  }

  state.currentPattern = 0;
  loadPat(0);
  start();
}
