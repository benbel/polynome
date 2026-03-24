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
  canonMode: 'crab',    // 'crab' | 'mirror' | 'table'
  canonCells: [],       // DOM elements for canon grid
  canonActive: {},      // computed from melActive
  trSteps: 16,
  trRem: 0,
  oldActives: [],
  oldInsts: [],
  oldMelActive: {},
  oldMelInst: 'pad',
  mode: null,        // current mode config
  modePatterns: {},   // per-mode pattern storage
};

// ======================== CANON ========================

export function computeCanon() {
  const s = state;
  const mode = s.mode;
  if (!mode || !mode.melodyGrids || !s.canonEnabled) { s.canonActive = {}; return; }

  const totalCols = mode.melodyGrids.reduce((sum, g) => sum + g.cols, 0);
  const melRows = mode.melodyGrids[0].rows;
  const result = {};

  for (const k of Object.keys(s.melActive)) {
    const [rs, cs] = k.split('-');
    const r = parseInt(rs), c = parseInt(cs);
    let nr = r, nc = c;

    if (s.canonMode === 'crab') {
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
      for (let gi = 0; gi < mode.mainGrids.length; gi++) {
        p.grids[gi] = { cells: {}, instrument: mode.mainGrids[gi].defaultInstrument };
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
      const period = g.cols - c;
      g.active[k] = { offset: s.step - (g.cols - 1) - Math.floor(Math.random() * period), vol: pat.cells[k] };
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
  if (mode.melodyGrids) {
    const totalMelCols = mode.melodyGrids.reduce((s, g) => s + g.cols, 0);
    const melRows = mode.melodyGrids[0].rows;
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

export function tick() {
  const s = state;
  const mode = s.mode;
  if (!mode) return;

  const hasMelody = mode.melodyGrids && mode.melodyGrids.length > 0;
  const totalMelCols = hasMelody ? mode.melodyGrids.reduce((sum, g) => sum + g.cols, 0) : 0;
  const melRows = hasMelody ? mode.melodyGrids[0].rows : 0;

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
          const period = g.cols - c;
          const a = oA[k];
          if (mod(s.step - a.offset - (g.cols - 1), period) === 0) {
            playNote(oI, parseInt(rs), a.vol * oldV, t);
          }
        }
      }
      if (hasMelody) {
        const melStep = Math.floor(s.step / s.melN);
        const mc = mod(melStep, totalMelCols);
        for (const k of Object.keys(s.oldMelActive)) {
          const [, cs] = k.split('-');
          if (parseInt(cs) === mc) {
            playNote(s.oldMelInst, parseInt(k.split('-')[0]), s.oldMelActive[k].vol * oldV, t);
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

    for (const k of Object.keys(g.active)) {
      const [rs, cs] = k.split('-');
      const r = parseInt(rs), c = parseInt(cs);
      const period = g.cols - c;
      const a = g.active[k];
      if (mod(s.step - a.offset - (g.cols - 1), period) === 0) {
        playNote(g.instrument, r, a.vol * newV, t);
        trig[r] = true;
      }
      for (let j = 0; j < g.cols; j++) {
        if (mod(s.step - a.offset - j, period) === 0) lit[r][j] = true;
      }
    }

    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const a = g.active[r + '-' + c];
        const isL = !!lit[r][c];
        const isT = trig[r] && c === g.cols - 1;
        if (isT && isL) applyStyle(g.cells[r][c], STYLES.TRIG);
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
          if (s.step % s.melN === 0) playNote(s.melInstrument, r, a.vol * newV, t);
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
            if (s.step % s.melN === 0) playNote(s.melInstrument, r, a.vol * newV, t);
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
  if (mode && mode.melodyGrids) {
    const totalMelCols = mode.melodyGrids.reduce((sum, g) => sum + g.cols, 0);
    const melRows = mode.melodyGrids[0].rows;
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
  if (mode && mode.melodyGrids) {
    const totalMelCols = mode.melodyGrids.reduce((s, g) => s + g.cols, 0);
    const melRows = mode.melodyGrids[0].rows;
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
  const primes = [3, 5, 7, 9, 11, 13, 17, 19];
  const accent = [4, 6, 8, 12, 16];

  // Generate base pattern in slot 0
  for (let gi = 0; gi < state.grids.length; gi++) {
    const g = state.grids[gi];
    const rows = g.rows, cols = g.cols;
    const cells = {};
    // Fill 15-40% of cells using rhythmic intervals
    const density = 0.15 + Math.random() * 0.25;
    const targetCells = Math.round(rows * cols * density);
    // Pick a few active rows with varying densities
    const activeRows = shuffle([...Array(rows).keys()]).slice(0, randInt(3, Math.min(rows, 6)));
    let placed = 0;
    for (const row of activeRows) {
      const interval = pick(primes.filter(x => x <= cols));
      const offset = randInt(0, interval - 1);
      for (let c = offset; c < cols; c += interval) {
        if (placed < targetCells) {
          cells[row + '-' + c] = Math.random() > 0.3 ? 1 : 0.5;
          placed++;
        }
      }
      // Scatter a few extra hits on this row
      const extras = randInt(0, 3);
      for (let e = 0; e < extras && placed < targetCells; e++) {
        const c = randInt(0, cols - 1);
        cells[row + '-' + c] = Math.random() > 0.5 ? 1 : 0.5;
        placed++;
      }
    }
    state.patterns[0].grids[gi] = { cells, instrument: pick(mode.mainInstruments).id };
  }

  // Generate melody seed
  if (mode.melodyGrids) {
    const totalCols = mode.melodyGrids.reduce((s, g) => s + g.cols, 0);
    const melRows = mode.melodyGrids[0].rows;
    const mc = {};
    const nV = randInt(3, Math.min(melRows, 6));
    const activeRows = shuffle([...Array(melRows).keys()]).slice(0, nV);
    for (const row of activeRows) {
      const interval = pick(primes.filter(x => x <= totalCols));
      const offset = randInt(0, interval - 1);
      for (let c = offset; c < totalCols; c += interval) {
        mc[row + '-' + c] = Math.random() > 0.3 ? 1 : 0.5;
      }
      const extras = randInt(0, 2);
      for (let e = 0; e < extras; e++) {
        mc[row + '-' + randInt(0, totalCols - 1)] = Math.random() > 0.5 ? 1 : 0.5;
      }
    }
    state.patterns[0].melody = { cells: mc, instrument: pick(mode.melodyInstruments || []).id || mode.melodyDefaultInstrument };
  }

  // Mutate into other pattern slots — each progressively more different
  for (let pi = 1; pi < state.patterns.length; pi++) {
    for (let gi = 0; gi < state.grids.length; gi++) {
      const base = state.patterns[0].grids[gi];
      const g = state.grids[gi];
      const cells = { ...base.cells };
      const mutations = randInt(2, 5 + pi);
      for (let m = 0; m < mutations; m++) {
        if (Math.random() > 0.4) {
          // Add a cell
          const r = randInt(0, g.rows - 1);
          const c = randInt(0, g.cols - 1);
          cells[r + '-' + c] = Math.random() > 0.5 ? 1 : 0.5;
        } else {
          // Remove a cell
          const k = Object.keys(cells);
          if (k.length > 0) delete cells[pick(k)];
        }
      }
      state.patterns[pi].grids[gi] = { cells, instrument: base.instrument };
    }
    if (mode.melodyGrids) {
      const baseMel = state.patterns[0].melody;
      const mc = { ...baseMel.cells };
      const totalCols = mode.melodyGrids.reduce((s, g) => s + g.cols, 0);
      const melRows = mode.melodyGrids[0].rows;
      const mutations = randInt(1, 3 + pi);
      for (let m = 0; m < mutations; m++) {
        if (Math.random() > 0.4) {
          mc[randInt(0, melRows - 1) + '-' + randInt(0, totalCols - 1)] = Math.random() > 0.5 ? 1 : 0.5;
        } else {
          const k = Object.keys(mc);
          if (k.length > 0) delete mc[pick(k)];
        }
      }
      state.patterns[pi].melody = { cells: mc, instrument: baseMel.instrument };
    }
  }

  state.currentPattern = 0;
  loadPat(0);
  start();
}
