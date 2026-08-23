export const CELL = { OFF: 0, HALF: 1, FULL: 2, LIT: 3, TRIG: 4 };
const CELL_CLASS = ['cell', 'cell half', 'cell on', 'cell lit', 'cell trig'];

const TICK_MS = 25;
const AHEAD = 0.1;
const HIDDEN_AHEAD = 2.0;
const FADE = 0.04;

export const state = {
  ctx: null,
  step: 0,
  visualStep: 0,
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
  masterGain: null,
  buses: {},
  live: new Set(),
  patterns: [],
  currentPattern: 0,
  autoCycle: false,
  cycleSteps: 64,
  patButtons: [],
  melButtons: [],
  canonEnabled: false,
  canonMode: 'simple',
  canonInterval: 4,
  canonOffset: 8,
  canonCells: [],
  canonActive: {},
  trSteps: 16,
  trRem: 0,
  oldActives: [],
  oldInsts: [],
  oldMelActive: {},
  oldMelInst: 'pad',
  balance: 50,
  simpleGrid: true,
  mode: null,
  modePatterns: {},

  schedTimer: null,
  nextStepTime: 0,
  paintQueue: [],
  rafId: null,
  dirty: false,
};

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

function mod(a, n) { return ((a % n) + n) % n; }

function melTotalCols() {
  const g = melGrids();
  return g ? g.reduce((sum, x) => sum + x.cols, 0) : 0;
}

export function computeCanon() {
  const s = state;
  const mode = s.mode;
  if (!mode || !melGrids() || !s.canonEnabled) { s.canonActive = {}; markDirty(); return; }

  const totalCols = melTotalCols();
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
  markDirty();
}

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

      if (i === 0 && melGrids() && melGrids().length > 0) {
        const totalCols = melTotalCols();
        const rows = melGrids()[0].rows;
        const mc = {};

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
  const mode = s.mode;
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
    const pat = s.patterns[idx].grids[gi] || { cells: {}, instrument: g.instrument };
    g.active = {};
    g.instrument = pat.instrument;
    for (const b of g.buttons) b.className = b.dataset.inst === g.instrument ? 'sel' : '';
    for (const k of Object.keys(pat.cells)) {
      const [, cs] = k.split('-');
      const c = parseInt(cs);
      if (mode.colSeqs) {

        g.active[k] = { offset: 0, vol: pat.cells[k] };
      } else {
        const period = g.cols - c;
        g.active[k] = { offset: s.step - (g.cols - 1) - Math.floor(Math.random() * period), vol: pat.cells[k] };
      }
    }
  }

  const mp = s.patterns[idx].melody;
  s.melActive = {};
  s.melInstrument = mp.instrument;
  for (const b of s.melButtons) b.className = b.dataset.inst === s.melInstrument ? 'sel' : '';
  for (const k of Object.keys(mp.cells)) s.melActive[k] = { vol: mp.cells[k] };

  s.currentPattern = idx;
  for (let i = 0; i < s.patButtons.length; i++) {
    s.patButtons[i].className = i === s.currentPattern ? 'pat-btn sel' : 'pat-btn';
  }

  computeCanon();
  markDirty();
}

export function selectPat(idx) {
  savePat(state.currentPattern);
  loadPat(idx);
}

export function teardownAudio() {
  const s = state;
  silence();
  for (const node of [s.masterGain, s.masterComp, s.reverbInput, s.delayInput]) {
    if (node) node.disconnect();
  }
  for (const id of Object.keys(s.buses)) {
    const b = s.buses[id];
    b.dry.disconnect(); b.del.disconnect(); b.rev.disconnect();
  }
  s.buses = {};
  s.masterGain = s.masterComp = s.reverbInput = s.delayInput = null;
}

export function buildEffects(config) {
  const ctx = state.ctx;
  teardownAudio();

  const master = ctx.createGain();
  master.gain.value = 1;
  master.connect(ctx.destination);
  state.masterGain = master;

  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = config.compThreshold || -12;
  comp.knee.value = 4;
  comp.ratio.value = config.compRatio || 8;
  comp.attack.value = 0.002;
  comp.release.value = 0.12;
  comp.connect(master);
  state.masterComp = comp;

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

  buildBuses();
}

function buildBuses() {
  const ctx = state.ctx;
  const fxAll = (state.mode && state.mode.fx) || {};
  state.buses = {};
  for (const inst of Object.keys(fxAll)) {
    const fx = fxAll[inst];
    const dry = ctx.createGain();
    dry.gain.value = fx.gain;
    dry.connect(state.masterComp);
    const del = ctx.createGain();
    del.gain.value = fx.delay * fx.gain;
    del.connect(state.delayInput);
    const rev = ctx.createGain();
    rev.gain.value = fx.reverb * fx.gain;
    rev.connect(state.reverbInput);
    state.buses[inst] = { dry, del, rev };
  }
}

// A grid can have fewer rows than its instrument has pitches -- the simple
// grid is 8 rows against a 16-pitch instrument. Spread those rows over the
// whole range instead of taking the top of it, so the low register does not
// vanish when the grid is simplified. The pitch tables are pentatonic, so
// every stride lands on a consonant degree.
function pitchIndex(inst, row, rows) {
  const buf = state.buffers[inst];
  if (!buf || buf.length <= rows) return row;
  return Math.min(buf.length - 1, row * Math.floor(buf.length / rows));
}

export function playNote(inst, fi, vol, time) {
  const s = state;
  const buf = s.buffers[inst];
  if (!buf || !buf[fi]) return;
  const bus = s.buses[inst];
  if (!bus) return;

  const src = s.ctx.createBufferSource();
  src.buffer = buf[fi];
  const g = s.ctx.createGain();
  g.gain.value = vol;
  src.connect(g);
  g.connect(bus.dry);
  g.connect(bus.del);
  g.connect(bus.rev);

  src.start(time);
  s.live.add(src);
  src.onended = () => {
    s.live.delete(src);
    src.disconnect();
    g.disconnect();
  };
}

function silence() {
  const s = state;
  if (!s.ctx) return;
  const t = s.ctx.currentTime;
  if (s.masterGain) {
    const gain = s.masterGain.gain;
    gain.cancelScheduledValues(t);
    gain.setValueAtTime(gain.value, t);
    gain.linearRampToValueAtTime(0, t + FADE);
  }
  for (const src of s.live) {
    try { src.stop(t + FADE); } catch (e) {  }
  }
  s.live.clear();
}

function scheduleStep(step, time) {
  const s = state;
  const mode = s.mode;
  const hasMelody = !!melGrids();
  const totalMelCols = melTotalCols();
  const melRows = hasMelody ? melGrids()[0].rows : 0;

  if (s.autoCycle && step > 0 && step % s.cycleSteps === 0) {
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

  const rhythmBal = Math.min(1, s.balance / 50);
  const melBal = Math.min(1, (100 - s.balance) / 50);

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
          if (!fires(mode, g, c, a, step)) continue;
          const vol = mode.stepVels
            ? a.vol * mode.stepVels[mod(step - (g.cols - 1), mode.seqLen)]
            : a.vol;
          playNote(oI, pitchIndex(oI, parseInt(rs), g.rows), vol * oldV * rhythmBal, time);
        }
      }
      if (hasMelody && step % s.melN === 0) {
        const mc = mod(Math.floor(step / s.melN), totalMelCols);
        for (const k of Object.keys(s.oldMelActive)) {
          const [rs, cs] = k.split('-');
          if (parseInt(cs) === mc) {
            playNote(s.oldMelInst, pitchIndex(s.oldMelInst, parseInt(rs), melRows),
                     s.oldMelActive[k].vol * oldV * melBal, time);
          }
        }
      }
    }
  }

  for (const g of s.grids) {
    for (const k of Object.keys(g.active)) {
      const [rs, cs] = k.split('-');
      const c = parseInt(cs);
      const a = g.active[k];
      if (!fires(mode, g, c, a, step)) continue;
      const vol = mode.stepVels
        ? a.vol * mode.stepVels[mod(step - (g.cols - 1), mode.seqLen)]
        : a.vol;
      playNote(g.instrument, pitchIndex(g.instrument, parseInt(rs), g.rows),
               vol * newV * rhythmBal, time);
    }
  }

  if (hasMelody && step % s.melN === 0) {
    const mc = mod(Math.floor(step / s.melN), totalMelCols);
    for (const k of Object.keys(s.melActive)) {
      const [rs, cs] = k.split('-');
      if (parseInt(cs) === mc) {
        playNote(s.melInstrument, pitchIndex(s.melInstrument, parseInt(rs), melRows),
                 s.melActive[k].vol * newV * melBal, time);
      }
    }
    if (s.canonEnabled) {
      for (const k of Object.keys(s.canonActive)) {
        const [rs, cs] = k.split('-');
        if (parseInt(cs) === mc) {
          playNote(s.melInstrument, pitchIndex(s.melInstrument, parseInt(rs), melRows),
                   s.canonActive[k].vol * newV * melBal, time);
        }
      }
    }
  }
}

function fires(mode, g, c, a, step) {
  if (mode.colSeqs) {
    return mode.colSeqs[c][mod(step - (g.cols - 1), mode.seqLen)] === 1;
  }
  return mod(step - a.offset - (g.cols - 1), g.cols - c) === 0;
}

function scheduler() {
  const s = state;
  if (!s.playing || !s.ctx) return;
  const ahead = document.hidden ? HIDDEN_AHEAD : AHEAD;
  const horizon = s.ctx.currentTime + ahead;

  if (s.nextStepTime < s.ctx.currentTime - 0.5) s.nextStepTime = s.ctx.currentTime;
  while (s.nextStepTime < horizon) {
    scheduleStep(s.step, s.nextStepTime);
    s.paintQueue.push({ step: s.step, time: s.nextStepTime });
    s.nextStepTime += s.stepMs / 1000;
    s.step++;
  }
}

export function markDirty() {
  state.dirty = true;
  if (state.rafId == null) state.rafId = requestAnimationFrame(frame);
}

function frame() {
  const s = state;
  s.rafId = null;
  let step = s.visualStep;
  let changed = s.dirty;
  s.dirty = false;

  if (s.playing && s.ctx) {
    const now = s.ctx.currentTime;
    while (s.paintQueue.length && s.paintQueue[0].time <= now) {
      step = s.paintQueue.shift().step;
      changed = true;
    }
  }

  if (changed) {
    s.visualStep = step;
    paint(step);
  }
  if (s.playing) s.rafId = requestAnimationFrame(frame);
}

function writeCells(cells, want, shown, cols) {
  for (let i = 0; i < want.length; i++) {
    if (want[i] === shown[i]) continue;
    shown[i] = want[i];
    const el = cells[(i / cols) | 0][i % cols];
    if (el) el.className = CELL_CLASS[want[i]];
  }
}

function paint(step) {
  const s = state;
  const mode = s.mode;
  if (!mode) return;
  const playing = s.playing;

  for (const g of s.grids) {
    const rows = g.rows, cols = g.cols, n = rows * cols;
    if (!g.want || g.want.length !== n) {
      g.want = new Int8Array(n);
      g.shown = new Int8Array(n).fill(-1);
    }
    const want = g.want;
    want.fill(CELL.OFF);

    for (const k of Object.keys(g.active)) {
      const [rs, cs] = k.split('-');
      const i = parseInt(rs) * cols + parseInt(cs);
      want[i] = g.active[k].vol >= 1 ? CELL.FULL : CELL.HALF;
    }

    if (playing) {

      const rowFired = new Uint8Array(rows);
      const lit = new Uint8Array(n);
      for (const k of Object.keys(g.active)) {
        const [rs, cs] = k.split('-');
        const r = parseInt(rs), c = parseInt(cs);
        const a = g.active[k];
        if (mode.colSeqs) {
          if (mode.colSeqs[c][mod(step - (cols - 1), mode.seqLen)]) rowFired[r] = 1;
          for (let d = 0; d < cols; d++) {
            if (mode.colSeqs[c][mod(step - d, mode.seqLen)]) lit[r * cols + d] = 1;
          }
        } else {
          const period = cols - c;
          if (mod(step - a.offset - (cols - 1), period) === 0) rowFired[r] = 1;
          for (let j = 0; j < cols; j++) {
            if (mod(step - a.offset - j, period) === 0) lit[r * cols + j] = 1;
          }
        }
      }
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c;
          if (!lit[i]) continue;
          if (rowFired[r] && c === cols - 1) want[i] = CELL.TRIG;
          else if (want[i] === CELL.OFF) want[i] = CELL.LIT;
        }
      }
    }

    writeCells(g.cells, want, g.shown, cols);
  }

  const mg = melGrids();
  if (!mg || s.melCells.length === 0) return;
  const cols = melTotalCols();
  const rows = mg[0].rows;
  const cursor = playing ? mod(Math.floor(step / s.melN), cols) : -1;

  paintVoice(s.melCells, s.melActive, rows, cols, cursor, 'mel');
  if (s.canonEnabled && s.canonCells.length > 0) {
    paintVoice(s.canonCells, s.canonActive, rows, cols, cursor, 'canon');
  }
}

const voiceBufs = {};

function paintVoice(cells, active, rows, cols, cursor, key) {
  const n = rows * cols;
  let buf = voiceBufs[key];
  if (!buf || buf.want.length !== n) {
    buf = voiceBufs[key] = { want: new Int8Array(n), shown: new Int8Array(n).fill(-1) };
  }
  const { want, shown } = buf;
  want.fill(CELL.OFF);

  for (const k of Object.keys(active)) {
    const [rs, cs] = k.split('-');
    const r = parseInt(rs), c = parseInt(cs);
    if (r >= rows || c >= cols) continue;
    want[r * cols + c] = active[k].vol >= 1 ? CELL.FULL : CELL.HALF;
  }
  if (cursor >= 0) {
    for (let r = 0; r < rows; r++) {
      const i = r * cols + cursor;
      want[i] = want[i] === CELL.OFF ? CELL.LIT : CELL.TRIG;
    }
  }

  writeCells(cells, want, shown, cols);
}

export function resetPainter() {
  for (const k of Object.keys(voiceBufs)) delete voiceBufs[k];
  markDirty();
}

export function start() {
  const s = state;
  if (!s.ctx || !s.mode || s.playing) return;

  if (s.ctx.state !== 'running') s.ctx.resume().catch(() => {});

  buildEffects(s.mode.effects);
  s.playing = true;
  s.paintQueue.length = 0;
  s.nextStepTime = s.ctx.currentTime + 0.05;
  s.schedTimer = setInterval(scheduler, TICK_MS);
  scheduler();
  markDirty();
}

export function stop() {
  const s = state;
  if (s.schedTimer) clearInterval(s.schedTimer);
  s.schedTimer = null;
  s.playing = false;
  silence();
  s.step = 0;
  s.visualStep = 0;
  s.paintQueue.length = 0;
  markDirty();
}

export function resetAll() {
  if (state.playing) stop();
  state.step = 0;
  for (const g of state.grids) g.active = {};
  state.melActive = {};
  initPatterns(state.mode);
  state.currentPattern = 0;
  for (let i = 0; i < state.patButtons.length; i++)
    state.patButtons[i].className = i === 0 ? 'pat-btn sel' : 'pat-btn';
  computeCanon();
  markDirty();
}

export function saveSong() {
  savePat(state.currentPattern);
  const song = {
    mode: state.mode.id,
    patterns: state.patterns,
    currentPattern: state.currentPattern,
    stepMs: state.stepMs,
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

export function loadSong(onLoaded, onError) {
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = '.json';
  inp.addEventListener('change', () => {
    if (!inp.files[0]) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const s = JSON.parse(reader.result);
        if (!s.patterns) throw new Error('not a polynome file');
        if (s.mode && s.mode !== state.mode.id) {
          if (onError) onError(`that file is for ${s.mode} mode`);
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
      } catch (e) {
        if (onError) onError(`could not read that file: ${e.message}`);
      }
    };
    reader.readAsText(inp.files[0]);
  });
  inp.click();
}

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
  const wasPlaying = state.playing;
  if (wasPlaying) stop();
  state.step = 0;
  const mode = state.mode;

  const maxCol = (cols) => cols - 2;

  const coprimeSets = [
    [3, 4], [3, 5], [3, 7], [4, 5], [4, 7], [5, 7], [5, 8],
    [3, 4, 7], [3, 5, 8], [4, 7, 11], [5, 7, 13], [3, 8, 11],
  ];

  for (let gi = 0; gi < state.grids.length; gi++) {
    const g = state.grids[gi];
    const rows = g.rows, cols = g.cols;
    const cells = {};

    if (mode.colSeqs) {

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

      const periods = pick(coprimeSets).filter(p => p <= mc);
      if (periods.length === 0) continue;

      const allRows = shuffle([...Array(rows).keys()]);
      let ri = 0;

      const anchorInfo = [];
      for (const period of periods) {
        if (ri >= rows) break;
        const row = allRows[ri++];
        const col = mc - period;
        const patType = pick(['pulse', 'skip', 'cluster']);

        if (patType === 'pulse') {

          cells[row + '-' + col] = 1;
        } else if (patType === 'skip') {

          cells[row + '-' + col] = 1;
          const col2 = Math.max(0, Math.min(mc, col - randInt(1, 3)));
          if (col2 !== col) cells[row + '-' + col2] = 0.5;
        } else {

          cells[row + '-' + col] = 1;
          if (col > 0) cells[row + '-' + (col - 1)] = 0.5;
          if (col > 1 && Math.random() > 0.5) cells[row + '-' + (col - 2)] = 0.5;
        }
        anchorInfo.push({ row, period, col });
      }

      const nResponse = randInt(1, 2);
      for (let rr = 0; rr < nResponse && ri < rows; rr++) {
        const row = allRows[ri++];
        const anchor = pick(anchorInfo);

        const offset = Math.floor(anchor.period / 2);
        const responseCol = Math.max(0, Math.min(mc, anchor.col + offset));
        if (responseCol !== anchor.col) {
          cells[row + '-' + responseCol] = 0.5;
        }
      }

      if (ri < rows && Math.random() > 0.3) {
        const row = allRows[ri++];

        const accentCol = pick([0, Math.floor(mc / 2)]);
        cells[row + '-' + accentCol] = 1;
      }
    }

    state.patterns[0].grids[gi] = { cells, instrument: pick(mode.mainInstruments).id };
  }

  if (melGrids()) {
    const totalCols = melTotalCols();
    const melRows = melGrids()[0].rows;
    const mc = {};

    const motifLen = randInt(3, 5);
    const motif = [];
    const intervalPool = [-3, -2, -1, 1, 2, 3];
    for (let i = 0; i < motifLen; i++) {
      motif.push(pick(intervalPool));
    }

    const clamp = (v) => Math.max(0, Math.min(melRows - 1, v));
    let startPitch = randInt(1, Math.max(1, melRows - 3));
    const spacing = randInt(2, 4);

    let col = 0;
    let pitch = startPitch;
    mc[pitch + '-' + col] = 1;
    for (let i = 0; i < motif.length && col + spacing < totalCols; i++) {
      col += spacing;
      pitch = clamp(pitch + motif[i]);
      mc[pitch + '-' + col] = (i === 0) ? 1 : 0.5;
    }

    col += spacing * randInt(2, 4);

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

    if (col < totalCols - 3 * spacing) {
      const varType = pick(['retrograde', 'truncated', 'augmented']);
      let varMotif;
      if (varType === 'retrograde') {
        varMotif = motif.slice().reverse().map(x => -x);
      } else if (varType === 'truncated') {
        varMotif = motif.slice(0, Math.max(2, motif.length - 1));
      } else {

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

  const nPat = state.patterns.length;
  for (let pi = 1; pi < nPat; pi++) {
    const progress = pi / (nPat - 1);

    for (let gi = 0; gi < state.grids.length; gi++) {
      const base = state.patterns[0].grids[gi];
      const g = state.grids[gi];
      const mc = maxCol(g.cols);
      const cells = { ...base.cells };

      const nAdd = Math.floor(1 + progress * 4);

      for (let a = 0; a < nAdd; a++) {
        if (progress < 0.5) {

          const r = randInt(0, g.rows - 1);
          const period = pick([3, 4, 5, 7, 8, 11]);
          const c = Math.max(0, Math.min(mc, mc - period));
          cells[r + '-' + c] = Math.random() > 0.3 ? 1 : 0.5;
        } else {

          const keys = Object.keys(cells);
          if (keys.length > 0) {
            const k = pick(keys);
            const [rs, cs] = k.split('-');
            const r = parseInt(rs);
            const c = parseInt(cs);

            const nr = Math.max(0, Math.min(g.rows - 1, r + pick([-1, 0, 1])));
            const nc = Math.max(0, Math.min(mc, c + pick([-2, -1, 1, 2])));
            cells[nr + '-' + nc] = 0.5;
          }
        }
      }

      if (progress > 0.2 && progress < 0.7 && Math.random() > 0.5) {
        const keys = Object.keys(cells);
        if (keys.length > 2) delete cells[pick(keys)];
      }

      state.patterns[pi].grids[gi] = { cells, instrument: base.instrument };
    }

    if (melGrids()) {
      const baseMel = state.patterns[Math.max(0, pi - 1)].melody;
      const melCells = { ...baseMel.cells };
      const totalCols = melTotalCols();
      const melRows = melGrids()[0].rows;

      const nMut = Math.floor(1 + progress * 3);
      for (let m = 0; m < nMut; m++) {
        const keys = Object.keys(melCells);
        if (keys.length === 0) break;

        const action = Math.random();
        if (action < 0.4) {

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
