import {
  state,
  initPatterns, selectPat, savePat, loadPat, buildEffects, teardownAudio,
  start, stop, resetAll, randomize, saveSong, loadSong, computeCanon,
  markDirty, resetPainter,
} from './engine.js';
import { loadMode, loadModeList, bufferCount } from './loader.js';

const STEPS_PER_BEAT = 4;
const DEFAULT_STEP_MS = 350;

const modeCache = {};
const bufferCache = {};
const failed = {};
let modeIds = [];
let statusText = '';

function setStatus(msg) {
  statusText = msg || '';
  const el = document.getElementById('status');
  el.textContent = statusText;
  el.style.display = statusText ? 'block' : 'none';
}

function refreshStatus() {
  if (Object.keys(failed).length > 0) {
    setStatus(`could not load ${Object.keys(failed).join(', ')}`);
  } else if (state.ctx && state.ctx.state !== 'running' && state.playing) {
    setStatus('click anywhere for sound');
  } else {
    setStatus('');
  }
}

function installAudioUnlock() {
  const resume = () => {
    if (state.ctx && state.ctx.state !== 'running') {
      state.ctx.resume().then(refreshStatus).catch(() => {});
    }
  };
  for (const ev of ['pointerdown', 'touchstart', 'keydown']) {
    window.addEventListener(ev, resume, { capture: true, passive: true });
  }
}

export async function switchMode(modeId) {
  if (state.playing) stop();
  teardownAudio();

  if (state.mode) {
    if (state.patterns.length > 0) savePat(state.currentPattern);
    state.modePatterns[state.mode.id] = JSON.parse(JSON.stringify(state.patterns));
  }

  if (!state.ctx) state.ctx = new AudioContext();

  const loading = document.getElementById('loading');
  const main = document.getElementById('main');
  const progressBar = document.getElementById('progress-bar');
  const progressFill = document.getElementById('progress-fill');

  if (!modeCache[modeId]) {
    loading.textContent = `loading ${modeId}`;
    loading.style.display = '';
    progressBar.style.display = '';
    progressFill.style.width = '0%';
    main.style.display = 'none';

    try {
      const { config, buffers } = await loadMode(modeId, state.ctx, pct => {
        progressFill.style.width = (pct * 100) + '%';
      });
      modeCache[modeId] = config;
      bufferCache[modeId] = buffers;
      delete failed[modeId];
    } catch (e) {
      failed[modeId] = e.message;
      loading.style.display = 'none';
      progressBar.style.display = 'none';
      main.style.display = state.mode ? 'flex' : 'none';
      markFailedModes();
      refreshStatus();
      return;
    }
  }

  const mode = modeCache[modeId];
  state.mode = mode;
  state.buffers = bufferCache[modeId];
  buildEffects(mode.effects);

  if (state.modePatterns[modeId]) {
    state.patterns = state.modePatterns[modeId];
    state.currentPattern = 0;
  } else {
    initPatterns(mode);
  }

  buildUI(mode);

  for (const b of document.querySelectorAll('#mode-col button')) {
    b.classList.toggle('sel', b.dataset.mode === modeId);
  }

  state.stepMs = mode.defaultStepMs || DEFAULT_STEP_MS;
  document.getElementById('speed').value = 550 - state.stepMs;
  showTempo();
  if (mode.defaultMelN) {
    state.melN = mode.defaultMelN;
    document.getElementById('melN').value = state.melN;
    showMelN();
  }

  loading.style.display = 'none';
  progressBar.style.display = 'none';
  main.style.display = 'flex';

  loadPat(state.currentPattern);
  updateTransport();
  refreshStatus();
}

function markFailedModes() {
  for (const b of document.querySelectorAll('#mode-col button')) {
    b.classList.toggle('err', !!failed[b.dataset.mode]);
  }
}

function buildUI(mode) {
  state.grids = [];
  state.melCells = [];
  state.melActive = {};
  state.melButtons = [];
  state.patButtons = [];
  state.step = 0;

  const isSimple = state.simpleGrid && mode.id !== 'original';
  const cellSize = isSimple ? 28 : (mode.cellSize || 16);
  const hidden = mode.hideControls || [];

  const patCol = document.getElementById('pat-col');
  patCol.innerHTML = '';
  patCol.style.display = (hidden.includes('pattern') || !state.autoCycle) ? 'none' : '';
  const numPat = mode.numPatterns || 8;
  for (let pi = 0; pi < numPat; pi++) {
    const pb = document.createElement('button');
    pb.className = pi === 0 ? 'pat-btn sel' : 'pat-btn';
    pb.textContent = pi + 1;
    pb.dataset.pi = pi;
    pb.addEventListener('click', () => { selectPat(pi); updateTransport(); });
    patCol.appendChild(pb);
    state.patButtons.push(pb);
  }

  const stage = document.getElementById('stage');
  stage.innerHTML = '';

  const mainGrids = isSimple
    ? [{ rows: 8, cols: 16, defaultInstrument: mode.mainGrids[0].defaultInstrument }]
    : mode.mainGrids;
  const layoutCols = isSimple ? 1 : ((mode.mainGridLayout && mode.mainGridLayout.cols) || 2);
  stage.style.gridTemplateColumns = `repeat(${layoutCols}, auto)`;

  for (let gi = 0; gi < mainGrids.length; gi++) {
    const def = mainGrids[gi];
    const panel = document.createElement('div');
    panel.className = 'panel';

    const instBar = document.createElement('div');
    instBar.className = 'inst';
    const g = {
      cells: [], active: {}, instrument: def.defaultInstrument,
      buttons: [], rows: def.rows, cols: def.cols,
    };

    for (const inst of mode.mainInstruments) {
      const btn = document.createElement('button');
      btn.textContent = inst.label;
      btn.dataset.inst = inst.id;
      if (inst.id === g.instrument) btn.className = 'sel';
      btn.addEventListener('click', () => {
        g.instrument = inst.id;
        for (const b of g.buttons) b.className = b.dataset.inst === inst.id ? 'sel' : '';
      });
      instBar.appendChild(btn);
      g.buttons.push(btn);
    }
    panel.appendChild(instBar);
    panel.appendChild(makeGrid(def.rows, def.cols, cellSize, g.cells, 0,
      (r, c) => toggleCell(g.active, g.cells, r, c, offsetFor(g, c))));

    stage.appendChild(panel);
    state.grids.push(g);
  }

  const melWrap = document.getElementById('mel-wrap');
  const melBar = document.getElementById('mel-inst-bar');
  const melGridsEl = document.getElementById('mel-grids');
  melBar.innerHTML = '';
  melGridsEl.innerHTML = '';

  const melGrids = isSimple && mode.melodyGrids ? [{ rows: 8, cols: 32 }] : mode.melodyGrids;
  const hasMelody = melGrids && melGrids.length > 0 && !hidden.includes('melody');

  if (hasMelody) {
    melWrap.style.display = 'flex';
    state.melInstrument = mode.melodyDefaultInstrument
      || (mode.melodyInstruments[0] && mode.melodyInstruments[0].id);

    for (const inst of (mode.melodyInstruments || [])) {
      const btn = document.createElement('button');
      btn.textContent = inst.label;
      btn.dataset.inst = inst.id;
      if (inst.id === state.melInstrument) btn.className = 'sel';
      btn.addEventListener('click', () => {
        state.melInstrument = inst.id;
        for (const b of state.melButtons) b.className = b.dataset.inst === inst.id ? 'sel' : '';
      });
      melBar.appendChild(btn);
      state.melButtons.push(btn);
    }

    for (let r = 0; r < melGrids[0].rows; r++) state.melCells[r] = [];
    let colBase = 0;
    for (const def of melGrids) {
      melGridsEl.appendChild(makeGrid(def.rows, def.cols, cellSize, state.melCells, colBase,
        (r, c) => { toggleCell(state.melActive, state.melCells, r, c); computeCanon(); }));
      colBase += def.cols;
    }
  } else {
    melWrap.style.display = 'none';
  }

  const canonWrap = document.getElementById('canon-wrap');
  const canonBar = document.getElementById('canon-mode-bar');
  const canonGridsEl = document.getElementById('canon-grids');
  canonBar.innerHTML = '';
  canonGridsEl.innerHTML = '';
  state.canonCells = [];
  state.canonButtons = [];

  if (hasMelody) {
    for (const cm of ['simple', 'interval', 'crab', 'mirror', 'table']) {
      const btn = document.createElement('button');
      btn.textContent = cm;
      if (cm === state.canonMode) btn.className = 'sel';
      btn.addEventListener('click', () => {
        state.canonMode = cm;
        for (const b of state.canonButtons) b.className = '';
        btn.className = 'sel';
        computeCanon();
      });
      canonBar.appendChild(btn);
      state.canonButtons.push(btn);
    }

    for (let r = 0; r < melGrids[0].rows; r++) state.canonCells[r] = [];
    let colBase = 0;
    for (const def of melGrids) {
      const el = makeGrid(def.rows, def.cols, cellSize, state.canonCells, colBase, null);
      canonGridsEl.appendChild(el);
      colBase += def.cols;
    }
    canonWrap.style.display = state.canonEnabled ? 'flex' : 'none';
  } else {
    canonWrap.style.display = 'none';
  }

  for (const [id, visible] of [
    ['melNWrap', hasMelody], ['canonToggle', hasMelody], ['balanceWrap', hasMelody],
    ['gridToggle', mode.id !== 'original'],
  ]) {
    document.getElementById(id).style.display = visible ? '' : 'none';
  }

  resetPainter();
}

function makeGrid(rows, cols, cellSize, cellStore, colBase, onToggle) {
  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
  for (let r = 0; r < rows; r++) {
    if (!cellStore[r]) cellStore[r] = [];
    for (let c = 0; c < cols; c++) {
      const el = document.createElement('div');
      el.className = 'cell';
      el.style.width = cellSize + 'px';
      el.style.height = cellSize + 'px';
      if (onToggle) {
        el.addEventListener('click', () => onToggle(r, colBase + c));
      } else {
        el.style.cursor = 'default';
      }
      grid.appendChild(el);
      cellStore[r][colBase + c] = el;
    }
  }
  return grid;
}

function offsetFor(g, c) {
  return state.mode.colSeqs ? 0 : state.step - (g.cols - 1);
}

function toggleCell(active, cells, r, c, offset) {
  const k = r + '-' + c;
  const simple = state.simpleGrid && state.mode.id !== 'original';
  if (!active[k]) {
    active[k] = { offset: offset || 0, vol: simple ? 1 : 0.5 };
  } else if (!simple && active[k].vol < 1) {
    active[k].vol = 1;
  } else {
    delete active[k];
  }
  markDirty();
  if (!state.playing) start();
  updateTransport();
}

function updateTransport() {
  const btn = document.getElementById('ctl');
  let any = Object.keys(state.melActive).length > 0;
  for (const g of state.grids) if (Object.keys(g.active).length > 0) any = true;
  btn.textContent = state.playing ? 'stop' : 'play';
  btn.style.visibility = (state.playing || any) ? '' : 'hidden';
  refreshStatus();
}

function showTempo() {
  document.getElementById('bpm').textContent =
    Math.round(60000 / (state.stepMs * STEPS_PER_BEAT)) + ' bpm';
}

function showMelN() {
  document.getElementById('melNDisp').textContent = 'melody ÷' + state.melN;
}

export async function initApp() {
  installAudioUnlock();

  try {
    modeIds = await loadModeList();
  } catch (e) {
    document.getElementById('loading').style.display = 'none';
    setStatus('no audio assets — run: python build/build.py');
    return;
  }

  const modeCol = document.getElementById('mode-col');
  for (const id of modeIds) {
    const btn = document.createElement('button');
    btn.textContent = id;
    btn.dataset.mode = id;
    btn.addEventListener('click', () => switchMode(id));
    modeCol.appendChild(btn);
  }

  const speed = document.getElementById('speed');
  speed.addEventListener('input', () => {
    state.stepMs = 550 - parseInt(speed.value);
    showTempo();
  });
  state.stepMs = 550 - parseInt(speed.value);
  showTempo();

  const cycleLen = document.getElementById('cycleLen');
  const cycleDisp = document.getElementById('cycleDisp');
  const showCycle = () => {
    state.cycleSteps = parseInt(cycleLen.value);
    cycleDisp.textContent = state.cycleSteps + ' steps';
  };
  cycleLen.addEventListener('input', showCycle);
  showCycle();

  const melN = document.getElementById('melN');
  melN.addEventListener('input', () => {
    state.melN = parseInt(melN.value);
    showMelN();
  });
  showMelN();

  const balance = document.getElementById('balance');
  const balDisp = document.getElementById('balDisp');
  const showBalance = () => {
    state.balance = parseInt(balance.value);
    balDisp.textContent = `melody ${100 - state.balance}`;
  };
  balance.addEventListener('input', showBalance);
  showBalance();

  document.getElementById('ctl').addEventListener('click', () => {
    if (state.playing) stop(); else start();
    updateTransport();
  });

  document.getElementById('gridToggle').addEventListener('click', () => {
    state.simpleGrid = !state.simpleGrid;
    document.getElementById('gridToggle').textContent = state.simpleGrid ? 'simple' : 'complex';
    if (!state.mode) return;
    const wasPlaying = state.playing;
    if (wasPlaying) stop();
    buildUI(state.mode);
    loadPat(state.currentPattern);
    if (wasPlaying) start();
    updateTransport();
  });

  document.getElementById('cycleToggle').addEventListener('click', () => {
    state.autoCycle = !state.autoCycle;
    document.getElementById('cycleToggle').textContent = state.autoCycle ? 'cycle on' : 'cycle off';
    document.getElementById('pat-col').style.display = state.autoCycle ? '' : 'none';
  });

  document.getElementById('canonToggle').addEventListener('click', () => {
    state.canonEnabled = !state.canonEnabled;
    document.getElementById('canonToggle').textContent = state.canonEnabled ? 'canon on' : 'canon off';
    document.getElementById('canon-wrap').style.display = state.canonEnabled ? 'flex' : 'none';
    computeCanon();
  });

  document.getElementById('btn-reset').addEventListener('click', () => { resetAll(); updateTransport(); });
  document.getElementById('btn-random').addEventListener('click', () => { randomize(); updateTransport(); });
  document.getElementById('btn-save').addEventListener('click', saveSong);
  document.getElementById('btn-load').addEventListener('click', () => {
    loadSong(song => {
      if (song.stepMs) state.stepMs = song.stepMs;
      else if (song.bpm) state.stepMs = Math.round(60000 / song.bpm);
      speed.value = 550 - state.stepMs;
      showTempo();
      if (song.cycleSteps) { cycleLen.value = song.cycleSteps; showCycle(); }
      if (typeof song.autoCycle === 'boolean') {
        state.autoCycle = song.autoCycle;
        document.getElementById('cycleToggle').textContent = song.autoCycle ? 'cycle on' : 'cycle off';
      }
      if (song.melN) { melN.value = song.melN; showMelN(); }
      updateTransport();
    }, setStatus);
  });

  await switchMode(modeIds[0]);
  preloadRest();
}

async function preloadRest() {
  for (const id of modeIds) {
    if (modeCache[id]) continue;
    try {
      const { config, buffers } = await loadMode(id, state.ctx, () => {});
      modeCache[id] = config;
      bufferCache[id] = buffers;
      if (bufferCount(buffers) === 0) failed[id] = 'no audio';
    } catch (e) {
      failed[id] = e.message;
    }
  }
  markFailedModes();
  refreshStatus();
}
