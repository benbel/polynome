// ui.js — DOM construction, controls, mode switcher

import {
  state, STYLES, applyStyle,
  initPatterns, selectPat, buildEffects,
  start, stop, restartTimer, resetAll, randomize,
  saveSong, loadSong, computeCanon,
} from './engine.js';
import { loadMode } from './loader.js';

// Available modes — lazy-loaded
const MODE_IDS = ['texture', 'original', 'clear', 'voices', 'speech'];
const modeCache = {};

// ======================== MODE SWITCHING ========================

export async function switchMode(modeId) {
  if (state.playing) stop();

  // Save current mode patterns
  if (state.mode) {
    const { savePat } = await import('./engine.js');
    if (state.patterns.length > 0) savePat(state.currentPattern);
    state.modePatterns[state.mode.id] = JSON.parse(JSON.stringify(state.patterns));
  }

  // Load mode config
  if (!modeCache[modeId]) {
    const mod = await import(`./modes/${modeId}.js`);
    modeCache[modeId] = mod.default;
  }
  const mode = modeCache[modeId];
  state.mode = mode;

  // Show loading
  const loading = document.getElementById('loading');
  const main = document.getElementById('main');
  const progressFill = document.getElementById('progress-fill');
  const progressBar = document.getElementById('progress-bar');
  loading.textContent = `loading ${modeId}...`;
  loading.style.display = '';
  progressBar.style.display = '';
  progressFill.style.width = '0%';
  main.style.display = 'none';

  // Init audio context (don't await resume — it needs a user gesture)
  if (!state.ctx) {
    state.ctx = new AudioContext();
  }

  // Load buffers (decodeAudioData works on suspended contexts)
  state.buffers = await loadMode(modeId, pct => {
    progressFill.style.width = (pct * 100) + '%';
  });

  // Build effects chain
  buildEffects(mode.effects);

  // Restore or init patterns
  if (state.modePatterns[modeId]) {
    state.patterns = state.modePatterns[modeId];
    state.currentPattern = 0;
  } else {
    initPatterns(mode);
  }

  // Rebuild UI
  buildUI(mode);

  // Update mode buttons
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.className = b.dataset.mode === modeId ? 'mode-btn sel' : 'mode-btn';
  });

  loading.style.display = 'none';
  progressBar.style.display = 'none';
  main.style.display = 'flex';

  // Load first pattern into state and auto-play
  const { loadPat } = await import('./engine.js');
  loadPat(state.currentPattern);
  start();
  updateCtl();
}

// ======================== BUILD UI ========================

function buildUI(mode) {
  state.grids = [];
  state.melCells = [];
  state.melActive = {};
  state.melButtons = [];
  state.patButtons = [];
  state.step = 0;

  const cellSize = mode.cellSize || 16;
  const hidden = mode.hideControls || [];

  // Pattern column
  const patCol = document.getElementById('pat-col');
  patCol.innerHTML = '';
  patCol.style.display = hidden.includes('pattern') ? 'none' : '';
  const numPat = mode.numPatterns || 8;
  for (let pi = 0; pi < numPat; pi++) {
    const pb = document.createElement('div');
    pb.className = pi === 0 ? 'pat-btn sel' : 'pat-btn';
    pb.textContent = pi + 1;
    pb.dataset.pi = pi;
    pb.addEventListener('click', function () { selectPat(parseInt(this.dataset.pi)); updateCtl(); });
    patCol.appendChild(pb);
    state.patButtons.push(pb);
  }

  // Main grids
  const stage = document.getElementById('stage');
  stage.innerHTML = '';
  const layoutCols = (mode.mainGridLayout && mode.mainGridLayout.cols) || 2;
  stage.style.gridTemplateColumns = `repeat(${layoutCols}, auto)`;

  for (let gi = 0; gi < mode.mainGrids.length; gi++) {
    const def = mode.mainGrids[gi];
    const panel = document.createElement('div');
    panel.className = 'panel';

    // Instrument bar
    const instBar = document.createElement('div');
    instBar.className = 'inst';
    const gObj = {
      cells: [], active: {}, instrument: def.defaultInstrument,
      buttons: [], rows: def.rows, cols: def.cols,
      freqs: mode.mainFreqs.slice(0, def.rows),
    };

    for (const inst of mode.mainInstruments) {
      const btn = document.createElement('button');
      btn.textContent = inst.label;
      btn.dataset.gi = gi;
      btn.dataset.inst = inst.id;
      if (inst.id === gObj.instrument) btn.className = 'sel';
      btn.addEventListener('click', function () {
        const g = state.grids[parseInt(this.dataset.gi)];
        g.instrument = this.dataset.inst;
        for (const b of g.buttons) b.className = '';
        this.className = 'sel';
      });
      instBar.appendChild(btn);
      gObj.buttons.push(btn);
    }
    panel.appendChild(instBar);

    // Grid cells
    const gridEl = document.createElement('div');
    gridEl.className = 'grid';
    gridEl.style.gridTemplateColumns = `repeat(${def.cols}, ${cellSize}px)`;
    for (let r = 0; r < def.rows; r++) {
      gObj.cells[r] = [];
      for (let c = 0; c < def.cols; c++) {
        const el = document.createElement('div');
        el.className = 'cell';
        el.style.width = cellSize + 'px';
        el.style.height = cellSize + 'px';
        el.dataset.gi = gi;
        el.dataset.r = r;
        el.dataset.c = c;
        el.addEventListener('click', function () {
          toggleMain(parseInt(this.dataset.gi), parseInt(this.dataset.r), parseInt(this.dataset.c));
        });
        gridEl.appendChild(el);
        gObj.cells[r][c] = el;
      }
    }
    panel.appendChild(gridEl);
    stage.appendChild(panel);
    state.grids.push(gObj);
  }

  // Melody section
  const melWrap = document.getElementById('mel-wrap');
  const mib = document.getElementById('mel-inst-bar');
  const melGridsDiv = document.getElementById('mel-grids');
  mib.innerHTML = '';
  melGridsDiv.innerHTML = '';

  if (mode.melodyGrids && mode.melodyGrids.length > 0 && !hidden.includes('melody')) {
    melWrap.style.display = '';
    state.melInstrument = mode.melodyDefaultInstrument || (mode.melodyInstruments[0] && mode.melodyInstruments[0].id) || 'pad';

    for (const inst of (mode.melodyInstruments || [])) {
      const btn = document.createElement('button');
      btn.textContent = inst.label;
      btn.dataset.inst = inst.id;
      if (inst.id === state.melInstrument) btn.className = 'sel';
      btn.addEventListener('click', function () {
        state.melInstrument = this.dataset.inst;
        for (const b of state.melButtons) b.className = '';
        this.className = 'sel';
      });
      mib.appendChild(btn);
      state.melButtons.push(btn);
    }

    const melRows = mode.melodyGrids[0].rows;
    for (let r = 0; r < melRows; r++) state.melCells[r] = [];

    for (let mg = 0; mg < mode.melodyGrids.length; mg++) {
      const mgDef = mode.melodyGrids[mg];
      const gel = document.createElement('div');
      gel.className = 'grid';
      gel.style.gridTemplateColumns = `repeat(${mgDef.cols}, ${cellSize}px)`;
      for (let r = 0; r < mgDef.rows; r++) {
        for (let lc = 0; lc < mgDef.cols; lc++) {
          const gc = mg > 0 ? mode.melodyGrids.slice(0, mg).reduce((s, g) => s + g.cols, 0) + lc : lc;
          const el = document.createElement('div');
          el.className = 'cell';
          el.style.width = cellSize + 'px';
          el.style.height = cellSize + 'px';
          el.dataset.r = r;
          el.dataset.c = gc;
          el.addEventListener('click', function () {
            toggleMel(parseInt(this.dataset.r), parseInt(this.dataset.c));
          });
          gel.appendChild(el);
          state.melCells[r][gc] = el;
        }
      }
      melGridsDiv.appendChild(gel);
    }
  } else {
    melWrap.style.display = 'none';
  }

  // Canon section
  const canonWrap = document.getElementById('canon-wrap');
  const canonModeBar = document.getElementById('canon-mode-bar');
  const canonGridsDiv = document.getElementById('canon-grids');
  canonModeBar.innerHTML = '';
  canonGridsDiv.innerHTML = '';
  state.canonCells = [];
  state.canonButtons = [];

  const hasMelody = mode.melodyGrids && mode.melodyGrids.length > 0 && !hidden.includes('melody');
  if (hasMelody) {
    // Build canon mode buttons (crab, mirror, table)
    for (const cm of ['simple', 'interval', 'crab', 'mirror', 'table']) {
      const btn = document.createElement('button');
      btn.textContent = cm;
      btn.dataset.canon = cm;
      if (cm === state.canonMode) btn.className = 'sel';
      btn.addEventListener('click', function () {
        state.canonMode = this.dataset.canon;
        for (const b of state.canonButtons) b.className = '';
        this.className = 'sel';
        computeCanon();
      });
      canonModeBar.appendChild(btn);
      state.canonButtons.push(btn);
    }

    // Build read-only canon grid (same dimensions as melody)
    const melRows = mode.melodyGrids[0].rows;
    for (let r = 0; r < melRows; r++) state.canonCells[r] = [];

    for (let mg = 0; mg < mode.melodyGrids.length; mg++) {
      const mgDef = mode.melodyGrids[mg];
      const gel = document.createElement('div');
      gel.className = 'grid';
      gel.style.gridTemplateColumns = `repeat(${mgDef.cols}, ${cellSize}px)`;
      for (let r = 0; r < mgDef.rows; r++) {
        for (let lc = 0; lc < mgDef.cols; lc++) {
          const gc = mg > 0 ? mode.melodyGrids.slice(0, mg).reduce((s, g) => s + g.cols, 0) + lc : lc;
          const el = document.createElement('div');
          el.className = 'cell';
          el.style.width = cellSize + 'px';
          el.style.height = cellSize + 'px';
          el.style.cursor = 'default';
          gel.appendChild(el);
          state.canonCells[r][gc] = el;
        }
      }
      canonGridsDiv.appendChild(gel);
    }

    canonWrap.style.display = state.canonEnabled ? 'flex' : 'none';
  } else {
    canonWrap.style.display = 'none';
  }

  // Controls visibility
  document.getElementById('cycleToggle').parentElement.querySelector('#cycleLen') && (
    document.getElementById('cycleToggle').style.display = hidden.includes('cycle') ? 'none' : '',
    document.getElementById('cycleLen').style.display = hidden.includes('cycle') ? 'none' : '',
    document.getElementById('cycleDisp').style.display = hidden.includes('cycle') ? 'none' : ''
  );

  const melNWrap = document.getElementById('melN');
  if (melNWrap) melNWrap.style.display = hidden.includes('melody') ? 'none' : '';

  const canonToggle = document.getElementById('canonToggle');
  if (canonToggle) canonToggle.style.display = hidden.includes('melody') ? 'none' : '';
}

// ======================== CELL TOGGLES ========================

function toggleMain(gi, r, c) {
  const g = state.grids[gi];
  const k = r + '-' + c;
  if (!g.active[k]) {
    g.active[k] = { offset: state.step - (g.cols - 1), vol: 0.5 };
    applyStyle(g.cells[r][c], STYLES.HALF);
  } else if (g.active[k].vol < 1) {
    g.active[k].vol = 1;
    applyStyle(g.cells[r][c], STYLES.FULL);
  } else {
    delete g.active[k];
    applyStyle(g.cells[r][c], STYLES.OFF);
  }
  if (!state.playing) start();
  updateCtl();
}

function toggleMel(r, c) {
  const k = r + '-' + c;
  if (!state.melActive[k]) {
    state.melActive[k] = { vol: 0.5 };
    applyStyle(state.melCells[r][c], STYLES.HALF);
  } else if (state.melActive[k].vol < 1) {
    state.melActive[k].vol = 1;
    applyStyle(state.melCells[r][c], STYLES.FULL);
  } else {
    delete state.melActive[k];
    applyStyle(state.melCells[r][c], STYLES.OFF);
  }
  computeCanon();
  if (!state.playing) start();
  updateCtl();
}

// ======================== CONTROLS ========================

function updateCtl() {
  const el = document.getElementById('ctl');
  let any = false;
  for (const g of state.grids) if (Object.keys(g.active).length > 0) any = true;
  if (Object.keys(state.melActive).length > 0) any = true;
  if (state.playing) el.innerHTML = '<span id="stop-btn">stop</span>';
  else if (any) el.innerHTML = '<span id="play-btn">play</span>';
  else el.innerHTML = '';

  const stopBtn = document.getElementById('stop-btn');
  if (stopBtn) stopBtn.addEventListener('click', () => { stop(); updateCtl(); });
  const playBtn = document.getElementById('play-btn');
  if (playBtn) playBtn.addEventListener('click', () => { start(); updateCtl(); });
}

// ======================== INIT ========================

export function initApp() {
  // Mode column
  const modeCol = document.getElementById('mode-col');
  for (const id of MODE_IDS) {
    const btn = document.createElement('button');
    btn.className = 'mode-btn';
    btn.textContent = id;
    btn.dataset.mode = id;
    btn.addEventListener('click', () => switchMode(id));
    modeCol.appendChild(btn);
  }

  // Controls wiring
  const slider = document.getElementById('speed');
  const bpmEl = document.getElementById('bpm');
  function updateBpm() {
    state.stepMs = 550 - parseInt(slider.value);
    bpmEl.textContent = Math.round(60000 / state.stepMs) + ' bpm';
  }
  updateBpm();
  slider.addEventListener('input', () => { updateBpm(); restartTimer(); });

  const cycleSlider = document.getElementById('cycleLen');
  const cycleDisp = document.getElementById('cycleDisp');
  function updateCycleDisp() {
    state.cycleSteps = parseInt(cycleSlider.value);
    cycleDisp.textContent = state.cycleSteps + 'st';
  }
  updateCycleDisp();
  cycleSlider.addEventListener('input', updateCycleDisp);

  const melSlider = document.getElementById('melN');
  const melNDisp = document.getElementById('melNDisp');
  function updateMelN() {
    state.melN = parseInt(melSlider.value);
    melNDisp.textContent = '\u00d7' + state.melN;
  }
  updateMelN();
  melSlider.addEventListener('input', updateMelN);

  document.getElementById('cycleToggle').addEventListener('click', () => {
    state.autoCycle = !state.autoCycle;
    document.getElementById('cycleToggle').textContent = state.autoCycle ? 'cycle on' : 'cycle off';
  });

  document.getElementById('canonToggle').addEventListener('click', () => {
    state.canonEnabled = !state.canonEnabled;
    document.getElementById('canonToggle').textContent = state.canonEnabled ? 'canon on' : 'canon off';
    const canonWrap = document.getElementById('canon-wrap');
    canonWrap.style.display = state.canonEnabled ? 'flex' : 'none';
    computeCanon();
  });

  document.getElementById('btn-reset').addEventListener('click', () => { resetAll(); updateCtl(); });
  document.getElementById('btn-random').addEventListener('click', () => { randomize(); updateCtl(); });
  document.getElementById('btn-save').addEventListener('click', saveSong);
  document.getElementById('btn-load').addEventListener('click', () => {
    loadSong(s => {
      if (s.bpm) { state.stepMs = Math.round(60000 / s.bpm); slider.value = 550 - state.stepMs; updateBpm(); }
      if (s.cycleSteps) { state.cycleSteps = s.cycleSteps; cycleSlider.value = s.cycleSteps; updateCycleDisp(); }
      if (typeof s.autoCycle === 'boolean') { state.autoCycle = s.autoCycle; document.getElementById('cycleToggle').textContent = s.autoCycle ? 'cycle on' : 'cycle off'; }
      if (s.melN) { state.melN = s.melN; melSlider.value = s.melN; updateMelN(); }
      updateCtl();
    });
  });

  // Auto-start default mode
  switchMode('original');
}
