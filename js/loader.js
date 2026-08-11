// loader.js — Fetch + decode audio buffers from pre-rendered assets.
// Falls back to in-browser OfflineAudioContext synthesis if assets not found.

import { state } from './engine.js';

// ======================== DSP HELPERS (for synthesis fallback) ========================

function normalize(buf, peak) {
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    let m = 0;
    for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > m) m = a; }
    if (m > 0) { const s = peak / m; for (let i = 0; i < d.length; i++) d[i] *= s; }
  }
}

function bitcrush(buf, bits, dn) {
  const s = Math.pow(2, -(bits - 1));
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    let h = 0;
    for (let i = 0; i < d.length; i++) {
      if (i % dn === 0) h = Math.round(d[i] / s) * s;
      d[i] = h;
    }
  }
}

function tapeSat(buf, dr) {
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < d.length; i++) {
      const x = d[i] * dr;
      d[i] = x > 0 ? Math.tanh(x * 1.2) / 1.2 : Math.tanh(x * 0.9) / 0.9;
    }
  }
}

function fadeIn(buf, ms) {
  const n = Math.round(buf.sampleRate * ms / 1000);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < n && i < d.length; i++) d[i] *= i / n;
  }
}

function fadeOut(buf, ms) {
  const n = Math.round(buf.sampleRate * ms / 1000);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    const s = d.length - n;
    for (let i = Math.max(0, s); i < d.length; i++) d[i] *= (d.length - i) / n;
  }
}

function mkDist(a) {
  const n = 1024, c = new Float32Array(n);
  for (let i = 0; i < n; i++) { const x = (i * 2 / n) - 1; c[i] = Math.tanh(x * a); }
  return c;
}

function mkFold(f) {
  const n = 2048, c = new Float32Array(n);
  for (let i = 0; i < n; i++) { const x = ((i * 2 / n) - 1) * f; c[i] = (4 / Math.PI) * Math.asin(Math.sin(Math.PI * x / 2)); }
  return c;
}

function nBuf(oc, dur) {
  const l = Math.round(oc.sampleRate * dur);
  const b = oc.createBuffer(1, l, oc.sampleRate);
  const d = b.getChannelData(0);
  for (let i = 0; i < l; i++) d[i] = Math.random() * 2 - 1;
  return b;
}

function sNoise(oc, dur) {
  const l = Math.round(oc.sampleRate * dur);
  const b = oc.createBuffer(2, l, oc.sampleRate);
  for (let c = 0; c < 2; c++) { const d = b.getChannelData(c); for (let i = 0; i < l; i++) d[i] = Math.random() * 2 - 1; }
  return b;
}

// ======================== ASSET LOADER ========================

export async function loadMode(modeName, onProgress) {
  const ctx = state.ctx;

  // Try loading pre-rendered assets first
  try {
    const resp = await fetch(`assets/${modeName}/manifest.json`);
    if (resp.ok) {
      const manifest = await resp.json();
      return await loadFromManifest(modeName, manifest, ctx, onProgress);
    }
  } catch (e) {
    // manifest not found, fall through to synthesis
  }

  // Fallback: check if mode has synthesis renderers
  const mode = await import(`./modes/${modeName}.js`);
  if (mode.default.renderers) {
    return await synthesizeBuffers(mode.default, ctx, onProgress);
  }

  return {};
}

async function loadFromManifest(modeName, manifest, ctx, onProgress) {
  const buffers = {};
  const promises = [];
  let loaded = 0;
  let total = 0;

  for (const inst of manifest.instruments) {
    buffers[inst.id] = new Array(inst.pitchCount);
    total += inst.pitchCount;
  }

  for (const inst of manifest.instruments) {
    for (let i = 0; i < inst.pitchCount; i++) {
      const url = `assets/${modeName}/${inst.id}_${i}.ogg`;
      promises.push(
        fetch(url)
          .then(r => r.arrayBuffer())
          .then(ab => ctx.decodeAudioData(ab))
          .then(buf => {
            buffers[inst.id][i] = buf;
            loaded++;
            if (onProgress) onProgress(loaded / total);
          })
      );
    }
  }

  await Promise.all(promises);
  return buffers;
}

// ======================== SYNTHESIS FALLBACK ========================
// Used when pre-rendered assets aren't available (e.g. texture mode during development)

async function synthesizeBuffers(mode, ctx, onProgress) {
  const SR = 44100;
  const renderers = mode.renderers;
  const crush = mode.crush || {};
  const durations = mode.durations || {};
  const fadeConfig = mode.fadeConfig || {};
  const buffers = {};
  const promises = [];
  let total = 0, loaded = 0;

  const allInsts = Object.keys(renderers);
  const melInsts = (mode.melodyInstruments || []).map(i => i.id);

  // Count total
  for (const id of allInsts) {
    const freqs = melInsts.includes(id) ? mode.melodyFreqs : mode.mainFreqs;
    total += freqs.length;
  }

  for (const id of allInsts) {
    const freqs = melInsts.includes(id) ? mode.melodyFreqs : mode.mainFreqs;
    buffers[id] = new Array(freqs.length);

    for (let fi = 0; fi < freqs.length; fi++) {
      const dur = durations[id] || 2.0;
      const len = Math.round(SR * dur);
      const oc = new OfflineAudioContext(2, len, SR);
      renderers[id](freqs[fi], oc);
      const cr = crush[id] || { bits: 16, down: 1, tape: 1 };
      const isMel = melInsts.includes(id);
      const fc = fadeConfig[id] || {};

      promises.push(
        oc.startRendering().then(buf => {
          if (cr.tape > 1) tapeSat(buf, cr.tape);
          if (cr.down > 1 || cr.bits < 16) bitcrush(buf, cr.bits, cr.down);
          normalize(buf, 0.85);
          fadeIn(buf, isMel ? 8 : (fc.fadeIn || 30));
          fadeOut(buf, isMel ? 80 : (fc.fadeOut || 60));
          buffers[id][fi] = buf;
          loaded++;
          if (onProgress) onProgress(loaded / total);
        })
      );
    }
  }

  await Promise.all(promises);
  return buffers;
}

export { normalize, bitcrush, tapeSat, fadeIn, fadeOut, mkDist, mkFold, nBuf, sNoise };
