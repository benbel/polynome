export async function loadModeList() {
  const resp = await fetch('assets/modes.json');
  if (!resp.ok) throw new Error(`no assets/modes.json (${resp.status})`);
  return resp.json();
}

export async function loadMode(modeId, ctx, onProgress) {
  const resp = await fetch(`assets/${modeId}/manifest.json`);
  if (!resp.ok) throw new Error(`no manifest for ${modeId} (${resp.status})`);
  const manifest = await resp.json();

  const config = buildConfig(manifest);
  const buffers = {};
  const errors = [];
  let loaded = 0;
  const total = manifest.instruments.length;

  await Promise.all(manifest.instruments.map(async (inst) => {
    try {
      buffers[inst.id] = await loadInstrument(modeId, inst, ctx);
    } catch (e) {
      buffers[inst.id] = [];
      errors.push(`${inst.id}: ${e.message}`);
      console.warn(`${modeId}/${inst.id}: ${e.message}`);
    }
    loaded++;
    if (onProgress) onProgress(loaded / total);
  }));

  return { config, buffers, errors };
}

async function loadInstrument(modeId, inst, ctx) {
  const resp = await fetch(`assets/${modeId}/${inst.sprite}`);
  if (!resp.ok) throw new Error(`${resp.status} ${inst.sprite}`);
  const decoded = await decode(ctx, await resp.arrayBuffer(), inst.sprite);
  return sliceSprite(decoded, inst, ctx);
}

// Safari only grew the promise form of decodeAudioData in iOS 14.5, and older
// WebKit rejects with a bare null instead of an Error.
function decode(ctx, data, name) {
  return new Promise((resolve, reject) => {
    const fail = (e) => reject(new Error((e && e.message) || `cannot decode ${name}`));
    let ret;
    try {
      ret = ctx.decodeAudioData(data, resolve, fail);
    } catch (e) {
      fail(e);
      return;
    }
    if (ret && typeof ret.then === 'function') ret.then(resolve, fail);
  });
}

// Lossy codecs do not decode to exactly the samples that were encoded: MP3
// carries encoder delay, and how much of it a browser trims varies. Sprite
// segments are separated by silence, so widen every slice into that silence
// rather than trusting the offsets to land sample-accurately -- a slice cut
// short would clip the note's tail into a click.
const SLICE_LEAD = 0.02;
const SLICE_TAIL = 0.15;

function sliceSprite(decoded, inst, ctx) {
  const rate = decoded.sampleRate;
  const channels = decoded.numberOfChannels;

  return inst.offsets.map(([start, duration]) => {
    const from = Math.max(0, Math.round((start - SLICE_LEAD) * rate));
    const to = Math.min(decoded.length, Math.round((start + duration + SLICE_TAIL) * rate));
    const length = Math.max(1, to - from);
    const out = ctx.createBuffer(channels, length, rate);
    const chunk = new Float32Array(length);
    for (let ch = 0; ch < channels; ch++) {
      decoded.copyFromChannel(chunk, ch, from);
      out.copyToChannel(chunk, ch);
    }
    return out;
  });
}

function buildConfig(manifest) {
  const config = { ...manifest };
  config.mainInstruments = manifest.instruments
    .filter(i => i.type === 'main')
    .map(i => ({ id: i.id, label: i.label }));
  config.melodyInstruments = manifest.instruments
    .filter(i => i.type === 'melody')
    .map(i => ({ id: i.id, label: i.label }));
  delete config.instruments;
  return config;
}

export function bufferCount(buffers) {
  let n = 0;
  for (const id of Object.keys(buffers || {})) {
    for (const b of buffers[id] || []) if (b) n++;
  }
  return n;
}
