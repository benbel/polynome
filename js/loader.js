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
  let loaded = 0;
  const total = manifest.instruments.length;

  await Promise.all(manifest.instruments.map(async (inst) => {
    try {
      buffers[inst.id] = await loadInstrument(modeId, inst, ctx);
    } catch (e) {
      buffers[inst.id] = [];
      console.warn(`${modeId}/${inst.id}: ${e.message}`);
    }
    loaded++;
    if (onProgress) onProgress(loaded / total);
  }));

  return { config, buffers };
}

async function loadInstrument(modeId, inst, ctx) {
  const resp = await fetch(`assets/${modeId}/${inst.sprite}`);
  if (!resp.ok) throw new Error(`${resp.status} ${inst.sprite}`);
  const decoded = await ctx.decodeAudioData(await resp.arrayBuffer());
  return sliceSprite(decoded, inst, ctx);
}

function sliceSprite(decoded, inst, ctx) {
  const scale = decoded.duration / inst.spriteDuration;
  const rate = decoded.sampleRate;
  const channels = decoded.numberOfChannels;

  return inst.offsets.map(([start, duration]) => {
    const from = Math.round(start * scale * rate);
    const length = Math.min(Math.round(duration * scale * rate), decoded.length - from);
    const out = ctx.createBuffer(channels, Math.max(1, length), rate);
    const chunk = new Float32Array(Math.max(1, length));
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
