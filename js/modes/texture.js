// texture mode — Deep, gritty electronic (Superpoze, Scratch Massive, Fakear)
// Includes OfflineAudioContext renderers as synthesis fallback when assets aren't available.

import { mkDist, mkFold, nBuf, sNoise } from '../loader.js';

const FREQS = [262, 220, 196, 165, 131, 110, 98, 82, 73, 65, 55, 49, 41, 33, 27, 21];
const MEL_FREQS = [523, 440, 392, 330, 262, 220, 196, 165];

// ======================== RENDERERS ========================

function renderSub(f0, oc) {
  const f = f0 * 0.25, dur = 2.0;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.9, 0); out.gain.linearRampToValueAtTime(0.7, 0.06);
  out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1); out.connect(oc.destination);
  const lp = oc.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 12;
  lp.frequency.setValueAtTime(f * 16, 0); lp.frequency.exponentialRampToValueAtTime(f * 2, 0.3);
  lp.frequency.exponentialRampToValueAtTime(f * 1.3, 1.0); lp.connect(out);
  const lp2 = oc.createBiquadFilter(); lp2.type = 'lowpass'; lp2.frequency.value = f * 8; lp2.Q.value = 2; lp2.connect(lp);
  const w = oc.createOscillator(); w.type = 'triangle'; w.frequency.value = 0.9;
  const wG = oc.createGain(); wG.gain.value = f * 1.5; w.connect(wG); wG.connect(lp.frequency);
  w.start(0); w.stop(dur);
  const n = 2048, fc = new Float32Array(n);
  for (let i = 0; i < n; i++) { const x = ((i * 2 / n) - 1) * 3; const fd = (4 / Math.PI) * Math.asin(Math.sin(Math.PI * x / 2)); fc[i] = x > 0 ? fd * 0.85 : fd; }
  const fold = oc.createWaveShaper(); fold.curve = fc; fold.oversample = '4x';
  const sat = oc.createWaveShaper(); sat.curve = mkDist(6); sat.oversample = '4x';
  fold.connect(sat); sat.connect(lp2);
  const mx = oc.createGain(); mx.gain.value = 0.4; mx.connect(fold);
  const dts = [-28, -18, -8, -3, 3, 8, 18, 28], pns = [-0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9];
  const dR = [0.13, 0.09, 0.17, 0.11, 0.14, 0.08, 0.16, 0.1], dA = [1.5, 2.2, 1.8, 2.5, 1.6, 2.0, 1.9, 2.3];
  for (let i = 0; i < 8; i++) {
    const o = oc.createOscillator(); o.type = i < 4 ? 'sawtooth' : 'square'; o.frequency.value = f; o.detune.value = dts[i];
    const dr = oc.createOscillator(); dr.type = 'sine'; dr.frequency.value = dR[i];
    const dg = oc.createGain(); dg.gain.value = dA[i]; dr.connect(dg); dg.connect(o.detune);
    const g = oc.createGain(); g.gain.value = i < 4 ? 0.14 : 0.1;
    const p = oc.createStereoPanner(); p.pan.value = pns[i];
    o.connect(g).connect(p).connect(mx); o.start(0); o.stop(dur); dr.start(0); dr.stop(dur);
  }
  const sub = oc.createOscillator(); sub.type = 'sine'; sub.frequency.value = f;
  const sB = oc.createGain(); sB.gain.value = 0.5; sub.connect(sB).connect(lp2);
  sub.start(0); sub.stop(dur);
  const hm = oc.createOscillator(); hm.type = 'sine'; hm.frequency.value = 50;
  const hG = oc.createGain(); hG.gain.value = 0.06; hm.connect(hG).connect(out);
  hm.start(0); hm.stop(dur);
  const nb = nBuf(oc, 0.5); const ns = oc.createBufferSource(); ns.buffer = nb;
  const nf = oc.createWaveShaper(); nf.curve = mkFold(2);
  const nl = oc.createBiquadFilter(); nl.type = 'lowpass'; nl.frequency.value = f * 8; nl.Q.value = 3;
  const ng = oc.createGain(); ng.gain.setValueAtTime(0.35, 0); ng.gain.exponentialRampToValueAtTime(0.001, 0.4);
  ns.connect(nf).connect(nl).connect(ng).connect(sat); ns.start(0);
}

function renderFm(f0, oc) {
  const f = f0, dur = 2.2;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.8, 0); out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1); out.connect(oc.destination);
  const lp = oc.createBiquadFilter(); lp.type = 'lowpass';
  lp.frequency.setValueAtTime(f * 6, 0); lp.frequency.exponentialRampToValueAtTime(f * 1.2, 0.8); lp.Q.value = 5; lp.connect(out);
  const ct = Math.min(1 / (f * 1.5), 0.05);
  const cm = oc.createDelay(0.05); cm.delayTime.value = ct;
  const cf = oc.createGain(); cf.gain.value = 0.55;
  const cl = oc.createBiquadFilter(); cl.type = 'lowpass'; cl.frequency.value = f * 3;
  cl.connect(cm); cm.connect(cf).connect(cl);
  const ci = oc.createGain(); ci.gain.value = 0.25; ci.connect(cm); cm.connect(lp);
  const fc2 = new Float32Array(2048);
  for (let i = 0; i < 2048; i++) { const x = ((i * 2 / 2048) - 1) * 1.8; const fd = (4 / Math.PI) * Math.asin(Math.sin(Math.PI * x / 2)); fc2[i] = x > 0 ? fd * 0.8 : fd * 1.1; }
  const fold = oc.createWaveShaper(); fold.curve = fc2; fold.oversample = '2x';
  fold.connect(lp); fold.connect(ci);
  const car = oc.createOscillator(); car.type = 'sine'; car.frequency.value = f;
  const cg = oc.createGain(); cg.gain.value = 0.55;
  const cp = oc.createStereoPanner(); cp.pan.value = -0.35;
  car.connect(cg).connect(cp).connect(fold);
  const car2 = oc.createOscillator(); car2.type = 'sine'; car2.frequency.value = f * 1.005;
  const cg2 = oc.createGain(); cg2.gain.value = 0.4;
  const cp2 = oc.createStereoPanner(); cp2.pan.value = 0.35;
  car2.connect(cg2).connect(cp2).connect(fold);
  const m1 = oc.createOscillator(); m1.type = 'sine'; m1.frequency.value = f * 1.4142;
  const m1g = oc.createGain(); m1g.gain.setValueAtTime(f * 4, 0); m1g.gain.exponentialRampToValueAtTime(f * 0.2, 0.9);
  m1.connect(m1g); m1g.connect(car.frequency); m1g.connect(car2.frequency);
  const m2 = oc.createOscillator(); m2.type = 'sine'; m2.frequency.value = f * 1.618;
  const m2g = oc.createGain(); m2g.gain.setValueAtTime(f * 2.5, 0); m2g.gain.exponentialRampToValueAtTime(f * 0.15, 1.0);
  m2.connect(m2g); m2g.connect(car.frequency);
  const m3 = oc.createOscillator(); m3.type = 'triangle'; m3.frequency.value = f * 0.5;
  const m3g = oc.createGain(); m3g.gain.setValueAtTime(f * 1.2, 0); m3g.gain.exponentialRampToValueAtTime(f * 0.5, 1.5);
  m3.connect(m3g); m3g.connect(car.frequency); m3g.connect(car2.frequency);
  car.start(0); car.stop(dur); car2.start(0); car2.stop(dur);
  m1.start(0); m1.stop(dur); m2.start(0); m2.stop(dur); m3.start(0); m3.stop(dur);
  const hm = oc.createOscillator(); hm.type = 'sine'; hm.frequency.value = 50;
  const hG = oc.createGain(); hG.gain.setValueAtTime(0.03, 0); hG.gain.exponentialRampToValueAtTime(0.001, dur * 0.6);
  hm.connect(hG).connect(out); hm.start(0); hm.stop(dur);
}

function renderGlass(f0, oc) {
  const f = f0, dur = 2.6;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.8, 0); out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1); out.connect(oc.destination);
  const mLp = oc.createBiquadFilter(); mLp.type = 'lowpass'; mLp.frequency.value = f * 8; mLp.Q.value = 1; mLp.connect(out);
  const fF = [f, f*1.5, f*2.76, f*3.51, f*4.23, f*5.87, f*7.1];
  const fQ = [10, 8, 7, 6, 5, 4, 3], fA = [.45, .25, .25, .18, .12, .08, .05];
  const fP = [-.5, -.2, .1, .35, -.6, .55, -.8], fD = [2.2, 1.8, 1.5, 1.2, .8, .5, .3];
  const fN = [];
  for (let i = 0; i < 7; i++) {
    const bp = oc.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = fF[i]; bp.Q.value = fQ[i];
    const fg = oc.createGain(); fg.gain.setValueAtTime(fA[i], 0); fg.gain.exponentialRampToValueAtTime(0.001, fD[i]);
    const fp = oc.createStereoPanner(); fp.pan.value = fP[i];
    bp.connect(fg).connect(fp).connect(mLp); fN.push(bp);
  }
  const nb = nBuf(oc, 0.02); const ns = oc.createBufferSource(); ns.buffer = nb;
  const eG = oc.createGain(); eG.gain.value = 0.8;
  ns.connect(eG);
  const d1 = Math.min(1 / f, 0.1);
  const c1 = oc.createDelay(0.1); c1.delayTime.value = d1;
  const c1f = oc.createGain(); c1f.gain.value = 0.72;
  const c1l = oc.createBiquadFilter(); c1l.type = 'lowpass'; c1l.frequency.value = f * 6;
  c1.connect(c1f).connect(c1l).connect(c1);
  const c1o = oc.createGain(); c1o.gain.setValueAtTime(0.35, 0); c1o.gain.exponentialRampToValueAtTime(0.001, 2.0);
  c1.connect(c1o); for (let i = 0; i < 4; i++) c1o.connect(fN[i]);
  eG.connect(c1);
  const ns2 = oc.createBufferSource(); ns2.buffer = nb;
  const e2 = oc.createGain(); e2.gain.value = 0.4;
  ns2.connect(e2); for (let i = 0; i < 7; i++) e2.connect(fN[i]);
  ns.start(0); ns2.start(0);
  const bd = oc.createOscillator(); bd.type = 'sine'; bd.frequency.value = f;
  const bF = oc.createWaveShaper(); bF.curve = mkFold(1.5);
  const bG = oc.createGain(); bG.gain.setValueAtTime(0.25, 0); bG.gain.exponentialRampToValueAtTime(0.001, 1.8);
  bd.connect(bF).connect(bG); for (let i = 0; i < 5; i++) bG.connect(fN[i]);
  bd.start(0); bd.stop(2);
}

function renderTape(f0, oc) {
  const f = f0, dur = 2.4;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.001, 0); out.gain.linearRampToValueAtTime(0.8, 0.04);
  out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1); out.connect(oc.destination);
  const lp = oc.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = f * 3.5; lp.Q.value = 2; lp.connect(out);
  const vw = oc.createBiquadFilter(); vw.type = 'peaking'; vw.frequency.value = f * 2.2; vw.Q.value = 4; vw.gain.value = 6; vw.connect(lp);
  const cv = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) { const x = (i * 2 / 1024) - 1; cv[i] = x > 0 ? Math.tanh(x * 2.5) / 2.5 * 2 : Math.tanh(x * 1.5) / 1.5; }
  const st = oc.createWaveShaper(); st.curve = cv; st.oversample = '4x'; st.connect(vw);
  const dt = [-15, -8, -3, 3, 8, 15], pV = [-.7, -.35, -.1, .1, .35, .7];
  const wR = [.6, .45, .7, .55, .8, .5], wD = [3, 4.5, 2.5, 5, 3.5, 4];
  for (let i = 0; i < 6; i++) {
    const o = oc.createOscillator(); o.type = 'sine'; o.frequency.value = f; o.detune.value = dt[i];
    const wo = oc.createOscillator(); wo.type = 'sine'; wo.frequency.value = wR[i];
    const wG = oc.createGain(); wG.gain.value = wD[i]; wo.connect(wG); wG.connect(o.detune);
    const g = oc.createGain(); g.gain.value = 0.18;
    const p = oc.createStereoPanner(); p.pan.value = pV[i];
    o.connect(g).connect(p).connect(st); o.start(0); o.stop(dur); wo.start(0); wo.stop(dur);
  }
  const hs = sNoise(oc, dur); const hS = oc.createBufferSource(); hS.buffer = hs;
  const hL = oc.createBiquadFilter(); hL.type = 'lowpass'; hL.frequency.value = f * 4;
  const hH = oc.createBiquadFilter(); hH.type = 'highpass'; hH.frequency.value = f * 0.5;
  const hG = oc.createGain(); hG.gain.value = 0.06;
  hS.connect(hH).connect(hL).connect(hG).connect(lp); hS.start(0);
}

function renderDust(f0, oc) {
  const f = f0 * 0.5, dur = 1.2;
  const out = oc.createGain(); out.gain.value = 0.9; out.connect(oc.destination);
  const mL = oc.createBiquadFilter(); mL.type = 'lowpass'; mL.frequency.value = f * 8; mL.Q.value = 1; mL.connect(out);
  const kD = Math.min(1 / f, 0.05);
  const ks = oc.createDelay(0.05); ks.delayTime.value = kD;
  const kF = oc.createGain(); kF.gain.value = 0.85;
  const kL = oc.createBiquadFilter(); kL.type = 'lowpass'; kL.frequency.value = f * 3;
  ks.connect(kF).connect(kL).connect(ks);
  const kO = oc.createGain(); kO.gain.setValueAtTime(0.35, 0); kO.gain.exponentialRampToValueAtTime(0.001, 0.8);
  ks.connect(kO).connect(mL);
  const cR = [1, 1.34, 1.87], cF = [.65, .58, .5], cP = [0, -.5, .5], cO = [];
  for (let ci = 0; ci < 3; ci++) {
    const ct = Math.min(1 / (f * cR[ci]), 0.05);
    const cd = oc.createDelay(0.05); cd.delayTime.value = ct;
    const cf = oc.createGain(); cf.gain.value = cF[ci];
    const cl = oc.createBiquadFilter(); cl.type = 'lowpass'; cl.frequency.value = f * (4 - ci);
    cd.connect(cf).connect(cl).connect(cd);
    const co = oc.createGain(); co.gain.setValueAtTime(0.2, 0); co.gain.exponentialRampToValueAtTime(0.001, 0.4 + ci * 0.1);
    const cp = oc.createStereoPanner(); cp.pan.value = cP[ci];
    cd.connect(co).connect(cp).connect(mL); cO.push(cd);
  }
  for (let i = 0; i < 12; i++) {
    const gs = oc.createBufferSource(); gs.buffer = nBuf(oc, 0.002 + Math.random() * 0.01);
    const gB = oc.createBiquadFilter(); gB.type = 'bandpass'; gB.frequency.value = f * (0.5 + Math.random() * 6); gB.Q.value = 2 + Math.random() * 10;
    const gF = oc.createWaveShaper(); gF.curve = mkFold(1.2 + Math.random() * 1.5);
    const gG = oc.createGain(); gG.gain.value = 0.1 + Math.random() * 0.2;
    const gP = oc.createStereoPanner(); gP.pan.value = (Math.random() - 0.5) * 1.6;
    gs.connect(gB).connect(gF).connect(gG).connect(gP).connect(mL);
    const tgt = i % 4; if (tgt < 3) gG.connect(cO[tgt]); else gG.connect(ks);
    gs.start(Math.random() * 0.025);
  }
  const tl = sNoise(oc, 0.6); const tS = oc.createBufferSource(); tS.buffer = tl;
  const tL = oc.createBiquadFilter(); tL.type = 'lowpass'; tL.frequency.value = f * 2.5; tL.Q.value = 7;
  const tF = oc.createWaveShaper(); tF.curve = mkFold(1.5);
  const tG = oc.createGain(); tG.gain.setValueAtTime(0.2, 0); tG.gain.exponentialRampToValueAtTime(0.001, 0.5);
  tS.connect(tL).connect(tF).connect(tG).connect(mL); tS.start(0);
}

function renderPad(f0, oc) {
  const f = f0, dur = 3.5;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.001, 0); out.gain.linearRampToValueAtTime(0.8, 0.8);
  out.gain.setValueAtTime(0.8, dur - 1.2); out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1);
  out.connect(oc.destination);
  const lp = oc.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = f * 3; lp.Q.value = 6; lp.connect(out);
  const l1 = oc.createOscillator(); l1.type = 'triangle'; l1.frequency.value = 0.2;
  const l1G = oc.createGain(); l1G.gain.value = f * 2; l1.connect(l1G); l1G.connect(lp.frequency);
  l1.start(0); l1.stop(dur);
  const pk = oc.createBiquadFilter(); pk.type = 'peaking'; pk.frequency.value = f * 2.5; pk.Q.value = 3; pk.gain.value = 6; pk.connect(lp);
  const fold = oc.createWaveShaper(); fold.curve = mkFold(1.5); fold.oversample = '2x'; fold.connect(pk);
  const dts = [-25, -18, -10, -4, 4, 10, 18, 25], pns = [-.9, -.65, -.35, -.1, .1, .35, .65, .9];
  for (let i = 0; i < 8; i++) {
    const o = oc.createOscillator(); o.type = 'sawtooth'; o.frequency.value = f; o.detune.value = dts[i];
    const g = oc.createGain(); g.gain.value = 0.08;
    const p = oc.createStereoPanner(); p.pan.value = pns[i];
    o.connect(g).connect(p).connect(fold); o.start(0); o.stop(dur);
  }
  const sub = oc.createOscillator(); sub.type = 'sine'; sub.frequency.value = f;
  const sg = oc.createGain(); sg.gain.value = 0.18; sub.connect(sg).connect(lp);
  sub.start(0); sub.stop(dur);
}

function renderOrgan(f0, oc) {
  const f = f0, dur = 2.8;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.001, 0); out.gain.linearRampToValueAtTime(0.8, 0.06);
  out.gain.setValueAtTime(0.8, dur - 0.8); out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1);
  out.connect(oc.destination);
  const cv = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) { const x = (i * 2 / 1024) - 1; cv[i] = x > 0 ? Math.tanh(x * 1.8) / 1.8 * 1.5 : Math.tanh(x * 1.3) / 1.3; }
  const sat = oc.createWaveShaper(); sat.curve = cv; sat.oversample = '2x';
  const trem = oc.createGain(); trem.gain.value = 1; trem.connect(sat).connect(out);
  const rotL = oc.createOscillator(); rotL.type = 'sine'; rotL.frequency.value = 5.8;
  const rotG = oc.createGain(); rotG.gain.value = 0.2; rotL.connect(rotG); rotG.connect(trem.gain);
  rotL.start(0); rotL.stop(dur);
  const drawbars = [1, 2, 3, 4, 6, 8], dbAmps = [0.8, 0.6, 0.3, 0.2, 0.15, 0.1];
  for (let i = 0; i < drawbars.length; i++) {
    const o = oc.createOscillator(); o.type = 'sine'; o.frequency.value = f * drawbars[i];
    const g = oc.createGain(); g.gain.value = dbAmps[i] * 0.15;
    o.connect(g).connect(trem); o.start(0); o.stop(dur);
  }
  const kc = oc.createBufferSource(); kc.buffer = nBuf(oc, 0.005);
  const kcg = oc.createGain(); kcg.gain.value = 0.08;
  const kch = oc.createBiquadFilter(); kch.type = 'highpass'; kch.frequency.value = 2000;
  kc.connect(kch).connect(kcg).connect(out); kc.start(0);
}

function renderPiano(f0, oc) {
  const f = f0, dur = 2.4;
  const out = oc.createGain();
  out.gain.setValueAtTime(0.9, 0); out.gain.linearRampToValueAtTime(0.7, 0.01);
  out.gain.exponentialRampToValueAtTime(0.001, dur - 0.1); out.connect(oc.destination);
  const lp = oc.createBiquadFilter(); lp.type = 'lowpass';
  lp.frequency.setValueAtTime(f * 6, 0); lp.frequency.exponentialRampToValueAtTime(f * 2, 0.5); lp.Q.value = 1; lp.connect(out);
  const c1d = Math.min(1 / f, 0.05);
  const c1 = oc.createDelay(0.05); c1.delayTime.value = c1d;
  const c1f = oc.createGain(); c1f.gain.value = 0.92;
  const c1l = oc.createBiquadFilter(); c1l.type = 'lowpass'; c1l.frequency.value = f * 4;
  c1.connect(c1f).connect(c1l).connect(c1);
  const c1o = oc.createGain(); c1o.gain.setValueAtTime(0.4, 0); c1o.gain.exponentialRampToValueAtTime(0.001, dur - 0.2);
  c1.connect(c1o).connect(lp);
  const c2d = Math.min(1 / (f * 2.01), 0.05);
  const c2 = oc.createDelay(0.05); c2.delayTime.value = c2d;
  const c2f = oc.createGain(); c2f.gain.value = 0.88;
  const c2l = oc.createBiquadFilter(); c2l.type = 'lowpass'; c2l.frequency.value = f * 3;
  c2.connect(c2f).connect(c2l).connect(c2);
  const c2o = oc.createGain(); c2o.gain.setValueAtTime(0.2, 0); c2o.gain.exponentialRampToValueAtTime(0.001, dur * 0.5);
  c2.connect(c2o).connect(lp);
  const exc = nBuf(oc, 0.008); const es = oc.createBufferSource(); es.buffer = exc;
  const eg = oc.createGain(); eg.gain.value = 0.7;
  const eh = oc.createBiquadFilter(); eh.type = 'bandpass'; eh.frequency.value = f * 2; eh.Q.value = 2;
  es.connect(eh).connect(eg); eg.connect(c1); eg.connect(c2); es.start(0);
  const thump = oc.createOscillator(); thump.type = 'sine';
  thump.frequency.setValueAtTime(f * 1.5, 0); thump.frequency.exponentialRampToValueAtTime(f, 0.02);
  const tg = oc.createGain(); tg.gain.setValueAtTime(0.15, 0); tg.gain.exponentialRampToValueAtTime(0.001, 0.04);
  thump.connect(tg).connect(lp); thump.start(0); thump.stop(0.05);
}

// ======================== EXPORT ========================

export default {
  id: 'texture',
  label: 'texture',

  mainGrids: [
    { rows: 16, cols: 32, defaultInstrument: 'glass' },
    { rows: 16, cols: 32, defaultInstrument: 'fm' },
    { rows: 16, cols: 32, defaultInstrument: 'sub' },
    { rows: 16, cols: 32, defaultInstrument: 'dust' },
  ],
  mainGridLayout: { cols: 2 },

  melodyGrids: [
    { rows: 8, cols: 32 },
    { rows: 8, cols: 32 },
  ],
  melodyDefaultInstrument: 'pad',

  mainInstruments: [
    { id: 'sub', label: 'sub' },
    { id: 'fm', label: 'fm' },
    { id: 'glass', label: 'glass' },
    { id: 'tape', label: 'tape' },
    { id: 'dust', label: 'dust' },
  ],
  melodyInstruments: [
    { id: 'pad', label: 'pad' },
    { id: 'organ', label: 'organ' },
    { id: 'piano', label: 'piano' },
  ],

  fx: {
    sub: { delay: 0.06, reverb: 0.10, gain: 0.34 },
    fm: { delay: 0.18, reverb: 0.25, gain: 0.22 },
    glass: { delay: 0.28, reverb: 0.40, gain: 0.20 },
    tape: { delay: 0.22, reverb: 0.35, gain: 0.20 },
    dust: { delay: 0.14, reverb: 0.22, gain: 0.28 },
    pad: { delay: 0.25, reverb: 0.50, gain: 0.16 },
    organ: { delay: 0.18, reverb: 0.30, gain: 0.18 },
    piano: { delay: 0.22, reverb: 0.38, gain: 0.20 },
  },

  effects: {
    reverbWet: 0.22,
    reverbDark: 0.8,
    reverbLength: 3.5,
    delayL: 0.45,
    delayR: 0.30,
    delayFeedback: 0.4,
    delayDarkLP: 1200,
    delayWet: 0.24,
    compThreshold: -12,
    compRatio: 8,
  },

  mainFreqs: FREQS,
  melodyFreqs: MEL_FREQS,
  numPatterns: 8,
  defaultMelN: 2,
  cellSize: 16,

  // Synthesis fallback — used when pre-rendered assets aren't available
  renderers: { sub: renderSub, fm: renderFm, glass: renderGlass, tape: renderTape, dust: renderDust, pad: renderPad, organ: renderOrgan, piano: renderPiano },
  durations: { sub: 2.0, fm: 2.2, glass: 2.6, tape: 2.4, dust: 1.2, pad: 3.5, organ: 2.8, piano: 2.4 },
  crush: {
    sub: { bits: 8, down: 5, tape: 2.5 }, fm: { bits: 10, down: 3, tape: 2.0 },
    glass: { bits: 11, down: 2, tape: 1.6 }, tape: { bits: 10, down: 4, tape: 2.8 },
    dust: { bits: 9, down: 3, tape: 2.0 }, pad: { bits: 10, down: 3, tape: 1.8 },
    organ: { bits: 12, down: 2, tape: 1.4 }, piano: { bits: 13, down: 1, tape: 1.2 },
  },
  fadeConfig: {
    sub: { fadeIn: 40, fadeOut: 80 }, fm: { fadeIn: 30, fadeOut: 60 },
    glass: { fadeIn: 20, fadeOut: 50 }, tape: { fadeIn: 35, fadeOut: 70 },
    dust: { fadeIn: 8, fadeOut: 20 }, pad: { fadeIn: 60, fadeOut: 120 },
    organ: { fadeIn: 15, fadeOut: 40 }, piano: { fadeIn: 5, fadeOut: 30 },
  },
};
