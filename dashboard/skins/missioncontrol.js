// Mission Control skin module for the SO-101 wall dashboard (see README.md in this folder for the contract).
// Attract: a six-pen strip-chart recorder. The last 60 s of joint trajectories scroll right-to-left on green graph
// paper, one lane per joint, with the live value as a seven-segment readout, a MET (mission elapsed time) counter
// and a REC / HOLD / IDLE lamp fed by api.record(). Samples api.arms() every 100 ms into a 600-slot ring buffer;
// the draw loop is requestAnimationFrame capped at 30 fps and allocates nothing per frame (strings only on change).
// Each lane autoscales to the last 60 s of its own pen (never narrower than MINSPAN, never outside the joint's full range): on the fixed
// +-120 deg scale a 2 deg sag is under a pixel and a resting arm draws six ruler-straight lines, which is why the chart looked dead.
// With no joint reading for 2.5 s (the arm is busy, unplugged, or the stream is down) the paper says so instead of leaving it blank.
(() => {
  if (!window.SKINS) return;
  const JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
  const JL = ["PAN", "LIFT", "ELBOW", "WRIST", "ROLL", "GRIP"];
  const JR = ["±120°", "±120°", "±120°", "±120°", "±180°", "0–100%"];
  const LO = [-120, -120, -120, -120, -180, 0], HI = [120, 120, 120, 120, 180, 100];
  const PEN = ["#e8dcc0", "#e2a93d", "#dc4a35", "#9ec7d8", "#8dbf7a", "#c7a4dc"];
  const TLBL = ["0", "-10 s", "-20 s", "-30 s", "-40 s", "-50 s", "-60 s"];
  const MINSPAN = [8, 8, 8, 8, 8, 12];   // narrowest lane range, degrees (gripper: percent)
  const rlo = LO.slice(), rhi = HI.slice();   // the lanes' current range, eased toward the target
  const N = 600, SPAN = 60000, SAMPLE_MS = 100, FRAME_MS = 1000 / 30;
  const PAPER = "#17301f", GRID_MINOR = "rgba(141,191,122,.16)", GRID_MAJOR = "rgba(141,191,122,.42)", LANE = "rgba(232,220,192,.38)";
  const FRAME = "#141516", BEZEL = "#0a0a0b", METAL = "#35363a", INK = "#e8dcc0", DIM = "#a39b86", READOUT = "#f2d98f", GO = "#8dbf7a", RED = "#dc4a35", AMBER = "#e2a93d";

  // ring buffer: ts[i] = sample wall time (ms), vals[i*6+j] = joint j value or NaN when the stream had no reading
  const ts = new Float64Array(N), vals = new Float32Array(N * 6);
  let head = 0, count = 0;
  // live text, rebuilt only when it changes
  const cur = ["--", "--", "--", "--", "--", "--"];
  let lastGood = 0, feedNote = "";   // wall time of the last sample with a reading; why there is none
  let metStr = "T+ 00:00:00", metSec = -1, epStr = "", lamp = 0;   // lamp: 0 idle, 1 rec, 2 hold
  // runtime
  let canvas = null, ctx = null, api = null, raf = 0, timer = 0, running = false, lastFrame = 0, t0 = 0, simEl = null;
  // layout (recomputed by size())
  let W = 0, H = 0, L = 0, R = 0, T = 0, B = 0, laneH = 0, small = false;
  let fLabel = "", fSmall = "", fRead = "", fMet = "", fTitle = "";

  const pad2 = n => (n < 10 ? "0" : "") + n;
  function met(now) {
    const s = Math.floor((now - t0) / 1000);
    if (s === metSec) return; metSec = s;
    metStr = "T+ " + pad2(Math.floor(s / 3600)) + ":" + pad2(Math.floor(s / 60) % 60) + ":" + pad2(s % 60);
  }

  function sample() {
    let pos = null, a = null, st = null;
    try { a = api.arms(); } catch (e) { a = null; }
    if (a && a.follower && a.follower.pos) pos = a.follower.pos;
    if (!pos) { try { st = api.state(); } catch (e) { st = null; } if (st && st.sim_arm && st.sim_arm.pos) pos = st.sim_arm.pos; }
    ts[head] = Date.now();
    feedNote = a && a.follower ? (a.follower.src === "busy" ? "ARM BUSY · NO STATE FILE" : a.follower.src === "unplugged" ? "FOLLOWER UNPLUGGED" : a.follower.src === "error" ? "BUS ERROR" : "NO READING") : "NO STREAM FROM THE JETSON";
    for (let j = 0; j < 6; j++) {
      const v = pos ? pos[JN[j]] : null;
      const ok = typeof v === "number" && v === v;
      vals[head * 6 + j] = ok ? v : NaN; if (ok) lastGood = ts[head];
      const s = ok ? v.toFixed(1) + (j === 5 ? "%" : "°") : "--";
      if (s !== cur[j]) cur[j] = s;
    }
    head = (head + 1) % N; if (count < N) count++;
    // recorder state lamp
    let rec = null; try { rec = api.record(); } catch (e) { rec = null; }
    let l = 0, e = "";
    if (rec && rec.running) {
      l = rec.paused_at ? 2 : 1;
      const tot = (rec.done_total || 0) + (rec.episode || 0), tgt = rec.episodes_target || rec.episodes || 0;
      e = (rec.phase === "reset" ? "RESET" : rec.phase === "starting" ? "STARTING" : "EP") + " " + tot + (tgt ? "/" + tgt : "") + (rec.name ? " · " + rec.name : "");
    } else if (rec && rec.done && !rec.err) e = "COMPLETE · " + (rec.name || "");
    lamp = l; if (e !== epStr) epStr = e;
  }

  function size() {
    const p = canvas.parentElement; if (!p) return;
    const r = p.getBoundingClientRect(), w = Math.round(r.width), h = Math.round(r.height);
    if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
    if (canvas.width === W && canvas.height === H) return;
    W = canvas.width; H = canvas.height; small = H < 300;
    const hd = Math.max(30, Math.min(48, Math.round(H * 0.085))), fm = Math.max(14, Math.min(22, Math.round(H * 0.035)));
    const lw = Math.max(64, Math.min(110, Math.round(W * 0.07))), rw = Math.max(90, Math.min(140, Math.round(W * 0.085)));
    L = lw; R = W - rw; T = hd; B = H - fm; laneH = (B - T) / 6;
    const fs = small ? 10 : 12, rs = small ? 15 : Math.max(16, Math.min(26, Math.round(laneH * 0.3)));
    fLabel = "600 " + fs + "px \"IBM Plex Mono\", Menlo, monospace";
    fSmall = "500 " + (fs - 1) + "px \"IBM Plex Mono\", Menlo, monospace";
    fRead = rs + "px \"Share Tech Mono\", Menlo, monospace";
    fMet = (small ? 15 : 22) + "px \"Share Tech Mono\", Menlo, monospace";
    fTitle = "600 " + (small ? 10 : 12) + "px \"IBM Plex Mono\", Menlo, monospace";
  }

  function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

  function autoscale(nowMs) {
    const first = (head - count + N) % N;
    for (let j = 0; j < 6; j++) {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < count; i++) { const idx = (first + i) % N; if (nowMs - ts[idx] > SPAN) continue; const v = vals[idx * 6 + j]; if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; } }
      let tl = LO[j], th = HI[j];
      if (mn <= mx) { const mid = (mn + mx) / 2, half = Math.max((mx - mn) * 0.6, MINSPAN[j] / 2); tl = Math.max(LO[j], mid - half); th = Math.min(HI[j], mid + half); if (th - tl < MINSPAN[j]) { if (tl === LO[j]) th = tl + MINSPAN[j]; else tl = th - MINSPAN[j]; } }
      rlo[j] += (tl - rlo[j]) * 0.15; rhi[j] += (th - rhi[j]) * 0.15;
    }
  }

  function draw(now) {
    const x = ctx, nowMs = Date.now(), pxMs = (R - L) / SPAN; autoscale(nowMs);
    // recorder frame and paper
    x.fillStyle = FRAME; x.fillRect(0, 0, W, H);
    x.fillStyle = PAPER; x.fillRect(L, T, R - L, B - T);
    // horizontal rules: 4 minor per lane, lane separators major
    x.lineWidth = 1; x.strokeStyle = GRID_MINOR; x.beginPath();
    for (let k = 0; k < 6; k++) for (let m = 1; m < 5; m++) { const y = Math.round(T + k * laneH + laneH * m / 5) + .5; x.moveTo(L, y); x.lineTo(R, y); }
    x.stroke();
    // vertical rules scroll with time: minor every second, major every 10 s, time labels on the majors
    const base = Math.floor(nowMs / 1000) * 1000;
    x.strokeStyle = GRID_MINOR; x.beginPath();
    for (let k = 0; k <= 60; k++) { const t = base - k * 1000; if ((t / 1000) % 10 === 0) continue; const gx = Math.round(R - (nowMs - t) * pxMs) + .5; if (gx < L) break; x.moveTo(gx, T); x.lineTo(gx, B); }
    x.stroke();
    x.strokeStyle = GRID_MAJOR; x.beginPath();
    for (let k = 0; k <= 60; k++) { const t = base - k * 1000; if ((t / 1000) % 10 !== 0) continue; const gx = Math.round(R - (nowMs - t) * pxMs) + .5; if (gx < L) break; x.moveTo(gx, T); x.lineTo(gx, B); }
    x.stroke();
    x.strokeStyle = LANE; x.beginPath();
    for (let k = 1; k < 6; k++) { const y = Math.round(T + k * laneH) + .5; x.moveTo(L, y); x.lineTo(R, y); }
    x.stroke();
    // sprocket holes along the paper's lower edge, moving with the grid
    x.fillStyle = BEZEL;
    const hole = Math.max(3, Math.min(5, (H - B) * 0.22)), hy = B + (H - B) / 2;
    for (let k = 0; k <= 60; k++) { const t = base - k * 1000; const gx = R - (nowMs - t) * pxMs; if (gx < L) break; x.beginPath(); x.arc(gx, hy, hole, 0, 6.2832); x.fill(); }
    // time labels under the majors
    x.fillStyle = DIM; x.font = fSmall; x.textAlign = "center"; x.textBaseline = "alphabetic";
    for (let k = 0; k <= 60; k += 10) { const t = Math.floor(nowMs / 10000) * 10000 - k * 1000; const gx = R - (nowMs - t) * pxMs; if (gx < L + 20) break; const i = Math.round((nowMs - t) / 10000); if (i >= 0 && i < 7 && !small) x.fillText(TLBL[i], gx, B - 4); }
    // traces: oldest to newest, pen up across gaps and off the left edge
    const first = (head - count + N) % N, pad = laneH * 0.08;
    x.lineWidth = small ? 1.4 : 2; x.lineJoin = "round"; x.lineCap = "round";
    x.save(); x.beginPath(); x.rect(L, T, R - L, B - T); x.clip();
    for (let j = 0; j < 6; j++) {
      const lo = rlo[j], span = rhi[j] - lo, top = T + j * laneH + pad, hgt = laneH - 2 * pad;
      x.strokeStyle = PEN[j]; x.beginPath(); let up = true;
      for (let i = 0; i < count; i++) {
        const idx = (first + i) % N, v = vals[idx * 6 + j];
        if (v !== v) { up = true; continue; }
        const px = R - (nowMs - ts[idx]) * pxMs; if (px < L - 4) { up = true; continue; }
        let f = (v - lo) / span; if (f < 0) f = 0; else if (f > 1) f = 1;
        const py = top + hgt * (1 - f);
        if (up) { x.moveTo(px, py); up = false; } else x.lineTo(px, py);
      }
      x.stroke();
    }
    x.restore();
    // pen heads at the right edge of the paper
    if (count) {
      const last = (head - 1 + N) % N;
      for (let j = 0; j < 6; j++) {
        const v = vals[last * 6 + j]; if (v !== v) continue;
        const lo = rlo[j], top = T + j * laneH + pad, hgt = laneH - 2 * pad; let f = (v - lo) / (rhi[j] - lo); if (f < 0) f = 0; else if (f > 1) f = 1;
        x.fillStyle = PEN[j]; x.beginPath(); x.arc(R, top + hgt * (1 - f), small ? 2.5 : 3.5, 0, 6.2832); x.fill();
      }
    }
    // left column: channel labels and ranges; right column: live readouts in the pen colour
    x.textBaseline = "middle";
    for (let j = 0; j < 6; j++) {
      const cy = T + j * laneH + laneH / 2;
      x.fillStyle = PEN[j]; x.fillRect(8, cy - (small ? 7 : 9), 4, small ? 14 : 18);
      x.fillStyle = INK; x.font = fLabel; x.textAlign = "left"; x.fillText(JL[j], 18, small ? cy : cy - 8);
      if (!small) { x.fillStyle = DIM; x.font = fSmall; x.fillText(Math.round(rlo[j]) + " … " + Math.round(rhi[j]) + (j === 5 ? "%" : "°"), 18, cy + 9); }
      x.fillStyle = BEZEL; roundRect(R + 10, cy - laneH * 0.22, W - R - 20, laneH * 0.44, 3); x.fill();
      x.fillStyle = PEN[j]; x.font = fRead; x.textAlign = "right"; x.fillText(cur[j], W - 18, cy + 1);
    }
    // no reading for a while: say so (a flat or empty chart must not look like a working one)
    if (nowMs - lastGood > 2500) {
      x.fillStyle = "rgba(10,12,11,.72)"; x.fillRect(L, T + (B - T) / 2 - 22, R - L, 44);
      x.fillStyle = AMBER; x.font = fLabel; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText("NO JOINT FEED · " + feedNote, (L + R) / 2, T + (B - T) / 2);
    }
    // header band: title, episode text, lamp, MET
    x.fillStyle = BEZEL; x.fillRect(0, T - 1, W, 1); x.fillStyle = METAL; x.fillRect(0, T - 2, W, 1);
    x.fillStyle = INK; x.font = fTitle; x.textAlign = "left"; x.textBaseline = "middle";
    x.fillText(small ? "STRIP CHART · JOINTS · 60 s" : "STRIP CHART RECORDER · JOINT TRAJECTORIES · LAST 60 s", 14, T / 2);
    const metW = small ? 120 : 170, lampW = small ? 54 : 66, lampH = small ? 16 : 22, right = W - 14;
    x.fillStyle = READOUT; x.font = fMet; x.textAlign = "right"; x.fillText(metStr, right, T / 2 + 1);
    x.fillStyle = DIM; x.font = fSmall; x.fillText("MET", right - metW, T / 2 + 1);
    const lx = right - metW - 44 - lampW, ly = T / 2 - lampH / 2;
    const lit = lamp === 1 ? (Math.floor(now / 500) % 2 === 0) : lamp === 2 ? (Math.floor(now / 1200) % 2 === 0) : false;
    x.fillStyle = lamp === 1 ? (lit ? RED : "#4a1a14") : lamp === 2 ? (lit ? AMBER : "#5a4210") : "#1e1f21";
    roundRect(lx, ly, lampW, lampH, 2); x.fill(); x.strokeStyle = BEZEL; x.lineWidth = 1; x.stroke();
    x.fillStyle = lamp === 0 ? DIM : (lit ? "#1a0c0a" : INK); x.font = fLabel; x.textAlign = "center";
    x.fillText(lamp === 1 ? "REC" : lamp === 2 ? "HOLD" : "IDLE", lx + lampW / 2, T / 2 + 1);
    if (epStr) { x.fillStyle = INK; x.font = fSmall; x.textAlign = "right"; x.fillText(epStr, lx - 16, T / 2 + 1); }
  }

  function frame(now) {
    if (!running) return;
    raf = requestAnimationFrame(frame);
    if (now - lastFrame < FRAME_MS - 1) return;
    lastFrame = now;
    size(); if (W <= 0 || H <= 0) return;
    if (simEl && !simEl.classList.contains("off")) return;   // the simulation view covers the canvas; skip the draw
    met(now);
    try { draw(now); } catch (e) { running = false; cancelAnimationFrame(raf); console.error("missioncontrol attract", e); }
  }
  function onResize() { if (canvas) { W = 0; H = 0; size(); } }

  const attract = {
    start(c, a) {
      canvas = c; api = a; ctx = c.getContext("2d"); simEl = document.getElementById("simview");
      window.so101StripRange = j => [rlo[j], rhi[j]];   // the lanes' current vertical ranges (the UI test reads them)
      head = 0; count = 0; metSec = -1; epStr = ""; lamp = 0; W = 0; H = 0;
      t0 = performance.now(); lastFrame = 0; running = true;
      size(); sample();
      timer = setInterval(sample, SAMPLE_MS);
      addEventListener("resize", onResize);
      raf = requestAnimationFrame(frame);
    },
    stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf); raf = 0;
      if (timer) clearInterval(timer); timer = 0;
      removeEventListener("resize", onResize);
      if (ctx && canvas) { try { ctx.clearRect(0, 0, canvas.width, canvas.height); } catch (e) {} }
      canvas = null; ctx = null; api = null; simEl = null;   // nothing was added to #gallery, so nothing to remove
    },
  };

  SKINS.register("missioncontrol", {
    attract,
    labels: {
      "#statusp h2": "FLIGHT DIRECTOR",
      "#jetsonp h2": "EECOM · JETSON ORIN NANO",
      "#armp h2": "GNC · ARM JOINTS",
      "section.panel:has(#loss) h2": "FIDO · TRAINING · ACT LOSS",
      "section.panel:has(#ds) h2": "TELMU · DATASETS",
      "section.panel:has(#pol) h2": "GUIDANCE · POLICIES",
      "section.panel:has(#events) h2": "COMM LOOP · EVENT LOG",
      "#shell .termbar span": "CAPCOM · SHELL",
      "#term .termbar span": "ONBOARD · TERMINAL",
      "header h1 small": "MISSION CONTROL · ARM LAB",
      "#recnext": "GO NEXT",
      "#recredo": "NO-GO REDO",
      "#recstop": "■ ABORT",
      "#rf_cancel": "SCRUB",
      "#recmodal h2": "FLIGHT PLAN · RECORD A DATASET",
      "#lockscreen .brand": "CAPCOM AUTHORIZATION",
      "#lockscreen .brand small": "CREW CLEARANCE · ARM LAB",
      "#lockscreen h1": "IDENTIFY · RETINAL SCAN",
    },
  });
})();
