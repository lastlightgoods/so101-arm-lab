// NOSTROMO skin module for the SO-101 wall dashboard. See README.md in this folder for the contract.
// Registers labels (the ship's computer's language), apply()/remove() (typewriter re-trigger, wireframe tweak on the
// exposed three.js scene) and the attract piece: a motion tracker whose blips are the two arms and the three cameras.
(() => {
  if (!window.SKINS) return;
  const GREEN = 0x9cff9c, DIMGREEN = 0x3f8f3f;
  let warned = 0; const warn = (where, e) => { if (warned++ < 3) console.warn("nostromo skin:", where, e); };   // never throw into the switcher, never fail silently either

  // ---------------------------------------------------------------- labels (static headings and captions only)
  const labels = {
    "#statusp h2": "SYSTEMS STATUS",
    "#jetsonp h2": "CORE PROCESSOR · JETSON ORIN",
    "#armp h2": "MANIPULATOR · JOINTS",
    ".rightcol .panel:nth-child(1) h2": "LEARNING · ACT LOSS",
    ".rightcol .panel:nth-child(2) h2": "DATA STORES",
    ".rightcol .panel:nth-child(3) h2": "PROTOCOLS",
    "main > .panel.wide:last-child h2": "SHIP LOG",
    "#term .termbar > span:first-child": "TERMINAL · JETSON",
    "#shell .termbar > span:first-child": "CLAUDE · SHELL",
    "h1 small": "LASTLIGHT ROBOTICS · ARM LAB · DECK C",
    "#lockbtn": "⬡ SEAL",
    "#lockscreen .brand": "MOTHER",
    "#lockscreen .brand small": "SHIP'S COMPUTER · ARM LAB",
    "#lockscreen h1": "CLEARANCE REQUIRED",
    "#lockscreen .hint": "PRESENT RETINA TO THE WRIST CAMERA  ·  HOLD U, OR PRESS AND HOLD THE SCREEN, FOR 2 S TO OVERRIDE",
    "#recmodal h2": "LOG A DATASET",
  };

  // ---------------------------------------------------------------- apply / remove
  // Wireframe: index.html exposes the three.js scene as window.__scene. The robot meshes are tagged userData.wire by its
  // wireify(); each carries a LineSegments child (edges, orange for the follower, cyan ghost for the leader). We hide the
  // mesh material (Material.visible=false keeps the child lines drawing) and recolour the edges phosphor green. Robots load
  // asynchronously, so a 1 s scan picks up meshes that arrive after apply(). Everything touched is restored in remove().
  // Trade-off: without the scene the CSS filter on #arm3d alone only tints the shaded arm green; it cannot remove faces.
  const touched = new Map(); let scanTimer = 0, headingNodes = [];
  function wireScan() {
    const scene = window.__scene; if (!scene || typeof scene.traverse !== "function") return;
    try {
      scene.traverse(o => {
        if (!o || !o.isMesh || !o.userData || !o.userData.wire || touched.has(o)) return;
        const rec = { mat: o.material, lines: [] };
        if (o.material) { rec.visible = o.material.visible; o.material.visible = false; }
        for (const ch of o.children || []) {
          if (ch.isLineSegments && ch.material && ch.material.color) {
            const hex = ch.material.color.getHex(), ghost = hex === 0x4fd6ff;
            rec.lines.push({ line: ch, hex, opacity: ch.material.opacity });
            ch.material.color.setHex(ghost ? DIMGREEN : GREEN); ch.material.opacity = ghost ? 0.55 : 0.95;
          }
        }
        touched.set(o, rec);
      });
    } catch (e) { warn("wireScan", e); }
  }
  function wireRestore() {
    for (const [o, rec] of touched) {
      try {
        if (o.material === rec.mat && rec.mat) rec.mat.visible = rec.visible !== false;
        for (const l of rec.lines) { if (l.line.material && l.line.material.color) { l.line.material.color.setHex(l.hex); l.line.material.opacity = l.opacity; } }
      } catch (e) {}
    }
    touched.clear();
  }
  function retype() {
    const html = document.documentElement;
    headingNodes = Array.from(document.querySelectorAll(".panel h2, .termbar"));
    headingNodes.forEach((h, i) => h.style.setProperty("--i", String(i)));
    html.classList.remove("nos-type"); void html.offsetWidth; html.classList.add("nos-type");
  }
  function apply() {
    retype();
    wireScan(); if (!scanTimer) scanTimer = setInterval(wireScan, 1000);
  }
  function remove() {
    if (scanTimer) { clearInterval(scanTimer); scanTimer = 0; }
    wireRestore();
    document.documentElement.classList.remove("nos-type");
    headingNodes.forEach(h => h.style.removeProperty("--i")); headingNodes = [];
  }

  // ---------------------------------------------------------------- attract: motion tracker
  // Origin at bottom centre, the upper half-plane is the tracker field. Bearing = shoulder_pan, range = shoulder_lift.
  // Sweep oscillates across the arc; blips flash when it passes them and decay. A range ring leaves the origin every
  // 2 s. The ping pulse runs at 3 s idle, 1 s while a record session runs, 2 s while it is paused.
  const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
  const TRAIL = 24;
  const T = {
    c: null, x: null, api: null, raf: 0, on: false, last: 0, t0: 0, W: 0, H: 0, ox: 0, oy: 0, R: 0,
    fontBig: "", fontMid: "", fontSmall: "", sb: 20, sm: 12, ss: 11, cost: 0, lowfx: false,
    ringT0: 0, pingT0: 0, pingPeriod: 3, pings: 0, sweepPhase: 0,
    blips: [
      { name: "FOLLOWER", key: "follower", ok: false, a: 0, r: 0, hot: 0, trail: new Float32Array(TRAIL * 2), ti: 0, tn: 0, tlast: 0, prev: new Float32Array(6), hasPrev: false, speed: 0 },
      { name: "LEADER", key: "leader", ok: false, a: 0, r: 0, hot: 0, trail: new Float32Array(TRAIL * 2), ti: 0, tn: 0, tlast: 0, prev: new Float32Array(6), hasPrev: false, speed: 0 },
    ],
    cams: [ { name: "CAM TOP", a: -Math.PI / 2, r: .86, hot: 0 }, { name: "CAM FRONT", a: -Math.PI / 2 - .72, r: .62, hot: 0 }, { name: "CAM WRIST", a: -Math.PI / 2 + .95, r: .36, hot: 0 } ],
    // readout strings are rebuilt at 4 Hz, not per frame
    txtAt: 0, txtRange: "RANGE ----", txtClosing: "CLOSING ----", txtContacts: "CONTACTS 03", txtPing: "PING 3.0 S", txtMode: "TRACKING", txtBearing: "", txtPings: "PINGS 0000", lastRecMode: "",
  };
  function onResize() {
    const c = T.c; if (!c) return; const p = c.parentElement; if (!p) return;
    const w = p.clientWidth, h = p.clientHeight; if (w <= 0 || h <= 0) return;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (T.W === w && T.H === h) return;
    T.W = w; T.H = h; T.ox = w / 2; T.oy = h - Math.max(26, h * .06);
    T.R = Math.min(T.oy - Math.max(44, h * .11), w / 2 - 28);
    const s = Math.max(12, Math.min(h * .07, 30));
    T.sb = Math.round(s * 1.7); T.sm = Math.round(s); T.ss = Math.round(Math.max(11, s * .62));   // px sizes, used for line spacing
    T.fontBig = `${T.sb}px "VT323","Share Tech Mono",monospace`;
    T.fontMid = `${T.sm}px "Share Tech Mono","VT323",monospace`;
    T.fontSmall = `${T.ss}px "Share Tech Mono","VT323",monospace`;
  }
  function recMode() {
    let R = null; try { R = T.api && T.api.record ? T.api.record() : null; } catch (e) { R = null; }
    if (!R || !R.running) return "idle";
    return R.paused_at ? "paused" : "running";
  }
  function readArms(now) {
    let A = null; try { A = T.api && T.api.arms ? T.api.arms() : null; } catch (e) { A = null; }
    for (let bi = 0; bi < 2; bi++) {
      const b = T.blips[bi], arm = A && A[b.key], pos = arm && arm.pos;
      const pan = pos && typeof pos.shoulder_pan === "number" ? pos.shoulder_pan : null;
      const lift = pos && typeof pos.shoulder_lift === "number" ? pos.shoulder_lift : null;
      if (pan === null || lift === null) { b.ok = false; b.hasPrev = false; b.speed = 0; continue; }
      b.ok = true;
      // bearing: pan -110..110 deg spread across the arc, 0 straight ahead (up). range: lift -100 (folded) .. 100 (reaching)
      b.a = -Math.PI / 2 + Math.max(-1, Math.min(1, pan / 110)) * (Math.PI / 2 - .08);
      b.r = .14 + .74 * Math.max(0, Math.min(1, (lift + 100) / 200));
      // joint speed (deg/s) from the previous sample: the "closing" readout
      let d2 = 0; for (let i = 0; i < 6; i++) { const v = typeof pos[JOINTS[i]] === "number" ? pos[JOINTS[i]] : 0; if (b.hasPrev) { const d = v - b.prev[i]; d2 += d * d; } b.prev[i] = v; }
      const dt = b.tlast ? (now - b.tlast) / 1000 : 0;
      if (b.hasPrev && dt > .05) { const sp = Math.sqrt(d2) / dt; if (sp > .01 || !b.speed) b.speed = b.speed * .6 + sp * .4; }
      b.hasPrev = true;
      if (now - b.tlast > 100) { b.tlast = now; b.trail[b.ti * 2] = b.a; b.trail[b.ti * 2 + 1] = b.r; b.ti = (b.ti + 1) % TRAIL; if (b.tn < TRAIL) b.tn++; }
    }
    return A;
  }
  function updateText(now, A) {
    const f = T.blips[0], l = T.blips[1];
    if (f.ok && l.ok && A && A.follower && A.leader && A.follower.pos && A.leader.pos) {
      let d2 = 0; for (let i = 0; i < 6; i++) { const a = A.follower.pos[JOINTS[i]] || 0, b = A.leader.pos[JOINTS[i]] || 0; d2 += (a - b) * (a - b); }
      T.txtRange = "RANGE " + String(Math.min(9999, Math.round(Math.sqrt(d2) * 10))).padStart(4, "0");
    } else if (f.ok) T.txtRange = "RANGE " + String(Math.min(9999, Math.round(f.r * 1000))).padStart(4, "0");
    else T.txtRange = "RANGE ----";
    T.txtClosing = f.ok ? "CLOSING " + String(Math.min(9999, Math.round(f.speed * 10))).padStart(4, "0") : "CLOSING ----";
    const n = 3 + (f.ok ? 1 : 0) + (l.ok ? 1 : 0);
    T.txtContacts = "CONTACTS " + String(n).padStart(2, "0");
    T.txtPing = "PING " + T.pingPeriod.toFixed(1) + " S";
    let locked = false; try { locked = !!(T.api && T.api.lock && T.api.lock()); } catch (e) {}
    const m = T.lastRecMode;
    T.txtMode = locked ? "SEALED · CLEARANCE REQUIRED" : m === "running" ? "RECORDING · TRACKING" : m === "paused" ? "HOLD · TRACKING" : (f.ok || l.ok) ? "TRACKING" : "NO CONTACT";
    T.txtBearing = f.ok ? "BRG " + String(Math.round((f.a + Math.PI / 2) * 180 / Math.PI)).padStart(4, " ") + "°" : "BRG ----";
    T.txtPings = "PINGS " + String(T.pings % 10000).padStart(4, "0");
  }
  // colours are constants and fades go through globalAlpha so the hot loop builds no strings
  const C_BG = "#020502", C_INK = "#9cff9c", C_DIM = "#3f8f3f", C_WARM = "#efe6c8";
  function draw(now) {
    const x = T.x, W = T.W, H = T.H, ox = T.ox, oy = T.oy, R = T.R; if (!x || R <= 0) return;
    const t = (now - T.t0) / 1000, lowfx = T.lowfx;
    x.setTransform(1, 0, 0, 1, 0, 0); x.globalAlpha = 1; x.globalCompositeOperation = "source-over"; x.shadowBlur = 0;
    x.fillStyle = C_BG; x.fillRect(0, 0, W, H);
    x.lineCap = "round"; x.lineJoin = "round";
    // static field: half-plane rings and bearing lines
    x.strokeStyle = C_DIM; x.lineWidth = 1; x.globalAlpha = .45;
    for (let k = 1; k <= 4; k++) { x.beginPath(); x.arc(ox, oy, R * k / 4, Math.PI, 2 * Math.PI); x.stroke(); }
    x.globalAlpha = .28;
    for (let k = 0; k <= 6; k++) { const a = Math.PI + Math.PI * k / 6; x.beginPath(); x.moveTo(ox, oy); x.lineTo(ox + Math.cos(a) * R, oy + Math.sin(a) * R); x.stroke(); }
    x.strokeStyle = C_INK; x.lineWidth = 1.5; x.globalAlpha = .6; x.beginPath(); x.moveTo(ox - R, oy + .5); x.lineTo(ox + R, oy + .5); x.stroke();
    // expanding range ring every 2 s
    const rf = (t - T.ringT0) / 2; if (rf >= 1) T.ringT0 = t - ((t - T.ringT0) % 2);
    const rr = R * Math.min(1, rf); x.globalAlpha = Math.max(0, 1 - rf) * .55; x.lineWidth = 2; x.beginPath(); x.arc(ox, oy, Math.max(1, rr), Math.PI, 2 * Math.PI); x.stroke();
    // ping pulse: cadence follows the record session
    const pf = (t - T.pingT0) / T.pingPeriod; if (pf >= 1) { T.pingT0 = t - ((t - T.pingT0) % T.pingPeriod); T.pings++; }
    const pa = Math.max(0, 1 - pf * 2.2); if (pa > 0) {
      x.fillStyle = C_WARM; x.globalAlpha = pa * .85; x.beginPath(); x.arc(ox, oy, 6 + pa * 10, 0, Math.PI * 2); x.fill();
      x.strokeStyle = C_WARM; x.globalAlpha = pa * .5; x.lineWidth = 3; x.beginPath(); x.arc(ox, oy, 14 + (1 - pa) * R * .22, Math.PI, 2 * Math.PI); x.stroke();
    }
    // sweep: oscillates across the arc with a fading wake
    T.sweepPhase = (Math.sin(t * .9) * .5 + .5);
    const sa = Math.PI + .02 + T.sweepPhase * (Math.PI - .04), dir = Math.cos(t * .9) >= 0 ? 1 : -1, sweepBearing = sa - 2 * Math.PI;
    x.strokeStyle = C_INK;
    for (let k = 10; k >= 0; k--) {
      const a = sa - dir * k * .028; if (a < Math.PI || a > 2 * Math.PI) continue;
      x.globalAlpha = (1 - k / 11) * (k ? .22 : .95); x.lineWidth = k ? 3 : 2;
      x.beginPath(); x.moveTo(ox, oy); x.lineTo(ox + Math.cos(a) * R, oy + Math.sin(a) * R); x.stroke();
    }
    if (!lowfx) { x.shadowColor = C_INK; x.shadowBlur = 10; }
    // camera contacts: fixed hollow diamonds
    x.font = T.fontSmall; x.textAlign = "left"; x.textBaseline = "middle";
    for (let i = 0; i < 3; i++) {
      const cm = T.cams[i];
      const ad = Math.abs(cm.a - sweepBearing); if (ad < .035) cm.hot = 1; else cm.hot *= .965;
      const px = ox + Math.cos(cm.a) * cm.r * R, py = oy + Math.sin(cm.a) * cm.r * R, s = 7 + cm.hot * 3;
      x.strokeStyle = C_INK; x.globalAlpha = .45 + cm.hot * .55; x.lineWidth = 1.5;
      x.beginPath(); x.moveTo(px, py - s); x.lineTo(px + s, py); x.lineTo(px, py + s); x.lineTo(px - s, py); x.closePath(); x.stroke();
      x.fillStyle = C_DIM; x.globalAlpha = .7 + cm.hot * .3; x.fillText(cm.name, px + 12, py);
    }
    // arm blips: trails then the square contact
    for (let bi = 0; bi < 2; bi++) {
      const b = T.blips[bi]; if (!b.ok) continue;
      const ad = Math.abs(b.a - sweepBearing); if (ad < .035) b.hot = 1; else b.hot *= .965;
      const main = bi === 0, col = main ? C_INK : C_WARM;
      x.fillStyle = col;
      for (let k = 0; k < b.tn; k++) {
        const idx = ((b.ti - 1 - k) % TRAIL + TRAIL) % TRAIL, ta = b.trail[idx * 2], tr = b.trail[idx * 2 + 1];
        const px = ox + Math.cos(ta) * tr * R, py = oy + Math.sin(ta) * tr * R;
        x.globalAlpha = (main ? .35 : .25) * (1 - k / TRAIL);
        x.fillRect(px - 1.5, py - 1.5, 3, 3);
      }
      const px = ox + Math.cos(b.a) * b.r * R, py = oy + Math.sin(b.a) * b.r * R, s = 9 + b.hot * 5;
      x.globalAlpha = (main ? .65 : .55) + b.hot * .35;
      x.fillRect(px - s / 2, py - s / 2, s, s);
      if (b.hot > .5) { x.strokeStyle = col; x.lineWidth = 1.5; x.beginPath(); x.arc(px, py, s + 6 + (1 - b.hot) * 18, 0, Math.PI * 2); x.stroke(); }
      x.globalAlpha = main ? .9 : .85; x.fillText(b.name, px + 14, main ? py - 2 : py + T.ss * .9);   // leader label sits below so the two never collide
    }
    x.shadowBlur = 0; x.globalAlpha = 1;
    // readouts: left and right columns, caret at the origin
    const pad = Math.max(14, W * .012), top = Math.max(14, H * .045), lm = T.sm * 1.35, ls = T.ss * 1.45;
    x.textBaseline = "top"; x.textAlign = "left"; x.font = T.fontMid; x.fillStyle = C_INK;
    x.fillText("MOTION TRACKER", pad, top);
    x.fillStyle = C_DIM; x.font = T.fontSmall;
    x.fillText(T.txtMode, pad, top + lm); x.fillText(T.txtContacts, pad, top + lm + ls); x.fillText(T.txtPing, pad, top + lm + ls * 2);
    x.textAlign = "right"; x.font = T.fontBig; x.fillStyle = C_WARM;
    x.fillText(T.txtRange, W - pad, top - 4);
    x.font = T.fontMid; x.fillStyle = C_INK; x.fillText(T.txtClosing, W - pad, top + T.sb * .95);
    x.fillStyle = C_DIM; x.font = T.fontSmall; x.fillText(T.txtBearing, W - pad, top + T.sb * .95 + lm);
    x.fillText(T.txtPings, W - pad, top + T.sb * .95 + lm + ls);
    x.textAlign = "center"; x.textBaseline = "bottom"; x.fillStyle = (Math.floor(t * 2) % 2) ? C_INK : C_DIM; x.font = T.fontSmall;
    x.fillText("\u25b2", ox, oy - 2);
  }
  function frame(now) {
    if (!T.on) return;
    T.raf = requestAnimationFrame(frame);
    if (now - T.last < 33) return;                       // <= 30 fps
    T.last = now;
    const t0 = performance.now();
    try {
      onResize();
      const mode = recMode(); if (mode !== T.lastRecMode) { T.lastRecMode = mode; T.pingPeriod = mode === "running" ? 1 : mode === "paused" ? 2 : 3; T.pingT0 = (now - T.t0) / 1000; }
      const A = readArms(now);
      if (now - T.txtAt > 250) { T.txtAt = now; updateText(now, A); }
      draw(now);
    } catch (e) { warn("frame", e); }
    const cost = performance.now() - t0; T.cost = T.cost * .9 + cost * .1;
    if (T.cost > 9 && !T.lowfx) T.lowfx = true; else if (T.cost < 4 && T.lowfx) T.lowfx = false;   // frame budget: drop the glow first
  }
  const attract = {
    start(canvas, api) {
      if (!canvas) return;
      T.c = canvas; T.x = canvas.getContext("2d"); T.api = api || {}; T.on = true; T.W = T.H = 0; T.last = 0; T.t0 = performance.now(); T.ringT0 = 0; T.pingT0 = 0; T.pings = 0; T.lastRecMode = ""; T.cost = 0; T.lowfx = false;
      for (const b of T.blips) { b.ok = false; b.hot = 0; b.tn = 0; b.ti = 0; b.tlast = 0; b.hasPrev = false; b.speed = 0; }
      addEventListener("resize", onResize);
      onResize(); T.raf = requestAnimationFrame(frame);
    },
    stop() {
      T.on = false; if (T.raf) cancelAnimationFrame(T.raf); T.raf = 0;
      removeEventListener("resize", onResize);
      if (T.x && T.c) { try { T.x.setTransform(1, 0, 0, 1, 0, 0); T.x.clearRect(0, 0, T.c.width, T.c.height); } catch (e) {} }
      T.c = null; T.x = null; T.api = null;   // nothing was added to #gallery; the HUD lives on the canvas
    },
  };

  window.SKINS.register("nostromo", { labels, apply, remove, attract });
})();
