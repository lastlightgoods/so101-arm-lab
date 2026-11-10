// DYSTOPIAN skin module for the SO-101 wall dashboard (Lastlight Goods sub-brand). See README.md for the contract.
// Attract: the all-silver ridge mark as a horizon (summit at 72% of the span), a faint grid under it, the live follower
// pose as a planar silhouette in front, and facility status cards cycling every 6 s. Silver only, never gold.
(() => {
  if (!window.SKINS) return;
  const SILVER = "#aeb4bb", INK = "#1a191f", SAND = "#f0e9dc", CLAY = "#b45c40";
  const GRID = "rgba(174,180,187,.10)", GRID2 = "rgba(174,180,187,.18)";
  // the mark: normalised ridge, x left to right, h as a fraction of the summit height; summit at 72% of the span
  const RIDGE = [[0, .16], [.17, .52], [.30, .34], [.46, .66], [.58, .44], [.72, 1], [.84, .62], [1, .30]];
  const STROKE = .0405;            // stroke = 4.05% of mark width, round caps and joins, uniform, no gradient
  const MARK_H = 426;              // summit height in a 1000-wide frame (126/296 of the delivered SVG span)
  const NS = "http://www.w3.org/2000/svg";

  function markSvg() {
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `-21 -21 1042 ${MARK_H + 42}`); s.setAttribute("class", "dyst-mark"); s.setAttribute("role", "img"); s.setAttribute("aria-label", "DYSTOPIAN");
    const p = document.createElementNS(NS, "polyline");
    p.setAttribute("points", RIDGE.map(([x, h]) => `${(x * 1000).toFixed(1)},${((1 - h) * MARK_H).toFixed(1)}`).join(" "));
    s.appendChild(p); return s;
  }

  // ---- lock screen additions (added in apply, removed in remove) ----
  let markEl = null, plateEl = null;
  function apply() {
    const ls = document.querySelector("#lockscreen"), sc = document.querySelector("#scanner");
    if (!ls) return;
    if (!markEl) markEl = markSvg();
    if (sc) ls.insertBefore(markEl, sc); else ls.appendChild(markEl);
    if (!plateEl) { plateEl = document.createElement("div"); plateEl.className = "dyst-plate"; plateEl.textContent = "SECTOR 01 · ARM LAB"; }
    ls.appendChild(plateEl);
  }
  function remove() {
    if (markEl && markEl.parentNode) markEl.parentNode.removeChild(markEl);
    if (plateEl && plateEl.parentNode) plateEl.parentNode.removeChild(plateEl);
  }

  // ---- attract ----
  const A = {
    c: null, x: null, api: null, raf: 0, on: false, last: 0, w: 0, h: 0,
    cards: ["SECTOR 01. NO SIGNAL.", "", "", "", "", ""], ncards: 1, card: 0, cardAt: 0, cardTimer: 0,
    fontTag: "", fontBig: "", fontSmall: "", fontCard: "", bigPx: 0, pad: 0,
    // planar chain: shoulder_lift 0 = horizontal, negative = raised; elbow_flex and wrist_flex are relative bends (degrees)
    lift: -100, elbow: 100, wrist: 70, grip: 20, tLift: -100, tElbow: 100, tWrist: 70, tGrip: 20, live: false,
    rx: new Float64Array(8), ry: new Float64Array(8),
  };
  const DEG = Math.PI / 180, FRAME = 1000 / 30;

  function layout() {
    const c = A.c, r = c.parentElement.getBoundingClientRect(), w = Math.round(r.width), h = Math.round(r.height);
    if (w > 0 && h > 0 && (c.width !== w || c.height !== h)) { c.width = w; c.height = h; }
    A.w = c.width; A.h = c.height;
    const H = A.h; A.pad = Math.round(Math.max(14, H * .035));
    A.bigPx = Math.round(Math.max(18, Math.min(H * .07, A.w * .034)));
    A.fontBig = `800 ${A.bigPx}px "JetBrains Mono", Menlo, monospace`; A.fontCard = A.fontBig;
    A.fontSmall = `800 ${Math.round(Math.max(10, H * .022))}px "JetBrains Mono", Menlo, monospace`;
    A.fontTag = `800 ${Math.round(Math.max(10, H * .019))}px "JetBrains Mono", Menlo, monospace`;
    // ridge geometry in canvas space: baseline at 64% of the height, summit height 32% of the height
    const base = H * .64, S = H * .32;
    for (let i = 0; i < 8; i++) { A.rx[i] = RIDGE[i][0] * A.w; A.ry[i] = base - RIDGE[i][1] * S; }
  }
  // shrink the card face so the longest card fits between the left pad and the index squares (runs on card change, not per frame)
  function fitCard() {
    if (!A.x) return;
    const avail = A.w - 2 * A.pad - Math.max(8, A.h * .016) * 1.6 * (A.ncards + 1) - A.w * .04;
    let px = A.bigPx; A.x.font = A.fontBig;
    const wid = A.x.measureText(A.cards[A.card] || "").width;
    if (wid > avail && wid > 0) px = Math.max(12, Math.floor(px * avail / wid));
    A.fontCard = px === A.bigPx ? A.fontBig : `800 ${px}px "JetBrains Mono", Menlo, monospace`;
  }
  const onResize = () => { if (A.c) { layout(); fitCard(); } };

  const KIND = { record: "RECORDING", teleoperate: "TELEOP", rollout: "POLICY", nudge: "POLICY", dance: "DANCE", demo_wave: "WAVE", replay: "REPLAY" };
  function buildCards() {
    let s = null, r = null, lk = false;
    try { s = A.api.state(); r = A.api.record(); lk = A.api.lock(); } catch (e) { s = null; r = null; }
    const cards = A.cards; let n = 0;
    const j = s && s.jetson, on = !!(j && j.online);
    const kinds = s && Array.isArray(s.procs) ? s.procs.map(p => KIND[p.kind] || String(p.kind || "").toUpperCase()).filter(Boolean) : [];
    cards[n++] = !s ? "SECTOR 01. NO SIGNAL." : lk ? "SECTOR 01. LOCKED." : "SECTOR 01. OPEN.";
    cards[n++] = !s ? "ARM BAY. NO SIGNAL." : !on ? "ARM BAY. OFFLINE." : kinds.length ? `ARM BAY. ${kinds.join(" + ")}.` : "ARM BAY. IDLE.";
    if (r && r.running) {
      const tot = (r.done_total || 0) + (r.episode || 0), tgt = r.episodes_target || r.episodes || 0;
      cards[n++] = `${r.paused_at ? "PAUSED" : r.phase === "reset" ? "RESET" : r.phase === "starting" ? "STARTING" : "RECORDING"}. ${tot} OF ${tgt}.`;
    } else if (r && r.name) cards[n++] = `LAST SESSION. ${String(r.name).toUpperCase()}. ${(r.done_total || 0) + (r.episode || 0)} EPISODES.`;
    else cards[n++] = "TRANSPORT. IDLE.";
    cards[n++] = !on ? "COMPUTE. UNREACHABLE." : `COMPUTE. ${Math.round(j.temp || 0)} C. ${(+j.watts || 0).toFixed(1)} W.`;
    if (on) cards[n++] = `COMPUTE. GPU ${Math.round(j.gpu || 0)}%. CPU ${Math.round(j.cpu || 0)}%. RAM ${((j.ram_used || 0) / 1024).toFixed(1)} G.`;
    if (s && s.datasets) {
      const ds = Object.values(s.datasets); let ep = 0; for (const d of ds) ep += (d && d.episodes) || 0;
      const pol = s.policies ? Object.keys(s.policies).length : 0;
      cards[n++] = `DATA VAULT. ${ds.length} SETS. ${ep} EPISODES. ${pol} POLICIES.`;
    }
    A.ncards = n; if (A.card >= n) A.card = 0; fitCard();
    // pose
    let a = null; try { a = A.api.arms(); } catch (e) { a = null; }
    const pos = a && a.follower && a.follower.pos;
    A.live = !!(pos && typeof pos.shoulder_lift === "number");
    if (A.live) { A.tLift = pos.shoulder_lift; A.tElbow = +pos.elbow_flex || 0; A.tWrist = +pos.wrist_flex || 0; A.tGrip = +pos.gripper || 0; }
    else { A.tLift = -100; A.tElbow = 100; A.tWrist = 70; A.tGrip = 20; }
  }

  function ridgePath(x) { x.beginPath(); x.moveTo(A.rx[0], A.ry[0]); for (let i = 1; i < 8; i++) x.lineTo(A.rx[i], A.ry[i]); }

  function draw(now) {
    const x = A.x, W = A.w, H = A.h; if (W === 0 || H === 0) return;
    x.setTransform(1, 0, 0, 1, 0, 0); x.globalAlpha = 1; x.shadowBlur = 0;
    x.fillStyle = INK; x.fillRect(0, 0, W, H);
    // faint grid under the ridge
    x.save(); ridgePath(x); x.lineTo(W, H); x.lineTo(0, H); x.closePath(); x.clip();
    x.lineWidth = 1; x.strokeStyle = GRID; x.beginPath();
    const gx = W / 24; for (let i = 1; i < 24; i++) { const px = Math.round(i * gx) + .5; x.moveTo(px, 0); x.lineTo(px, H); }
    const gy = H / 16; for (let i = 1; i < 16; i++) { const py = Math.round(i * gy) + .5; x.moveTo(0, py); x.lineTo(W, py); }
    x.stroke(); x.restore();
    // the ridge: uniform silver stroke, 4.05% of the span, round caps and joins, no gradient
    x.lineCap = "round"; x.lineJoin = "round"; x.strokeStyle = SILVER; x.lineWidth = STROKE * W; ridgePath(x); x.stroke();
    // arm silhouette in front of the ridge (ink halo so it cuts the horizon)
    const k = .18; A.lift += (A.tLift - A.lift) * k; A.elbow += (A.tElbow - A.elbow) * k; A.wrist += (A.tWrist - A.wrist) * k; A.grip += (A.tGrip - A.grip) * k;
    const L = H * .40, L1 = L * .40, L2 = L * .37, L3 = L * .23, bx = W * .30, by = H * .84, floor = by - Math.max(3, H * .01);
    const t1 = -A.lift * DEG, t2 = t1 - A.elbow * DEG, t3 = t2 - A.wrist * DEG;
    // joints never drop below the plinth: the real arm sits on a table
    const x1 = bx + L1 * Math.cos(t1), y1 = Math.min(floor, by - L1 * Math.sin(t1)), x2 = x1 + L2 * Math.cos(t2), y2 = Math.min(floor, y1 - L2 * Math.sin(t2)), x3 = x2 + L3 * Math.cos(t3), y3 = Math.min(floor, y2 - L3 * Math.sin(t3));
    const ga = Math.min(1, Math.max(0, A.grip / 100)) * 28 * DEG, jaw = L3 * .42, plinth = Math.max(10, H * .035);
    x.globalAlpha = A.live ? 1 : .45;
    for (let pass = 0; pass < 2; pass++) {
      x.strokeStyle = pass ? SILVER : INK; x.lineWidth = pass ? Math.max(4, H * .012) : Math.max(12, H * .03);
      x.beginPath(); x.moveTo(bx, by); x.lineTo(x1, y1); x.lineTo(x2, y2); x.lineTo(x3, y3); x.stroke();
      x.beginPath(); x.moveTo(x3, y3); x.lineTo(x3 + jaw * Math.cos(t3 + ga), y3 - jaw * Math.sin(t3 + ga)); x.moveTo(x3, y3); x.lineTo(x3 + jaw * Math.cos(t3 - ga), y3 - jaw * Math.sin(t3 - ga)); x.stroke();
      x.beginPath(); x.rect(bx - plinth, by, plinth * 2, plinth * .6); x.stroke();
    }
    x.fillStyle = INK; x.fillRect(bx - plinth, by, plinth * 2, plinth * .6);
    const jr = Math.max(3, H * .009); x.fillStyle = SILVER;
    x.beginPath(); x.arc(bx, by, jr, 0, 7); x.fill(); x.beginPath(); x.arc(x1, y1, jr, 0, 7); x.fill(); x.beginPath(); x.arc(x2, y2, jr, 0, 7); x.fill(); x.beginPath(); x.arc(x3, y3, jr, 0, 7); x.fill();
    x.globalAlpha = 1;
    // tag, top left
    x.fillStyle = SILVER; x.textBaseline = "top"; x.textAlign = "left"; x.font = A.fontTag;
    x.fillText("DYSTOPIAN   ·   SECTOR 01   ·   ARM LAB", A.pad, A.pad);
    x.textAlign = "right"; x.fillText(A.live ? "FOLLOWER · LIVE" : "FOLLOWER · NO STREAM", W - A.pad, A.pad);
    // status card, bottom left: plate band, then flat declarative text
    const bandY = H - A.pad - A.bigPx - Math.round(H * .055);
    x.fillStyle = SILVER; x.fillRect(A.pad, bandY, Math.max(80, W * .08), 3);
    // why: on the wall's compact layout the panel is ~180 px tall and the caption collided with the card text; keep the plate band, drop the words
    x.textAlign = "left";   // the tag line above draws right-aligned; the band and card are left-aligned whether or not the caption is shown
    if (H >= 240) { x.font = A.fontSmall; x.fillStyle = SILVER; x.fillText("FACILITY STATUS", A.pad, bandY + 10); }
    x.font = A.fontCard; x.textBaseline = "alphabetic"; x.fillStyle = SAND;
    const txt = A.cards[A.card] || ""; x.fillText(txt, A.pad, H - A.pad);
    // card index squares, bottom right; the current one fills over its 6 s
    const sq = Math.max(8, H * .016), gap = sq * .6; let sx0 = W - A.pad - A.ncards * (sq + gap) + gap;
    const frac = Math.min(1, (now - A.cardAt) / 6000);
    for (let i = 0; i < A.ncards; i++, sx0 += sq + gap) {
      x.strokeStyle = SILVER; x.lineWidth = 1; x.strokeRect(sx0 + .5, H - A.pad - sq + .5, sq - 1, sq - 1);
      if (i < A.card) { x.fillStyle = SILVER; x.fillRect(sx0, H - A.pad - sq, sq, sq); }
      else if (i === A.card) { x.fillStyle = SILVER; x.fillRect(sx0, H - A.pad - sq, sq * frac, sq); }
    }
    // hazard tick when recording
    if (txt.indexOf("RECORDING") === 0) { x.fillStyle = CLAY; x.fillRect(A.pad, bandY, Math.max(80, W * .08), 3); }
  }

  function frame(now) {
    if (!A.on) return;
    A.raf = requestAnimationFrame(frame);
    if (now - A.last < FRAME) return;        // <= 30 fps budget
    A.last = now;
    if (now - A.cardAt >= 6000) { A.card = (A.card + 1) % Math.max(1, A.ncards); A.cardAt = now; fitCard(); }
    const c = A.c, r = c.parentElement.getBoundingClientRect();
    if (c.width !== Math.round(r.width) || c.height !== Math.round(r.height) || A.w !== c.width || A.h !== c.height) { layout(); fitCard(); }
    try { draw(now); } catch (e) { /* never let a bad frame kill the kiosk */ }
  }

  const attract = {
    start(canvas, api) {
      A.c = canvas; A.x = canvas.getContext("2d"); A.api = api; A.on = true; A.last = 0; A.card = 0; A.cardAt = performance.now();
      layout(); buildCards(); fitCard(); addEventListener("resize", onResize);
      A.cardTimer = setInterval(buildCards, 1000);
      A.raf = requestAnimationFrame(frame);
    },
    stop() {
      A.on = false; if (A.raf) cancelAnimationFrame(A.raf); A.raf = 0;
      if (A.cardTimer) clearInterval(A.cardTimer); A.cardTimer = 0;
      removeEventListener("resize", onResize);
      if (A.x && A.c) { try { A.x.setTransform(1, 0, 0, 1, 0, 0); A.x.clearRect(0, 0, A.c.width, A.c.height); } catch (e) {} }
    },
  };

  const labels = {
    "#statusp h2": "SECTOR 01 · STATUS",
    "#jetsonp h2": "COMPUTE",
    "#armp h2": "ARM BAY",
    "h2:has(#tr_sub)": "DATA VAULT · LOSS",
    "h2:has(#ds_sub)": "DATA VAULT",
    "h2:has(#pol_sub)": "POLICIES",
    "#recnext": "NEXT", "#recredo": "REDO", "#recstop": "END",
    "header h1 small": "DYSTOPIAN · ARM LAB",
    "#lockscreen .brand": "DYSTOPIAN",
    "#lockscreen .brand small": "MACHINE-CUT. HUMAN-FINISHED.",
  };

  window.SKINS.register("dystopian", { attract, labels, apply, remove });
})();
