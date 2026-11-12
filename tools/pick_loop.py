#!/usr/bin/env python3
"""pick_loop.py - repeat "pick the nut, place it on the mat centre" and get better each time (iterative learning).

Per run: find the nut in the overhead frame (dark blob near its last known spot; failing that the scene map or a whole-mat search),
build a plan from /api/ik, bias every waypoint by the servo sag measured at the SAME waypoint last time (iterative learning control),
shift the place target by the camera-measured landing error of the previous run (outcome learning), run jetson/pick_seq.py over ssh in
one bus session per stage, then measure where the nut landed. Stops when the nut is lost, not picked up, not in the jaws, or the error grows.

Closed loop (default, docs/pick-loop.md): the open-loop command reaches the hover 5-8 mm short along the reach direction (10 Nov 2026),
but the servo readback plus the arm fit is good to a millimetre, so the pick is staged pick_seq runs: (a) hover, read back the
reached joints, FK them to the mat, correction c1 = commanded - reached; (b) corrected hover + grasp height with the jaws open, read
back again, c2, and the tip height h decides whether the grasp target is lowered (h > 6 mm: by min(h - 3, 6) mm through /api/ik);
(c) grasp at the twice-corrected point, close in two steps (20, then 0: the moving jaw meets the nut slowly) and lift; (d) the wrist
camera looks for the nut between the jaw tips (jaw check); not there -> ONE more corrected grasp (b + c); still not there -> open over
the spot and rest, `picked=false, reason="not_in_jaws"`; (e) carry, place, open, lift, rest. `--open-loop` is the single-run plan of before.

  tools/pick_loop.py --runs 4 [--target 9 6] [--open-loop] [--jaw-open 35] [--grasp-aim-mm 10] [--grasp-tilt-deg -3] [--grasp-roll auto] [--wrist-check] [--no-jaw-check] [--fast-close]
                     [--jaw-window 300 300 380 360] [--lost-after 3] [--state jetson/calib/pick_state.json]
  tools/pick_loop.py --reset [--safe-box 6 4 13 8] [--seed N]      # put the nut at a random clear spot inside the safe box
State (biases, place offset, history) persists in --state so later sessions start where this one ended.
Every pick is also one MLflow run in experiment `so101-picks` (jetson/labtrack.py: params, landing error, per-waypoint sag, the stage
corrections, the overhead frame at landing, the jaw-check frames, the plan and the state file); with no tracking server the loop runs as before.
"""
import argparse, json, math, os, random, select, subprocess, sys, time, urllib.error, urllib.request
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.environ.get("SO101_DASH", "http://localhost:8765")
REST = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 72.0, "wrist_roll": 0.5}
# why: at 0.12 in (3 mm by the IK) the jaws closed about 2 mm above the mat over a ~3 mm thick nut and rode over it (11 Nov 2026; a touch-off put mat contact at an
# IK height of about 0.5 mm); the 16:20 test, grasp and close commanded at 1.0 mm, gripped the nut. The closed loop's lower/raise rule is unchanged.
GRASP_H_IN = 1.0 / 25.4
# why: the staged grasp used to re-aim at grasp height, so the readback correction (c2, 2-6.5 mm on 10 Nov) dragged the fixed jaw along the mat
# into the nut: TC saw it "land a 1/4 in too far back, then slide up to the nut and hit it". The last stage now rises to this height over the
# corrected point first and comes straight down.
PREGRASP_H_IN = 0.4
# why: jaws at 35 grasp the nut; at 45 the SAME twice-corrected pose failed twice (10 Nov 2026): the moving jaw's closing arc flicked the nut 5-13 mm before the fixed jaw met it
JAW_OPEN = 35
# why: the two-step close parks the moving jaw at 20 (just clear of a 10 mm nut) first, so the final squeeze starts from a standstill next to the nut instead of flicking it (open-loop runs landed 10-40 mm off partly because the nut slid in the single-moving-jaw gripper)
CLOSE_MID = 20
# why: the arm fit's tip is where the CLOSED jaws meet (the fixed jaw's inner face), so a grasp aimed at the nut's centre brought the fixed jaw
# down on the nut's near half and pushed it forward into the gap (10 Nov 2026, seen by TC). The grasp puts the nut this far into the open gap
# instead. 10 mm (the first value: the nut's ~4 mm radius plus clearance) was too much: TC watched the fixed jaw land "a 1/4 in too far back,
# then slide up to the nut and hit it" (session first10c, 10 Nov 2026), so 10 - 6.35 = 3.65 -> 4 mm, the fixed jaw's face at the nut's edge.
GRASP_AIM_MM = 4.0
# why: at 1.0 mm the front finger (the moving jaw, away from the base) dragged on the mat and pulled the fixed thumb forward (TC, 11 Nov 2026: "the wrist needs to tilt a bit so
# the front finger doesn't drag"). A -3 deg wrist_flex swings the jaw tips away from the base (~2.8 mm per degree at the grasp), so the front finger rides higher than the fixed
# thumb; the IK target is shifted back by the tilt's own tip shift so the tilted tip lands where the untilted one would. 16:35 test pick: picked and placed, 5.9 mm from the target.
GRASP_TILT_DEG = -3.0
TILT_WPS = ("pregrasp", "grasp", "close1", "close2", "close")   # not hover_nut, lift, hover_tgt, place, open, lift2, rest
# why: 11 Nov 2026 18:2x a ~6 x 13 mm spacer lying along the reach was pushed 1.8 in instead of picked: the jaws close along jaw_dir, which with wrist_roll fixed is roughly the reach direction,
# so they closed along its length. --grasp-roll turns the wrist so jaw_dir is perpendicular to the object's long axis; the nut-side waypoints carry the roll (hover_pre and pregrasp_up through
# hover_nut / pregrasp), the carry and rest do not. Off by default.
ROLL_WPS = ("hover_nut", "pregrasp", "grasp", "close1", "close2", "close", "lift")
ROLL_MIN_RATIO = 1.3     # long / short below this is round: no roll
ROLL_OBJ_MM = 15.0       # --grasp-roll auto: the scene object used must be this close to the nut position
# why: the final stage goes from the stage-2 grasp pose UP to pregrasp, and lifting is the servo's bad side (P-only, I=0: it stops short; the 11:56 sweep on 11 Nov 2026 measured
# 1.7-4.7 deg of droop on the shoulder, 3 deg on the elbow). A live tilt pick had a pregrasp sag of 4.35 deg, and the commanded pregrasp -> grasp descent is only ~1.4 deg on shoulder_lift
# and elbow_flex, less than the droop, so the real shoulder move was a reversal from the bad side: the grasp landed 6.4 mm high and 4 mm past the aim. A final move that INCREASES
# shoulder_lift and elbow_flex landed within 0.04-0.6 deg. pregrasp_up is pregrasp's command with both reduced by APPROACH_DEG: with the arm fit at 5 mat spots it sits 50-63 mm above
# the mat and 33-40 mm back, the straight joint path down to pregrasp never goes below pregrasp's 10 mm, and 8 deg beats the largest measured droop (4.7 deg).
APPROACH_DEG = 8.0
APPROACH_JOINTS = ("shoulder_lift", "elbow_flex")
# why: the shoulder_pan servo has stiction: every move stops short in the direction it was moving (live pick 11 Nov 2026 17:17, closed loop with the -3 deg tilt): stage 1 hover commanded 9.280 reached 8.830 in (11 mm short);
# stage 2 grasp pose after c1 commanded 9.728 reached 9.613 (3 mm short); stage 3 final grasp after c2 commanded 9.393 reached 9.548: the arm moved 1.6 mm of a 5.6 mm correction, about 4 mm (~1 deg at ~230 mm reach) short, and the
# final grasp landed 7.5 mm off the aim. The readback corrections c1, c2 assume the next arrival behaves like the last one, which only holds if every arrival at a pan target comes from the SAME side. Pan has no gravity side, so
# the back-off picks one: every nut-side target is approached from PAN_APPROACH_DEG lower pan (the last pan move always increases), 3 deg beating the ~1 deg stiction band with margin (~16 mm sideways at 300 mm reach).
PAN_APPROACH_DEG = 3.0
# back-off waypoint -> (the waypoint whose command it copies, degrees taken off each joint): never a mat point, never learned (NO_LEARN)
BACKOFF = {"pregrasp_up": ("pregrasp", {**{k: APPROACH_DEG for k in APPROACH_JOINTS}, "shoulder_pan": PAN_APPROACH_DEG}), "hover_pre": ("hover_nut", {"shoulder_pan": PAN_APPROACH_DEG})}
WP = [("hover_pre", "nut", 1.5, JAW_OPEN, 1.6, 0.4), ("hover_nut", "nut", 1.5, JAW_OPEN, 0.5, 0.4), ("pregrasp_up", "pregrasp", PREGRASP_H_IN, JAW_OPEN, 0.8, 0.3), ("pregrasp", "nut", PREGRASP_H_IN, JAW_OPEN, 0.6, 0.3), ("grasp", "nut", GRASP_H_IN, JAW_OPEN, 0.9, 0.5),
      ("close1", "nut", GRASP_H_IN, CLOSE_MID, 0.5, 0.4), ("close2", "nut", GRASP_H_IN, 0, 0.6, 0.6),
      ("lift", "nut", 1.5, 0, 0.8, 0.2), ("hover_tgt", "tgt", 1.5, 0, 1.2, 0.3), ("place", "tgt", 0.3, 0, 0.9, 0.5),
      ("open", "tgt", 0.3, JAW_OPEN, 0.5, 0.4), ("lift2", "tgt", 1.5, JAW_OPEN, 0.7, 0.2), ("rest", None, 0, 0, 1.8, 0.3)]
WP_FAST_CLOSE = ("close", "nut", GRASP_H_IN, 0, 0.6, 0.4)   # --fast-close: the single-step close of before, in place of close1 + close2
GRIPPER_ONLY = ("close", "close1", "close2", "open")          # why: pure jaw moves say nothing about arm sag; the bias learning skips them
GRASP_HEIGHT_WPS = ("grasp", "close", "close1", "close2")      # why: the close waypoints sit at the grasp height too, lowered or not (a close at the unlowered height would lift the tip while squeezing)
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
NO_LEARN = tuple(BACKOFF)   # why: pregrasp_up and hover_pre are defined from another waypoint's command, not mat points: their droop is not sag to learn, and they must not change the biases
STAGES = (("hover_pre", "hover_nut"), ("hover_pre", "hover_nut", "grasp"), ("pregrasp_up", "pregrasp", "grasp", "close1", "close2", "lift", "hover_tgt", "place", "open", "lift2", "rest"))
LIFT_STAGE, CARRY_STAGE, ABORT_STAGE = ("pregrasp_up", "pregrasp", "grasp", "close1", "close2", "lift"), ("hover_tgt", "place", "open", "lift2", "rest"), ("hover_pre", "hover_nut", "rest")
# why: 11 Nov 2026 22:2x the stage-2 height reading did not carry to the final approach (pregrasp_up -> pregrasp -> grasp lands up to 4 mm off it: 1.6 vs 5.4, 5.4 vs 2.5 mm) and several misses closed 3.4-7 mm above a ~3 mm nut,
# so with a persistent session (a stage costs ~0.2 s) the final stage is split at the grasp: read the tip back there, and send ONE corrective move at grasp height when it is off (docs/pick-loop.md)
PRE_STAGE, CORR_STAGE, CLOSE_STAGE = ("pregrasp_up", "pregrasp", "grasp"), ("grasp",), ("close1", "close2", "lift")
FINAL_H_TARGET_MM, FINAL_H_BAND_MM = 1.0, (0.5, 2.0)   # the corrective move fires when the grasp readback is outside the band; it aims at the target
FINAL_XY_TOL_MM, FINAL_XY_CAP_MM = 1.5, 4.0            # ... or when the xy residual against the aim is over the tolerance; the xy move is capped
LOWER_CAP_MM = (-5.0, 6.0)                              # (raise, lower) caps of the grasp height, the same as the stage-2 rule's
# why: 11 Nov 2026 live: the gripper readback (0-100) after the lift is 0.5-1.5 empty, 4.0-4.8 holding a nut across its flats or a 6 mm spacer, 1.3-1.5 holding a wing nut by a thin wing (= empty); close2 read 4.49 once but the lift 0.54 (the nut slipped out), so only the reading after the lift counts
GRIP_EMPTY_MAX = 2.0
SPEED = {"closed": 50.0, "open": 80.0}   # why: the staged runs start each stage from a standstill close to the nut; 50 deg/s settles cleaner for the readback
JAW_HOLE_MAX = 100.0   # mean grey (0-255) allowed in a ring's hole inside the jaw window: the held nut's hole measured 19-56, the bright jaw edge 204-222 (frames 11 Nov 2026)
JAW_WINDOW = (300, 300, 380, 360)   # wrist-frame pixels (x0, y0, x1, y1) between the jaw tips after close + lift (diag frames, 10 Nov 2026: the nut's ring sat at about (340, 325))
SAFE_BOX = (6.0, 4.0, 13.0, 8.0)   # --reset drops the nut inside this (mat inches): reachable, clear of the markers and of the mat edges
MAT_IN = (18.0, 12.0)
NUT_W_MM = (8.0, 20.0)   # a scene-map object this wide (long side) with a short side <= 20 mm is a nut candidate (an M4 nut reads 10-12 mm)
RESET_MM = {"avoid": 40.0, "marker": 50.0, "move": 60.0}


def api(path):
    return json.load(urllib.request.urlopen(DASH + path, timeout=10))


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "jetson" if name != "wrist_check" else "tools", f"{name}.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def cam_calib(): return _load("cam_calib")


def wrist_mod(): return _load("wrist_check")


def labtrack(): return _load("labtrack").LabTrack("so101-picks")


_SERVO = []
def servo_tele():
    """jetson/servo_telemetry.py (the decoder and the Summary the Jetson side fills); loaded once."""
    if not _SERVO: _SERVO.append(_load("servo_telemetry"))
    return _SERVO[0]


def tele_of(rows):
    """One summary of the servo telemetry pick_seq attached to these waypoint rows (None when the Jetson sent none): peak temperature, load, current, sag, lowest voltage."""
    try: return servo_tele().merge([r.get("tele") for r in rows]) if any(r.get("tele") for r in rows) else None
    except Exception: return None   # why: health numbers are a bonus; a malformed row must never cost the pick


def fetch_top(tries=6, wait_s=0.5, sleep=None):
    """One fresh overhead JPEG from the dashboard (bytes). The dashboard answers 404 while its newest frame is older than 3 s, so a short
    feed hiccup is retried (tries x wait_s) before the error is raised. why: 11 Nov 2026 18:2x a 404 right after the place crashed the pick
    after the arm had finished, before the record and the state were written."""
    sleep = sleep or time.sleep
    for i in range(tries):
        try: return urllib.request.urlopen(DASH + f"/cam/top.jpg?t={time.time():f}", timeout=5).read()
        except urllib.error.HTTPError as e:
            if e.code != 404 or i == tries - 1: raise
            sleep(wait_s)


def fetch_wrist():
    """One fresh wrist-camera JPEG from the dashboard (bytes), or None when the feed is down (the checks are optional)."""
    try: return urllib.request.urlopen(DASH + f"/cam/wrist.jpg?t={time.time():f}", timeout=5).read()
    except Exception: return None


def fetch_scene_objects():
    """The scene map's objects (/api/scene, docs/mat-scene.md), or [] when the dashboard, the camera or the background is missing."""
    try: r = api("/api/scene"); return list(r.get("objects") or []) if r.get("ok", True) else []
    except Exception: return []


def fetch_scene_state():
    """(objects, pending) of /api/scene: the nut-like or not objects the map confirmed and how many it is still confirming (SCENE_MIN_HITS, ~1.5 s); ([], 0) when the dashboard, the camera or the background is missing."""
    try: r = api("/api/scene"); return (list(r.get("objects") or []), int(r.get("pending") or 0)) if r.get("ok", True) else ([], 0)
    except Exception: return [], 0


class Found(tuple):
    """find_nut's (mat_in, pixel, area) with `.source` ("scene" | "dark") saying which search answered."""
    source = None


def find_nut_near(cc, table, near_in, radius_px=70, radius_mm=None, wait_s=3.0, jpeg=None, joints="live", sleep=None, dark=True):
    """The nut near `near_in` (mat inches) -> Found (mat_in, pixel, area; `.source` "scene" | "dark") or None. The scene map first: poll /api/scene every
    0.5 s while it reports pending objects (at most `wait_s`; a new object needs ~1.5 s), take the nut-like object nearest `near_in` within `radius_mm`
    (default: what a `radius_px` box covers). None there: the dark-blob search find_nut with the same radius (dark=False: None). Never raises on a dashboard error.
    why: 11 Nov 2026 the dark-blob search kept taking fragments of the printed squares for the nut (6.78, 4.81) and never saw the real one on a light square
    (10.17, 7.32), while the scene map found it every time."""
    sleep = sleep or time.sleep
    try:
        if radius_mm is None:
            try: u, v = cc.table_to_pixel(table, near_in[0] * 25.4, near_in[1] * 25.4); (x0, y0), (x1, y1) = cc.pixel_to_table(table, u, v), cc.pixel_to_table(table, u + 1.0, v); radius_mm = radius_px * math.hypot(x1 - x0, y1 - y0)
            except Exception: radius_mm = radius_px * 2.0   # the mat camera's nominal 0.5 px/mm
        objs, pend = fetch_scene_state(); waited = 0.0
        while pend > 0 and waited < wait_s: sleep(0.5); waited += 0.5; objs, pend = fetch_scene_state()
        c = [(math.hypot(float(o["x_in"]) - near_in[0], float(o["y_in"]) - near_in[1]) * 25.4, o) for o in nut_candidates(objs) if "x_in" in o and "y_in" in o]
        c = [t for t in c if t[0] <= radius_mm]
        if c:
            o = min(c, key=lambda t: t[0])[1]; xy = [float(o["x_in"]), float(o["y_in"])]; u, v = cc.table_to_pixel(table, xy[0] * 25.4, xy[1] * 25.4)
            f = Found((xy, [float(u), float(v)], None)); f.source = "scene"; return f
    except Exception as e: print(json.dumps({"find_nut_near": "scene failed", "err": f"{type(e).__name__}: {e}"}), file=sys.stderr, flush=True)
    if not dark: return None   # scene only: the callers try the scene map at every hint before any dark-blob search
    kw = {"jpeg": jpeg} if jpeg is not None else {}; kw.update({"joints": joints} if joints != "live" else {})
    d = find_nut(cc, table, near_in, radius_px=radius_px, **kw)
    if d is None: return None
    f = Found(d); f.source = "dark"; return f


def find_nut_hints(cc, table, hints, sleep=None):
    """The nut near any of `hints` ((near_in, radius_px), ...) -> Found or None: the scene map at EVERY hint first, then the dark blob at every hint.
    why: 11 Nov 2026 asking hint by hint let a printed-square fragment found by the dark search at the first hint beat a nut the scene map saw from the second."""
    return (next((f for f in (find_nut_near(cc, table, h, radius_px=r, sleep=sleep, dark=False) for h, r in hints) if f), None)
            or next((f for f in (find_nut_near(cc, table, h, radius_px=r, sleep=sleep, wait_s=0.0) for h, r in hints) if f), None))


def find_nut(cc, table, near_in, radius_px=70, jpeg=None, joints="live"):
    """Dark blob nearest to `near_in` (mat inches) in a fresh overhead frame (or the given JPEG bytes) -> (mat_in, pixel, area) or None. `joints`: the
    follower's joints for the arm cut-out, as in find_nut_mat ("live" reads /api/state; None skips it; a dict is used as is). why: 11 Nov 2026 13:45 the
    near search took a dark slot on the gripper (arm parked over the mat centre) for the nut, 14 mm from the real one."""
    img = cv2.imdecode(np.frombuffer(fetch_top() if jpeg is None else jpeg, np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); cu, cv_ = cc.table_to_pixel(table, near_in[0] * 25.4, near_in[1] * 25.4)
    x0, y0 = max(0, int(cu) - radius_px), max(0, int(cv_) - radius_px); roi = g[y0:y0 + 2 * radius_px, x0:x0 + 2 * radius_px]
    keep = arm_cut(cc, table, np.full(g.shape, 255, np.uint8), follower_joints() if joints == "live" else joints)[y0:y0 + 2 * radius_px, x0:x0 + 2 * radius_px]   # why: an arm_cut failure leaves the mask whole (no cut), never breaks the search
    m = ((roi < 60) & (keep > 0)).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    H, W = m.shape; whole = lambda i: st[i, cv2.CC_STAT_LEFT] > 0 and st[i, cv2.CC_STAT_TOP] > 0 and st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] < W and st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] < H
    # why: 11 Nov 2026 16:32 the box's edge cut a strip off a printed dark square at mat (6.74, 5.93) and the search took it for the nut; a nut lies inside the box, not cut by it
    cands = [(math.hypot(x0 + cen[i][0] - cu, y0 + cen[i][1] - cv_), int(st[i, cv2.CC_STAT_AREA]), x0 + cen[i][0], y0 + cen[i][1]) for i in range(1, n) if 8 <= st[i, cv2.CC_STAT_AREA] <= 300 and whole(i)]
    if not cands: return None
    _, area, u, v = min(cands); X, Y = cc.pixel_to_table(table, u, v)
    return [X / 25.4, Y / 25.4], [u, v], area


# ---------------------------------------------------------------- the whole-mat finder (2B)
def load_layout():
    """jetson/calib/top_layout.json when present (the marker centres and size the mat was glued to), else None."""
    p = os.path.join(ROOT, "jetson", "calib", "top_layout.json")
    try: return json.load(open(p)) if os.path.exists(p) else None
    except (OSError, ValueError): return None


def _units_to_in(units): u = str(units or "mm").lower(); return 1.0 if u.startswith("in") else (10.0 / 25.4 if u == "cm" else 1.0 / 25.4)


def marker_centres_in(table, layout=None):
    """Marker square centres in mat inches: from the layout file when given (its own units), else the table calibration's layout_markers (mm)."""
    if layout and layout.get("markers"): k = _units_to_in(layout.get("units")); return [[round(float(x) * k, 6), round(float(y) * k, 6)] for x, y in layout["markers"].values()]  # why: rounded so 76.2 mm is 3.0 in, not 2.9999999999999996
    return [[round(float(x) / 25.4, 6), round(float(y) / 25.4, 6)] for x, y in (table.get("layout_markers") or {}).values()]


def marker_size_in(table, layout=None):
    if layout and layout.get("marker_size") is not None: return float(layout["marker_size"]) * _units_to_in(layout.get("units"))
    return float((layout or table).get("marker_size_mm") or 50.8) / 25.4


def mat_mask(cc, table, shape, markers_in, marker_in=2.0, margin_mm=10.0, mat_in=MAT_IN):
    """uint8 mask (255 = search here) of the mat area in the overhead frame with the marker squares (+ margin) cut out."""
    m = np.zeros(shape[:2], np.uint8)
    def poly(pts): return np.array([cc.table_to_pixel(table, x * 25.4, y * 25.4) for x, y in pts], np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [poly([(0.0, 0.0), (mat_in[0], 0.0), (mat_in[0], mat_in[1]), (0.0, mat_in[1])])], 255); hm = marker_in / 2.0 + margin_mm / 25.4
    for cx, cy in markers_in: cv2.fillPoly(m, [poly([(cx - hm, cy - hm), (cx + hm, cy - hm), (cx + hm, cy + hm), (cx - hm, cy + hm)])], 0)
    return m


def follower_joints():
    """The follower's joints (LeRobot degrees) from /api/state, or None when the dashboard or the arm stream is down."""
    try:
        pos = ((api("/api/state").get("arms") or {}).get("follower") or {}).get("pos") or {}
        return {k: float(pos[k]) for k in ARM} if all(k in pos for k in ARM) else None
    except Exception: return None


def arm_cut(cc, table, mask, joints):
    """`mask` with the arm cut out: cam_calib.arm_mask_px_from_joints (links, wrist camera, cables, +30 % for the shadow), each point a filled
    circle and consecutive points joined by a stroke 2 x the larger radius, as the scene map's kinematic mask does. Unchanged without joints."""
    if not joints: return mask
    try: arm = cc.arm_mask_px_from_joints(joints, table)
    except Exception: return mask
    P = [(int(round(u)), int(round(v)), max(1, int(math.ceil(r)))) for (u, v), r in zip(arm.get("points") or [], arm.get("radii_px") or []) if all(math.isfinite(x) for x in (u, v, r))]
    for u, v, r in P: cv2.circle(mask, (u, v), r, 0, -1)
    for (u0, v0, r0), (u1, v1, r1) in zip(P, P[1:]): cv2.line(mask, (u0, v0), (u1, v1), 0, thickness=2 * max(r0, r1))
    return mask


def find_nut_mat(cc, table, near_in=None, jpeg=None, markers_in=None, marker_in=2.0, dark=60, area=(8, 300), joints="live"):
    """find_nut over the WHOLE mat (markers AND the arm masked): the dark blob nearest `near_in` (else the mat centre) -> (mat_in, pixel, area)
    or None. Never raises. `joints`: the follower's joints for the arm cut-out ("live" reads /api/state; None skips it). why: 10 Nov 2026, with
    no hint the search took the parked arm's dark parts at mat (8.1, 3.8) for the nut, closer to the mat centre than the real nut; first10b
    cycles 9-10 went there and found nothing in the jaws."""
    try:
        img = cv2.imdecode(np.frombuffer(fetch_top() if jpeg is None else jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); mask = mat_mask(cc, table, g.shape, marker_centres_in(table) if markers_in is None else markers_in, marker_in)
        mask = arm_cut(cc, table, mask, follower_joints() if joints == "live" else joints)
        n, lab, st, cen = cv2.connectedComponentsWithStats(((g < dark) & (mask > 0)).astype(np.uint8))
        ref = near_in if near_in else (MAT_IN[0] / 2.0, MAT_IN[1] / 2.0); cu, cv_ = cc.table_to_pixel(table, ref[0] * 25.4, ref[1] * 25.4)
        cands = [(math.hypot(cen[i][0] - cu, cen[i][1] - cv_), int(st[i, cv2.CC_STAT_AREA]), float(cen[i][0]), float(cen[i][1])) for i in range(1, n) if area[0] <= st[i, cv2.CC_STAT_AREA] <= area[1]]
        if not cands: return None
        _, a, u, v = min(cands); X, Y = cc.pixel_to_table(table, u, v); return [X / 25.4, Y / 25.4], [u, v], a
    except Exception as e:
        print(json.dumps({"find_nut_mat": "failed", "err": f"{type(e).__name__}: {e}"}), flush=True); return None


def nut_candidates(objects, w_mm=NUT_W_MM):
    """Scene-map objects shaped like the nut: long side 8..20 mm, short side <= 20 mm."""
    out = []
    for o in objects or []:
        try:
            if w_mm[0] <= float(o["w_mm"]) <= w_mm[1] and float(o["h_mm"]) <= w_mm[1]: out.append(o)
        except (KeyError, TypeError, ValueError): continue
    return out


def find_nut_anywhere(cc, table, scene_objects=None, near_in=None, jpeg=None, markers_in=None):
    """The nut anywhere on the mat: the scene map first (/api/scene objects, or the given list; the candidate nearest `near_in`, else
    any), then the whole-mat dark-blob search with the markers masked. Returns (xy_in, source) with source "scene" | "mat", or None."""
    c = nut_candidates(fetch_scene_objects() if scene_objects is None else scene_objects)
    if c:
        o = min(c, key=lambda o: math.hypot(float(o["x_in"]) - near_in[0], float(o["y_in"]) - near_in[1])) if near_in else c[0]
        return [float(o["x_in"]), float(o["y_in"])], "scene"
    f = find_nut_mat(cc, table, near_in, jpeg, markers_in)
    return (f[0], "mat") if f else None


# ---------------------------------------------------------------- plans and the arm
def waypoints(fast_close=False):
    """WP, or WP with the single-step `close` in place of close1 + close2 (--fast-close)."""
    return list(WP) if not fast_close else [WP_FAST_CLOSE if w[0] == "close1" else w for w in WP if w[0] != "close2"]


_CALIB = {}
def calib():
    """(cam_calib, top_table) of this ROOT, loaded once: build_plan's tilt shift needs them and not every caller has them to hand."""
    if ROOT not in _CALIB: _CALIB[ROOT] = (cam_calib(), json.load(open(os.path.join(ROOT, "jetson", "calib", "top_table.json"))))
    return _CALIB[ROOT]


def build_plan(nut_in, tgt_in, biases, names=None, jaw_open=JAW_OPEN, grasp_h_in=None, fast_close=False, tilt_deg=0.0, cc=None, table=None, roll_deg=0.0):
    """The waypoints of WP (all, or only `names`, in WP order) through /api/ik with the learned biases; `jaw_open` replaces every
    open-jaw value, `grasp_h_in` the grasp (and close) height (the closed loop lowers it when the readback says the tip sits high).
    `tilt_deg` is added to wrist_flex on the TILT_WPS; their IK target is shifted back by the tilt's own tip shift (FK through the arm
    fit, `cc`/`table` or calib()), so the tilted tip lands on the commanded point; `desired` carries the tilt (the sag learning must not learn it).
    `roll_deg` (0/None: the plan of before) is added to the arm fit's wrist_roll on the ROLL_WPS (the nut side; hover_pre and pregrasp_up through the waypoint they copy) and goes to
    /api/ik as &roll=, so the tip still lands on the commanded point (the tip is off the roll axis); the carry and rest keep the fit's roll."""
    roll_deg = float(roll_deg or 0.0); base_roll = float((((cc, table) if cc is not None and table is not None else calib())[1].get("arm_fit") or {}).get("wrist_roll_deg", 0.0)) if roll_deg else 0.0
    def ik(x, y, h, roll=None):
        r = api(f"/api/ik?x_in={x:.4f}&y_in={y:.4f}&h_in={h:.4f}" + ("" if roll is None else f"&roll={roll:.4f}"))
        if not r.get("ok"): raise SystemExit(f"ik failed for ({x:.2f},{y:.2f},{h}): {r.get('err')}")
        return r["joints_deg"]
    want = None if names is None else set(names) | ({"close"} if {"close1", "close2"} & set(names) else set())   # why: a stage naming the two-step close also names its --fast-close stand-in
    def arm_wp(name, which, h):   # (desired, command) joints of one mat-point or rest waypoint: IK, the tilt's shift, the bias, the clamp
        tilt = float(tilt_deg or 0.0) if name in TILT_WPS else 0.0; pt = nut_in if which == "nut" else tgt_in
        roll = base_roll + roll_deg if (roll_deg and which == "nut" and name in ROLL_WPS) else None
        j = dict(REST) if which is None else ik(*pt, h, roll)
        if tilt and which is not None:
            c, t = (cc, table) if cc is not None and table is not None else calib()
            for _ in range(2):   # why: the shift barely depends on the pose, so twice is enough (11 Nov 2026); the tilted tip must land on the commanded point
                # ... in height too: xy alone left the tilted tip -2.8 mm (8, 5) to +1.75 mm (14, 8) off the commanded height (arm fit), and the grasp height is what made the 11 Nov misses
                x0, y0, h0 = fk_mat(c, t, j); x1, y1, h1 = fk_mat(c, t, dict(j, wrist_flex=j["wrist_flex"] + tilt)); j = ik(pt[0] - (x1 - x0), pt[1] - (y1 - y0), h - (h1 - h0) / 25.4, roll)
            j = dict(j, wrist_flex=j["wrist_flex"] + tilt)
        b = biases.get(name, {}); cmd = {k: j[k] + b.get(k, 0.0) for k in j}; cmd["wrist_flex"] = max(-93.0, min(93.0, cmd["wrist_flex"])); return j, cmd
    wps = []
    for name, which, h, grip, secs, settle in waypoints(fast_close):
        if want is not None and name not in want: continue
        if name in GRASP_HEIGHT_WPS and grasp_h_in is not None: h = grasp_h_in
        if name in BACKOFF:   # pregrasp_up / hover_pre: the target's final command with the approach joints backed off, so the move on to the target increases them
            tgt_wp, drop = BACKOFF[name]; j, cmd = arm_wp(tgt_wp, "nut", h); j, cmd = ({k: v - drop.get(k, 0.0) for k, v in d.items()} for d in (j, cmd))
        else: j, cmd = arm_wp(name, which, h)
        wps.append({"name": name, "joints": cmd, "desired": j, "gripper": jaw_open if grip == JAW_OPEN else grip, "seconds": secs, "settle": settle, "gripper_only": name in GRIPPER_ONLY})
    return {"waypoints": wps}


PLAN_TIMEOUT_S = 180
PICK_SEQ_ARGS = ""   # why: extra pick_seq flags, set by tools/autoloop.py --record-demos to "--tick-sink PATH" before the session starts, empty otherwise (2026-11-12); --serve writes one sink file (with its header line) per plan, like --plan runs
SERVE_CMD = "cd ~/so101 && source env.sh >/dev/null 2>&1; exec ~/so101/.venv/bin/python pick_seq.py --serve --idle-s 60"
SESSION = None   # why: 12 Nov 2026 a pick_seq session costs 5.4 s of start-up and a closed-loop pick has 3-5 stages; main() / autoloop set this to a PickSession (--session, default on), run_plan then reuses one connection


class SessionDead(Exception):
    """The persistent pick_seq is gone. `rows` = the waypoint rows of THIS plan already received (non-empty: the arm moved, never replay it silently)."""
    def __init__(self, msg, rows=()): super().__init__(msg); self.rows = list(rows)


class PickSession:
    """One `ssh jetson ... pick_seq.py --serve` kept for many plans (jetson/pick_seq.py --serve): started lazily on the first plan, plans go in as JSON lines on stdin, rows and
    {"plan_done": n} come back on stdout. `argv` / `popen` are for tests. A session that never completed a plan (old pick_seq without --serve, arm busy) is `broken`: run_plan uses
    the scp+ssh path from then on. One that ran plans and then ended (60 s idle timeout, ssh drop) is simply restarted for the next plan."""
    def __init__(self, argv=None, timeout=PLAN_TIMEOUT_S, popen=subprocess.Popen):
        self.argv = argv or ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "jetson", SERVE_CMD + ((" " + PICK_SEQ_ARGS) if PICK_SEQ_ARGS else "")]
        self.timeout, self.popen, self.proc, self.buf, self.served, self.broken, self.starts = timeout, popen, None, b"", 0, False, 0

    def _start(self):
        self.proc = self.popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0); self.buf = b""; self.starts += 1

    def _line(self, deadline):
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0: raise subprocess.TimeoutExpired(self.argv, self.timeout)
            if not select.select([self.proc.stdout], [], [], min(left, 1.0))[0]: continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk: return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1); return line.decode("utf-8", "replace")

    def _run_once(self, plan, max_deg_s):
        if self.proc is None or self.proc.poll() is not None: self._start()
        rows = []; deadline = time.monotonic() + self.timeout
        try: self.proc.stdin.write((json.dumps({"waypoints": plan["waypoints"], "max_deg_s": max_deg_s, "correct": 0}) + "\n").encode()); self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e: raise SessionDead(f"pick_seq session: write failed ({e})") from e
        while True:
            try: line = self._line(deadline)
            except subprocess.TimeoutExpired: self.close(); raise   # why: the arm may still be moving; no replay, the caller trips
            if line is None: raise SessionDead("pick_seq session ended mid-plan" if rows else "pick_seq session ended before the plan ran", rows)
            try: d = json.loads(line)
            except ValueError: continue
            if not isinstance(d, dict): continue
            if "wp" in d: rows.append(d)
            elif "plan_done" in d: self.served += 1; return rows
            elif d.get("stopped"): self.close(); raise SystemExit("pick_seq was stopped (so101-stop); not retrying")
            elif d.get("fatal"): raise SessionDead("pick_seq session failed: " + str(d.get("error"))[:200], rows)
            elif "error" in d: raise SystemExit("pick_seq refused the plan: " + str(d["error"])[:200])
            elif "bye" in d: raise SessionDead(f"pick_seq session left ({d['bye']})", rows)

    def run(self, plan, max_deg_s):
        for attempt in (0, 1):
            try: return self._run_once(plan, max_deg_s)
            except SessionDead as e:
                self.close()
                if e.rows: raise
                if attempt == 0 and self.served: continue   # why: it served plans and then left (idle timeout): nothing ran, start a new one
                if not self.served: self.broken = True
                raise

    def close(self):
        p, self.proc = self.proc, None
        if p is None: return
        try: p.stdin.write(b'{"quit": true}\n'); p.stdin.flush(); p.stdin.close()
        except (BrokenPipeError, OSError, ValueError): pass
        try: p.wait(timeout=5)
        except Exception:
            try: p.terminate(); p.wait(timeout=3)
            except Exception:
                try: p.kill()
                except Exception: pass
        try: p.stdout.close()
        except Exception: pass


def run_plan(plan, max_deg_s, session=None):
    s = session or SESSION
    if s is not None and not s.broken:
        try: return s.run(plan, max_deg_s)
        except SessionDead as e:
            if e.rows: raise SystemExit(f"{e}; {len(e.rows)} rows of the plan arrived, not replaying it") from e   # why: the arm moved part of the plan; never a silent half-plan
            print(json.dumps({"note": "pick_seq session unavailable, one attempt on the scp+ssh path", "why": str(e)}), file=sys.stderr, flush=True)
    return run_plan_ssh(plan, max_deg_s)


def run_plan_ssh(plan, max_deg_s):
    p = "/tmp/pick_plan.json"; open("/tmp/pick_plan_local.json", "w").write(json.dumps(plan))
    subprocess.run(["scp", "-q", "/tmp/pick_plan_local.json", f"jetson:{p}"], check=True)
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "jetson", f"cd ~/so101 && source env.sh >/dev/null 2>&1; ~/so101/.venv/bin/python pick_seq.py --plan {p} --correct 0 --max-deg-s {max_deg_s}{(' ' + PICK_SEQ_ARGS) if PICK_SEQ_ARGS else ''}"], capture_output=True, text=True, timeout=PLAN_TIMEOUT_S)
    rows = []
    for line in out.stdout.splitlines():
        try: d = json.loads(line)
        except ValueError: continue
        if "wp" in d: rows.append(d)
    if not rows: raise SystemExit("pick_seq produced no waypoint rows:\n" + out.stdout[-800:] + out.stderr[-800:])
    return rows


def open_session(on=True):
    """Install a PickSession as the module's session (lazy: nothing connects until the first plan). Returns it, or None when off."""
    global SESSION
    SESSION = PickSession() if on else None; return SESSION


def close_session():
    global SESSION
    s, SESSION = SESSION, None
    if s is not None: s.close()


def fk_mat(cc, table, joints):
    """Where the tip really is for read-back joints: (mat x_in, mat y_in, height above the mat in mm), through the arm fit's zero offsets
    and tip offset. This is the measurement the closed loop corrects on (10 Nov 2026: it predicted the nut's wrist pixel within ~10 px)."""
    fit = table["arm_fit"]; off = fit.get("joint_offsets_deg") or {}
    tip, _ = cc.fk_tip({k: float(joints[k]) + float(off.get(k, 0.0)) for k in ARM}, tip_offset_mm=fit.get("tip_offset_mm"))
    x, y = cc.arm_to_table(table, float(tip[0]), float(tip[1])); return x / 25.4, y / 25.4, float(tip[2]) - float(fit.get("table_z_mm", 0.0) or 0.0)


def jaw_dir_mat(cc, table, joints):
    """Unit (mat x, mat y) from the fixed jaw toward the open moving jaw for read-back joints: gripper_link's +x axis on the mat (the URDF
    moving jaw pivots 28 mm on the +x side of the fixed tip and opens along +x; tests/tools/test_pick_aim.py checks it against the URDF
    jaw joint and the real wrist frame). None when that axis is near vertical (no direction on the mat)."""
    fit = table["arm_fit"]; G = cc.fk_frames({k: float(joints[k]) for k in ARM}, fit.get("joint_offsets_deg") or {})["gripper_link"]
    ax = G[:3, 0]; p0 = np.array(cc.arm_to_table(table, 0.0, 0.0)); p1 = np.array(cc.arm_to_table(table, float(ax[0]) * 1000.0, float(ax[1]) * 1000.0))
    d = p1 - p0; n = float(np.linalg.norm(d))   # mm of a 1 m axis: its horizontal share
    return None if n < 300.0 else (float(d[0] / n), float(d[1] / n))


def aim_point(nut_in, jaw_dir, aim_mm):
    """Where the fixed jaw's tip goes so the nut sits `aim_mm` inside the open gap: the nut moved back along the jaw direction (mat inches)."""
    if not jaw_dir or not aim_mm: return [float(nut_in[0]), float(nut_in[1])]
    return [nut_in[0] - aim_mm / 25.4 * jaw_dir[0], nut_in[1] - aim_mm / 25.4 * jaw_dir[1]]


def _scene_mod():
    """dashboard/scene_detect.py by path (not through ROOT, which the tests move): long_axis_mat lives next to the detector whose angle it reads."""
    import importlib.util
    if "_scene_detect" not in globals():
        spec = importlib.util.spec_from_file_location("scene_detect_for_pick", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "scene_detect.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); globals()["_scene_detect"] = m
    return globals()["_scene_detect"]


def long_axis_mat(obj, min_ratio=ROLL_MIN_RATIO): return _scene_mod().long_axis_mat(obj, min_ratio)


def parse_grasp_roll(v):
    """--grasp-roll: "off" (None / ""), "auto", or degrees -> "off" | "auto" | float (SystemExit on anything else)."""
    t = "off" if v is None or str(v).strip() == "" else str(v).strip().lower()
    if t in ("off", "auto"): return t
    try: return float(t)
    except ValueError: raise SystemExit(f"--grasp-roll wants auto, off or degrees, not {v!r}") from None


def roll_to_perpendicular(cc, table, joints, axis, span=90.0):
    """The wrist_roll OFFSET (degrees, |offset| <= span) that turns jaw_dir_mat of `joints` perpendicular to `axis` (a unit mat vector): the root of
    jaw_dir . axis closest to zero, found by scanning the offset and bisecting, with jaw_dir from FK through the arm fit (the sign of wrist_roll vs the
    jaw's turn on the mat is whatever the fit says, never assumed). None when jaw_dir is undefined (gripper not near vertical) or no root lies in range."""
    def f(r):
        u = jaw_dir_mat(cc, table, dict(joints, wrist_roll=float(joints["wrist_roll"]) + r)); return None if u is None else u[0] * axis[0] + u[1] * axis[1]
    grid = [i * 1.0 for i in range(-int(span), int(span) + 1)]; vals = [f(r) for r in grid]
    if any(v is None for v in vals): return None
    roots = []
    for (r0, v0), (r1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if v0 == 0.0: roots.append(r0)
        elif v0 * v1 < 0:
            lo, hi, flo = r0, r1, v0
            for _ in range(40):
                mid = (lo + hi) / 2.0; fm = f(mid)
                if fm * flo <= 0: hi = mid
                else: lo, flo = mid, fm
            roots.append((lo + hi) / 2.0)
    if vals[-1] == 0.0: roots.append(grid[-1])
    return min(roots, key=abs) if roots else None


def resolve_grasp_roll(spec, cc, table, nut_in, tilt_deg=0.0, objects=None):
    """(roll offset in degrees or None, info) for --grasp-roll `spec`. "off": (None, {}). A number: that offset. "auto": the scene-map object nearest
    `nut_in` (within ROLL_OBJ_MM; `objects` or /api/scene), its long axis, and roll_to_perpendicular at the IK pose for the grasp (tilted). No object,
    a round one or no axis: (None, info with why). info: {"spec", "axis", "obj", "why"} for the stage line and MLflow."""
    spec = parse_grasp_roll(spec)
    if spec == "off": return None, {}
    if spec != "auto": return float(spec), {"spec": spec}
    info = {"spec": "auto", "axis": None, "obj": None, "why": None}
    objs = fetch_scene_objects() if objects is None else objects
    near = sorted(((math.hypot(float(o["x_in"]) - nut_in[0], float(o["y_in"]) - nut_in[1]) * 25.4, o) for o in objs if "x_in" in o and "y_in" in o), key=lambda t: t[0])
    if not near or near[0][0] > ROLL_OBJ_MM: info["why"] = "no scene object at the nut position"; return None, info
    o = near[0][1]; info["obj"] = {k: o.get(k) for k in ("id", "x_in", "y_in", "w_mm", "h_mm", "angle_deg")}; ax = long_axis_mat(o)
    if ax is None: info["why"] = f"round (long/short < {ROLL_MIN_RATIO:g})"; return None, info
    j = api(f"/api/ik?x_in={nut_in[0]:.4f}&y_in={nut_in[1]:.4f}&h_in={GRASP_H_IN:.4f}")["joints_deg"]; j = dict(j, wrist_flex=j["wrist_flex"] + float(tilt_deg or 0.0))
    r = roll_to_perpendicular(cc, table, j, ax); info["axis"] = [round(ax[0], 3), round(ax[1], 3)]
    if r is None: info["why"] = "jaw_dir undefined at the grasp pose"; return None, info
    return round(r, 2), info


def _session_on(): return SESSION is not None and not SESSION.broken


def reached_joints(rows, name):
    r = [r for r in rows if r["wp"] == name][-1]; return {k: float(r["reached"][k]) for k in ARM}


def _mm(c): return [round(c[0] * 25.4, 1), round(c[1] * 25.4, 1)]


# ---------------------------------------------------------------- the jaw check (2A)
def gripper_after_lift(rows):
    """The gripper readback (0-100) of the lift waypoint of `rows`, or None (no rows, no lift row, no reading). Only this one counts: close2 can read high and the nut still slip out on the lift (11 Nov 2026)."""
    try:
        r = [r for r in (rows or []) if r.get("wp") == "lift"]; return float(r[-1]["reached"]["gripper"]) if r else None
    except Exception: return None


def grip_held(camera, gripper, grip_max=GRIP_EMPTY_MAX):
    """held = (gripper after the lift > grip_max) OR (the camera ring check says in_jaws). True / False, or None when the gripper reads empty and the camera is unknown (carry on, as before)."""
    if gripper is not None and gripper > grip_max: return True
    return camera


def jaw_check(jpeg=None, window=JAW_WINDOW, out_path=None, min_score=None, rows=None, grip_max=GRIP_EMPTY_MAX):
    """The object-agnostic grip check: the wrist-camera ring check (`_camera_check`, unchanged) OR the gripper readback after the lift (`rows`: the lift stage's rows;
    None: camera only). {"in_jaws" (the verdict), "camera" (the ring check's), "gripper" (the lift reading or None), ...the camera's keys}. Never raises."""
    res = _camera_check(jpeg, window, out_path, min_score); grip = gripper_after_lift(rows)
    res.update({"camera": res["in_jaws"], "gripper": None if grip is None else round(grip, 2)}); res["in_jaws"] = grip_held(res["camera"], grip, grip_max)
    if res["in_jaws"] is True and res["camera"] is not True: res["note"] = res["note"] or f"held by the gripper reading ({grip:.2f} > {grip_max})"
    return res


def _camera_check(jpeg=None, window=JAW_WINDOW, out_path=None, min_score=None):
    """Is the nut between the jaw tips? wrist_check.find_ring over the jaw window of a wrist frame taken after the lift (`jpeg`; None
    fetches one). Never raises: {"in_jaws": True | False | None, "ring": {uv, r_px, score} | None, "frame": out_path | None, "window", "note"};
    in_jaws None means unknown (no frame, a bad frame, a failure) and the caller carries on as if it were held."""
    res = {"in_jaws": None, "ring": None, "frame": None, "window": [int(v) for v in window], "note": None}
    try:
        jpeg = fetch_wrist() if jpeg is None else jpeg
        img = (cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if len(jpeg) else None) if isinstance(jpeg, (bytes, bytearray)) else jpeg
        if img is None: res["note"] = "no wrist frame; carrying unchecked"; return res
        x0, y0, x1, y1 = res["window"]; cu, cv_ = (x0 + x1) / 2.0, (y0 + y1) / 2.0; wm = wrist_mod()
        kw = {"min_score": min_score} if min_score is not None else {}
        ring = wm.find_ring(img, (cu, cv_), radius_px=math.hypot(x1 - x0, y1 - y0) / 2.0, rect=(x0, y0, x1, y1), hole_max=JAW_HOLE_MAX, **kw)   # why: the best ring INSIDE the rectangle; the disc's best was a false ring on the jaw edge at (357, 366) and hid the held nut at (335, 330) (11 Nov 2026 16:20)
        inside = ring is not None and x0 <= ring["uv"][0] <= x1 and y0 <= ring["uv"][1] <= y1   # why: the jaw gap is a rectangle, so a ring just outside it is a nut on the mat, not one held (kept as a guard)
        res.update({"in_jaws": bool(inside), "ring": ring if inside else None})
        if ring and not inside: res["note"] = "ring outside the jaw window"
        if out_path:
            raw = os.path.splitext(out_path)[0] + "_raw.jpg"; os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True); cv2.imwrite(raw, img)   # why: the annotated frame has circles drawn on it and cannot be a test fixture; keep the camera's pixels beside it (11 Nov 2026)
            out = img.copy(); cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)
            text = f"jaw window {x0}-{x1} x {y0}-{y1}  " + (f"nut in jaws: ring ({ring['uv'][0]:.0f},{ring['uv'][1]:.0f}) r{ring['r_px']} score {ring['score']:.2f}" if inside else "nut NOT in jaws")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True); cv2.imwrite(out_path, wm.annotate(out, (cu, cv_), ring, text)); res["frame"] = out_path
        return res
    except Exception as e:   # why: a check must never take the pick down with it; unknown -> carry, and the note says why
        res["note"] = f"jaw check failed: {type(e).__name__}: {e}"; return res


def final_fix(h_mm, res_in, lower_mm):
    """The final-pose correction decision: (xy move in mat inches, new lowered_mm) or None. Fires when the grasp readback height `h_mm` is outside FINAL_H_BAND_MM or the xy residual
    `res_in` (aim - reached, inches) is over FINAL_XY_TOL_MM; the xy move is the residual capped to FINAL_XY_CAP_MM, the height moves to FINAL_H_TARGET_MM within LOWER_CAP_MM."""
    off_h = not (FINAL_H_BAND_MM[0] <= h_mm <= FINAL_H_BAND_MM[1]); r_mm = math.hypot(*res_in) * 25.4; off_xy = r_mm > FINAL_XY_TOL_MM
    new_lower = max(LOWER_CAP_MM[0], min(LOWER_CAP_MM[1], lower_mm + (h_mm - FINAL_H_TARGET_MM))) if off_h else lower_mm
    if not off_xy and abs(new_lower - lower_mm) < 0.05: return None   # (off_h pinned at a cap: nothing to move)
    k = min(1.0, FINAL_XY_CAP_MM / r_mm) if r_mm > 0 else 1.0
    return (res_in[0] * k, res_in[1] * k), new_lower


def closed_loop_pick(cc, table, nut_in, place_in, biases, a, run, say, wpose=None, check_path=None, jaw_path=None):
    """The staged pick_seq runs of the closed loop (module docstring). Returns (plan, rows, info): plan and rows span the final stages
    (grasp .. lift, then carry .. rest, or the abort), info carries corr1_mm, corr2_mm, grasp_h_mm, grasp_lowered_mm, final_residual_mm,
    wrist_check, in_jaws, jaw_ring, jaw_retry, jaw_frames and reason ("not_in_jaws" when the nut was not carried) for the record and MLflow."""
    tilt = float(getattr(a, "grasp_tilt_deg", GRASP_TILT_DEG) or 0.0); jaw = getattr(a, "jaw_check", True); fast = getattr(a, "fast_close", False); window = tuple(getattr(a, "jaw_window", None) or JAW_WINDOW)
    spec = parse_grasp_roll(getattr(a, "grasp_roll", "off")); roll, rinfo = resolve_grasp_roll(spec, cc, table, nut_in, tilt)   # why: once per pick, from the scene map; every nut-side stage (and the retry) turns the wrist the same way
    if spec != "off": say({"run": run, "stage": "grasp_roll", "roll_deg": roll, **rinfo})
    def stage(xy, names, grasp_h_in=None):
        p = build_plan(list(xy), list(place_in), biases, names=names, jaw_open=a.jaw_open, grasp_h_in=grasp_h_in, fast_close=fast, tilt_deg=tilt, cc=cc, table=table, roll_deg=roll); return p, run_plan(p, a.max_deg_s)
    # (a) hover over the commanded point; the readback says where the tip really went
    _, r1 = stage(nut_in, STAGES[0]); j1 = reached_joints(r1, "hover_nut"); x, y, h = fk_mat(cc, table, j1)
    aim_mm = float(getattr(a, "grasp_aim_mm", GRASP_AIM_MM) or 0.0); u = jaw_dir_mat(cc, table, j1) if aim_mm else None
    aim = aim_point(nut_in, u, aim_mm)   # the fixed jaw's tip goes here; every correction below steers the readback onto it
    c1 = (aim[0] - x, aim[1] - y); info = {"corr1_mm": _mm(c1), "wrist_check": None, "in_jaws": None, "jaw_ring": None, "jaw_retry": False, "jaw_frames": [], "reason": None,
                                          "aim_mm": aim_mm if u else 0.0, "jaw_dir": [round(u[0], 3), round(u[1], 3)] if u else None, "aim_in": [round(v, 3) for v in aim],
                                          "grasp_roll_deg": roll, "grasp_roll_info": rinfo or None}
    say({"run": run, "stage": "hover", "cmd_in": [round(v, 3) for v in nut_in], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "corr1_mm": info["corr1_mm"], "aim_in": info["aim_in"], "aim_mm": info["aim_mm"], "jaw_dir": info["jaw_dir"]})
    if a.wrist_check:   # a look, not a correction: does the camera see the nut where the command says it is?
        info["wrist_check"] = wrist_mod().wrist_check(cc, table, wpose, nut_in, j1, fetch_wrist() or b"", check_path) if wpose else None
        say({"run": run, "stage": "wrist_check", **(info["wrist_check"] or {"ring": None, "annotated": check_path if wpose else None})})
    xy1 = [nut_in[0] + c1[0], nut_in[1] + c1[1]]   # the hover was commanded at the nut; c1 moves it onto the aim point

    def grasp_pose(attempt):   # (b) corrected hover (jaws open), then down to grasp height; read back again, and measure the tip height
        _, r2 = stage(xy1, STAGES[1]); j2 = reached_joints(r2, "grasp"); x, y, h = fk_mat(cc, table, j2)
        c2 = (aim[0] - x, aim[1] - y); xy2 = [xy1[0] + c2[0], xy1[1] + c2[1]]
        lower = min(h - 1.0, 6.0) if h > 2.5 else (-min(1.0 - h, 5.0) if h < 1.0 else 0.0)   # why: the jaws must reach the mat to get under the nut; the cap keeps the IK target within 3 mm below the mat plane the readback contradicts
        # ... and 11 Nov 2026 the old band (accept 1-6 mm) let final grasps sit 3.4-7 mm up over a ~3 mm nut (misses at 5.3, 5.6, 6.7 mm, the jaws closed above it): above 2.5 mm, lower to land at 1 mm
        # ... and a tip that reads below 1 mm is pressing on the mat (10 Nov: -3.4 to -0.8 mm at the longer reaches), which stalls it short and
        # drags it when it moves on: raise the grasp so the readback lands at 1 mm (at most 5 mm). Negative lowered_mm = raised.
        info.update({"corr2_mm": _mm(c2), "grasp_h_mm": round(h, 1), "grasp_lowered_mm": round(lower, 1)})
        say({"run": run, "stage": "grasp_pose", "attempt": attempt, "cmd_in": [round(v, 3) for v in xy1], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "corr2_mm": info["corr2_mm"], "lowered_mm": round(lower, 1)})
        return xy2, lower

    def readback(rows):   # the tip at the grasp row: (x_in, y_in, h_mm, residual against the aim in inches)
        x, y, h = fk_mat(cc, table, reached_joints(rows, "grasp")); return x, y, h, (aim[0] - x, aim[1] - y)

    def grasp(xy2, lower, attempt):   # (c) the pick at the twice-corrected point; returns (plan, rows, xy, lower) of the final pose (the corrected one when the final-pose correction fired)
        hi = lambda lw: (GRASP_H_IN - lw / 25.4) if lw else None
        if not _session_on():   # why: without the persistent session each stage costs a connect (5.8 s), so the split (and its correction) is off: the single lift stage of before (12 Nov 2026)
            plan, rows = stage(xy2, LIFT_STAGE if jaw else STAGES[2], grasp_h_in=hi(lower)); x, y, h, res = readback(rows)
            info.update({"final_residual_mm": round(math.hypot(*res) * 25.4, 2), "grasp_cmd_in": [round(v, 3) for v in xy2], "final_corr_mm": None, "final_h_before": None, "final_h_after": None})
            say({"run": run, "stage": "final", "attempt": attempt, "cmd_in": info["grasp_cmd_in"], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "residual_mm": _mm(res), "sag_deg": {r["wp"]: r["max_err_deg"] for r in rows}})
            return plan, rows, xy2, lower
        plan, rows = stage(xy2, PRE_STAGE, grasp_h_in=hi(lower)); x, y, h, res = readback(rows)
        say({"run": run, "stage": "final", "attempt": attempt, "cmd_in": [round(v, 3) for v in xy2], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "residual_mm": _mm(res), "sag_deg": {r["wp"]: r["max_err_deg"] for r in rows}})
        info.update({"final_residual_mm": round(math.hypot(*res) * 25.4, 2), "grasp_cmd_in": [round(v, 3) for v in xy2], "final_corr_mm": 0.0, "final_h_before": round(h, 1), "final_h_after": round(h, 1)})
        fix = final_fix(h, res, lower)
        if fix:   # ONE corrective move: the grasp waypoint alone, at the corrected xy and height, jaws open (no sweep: it starts at the grasp and moves a few mm)
            (cx, cy), lower2 = fix; xy3 = [xy2[0] + cx, xy2[1] + cy]; p2, r2 = stage(xy3, CORR_STAGE, grasp_h_in=hi(lower2)); x2, y2, h2, res2 = readback(r2)
            info.update({"final_residual_mm": round(math.hypot(*res2) * 25.4, 2), "grasp_cmd_in": [round(v, 3) for v in xy3], "final_corr_mm": round(math.hypot(cx, cy) * 25.4, 2), "final_h_after": round(h2, 1), "grasp_lowered_mm": round(lower2, 1)})
            say({"run": run, "stage": "final_corr", "attempt": attempt, "cmd_in": info["grasp_cmd_in"], "reached_in": [round(x2, 3), round(y2, 3)], "corr_mm": _mm((cx, cy)), "lowered_mm": round(lower2, 1), "h_before": round(h, 1), "h_after": round(h2, 1), "residual_mm": _mm(res2)})
            plan = {"waypoints": [w for w in plan["waypoints"] if w["name"] != "grasp"] + p2["waypoints"]}; rows = [r for r in rows if r["wp"] != "grasp"] + r2; xy2, lower = xy3, lower2
        p3, r3 = stage(xy2, CLOSE_STAGE if jaw else CLOSE_STAGE + CARRY_STAGE, grasp_h_in=hi(lower))   # (close1, close2, lift [, the carry]) at the final pose
        return {"waypoints": plan["waypoints"] + p3["waypoints"]}, rows + r3, xy2, lower

    xy2, lower = grasp_pose(1)
    plan, rows, xy2, lower = grasp(xy2, lower, 1)
    if not jaw: return plan, rows, info   # --no-jaw-check: grasp .. rest as before (one stage with the session off; with it, split at the grasp for the correction)
    def look(attempt, path, rows):   # (d) the grip check: the gripper readback after the lift OR the wrist camera (is the nut between the jaw tips?)
        jc = jaw_check(None, window, path, rows=rows)
        if jc["frame"]: info["jaw_frames"].append(jc["frame"])
        say({"run": run, "stage": "jaw_check", "attempt": attempt, **jc}); return jc
    jc = look(1, jaw_path, rows)
    if jc["in_jaws"] is False:   # one retry: the corrected hover opens the jaws over the spot, then the readback-corrected grasp again
        info["jaw_retry"] = True; xy2, lower = grasp_pose(2); plan, rows, xy2, lower = grasp(xy2, lower, 2)
        jc = look(2, jaw_path.replace(".jpg", "_retry.jpg") if jaw_path else None, rows)
    info.update({"in_jaws": jc["in_jaws"], "jaw_ring": jc["ring"]["uv"] if jc["ring"] else None, "gripper": jc["gripper"], "camera": jc["camera"]})
    if jc["in_jaws"] is False: info["reason"] = "not_in_jaws"; p2, r2 = stage(xy2, ABORT_STAGE)   # open over the nut's spot, then rest: no carry
    else: p2, r2 = stage(xy2, CARRY_STAGE)   # (e) True, or None (no frame): carry
    return {"waypoints": plan["waypoints"] + p2["waypoints"]}, rows + r2, info


# ---------------------------------------------------------------- one recorded pick (the loop and --reset share it)
def pick_once(cc, table, state, a, run, nut_in, tgt, place_in, off, say, session, track, mode, wpose=None, extra=None):
    """One pick, recorded: the staged pick (or the open-loop plan), sag learning, the landing measurement, the history record (appended
    to state and saved), MLflow. Returns (record, landed, picked, stop) with stop a dict to print and stop on, or None."""
    key = f"{tgt[0]:.2f},{tgt[1]:.2f}"; control = "closed" if a.closed_loop else "open"
    with track.run(f"{'reset' if mode == 'reset' else 'pick'}-{session}-r{run}", tags={"host": "mac", "run": run, "nut_in": f"{nut_in[0]:.2f},{nut_in[1]:.2f}", "target_in": key, "session": session, "mode": mode}) as tr:   # why: one MLflow run per pick; a no-op when there is no server
        tr.params({"target_in": f"{a.target[0]:.2f},{a.target[1]:.2f}", "alt_in": f"{a.alt[0]:.2f},{a.alt[1]:.2f}", "place_in": f"{place_in[0]:.3f},{place_in[1]:.3f}", "gain": a.gain, "outcome_gain": a.outcome_gain,
                   "max_deg_s": a.max_deg_s, "runs": a.runs, "state": os.path.relpath(a.state, ROOT) if a.state.startswith(ROOT) else a.state, "dash": DASH, "mode": mode, "jaw_open": a.jaw_open, "wrist_check": int(a.wrist_check),
                   "jaw_check": int(a.jaw_check), "fast_close": int(a.fast_close), "jaw_window": ",".join(str(int(v)) for v in a.jaw_window),
                   "grasp_aim_mm": float(getattr(a, "grasp_aim_mm", GRASP_AIM_MM)) if control == "closed" else None, "grasp_tilt_deg": float(getattr(a, "grasp_tilt_deg", GRASP_TILT_DEG)), "grasp_roll": str(parse_grasp_roll(getattr(a, "grasp_roll", "off")))})
        t0 = time.time(); check_path = os.path.join(ROOT, "outputs", "run", f"wrist_check_{session}_r{run}.jpg"); jaw_path = os.path.join(ROOT, "outputs", "run", f"in_jaws_{session}_r{run}.jpg")
        if a.closed_loop: plan, rows, info = closed_loop_pick(cc, table, nut_in, place_in, state["biases"], a, run, say, wpose, check_path, jaw_path); tr.json(plan, "plan.json")
        else:
            tilt = float(getattr(a, "grasp_tilt_deg", GRASP_TILT_DEG) or 0.0); roll, rinfo = resolve_grasp_roll(getattr(a, "grasp_roll", "off"), cc, table, nut_in, tilt)
            plan = build_plan(nut_in, place_in, state["biases"], jaw_open=a.jaw_open, fast_close=a.fast_close, tilt_deg=tilt, cc=cc, table=table, roll_deg=roll); tr.json(plan, "plan.json"); rows = run_plan(plan, a.max_deg_s); info = {"grasp_roll_deg": roll, "grasp_roll_info": rinfo or None}
        dt = time.time() - t0
        # learn the servo sag per waypoint: bias += gain * (target - reached), capped; closed loop: on the FINAL stages only (their waypoints are the ones the corrected command drove)
        desired = {w["name"]: w["desired"] for w in plan["waypoints"]}
        for r in rows:
            if r["wp"] in GRIPPER_ONLY or r["wp"] in NO_LEARN: continue
            b = state["biases"].setdefault(r["wp"], {})
            for k in ARM: b[k] = max(-6.0, min(6.0, b.get(k, 0.0) + a.gain * (desired[r["wp"]][k] - r["reached"][k])))   # drive DESIRED - reached to zero, not command - reached
            r["max_err_deg"] = round(max(abs(desired[r["wp"]][k] - r["reached"][k]) for k in ARM), 2)
        time.sleep(1.0)
        not_in_jaws = info.get("reason") == "not_in_jaws"
        landing_jpg = fetch_top(); landed = find_nut_near(cc, table, nut_in if not_in_jaws else place_in, radius_px=60, jpeg=landing_jpg)   # why: an aborted pick left the nut where it was; look there, not at the target
        landing_source = getattr(landed, "source", None); tr.params({"landing_source": landing_source, "nut_source": (extra or {}).get("nut_source")})   # why: 11 Nov 2026 which search saw the nut (scene map or dark blob) is the first thing to check when a cycle goes wrong
        before_mm = math.hypot(nut_in[0] - tgt[0], nut_in[1] - tgt[1]) * 25.4
        err_mm = math.hypot(landed[0][0] - tgt[0], landed[0][1] - tgt[1]) * 25.4 if landed else None
        picked = (not not_in_jaws) and landed is not None and (err_mm < before_mm - 5.0 or before_mm < 40.0)   # it got closer, or it was already there and stayed within the search radius
        if landed and picked:
            off[0] = max(-0.6, min(0.6, off[0] + a.outcome_gain * (tgt[0] - landed[0][0]))); off[1] = max(-0.6, min(0.6, off[1] + a.outcome_gain * (tgt[1] - landed[0][1])))
        wc = info.get("wrist_check")
        rec = {"run": run, "target_in": tgt, "nut_in": [round(v, 2) for v in nut_in], "picked": picked, "landed_in": [round(v, 2) for v in landed[0]] if landed else None, "err_mm": round(err_mm, 1) if err_mm is not None else None,
               "seconds": round(dt, 1), "sag_deg": {r["wp"]: r["max_err_deg"] for r in rows}, "place_offset_in": [round(v, 3) for v in off], "mode": mode, "control": control, "jaw_open": a.jaw_open,
               "corr1_mm": info.get("corr1_mm"), "corr2_mm": info.get("corr2_mm"), "grasp_h_mm": info.get("grasp_h_mm"), "grasp_lowered_mm": info.get("grasp_lowered_mm"), "final_residual_mm": info.get("final_residual_mm"),
               "final_corr_mm": info.get("final_corr_mm"), "final_h_before": info.get("final_h_before"), "final_h_after": info.get("final_h_after"), "gripper": info.get("gripper"), "camera": info.get("camera"),
               "aim_mm": info.get("aim_mm"), "jaw_dir": info.get("jaw_dir"), "aim_in": info.get("aim_in"),
               "wrist_offset_mm": wc["offset_mm"] if wc else None, "in_jaws": info.get("in_jaws"), "jaw_ring": info.get("jaw_ring"), "jaw_retry": bool(info.get("jaw_retry")), "reason": info.get("reason"), "landing_source": landing_source,
               "fast_close": bool(a.fast_close), **(extra or {}), "t": time.strftime("%H:%M:%S")}
        state["history"].append(rec); json.dump(state, open(a.state, "w"), indent=1); print(json.dumps(rec), flush=True)
        c1, c2 = info.get("corr1_mm"), info.get("corr2_mm"); ij = info.get("in_jaws")
        tr.metrics({"landing_err_mm": err_mm, "before_mm": before_mm, "seconds": dt, "picked": int(picked), "found_at_landing": int(landed is not None), **{f"sag_{r['wp']}": r["max_err_deg"] for r in rows if "max_err_deg" in r},
                    "corr1_mm": math.hypot(*c1) if c1 else None, "corr1_x_mm": c1[0] if c1 else None, "corr1_y_mm": c1[1] if c1 else None, "corr2_mm": math.hypot(*c2) if c2 else None, "corr2_x_mm": c2[0] if c2 else None, "corr2_y_mm": c2[1] if c2 else None,
                    "grasp_h_mm": info.get("grasp_h_mm"), "grasp_lowered_mm": info.get("grasp_lowered_mm"), "final_residual_mm": info.get("final_residual_mm"), "final_corr_mm": info.get("final_corr_mm"), "final_h_before": info.get("final_h_before"), "final_h_after": info.get("final_h_after"), "gripper_after_lift": info.get("gripper"), "wrist_offset_mm": wc["offset_norm_mm"] if wc else None, "jaw_open": a.jaw_open, "aim_mm": info.get("aim_mm"), "grasp_roll_deg": info.get("grasp_roll_deg"),
                    "in_jaws": None if ij is None else int(ij), "jaw_retry": int(bool(info.get("jaw_retry"))), "not_in_jaws": int(not_in_jaws),
                    **(servo_tele().metrics(tele) if (tele := tele_of(rows)) else {})})   # why: sag, heat, current and voltage per pick, so sessions compare (docs/telemetry.md)
        if tele: tr.json(tele, "servo_telemetry.json")
        tr.file("landing.jpg", landing_jpg, "frames"); tr.json(rows, "waypoints.json"); tr.artifact(a.state, "state")
        if a.wrist_check: tr.artifact(check_path, "frames")
        for p in info.get("jaw_frames") or []: tr.artifact(p, "frames")
        stop = None
        if not_in_jaws: stop = {"run": run, "stop": "nut not in the jaws after two grasps; opened over the spot and went to rest", "reason": "not_in_jaws"}
        elif not landed: stop = {"run": run, "stop": "nut not found near the target after placing; it may have been dropped elsewhere"}
        elif not picked: stop = {"run": run, "stop": "nut did not get closer to the target; grasp probably missed"}
        return rec, landed, picked, stop


# ---------------------------------------------------------------- reset (2B)
def sample_reset_target(nut_in, safe_box, avoid, markers, rng, min_avoid_mm=RESET_MM["avoid"], min_marker_mm=RESET_MM["marker"], min_move_mm=RESET_MM["move"], tries=200):
    """A random point inside `safe_box` (x0, y0, x1, y1 mat inches) at least min_avoid_mm from every point of `avoid`, min_marker_mm from every
    marker centre and min_move_mm from the nut's current spot; rejection sampling with `rng.uniform`, SystemExit after `tries` misses."""
    x0, y0, x1, y1 = safe_box; lo, hi = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
    for _ in range(int(tries)):
        p = (round(rng.uniform(lo[0], hi[0]), 3), round(rng.uniform(lo[1], hi[1]), 3))   # why: test the point we return; rounding after the check could slip a spot under a limit
        if math.hypot(p[0] - nut_in[0], p[1] - nut_in[1]) * 25.4 < min_move_mm: continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) * 25.4 < min_avoid_mm for q in avoid): continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) * 25.4 < min_marker_mm for q in markers): continue
        return [p[0], p[1]]
    raise SystemExit(f"reset: no clear spot in the safe box {list(safe_box)} in {tries} tries (nut at ({nut_in[0]:.2f}, {nut_in[1]:.2f}) in, {len(avoid)} objects to keep {min_avoid_mm:.0f} mm from, "
                     f"{len(markers)} markers at {min_marker_mm:.0f} mm, move >= {min_move_mm:.0f} mm): clear the mat or widen --safe-box")


def avoid_from_scene(objects, nut_in, exclude_mm=15.0):
    """Every scene-map object centre (mat inches) except the nut itself (anything within exclude_mm of it)."""
    out = []
    for o in objects or []:
        try: p = [float(o["x_in"]), float(o["y_in"])]
        except (KeyError, TypeError, ValueError): continue
        if math.hypot(p[0] - nut_in[0], p[1] - nut_in[1]) * 25.4 > exclude_mm: out.append(p)
    return out


def default_args():
    """The CLI defaults as a namespace (for callers of reset_nut / closed_loop_pick that have no argparse of their own)."""
    a = parser().parse_args([]); a.max_deg_s = SPEED["closed"]; return a


def reset_nut(state, cc, table, safe_box, avoid, rng, a=None, say=None, markers_in=None, session=None, track=None, run=0, scene_objects=None):
    """--reset: pick the nut from wherever it is (near its last spot, else the scene map, else the whole mat) and place it at a random clear
    point inside `safe_box` (sample_reset_target; `avoid` = other objects' centres in mat inches, the nut itself is dropped from it) with the
    same closed-loop pick. The history record gets mode="reset", the target and the box. Returns the record; SystemExit when no nut or no spot."""
    a = a or default_args(); say = say or (lambda d: print(json.dumps(d), flush=True)); track = track or labtrack(); session = session or time.strftime("%Y%m%d-%H%M%S")
    markers = marker_centres_in(table, load_layout()) if markers_in is None else markers_in
    near = state["history"][-1]["landed_in"] if state.get("history") and state["history"][-1].get("landed_in") else list(a.target)
    found = find_nut_near(cc, table, near); source = found.source if found else None
    if not found:
        anywhere = find_nut_anywhere(cc, table, scene_objects, near_in=near, markers_in=markers)
        if not anywhere: raise SystemExit(f"reset: nut not found anywhere on the mat (last seen near {[round(v, 2) for v in near]})")
        found, source = (anywhere[0], None, None), anywhere[1]
    nut_in = found[0]; avoid = avoid_from_scene([{"x_in": p[0], "y_in": p[1]} for p in (avoid or [])], nut_in)
    tgt = sample_reset_target(nut_in, safe_box, avoid, markers, rng)
    say({"reset": True, "nut_in": [round(v, 3) for v in nut_in], "nut_source": source, "target_in": tgt, "safe_box": [float(v) for v in safe_box], "avoid": len(avoid), "markers": len(markers)})
    rec, _, _, stop = pick_once(cc, table, state, a, run, nut_in, tgt, list(tgt), [0.0, 0.0], say, session, track, "reset", extra={"safe_box": [float(v) for v in safe_box], "nut_source": source})
    if stop: print(json.dumps(stop), flush=True)
    return rec


# ---------------------------------------------------------------- CLI
def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=4); ap.add_argument("--target", type=float, nargs=2, default=[9.0, 6.0])
    ap.add_argument("--max-deg-s", type=float, default=None, help="pick_seq speed cap (default 50 closed loop, 80 open loop)")
    ap.add_argument("--state", default=os.path.join(ROOT, "jetson", "calib", "pick_state.json")); ap.add_argument("--gain", type=float, default=0.7); ap.add_argument("--outcome-gain", type=float, default=0.5)
    ap.add_argument("--alt", type=float, nargs=2, default=[6.0, 5.0], help="second target; runs alternate alt, target, alt, target ... so every pick and place is unambiguous; the LAST run always goes to --target")
    ap.add_argument("--closed-loop", dest="closed_loop", action="store_true", default=True, help="hover, read back, correct; grasp pose, read back, correct; pick (default)")
    ap.add_argument("--open-loop", dest="closed_loop", action="store_false", help="one pick_seq run of the whole plan, no readback correction")
    ap.add_argument("--grasp-aim-mm", type=float, default=GRASP_AIM_MM, help=f"closed loop: put the nut this far inside the open jaw gap, so the fixed jaw lands beside it, not on it (default {GRASP_AIM_MM:g}; 0 aims the fixed jaw's tip at the nut as before 10 Nov 2026)")
    ap.add_argument("--grasp-tilt-deg", type=float, default=GRASP_TILT_DEG, help=f"wrist_flex tilt on pregrasp, grasp and close (default {GRASP_TILT_DEG:g}; negative lifts the front finger off the mat; the tip is held on the commanded point; 0 = the vertical gripper of before 11 Nov 2026)")
    ap.add_argument("--grasp-roll", type=parse_grasp_roll, default="off", metavar="auto|off|DEG", help="turn the wrist so the jaws close across the object's long axis: auto = the scene-map object at the nut position (round ones: no roll), DEG = that wrist_roll offset from the arm fit's roll, off = today (default); nut-side waypoints only, see docs/pick-loop.md")
    ap.add_argument("--jaw-open", type=int, default=JAW_OPEN, help=f"gripper opening for hover/grasp/open/lift2 (default {JAW_OPEN}; 45 flicks the nut away, see docs/pick-loop.md)")
    ap.add_argument("--wrist-check", action="store_true", help="closed loop: at hover, look for the nut in the wrist frame near its predicted pixel and report the offset (check only; annotated JPEG in outputs/run/)")
    ap.add_argument("--no-jaw-check", dest="jaw_check", action="store_false", default=True, help="skip the wrist-camera nut-in-jaws check after the lift (and its one retry); the final stage runs in one session as before")
    ap.add_argument("--jaw-window", type=int, nargs=4, default=list(JAW_WINDOW), metavar=("X0", "Y0", "X1", "Y1"), help=f"wrist-frame rectangle between the jaw tips where the held nut's ring must appear (default {' '.join(map(str, JAW_WINDOW))})")
    ap.add_argument("--fast-close", action="store_true", help=f"single-step close (jaws straight to 0) instead of close1 to {CLOSE_MID} then close2 to 0")
    ap.add_argument("--lost-after", type=int, default=3, help="stop with 'lost' after this many consecutive runs in which the nut was found nowhere (near, scene map, whole mat)")
    ap.add_argument("--reset", action="store_true", help="instead of the runs: pick the nut from wherever it is and place it at a random clear spot inside --safe-box")
    ap.add_argument("--safe-box", type=float, nargs=4, default=list(SAFE_BOX), metavar=("X0", "Y0", "X1", "Y1"), help=f"--reset: the box the nut is dropped in, mat inches (default {' '.join(map(str, SAFE_BOX))})")
    ap.add_argument("--seed", type=int, default=None, help="--reset: seed for the target sampling (default: random)")
    ap.add_argument("--session", dest="session", action="store_true", default=True, help="one persistent pick_seq --serve over ssh for all stages (saves ~5 s of start-up per stage; needs jetson/pick_seq.py with --serve; falls back to scp+ssh per stage) (default)")
    ap.add_argument("--no-session", dest="session", action="store_false", help="one scp+ssh+pick_seq start per stage, as before 12 Nov 2026")
    return ap


def main():
    a = parser().parse_args(); open_session(a.session)
    try: _main(a)
    finally: close_session()   # why: the ssh must not outlive the pick, on any exit


def _main(a):
    cc = cam_calib(); table = json.load(open(os.path.join(ROOT, "jetson", "calib", "top_table.json"))); track = labtrack()
    mode = "closed" if a.closed_loop else "open"; a.max_deg_s = a.max_deg_s if a.max_deg_s is not None else SPEED[mode]
    if a.closed_loop and not (table.get("arm_fit") and table.get("arm_base_mm")): raise SystemExit("closed loop needs the arm fit in jetson/calib/top_table.json (tools/jog_capture.py --fit); or run --open-loop")
    wpath = os.path.join(ROOT, "jetson", "calib", "wrist_campose.json"); wpose = json.load(open(wpath)) if a.wrist_check and os.path.exists(wpath) else None
    if a.wrist_check and wpose is None: print(json.dumps({"note": "no wrist_campose.json; --wrist-check skipped"}), flush=True)
    state = json.load(open(a.state)) if os.path.exists(a.state) else {"biases": {}, "place_offsets_in": {}, "history": []}
    state.setdefault("place_offsets_in", {}); state.setdefault("history", []); near = state["history"][-1]["landed_in"] if state["history"] and state["history"][-1].get("landed_in") else list(a.target)
    session = time.strftime("%Y%m%d-%H%M%S"); say = lambda d: print(json.dumps(d), flush=True); markers = marker_centres_in(table, load_layout())
    if a.reset:
        objs = fetch_scene_objects()
        reset_nut(state, cc, table, a.safe_box, [[o["x_in"], o["y_in"]] for o in objs if "x_in" in o and "y_in" in o], random.Random(a.seed), a=a, say=say, markers_in=markers, session=session, track=track, run=1, scene_objects=objs); return
    run, misses = 0, 0
    while run < a.runs:
        nxt = run + 1; tgt = list(a.target) if (nxt == a.runs or nxt % 2 == 0) else list(a.alt); key = f"{tgt[0]:.2f},{tgt[1]:.2f}"
        found = find_nut_hints(cc, table, ((near, 70), (list(a.target), 140), (list(a.alt), 140))); source = found.source if found else None
        if not found:   # the whole-mat finder: the scene map, then a dark-blob search over the whole mat with the markers masked
            anywhere = find_nut_anywhere(cc, table, near_in=near, markers_in=markers)
            if anywhere: found, source = (anywhere[0], None, None), anywhere[1]; say({"run": nxt, "note": "nut found by the whole-mat finder", "source": source, "nut_in": [round(v, 3) for v in anywhere[0]]})
        if not found:
            misses += 1; say({"run": nxt, "miss": misses, "near": [round(v, 3) for v in near], "lost_after": a.lost_after})
            if misses >= a.lost_after: print(json.dumps({"run": nxt, "stop": "lost", "misses": misses, "near": [round(v, 3) for v in near]}), flush=True); break
            time.sleep(1.0); continue   # why: a miss does not spend a run; the scene map updates at ~2 Hz, so look again
        run, misses = nxt, 0; off = state["place_offsets_in"].setdefault(key, [0.0, 0.0]); nut_in = found[0]; place_in = [tgt[0] + off[0], tgt[1] + off[1]]
        rec, landed, picked, stop = pick_once(cc, table, state, a, run, nut_in, tgt, place_in, off, say, session, track, mode, wpose, extra={"nut_source": source})
        if stop: print(json.dumps(stop), flush=True); break
        near = landed[0]


if __name__ == "__main__": main()
