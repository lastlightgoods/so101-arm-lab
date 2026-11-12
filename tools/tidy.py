#!/usr/bin/env python3
"""tidy.py - put every toy on the mat into the 2 in tall blue tape ring, one recorded pick per toy, and (reset) get the mat ready for the next round (docs/tidy.md).

  tools/tidy.py plan                         # no motion: the toys, the ring, reachable / blocked for each, the order
  tools/tidy.py run [--record] [--max N]     # tidy the toys one by one, re-reading the scene map between picks
  tools/tidy.py reset [--record-spread]      # lift the ring off the pile, set it down elsewhere, spread the toys on clear spots
  tools/tidy.py loop --rounds K [--record]   # tidy all -> reset -> tidy all ... with the guardrails

The pick is tools/pick_loop.closed_loop_pick (readback-corrected hover, grasp, lift) with: the ring found by its blue colour; every planned waypoint and the joint paths between them
checked against the ring as a solid cylinder before anything moves; a release over the ring at 110 mm with a tipped gripper; an adaptive route; the "fluid" trims; a grip check.
Nothing here edits pick_loop: the module's tables and functions are patched for the length of one pick (Rig.patched) and put back.
"""
import argparse, contextlib, importlib.util, json, math, os, random, signal, sys, time, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = "so101-tidy"
TASK_RING = "Pick up the toy and put it in the tape ring"   # a new task string, its own dataset (lastlight/so101-toys-to-ring)
TASK_SPREAD = "Pick up the toy and put it down on the mat"
RING_REPO, SPREAD_REPO = "lastlight/so101-toys-to-ring", "lastlight/so101-toys-spread"
IN = 25.4
RING_R_MM, RING_H_MM = 52.0, 51.0   # outer radius; 2 in tape (TC, 11 Nov 2026)
RING_DONE_MM = 48.0                 # a toy this close to the ring's centre is inside it (already done)
RING_WALL_IN_MM, RING_WALL_OUT_MM = 20.0, 10.0   # a blob centred this far inside / outside the ring's outer radius is the ring's own wall, not a toy
PART_R_MM, TIP_R_MM, MARGIN_MM = 20.0, 8.0, 5.0   # the body / the jaw tip vs the rim (+5 mm over the 51 mm wall)
DROP_IN_MAX = 3.3                   # highest vertical-gripper height anywhere (inches)
GRIP_EMPTY_MAX = 2.0                # gripper readback (0-100) after the lift at or below this = nothing held (11 Nov 2026 live)
REST_TIP_IN = (9.02, 3.21)          # straight above the rest tip is where the arm goes up before crossing to the ring
HIGH_MM = 110.0
MAX_FAILS = 3
CLEAN_ENDS = ('all_tidy', 'max', 'max_tries', 'max_minutes')   # exit code 0: finished, or a cap the owner set
BLUE_ROWS = 5 / 6                   # the bottom sixth of the frame holds the camera's own blue tape
# -- the ring lift (reset)
RING_GRIP_H_MM, RING_PRE_H_MM, RING_HOVER_H_MM, RING_JAW, RING_AIM_MM, WALL_MID_MM = 15.0, 63.0, 76.0, 50, 12.0, 41.0   # 11 Nov 21:0x live; the grip is 10 mm lower than that first try
RING_LIFT_CAP_MM, MIN_BOTTOM_MM, HARD_MIN_BOTTOM_MM = 110.0, 40.0, 25.0
RING_BOX_IN = (5.5, 3.3, 12.0, 8.7)     # where the ring's centre may be set down. Wide on purpose: the ring has to hop >= 3.6 in to leave 1.5 in of floor between its wall and the pile, and the reach check (RingMover.plan) trims it; 2+ in from the mat edges
SPREAD_BOX_IN = (5.5, 3.0, 12.5, 9.0)   # where spread toys go: comfortable reach, off the far edges
PILE_R_MM = RING_R_MM + 15.0   # toys this close to where the ring stood are its pile
TOY_AREA_MM2, TOY_LONG_MM = 90.0, 22.0  # a nut reads ~12 x 10 mm, area ~88 (docs/mat-scene.md)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def load_pl(): return load(os.path.join(ROOT, "tools", "pick_loop.py"), "pick_loop_for_tidy")
def load_gr(): return load(os.path.join(ROOT, "tools", "guardrails.py"), "guardrails_for_tidy")
def load_al(): return load(os.path.join(ROOT, "tools", "autoloop.py"), "autoloop_for_tidy")


ko = load(os.path.join(ROOT, "tools", "keepout.py"), "keepout_for_tidy")   # keep-outs: the toy frogs and anything else that is not a toy (TC 2026-11-11: "be careful not to hurt the frogs")


def dist_mm(a, b): return math.hypot(a[0] - b[0], a[1] - b[1]) * IN


# ---------------------------------------------------------------- the ring in the top frame, the toy kinds (shared with the dashboard: tools/scene_labels.py)
sl = load(os.path.join(ROOT, "tools", "scene_labels.py"), "scene_labels_for_tidy")
fit_circle, blue_pixels, ring_band, fit_ring, toy_kind, on_ring_wall = sl.fit_circle, sl.blue_pixels, sl.ring_band, sl.fit_ring, sl.toy_kind, sl.on_ring_wall


TOY_MIN_SIDE_MM, MAT_EDGE_IN, MAT_IN = 6.0, 0.8, (18.0, 12.0)   # why: 12 Nov 2026 a 1-2 mm mat shift against the stored background gave ghost blobs along the mat's edges, mostly 4 mm-thin strips (18 x 4, 66 x 4 mm) and one nut-sized one at the top edge by the arm's base; real toys are >= 6 mm across (a standing spacer)


def is_toy(o, area_max=2000.0, long_max=60.0, ring=None):
    """A scene blob that is a toy: not the whole ring (104 mm across), not a thin strip or a blob at the mat's border (background ghosts after the mat
    shifts), and, with `ring`, not a piece of the ring's wall (on_ring_wall)."""
    if not ("x_in" in o and "y_in" in o and float(o.get("area_mm2") or 0) < area_max and max(float(o.get("w_mm") or 0), float(o.get("h_mm") or 0)) <= long_max): return False
    if min(float(o.get("w_mm") or 0), float(o.get("h_mm") or 0)) < TOY_MIN_SIDE_MM: return False
    x, y = float(o["x_in"]), float(o["y_in"])
    if not (MAT_EDGE_IN <= x <= MAT_IN[0] - MAT_EDGE_IN and MAT_EDGE_IN <= y <= MAT_IN[1] - MAT_EDGE_IN): return False
    return ring is None or not on_ring_wall(o, ring)


def inside_ring(o, ring, r_mm=RING_DONE_MM): return dist_mm((o["x_in"], o["y_in"]), ring) < r_mm


def order_toys(toys, verdicts, base_in, ring, how="clean"):
    """Toys with an ok verdict in pick order. near: the nearest to the ring first (shortest carry). clean: the toys whose straight route from rest is clean first, then nearest to the arm's base."""
    ok = [t for t, v in zip(toys, verdicts) if v.get("ok")]
    if how == "near": return sorted(ok, key=lambda t: dist_mm((t["x_in"], t["y_in"]), ring))
    vd = {id(t): v for t, v in zip(toys, verdicts)}
    return sorted(ok, key=lambda t: (not vd[id(t)].get("direct"), dist_mm((t["x_in"], t["y_in"]), base_in)))


# ---------------------------------------------------------------- the grip check
CAM_ONLY_MIN_SCORE = 0.45   # the wrist camera passes a wing nut on its own only at this ring score or higher: its false positives on empty jaws scored 0.37, 0.388, 0.395 (11-12 Nov 2026)


def grip_decision(g_lift, cam_in_jaws, kind="nut", cam_score=None):
    """Held after the lift: the gripper readback > 2.0 (the jaws stopped on something). The wrist camera counts ONLY for a wing nut (its thin wing reads 0.5-1.5, like empty jaws); for a nut or a spacer the
    camera alone never passes. True / False / None (unknown: no readback, and no camera answer that counts)."""
    if g_lift is not None and g_lift > GRIP_EMPTY_MAX: return True
    if kind == "wing nut" and cam_in_jaws is True and (cam_score is None or cam_score >= CAM_ONLY_MIN_SCORE): return True   # why: 12 Nov a 0.37 false positive sent an empty carry to the ring
    if g_lift is None and not (kind == "wing nut" and cam_in_jaws is not None): return None
    return False   # why: 11 Nov 21:32 pilot: readback 0.54 (empty) but the camera found a false ring (score 0.388) on the mat pattern between the empty jaws, and a mislabelled episode was kept


def interior_diff(before, after, circ, frac=0.8, thr=30):
    """Pixels inside the ring (a disc of `frac` x the fitted radius around the fitted centre, so the rim and its shadow stay out) whose grey level differs by more than `thr` between two frames.
    before / after: 2-D uint8 arrays; circ: (cx, cy, r) in pixels."""
    h, w = before.shape[:2]; yy, xx = np.ogrid[:h, :w]; m = (xx - circ[0]) ** 2 + (yy - circ[1]) ** 2 <= (frac * circ[2]) ** 2
    return int(np.count_nonzero((np.abs(before.astype(int) - after.astype(int)) > thr) & m))


def placed_verdict(before, after, circ, area_mm2, ppm, min_frac=0.4, min_px=6):
    """Did a toy really arrive in the ring: the interior diff >= min_frac of the toy's scene-map area in pixels (at least min_px). -> {placed, diff_px, need_px}; the numbers go into the verdict."""
    need = max(float(min_px), min_frac * float(area_mm2) * float(ppm) ** 2); d = interior_diff(before, after, circ)
    return {"placed": d >= need, "diff_px": d, "need_px": round(need, 1)}


# ---------------------------------------------------------------- the rig: geometry, obstacle check, route, trims
class Rig:
    """Everything that needs the arm fit and the ring. `pl` is pick_loop, `ik_free(x_in, y_in, h_mm, roll=None)` the tipped-gripper IK (default: cam_calib.ik_tip down=False)."""
    def __init__(self, pl, cc, table, ring, ik_free=None, fluid=True, part_r_mm=PART_R_MM):
        self.pl, self.cc, self.table, self.ring, self.fluid, self.part_r = pl, cc, table, tuple(ring), fluid, part_r_mm
        self.off = table["arm_fit"].get("joint_offsets_deg") or {}; self.tz = float(table["arm_fit"].get("table_z_mm", 0.0) or 0.0)
        self.base = tuple(v / IN for v in cc.arm_to_table(table, 0.0, 0.0)); self._ik_free = ik_free or self.default_ik_free; self.wp0 = list(pl.WP); self.end_direct = None
        self._cache = {}; self.keepouts = []

    def default_ik_free(self, x, y, h_mm, roll=None):
        tg = self.pl.api(f"/api/ik?x_in={x}&y_in={y}&h_in=2.0")["target_mm"]; s = self.cc.ik_tip([tg[0], tg[1], tg[2] + h_mm - 50.8], self.off, roll_deg=0.0, down=False)
        return {**s["joints_deg"], "wrist_roll": self.pl.REST["wrist_roll"] if roll is None else roll}

    def ik_free(self, x, y, h_mm, roll=None): return self._ik_free(x, y, h_mm, roll)

    def set_ring(self, ring): self.ring = tuple(ring); self._cache = {}

    @property
    def high(self):   # over the ring's centre at 110 mm, tipped: the vertical IK only reaches 76-84 mm there. None when even that has no answer.
        if "high" not in self._cache:
            try: self._cache["high"] = self.ik_free(self.ring[0], self.ring[1], HIGH_MM)
            except Exception: self._cache["high"] = None
        return self._cache["high"]

    @property
    def hb(self):   # straight above the rest tip, 71 mm
        if "hb" not in self._cache: self._cache["hb"] = self.pl.api(f"/api/ik?x_in={REST_TIP_IN[0]}&y_in={REST_TIP_IN[1]}&h_in=2.8")["joints_deg"]
        return self._cache["hb"]

    @property
    def hb_high(self):
        if "hbh" not in self._cache: self._cache["hbh"] = self.ik_free(REST_TIP_IN[0], REST_TIP_IN[1], HIGH_MM)
        return self._cache["hbh"]

    # -- reach
    def ik_ok(self, x, y, h): return bool(self.pl.api(f"/api/ik?x_in={x:.3f}&y_in={y:.3f}&h_in={h:.3f}").get("ok"))

    def max_h(self, x, y, top=DROP_IN_MAX):
        h = top
        while h > 1.0 and not self.ik_ok(x, y, h): h = round(h - 0.1, 2)
        return h

    # -- the obstacle check: the ring as a solid cylinder
    def droop_mm(self, x, y):   # the 11 Nov sweep: ~4 mm at 150 mm reach, ~16 at 220, ~24 at 290 (shoulder 1.7 / 4.2 / 4.7 deg)
        r = math.hypot(x - self.base[0], y - self.base[1]) * IN; return max(4.0, 4.0 + (r - 150.0) / 140.0 * 20.0)

    def points(self, j):
        """(x_in, y_in, h_mm) along wrist -> gripper -> gripper_frame, then the tip (last)."""
        F = self.cc.fk_frames({k: float(j[k]) for k in self.pl.ARM}, self.off); chain = [F[n][:3, 3] * 1000.0 for n in ("wrist_link", "gripper_link", "gripper_frame_link")]; out = []
        for a, b in zip(chain, chain[1:]):
            for f in np.linspace(0, 1, 5):
                p = a + f * (b - a); X, Y = self.cc.arm_to_table(self.table, float(p[0]), float(p[1])); out.append((X / IN, Y / IN, float(p[2]) - self.tz))
        out.append(self.pl.fk_mat(self.cc, self.table, j)); return out

    def hits(self, j, ring=True):
        """Body points of pose j that touch the ring (x, y, h, clearance) or a keep-out (x, y, h, clearance, "keep-out"); ring=False: keep-outs only (the ring lift grips the ring)."""
        bad = []; pts = self.points(j); tx, ty, th = pts[-1]
        for x, y, h in pts:
            at_tip = math.dist((x * IN, y * IN, h), (tx * IN, ty * IN, th)) < 15.0; rad = TIP_R_MM if at_tip else self.part_r; droop = self.droop_mm(x, y)
            wall = math.hypot(x - self.ring[0], y - self.ring[1]) * IN - RING_R_MM
            if ring and wall < rad and h - droop < RING_H_MM + MARGIN_MM: bad.append((round(x, 2), round(y, 2), round(h, 1), round(wall, 1)))
            for k in ko.point_hits(x, y, h, rad, self.keepouts, droop, MARGIN_MM): bad.append((round(x, 2), round(y, 2), round(h, 1), round(ko.dist_mm((x, y), (k["x_in"], k["y_in"])) - k["r_mm"], 1), "keep-out"))
        return bad

    def lerp(self, a, b, f): return {k: a[k] + f * (b[k] - a[k]) for k in self.pl.ARM}

    def path_clean(self, a, b, n=24): return not any(self.hits(self.lerp(a, b, f)) for f in np.linspace(0, 1, n))

    def seq_hits(self, seq, n=16):
        bad = []
        for a, b in zip(seq, seq[1:]):
            for f in np.linspace(0, 1, n): bad += self.hits(self.lerp(a, b, f))
        return bad

    # -- route and trims
    def above(self, obj, h_in):
        r = self.pl.api(f"/api/ik?x_in={obj[0]:.3f}&y_in={obj[1]:.3f}&h_in={h_in:.3f}")
        if not r.get("ok"): raise SystemExit(f"ik failed above ({obj[0]:.2f},{obj[1]:.2f}) at {h_in} in")
        return r["joints_deg"]

    def start_route(self, obj, lift):
        """[] when rest -> straight to the toy's stage 1 is clean; else up in front of the base and over at 110 mm (high_back, high_back_up, high, above)."""
        A = self.above(obj, lift)
        if self.path_clean(self.pl.REST, A): return []
        return [("high_back", self.hb), ("high_back_up", self.hb_high), ("high", self.high), ("above", A)]

    def trim(self, plan):
        """The ring carry and the fluid trims on one stage plan (pure on the plan): the carry ends over the ring at HIGH (closed, then open) and leaves by hb_high, or straight home
        when HIGH -> rest is clean; no `place`; no `lift2` that repeats the pose reached; stage 2's hover_nut merged into the descent; short settles on back-offs."""
        wps = plan["waypoints"]; names = [w["name"] for w in wps]
        keep = []
        for w in wps:
            n = w["name"]
            if n == "place" or (n == "lift2" and self.end_direct): continue   # why: place and lift2 repeated the pose already reached (video 11 Nov 20:5x)
            if self.fluid and n == "hover_nut" and len(wps) > 2 and "grasp" in names: continue   # stage 2's hover merges into the descent
            if self.fluid and n == "hover_pre" and "rest" in names: continue   # the abort's back-off hover: nothing to measure on the way home
            keep.append(w)
        for w in keep:
            if self.fluid and w["name"] in ("hover_pre", "pregrasp_up"): w["settle"] = 0.1
            if w["name"] in ("hover_tgt", "place", "open"): w["joints"], w["desired"] = dict(self.high), dict(self.high)
            if w["name"] == "lift2": w["joints"] = w["desired"] = dict(self.hb_high)
        plan["waypoints"] = keep; return plan

    @contextlib.contextmanager
    def patched(self, lift_in, hover_in=1.5, trim=True):
        """pick_loop's WP table (hover and lift heights, carry rows as placeholders) and build_plan (the trims) for one pick, put back after."""
        pl = self.pl; wp, bp = pl.WP, pl.build_plan
        pl.WP = [(w[0], w[1], 1.0 if (trim and w[0] in ("hover_tgt", "place", "open", "lift2")) else (lift_in if w[0] == "lift" else (hover_in if w[0] in ("hover_pre", "hover_nut") else w[2])), *w[3:]) for w in self.wp0]
        if trim: pl.build_plan = lambda *a, **k: self.trim(bp(*a, **k))
        try: yield
        finally: pl.WP, pl.build_plan = wp, bp

    def check(self, obj, dest=None):
        """The verdict for one toy, nominal plan, no motion: {reachable, ok, lift_in, hover_in, direct, end_direct, n_bad, worst, why}. dest None: the ring carry; else a spot (plain place) for the spread."""
        pl = self.pl; v = {"reachable": False, "ok": False, "lift_in": None, "hover_in": None, "direct": None, "end_direct": None, "n_bad": 0, "worst": None, "why": None}
        try:
            mh = self.max_h(*obj)
            if mh < 0.9 or not self.ik_ok(obj[0], obj[1], pl.GRASP_H_IN): v["why"] = "out of reach"; return v
            lift = max(1.0, min(DROP_IN_MAX, mh - 0.1)); hover = max(0.8, min(1.5, mh - 0.1)); v.update({"reachable": True, "lift_in": lift, "hover_in": hover})
            if dest is None:
                if self.high is None: v["why"] = "no tipped release pose over the ring"; return v
                self.end_direct = self.path_clean(self.high, pl.REST); v["end_direct"] = self.end_direct
                route = self.start_route(obj, lift); v["direct"] = not route; seq = [pl.REST] + [j for _, j in route]
                with self.patched(lift, hover):
                    for names in (pl.STAGES[0], pl.STAGES[1], pl.STAGES[2]): seq += [w["joints"] for w in pl.build_plan(list(obj), list(self.ring), {}, names=names, tilt_deg=pl.GRASP_TILT_DEG, cc=self.cc, table=self.table)["waypoints"]]
                    lift_j = pl.build_plan(list(obj), list(self.ring), {}, names=("lift",), tilt_deg=pl.GRASP_TILT_DEG, cc=self.cc, table=self.table)["waypoints"][0]["joints"]
                    seq_abort = [lift_j] + [w["joints"] for w in pl.build_plan(list(obj), list(self.ring), {}, names=pl.ABORT_STAGE, tilt_deg=pl.GRASP_TILT_DEG, cc=self.cc, table=self.table)["waypoints"]]
                bad = self.seq_hits(seq) + self.seq_hits(seq_abort)
            else:
                v["direct"] = self.path_clean(pl.REST, self.above(obj, lift)); seq = [pl.REST]
                with self.patched(lift, hover, trim=False):
                    for names in (pl.STAGES[0], pl.STAGES[1], pl.STAGES[2]): seq += [w["joints"] for w in pl.build_plan(list(obj), list(dest), {}, names=names, tilt_deg=pl.GRASP_TILT_DEG, cc=self.cc, table=self.table)["waypoints"]]
                bad = self.seq_hits(seq)
            v["n_bad"] = len(bad); v["worst"] = min(bad, key=lambda t: t[3]) if bad else None; v["ok"] = not bad
            v["why"] = None if not bad else ("route passes a keep-out (a frog?)" if any(len(t) > 4 for t in bad) else "route passes the ring")
        except SystemExit as e: v["why"] = str(e)[:90]; v["ok"] = False
        return v


# ---------------------------------------------------------------- the pick
class TripError(Exception):
    def __init__(self, trip): super().__init__(trip.reason); self.trip = trip


class Episode:
    """One recorded pick around a DemoRecorder / RemoteRecorder-shaped `rec` (open(meta, t), end(t) -> health, commit(keep, verdict) -> {dataset, episode_index, frames, health}, abort(reason)).
    A recorder that refuses to open never stops the arm: the pick runs unrecorded and the result says why. rec None: no recording."""
    active = {"rec": None}   # the resident recorder holds ONE dataset configuration: switching between the ring and the spread dataset reconfigures it (untested live)

    def __init__(self, rec, repo): self.rec, self.repo, self.open_err, self.health, self.on = rec, repo, None, None, False

    def activate(self):
        if self.rec is not None and Episode.active["rec"] is not self.rec:
            if hasattr(self.rec, "configure"): self.rec.configure()
            Episode.active["rec"] = self.rec

    def open(self, meta):
        if self.rec is None: return False
        try: self.activate(); self.rec.open(meta, None); self.on = True
        except Exception as e: self.open_err = f"{type(e).__name__}: {e}"[:160]; self.on = False
        return self.on

    def end(self):
        if self.on: self.health = self.rec.end(None); self.on = False; return True
        return False

    def abort(self, reason):
        if self.rec is not None and (self.on or self.health is not None):
            try: self.rec.abort(reason)
            except Exception: pass
        self.on = False; self.health = None

    def commit(self, keep, verdict, meta_extra=None):
        if self.rec is None: return None
        if self.open_err: return {"kept": False, "dataset": None, "dataset_repo": None, "episode_index": None, "frames": 0, "discard_reason": f"recorder_open_failed:{self.open_err}"}
        ep = getattr(self.rec, "ep", None)
        if ep is not None and meta_extra: ep.meta.setdefault("teacher", {}).update(meta_extra)
        res = self.rec.commit(keep, verdict); self.health = None
        if res is None: return None
        return {"kept": bool(keep) and res.get("dataset") == "main", "dataset": res.get("dataset"), "dataset_repo": self.repo if keep else getattr(self.rec, "rejects_repo", None), "episode_index": res.get("episode_index"), "frames": res.get("frames"),
                "discard_reason": None if keep else verdict.get("discard_reason")}


def keep_decision(held, spot_empty, placed, trip, health_ok=True):
    """KEEP only if held after the lift (the gripper readback; the camera only for a wing nut) AND placed (the ring's interior changed by about the toy's area) AND the toy's spot is empty (an extra, never the proof) AND no trip AND a healthy recording. -> (keep, reasons)."""
    why = []
    if held is not True: why.append("not_held" if held is False else "grip_unknown")
    if not spot_empty: why.append("toy_still_at_spot")
    if not placed: why.append("not_in_ring")
    if trip: why.append(f"guardrail:{trip}")
    if not health_ok: why.append("recording_unhealthy")
    return (not why), why


def temp_trip(state, limit_c, mk):
    """A Trip when any follower temperature in /api/state is above limit_c; None when none is published (the caller notes it once)."""
    fol = ((state or {}).get("arms") or {}).get("follower") or {}; vals = []
    for k, v in list(fol.items()) + list((state or {}).items()):
        if "temp" in str(k).lower():
            vals += [float(x) for x in (v.values() if isinstance(v, dict) else (v if isinstance(v, (list, tuple)) else [v])) if isinstance(x, (int, float))]
    return mk("temperature", {"max_c": max(vals), "limit_c": limit_c}) if vals and max(vals) > limit_c else None


def has_temp(state): return any("temp" in str(k).lower() for k in list(((state or {}).get("arms") or {}).get("follower") or {}) + list(state or {}))


class Tidy:
    """The runner. `guard` a guardrails.Guard, `ctx` its Ctx, `track` a LabTrack, `eps` {"ring": Episode, "spread": Episode} (rec None = off), `args` the parsed CLI."""
    def __init__(self, pl, cc, table, rig, guard, ctx, track, args, rng, eps=None, out=print, fetch_top=None):
        self.pl, self.cc, self.table, self.rig, self.guard, self.ctx, self.track, self.a, self.rng, self.out = pl, cc, table, rig, guard, ctx, track, args, rng, out
        self.eps = eps or {}; self.fetch_top = fetch_top or pl.fetch_top; self.n = 0; self.tl = []; self.last_rows = []; self.grip_log = []; self.cur_kind = "nut"; self.stages = []; self.fails = 0; self.in_motion = False
        self.markers_mm = list((table.get("layout_markers") or {}).values()); self.noted = set()
        self.biases = load_biases(os.path.join(getattr(pl, "ROOT", ROOT), "jetson", "calib", "pick_state.json"))

    def say(self, d): self.out(json.dumps(d))
    def stage(self, d): self.stages.append(d); self.say({"event": "stage", "pick": self.n, **d})
    def settle(self, s): self.ctx.sleep(s)

    # -- the world
    def objects(self):
        sc = self.ctx.fetch_scene(); return list((sc or {}).get("objects") or []) if isinstance(sc, dict) and sc.get("ok", True) else []

    def snap(self):
        """(grey frame, (cx, cy, r) of the ring's rim in pixels) from the top camera, or (None, None) when the ring is not seen."""
        import cv2
        img = cv2.imdecode(np.frombuffer(self.fetch_top(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None, None
        band = ring_band(img, self.cc, self.table)
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), band[2]) if band is not None else (None, None)

    def ppm(self, o):
        u, v = self.cc.table_to_pixel(self.table, o["x_in"] * IN, o["y_in"] * IN); return abs(self.cc.table_to_pixel(self.table, o["x_in"] * IN + 10, o["y_in"] * IN)[0] - u) / 10.0

    def settled_objects(self, wait=1.5, cap=4.0):
        self.settle(wait); waited = 0.0
        while True:
            sc = self.ctx.fetch_scene(); pend = int((sc or {}).get("pending") or 0) if isinstance(sc, dict) else 0
            if pend == 0 or waited >= cap: return list((sc or {}).get("objects") or []) if isinstance(sc, dict) else []
            self.settle(0.5); waited += 0.5

    def refit_ring(self):
        r = sl.fit_ring_multi(self.fetch_top, self.cc, self.table)
        if r is None: raise SystemExit("tape ring not found in the top frame")
        self.rig.set_ring(r); return r

    def toys(self, objs=None): return [o for o in (self.objects() if objs is None else objs) if is_toy(o, ring=self.rig.ring) and not ko.is_keepout_object(o, self.rig.keepouts)]

    def refresh_keepouts(self, objs=None):
        """Keep-outs from the scene map + the top frame -> rig.keepouts (the obstacle check), logged when they change. -> the list."""
        import cv2
        objs = self.objects() if objs is None else objs; img = cv2.imdecode(np.frombuffer(self.fetch_top(), np.uint8), cv2.IMREAD_COLOR)
        k = ko.find_keepouts(objs, img, self.cc, self.table, self.rig.ring, RING_R_MM, RING_WALL_IN_MM, RING_WALL_OUT_MM); self.rig.keepouts = k
        key = tuple((round(x["x_in"], 1), round(x["y_in"], 1)) for x in k)
        if key != getattr(self, "_ko_key", None): self._ko_key = key; self.say({"event": "keepouts", "keepouts": [{kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in x.items()} for x in k]})
        return k

    # -- guarded motion
    @contextlib.contextmanager
    def guarded(self):
        pl, orig = self.pl, self.pl.run_plan; self.orig_run = orig; t00 = time.time()
        def run(plan, max_deg_s):
            if t := self.guard.check_plan(plan): raise TripError(t)
            if plan["waypoints"] and plan["waypoints"][0]["name"] == "hover_tgt" and self.rig.fluid: max_deg_s = max(max_deg_s, 80.0)   # the carry needs no precision
            t0 = time.time(); self.in_motion = True
            try: rows = orig(plan, max_deg_s)
            except (Exception, SystemExit) as e: raise TripError(self.guard.trip("run_plan", {"err": f"{type(e).__name__}: {e}"[:300]})) from e
            if rows and rows[-1].get("wp") == "rest": self.in_motion = False   # the plan ended at rest
            if t := self.guard.check_rows(rows): raise TripError(t)   # motion error, then servo health (heat, current, supply, error bits, stall) of every row
            self.last_rows = rows; self.tl.append({"t0": round(t0 - t00, 2), "t1": round(time.time() - t00, 2), "wps": [(x.get("wp"), x.get("t"), x.get("seconds"), (x.get("reached") or {}).get("gripper")) for x in rows]}); return rows
        pl.run_plan = run
        try: yield
        finally: pl.run_plan = orig

    def grip_check_fn(self):
        orig = self.pl.jaw_check
        def grip_check(jpeg=None, window=None, out_path=None, min_score=None, rows=None, **_):   # rows: the lift stage's rows, passed by closed_loop_pick since final-correct
            g = [(x.get("reached") or {}).get("gripper") for x in (rows if rows is not None else self.last_rows) if x.get("wp") == "lift"]; g = g[-1] if g else None; cam = None
            if not (g is not None and g > GRIP_EMPTY_MAX):   # asked for the log (and decisive for a wing nut only)
                try: cam = orig(None, window or self.pl.JAW_WINDOW, out_path)
                except Exception: cam = None
            held = grip_decision(g, (cam or {}).get("in_jaws"), self.cur_kind, ((cam or {}).get("ring") or {}).get("score"))
            self.grip_log.append({"gripper_after_lift": g, "camera_in_jaws": (cam or {}).get("in_jaws"), "camera_score": ((cam or {}).get("ring") or {}).get("score"), "held": held})
            return {"in_jaws": held, "ring": (cam or {}).get("ring"), "frame": (cam or {}).get("frame"), "window": [], "gripper": g, "camera": (cam or {}).get("in_jaws"), "note": f"gripper after the lift {g}; camera {(cam or {}).get('in_jaws')}"}   # gripper/camera: the keys pick_loop.jaw_check returns since final-correct
        return grip_check

    def go_home(self):
        try:
            if self.guard.readback_healthy(): self.orig_run(self.pl.build_plan(None, None, {}, names=("rest",)), self.pl.SPEED["closed"]); self.in_motion = False
        except (Exception, SystemExit): pass

    def send_rest(self, fresh=False):
        """The rest plan on the unwrapped run_plan (the stop file must not block the way home). fresh: end the pick_seq that may still be finishing the interrupted plan and start a new one (only when a session was open)."""
        if fresh and getattr(self.pl, "SESSION", None) is not None: self.pl.close_session(); self.pl.open_session(True)
        (getattr(self, "orig_run", None) or self.pl.run_plan)(self.pl.build_plan(None, None, {}, names=("rest",)), self.pl.SPEED["closed"])

    def park(self, reason, fresh=False):
        """Every exit path ends here: rest if the arm is not already there (guardrails.park_on_exit decides). Torque stays on at rest (pick_seq never releases it)."""
        r = self.guard.park(lambda: self.send_rest(fresh), self.in_motion, reason, self.ctx)
        if r["parked"]: self.in_motion = False
        self.say({"event": "park", "reason": reason, **r}); return r

    def pick_args(self, aim, roll):
        pl = self.pl; return argparse.Namespace(jaw_open=self.a.jaw_open or pl.JAW_OPEN, max_deg_s=pl.SPEED["closed"], wrist_check=False, grasp_aim_mm=aim, grasp_tilt_deg=pl.GRASP_TILT_DEG,
                                                grasp_roll=(f"{roll:.2f}" if roll is not None else "off"), jaw_check=True, fast_close=False, jaw_window=list(pl.JAW_WINDOW))

    def closed_pick(self, obj, dest, aim, roll):
        """closed_loop_pick with this runner's grip check; returns (rows, info)."""
        pl = self.pl; jc = pl.jaw_check
        if self.a.grip_check == "own": pl.jaw_check = self.grip_check_fn()
        try: _, rows, info = pl.closed_loop_pick(self.cc, self.table, list(obj), list(dest), self.biases, self.pick_args(aim, roll), self.n, self.stage)
        finally: pl.jaw_check = jc
        return rows, info

    def ring_pick(self, toy):
        """One toy into the ring, recorded when --record. Returns the result dict (never raises on a trip: it is in the result)."""
        pl, rig, a = self.pl, self.rig, self.a; self.n += 1; n = self.n; obj = (float(toy["x_in"]), float(toy["y_in"])); kind, aim = toy_kind(toy); ep = self.eps.get("ring") or Episode(None, RING_REPO)
        self.tl = []; self.stages = []; t0 = time.time(); trip = None; rec = {"pick": n, "kind": kind, "toy_in": [round(v, 3) for v in obj], "ring_in": [round(v, 3) for v in rig.ring], "aim_mm": aim}
        chk = rig.check(obj); rec["check"] = {k: chk[k] for k in ("reachable", "ok", "lift_in", "direct", "end_direct", "n_bad", "worst", "why")}
        if not chk["ok"]: return {**rec, "refused": chk["why"] or "check failed", "ok": False, "trip": None}
        before = self.count_in_ring(self.objects()); snap0 = self.snap(); self.grip_log = []; self.cur_kind = kind
        roll, rinfo = (None, {"why": "off"}) if a.no_roll else self.roll_at_rest(obj); rec["grasp_roll_deg"] = roll
        meta = {"task": TASK_RING, "scene": {"toy_kind": kind, "toy_in": rec["toy_in"], "ring_in": rec["ring_in"], "grasp_roll_deg": roll, "aim_mm": aim, "lighting_tag": a.lighting_tag or "untagged"},
                "teacher": {"session": a.session, "pick": n}}
        ep.open(meta); held = None; info = {}; started = False
        with self.track.run(f"tidy-{a.session}-p{n}", tags={"pick": n, "kind": kind, "toy_in": f"{obj[0]:.2f},{obj[1]:.2f}", "session": a.session}) as tr:
            tr.params({"kind": kind, "aim_mm": aim, "grasp_roll": roll, "ring_in": f"{rig.ring[0]:.3f},{rig.ring[1]:.3f}", "toy_in": f"{obj[0]:.3f},{obj[1]:.3f}", "grip_check": a.grip_check, "direct_route": int(bool(chk["direct"])), "end_direct": int(bool(chk["end_direct"])),
                       "lift_in": chk["lift_in"], "recording": int(ep.rec is not None), "lighting_tag": a.lighting_tag or "untagged"})
            try:
                with self.guarded(), rig.patched(chk["lift_in"], chk["hover_in"]):
                    rig.end_direct = chk["end_direct"]; route = rig.start_route(obj, chk["lift_in"]); started = True
                    if route:
                        self.pl.run_plan({"waypoints": [{"name": nm, "joints": jj, "desired": jj, "gripper": a.jaw_open or pl.JAW_OPEN, "seconds": 1.2, "settle": 0.2, "gripper_only": False} for nm, jj in route]}, pl.SPEED["closed"])
                    rows, info = self.closed_pick(obj, rig.ring, aim, roll)
                held = info.get("in_jaws")
            except TripError as e: trip = e.trip
            except SystemExit as e: trip = self.guard.trip("pick_error", {"err": str(e)[:200]})
            if trip:   # a trip mid-pick: nothing partial is saved (aborted, not rejected: the arm never finished the pick)
                ep.abort(trip.reason); self.say({"event": "trip", "pick": n, **trip.as_json()})
                if started: self.go_home()
            else: ep.end()   # why: the episode ends AFTER the arm is back at rest (the carry stage is one pick_seq plan: hover_tgt, open, rest): the end state equals the start state, so a policy learns to park, and no stage boundary is added just for the recorder (2026-11-11)
            objs = self.settled_objects(); spot_empty = not any(dist_mm((o["x_in"], o["y_in"]), obj) < 15.0 for o in self.toys(objs)); after = self.count_in_ring(objs); in_ring = after > before
            snap1 = self.snap(); circ = snap0[1] or snap1[1]
            pv = placed_verdict(snap0[0], snap1[0], circ, float(toy.get("area_mm2") or TOY_AREA_MM2), self.ppm(toy)) if (snap0[0] is not None and snap1[0] is not None and circ is not None) else {"placed": False, "diff_px": None, "need_px": None, "why": "no frame"}
            if trip: pv = {**pv, "placed": False}
            health_ok = True if ep.health is None else bool((ep.health or {}).get("ok", True))
            keep, why = keep_decision(held, spot_empty, pv["placed"], trip.reason if trip else None, health_ok)
            res = {**rec, "held": held, "placed": pv["placed"], "interior": pv, "grip": self.grip_log, "spot_empty": spot_empty, "in_ring": in_ring, "ring_count": [before, after], "reason": info.get("reason"), "residual_mm": info.get("final_residual_mm"), "route": "direct" if chk["direct"] else "over at 110 mm", "seconds": round(time.time() - t0, 1),
                   "trip": trip.reason if trip else None, "ok": bool(held is True and spot_empty and pv["placed"] and not trip), "keep": keep, "keep_reasons": why}
            out = None if trip else ep.commit(keep, {"keep": keep, "discard_reason": why[0] if why else None, "reasons": why, "held": held, "placed": pv, "spot_empty": spot_empty, "grip": self.grip_log}, {"keep": keep, "held": held, "residual_mm": info.get("final_residual_mm"), "reasons": why, "grip": self.grip_log, "interior": pv, "spot_empty": spot_empty})
            res["demo"] = out
            tr.metrics({"held": None if held is None else int(held), "spot_empty": int(spot_empty), "in_ring": int(in_ring), "placed": int(pv["placed"]), "interior_diff_px": pv["diff_px"], "interior_need_px": pv["need_px"], "gripper_after_lift": (self.grip_log[-1] if self.grip_log else {}).get("gripper_after_lift"), "camera_score": (self.grip_log[-1] if self.grip_log else {}).get("camera_score"), "ok": int(res["ok"]), "seconds": res["seconds"], "final_residual_mm": info.get("final_residual_mm"), "aim_mm": aim, "kept": None if out is None else int(bool(out["kept"]))})
            if out: tr.params({"dataset_repo": out.get("dataset_repo"), "kept": int(bool(out["kept"])), "discard_reason": out.get("discard_reason"), "episode_index": out.get("episode_index")})
            tr.json({"pick": res, "stages": self.stages, "timeline": self.tl}, "timeline.json")
        self.save_timeline(n, res); self.say({"event": "pick", **res}); return res

    def save_timeline(self, n, res):
        d = os.path.join(getattr(self.a, "out_dir", None) or os.path.join(ROOT, "outputs", "run"), f"tidy_{self.a.session}"); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"pick_{n}.json"), "w") as f: json.dump({"pick": res, "timeline": self.tl, "stages": self.stages}, f, indent=1, default=str)

    def count_in_ring(self, objs): return sum(1 for o in self.toys(objs) if inside_ring(o, self.rig.ring))

    def roll_at_rest(self, obj):
        """Wrist roll for an elongated toy from the scene map's angle, resolved with the arm at rest (the map cannot see a toy under the arm)."""
        try: return self.pl.resolve_grasp_roll("auto", self.cc, self.table, list(obj), tilt_deg=self.pl.GRASP_TILT_DEG)
        except (Exception, SystemExit) as e: return None, {"why": f"{type(e).__name__}: {e}"[:80]}

    # -- the round
    def before(self):
        """A Trip or None: stop file, panel, arm_busy, overload, scene health (Guard.check_before_cycle), temperature."""
        if t := self.guard.check_before_cycle(self.ctx): return t
        try: st = self.ctx.fetch_state()
        except Exception: return None
        t = temp_trip(st, self.a.temp_limit, self.guard.trip)
        if t: return t
        if "temp" not in self.noted and not has_temp(st): self.noted.add("temp"); self.say({"note": "temperature not checked: /api/state publishes none"})
        for e in self.eps.values():
            if e.rec is not None and hasattr(e.rec, "alive") and not e.rec.alive(): return self.guard.trip("recorder", {"err": "the demo recorder does not answer"})
        return t

    def round(self, max_n=None):
        """Tidy until no toy is left outside the ring, max_n picks, a trip, or MAX_FAILS failures in a row. -> summary dict."""
        s = {"picked": 0, "ok": 0, "failed": 0, "refused": [], "stop": None, "results": []}
        while True:
            if max_n is not None and s["picked"] >= max_n: s["stop"] = "max"; break
            if t := self.before(): s["stop"] = t.reason; s["trip"] = t.as_json(); self.say({"event": "stop", **t.as_json()}); break
            self.refit_ring(); objs = self.objects(); self.refresh_keepouts(objs)
            if bad := ko.inside_ring(self.rig.keepouts, self.rig.ring, RING_R_MM): s["stop"] = "keep-out inside the ring"; self.say({"event": "stop", "reason": s["stop"], "keepouts": bad}); break   # never drop a toy on a frog
            toys = [o for o in self.toys(objs) if not inside_ring(o, self.rig.ring)]
            if not toys: s["stop"] = "all_tidy"; break
            verdicts = [self.rig.check((o["x_in"], o["y_in"])) for o in toys]; order = order_toys(toys, verdicts, self.rig.base, self.rig.ring, self.a.order)
            s["refused"] = [{"toy_in": [round(o["x_in"], 2), round(o["y_in"], 2)], "why": v["why"]} for o, v in zip(toys, verdicts) if not v["ok"]]
            if not order: s["stop"] = "refused: no toy has a clean route"; self.say({"event": "stop", "reason": s["stop"], "refused": s["refused"]}); break
            res = self.ring_pick(order[0]); s["picked"] += 1; s["results"].append(res)
            if res.get("trip"): s["stop"] = res["trip"]; break
            if res["ok"]: s["ok"] += 1
            else: s["failed"] += 1
            if t := self.guard.check_after_cycle({"ok": res["ok"], "found": True}): s["stop"] = t.reason; s["trip"] = t.as_json(); self.say({"event": "stop", **t.as_json()}); break
        return s


def load_biases(path):
    try:
        with open(path) as f: b = json.load(f).get("biases") or {}
        return b if isinstance(b, dict) else {}
    except (OSError, ValueError): return {}


# ---------------------------------------------------------------- reset: lift the ring, set it down elsewhere
def loaded_sag_mm(reach_mm): return max(8.0, 24.0 * reach_mm / 230.0)   # why: 11 Nov 21:0x the arm sagged ~24 mm under the ring (commanded 78, reached 54-59) at a ~230 mm reach; scaled linearly with reach (assumption, not measured)


def ring_bottom_mm(tip_h_mm, reach_mm, grip_h_mm=RING_GRIP_H_MM): return tip_h_mm - loaded_sag_mm(reach_mm) - grip_h_mm


def lift_height_mm(reach_mm, max_free_mm, want_bottom=MIN_BOTTOM_MM, grip_h_mm=RING_GRIP_H_MM, cap=RING_LIFT_CAP_MM):
    """(commanded tip height, predicted ring bottom): as high as the tipped IK reaches up to `cap`, aiming for want_bottom after the sag."""
    cmd = min(cap, max_free_mm); return cmd, ring_bottom_mm(cmd, reach_mm, grip_h_mm)


RING_MARKER_MM = 61.0   # ring wall to marker centre: the marker's half-diagonal 36 (50.8 mm square) + ~10 of white border + 15 of set-down error (12 Nov 09:2x: the ring landed 13 mm off and covered marker 2, the camera lost it and the guard stopped the run)


def ring_spot_ok(p, cur_in, pile_in, others_in, markers_mm, ring_r_mm=RING_R_MM, clear_mm=1.5 * IN, edge_mm=30.0, marker_mm=RING_MARKER_MM, min_move_mm=60.0, mat_mm=(457.2, 304.8)):
    """None when p (mat inches) is a fine new ring centre, else why not: mat edges, clearance to the pile and every toy, markers, and a real move."""
    px, py = p[0] * IN, p[1] * IN
    if min(px, py, mat_mm[0] - px, mat_mm[1] - py) < ring_r_mm + edge_mm: return "too close to the mat edge"
    if any(dist_mm(p, q) < ring_r_mm + clear_mm for q in list(pile_in) + list(others_in)): return "too close to a toy or the pile"
    if any(math.hypot(px - m[0], py - m[1]) < ring_r_mm + marker_mm for m in markers_mm): return "would cover or crowd a marker"
    if cur_in is not None and dist_mm(p, cur_in) < min_move_mm: return "not a real move"
    return None


def sample_ring_spot(rng, box_in, cur_in, pile_in, others_in, markers_mm, feasible=None, ring_r_mm=RING_R_MM, clear_mm=1.5 * IN, edge_mm=30.0, marker_mm=RING_MARKER_MM, min_move_mm=60.0, mat_mm=(457.2, 304.8), tries=400):
    """A new centre (mat inches) for the ring inside `box_in`: >= clear_mm (1.5 in) of free space between the ring's wall and the pile and every other toy, ring wall >= edge_mm from the mat edges and
    marker_mm from every marker centre, >= min_move_mm from where it stands, and `feasible(p)` true (the arm can grip and set it down there). None when `tries` draws find none."""
    x0, y0, x1, y1 = box_in
    for _ in range(tries):
        p = [round(rng.uniform(x0, x1), 3), round(rng.uniform(y0, y1), 3)]
        if ring_spot_ok(p, cur_in, pile_in, others_in, markers_mm, ring_r_mm, clear_mm, edge_mm, marker_mm, min_move_mm, mat_mm): continue
        if feasible is not None and not feasible(p): continue
        return p
    return None


class RingMover:
    """Lift the ring by its wall (jaws radial on the side facing the base, tips 15 mm up the 51 mm wall), carry it on a tipped lift, set it down and go home. `t` is the Tidy runner."""
    def __init__(self, t): self.t, self.pl, self.rig, self.cc, self.table = t, t.pl, t.rig, t.cc, t.table

    def ik(self, x, y, h_mm, roll=None):
        q = f"/api/ik?x_in={x:.4f}&y_in={y:.4f}&h_in={h_mm / IN:.4f}" + (f"&roll={roll:.3f}" if roll is not None else ""); r = self.pl.api(q)
        if not r.get("ok"): raise SystemExit(f"ik failed ({x:.2f},{y:.2f},{h_mm:.0f} mm): {r.get('err')}")
        return r["joints_deg"]

    def grip_point(self, c, rot_deg=0.0):
        """A point on the wall midline and the unit vector to it from the centre: the side facing the arm's base, turned by rot_deg (another side when that one is out of reach)."""
        u = (self.rig.base[0] - c[0], self.rig.base[1] - c[1]); n = math.hypot(*u); a = math.radians(rot_deg); u = (u[0] / n, u[1] / n); u = (u[0] * math.cos(a) - u[1] * math.sin(a), u[0] * math.sin(a) + u[1] * math.cos(a))
        r = WALL_MID_MM / IN; return (c[0] + r * u[0], c[1] + r * u[1]), u

    def vertical_h(self, x, y, want_mm, floor_mm, roll=None):
        """The highest vertical-gripper height <= want_mm (3 mm steps) the IK reaches at (x, y), or None below floor_mm."""
        h = want_mm
        while h >= floor_mm:
            try: self.ik(x, y, h, roll); return h
            except SystemExit: h -= 3.0
        return None

    def radial_roll(self, p, u):
        pl = self.pl; j0 = self.ik(p[0], p[1], RING_GRIP_H_MM); d0 = pl.jaw_dir_mat(self.cc, self.table, j0); a0 = math.atan2(d0[1], d0[0]); best = None
        for tgt in (math.atan2(u[1], u[0]), math.atan2(-u[1], -u[0])):
            dd = math.degrees((tgt - a0 + math.pi) % (2 * math.pi) - math.pi) / 1.02   # +1 deg roll turns jaw_dir ~+1.02 deg ccw (measured)
            if best is None or abs(dd) < abs(best): best = dd
        roll = float(j0["wrist_roll"]) + best; return roll, pl.jaw_dir_mat(self.cc, self.table, self.ik(p[0], p[1], RING_GRIP_H_MM, roll))

    def max_free_h(self, x, y, roll):
        h = RING_LIFT_CAP_MM
        while h >= 40.0:
            try: return h, self.rig.ik_free(x, y, h, roll)
            except Exception: h -= 5.0
        return 0.0, None

    def reach_mm(self, x, y): return math.hypot(x - self.rig.base[0], y - self.rig.base[1]) * IN

    def way_home_hits(self, dest, up_joints):
        """True when, with the ring standing at dest, the arm's rest pose or its path there from the lift-off pose touches the ring. why: 12 Nov 2026 the ring was
        set down at (8.72, 3.82); the rest pose's gripper tip (9.02, 3.21, 13 mm up) is inside that footprint, and going home after the release shoved the ring 4 in."""
        old = self.rig.ring
        try:
            self.rig.set_ring(dest); return bool(self.rig.hits(self.pl.REST)) or not self.rig.path_clean(up_joints, self.pl.REST)
        finally: self.rig.set_ring(old)

    GRIP_SIDES_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)

    def plan(self, c, dest):
        """Everything the move needs at nominal positions: grip side and point, roll, aim, hover height, lift height and the predicted ring bottom.
        Every pose the move uses is checked here (12 Nov 2026: a vertical hover 133 mm from the base had no IK and the move stopped after the spot was chosen);
        the side facing the base first, then the wall turned by GRIP_SIDES_DEG. The ring hangs rigidly from the jaws, so the set-down aim is the grip aim moved
        by exactly the ring's displacement (12 Nov: re-picking the base-facing side at the destination shifted the set-down 13 mm and the ring covered a marker).
        SystemExit when no side works."""
        why = []
        for rot in self.GRIP_SIDES_DEG:
            try:
                p, u = self.grip_point(c, rot); roll, jd = self.radial_roll(p, u); a = (p[0] - RING_AIM_MM / IN * jd[0], p[1] - RING_AIM_MM / IN * jd[1])
                b = (a[0] + dest[0] - c[0], a[1] + dest[1] - c[1])
                hov = self.vertical_h(a[0], a[1], RING_HOVER_H_MM, RING_PRE_H_MM + 3.0, roll)
                if hov is None: why.append(f"{rot:+.0f}: no hover"); continue
                self.ik(*a, RING_PRE_H_MM, roll); self.ik(*a, RING_GRIP_H_MM, roll); self.ik(*b, RING_GRIP_H_MM + 1.0, roll)
                ha, ja = self.max_free_h(a[0], a[1], roll); hb, jb = self.max_free_h(b[0], b[1], roll)
                if ja is None or jb is None: why.append(f"{rot:+.0f}: no tipped lift"); continue
                cmd = min(ha, hb); reach = max(self.reach_mm(*a), self.reach_mm(*b)); cmd, bottom = lift_height_mm(reach, cmd)
                if bottom < HARD_MIN_BOTTOM_MM: why.append(f"{rot:+.0f}: ring bottom {bottom:.0f} mm after the sag at reach {reach:.0f} mm"); continue
                if self.way_home_hits(dest, self.rig.ik_free(*b, cmd, roll)): raise SystemExit("the set-down spot is in the arm's way home (rest or the path to it would hit the ring there)")
                return {"grip_in": p, "aim_in": a, "drop_aim_in": b, "roll": roll, "jaw_dir": jd, "hover_mm": hov, "grip_side_deg": rot, "lift_mm": cmd, "bottom_mm": bottom, "reach_mm": reach, "min_bottom_wanted": MIN_BOTTOM_MM}
            except SystemExit as e: why.append(f"{rot:+.0f}: {str(e)[:60]}")
        raise SystemExit("ring lift: no grip side works (" + "; ".join(why)[:300] + ")")

    def W(self, n, j, g, s=1.0, st=0.2): return {"name": n, "joints": j, "desired": j, "gripper": g, "seconds": s, "settle": st, "gripper_only": False}

    def lift_ring(self, dest):
        """The whole move; returns {"ok", "why", "plan", "measured_in", "err_mm"}. The caller holds the guard."""
        t, pl, rig = self.t, self.pl, self.rig; c = rig.ring; P = self.plan(c, dest); a, b, roll, cmd = P["aim_in"], P["drop_aim_in"], P["roll"], P["lift_mm"]; out = {"ok": False, "why": None, "plan": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items() if k != "jaw_dir"}}
        if rig.keepouts:   # the arm's own path (nominal poses) vs the keep-outs, before anything moves; the ring itself is what it grips, so only keep-outs count
            seq = [dict(pl.REST), self.ik(*a, P["hover_mm"], roll), self.ik(*a, RING_PRE_H_MM, roll), self.ik(*a, RING_GRIP_H_MM, roll), rig.ik_free(*a, cmd, roll), rig.ik_free(*b, cmd, roll), self.ik(*b, RING_GRIP_H_MM + 1.0, roll), rig.ik_free(*b, cmd, roll), dict(pl.REST)]
            if any(rig.hits(rig.lerp(u, v, f), ring=False) for u, v in zip(seq, seq[1:]) for f in np.linspace(0, 1, 12)): out["why"] = "ring move passes a keep-out"; return out
        with t.guarded():
            r1 = pl.run_plan({"waypoints": [self.W("hover", self.ik(*a, P["hover_mm"], roll), RING_JAW, 1.6, 0.4)]}, 50.0); x, y, _ = pl.fk_mat(self.cc, self.table, pl.reached_joints(r1, "hover"))
            a2 = (a[0] + (a[0] - x), a[1] + (a[1] - y)); out["corr_mm"] = [round((a2[0] - a[0]) * IN, 1), round((a2[1] - a[1]) * IN, 1)]   # one readback correction at the hover
            lift_j = rig.ik_free(*a2, cmd, roll)
            rows = pl.run_plan({"waypoints": [self.W("pregrasp", self.ik(*a2, RING_PRE_H_MM, roll), RING_JAW, 0.8), self.W("grip", self.ik(*a2, RING_GRIP_H_MM, roll), RING_JAW, 1.0, 0.4), self.W("close", self.ik(*a2, RING_GRIP_H_MM, roll), 0, 0.8, 0.6),
                                            self.W("lift", lift_j, 0, 1.2, 0.3)]}, 50.0)
            g = [(r.get("reached") or {}).get("gripper") for r in rows if r["wp"] == "lift"]; g = g[-1] if g else None; tip = pl.fk_mat(self.cc, self.table, pl.reached_joints(rows, "lift"))
            real_bottom = tip[2] - RING_GRIP_H_MM; out.update({"gripper_after_lift": g, "reached_tip_mm": round(tip[2], 1), "bottom_after_lift_mm": round(real_bottom, 1)})   # the readback tip height already contains the sag the arm settled into
            if g is None or g <= GRIP_EMPTY_MAX: out["why"] = "ring not gripped"; pl.run_plan({"waypoints": [self.W("open", self.ik(*a2, RING_PRE_H_MM, roll), RING_JAW, 0.8), self.W("rest", dict(pl.REST), 0, 1.8, 0.3)]}, 50.0); return out
            if real_bottom < HARD_MIN_BOTTOM_MM:   # the ring hangs too low to carry over the pile: put it back where it was
                out["why"] = f"ring bottom {real_bottom:.0f} mm after the lift: put back"; pl.run_plan({"waypoints": [self.W("lower", self.ik(*a2, RING_GRIP_H_MM + 1.0, roll), 0, 1.2, 0.4), self.W("open", self.ik(*a2, RING_GRIP_H_MM + 1.0, roll), RING_JAW, 0.6, 0.4),
                                                                                                                        self.W("up", rig.ik_free(*a2, cmd, roll), RING_JAW, 1.0), self.W("rest", dict(pl.REST), 0, 1.8, 0.3)]}, 50.0); return out
            pl.run_plan({"waypoints": [self.W("carry", rig.ik_free(*b, cmd, roll), 0, 1.6, 0.3), self.W("lower", self.ik(*b, RING_GRIP_H_MM + 1.0, roll), 0, 1.2, 0.4)]}, 80.0)   # lower until the tip is ~1 mm above the grip height
            pl.run_plan({"waypoints": [self.W("open", self.ik(*b, RING_GRIP_H_MM + 1.0, roll), RING_JAW, 0.6, 0.4), self.W("up", rig.ik_free(*b, cmd, roll), RING_JAW, 1.0, 0.2), self.W("rest", dict(pl.REST), 0, 1.8, 0.3)]}, 50.0)
        measured = t.refit_ring(); out.update({"ok": True, "measured_in": [round(v, 3) for v in measured], "err_mm": round(dist_mm(measured, dest), 1), "commanded_in": [round(v, 3) for v in dest]})   # the next round uses the measured centre
        return out


# ---------------------------------------------------------------- reset: spread the pile
def sample_spread_spot(rng, box_in, others_in, markers_mm, ring_in, cur_in, ring_r_mm=RING_R_MM, ring_gap_mm=40.0, min_obj=40.0, min_marker=50.0, min_cur=60.0, avoid=(), tries=500):
    """Autoloop's sample_spot rules (>= 40 mm from every other object, 50 mm from a marker centre, 60 mm from where the toy is) plus: clear of the ring's wall by ring_gap_mm and outside every (centre_in, radius_mm) of `avoid` (the pile)."""
    x0, y0, x1, y1 = box_in
    for _ in range(tries):
        p = [round(rng.uniform(x0, x1), 3), round(rng.uniform(y0, y1), 3)]
        if dist_mm(p, ring_in) < ring_r_mm + ring_gap_mm or any(dist_mm(p, c) < r for c, r in avoid): continue
        if any(dist_mm(p, o) < min_obj for o in others_in) or any(dist_mm(p, [m[0] / IN, m[1] / IN]) < min_marker for m in markers_mm): continue
        if cur_in is not None and dist_mm(p, cur_in) < min_cur: continue
        return p
    return None


def blob_count(o): return max(1, int(round(float(o.get("area_mm2") or TOY_AREA_MM2) / TOY_AREA_MM2)))


def is_single(o):
    long_ = max(float(o.get("w_mm") or 0), float(o.get("h_mm") or 0)); return long_ <= TOY_LONG_MM and float(o.get("area_mm2") or 0) <= 1.6 * TOY_AREA_MM2


def split_mask(mask, min_area_px=12, max_steps=8):
    """The toy-sized pieces of a blob mask (uint8 0/255), by growing erosion until it falls apart: [(cx, cy, area_px, solidity)] in pixels, most compact first. One piece when it never splits."""
    import cv2
    m = (mask > 0).astype(np.uint8); k = np.ones((3, 3), np.uint8); cur = m; best = None
    for _ in range(max_steps):
        n, lab, stats, cen = cv2.connectedComponentsWithStats(cur, connectivity=8); comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
        if len(comps) >= 2 or not comps: best = comps and (lab, stats, cen, comps) if comps else best; break
        best = (lab, stats, cen, comps); cur = cv2.erode(cur, k)
    if not best: return []
    lab, stats, cen, comps = best; out = []
    for i in comps:
        c = (lab == i).astype(np.uint8); cs, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); hull = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))); area = float(stats[i, cv2.CC_STAT_AREA])
        out.append((float(cen[i][0]), float(cen[i][1]), area, area / hull if hull > 0 else 0.0))
    return sorted(out, key=lambda t: -t[3])


def geometric_candidates(o, r_mm=6.0):
    """Without a mask: the blob's rotated rectangle (w_mm x h_mm at angle_deg) ends / corners, a toy radius in from the edge; outermost first. A single toy: its centre."""
    c = (float(o["x_in"]), float(o["y_in"]))
    if is_single(o): return [c]
    w, h = float(o.get("w_mm") or 0), float(o.get("h_mm") or 0); ang = math.radians(float(o.get("angle_deg") or 0.0)); u = (math.cos(ang), math.sin(ang)); v = (-u[1], u[0])
    sx = max(w / 2 - r_mm, 0.0); sy = max(h / 2 - r_mm, 0.0); outs = []
    for a in ((1, -1) if sx > 1 else (0,)):
        for b in ((1, -1) if sy > 1 else (0,)): outs.append((c[0] + (a * sx * u[0] + b * sy * v[0]) / IN, c[1] + (a * sx * u[1] + b * sy * v[1]) / IN))
    return sorted(outs, key=lambda p: -dist_mm(p, c)) or [c]


def pile_candidates(o, mask_fn=None):
    """Pick points (mat inches) for one scene blob, best first: the most compact toy-sized piece of the blob's mask when `mask_fn(o)` gives one ([(x_in, y_in, area_px, solidity)]), else the geometric ends."""
    if is_single(o): return [(float(o["x_in"]), float(o["y_in"]))]
    if mask_fn is not None:
        try:
            pieces = mask_fn(o)
            if pieces: return [(p[0], p[1]) for p in pieces]
        except Exception: pass
    return geometric_candidates(o)


def nudge_waypoints(c_in, ring_in, ik, jaw_open, sweep_mm=30.0, low_mm=7.0, up_mm=50.0):
    """A short low sweep through the blob centre with the jaws open, away from the ring: [(name, joints, gripper)] — approach, down, sweep, up. `ik(x, y, h_mm)` -> joints."""
    d = (c_in[0] - ring_in[0], c_in[1] - ring_in[1]); n = math.hypot(*d) or 1.0; d = (d[0] / n, d[1] / n); s = sweep_mm / IN
    p0, p1 = (c_in[0] - s * d[0], c_in[1] - s * d[1]), (c_in[0] + s * d[0], c_in[1] + s * d[1])
    return [("nudge_above", ik(*p0, up_mm), jaw_open), ("nudge_in", ik(*p0, low_mm), jaw_open), ("nudge_through", ik(*p1, low_mm), jaw_open), ("nudge_up", ik(*p1, up_mm), jaw_open)]


def make_blob_mask_fn(t):
    """mask_fn for pile_candidates from the live top frame: the blob's box in pixels, foreground = differs from the box's border by > 25 grey levels, split_mask on that. Not validated live (docs/tidy.md)."""
    import cv2
    def fn(o):
        img = cv2.imdecode(np.frombuffer(t.fetch_top(), np.uint8), cv2.IMREAD_GRAYSCALE); cc, tab = t.cc, t.table; u, v = cc.table_to_pixel(tab, o["x_in"] * IN, o["y_in"] * IN); ppm = float(cc.table_to_pixel(tab, o["x_in"] * IN + 10, o["y_in"] * IN)[0] - u) / 10.0
        hw = int((max(float(o.get("w_mm") or 0), float(o.get("h_mm") or 0)) / 2 + 8) * abs(ppm)); x0, y0 = max(int(u) - hw, 0), max(int(v) - hw, 0); roi = img[y0:int(v) + hw, x0:int(u) + hw]
        if roi.size == 0: return []
        border = np.median(np.r_[roi[0], roi[-1], roi[:, 0], roi[:, -1]]); mask = (np.abs(roi.astype(int) - border) > 25).astype(np.uint8) * 255; out = []
        for px, py, ar, sol in split_mask(mask):
            X, Y = cc.pixel_to_table(tab, px + x0, py + y0); out.append((X / IN, Y / IN, ar, sol))
        return out
    return fn


def spread_pile(t, pile_in, box_in, nudge=True, max_attempts=None):
    """Spread the toys of the pile onto random clear spots. One closed-loop pick each (the normal place); the pile's blobs are split into toy-sized pieces; a blob that fails twice is nudged once. -> summary."""
    rig = t.rig; s = {"moved": 0, "failed": 0, "nudged": 0, "attempts": 0, "stop": None, "spots": []}; fails = {}; nudged = set(); n_toys = len([o for o in t.toys() if dist_mm((o["x_in"], o["y_in"]), pile_in) < PILE_R_MM])
    cap = max_attempts or 3 * max(n_toys, 1) + 6; mask_fn = make_blob_mask_fn(t) if t.a.blob_mask else None
    while s["attempts"] < cap:
        if tr := t.before(): s["stop"] = tr.reason; break
        objs = t.settled_objects(); t.refresh_keepouts(objs); pile = [o for o in t.toys(objs) if dist_mm((o["x_in"], o["y_in"]), pile_in) < PILE_R_MM]
        if not pile: break
        blob = max(pile, key=lambda o: (blob_count(o), -dist_mm((o["x_in"], o["y_in"]), pile_in))); key = (round(blob["x_in"], 1), round(blob["y_in"], 1))
        if fails.get(key, 0) >= 2 and nudge and key not in nudged and dist_mm((blob["x_in"], blob["y_in"]), rig.ring) > RING_R_MM + 90.0 and all(dist_mm((blob["x_in"], blob["y_in"]), (k["x_in"], k["y_in"])) > k["r_mm"] + 90.0 for k in rig.keepouts):
            nudged.add(key); s["nudged"] += 1; do_nudge(t, blob); continue
        others = [[o["x_in"], o["y_in"]] for o in t.toys(objs) if o is not blob]
        avoid = [(pile_in, PILE_R_MM + 15.0)] + [((k["x_in"], k["y_in"]), k["r_mm"] + 40.0) for k in rig.keepouts]
        spot = sample_spread_spot(t.rng, box_in, others + list(s["spots"]), t.markers_mm, rig.ring, (blob["x_in"], blob["y_in"]), avoid=avoid)
        if spot is None: s["stop"] = "no free spot in the spread box"; break
        for _try in range(12):   # a spot whose carry passes the ring is not used: draw another
            cand = next((c for c in pile_candidates(blob, mask_fn) if rig.check(c, dest=spot)["ok"]), None)
            if cand is not None: break
            spot = sample_spread_spot(t.rng, box_in, others + list(s["spots"]), t.markers_mm, rig.ring, (blob["x_in"], blob["y_in"]), avoid=avoid)
            if spot is None: break
        if cand is None or spot is None: s["stop"] = "refused: no clean route for the pile"; break
        s["attempts"] += 1; t.guard.count_try(); res = spread_pick(t, blob, cand, spot); t.say({"event": "spread", **res})
        if res.get("trip"): s["stop"] = res["trip"]; break
        if res["ok"]: s["moved"] += 1; s["spots"].append(spot); fails.pop(key, None); t.fails = 0
        else:
            s["failed"] += 1; fails[key] = fails.get(key, 0) + 1; t.fails += 1
            if t.fails >= MAX_FAILS: s["stop"] = "consecutive_failures"; break
    return s


def spread_pick(t, blob, cand, spot):
    """One toy of the pile to `spot`, the plain closed-loop carry; optionally recorded into the spread dataset. -> {ok, held, landed...}."""
    t.n += 1; n = t.n; kind, aim = toy_kind(blob if is_single(blob) else {"w_mm": 12, "h_mm": 10}); ep = t.eps.get("spread") or Episode(None, SPREAD_REPO); t.cur_kind = kind; t.grip_log = []; t.tl = []; t.stages = []; trip = None; held = None; info = {}
    ep.open({"task": TASK_SPREAD, "scene": {"toy_kind": kind, "toy_in": [round(v, 3) for v in cand], "spot_in": spot, "aim_mm": aim, "lighting_tag": t.a.lighting_tag or "untagged"}, "teacher": {"session": t.a.session, "pick": n, "phase": "spread"}})
    try:
        with t.guarded(): _, info = t.closed_pick(cand, spot, aim, None)
        held = info.get("in_jaws")
    except TripError as e: trip = e.trip; ep.abort(trip.reason); t.go_home()
    except SystemExit as e: trip = t.guard.trip("pick_error", {"err": str(e)[:200]}); ep.abort("pick_error"); t.go_home()
    ep.end(); objs = t.settled_objects(); landed = [o for o in t.toys(objs) if dist_mm((o["x_in"], o["y_in"]), spot) < 15.0]; ok = bool(held is True and landed and not trip)
    keep, why = keep_decision(held, True, bool(landed), trip.reason if trip else None, True)
    out = ep.commit(keep, {"keep": keep, "discard_reason": why[0] if why else None, "reasons": why}, {"keep": keep}) if ep.rec is not None and not trip else None
    return {"pick": n, "ok": ok, "held": held, "spot_in": spot, "from_in": [round(v, 3) for v in cand], "landed": bool(landed), "trip": trip.reason if trip else None, "demo": out}


def do_nudge(t, blob):
    pl = t.pl; c = (float(blob["x_in"]), float(blob["y_in"])); jaw = t.a.jaw_open or pl.JAW_OPEN
    def ik(x, y, h):
        r = pl.api(f"/api/ik?x_in={x:.4f}&y_in={y:.4f}&h_in={h / IN:.4f}")
        if not r.get("ok"): raise SystemExit("nudge: ik failed")
        return r["joints_deg"]
    try:
        wps = nudge_waypoints(c, t.rig.ring, ik, jaw)
        with t.guarded(): pl.run_plan({"waypoints": [{"name": n, "joints": j, "desired": j, "gripper": g, "seconds": 1.2, "settle": 0.2, "gripper_only": False} for n, j, g in wps] + [{"name": "rest", "joints": dict(pl.REST), "desired": dict(pl.REST), "gripper": 0, "seconds": 1.8, "settle": 0.3, "gripper_only": False}]}, 30.0)
        t.say({"event": "nudge", "blob_in": [round(v, 3) for v in c]})
    except (TripError, SystemExit, Exception) as e: t.say({"event": "nudge_failed", "err": str(e)[:120]}); t.go_home()


def reset_mat(t, dest=None):
    """Lift the ring off the pile, set it down at a random clear spot (or `dest`), re-fit it, spread the pile. -> summary."""
    rig = t.rig; objs = t.settled_objects(); kos = t.refresh_keepouts(objs); toys = t.toys(objs); pile = [(o["x_in"], o["y_in"]) for o in toys if inside_ring(o, rig.ring, RING_R_MM)]; pile_c = rig.ring; others = [(o["x_in"], o["y_in"]) for o in toys if (o["x_in"], o["y_in"]) not in pile]
    if ko.inside_ring(kos, rig.ring, RING_R_MM): return {"ok": False, "stop": "keep-out inside the ring"}   # lifting the ring would drag its wall over it
    mover = RingMover(t)
    def feasible(p):   # the set-down clear of every keep-out, and the carried ring's path (its full width) passing none
        if any(dist_mm(p, (k["x_in"], k["y_in"])) < RING_R_MM + k["r_mm"] + 10.0 for k in kos) or not ko.segment_clear(rig.ring, p, kos, RING_R_MM + 10.0): return False
        try: mover.plan(rig.ring, p); return True
        except (SystemExit, Exception): return False
    spot = dest or sample_ring_spot(t.rng, t.a.ring_box, rig.ring, pile, others, t.markers_mm, feasible)
    if spot is None: return {"ok": False, "stop": "no free spot for the ring"}
    if dest is not None and not (all(dist_mm(dest, (k["x_in"], k["y_in"])) >= RING_R_MM + k["r_mm"] + 10.0 for k in kos) and ko.segment_clear(rig.ring, dest, kos, RING_R_MM + 10.0)):
        return {"ok": False, "stop": "--ring-to spot or path too close to a keep-out"}
    if dest is not None and (why := ring_spot_ok(dest, rig.ring, pile, others, t.markers_mm)): return {"ok": False, "stop": f"--ring-to spot refused: {why}"}   # why: 12 Nov the explicit spot skipped these checks and the ring covered marker 2
    t.say({"event": "ring_dest", "from_in": [round(v, 3) for v in rig.ring], "to_in": spot})
    with_trip = None
    try: mv = mover.lift_ring(spot)
    except TripError as e: with_trip = e.trip; mv = {"ok": False, "why": f"trip:{e.trip.reason}"}; t.go_home()
    except SystemExit as e: mv = {"ok": False, "why": str(e)[:160]}; t.go_home()
    t.say({"event": "ring_move", **mv})
    if with_trip: return {"ok": False, "stop": with_trip.reason, "ring": mv}
    if not mv["ok"]: return {"ok": False, "stop": mv["why"], "ring": mv}
    sp = spread_pile(t, pile_c, t.a.spread_box, nudge=t.a.nudge); return {"ok": sp["stop"] is None, "stop": sp["stop"], "ring": mv, "spread": sp}


# ---------------------------------------------------------------- the loop
def loop(t, rounds, max_n=None):
    """tidy all -> reset -> tidy all ... `rounds` tidy rounds with a reset between them; stops on the first stop of any step."""
    out = {"rounds": [], "stop": None}
    for k in range(1, int(rounds) + 1):
        r = t.round(max_n); out["rounds"].append({"round": k, "tidy": {x: r[x] for x in ("picked", "ok", "failed", "stop")}})
        if r["stop"] != "all_tidy": out["stop"] = f"round {k}: {r['stop']}"; break
        if k == rounds: break
        rs = reset_mat(t); out["rounds"][-1]["reset"] = {x: rs.get(x) for x in ("ok", "stop")}
        if not rs["ok"]: out["stop"] = f"reset after round {k}: {rs.get('stop')}"; break
    return out


# ---------------------------------------------------------------- recorders and the CLI
def make_recorder(a, pl, repo, name):
    """A RemoteRecorder to the resident recorder on the Jetson (phase 2, branch demos-phase2), preflighted. SystemExit with the fix when it is missing."""
    ds = load(os.path.join(ROOT, "tools", "demo_session.py"), "demo_session_for_tidy")
    if not hasattr(ds, "RemoteRecorder"): raise SystemExit("--record needs RemoteRecorder (tools/demo_session.py of the demos-phase2 branch); not in this checkout yet")
    lc = load(os.path.join(ROOT, "jetson", "lab_config.py"), "lab_config_for_tidy"); lab = lc.LabConfig.from_file(a.lab_config) if a.lab_config else lc.LabConfig()
    if a.lighting_tag: lab.lighting_tag = a.lighting_tag
    root = (a.demo_root or f"~/so101/outputs/demos/{a.session}").rstrip("/") + "/" + name; rec = ds.RemoteRecorder(ds.ssh_transport(a.demo_host), root, repo, lab, rejects_root=root + "-rejects", vcodec=a.demo_vcodec or "libsvtav1")
    if not rec.alive(): raise SystemExit("--record: the demo recorder does not answer; start it on the Jetson (docs/scripted-demos.md)")
    st = rec.status()
    if not st.get("ok") or st.get("cams_missing") or st.get("busy"): raise SystemExit(f"--record: recorder not ready: {st}")
    pl.PICK_SEQ_ARGS = f"--tick-sink {a.demo_tick_sink}"   # why: pick_seq writes ticks only when recording (the phase-2 hook in PickSession reads PICK_SEQ_ARGS), 2026-11-11
    return rec


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("cmd", choices=("plan", "run", "reset", "loop"))
    ap.add_argument("--record", action="store_true", help="record each ring pick as a LeRobot episode (kept / rejected) via the resident recorder on the Jetson"); ap.add_argument("--record-spread", action="store_true", help="also record the reset's spread picks into a separate dataset (off)")
    ap.add_argument("--max", type=int, default=None, help="at most this many picks per tidy round"); ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--order", choices=("clean", "near"), default="clean")
    ap.add_argument("--grip-check", choices=("own", "loop"), default="own", help="own: gripper readback after the lift > 2.0 OR the wrist camera; loop: closed_loop_pick's own (switch once final-correct is on main)")
    ap.add_argument("--no-roll", action="store_true"); ap.add_argument("--jaw-open", type=int, default=None); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--session", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--temp-limit", type=float, default=65.0, help="servo temperature (C) at which the run stops (plan rows and /api/tele; the servos' own cut is 70, idle after a long day is 52 to 57)"); ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-minutes", type=float, default=None, help="wall-clock cap for the whole session, checked between picks (a clean stop)"); ap.add_argument("--max-tries", type=int, default=None, help="cap on picks attempted (ring and spread), whatever their outcome (a clean stop)")
    ap.add_argument("--require-health", action="store_true", help="a plan whose rows carry no servo telemetry is a trip"); ap.add_argument("--unattended", action="store_true", help="overnight mode: caps default to --max-tries 60 and --max-minutes 240, implies --require-health, refuses to start while the stop file exists, keeps the Mac awake (caffeinate), and after parking at rest lets go of the servo torque if the readback confirms rest")
    ap.add_argument("--ring-box", type=float, nargs=4, default=list(RING_BOX_IN)); ap.add_argument("--spread-box", type=float, nargs=4, default=list(SPREAD_BOX_IN)); ap.add_argument("--no-nudge", dest="nudge", action="store_false"); ap.add_argument("--no-blob-mask", dest="blob_mask", action="store_false")
    ap.add_argument("--ring-to", type=float, nargs=2, default=None, help="reset: set the ring down at this centre (mat inches) instead of a random one")
    ap.add_argument("--demo-host", default="jetson"); ap.add_argument("--demo-root", default=None); ap.add_argument("--demo-vcodec", default=None); ap.add_argument("--demo-tick-sink", default="/dev/shm/so101_ticks.jsonl"); ap.add_argument("--lab-config", default=None)
    ap.add_argument("--lighting-tag", default=None); ap.add_argument("--ring-repo", default=RING_REPO); ap.add_argument("--spread-repo", default=SPREAD_REPO)
    return ap


def plan_report(pl, rig, toys, ring, order):
    verdicts = [rig.check((o["x_in"], o["y_in"])) for o in toys]; rows = []
    for o, v in zip(toys, verdicts):
        kind, aim = toy_kind(o)
        rows.append({"toy_in": [round(o["x_in"], 2), round(o["y_in"], 2)], "kind": kind, "aim_mm": aim, "inside_ring": inside_ring(o, ring), "verdict": "done" if inside_ring(o, ring) else ("ok" if v["ok"] else ("unreachable" if not v["reachable"] else "blocked")),
                     "route": None if v["direct"] is None else ("direct" if v["direct"] else "over at 110 mm"), "lift_in": v["lift_in"], "why": v["why"], "worst": v["worst"]})
    pending = [(o, v) for o, v in zip(toys, verdicts) if not inside_ring(o, ring)]; ordered = order_toys([p[0] for p in pending], [p[1] for p in pending], rig.base, ring, order)
    return {"ring_in": [round(v, 3) for v in ring], "ring_r_mm": RING_R_MM, "ring_h_mm": RING_H_MM, "toys": rows, "order": [[round(o["x_in"], 2), round(o["y_in"], 2)] for o in ordered], "n_outside": len(pending), "n_refused": sum(1 for o, v in pending if not v["ok"])}


def main(argv=None):
    a = build_parser().parse_args(argv); gr = load_gr()
    if a.unattended:   # before anything is loaded or fetched: caps default to 60 tries / 240 min, a leftover stop file refuses the start, the Mac is kept awake
        a.max_tries, a.max_minutes = gr.unattended_gate(a.max_tries, a.max_minutes, os.path.join(ROOT, "outputs", "run", "autoloop.stop"))
        gr.start_caffeinate()
    al = load_al(); pl = load_pl(); cc = pl.cam_calib(); table = json.load(open(os.path.join(ROOT, "jetson", "calib", "top_table.json")))
    if not (table.get("arm_fit") and table.get("arm_base_mm")): raise SystemExit("tidy needs the arm fit in jetson/calib/top_table.json")
    ring = sl.fit_ring_multi(pl.fetch_top, cc, table)
    if ring is None: raise SystemExit("tape ring not found in the top frame (fewer than 150 blue pixels)")
    rig = Rig(pl, cc, table, ring)
    if a.cmd == "plan":
        import cv2
        objs = pl.fetch_scene_objects(); rig.keepouts = ko.find_keepouts(objs, cv2.imdecode(np.frombuffer(pl.fetch_top(), np.uint8), cv2.IMREAD_COLOR), cc, table, ring, RING_R_MM, RING_WALL_IN_MM, RING_WALL_OUT_MM)
        rep = plan_report(pl, rig, [o for o in objs if is_toy(o, ring=ring) and not ko.is_keepout_object(o, rig.keepouts)], ring, a.order); rep["keepouts"] = rig.keepouts; rep["keepout_inside_ring"] = bool(ko.inside_ring(rig.keepouts, ring, RING_R_MM))
        print(json.dumps(rep, indent=1)); return 0
    get = lambda path, t: json.load(urllib.request.urlopen(pl.DASH + path, timeout=t))
    ctx = gr.Ctx(lambda: get("/api/state", gr.Limits().panel_s), lambda: get("/api/scene", 5), time.time, time.sleep, lambda d: print(json.dumps(d), flush=True), arm_jobs=al.jetson_arm_jobs, fetch_tele=lambda: get("/api/tele", 3))
    lim = gr.Limits(ceiling_in=4.6, stop_file=os.path.join(ROOT, "outputs", "run", "autoloop.stop"), temp_limit_c=a.temp_limit, max_tries=a.max_tries, max_minutes=a.max_minutes, require_health=a.require_health or a.unattended)
    guard = gr.Guard(cc, table, lim, ctx); track = pl._load("labtrack").LabTrack(EXPERIMENT)
    eps = {"ring": Episode(make_recorder(a, pl, a.ring_repo, "so101-toys-to-ring") if a.record else None, a.ring_repo), "spread": Episode(make_recorder(a, pl, a.spread_repo, "so101-toys-spread") if a.record_spread else None, a.spread_repo)}
    pl.open_session(True)
    t = None
    try:
        eps['ring'].activate()   # fail before any motion when the recorder will not take its configuration
        t = Tidy(pl, cc, table, rig, guard, ctx, track, a, random.Random(a.seed), eps, out=lambda l: print(l, flush=True))
        try:
            if a.cmd == "run": s = t.round(a.max)
            elif a.cmd == "reset": s = reset_mat(t, tuple(a.ring_to) if a.ring_to else None)
            else: s = loop(t, a.rounds, a.max)
        except KeyboardInterrupt:   # Ctrl-C or SIGTERM: park, then leave
            t.say({"event": "interrupted"}); t.park("interrupted", fresh=True); s = {"stop": "interrupted"}
        except Exception as e:   # a crash must not leave the arm mid-pick; park, then let it propagate
            t.say({"event": "error", "err": f"{type(e).__name__}: {e}"[:300]}); t.park("exception", fresh=True); raise
        if s.get("stop") and s["stop"].split(": ")[-1] not in ("all_tidy", "max") and t.in_motion: t.park(s["stop"])   # any stop that left the arm out of rest (a trip's go_home failed, a cap hit mid-motion)
        print(json.dumps({"event": "summary", **{k: v for k, v in s.items() if k != "results"}}), flush=True)
        return 0 if not s.get("stop") or s["stop"].split(": ")[-1] in CLEAN_ENDS else 2
    finally:
        if a.unattended and t is not None: guard.release_torque(ctx, dict(pl.REST), pl.close_session, gr.ssh_torque_off, t.say)   # after the park, whatever ended the run
        pl.close_session()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # why: kill <pid> takes the same clean exit as Ctrl-C (arm parked)
    sys.exit(main())
