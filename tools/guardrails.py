#!/usr/bin/env python3
"""guardrails.py - the safety checks of the unattended session runner (tools/autoloop.py, docs/autoloop.md).

Pure functions (box_check, fk_mat_mm, plan_tips, motion_check, stopfile_present) and a Guard that ties them to the limits and to the
counters that live across cycles. Every check returns a Trip(reason, detail) or None; a Trip ends the session. Nothing here moves the arm
or talks to the arm: the dashboard is reached through the callables of a Ctx, so a test (or --dry-run) hands in fakes and a fake clock.
"""
import dataclasses, os, time

MAT_MM = (457.2, 304.8)   # the Fiskars mat, 18 x 12 in (docs/mat-scene.md)
IN = 25.4
STOP_FILE = os.path.expanduser("~/so101/outputs/run/autoloop.stop")
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
NO_BOX_CHECK = ("rest",)   # why: rest is folded up behind the mat edge on purpose; the box guards what the tip does over the mat


@dataclasses.dataclass
class Trip:
    reason: str
    detail: dict = dataclasses.field(default_factory=dict)
    clean: bool = False   # a stop the owner asked for (stop file), not a fault

    def as_json(self): return {"reason": self.reason, "detail": self.detail, "clean": self.clean}


@dataclasses.dataclass
class Limits:
    margin_mm: float = 20.0          # tip stays this far inside the mat edge
    ceiling_in: float = 4.0          # and below this height above the mat
    floor_mm: float = -10.0          # and not driven into the mat (the closed loop lowers the grasp by at most 3 mm below the plane)
    max_err_deg: float = 8.0         # pick_seq row: commanded vs reached
    min_markers: int = 3             # overhead camera must see this many ArUco markers ...
    markers_s: float = 10.0          # ... within this long
    holding_s: float = 10.0          # dashboard "holding" (calibration held without a fresh solve) longer than this trips
    lost_after: int = 3              # consecutive cycles without finding the nut
    max_failures: int = 3            # consecutive cycles that did not land the nut
    panel_s: float = 3.0             # /api/state must answer within this
    min_fps: float = 25.0
    poll_s: float = 1.0              # perception polling step
    stop_file: str = STOP_FILE


@dataclasses.dataclass
class Ctx:
    """What the guard may look at. fetch_state / fetch_scene return the dashboard JSON or raise; now / sleep default to the wall clock."""
    fetch_state: object
    fetch_scene: object
    now: object = time.time
    sleep: object = time.sleep
    note: object = print   # one-off notes ("fps not checked"), called with a dict


# ---------------------------------------------------------------- pure functions
def stopfile_present(path): return bool(path) and os.path.exists(path)


def box_check(x_mm, y_mm, h_mm, lim=None):
    """None when the tip (mat mm, height above the mat in mm) is inside the shrunk mat and below the ceiling, else a short reason. Edges are inside."""
    lim = lim or Limits(); m = lim.margin_mm
    if not (m <= x_mm <= MAT_MM[0] - m and m <= y_mm <= MAT_MM[1] - m): return f"tip ({x_mm / IN:.2f}, {y_mm / IN:.2f}) in outside the mat shrunk by {m:g} mm"
    if h_mm > lim.ceiling_in * IN: return f"tip {h_mm / IN:.2f} in above the mat, ceiling {lim.ceiling_in:g} in"
    if h_mm < lim.floor_mm: return f"tip {h_mm:.1f} mm below the mat plane, floor {lim.floor_mm:g} mm"
    return None


def fk_mat_mm(cc, table, joints):
    """Where the tip of COMMANDED joints sits: (mat x mm, mat y mm, height above the mat mm), through the arm fit's zero offsets and tip offset
    (the same chain as pick_loop.fk_mat, which reports inches)."""
    fit = table["arm_fit"]; off = fit.get("joint_offsets_deg") or {}
    tip, _ = cc.fk_tip({k: float(joints[k]) + float(off.get(k, 0.0)) for k in ARM}, tip_offset_mm=fit.get("tip_offset_mm"))
    x, y = cc.arm_to_table(table, float(tip[0]), float(tip[1])); return float(x), float(y), float(tip[2]) - float(fit.get("table_z_mm", 0.0) or 0.0)


def plan_tips(cc, table, plan):
    """[(waypoint name, x mm, y mm, h mm)] for every waypoint that is checked against the box."""
    return [(w["name"], *fk_mat_mm(cc, table, w["joints"])) for w in plan["waypoints"] if w["name"] not in NO_BOX_CHECK]


def motion_check(row, limit):
    """Trip when a pick_seq row's commanded-vs-reached error is above `limit` deg (equal is fine); rows without the field pass."""
    e = row.get("max_err_deg")
    return Trip("motion", {"wp": row.get("wp"), "max_err_deg": e, "limit_deg": limit}) if e is not None and float(e) > limit else None


# ---------------------------------------------------------------- the Guard
class Guard:
    def __init__(self, cc, table, limits=None, ctx=None):
        self.cc, self.table, self.lim, self.ctx = cc, table, limits or Limits(), ctx
        self.lost = self.failures = 0; self.healthy = False
        self._low_since = self._hold_since = None; self._noted = set()

    def _note(self, ctx, key, msg):
        if key not in self._noted: self._noted.add(key); ctx.note({"note": msg})

    def trip(self, reason, detail=None, clean=False): return Trip(reason, detail or {}, clean)   # why: the runner loads this module by path, so it builds trips through the Guard

    # -- between cycles
    def check_stop(self):
        return Trip("stop_file", {"path": self.lim.stop_file}, clean=True) if stopfile_present(self.lim.stop_file) else None

    def poll_state(self, ctx=None):
        """(state dict or None, Trip or None): the STOP-liveness check. The panel must answer within panel_s and report the Jetson up."""
        ctx = ctx or self.ctx; t0 = ctx.now()
        try: st = ctx.fetch_state()
        except Exception as e: self.healthy = False; return None, Trip("panel", {"err": f"{type(e).__name__}: {e}"[:200]})
        dt = ctx.now() - t0
        if dt > self.lim.panel_s: self.healthy = False; return st, Trip("panel", {"answered_s": round(dt, 2), "limit_s": self.lim.panel_s})
        if not isinstance(st, dict) or st.get("error"): self.healthy = False; return None, Trip("panel", {"err": str((st or {}).get("error") if isinstance(st, dict) else st)[:200]})
        if not (st.get("jetson") or {}).get("online"): self.healthy = False; return st, Trip("panel", {"err": "jetson not online"})
        self.healthy = True; return st, None

    def readback_healthy(self, ctx=None):
        """A fresh /api/state poll: is it safe to command the arm to rest? (panel answered, Jetson up, the follower stream is alive)"""
        st, t = self.poll_state(ctx)
        if t: return False
        fol = ((st.get("arms") or {}).get("follower") or {}); self.healthy = bool(fol.get("pos")); return self.healthy

    def _scene_trip(self, sc, now):
        """(Trip or None, keep polling?)"""
        lim = self.lim
        if not isinstance(sc, dict) or not sc.get("ok", True): return Trip("perception", {"err": str((sc or {}).get("err") if isinstance(sc, dict) else sc)[:200]}), False
        if sc.get("calib_ok") is False: return Trip("calib", {"calib_note": sc.get("calib_note")}), False
        low = int(sc.get("n_markers_seen", lim.min_markers)) < lim.min_markers
        self._low_since = (self._low_since if self._low_since is not None else now) if low else None
        self._hold_since = (self._hold_since if self._hold_since is not None else now) if sc.get("holding") else None
        if low and now - self._low_since >= lim.markers_s: return Trip("markers", {"n_markers_seen": sc.get("n_markers_seen"), "seconds": round(now - self._low_since, 1), "limit_s": lim.markers_s}), False
        if self._hold_since is not None and now - self._hold_since > lim.holding_s: return Trip("holding", {"seconds": round(now - self._hold_since, 1), "limit_s": lim.holding_s}), False
        return None, low or self._hold_since is not None

    def check_before_cycle(self, ctx=None):
        ctx = ctx or self.ctx
        if t := self.check_stop(): return t
        st, t = self.poll_state(ctx)
        if t: return t
        fol = (st.get("arms") or {}).get("follower") or {}
        if "overload" in fol or "overload" in st:   # the dashboard does not publish one yet (docs/autoloop.md); honoured the day it does
            ov = fol.get("overload", st.get("overload"))
            if ov: return Trip("overload", {"overload": ov})
        else: self._note(ctx, "overload", "overload not checked: /api/state exposes no overload signal")
        fps = st.get("fps")
        if fps is None: self._note(ctx, "fps", "fps not checked: the dashboard exposes no camera fps")
        else:
            worst = min(fps.values()) if isinstance(fps, dict) and fps else fps
            if isinstance(worst, (int, float)) and worst < self.lim.min_fps: return Trip("fps", {"fps": fps, "limit": self.lim.min_fps})
        self._low_since = self._hold_since = None
        while True:   # poll the scene until it is fine or a threshold is crossed (a bumped marker or a hand over the mat is usually gone in a second)
            try: sc = ctx.fetch_scene()
            except Exception as e: sc = {"ok": False, "err": f"{type(e).__name__}: {e}"}
            trip, waiting = self._scene_trip(sc, ctx.now())
            if trip or not waiting: return trip
            ctx.sleep(self.lim.poll_s)

    # -- inside a cycle
    def check_plan(self, plan):
        """Called before every run_plan: stop file, then the workspace box on every waypoint's FK tip. A plan that fails is never run."""
        if t := self.check_stop(): return t
        for name, x, y, h in plan_tips(self.cc, self.table, plan):
            why = box_check(x, y, h, self.lim)
            if why: return Trip("workspace", {"wp": name, "x_in": round(x / IN, 3), "y_in": round(y / IN, 3), "h_mm": round(h, 1), "why": why})
        return None

    def check_waypoint(self, row): return motion_check(row, self.lim.max_err_deg)

    # -- after a cycle
    def check_after_cycle(self, result):
        """result: {"ok": the cycle landed the nut, "found": the nut was located}. Consecutive counters; nut_lost is checked before failures."""
        self.lost = 0 if result.get("found", True) else self.lost + 1
        self.failures = 0 if result.get("ok") else self.failures + 1
        if self.lost >= self.lim.lost_after: return Trip("nut_lost", {"cycles": self.lost})
        if self.failures >= self.lim.max_failures: return Trip("consecutive_failures", {"failures": self.failures})
        return None
