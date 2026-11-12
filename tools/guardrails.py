#!/usr/bin/env python3
"""guardrails.py - the safety checks of the unattended session runner (tools/autoloop.py, docs/autoloop.md).

Pure functions (box_check, fk_mat_mm, plan_tips, motion_check, stopfile_present) and a Guard that ties them to the limits and to the
counters that live across cycles. Every check returns a Trip(reason, detail) or None; a Trip ends the session. Nothing here moves the arm
or talks to the arm: the dashboard is reached through the callables of a Ctx, so a test (or --dry-run) hands in fakes and a fake clock.
"""
import dataclasses, importlib.util, math, os, time

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
    # -- servo health (jetson/servo_telemetry.py numbers: pick_seq attaches a summary to every waypoint row; /api/tele has the live ones)
    temp_limit_c: float = 65.0       # hot servo: the servos' firmware cut is 70 C on all six (read from the real arm, verify_telemetry 11 Nov 22:30) so 65 keeps 5 C; it is also the red mark (servo_telemetry.TEMP_HOT_C). The old 60 was 3 to 8 C ABOVE the idle temperatures measured after 5 h of picks (52 to 57 C), i.e. it would have stopped every long run. A rise-rate rule is not built: no heating curve has been measured
    temp_start_c: float = 60.0       # an unattended run does not START with a servo at or above this (idle 52 to 57 C measured after a long day): it would have no headroom for the night. Checked once, before the first pick, from /api/tele
    cur_limit_a: float = 2.7         # one ARM joint at its stall current: STS3215 at 12 V stalls at 2.7 A (datasheet figure in docs/stronger-arm.md; servo_telemetry.MODELS). The arm joints' own protection is set at 19.5 A (read from the real arm), so it will never save a stalled joint: this software trip is the only one. The gripper is exempt: it stalls on the object by design. Counts->A uses 6.5 mA/count, unverified on the arm
    bus_limit_a: float = 5.0         # all servos together: the 12 V 5 A supply / Waveshare board rating (servo_telemetry.BUS_A_MARK); Robonine measured 3.28 A peak for this arm under 300 g
    volt_min_v: float = 10.0         # supply sag: below this the 12 V rail is browning out (servo_telemetry.VOLT_LOW_V, lab choice; the servos run 6 to 12.6 V)
    stall_s: float = 3.0             # stall = position error above servo_telemetry.STALL_ERR_DEG with load above STALL_LOAD_PCT for this long (both are assumptions, see there); 3 s is longer than any settle in a plan (<= 0.7 s)
    tele_max_age_s: float = 5.0      # a /api/tele sample older than this is not looked at (while a PickSession idles the stream goes quiet by design)
    require_health: bool = False     # True: a plan that comes back without servo telemetry is a trip (no data is not "healthy"); --unattended turns it on
    # -- the session's own caps (None = off)
    max_tries: int = None            # picks / cycles attempted, whatever their outcome
    max_minutes: float = None        # wall clock from the first cycle; checked between picks, never mid-pick


@dataclasses.dataclass
class Ctx:
    """What the guard may look at. fetch_state / fetch_scene return the dashboard JSON or raise; now / sleep default to the wall clock."""
    fetch_state: object
    fetch_scene: object
    now: object = time.time
    sleep: object = time.sleep
    note: object = print   # one-off notes ("fps not checked"), called with a dict
    arm_jobs: object = None   # () -> list of command lines of other jobs holding the arm (lerobot teleop/record/...), or None when unknown
    fetch_tele: object = None   # () -> the dashboard's /api/tele JSON (live servo health), or None = not looked at between cycles


# ---------------------------------------------------------------- pure functions
_KO = []
def keepout_mod():
    """tools/keepout.py by path (the frogs and anything else that is not a toy), loaded once."""
    if not _KO:
        spec = importlib.util.spec_from_file_location("keepout_for_guardrails", os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepout.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _KO.append(m)
    return _KO[0]


PART_R_MM, TIP_R_MM, KO_MARGIN_MM = 20.0, 8.0, 5.0   # arm body / jaw tip radius and the height margin: the same numbers as tools/tidy.py (the ring check)


def droop_mm(x_in, y_in, base_in):   # tools/tidy.py Rig.droop_mm: the 11 Nov sweep, ~4 mm at 150 mm reach, ~16 at 220, ~24 at 290
    r = math.hypot(x_in - base_in[0], y_in - base_in[1]) * IN; return max(4.0, 4.0 + (r - 150.0) / 140.0 * 20.0)


def body_points(cc, table, joints):
    """(x_in, y_in, h_mm) along wrist -> gripper -> gripper frame, the tip last (the same chain as tidy.Rig.points)."""
    import numpy as np
    fit = table["arm_fit"]; off = fit.get("joint_offsets_deg") or {}; tz = float(fit.get("table_z_mm", 0.0) or 0.0)
    F = cc.fk_frames({k: float(joints[k]) for k in ARM}, off); chain = [F[n][:3, 3] * 1000.0 for n in ("wrist_link", "gripper_link", "gripper_frame_link")]; out = []
    for a, b in zip(chain, chain[1:]):
        for f in np.linspace(0, 1, 5):
            p = a + f * (b - a); X, Y = cc.arm_to_table(table, float(p[0]), float(p[1])); out.append((X / IN, Y / IN, float(p[2]) - tz))
    x, y, h = fk_mat_mm(cc, table, joints); out.append((x / IN, y / IN, h)); return out


def keepout_hits(cc, table, plan, keepouts, steps=8):
    """[(waypoint pair, x_in, y_in, h_mm, clearance_mm)] of arm body points that touch a keep-out cylinder (droop included) on the straight joint-space move
    between consecutive waypoints (and at the first one). Empty when no keep-out."""
    if not keepouts: return []
    ko = keepout_mod(); base = tuple(v / IN for v in cc.arm_to_table(table, 0.0, 0.0)); wps = plan["waypoints"]; hits = []
    for i, w in enumerate(wps):
        a = w["joints"]; b = wps[i - 1]["joints"] if i else None
        for f in ([1.0] if b is None else [k / steps for k in range(1, steps + 1)]):
            j = a if b is None else {k: float(b[k]) + f * (float(a[k]) - float(b[k])) for k in ARM}; pts = body_points(cc, table, j); tx, ty, th = pts[-1]
            for x, y, h in pts:
                at_tip = math.dist((x * IN, y * IN, h), (tx * IN, ty * IN, th)) < 15.0
                for k in ko.point_hits(x, y, h, TIP_R_MM if at_tip else PART_R_MM, keepouts, droop_mm(x, y, base), KO_MARGIN_MM):
                    hits.append((f"{wps[i - 1]['name'] if i else 'start'}>{w['name']}", round(x, 2), round(y, 2), round(h, 1), round(ko.dist_mm((x, y), (k["x_in"], k["y_in"])) - k["r_mm"], 1)))
    return hits


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


# ---------------------------------------------------------------- servo health
TELE_STATUS_BITS = {0: "voltage", 1: "sensor", 2: "temp", 3: "current", 5: "overload"}   # Feetech Status register (docs/telemetry.md); bit 4 is undocumented and ignored
TELE_KEYS = ("temp", "load", "cur", "volt", "pos", "goal", "sag", "vel", "on", "moving", "status")   # per-joint cell order of a /api/tele row (dashboard/telemetry_store.KEYS)


def servo_health_trip(s, lim=None):
    """Trip for a servo-telemetry summary (Summary.out() shape: {"j": {joint: {temp_max_c, load_abs_max_pct, cur_max_a, stall_s_max}}, "total_a_max", "volt_min", "flags"}) or None.
    Order: the servos' own error bits, heat, supply voltage, bus current, one joint at stall current, a stall. The gripper is checked for heat only (it holds objects at its torque limit on purpose)."""
    lim = lim or Limits()
    if not isinstance(s, dict): return None
    j = s.get("j") or {}; flags = s.get("flags") or {}
    errs = sorted(f for f in (flags if isinstance(flags, (dict, list)) else []) if not str(f).startswith("gripper:") and not str(f).endswith(":bit4"))   # bit 4 is undocumented
    if errs: return Trip("servo_error", {"flags": errs[:6]})
    for n, e in j.items():
        t = e.get("temp_max_c")
        if t is not None and t >= lim.temp_limit_c: return Trip("servo_hot", {"joint": n, "temp_c": t, "limit_c": lim.temp_limit_c})
    v = s.get("volt_min")
    if v is not None and v < lim.volt_min_v: return Trip("servo_voltage", {"volt_min_v": v, "limit_v": lim.volt_min_v})
    a = s.get("total_a_max")
    if a is not None and a > lim.bus_limit_a: return Trip("servo_current", {"total_a": a, "limit_a": lim.bus_limit_a, "scope": "bus"})
    for n, e in j.items():
        if n == "gripper": continue
        if (e.get("cur_max_a") or 0.0) >= lim.cur_limit_a: return Trip("servo_current", {"joint": n, "cur_a": e["cur_max_a"], "limit_a": lim.cur_limit_a, "scope": "joint"})
        if (e.get("stall_s_max") or 0.0) >= lim.stall_s: return Trip("servo_stall", {"joint": n, "stall_s": round(e["stall_s_max"], 2), "limit_s": lim.stall_s, "load_pct": e.get("load_abs_max_pct")})
    return None


def summary_of_tele_api(api, lim=None):
    """The newest row of the dashboard's /api/tele as a one-sample summary, or None when there is none or it is older than lim.tele_max_age_s."""
    lim = lim or Limits()
    if not isinstance(api, dict) or not api.get("rows"): return None
    age = api.get("age")
    if age is None or age > lim.tele_max_age_s: return None
    row = api["rows"][-1]; joints = api.get("joints") or []; keys = api.get("keys") or list(TELE_KEYS); ix = {k: i for i, k in enumerate(keys)}
    j = {}; flags = []
    for n, cells in zip(joints, row[3]):
        if not cells: continue
        g = lambda k: cells[ix[k]] if k in ix and ix[k] < len(cells) else None
        j[n] = {"temp_max_c": g("temp"), "load_abs_max_pct": abs(g("load") or 0.0), "cur_max_a": g("cur") or 0.0, "stall_s_max": 0.0}   # a live sample has no duration: stalls come from the plan rows
        st = g("status") or 0
        flags += [f"{n}:{name}" for bit, name in TELE_STATUS_BITS.items() if st & (1 << bit)]
    return {"j": j, "total_a_max": row[1], "volt_min": row[2], "flags": flags}


def unattended_problems(lim):
    """What an unattended (overnight) start refuses to run without: both session caps. [] when fine."""
    return [m for m, ok in (("--max-minutes (a wall-clock cap)", lim.max_minutes), ("--max-tries (a cap on picks)", lim.max_tries)) if not ok]


def park_on_exit(guard, ctx, send_rest, in_motion, reason):
    """The one place that decides whether to send the arm to rest when a session ends (any reason, including an exception or SIGTERM of the runner).
    -> {"parked": bool, "why": str}. No motion command when: the arm is already at rest (not in_motion); the owner stopped the arm or the link is dead (reason run_plan);
    the dashboard does not read back a live follower. Torque policy: pick_seq disconnects with disable_torque_on_disconnect=False, so torque stays ON at rest. Never raises."""
    if not in_motion: return {"parked": False, "why": "already at rest"}
    if reason == "run_plan": return {"parked": False, "why": "arm stopped or link dead: no further motion command"}
    try:
        if not guard.readback_healthy(ctx): return {"parked": False, "why": "dashboard does not read back a live follower"}
        send_rest(); return {"parked": True, "why": reason}
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, GeneratorExit)): raise
        return {"parked": False, "why": f"rest failed: {type(e).__name__}: {e}"[:200]}


# ---------------------------------------------------------------- the Guard
class Guard:
    def __init__(self, cc, table, limits=None, ctx=None):
        self.cc, self.table, self.lim, self.ctx = cc, table, limits or Limits(), ctx
        self.lost = self.failures = 0; self.healthy = False; self.tries = 0; self.t0 = None; self.keepouts = []   # keepouts: set by the runner every cycle (tools/keepout.py find_keepouts)
        self._low_since = self._hold_since = None; self._noted = set()

    def _note(self, ctx, key, msg):
        if key not in self._noted: self._noted.add(key); ctx.note({"note": msg})

    def keepout_mod(self): return keepout_mod()

    def trip(self, reason, detail=None, clean=False): return Trip(reason, detail or {}, clean)   # why: the runner loads this module by path, so it builds trips through the Guard

    # -- between cycles
    def check_stop(self):
        return Trip("stop_file", {"path": self.lim.stop_file}, clean=True) if stopfile_present(self.lim.stop_file) else None

    def count_try(self): self.tries += 1   # a pick that did not go through check_after_cycle (tidy's spread picks)

    def check_budget(self, ctx):
        """The session caps, checked between picks: tries first, then wall clock from the first call. Clean stops (the owner set them), not faults."""
        now = ctx.now()
        if self.t0 is None: self.t0 = now
        lim = self.lim
        if lim.max_tries and self.tries >= lim.max_tries: return Trip("max_tries", {"tries": self.tries, "limit": lim.max_tries}, clean=True)
        if lim.max_minutes and now - self.t0 >= lim.max_minutes * 60: return Trip("max_minutes", {"minutes": round((now - self.t0) / 60, 1), "limit": lim.max_minutes}, clean=True)
        return None

    def check_tele(self, ctx):
        """Live servo health from /api/tele between picks (heat, current, supply, error bits at rest). Between picks of a running session the stream is quiet by design
        (the plan rows are the source then), so a missing sample is only noted. FAIL CLOSED before the first pick of a run with require_health: no fresh sample (feed down,
        Jetson on old files) is a trip, and a servo already at temp_start_c is a trip."""
        if not ctx.fetch_tele: return None
        first = self.tries == 0 and self.lim.require_health
        try: api = ctx.fetch_tele()
        except Exception as e:
            self._note(ctx, "tele", f"live servo health not checked: {type(e).__name__}: {e}")
            return Trip("no_servo_telemetry", {"err": f"{type(e).__name__}: {e}"[:160]}) if first else None
        s = summary_of_tele_api(api, self.lim)
        if s is None:
            self._note(ctx, "tele_stale", "live servo health not checked: /api/tele has no fresh sample (plan rows are the source)")
            return Trip("no_servo_telemetry", {"why": f"no sample in /api/tele younger than {self.lim.tele_max_age_s:g} s before the first pick"}) if first else None
        if t := servo_health_trip(s, self.lim): return t
        if first:
            hot = max(((e.get("temp_max_c") or 0, n) for n, e in s["j"].items()), default=(0, None))
            if hot[0] >= self.lim.temp_start_c: return Trip("servo_hot_start", {"joint": hot[1], "temp_c": hot[0], "start_limit_c": self.lim.temp_start_c})
        return None

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
        if t := self.check_budget(ctx): return t
        st, t = self.poll_state(ctx)
        if t: return t
        if t := self.check_tele(ctx): return t
        if ctx.arm_jobs:   # why: 11 Nov first10 cycle 4: a lerobot-teleoperate started mid-session and only pick_seq's own refusal stopped the pick
            try: jobs = ctx.arm_jobs()
            except Exception as e: jobs = None; self._note(ctx, "arm_jobs", f"arm jobs not checked: {type(e).__name__}: {e}")
            if jobs: return Trip("arm_busy", {"jobs": [j[:160] for j in jobs][:3]})
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
        if self.keepouts and (hits := keepout_hits(self.cc, self.table, plan, self.keepouts)):
            return Trip("keepout", {"at": hits[0][0], "x_in": hits[0][1], "y_in": hits[0][2], "h_mm": hits[0][3], "clearance_mm": hits[0][4], "n": len(hits)})   # a plan that passes a keep-out is never run
        return None

    def check_waypoint(self, row):
        """One pick_seq row: the commanded-vs-reached error, then the servo health summary pick_seq attached to it (heat, current, supply, error bits, stall)."""
        return motion_check(row, self.lim.max_err_deg) or (servo_health_trip(row["tele"], self.lim) if row.get("tele") else None)

    def check_rows(self, rows):
        """All rows of a finished plan. With lim.require_health, a plan whose rows carry no telemetry at all is a trip: an unattended run does not fly blind."""
        for r in rows:
            if t := self.check_waypoint(r): return t
        if self.lim.require_health and rows and not any(r.get("tele") for r in rows): return Trip("no_servo_telemetry", {"rows": len(rows)})
        return None

    # -- the end
    def park(self, send_rest, in_motion, reason, ctx=None): return park_on_exit(self, ctx or self.ctx, send_rest, in_motion, reason)   # why: runners load this module by path, so they reach park_on_exit through the Guard

    # -- after a cycle
    def check_after_cycle(self, result):
        """result: {"ok": the cycle landed the nut, "found": the nut was located}. Consecutive counters; nut_lost is checked before failures."""
        self.tries += 1; self.lost = 0 if result.get("found", True) else self.lost + 1
        self.failures = 0 if result.get("ok") else self.failures + 1
        if self.lost >= self.lim.lost_after: return Trip("nut_lost", {"cycles": self.lost})
        if self.failures >= self.lim.max_failures: return Trip("consecutive_failures", {"failures": self.failures})
        return None
