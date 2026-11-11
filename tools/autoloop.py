#!/usr/bin/env python3
"""autoloop.py - the session runner of "the arm that learns on its own" (docs/autoloop.md): pick the nut to a target, judge the
outcome from the overhead camera, put the nut somewhere new, repeat, with tools/guardrails.py between every step.

  tools/autoloop.py --cycles 10 [--minutes 60] [--target 9 6 | --random-target] [--safe-box X0 Y0 X1 Y1] [--radius-mm 15] [--session NAME] [--dry-run]

One cycle: guard.before -> TRY (closed-loop pick to the target, tools/pick_loop.py) -> JUDGE (/api/scene after a short settle) -> RESET
(a closed-loop pick to a spot sampled with the box rules; never pick_loop.reset_nut, which writes the pick state file) -> guard.after. One JSON line per cycle,
one MLflow run per cycle in `so101-autoloop`, a parent run and outputs/run/autoloop_<session>.json for the session. Any trip: one JSON line,
the session ends, the arm goes to rest only if the dashboard still reads back. --dry-run runs the whole loop on a fake arm and a fake dashboard.
"""
import argparse, contextlib, importlib.util, json, math, os, random, statistics, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = "so101-autoloop"
SAFE_BOX_IN = (6.0, 4.0, 13.0, 8.0)   # default safe box in mat inches, the box the owner approved (plan doc 42, prep doc 44); same as pick_loop.SAFE_BOX
IN = 25.4


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def load_guardrails(): return load(os.path.join(ROOT, "tools", "guardrails.py"), "guardrails_for_autoloop")


def load_pick_loop(): return load(os.path.join(ROOT, "tools", "pick_loop.py"), "pick_loop_for_autoloop")


class TripError(Exception):
    """Raised inside a (guarded) run_plan so closed_loop_pick unwinds at once; carries the Trip."""
    def __init__(self, trip): super().__init__(trip.reason); self.trip = trip


# ---------------------------------------------------------------- spots for the nut
def dist_mm(a_in, b_in): return math.hypot(a_in[0] - b_in[0], a_in[1] - b_in[1]) * IN


def sample_spot(rng, box_in, others_in, markers_mm, cur_in=None, min_obj=40.0, min_marker=50.0, min_cur=60.0, tries=500):
    """A random mat point (inches) inside the safe box, >= min_obj mm from every other object, >= min_marker mm from every marker centre and
    >= min_cur mm from `cur_in`; None when `tries` draws find none."""
    x0, y0, x1, y1 = box_in
    for _ in range(tries):
        p = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
        if any(dist_mm(p, o) < min_obj for o in others_in) or any(dist_mm(p, [m[0] / IN, m[1] / IN]) < min_marker for m in markers_mm): continue
        if cur_in is not None and dist_mm(p, cur_in) < min_cur: continue
        return [round(p[0], 3), round(p[1], 3)]
    return None


def nearest_object(objs, near_in):
    """The scene object closest to `near_in` (inches) -> (object, distance mm), or (None, None)."""
    c = [(dist_mm([o["x_in"], o["y_in"]], near_in), i, o) for i, o in enumerate(objs)]
    return (min(c)[2], min(c)[0]) if c else (None, None)


# ---------------------------------------------------------------- the runner
ARM_JOB_RE = "lerobot-(teleoperate|record|rollout|calibrate|replay)"


def jetson_arm_jobs(host="jetson", timeout=8):
    """Command lines of LeRobot jobs holding the arm on the Jetson (same pattern pick_seq refuses on); [] when none. Raises when ssh fails."""
    import subprocess
    r = subprocess.run(["ssh", "-o", f"ConnectTimeout={timeout}", host, f"pgrep -af '{ARM_JOB_RE}' | grep -v pgrep || true"], capture_output=True, text=True, timeout=timeout + 5)
    if r.returncode not in (0, 1): raise RuntimeError(f"ssh rc {r.returncode}: {r.stderr.strip()[:120]}")
    return [l for l in r.stdout.splitlines() if l.strip()]


def load_biases(path):
    """The pick loop's learned per-waypoint biases from its state file, or {} (missing or unreadable). Read only."""
    try:
        with open(path) as f: b = json.load(f).get("biases") or {}
        return b if isinstance(b, dict) else {}
    except (OSError, ValueError): return {}


class Runner:
    """One session. `pl` is the pick_loop module, `guard` a Guard, `ctx` its Ctx (fetch_state / fetch_scene / now / sleep), `track` a LabTrack,
    `rng` a random.Random. Everything that moves the arm goes through pl.run_plan, wrapped so the guard sees every plan and every row."""
    def __init__(self, pl, cc, table, guard, ctx, cfg, track, rng, out=print):
        self.pl, self.cc, self.table, self.guard, self.ctx, self.cfg, self.track, self.rng, self.out = pl, cc, table, guard, ctx, cfg, track, rng, out
        self.in_motion = False; self.cycles = []; self.stops = []; self.stages = []; self.cur_in = None
        self.markers_mm = list((table.get("layout_markers") or {}).values())
        self.pick_args = argparse.Namespace(jaw_open=pl.JAW_OPEN, max_deg_s=pl.SPEED["closed"], wrist_check=False)
        self.biases = load_biases(os.path.join(getattr(pl, "ROOT", "."), "jetson", "calib", "pick_state.json"))   # read only: the pick loop's learned sag biases; the runner never writes that file

    def say(self, d): self.out(json.dumps(d))

    # -- the world, through the dashboard
    def scene(self):
        try: sc = self.ctx.fetch_scene()
        except Exception: return None
        return sc if isinstance(sc, dict) and sc.get("ok", True) else None

    def locate(self, last_in):
        """The nut as (mat_in, pixel, area) or None: pick_loop.find_nut_anywhere when it exists, else the overhead search near the last spot and
        the target, else the scene object nearest the last spot."""
        pl, cc, tab = self.pl, self.cc, self.table; sc = self.scene(); objs = (sc or {}).get("objects") or []
        try:
            fn = getattr(pl, "find_nut_anywhere", None)
            if fn:   # why: pick_loop's finder takes near_in and returns (xy_in, source); the runner works in find_nut's (xy_in, px, area)
                got = fn(cc, tab, scene_objects=objs, near_in=last_in)
                return (list(got[0]), None, None) if got else None
            near = last_in or list(self.cfg.target)
            return pl.find_nut(cc, tab, near) or pl.find_nut(cc, tab, list(self.cfg.target), radius_px=140) or self.from_scene(objs, near)
        except (Exception, SystemExit): return None

    @staticmethod
    def from_scene(objs, near_in):
        o, d = nearest_object([o for o in objs if max(o.get("w_mm", 0), o.get("h_mm", 0)) <= 40], near_in)   # why: a nut is ~12 mm; a hand or a tool is not the nut
        return ([o["x_in"], o["y_in"]], None, None) if o else None

    # -- the arm, guarded
    @contextlib.contextmanager
    def guarded(self):
        pl, orig = self.pl, self.pl.run_plan; self.orig_run = orig
        def run(plan, max_deg_s):
            if t := self.guard.check_plan(plan): raise TripError(t)
            self.in_motion = True
            try: rows = orig(plan, max_deg_s)
            except (Exception, SystemExit) as e: raise TripError(self.guard.trip("run_plan", {"err": f"{type(e).__name__}: {e}"[:300]})) from e   # why: so101-stop kills pick_seq, so the ssh run returns no rows; that is the owner stopping the arm, not a failed try to repeat
            for r in rows:
                if t := self.guard.check_waypoint(r): raise TripError(t)
            return rows
        pl.run_plan = run
        try: yield
        finally: pl.run_plan = orig

    def pick(self, nut_in, place_in, n):
        _, rows, info = self.pl.closed_loop_pick(self.cc, self.table, list(nut_in), list(place_in), self.biases, self.pick_args, n, self.stages.append)
        self.in_motion = False; return rows, info

    def judge(self, nut_before_in, target_in, info):
        """After a short settle: where the nut is now (the scene object nearest the target) and what that means."""
        self.ctx.sleep(self.cfg.settle_s); sc = self.scene(); objs = (sc or {}).get("objects") or []
        o, _ = nearest_object(objs, target_in)
        if o is None: return {"after_in": None, "moved_mm": None, "picked": False, "err_mm": None, "landed": False, "objects": objs}
        after = [o["x_in"], o["y_in"]]; moved = dist_mm(after, nut_before_in); err = dist_mm(after, target_in)
        jaws = [v for v in (info.get("in_jaws"), o.get("in_jaws")) if v is not None]
        picked = moved > self.cfg.moved_mm and all(jaws)
        return {"after_in": after, "moved_mm": round(moved, 1), "picked": picked, "err_mm": round(err, 1), "landed": picked and err <= self.cfg.radius_mm, "objects": objs, "obj": o}

    def reset(self, after_in, objs, nut_obj):
        """Put the nut on a new spot; returns {"spot_in", "error"}."""
        others = [[o["x_in"], o["y_in"]] for o in objs if o is not nut_obj]
        spot = sample_spot(self.rng, self.cfg.safe_box, others, self.markers_mm, after_in, *self.cfg.spot_mm)
        if spot is None: return {"spot_in": None, "error": "no free spot in the safe box"}
        try:   # why: never pl.reset_nut here: it saves its history over the pick loop's own state file and samples its own spot, so spot_in would lie
            nut = self.locate(after_in)
            if not nut: return {"spot_in": spot, "error": "nut not found for the reset"}
            self.pick(nut[0], spot, 0)
        except (Exception, SystemExit) as e: return {"spot_in": spot, "error": f"{type(e).__name__}: {e}"[:200]}
        return {"spot_in": spot, "error": None}

    def target_for(self, objs):
        if not self.cfg.random_target: return list(self.cfg.target)
        others = [[o["x_in"], o["y_in"]] for o in objs]
        return sample_spot(self.rng, self.cfg.safe_box, others, self.markers_mm, self.cur_in, *self.cfg.spot_mm) or list(self.cfg.target)

    def cycle(self, n):
        """One TRY / JUDGE / RESET; returns (record, Trip or None). A TripError from the guarded run_plan ends the cycle where it stands."""
        t0 = self.ctx.now(); rec = {"event": "cycle", "cycle": n, "found": True, "picked": False, "landed": False, "ok": False, "landed_err_mm": None, "retries": 0, "target_in": None}
        trip = None
        try:
            with self.guarded():
                sc = self.scene(); rec["target_in"] = tgt = self.target_for((sc or {}).get("objects") or []); last = self.cur_in
                for attempt in range(self.cfg.max_retries + 1):
                    nut = self.locate(last)
                    if not nut: rec["found"] = False; break
                    rec["nut_in"] = [round(v, 3) for v in nut[0]]; last = nut[0]
                    try: rows, info = self.pick(nut[0], tgt, n)
                    except SystemExit as e: rec["error"] = str(e)[:200]; break   # why: pick_loop raises SystemExit when ik fails; that is a failed try, not a dead runner
                    j = self.judge(nut[0], tgt, info); rec.update(picked=j["picked"], landed=j["landed"], landed_err_mm=j["err_mm"], moved_mm=j["moved_mm"])
                    if j["after_in"]: self.cur_in = last = j["after_in"]
                    if j["picked"] or attempt == self.cfg.max_retries: break
                    rec["retries"] += 1   # not picked: the nut has not moved, so one more try is cheap; a nut that was picked and dropped is not retried
                rec["ok"] = rec["picked"] and rec["landed"]
                if rec["found"] and rec["landed_err_mm"] is not None and "error" not in rec:
                    r = self.reset(self.cur_in, j["objects"], j.get("obj")); rec["reset_to_in"] = r["spot_in"]
                    if r["error"]: rec["reset_error"] = r["error"]
                    elif r["spot_in"]: self.cur_in = r["spot_in"]
        except TripError as e: trip = e.trip
        rec["cycle_s"] = round(self.ctx.now() - t0, 2)
        return rec, trip or self.guard.check_after_cycle(rec)

    # -- the session
    def finish_cycle(self, rec, trip, parent_id):
        rec["trip"] = trip.reason if trip else None; self.cycles.append(rec); tgt = rec["target_in"]
        with self.track.run(f"autoloop-{self.cfg.session}-c{rec['cycle']}", tags={"host": "mac", "session": self.cfg.session, "cycle": rec["cycle"], "trip": rec["trip"] or "none", **({"mlflow.parentRunId": parent_id} if parent_id else {})}) as tr:
            tr.params({"session": self.cfg.session, "cycle": rec["cycle"], "target": f"{tgt[0]:.2f},{tgt[1]:.2f}" if tgt else None, "mode": "random" if self.cfg.random_target else "fixed", "radius_mm": self.cfg.radius_mm, "dry_run": int(self.cfg.dry_run)})
            tr.metrics({"landed_err_mm": rec["landed_err_mm"], "picked": int(rec["picked"]), "landed": int(rec["landed"]), "cycle_s": rec["cycle_s"], "retries": rec["retries"], "trip": int(trip is not None)})
            tr.json({"cycle": rec, "stages": self.stages}, "cycle.json"); self.stages.clear()
        self.say(rec)

    def stop(self, trip, n):
        """Print the trip line; go to rest only if the arm was mid-motion AND the dashboard still reads back (a fresh /api/state poll)."""
        rest = False
        if self.in_motion and trip.reason != "run_plan" and self.guard.readback_healthy(self.ctx):   # why: after a dead run_plan (so101-stop, ssh down) no further motion command is sent
            try: self.orig_run(self.pl.build_plan(None, None, {}, names=("rest",)), self.pick_args.max_deg_s); rest = True
            except (Exception, SystemExit): rest = False
        self.stops.append({**trip.as_json(), "cycle": n}); self.say({"event": "stop" if trip.clean else "trip", "reason": trip.reason, "detail": trip.detail, "cycle": n, "rest": rest})

    def summary(self, end, t0):
        errs = [c["landed_err_mm"] for c in self.cycles if c["landed_err_mm"] is not None]; n = len(self.cycles); ok = sum(1 for c in self.cycles if c["ok"])
        return {"session": self.cfg.session, "cycles": n, "landed": ok, "picked": sum(1 for c in self.cycles if c["picked"]), "success_rate": round(ok / n, 3) if n else None,
                "mean_err_mm": round(statistics.mean(errs), 1) if errs else None, "median_err_mm": round(statistics.median(errs), 1) if errs else None, "retries": sum(c["retries"] for c in self.cycles),
                "stops": self.stops, "end": end, "duration_s": round(self.ctx.now() - t0, 1), "radius_mm": self.cfg.radius_mm, "dry_run": self.cfg.dry_run}

    def run(self):
        cfg, t0, end = self.cfg, self.ctx.now(), "cycles"
        with self.track.run(f"autoloop-{cfg.session}", tags={"host": "mac", "session": cfg.session, "kind": "session"}) as parent:
            parent.params({"session": cfg.session, "cycles": cfg.cycles, "minutes": cfg.minutes, "mode": "random" if cfg.random_target else "fixed", "target": f"{cfg.target[0]:g},{cfg.target[1]:g}", "safe_box": ",".join(f"{v:g}" for v in cfg.safe_box),
                           "radius_mm": cfg.radius_mm, "max_retries": cfg.max_retries, "dry_run": int(cfg.dry_run)})
            try:
                for n in range(1, cfg.cycles + 1):
                    if cfg.minutes and self.ctx.now() - t0 >= cfg.minutes * 60: end = "minutes"; break
                    trip = self.guard.check_before_cycle(self.ctx)
                    if trip: self.stop(trip, n); end = trip.reason; break
                    rec, trip = self.cycle(n); self.finish_cycle(rec, trip, getattr(parent, "id", None))
                    if trip: self.stop(trip, n); end = trip.reason; break
            except KeyboardInterrupt: end = "interrupted"; self.say({"event": "interrupted"})
            s = self.summary(end, t0); out = os.path.join(cfg.out_dir, f"autoloop_{cfg.session}.json"); os.makedirs(cfg.out_dir, exist_ok=True)
            with open(out, "w") as f: json.dump(s, f, indent=1)
            parent.metrics({"cycles": s["cycles"], "success_rate": s["success_rate"], "mean_err_mm": s["mean_err_mm"], "median_err_mm": s["median_err_mm"], "duration_s": s["duration_s"], "trips": sum(1 for x in self.stops if not x["clean"])})
            parent.json(s, "summary.json")
        self.say({"event": "summary", **s}); return s


# ---------------------------------------------------------------- fakes for --dry-run (and tests/tools/autohelpers.py)
class FakeWorld:
    """The dashboard and the mat, in memory: /api/state, /api/scene and a nut that a plan with a `place` waypoint moves. `script[i]` is the
    outcome of the i-th pick ("ok" | "miss" | "drop"; default ok); `miss_p` makes random misses from `rng`. Knobs: markers, calib_ok, holding,
    jetson_up, extra (other objects, mm), in_jaws."""
    def __init__(self, cc, table, fk, nut_in=(9.0, 6.0), rng=None, script=(), miss_p=0.0, land_noise_mm=3.0):
        self.cc, self.table, self.fk, self.rng, self.script, self.miss_p, self.noise = cc, table, fk, rng or random.Random(0), list(script), miss_p, land_noise_mm
        self.nut = [nut_in[0] * IN, nut_in[1] * IN]; self.picks = 0; self.markers = 4; self.calib_ok = True; self.holding = False; self.jetson_up = True; self.extra = []; self.in_jaws = None
        self.hidden = False; self.state_extra = {}; self.follower = {"pos": {"shoulder_pan": 0.0}, "src": "bus"}

    def api_state(self): return {"jetson": {"online": self.jetson_up, "last": time.time()}, "arms": {"follower": self.follower}, "procs": [], **self.state_extra}

    def api_scene(self):
        objs = ([] if self.hidden else [self.obj(self.nut)]) + [self.obj(e) for e in self.extra]
        return {"ok": True, "objects": objs, "n_markers_seen": self.markers, "calib_ok": self.calib_ok, "holding": self.holding, "age_s": 0.2}

    def obj(self, mm):
        o = {"id": 1, "x_mm": round(mm[0], 1), "y_mm": round(mm[1], 1), "x_in": round(mm[0] / IN, 3), "y_in": round(mm[1] / IN, 3), "w_mm": 12.0, "h_mm": 10.0}
        if self.in_jaws is not None: o["in_jaws"] = self.in_jaws
        return o

    def find_nut(self, cc, table, near_in, radius_px=70, jpeg=None):
        return None if self.hidden else ([self.nut[0] / IN, self.nut[1] / IN], [320, 180], 40)

    def on_plan(self, plan):
        """A plan with `place` is a pick: the nut moves to the FK of the place waypoint when the grasp was within 15 mm of it."""
        names = {w["name"]: w for w in plan["waypoints"]}
        if "grasp" in names and {"close", "close1", "close2"} & names.keys():   # why: with the jaw check the pick is two plans (grasp..lift, then carry..rest)
            self.grasp_at = self.fk(self.cc, self.table, names["grasp"]["desired"])[:2]
        if "place" not in names: return
        g = getattr(self, "grasp_at", None); self.grasp_at = None
        if g is None: return
        gx, gy = g; px, py, _ = self.fk(self.cc, self.table, names["place"]["desired"])
        want = self.script[self.picks] if self.picks < len(self.script) else ("miss" if self.rng.random() < self.miss_p else "ok"); self.picks += 1
        if math.hypot(gx - self.nut[0], gy - self.nut[1]) > 15.0: want = "miss"
        if want == "ok": self.nut = [px + self.rng.gauss(0, self.noise / 2), py + self.rng.gauss(0, self.noise / 2)]
        elif want == "drop": self.nut = [px + 45.0, py + 30.0]


class DryArm:
    """pick_loop's api / run_plan on the real kinematics (damped Gauss-Newton on fk_tip): every arm waypoint is reached `shortfall_mm` short
    along the base->tip direction, which is what the closed loop exists to correct. Gripper-only waypoints and rest are reached exactly."""
    def __init__(self, cc, table, world, shortfall_mm=6.0):
        self.cc, self.table, self.world, self.short = cc, table, world, shortfall_mm
        self.ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"); self.seed = {"shoulder_pan": 15.0, "shoulder_lift": 17.0, "elbow_flex": 2.0, "wrist_flex": 80.0, "wrist_roll": 0.46}

    def tip(self, q):
        fit = self.table["arm_fit"]; off = fit.get("joint_offsets_deg") or {}
        return self.cc.fk_tip({k: float(q[k]) + float(off.get(k, 0.0)) for k in self.ARM}, tip_offset_mm=fit.get("tip_offset_mm"))[0]

    def solve(self, seed, want):
        import numpy as np
        q = {k: float(seed[k]) for k in self.ARM}; names = self.ARM[:4]
        for _ in range(10):
            t = self.tip(q); e = np.asarray(want) - t
            if np.linalg.norm(e) < 1e-4: break
            J = np.zeros((3, 4))
            for i, k in enumerate(names): q2 = dict(q); q2[k] += 0.01; J[:, i] = (self.tip(q2) - t) / 0.01
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
            for i, k in enumerate(names): q[k] += float(dq[i])
        return q

    def api(self, path):
        import numpy as np
        q = dict(p.split("=") for p in path.split("?")[1].split("&")); x, y, h = float(q["x_in"]), float(q["y_in"]), float(q.get("h_in", 0.0))
        ax, ay = self.cc.table_to_arm(self.table, x * IN, y * IN); j = self.solve(self.seed, np.array([ax, ay, float(self.table["arm_fit"]["table_z_mm"]) + h * IN]))
        j["wrist_roll"] = float(self.table["arm_fit"].get("wrist_roll_deg", 0.0)); return {"ok": True, "joints_deg": {k: round(v, 3) for k, v in j.items()}}

    def run_plan(self, plan, max_deg_s):
        import numpy as np
        rows = []
        for i, w in enumerate(plan["waypoints"]):
            j = {k: float(w["joints"][k]) for k in self.ARM}
            if w["gripper_only"] or w["name"] == "rest": got = j
            else:
                t = self.tip(j); d = t[:2] - np.array([38.8353, 0.0]); d = d / np.linalg.norm(d); want = t.copy(); want[:2] -= self.short * d; got = self.solve(j, want)
            rows.append({"wp": w["name"], "seconds": 1.0, "t": i, "target": {**j, "gripper": w["gripper"]}, "reached": {**{k: round(v, 3) for k, v in got.items()}, "gripper": w["gripper"]}, "max_err_deg": round(max(abs(j[k] - got[k]) for k in self.ARM), 2)})
        self.world.on_plan(plan); return rows


# ---------------------------------------------------------------- command line
def make_cfg(a, out_dir):
    box = tuple(a.safe_box) if a.safe_box else SAFE_BOX_IN
    return argparse.Namespace(cycles=a.cycles, minutes=a.minutes, target=tuple(a.target), random_target=a.random_target, safe_box=box, radius_mm=a.radius_mm, session=a.session or time.strftime("%Y%m%d-%H%M%S"),
                              dry_run=a.dry_run, max_retries=a.max_retries, settle_s=a.settle, moved_mm=20.0, spot_mm=(40.0, 50.0, 60.0), out_dir=out_dir)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=10); ap.add_argument("--minutes", type=float, default=None, help="stop starting new cycles after this long")
    g = ap.add_mutually_exclusive_group(); g.add_argument("--target", type=float, nargs=2, default=[9.0, 6.0], metavar=("X", "Y")); g.add_argument("--random-target", action="store_true", help="a fresh target each cycle, sampled with the reset rules")
    ap.add_argument("--safe-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help=f"mat inches where resets and random targets go (default {' '.join(f'{v:g}' for v in SAFE_BOX_IN)})")
    ap.add_argument("--radius-mm", type=float, default=15.0, help="landed = nut within this of the target"); ap.add_argument("--session", default=None)
    ap.add_argument("--dry-run", action="store_true", help="fake arm and fake dashboard: the whole loop, nothing moves"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-consecutive-failures", type=int, default=3); ap.add_argument("--max-retries", type=int, default=1); ap.add_argument("--lost-after", type=int, default=3)
    ap.add_argument("--stop-file", default=None, help="while this file exists the session stops cleanly (default ~/so101/outputs/run/autoloop.stop)")
    ap.add_argument("--max-err-deg", type=float, default=8.0); ap.add_argument("--margin-mm", type=float, default=20.0); ap.add_argument("--ceiling-in", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to wait before judging the landing"); ap.add_argument("--table", default=None, help="top_table.json (default jetson/calib/top_table.json)")
    ap.add_argument("--dry-miss", type=float, default=0.1, help="--dry-run: chance that a pick misses the nut"); ap.add_argument("--out-dir", default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv); gr = load_guardrails(); pl = load_pick_loop(); cc = pl.cam_calib(); out_dir = a.out_dir or os.path.join(ROOT, "outputs", "run")
    tpath = a.table or os.path.join(ROOT, "jetson", "calib", "top_table.json")
    if a.dry_run and not os.path.exists(tpath): tpath = os.path.join(ROOT, "tests", "calib", "fixtures", "top_table_real.json")
    table = json.load(open(tpath))
    if not (table.get("arm_fit") and table.get("arm_base_mm")): raise SystemExit("autoloop needs the arm fit in top_table.json (tools/jog_capture.py --fit)")
    cfg = make_cfg(a, out_dir); rng = random.Random(a.seed)
    lim = gr.Limits(margin_mm=a.margin_mm, ceiling_in=a.ceiling_in, max_err_deg=a.max_err_deg, lost_after=a.lost_after, max_failures=a.max_consecutive_failures, **({"stop_file": a.stop_file} if a.stop_file else {}))
    lt = load(os.path.join(ROOT, "jetson", "labtrack.py"), "labtrack_for_autoloop")
    if a.dry_run:
        world = FakeWorld(cc, table, gr.fk_mat_mm, nut_in=(6.0, 8.0), rng=rng, miss_p=a.dry_miss); arm = DryArm(cc, table, world)
        pl.api, pl.run_plan, pl.find_nut, pl.fetch_top, pl.fetch_wrist = arm.api, arm.run_plan, world.find_nut, (lambda: b""), (lambda: None)
        ctx = gr.Ctx(world.api_state, world.api_scene, time.time, lambda s: None, lambda d: print(json.dumps(d), flush=True)); track = lt.LabTrack(EXPERIMENT + "-dry")
    else:
        get = lambda path, t: json.load(urllib.request.urlopen(pl.DASH + path, timeout=t))
        ctx = gr.Ctx(lambda: get("/api/state", gr.Limits().panel_s), lambda: get("/api/scene", 5), time.time, time.sleep, lambda d: print(json.dumps(d), flush=True), arm_jobs=jetson_arm_jobs); track = lt.LabTrack(EXPERIMENT)
    guard = gr.Guard(cc, table, lim, ctx); s = Runner(pl, cc, table, guard, ctx, cfg, track, rng, out=lambda l: print(l, flush=True)).run()
    return 0 if all(x["clean"] for x in s["stops"]) else 2


if __name__ == "__main__": sys.exit(main())
