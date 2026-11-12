#!/usr/bin/env python3
"""learn_loop_eval.py - the `evaluate` stage of tools/learn_loop.py (docs/learn-loop.md): run a deployed policy on the real task for M trials,
each judged by the autoloop's own judge, under the autoloop's guardrails.

One trial = guard.before -> (pre-reset if the nut sits on the target) -> POLICY ROLLOUT on the Jetson for at most T seconds, watched from the Mac
-> park the arm (the scripted `rest` plan) -> JUDGE (autoloop.Runner.judge: the overhead camera, the same radius and moved_mm) -> scripted RESET
(autoloop.Runner.reset_and_judge) -> guard.after (consecutive failures, nut lost). One JSON line per trial, same shape as the autoloop's cycle line.

The safety envelope (what the scripted pick has and what a policy can have):
  speed cap       lerobot's `--robot.max_relative_target` (joint step per control tick, SOFollower.send_action) = MAX_REL_DEG, the scripted pick's 50-80 deg/s at 30 Hz.
  workspace box   NOT enforced inside lerobot-rollout. RolloutWatch polls the follower's joints (/api/state, fed by the busheal plugin during a rollout), runs the
                  guardrails' own FK + box_check and stop-file check every POLL_S, and has the transport run so101-stop on a trip. A tripwire with ~0.3-1 s latency,
                  not a clamp: the speed cap bounds how far the arm can get meanwhile. An in-process FK clamp in the rollout loop is the missing piece (docs/learn-loop.md).
"""
import json, math, os, shlex, statistics, subprocess, time

TASK = "Pick up the nut and place it on the target"   # the recorder's one fixed task string (jetson/demo_recorder.TASK)
MAX_REL_DEG = 2.0       # why: 2 deg per 30 Hz tick = 60 deg/s, between pick_loop.SPEED closed (50) and open (80); the policy gets no faster than the scripted pick (11 Nov 2026)
POLL_S = 0.25           # watchdog period
STALE_S = 2.5           # a follower fix older than this (or without degrees) is "state lost": the box cannot be checked, so the arm is stopped
STATE_GRACE_S = 30.0    # nudge + model load before the first fix; the arm is not under the policy yet
END_GRACE_S = 40.0      # lerobot-rollout returns the arm to its start pose after --duration; give it this long before calling it hung
CLEAN_RC = (143, -15, -2, 130)   # so101-stop (SIGTERM) or Ctrl-C: the owner stopping the arm, not a fault
ALL_CAMS = ("front", "top", "wrist")


def ckpt_images(ckpt_dir):
    """Camera names a checkpoint's config.json expects (observation.images.<name>), sorted; [] when unreadable."""
    try: feats = json.load(open(os.path.join(ckpt_dir, "config.json"))).get("input_features") or {}
    except (OSError, ValueError): return []
    return sorted(k.rsplit(".", 1)[-1] for k in feats if k.startswith("observation.images."))


def cams_expr(images):
    """The Jetson env var holding the camera set a policy was trained on: all three -> $SO101_CAMS, top+wrist -> $SO101_CAMS_2 (jetson/env.sh)."""
    s = tuple(sorted(images))
    if s == ALL_CAMS: return "$SO101_CAMS"
    if s == ("top", "wrist"): return "$SO101_CAMS_2"
    raise ValueError(f"no camera set on the Jetson matches the policy's images {list(s)}; env.sh offers top+front+wrist and top+wrist")


def rollout_script(name, kind, chunk, duration_s, images, max_rel=MAX_REL_DEG, task=TASK):
    """The bash the Jetson runs for one trial (piped to `ssh jetson bash -s`): the dashboard's POLICY button (nudge, cams-take, lerobot-rollout rtc, cams-release)
    with the speed cap added and torque kept on disconnect so the arm holds its pose until the scripted park."""
    ck = f'"$HOME/so101/policies/{name}/current"'; cams = cams_expr(images)
    if kind == "smolvla":   # so101-run-vla: cameras renamed as in training, execution horizon 10
        infer = "--inference.type=rtc --inference.rtc.execution_horizon=10 --inference.queue_threshold=40"
        extra = ' --rename_map="{\\"observation.images.top\\": \\"observation.images.camera1\\", \\"observation.images.wrist\\": \\"observation.images.camera2\\"}"'; pre = "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    else: infer = f"--inference.type=rtc --inference.queue_threshold={max(1, int(0.9 * int(chunk)))}"; extra = ""; pre = ""   # so101-run-async: re-plan when the queue is down to 90 percent
    return (f'source "$HOME/so101/env.sh" >/dev/null 2>&1\n{pre}so101-nudge || {{ echo "nudge failed" >&2; exit 3; }}\nso101-cams-take\n'
            f'lerobot-rollout --strategy.type=base {infer} --policy.path={ck} --policy.device=cuda{extra} --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID '
            f'--robot.max_relative_target={max_rel:g} --robot.disable_torque_on_disconnect=false --robot.cameras="{cams}" --task={shlex.quote(task)} --duration={duration_s:g} --display_data=false\n'
            'rc=$?\nso101-cams-release\nexit $rc\n')


class RolloutWatch:
    """The Mac-side tripwire for a policy rollout. check(elapsed) -> a guardrails.Trip or None: stop file (clean), workspace (FK tip of the follower's read-back
    joints through box_check, the same Limits as the scripted pick), state_lost (no usable fix for STALE_S; the first fix gets STATE_GRACE_S)."""
    def __init__(self, gr, guard, cc, table, ctx, stale_s=STALE_S, grace_s=STATE_GRACE_S):
        self.gr, self.guard, self.cc, self.table, self.ctx, self.stale_s, self.grace_s = gr, guard, cc, table, ctx, stale_s, grace_s
        self.seen = False; self.last_ok = 0.0; self.fixes = 0; self.worst = None

    def check(self, elapsed):
        if t := self.guard.check_stop(): return t
        pos = None
        try:
            fol = ((self.ctx.fetch_state() or {}).get("arms") or {}).get("follower") or {}; p = fol.get("pos") or {}
            if all(k in p for k in self.gr.ARM) and fol.get("mode", "deg") == "deg" and self.ctx.now() - float(fol.get("t", self.ctx.now())) <= self.stale_s: pos = {k: float(p[k]) for k in self.gr.ARM}
        except Exception: pos = None
        if pos is None:
            if (self.seen and elapsed - self.last_ok >= self.stale_s) or (not self.seen and elapsed >= self.grace_s):
                return self.gr.Trip("state_lost", {"elapsed_s": round(elapsed, 1), "stale_s": self.stale_s, "seen": self.seen})
            return None
        self.seen = True; self.last_ok = elapsed; self.fixes += 1
        x, y, h = self.gr.fk_mat_mm(self.cc, self.table, pos); self.worst = (round(x / 25.4, 2), round(y / 25.4, 2), round(h, 1))
        if why := self.gr.box_check(x, y, h, self.guard.lim): return self.gr.Trip("workspace", {"phase": "policy", "x_in": round(x / 25.4, 3), "y_in": round(y / 25.4, 3), "h_mm": round(h, 1), "why": why, "elapsed_s": round(elapsed, 1)})
        return None


class SshTransport:
    """The real thing: the trial script over `ssh <host> bash -s`, polled every POLL_S by `tick(elapsed)` (a RolloutWatch.check); a Trip or an overrun ends in
    `so101-stop` on the Jetson. rollout() -> {"rc", "elapsed_s", "trip", "overrun"}. The tests drive it with a fake Popen and clock; nothing here runs without a Jetson."""
    def __init__(self, host="jetson", popen=subprocess.Popen, run=subprocess.run, clock=time.monotonic, sleep=time.sleep):
        self.host, self.popen, self.run, self.clock, self.sleep = host, popen, run, clock, sleep; self.opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]

    def stop(self):
        try: self.run(["ssh", *self.opts, self.host, "bash -s"], input='source "$HOME/so101/env.sh" >/dev/null 2>&1; so101-stop\n', capture_output=True, text=True, timeout=30)
        except Exception: pass   # why: ssh down means the arm cannot be told anything; the owner's STOP button is the other road

    def rollout(self, script, duration_s, tick, poll_s=POLL_S):
        p = self.popen(["ssh", *self.opts, self.host, "bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        p.stdin.write(script); p.stdin.close(); t0 = self.clock(); trip = None; overrun = False
        while p.poll() is None:
            self.sleep(poll_s); el = self.clock() - t0
            if (trip := tick(el)) is not None: break
            if el > duration_s + END_GRACE_S: overrun = True; break
        if trip is not None or overrun:
            self.stop()
            for _ in range(20):
                if p.poll() is not None: break
                self.sleep(0.5)
            else: p.kill()
        return {"rc": p.poll(), "elapsed_s": round(self.clock() - t0, 1), "trip": trip, "overrun": overrun}


def wilson(k, n, z=1.96):
    """95 percent Wilson interval for k successes in n trials -> (low, high), or (None, None) when n is 0."""
    if not n: return None, None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - w), 3), round(min(1.0, c + w), 3)


def summarize(trials, radius_mm):
    n = len(trials); ok = sum(1 for t in trials if t.get("ok")); errs = [t["landed_err_mm"] for t in trials if t.get("landed_err_mm") is not None]; placed = [t["landed_err_mm"] for t in trials if t.get("ok")]
    lo, hi = wilson(ok, n)
    return {"trials": n, "landed": ok, "picked": sum(1 for t in trials if t.get("picked")), "success_rate": round(ok / n, 3) if n else None, "ci95": [lo, hi],
            "mean_err_mm": round(statistics.mean(errs), 1) if errs else None, "median_err_mm": round(statistics.median(errs), 1) if errs else None,
            "median_placed_err_mm": round(statistics.median(placed), 1) if placed else None, "radius_mm": radius_mm,
            "policy_s_mean": round(statistics.mean(t["policy_s"] for t in trials if t.get("policy_s") is not None), 1) if any(t.get("policy_s") is not None for t in trials) else None}


class PolicyEval:
    """`rn` is an autoloop.Runner (its pick_loop, guard, ctx, judge and reset are the autoloop's own); `transport.rollout(script, T, tick)` / `.stop()` is the Jetson side;
    `lineage` are tags that tie every trial's MLflow run to the registry version, the train run and the dataset hash."""
    def __init__(self, rn, al, gr, transport, script, duration_s, track=None, lineage=None, settle_s=3.0, fresh_s=20.0):
        self.rn, self.al, self.gr, self.tr, self.script, self.T, self.track, self.lineage = rn, al, gr, transport, script, float(duration_s), track, dict(lineage or {})
        self.settle_s, self.fresh_s = settle_s, fresh_s; self.stops = []; self.trials = []

    # -- pieces
    def wait_fresh_scene(self):
        """The rollout held the cameras (so101-cams-take): wait until the dashboard's scene is live again (age_s small) before judging; False when it never is."""
        rn = self.rn; t0 = rn.ctx.now()
        while rn.ctx.now() - t0 <= self.fresh_s:
            sc = rn.scene()
            if sc is not None and float(sc.get("age_s", 0.0) or 0.0) < 2.0: return True
            rn.ctx.sleep(1.0)
        return False

    def park(self):
        rn = self.rn; rn.pl.run_plan(rn.pl.build_plan(None, None, {}, names=("rest",)), rn.pick_args.max_deg_s)   # rn.pl.run_plan is the guarded one inside rn.guarded()

    def rollout(self, rec):
        """Run the policy; returns None or the Trip that ended it. Fills rec["policy_s"], rec["rollout"]."""
        rn = self.rn; watch = RolloutWatch(self.gr, rn.guard, rn.cc, rn.table, rn.ctx); rn.in_motion = True
        out = self.tr.rollout(self.script, self.T, watch.check); rec["policy_s"] = out["elapsed_s"]; rec["rollout"] = {"rc": out["rc"], "overrun": out["overrun"], "fixes": watch.fixes, "last_tip": watch.worst}
        if out["trip"] is not None: self.tr.stop(); return out["trip"]
        if watch.fixes == 0 and out["rc"] in (0, None) and not out["overrun"]: return rn.guard.trip("unwatched", {"elapsed_s": out["elapsed_s"], "note": "the arm never reported joints during the rollout, so the workspace box was not checked"})   # why: a silent state stream must not look like a clean run
        if out["overrun"]: self.tr.stop(); return rn.guard.trip("rollout_hung", {"after_s": out["elapsed_s"], "limit_s": self.T + END_GRACE_S})
        rc = out["rc"]
        if rc not in (0, None) and out["elapsed_s"] < self.T - 2.0:   # ended early and not by itself
            return rn.guard.trip("stop_arm" if rc in CLEAN_RC else "rollout_failed", {"rc": rc, "elapsed_s": out["elapsed_s"]}, clean=rc in CLEAN_RC)
        if rc not in (0, None): return rn.guard.trip("rollout_failed", {"rc": rc, "elapsed_s": out["elapsed_s"]})
        return None

    # -- one trial
    def trial(self, n):
        rn = self.rn; t0 = rn.ctx.now(); rn.n = n
        rec = {"event": "trial", "trial": n, "found": True, "picked": False, "landed": False, "ok": False, "landed_err_mm": None, "moved_mm": None, "policy_s": None, "target_in": None}; trip = None; policy_trip = False
        try:
            with rn.guarded():
                sc = rn.scene(); rec["target_in"] = tgt = rn.target_for((sc or {}).get("objects") or []); nut = rn.locate(rn.cur_in)
                if not nut: rec["found"] = False
                else:
                    if self.al.dist_mm(nut[0], tgt) < rn.cfg.moved_mm:   # why: same rule as Runner.cycle, a nut already on the target cannot be judged as moved by the policy
                        objs = (rn.scene() or {}).get("objects") or []; nut_obj, _ = self.al.nearest_object(objs, nut[0])
                        pr = rn.reset_and_judge(nut[0], objs, nut_obj, f"t{n}_prereset"); rec["pre_reset"] = {k: pr.get(k) for k in ("spot_in", "at_in", "err_mm", "ok", "how", "error")}
                        if pr.get("at_in") is None or self.al.dist_mm(pr["at_in"], tgt) < rn.cfg.moved_mm: rec["error"] = "nut on the target and the reset did not move it away"
                        else: nut = rn.locate(pr["at_in"]); rec["found"] = bool(nut)
                if rec["found"] and "error" not in rec:
                    rec["nut_in"] = [round(v, 3) for v in nut[0]]; rn.cur_in = nut[0]
                    t = self.rollout(rec)
                    if t is not None: policy_trip = True; rn.in_motion = False; raise self.al.TripError(t)   # why: after a policy escape no further arm command is sent (not even `rest`): the owner looks first
                    self.park(); rn.in_motion = False; rec["scene_live"] = self.wait_fresh_scene()
                    j = rn.judge(nut[0], tgt, {}, f"t{n}_judge"); rec.update(picked=j["picked"], landed=j["landed"], landed_err_mm=j["err_mm"], moved_mm=j["moved_mm"], judge_wait_s=j["wait_s"], judge_source=j.get("source"))
                    if j["after_in"]: rn.cur_in = j["after_in"]
                    rec["ok"] = rec["picked"] and rec["landed"]
                    if j["after_in"]:
                        r = rn.reset_and_judge(rn.cur_in, j["objects"], j.get("obj"), f"t{n}_reset"); rec["reset_to_in"] = r["spot_in"]
                        if r["error"]: rec["reset_error"] = r["error"]
                        rec.update(reset_ok=r.get("ok"), reset_err_mm=r.get("err_mm"), reset_how=r.get("how"))
        except self.al.TripError as e: trip = e.trip
        rec["trial_s"] = round(rn.ctx.now() - t0, 2); rec["policy_trip"] = policy_trip
        return rec, trip or rn.guard.check_after_cycle(rec)

    def log_trial(self, rec, trip, parent_id=None):
        """One MLflow run per trial, tagged with the lineage; never raises into the arm loop."""
        if self.track is None: return None
        try:
            with self.track.run(f"eval-trial-{rec['trial']}", tags={"kind": "eval-trial", "trial": rec["trial"], "trip": (trip.reason if trip else "none"), **({"mlflow.parentRunId": parent_id} if parent_id else {}), **self.lineage}) as tr:
                tr.params({"target": ",".join(f"{v:.2f}" for v in rec["target_in"]) if rec.get("target_in") else None, "duration_s": self.T, "max_rel_deg": MAX_REL_DEG})
                tr.metrics({"ok": int(rec["ok"]), "picked": int(rec["picked"]), "landed": int(rec["landed"]), "landed_err_mm": rec["landed_err_mm"], "moved_mm": rec["moved_mm"], "policy_s": rec["policy_s"], "trial_s": rec["trial_s"], "trip": int(trip is not None)})
                tr.json(rec, "trial.json"); return getattr(tr, "id", None)
        except Exception: return None

    # -- the session
    def run(self, first, last, parent_id=None, on_trial=None, say=None):
        """Trials first..last (1-based). Returns {"trials": [...], "end": "trials"|<trip reason>, "stop": trip json or None}; stops at the first trip like the autoloop."""
        rn = self.rn; say = say or rn.say; end = "trials"; stop = None
        for n in range(first, last + 1):
            trip = rn.guard.check_before_cycle(rn.ctx)
            if trip: rn.stop(trip, n); stop = {**trip.as_json(), "trial": n}; end = trip.reason; break
            rec, trip = self.trial(n); rec["trip"] = trip.reason if trip else None; rec["mlflow_run"] = self.log_trial(rec, trip, parent_id); self.trials.append(rec); say(rec)
            if on_trial: on_trial(rec)
            if trip: rn.stop(trip, n); stop = {**trip.as_json(), "trial": n}; end = trip.reason; break
        return {"trials": list(self.trials), "end": end, "stop": stop}
