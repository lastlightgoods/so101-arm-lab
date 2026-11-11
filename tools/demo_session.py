#!/usr/bin/env python3
"""demo_session.py - the Mac side of scripted-demo recording (docs/scripted-demos.md): the keep/discard rule and the glue between
tools/autoloop.py (--record-demos) and jetson/demo_recorder.py.

    keep_decision(cycle, ...)   pure: does this try become a training episode, which tier, and why not
    DemoSession(...)            open_try / on_plan / end_try / commit around each TRY; MLflow links; the dataset-version run at the end

PHASE 1: the recorder runs in this process against a fake arm (virtual clock, ticks synthesised from the plan with pick_seq's own easing)
and a fake camera. Phase 2 puts the recorder on the Jetson behind a socket; DemoSession then talks to it and `fake` is None.
"""
import hashlib, json, os, subprocess

EXPERIMENT_DATASETS = "so101-datasets"
PRE_RESET_ERROR = "nut on the target"   # autoloop's rec["error"] prefix when the pre-reset could not move the nut off the target


class Decision:
    def __init__(self, keep, tier, reasons):
        self.keep, self.tier, self.reasons = keep, tier, list(reasons); self.reason = None if keep else (self.reasons[0] if self.reasons else "unknown")

    def as_json(self): return {"keep": self.keep, "tier": self.tier, "reasons": self.reasons, "discard_reason": self.reason}

    def __repr__(self): return f"Decision(keep={self.keep}, tier={self.tier}, reasons={self.reasons})"


def keep_decision(cycle, radius_mm=15.0, trip=None, jaw_held=None, health=None):
    """KEEP only if ALL hold (plan section 5):
      1. the nut was picked and ended within `radius_mm` of the target (cycle["picked"], cycle["landed_err_mm"]; cycle["landed"] alone is not trusted: the radius is rechecked here)
      2. no guardrail tripped in the cycle (`trip` = the Trip reason or None)
      3. the final jaw check held (`jaw_held` is True; False and None, "unknown", both discard)
      4. recording health is fine (`health["ok"]`; no health dict discards)
      5. not a pre-reset artefact: the cycle has no error (the "nut on the target" case, or any other)
    Tier "A" = first attempt, "R" = success after the one retry. Never raises: any malformed input is a discard ("decision_error")."""
    try:
        reasons = []; c = cycle or {}
        if c.get("found") is False: reasons.append("nut_not_found")
        err = c.get("landed_err_mm")
        if not c.get("picked"): reasons.append("not_picked")
        if err is None: reasons.append("no_landing_measured")
        elif not (float(err) <= float(radius_mm)): reasons.append("outside_radius")
        if trip: reasons.append(f"guardrail:{trip}")
        if jaw_held is not True: reasons.append("jaw_check_failed" if jaw_held is False else "jaw_check_missing")
        if not health: reasons.append("health_missing")
        elif not health.get("ok"): reasons.append("recording_health:" + ",".join(health.get("reasons") or ["not_ok"]))
        if c.get("error"): reasons.append("pre_reset_artefact" if str(c["error"]).startswith(PRE_RESET_ERROR) else "cycle_error")
        if reasons: return Decision(False, None, reasons)
        return Decision(True, "R" if int(c.get("retries") or 0) > 0 else "A", [])
    except Exception as e: return Decision(False, None, [f"decision_error:{type(e).__name__}"])


def jaw_state(info, obj):
    """Fold the pick record's in_jaws and the scene object's in_jaws into True (held) / False (a check said no) / None (nobody checked)."""
    vals = [v for v in ((info or {}).get("in_jaws"), (obj or {}).get("in_jaws")) if v is not None]
    return None if not vals else all(bool(v) for v in vals)


def git_sha(root):
    try: return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception: return None


def features_hash(features): return hashlib.sha256(json.dumps(features, sort_keys=True, default=str).encode()).hexdigest()[:12]


class DemoSession:
    """One recording session. `recorder` = jetson/demo_recorder.DemoRecorder, `fake` = FakeRig (below) or None, `track` = a LabTrack for the
    so101-datasets experiment (its runs are no-ops when tracking is off)."""
    def __init__(self, recorder, session, dr, track=None, radius_mm=15.0, fake=None, repo_root=None, out=None):
        self.rec, self.session, self.dr, self.track, self.radius, self.fake, self.out = recorder, session, dr, track, radius_mm, fake, out or (lambda d: None)
        self.sha = git_sha(repo_root) if repo_root else None; self.parent_id = None; self.pending = None; self.max_err_deg = 0.0; self.outcomes = []; self.recording = False

    def set_parent(self, run_id): self.parent_id = run_id

    def alive(self): return self.rec.alive()

    def jaw_state(self, info, obj): return jaw_state(info, obj)

    def open_try(self, n, info):
        """The try begins: info = {"nut_in", "target_in"}. Episode meta is fixed here; the verdict fields are added at commit."""
        t = self.fake.now() if self.fake else None
        self.max_err_deg = 0.0
        meta = {"scene": {"start_in": info.get("nut_in"), "target_in": info.get("target_in"), "start_region": None},   # TODO(phase 2): start regions (plan 6.1) are not defined in the code yet
                "teacher": {"session": self.session, "cycle": n, "git_sha": self.sha},
                "links": {"mlflow_cycle_run_name": f"autoloop-{self.session}-c{n}", "mlflow_session_run_id": self.parent_id}}
        self.cycle_n = n; self.rec.open(meta, t); self.recording = True; self.pending = None

    def on_plan(self, plan, rows, max_deg_s):
        """Called by the runner after every plan the arm ran. With a fake rig this is where the plan's ticks come from; with the real arm (phase 2) pick_seq writes them."""
        if self.recording and rows: self.max_err_deg = max(self.max_err_deg, max(float(r.get("max_err_deg") or 0.0) for r in rows))
        if self.fake: self.fake.run_plan(self.rec if self.recording else None, plan, rows, max_deg_s)

    def end_try(self, cycle, jaw_held, stages):
        """The last stage returned: stop sampling (the judge and the reset run after this and are not recorded)."""
        t = self.fake.now() if self.fake else None
        health = self.rec.end(t); self.recording = False
        self.pending = {"jaw_held": jaw_held, "stages": list(stages), "health": health}

    def abort_if_recording(self, reason):
        """A trip or stop in the middle of the try: nothing partial is saved. A trip after end_try (in the reset) leaves the episode pending for commit()."""
        if self.recording: self.rec.abort(reason); self.recording = False; self.pending = None; self.outcomes.append({"kept": False, "aborted": reason}); return True
        return False

    def abort(self, reason):
        was = self.rec.abort(reason); self.recording = False
        if was: self.outcomes.append({"kept": False, "aborted": reason})
        self.pending = None; return was

    def commit(self, cycle, trip):
        """Decide and write. Sets cycle["demo"] to the outcome."""
        p = self.pending
        if p is None: return None
        self.pending = None
        d = keep_decision(cycle, self.radius, trip.reason if trip else None, p["jaw_held"], p["health"])
        jaws = [{k: s.get(k) for k in ("attempt", "in_jaws", "note")} for s in p["stages"] if s.get("stage") == "jaw_check"]
        stage_info = [{k: s[k] for k in ("stage", "attempt", "corr1_mm", "corr2_mm", "residual_mm", "sag_deg", "lowered_mm") if k in s} for s in p["stages"] if s.get("stage") in ("hover", "grasp_pose", "final")]
        self.rec.ep.meta["teacher"].update({"n_attempts": int(cycle.get("retries") or 0) + 1, "jaw_check": jaws, "landed_err_mm": cycle.get("landed_err_mm"), "stages": stage_info, "max_err_deg": round(self.max_err_deg, 2),
                                            "trip": trip.reason if trip else None})
        verdict = {**d.as_json(), "radius_mm": self.radius, "jaw_held": p["jaw_held"], "retries": cycle.get("retries"), "landed_err_mm": cycle.get("landed_err_mm")}
        res = self.rec.commit(d.keep, verdict)
        out = {"kept": d.keep, "tier": d.tier, "discard_reason": d.reason, "dataset": res["dataset"], "dataset_repo": self.rec.repo_id if d.keep else self.rec.rejects_repo, "episode_index": res["episode_index"],
               "frames": res["frames"], "config_id": self.rec.cfg.config_id, "cadence_over_budget_pct": p["health"]["over_budget_pct"], "dropped_idle_frames": p["health"]["dropped_idle_frames"]}
        cycle["demo"] = out; self.outcomes.append(out)
        return out

    # -- MLflow
    def link_cycle(self, cycle, run):
        """Inside finish_cycle: the cycle run gets the dataset link; the cycle's run id goes back to meta/lab/links.jsonl (the sidecar itself stays append-only)."""
        o = cycle.get("demo")
        if not o: return
        run.params({"dataset_repo": o["dataset_repo"], "kept": int(o["kept"]), "discard_reason": o["discard_reason"], "config_id": o["config_id"], "tier": o["tier"], "dataset": o["dataset"]})
        run.metrics({"episode_frames": o["frames"], "episode_index": o["episode_index"], "cadence_over_budget_pct": o["cadence_over_budget_pct"]})
        run.json({k: v for k, v in o.items()}, "episode_meta.json")
        if o["dataset"] and o["episode_index"] is not None:
            root = self.rec.root if o["dataset"] == "main" else self.rec.rejects_root
            self.dr.append_jsonl(os.path.join(self.dr.sidecar_dir(root), "links.jsonl"), {"episode_index": o["episode_index"], "dataset": o["dataset"], "mlflow_cycle_run_name": f"autoloop-{self.session}-c{cycle['cycle']}",
                                                                                           "mlflow_cycle_run_id": getattr(run, "id", None), "mlflow_session_run_id": self.parent_id})

    def finish(self, summary=None):
        """End of session: finalize the datasets, log the dataset-version run, return the numbers for the session summary."""
        self.abort("session_end"); self.rec.close(); cfgs = {}; frames = 0
        for r in self.dr.read_sidecar(self.rec.root):
            cid = ((r.get("configuration") or {}).get("config_id")) or "unknown"; cfgs[cid] = cfgs.get(cid, 0) + 1; frames += int(r.get("n_frames") or 0)
        have = lambda root: os.path.exists(os.path.join(root, "meta", "info.json"))
        main_hash = self.dr.dataset_hash(self.rec.root) if have(self.rec.root) else None; rej_hash = self.dr.dataset_hash(self.rec.rejects_root) if have(self.rec.rejects_root) else None
        out = {"kept": self.rec.counts["kept"], "rejected": self.rec.counts["rejected"], "aborted": self.rec.counts["aborted"], "empty": self.rec.counts["empty"], "episodes_total": sum(cfgs.values()), "frames_total": frames,
               "config_counts": cfgs, "dataset_hash": main_hash, "rejects_hash": rej_hash, "repo_id": self.rec.repo_id, "rejects_repo": self.rec.rejects_repo, "fps": self.rec.fps}
        if self.track is not None and main_hash:
            name = f"dataset-{self.rec.repo_id.split('/')[-1]}-{main_hash[:8]}"
            with self.track.run(name, tags={"kind": "dataset-version", "session": self.session}) as r:
                r.params({"repo_id": self.rec.repo_id, "rejects_repo": self.rec.rejects_repo, "dataset_hash": main_hash, "rejects_hash": rej_hash, "fps": self.rec.fps, "features_hash": features_hash(self.rec.features),
                          "session": self.session, "episodes_total": out["episodes_total"], "frames_total": frames, **{f"config_{k}": v for k, v in cfgs.items()}})
                r.metrics({"episodes_total": out["episodes_total"], "frames_total": frames, "kept": out["kept"], "rejected": out["rejected"], "aborted": out["aborted"],
                           "yield": out["kept"] / (out["kept"] + out["rejected"] + out["aborted"]) if (out["kept"] + out["rejected"] + out["aborted"]) else None})
                for rel in ("meta/info.json", "meta/lab/episodes.jsonl"): r.artifact(os.path.join(self.rec.root, rel), "dataset")
                r.json({"counts": self.rec.counts, "rejects": [{k: x.get(k) for k in ("episode_index", "verdict")} for x in self.dr.read_sidecar(self.rec.rejects_root)]}, "rejects_summary.json")
        return out


class FakeRig:
    """The arm and its tick sink, simulated: a virtual clock, pick_seq's own easing (`ps`), a TickBuffer shared with the recorder, `synth` =
    demo_recorder.synth_plan_ticks. Each plan the fake arm runs appends its ticks; while an episode is open the recorder is pumped at every tick
    time. `think_s` = the idle time before each plan (the Mac thinking between staged runs)."""
    def __init__(self, ps, buf, synth, t0=1000.0, think_s=0.8):
        self.ps, self.buf, self.synth, self.t, self.think = ps, buf, synth, t0, think_s; self.cur = {j: 0.0 for j in ps.J}

    def now(self): return self.t

    def run_plan(self, recorder, plan, rows, max_deg_s):
        self.t += self.think; t_end, self.cur, times = self.synth(self.buf, self.ps, plan, rows, max_deg_s, self.t, self.cur)
        if recorder is not None:
            for tm in times: recorder.pump(tm)
            recorder.pump(t_end)
        self.t = t_end
