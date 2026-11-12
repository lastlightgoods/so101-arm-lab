#!/usr/bin/env python3
"""demo_session.py - the Mac side of scripted-demo recording (docs/scripted-demos.md): the keep/discard rule and the glue between
tools/autoloop.py (--record-demos) and jetson/demo_recorder.py.

    keep_decision(cycle, ...)   pure: does this try become a training episode, which tier, and why not
    DemoSession(...)            open_try / on_plan / end_try / commit around each TRY; MLflow links; the dataset-version run at the end

    RemoteRecorder(transport, ...)  PHASE 2: the same interface as jetson/demo_recorder.DemoRecorder, but every call is one request to the resident
                                    jetson/demo_recorderd.py (over ssh in real runs; `transport` is injected so tests and the dry run use no ssh)

PHASE 1 (--dry-run): the recorder runs in this process against a fake arm (virtual clock, ticks synthesised from the plan with pick_seq's own easing)
and a fake camera. PHASE 2: the recorder is the resident process on the Jetson, DemoSession gets a RemoteRecorder and `fake` is None.
"""
import hashlib, json, os, subprocess, tempfile

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
        self.sha = git_sha(repo_root) if repo_root else None; self.parent_id = None; self.pending = None; self.max_err_deg = 0.0; self.outcomes = []; self.recording = False; self.open_err = None

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
        self.cycle_n = n; self.pending = None; self.open_err = None
        try: self.rec.open(meta, t); self.recording = True
        except RecorderError as e: self.recording = False; self.open_err = str(e)[:160]   # why: a recorder that refuses or does not answer must never stop the arm; this try is simply not recorded (2026-11-11)

    def on_plan(self, plan, rows, max_deg_s):
        """Called by the runner after every plan the arm ran. With a fake rig this is where the plan's ticks come from; with the real arm (phase 2) pick_seq writes them."""
        if self.recording and rows: self.max_err_deg = max(self.max_err_deg, max(float(r.get("max_err_deg") or 0.0) for r in rows))
        if self.fake: self.fake.run_plan(self.rec if self.recording else None, plan, rows, max_deg_s)

    def end_try(self, cycle, jaw_held, stages):
        """The last stage returned: stop sampling (the judge and the reset run after this and are not recorded)."""
        if not self.recording: return   # the recorder refused this try at open_try (commit() reports it)
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
        if p is None and self.open_err:
            out = {"kept": False, "tier": None, "discard_reason": f"recorder_open_failed:{self.open_err}", "dataset": None, "dataset_repo": self.rec.rejects_repo, "episode_index": None, "frames": 0, "config_id": self.rec.cfg.config_id,
                   "cadence_over_budget_pct": 0.0, "dropped_idle_frames": 0}
            cycle["demo"] = out; self.outcomes.append(out); self.open_err = None; return out
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
            self.rec.append_link(o["dataset"], {"episode_index": o["episode_index"], "dataset": o["dataset"], "mlflow_cycle_run_name": f"autoloop-{self.session}-c{cycle['cycle']}",
                                                                                           "mlflow_cycle_run_id": getattr(run, "id", None), "mlflow_session_run_id": self.parent_id})

    def finish(self, summary=None):
        """End of session: finalize the datasets, log the dataset-version run, return the numbers for the session summary."""
        self.abort("session_end"); self.rec.close()
        try: sm = self.rec.summary()
        except RecorderError as e:   # why: a recorder that died during the session must not crash the session summary; the counts the runner kept are still right (2026-11-11)
            c = self.rec.counts; return {**c, "episodes_total": None, "frames_total": None, "config_counts": {}, "dataset_hash": None, "rejects_hash": None, "repo_id": self.rec.repo_id, "rejects_repo": self.rec.rejects_repo, "fps": self.rec.fps, "summary_error": str(e)[:120]}
        cfgs = {}; frames = 0
        for r in sm["main"]["rows"]:
            cid = ((r.get("configuration") or {}).get("config_id")) or "unknown"; cfgs[cid] = cfgs.get(cid, 0) + 1; frames += int(r.get("n_frames") or 0)
        main_hash, rej_hash = sm["main"]["hash"], sm["rejects"]["hash"]; c = self.rec.counts
        out = {"kept": c["kept"], "rejected": c["rejected"], "aborted": c["aborted"], "empty": c["empty"], "episodes_total": sum(cfgs.values()), "frames_total": frames,
               "config_counts": cfgs, "dataset_hash": main_hash, "rejects_hash": rej_hash, "repo_id": self.rec.repo_id, "rejects_repo": self.rec.rejects_repo, "fps": self.rec.fps}
        if self.track is not None and main_hash:
            name = f"dataset-{self.rec.repo_id.split('/')[-1]}-{main_hash[:8]}"
            with self.track.run(name, tags={"kind": "dataset-version", "session": self.session}) as r:
                r.params({"repo_id": self.rec.repo_id, "rejects_repo": self.rec.rejects_repo, "dataset_hash": main_hash, "rejects_hash": rej_hash, "fps": self.rec.fps, "features_hash": features_hash(sm["features"]),
                          "session": self.session, "episodes_total": out["episodes_total"], "frames_total": frames, **{f"config_{k}": v for k, v in cfgs.items()}})
                r.metrics({"episodes_total": out["episodes_total"], "frames_total": frames, "kept": out["kept"], "rejected": out["rejected"], "aborted": out["aborted"],
                           "yield": out["kept"] / (out["kept"] + out["rejected"] + out["aborted"]) if (out["kept"] + out["rejected"] + out["aborted"]) else None})
                with tempfile.TemporaryDirectory() as td:   # why: the datasets may live on the Jetson; the artifacts are written from the summary text, same file names as before
                    for fn, txt in (("info.json", sm["main"]["info"]), ("episodes.jsonl", sm["main"]["episodes"])):
                        if txt is not None: pth = os.path.join(td, fn); open(pth, "w").write(txt); r.artifact(pth, "dataset")
                r.json({"counts": c, "rejects": [{k: x.get(k) for k in ("episode_index", "verdict")} for x in sm["rejects"]["rows"]]}, "rejects_summary.json")
        return out


class RecorderError(Exception):
    """The resident recorder refused a request or did not answer."""


def ssh_transport(host="jetson", python="~/so101/.venv/bin/python", script="~/so101/demo_recorderd.py", connect_timeout=8):
    """transport(op, payload, timeout) -> reply dict, over one ssh per call (payload on stdin, reply on stdout). Never raises: a failure is {"ok": False, "err": ...}.
    A request is one call, so a lost connection costs one episode, never the arm."""
    def call(op, payload, timeout=30.0):
        if not op.isalpha(): return {"ok": False, "err": "bad op"}
        cmd = f"cd ~/so101 && source env.sh >/dev/null 2>&1; {python} {script} ctl {op} --timeout {int(timeout)}"
        try: r = subprocess.run(["ssh", "-o", f"ConnectTimeout={connect_timeout}", host, cmd], input=json.dumps(payload), capture_output=True, text=True, timeout=timeout + 15)
        except Exception as e: return {"ok": False, "err": f"ssh: {type(e).__name__}: {str(e)[:80]}"}
        for line in reversed(r.stdout.strip().splitlines()):
            try: return json.loads(line)
            except ValueError: continue
        return {"ok": False, "err": f"no reply (rc {r.returncode}): {r.stderr.strip()[-120:]}"}
    return call


class _EpStub:
    def __init__(self, meta): self.meta = meta


class RemoteRecorder:
    """DemoRecorder's interface (open / end / commit / abort / alive / close / summary / append_link and the attributes DemoSession reads) over a control channel to
    jetson/demo_recorderd.py. `transport(op, payload, timeout)` returns the reply dict (ssh_transport, or a fake in tests).
    Only open() raises (RecorderError, and DemoSession turns that into "this try is not recorded"); end / commit / abort / alive never do: a dead recorder gives an
    unhealthy episode and a False alive(), which the runner's `recorder` guardrail stops the session on between cycles, never in the middle of a pick."""
    def __init__(self, transport, root, repo_id, cfg, rejects_root=None, rejects_repo=None, vcodec="libsvtav1", fps=30):
        self.t, self.root, self.repo_id, self.cfg, self.vcodec, self.fps = transport, root, repo_id, cfg, vcodec, fps
        self.rejects_root = rejects_root or root.rstrip("/") + "-rejects"; self.rejects_repo = rejects_repo or repo_id + "-rejects"
        self.counts = {"kept": 0, "rejected": 0, "aborted": 0, "empty": 0}; self.ep = None; self._summary = None; self.features = {}

    def configure(self):
        r = self.t("configure", {"repo": self.repo_id, "root": self.root, "rejects_root": self.rejects_root, "rejects_repo": self.rejects_repo, "lab": self.cfg.to_dict(), "vcodec": self.vcodec}, 120.0)
        if not r.get("ok"): raise RecorderError(f"configure: {r.get('err')}")
        self.root, self.rejects_root, self.fps = r.get("root", self.root), r.get("rejects_root", self.rejects_root), r.get("fps", self.fps); self._summary = None; return r

    def alive(self): return bool(self.t("ping", {}, 10.0).get("ok"))

    def status(self): return self.t("status", {}, 15.0)

    def open(self, meta=None, t=None):
        self._summary = None; r = self.t("open", {"meta": meta or {}}, 20.0)
        if not r.get("ok"): raise RecorderError(str(r.get("err")))
        self.ep = _EpStub(meta or {}); return r

    def end(self, t=None):
        r = self.t("end", {}, 30.0)
        if r.get("ok"): return r["health"]
        return {"ok": False, "reasons": [f"recorder_unreachable:{str(r.get('err'))[:60]}"], "frames": 0, "dropped_idle_frames": 0, "over_budget_pct": 0.0, "cam_age_max_s": {}, "banding": {}}

    def abort(self, reason="aborted"):
        if self.ep is None: return False
        r = self.t("abort", {"reason": reason}, 20.0); was = bool(r.get("was_open"))
        if was: self.counts["aborted"] += 1
        self.ep = None; return was

    def commit(self, keep, verdict=None):
        meta = self.ep.meta if self.ep is not None else {}; r = self.t("commit", {"keep": bool(keep), "verdict": verdict or {}, "meta": meta}, 900.0); self.ep = None
        if not r.get("ok"):
            self.counts["aborted"] += 1; return {"dataset": None, "episode_index": None, "frames": 0, "health": {"ok": False, "reasons": [f"commit_failed:{str(r.get('err'))[:80]}"]}}
        if r.get("dataset") is None: self.counts["empty"] += 1
        else: self.counts["kept" if r["dataset"] == "main" else "rejected"] += 1
        return {k: r.get(k) for k in ("dataset", "episode_index", "frames", "health")}

    def append_link(self, kind, row): self.t("link", {"kind": kind, "row": row}, 20.0)

    def close(self):
        r = self.t("finish", {}, 300.0)
        if r.get("ok"): self._summary = r; self.features = r.get("features") or {}

    def summary(self):
        if self._summary is None:
            r = self.t("summary", {}, 120.0)
            if not r.get("ok"): raise RecorderError(f"summary: {r.get('err')}")
            self._summary = r; self.features = r.get("features") or {}
        return self._summary


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
