#!/usr/bin/env python3
"""learn_loop.py - record -> train -> register -> deploy -> evaluate -> report, one resumable routine (docs/learn-loop.md).

  tools/learn_loop.py plan|status|record|train|register|deploy|evaluate|report|all --name NAME [--cycles 30 --trials 10 --trial-s 25 --steps 20000] [--dry-run]

  record    tools/autoloop.py --record-demos: the closed-loop scripted pick; only episodes the judge's keep rule accepts land in the main dataset (rejects go beside it)
  train     lerobot-train ACT on that dataset (Mac MPS by default), input features built from the dataset's meta/info.json, tracked in MLflow with the dataset hash
  register  tools/so101_policy.py register: a registry version carrying the metrics, the dataset hash and the train run
  deploy    tools/so101_policy.py deploy: rsync + sha256 check to the Jetson, `current` flipped, `status` must agree
  evaluate  M trials of the deployed policy on the real task under the autoloop's guardrails and judge (tools/learn_loop_eval.py)
  report    success rate and placement error against the scripted baseline of so101-autoloop

State lives in <out-dir>/<name>/state.json: a stage that is `done` is skipped, a failed or interrupted one is re-run by repeating the same command (train resumes from its
last checkpoint, evaluate from its last finished trial). Re-running a stage with --force invalidates the stages after it. --dry-run swaps the arm, the Jetson, lerobot-train and the
dashboard for the fakes of tools/learn_loop_fakes.py and keeps MLflow in <run dir>/mlflow.db. Exit 0 ok, 1 a stage failed, 2 a guardrail tripped or the stop file was set.
"""
import argparse, contextlib, importlib.util, io, json, os, random, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools"); JETSON = os.path.join(ROOT, "jetson")
STAGES = ("record", "train", "register", "deploy", "evaluate", "report")
EXP_LOOP, EXP_TRAIN, EXP_DATASETS, EXP_AUTOLOOP = "so101-learn-loop", "so101-act", "so101-datasets", "so101-autoloop"
# why: measured steps/s of past runs (outputs/train/*.log, docs/learn-loop.md): Mac MPS batch 8 on the real 2-camera lighter data 1.45-1.49 step/s (act_lighter 25k in 4h38m, ft_lighter 8k in 1h34m);
# Orin Nano CUDA batch 4 ~1.9 step/s (so101-finetune) and it OOMs above that on the shared 8 GB; sim data at 320x240 ran 3.0-4.0 step/s on the Mac (11 Nov 2026)
RATE = {"mps": (1.45, 8), "cuda": (1.9, 4), "cpu": (0.3, 8)}   # device -> (steps/s, batch it was measured at)
CYCLE_S = 86.0          # median real scripted cycle (try + judge + reset) in so101-autoloop, 19 cycles (11 Nov 2026)
TRIAL_OVERHEAD_S = 45.0  # park + wait for the cameras + judge + scripted reset around the policy's T seconds, an estimate until the first real evaluation
STEP_RE = re.compile(r"step:\s*([\d.]+)([KM]?)"); LOSS_RE = re.compile(r"loss:\s*([\d.]+)"); RATE_RE = re.compile(r"([\d.]+)\s*step/s")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def mod(rel, name): return load(os.path.join(ROOT, rel), name)


class StageError(Exception):
    def __init__(self, msg, code=1): super().__init__(msg); self.code = code


class RealSys:
    """subprocess with the output teed to a log file and the console; run(argv, log) -> return code."""
    def run(self, argv, log=None):
        f = open(log, "a") if log else None
        try:
            p = subprocess.Popen([str(a) for a in argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                if f: f.write(line)
            return p.wait()
        finally:
            if f: f.close()


class HybridSys:
    """--dry-run --dry-real-record: the real `autoloop.py --dry-run --record-demos` (fake arm, fake cameras, real LeRobot dataset writer) for record, the fake lerobot-train for the rest."""
    def __init__(self, fake, real): self.fake, self.real = fake, real
    def run(self, argv, log=None): return (self.real if any(str(x).endswith("autoloop.py") for x in argv) else self.fake).run(argv, log)


def default_sys(a):
    if not a.dry_run: return RealSys()
    fake = mod("tools/learn_loop_fakes.py", "learn_loop_fakes_for_sys").FakeSys(); return HybridSys(fake, RealSys()) if a.dry_real_record else fake


def parse_log(text):
    """[(step, loss)] from lerobot-train's log lines (the dashboard's and mlflow_backfill's regexes) and the last reported step/s."""
    pts = []; rate = None
    for line in text.replace("\r", "\n").splitlines():
        m = STEP_RE.search(line); l = LOSS_RE.search(line)
        if m and l: pts.append((int(float(m.group(1)) * {"K": 1e3, "M": 1e6}.get(m.group(2), 1)), float(l.group(1))))
        r = RATE_RE.search(line)
        if r: rate = float(r.group(1))
    return pts, rate


# ---------------------------------------------------------------- pure builders
def train_cfg(info, a):
    """Everything lerobot-train needs, from the dataset's meta/info.json: the camera set (must be one the Jetson can serve), the input features WITHOUT observation.environment_state
    (the recorder writes it, lerobot-rollout cannot supply it at run time), fps, sizes. Raises StageError when the dataset cannot make a policy that runs on the Jetson."""
    ev = mod("tools/learn_loop_eval.py", "learn_loop_eval_for_cfg"); feats = info.get("features") or {}
    if "action" not in feats or "observation.state" not in feats: raise StageError("dataset has no action / observation.state features")
    imgs = sorted(k.rsplit(".", 1)[-1] for k in feats if k.startswith("observation.images."))
    try: ev.cams_expr(imgs)
    except ValueError as e: raise StageError(str(e)) from e
    inp = {"observation.state": {"type": "STATE", "shape": list(feats["observation.state"]["shape"])}}
    for c in imgs: h, w, ch = feats[f"observation.images.{c}"]["shape"]; inp[f"observation.images.{c}"] = {"type": "VISUAL", "shape": [ch, h, w]}
    return {"policy": a.policy, "device": a.device, "batch_size": a.batch, "steps": a.steps, "chunk_size": a.chunk, "fps": info.get("fps"), "images": imgs, "input_features": inp,
            "dropped_features": sorted(k for k in feats if k.startswith("observation.") and k not in inp), "episodes": info.get("total_episodes"), "frames": info.get("total_frames")}


def train_argv(cfg, repo, root, out_dir, job, a, binary=None):
    binary = binary or os.path.join(os.path.dirname(sys.executable), "lerobot-train"); base = [binary, f"--dataset.repo_id={repo}", f"--dataset.root={root}", "--policy.push_to_hub=false", f"--policy.device={cfg['device']}",
                                                                                             f"--output_dir={out_dir}", f"--job_name={job}", f"--steps={cfg['steps']}", f"--batch_size={cfg['batch_size']}", f"--num_workers={a.workers}",
                                                                                             f"--save_freq={a.save_freq}", "--log_freq=100", "--wandb.enable=false"]
    if cfg["policy"] == "smolvla":   # why: the recipe of NEXT_STEPS.md (16 Oct 2026, HF Jobs): base weights, top/wrist renamed to camera1/2, one empty camera; not run from here yet
        return base + ["--policy.path=lerobot/smolvla_base", "--policy.empty_cameras=1", '--rename_map={"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}']
    return base + ["--policy.type=act", f"--policy.chunk_size={cfg['chunk_size']}", f"--policy.n_action_steps={cfg['chunk_size']}", f"--policy.input_features={json.dumps(cfg['input_features'], separators=(',', ':'))}"]


def resume_argv(out_dir, binary=None):
    binary = binary or os.path.join(os.path.dirname(sys.executable), "lerobot-train")
    return [binary, "--resume=true", f"--config_path={os.path.join(out_dir, 'checkpoints', 'last', 'pretrained_model', 'train_config.json')}"]


def estimates(a, kept=None):
    """Seconds per stage from the measured rates above; the numbers docs/learn-loop.md quotes."""
    rate, at = RATE.get(a.device, RATE["mps"]); sps = rate * (at / a.batch) if a.batch else rate   # throughput scales with samples/s, not steps/s
    return {"record": a.cycles * CYCLE_S, "train": a.steps / sps, "register": 30.0, "deploy": 90.0, "evaluate": a.trials * (a.trial_s + TRIAL_OVERHEAD_S), "report": 5.0}


def fmt_s(s): return f"{s / 3600:.1f} h" if s >= 5400 else f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s"


# ---------------------------------------------------------------- the loop
class Loop:
    def __init__(self, a, sys_=None, transport=None, runtime=None, clock=time.time):
        self.a, self.sys, self.transport, self.runtime, self.clock = a, sys_ or default_sys(a), transport, runtime, clock
        self.base = os.path.abspath(os.path.expanduser(a.out_dir)) if a.out_dir else os.path.join(ROOT, "outputs", "learn_loop"); self.dir = os.path.join(self.base, a.name); os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json"); self.state = self.load_state(); self._lt = None; self._tracks = {}; self.out = print
        self.uri = a.mlflow_uri or (("sqlite:///" + os.path.join(self.dir, "mlflow.db")) if a.dry_run and not (os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("SO101_MLFLOW_URI")) else None); self.env = {}
        if self.uri: self.env.update(MLFLOW_TRACKING_URI=self.uri, SO101_MLFLOW_URI=self.uri, SO101_MLFLOW_ARTIFACTS=os.path.join(self.dir, "artifacts"))
        elif os.environ.get("SO101_MLFLOW_URI") and not os.environ.get("MLFLOW_TRACKING_URI"): self.env["MLFLOW_TRACKING_URI"] = os.environ["SO101_MLFLOW_URI"]   # why: so101_policy reads only MLFLOW_TRACKING_URI, the loggers read SO101_MLFLOW_URI; one server for both
        self.sfx = "-dry" if a.dry_run else ""
        if a.dry_run and not a.stop_file: a.stop_file = os.path.join(self.dir, "stop")   # why: a dry run must not be tripped by (or touch) the real autoloop.stop
        if a.dry_run: self.env["SO101_DEPLOY_LOCAL_ROOT"] = os.path.join(self.dir, "jetson_home")   # why: so101_policy's own stand-in for the Jetson: ssh becomes bash -s, rsync a local copy

    @contextlib.contextmanager
    def envscope(self):
        """The MLflow and deploy environment for the stages (and the child processes), restored afterwards: a library caller or a test must not inherit a dry run's tmp paths."""
        old = {k: os.environ.get(k) for k in self.env}; os.environ.update(self.env)
        try: yield
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

    def sh(self, argv, log=None): return self.sys.run([str(x) for x in argv], log)

    # -- state
    def load_state(self):
        if os.path.exists(self.state_path):
            try: return json.load(open(self.state_path))
            except ValueError: raise StageError(f"{self.state_path} is not valid JSON; fix or delete it") from None
        return {"name": self.a.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run": bool(self.a.dry_run), "ids": {}, "stages": {s: {"status": "pending"} for s in STAGES}}

    def save(self):
        tmp = self.state_path + ".tmp"; json.dump(self.state, open(tmp, "w"), indent=1, default=str); os.replace(tmp, self.state_path)

    def st(self, s): return self.state["stages"][s]
    def res(self, s):
        if self.st(s).get("status") != "done": raise StageError(f"stage `{s}` has not finished; run it first (learn_loop.py {s} --name {self.a.name})")
        return self.st(s)["result"]

    # -- MLflow (LabTrack: a missing server never stops a stage)
    def lt(self):
        if self._lt is None: self._lt = mod("jetson/labtrack.py", "labtrack_for_learn_loop")
        return self._lt
    def track(self, exp):
        if exp not in self._tracks: self._tracks[exp] = self.lt().LabTrack(exp, uri=self.uri)
        return self._tracks[exp]
    def parent(self):
        """The loop's parent run in so101-learn-loop (created once, id kept in state.json)."""
        t = self.track(EXP_LOOP + self.sfx); rid = self.state["ids"].get("parent")
        if rid is None:
            r = t.start(f"learnloop-{self.a.name}", tags={"kind": "learn-loop", "name": self.a.name, "dry_run": int(bool(self.a.dry_run))}); rid = r.id
            if rid: self.state["ids"]["parent"] = rid; self.save()
            r.params({"name": self.a.name, "cycles": self.a.cycles, "trials": self.a.trials, "trial_s": self.a.trial_s, "policy": self.a.policy, "steps": self.a.steps, "batch": self.a.batch, "device": self.a.device})
            return r
        return t.attach(rid)

    def mlflow_client(self):
        try:
            import mlflow; return mlflow.MlflowClient(tracking_uri=self.uri or os.environ.get("MLFLOW_TRACKING_URI") or self.lt().resolve_uri())
        except Exception: return None

    def find_run(self, exp, filt):
        c = self.mlflow_client()
        try:
            e = c.get_experiment_by_name(exp)
            rs = c.search_runs([e.experiment_id], filt, max_results=1, order_by=["attributes.start_time DESC"]) if e else []
            return rs[0].info.run_id if rs else None
        except Exception: return None

    # -- driver
    def run_stage(self, s):
        st = self.st(s)
        if st.get("status") == "done" and not self.a.force: self.out(f"[{s}] done earlier ({st.get('finished', '?')}); skipping (--force to redo)"); return 0
        st.update(status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), error=None); self.save(); self.out(f"[{s}] start")
        try: result = getattr(self, "stage_" + s)()
        except StageError as e: st.update(status="failed", error=str(e), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")); self.save(); self.out(f"[{s}] FAILED: {e}"); self.tag_parent(failed_stage=s); return e.code
        except KeyboardInterrupt: st.update(status="failed", error="interrupted"); self.save(); self.out(f"[{s}] interrupted; repeat the command to resume"); return 1
        st.update(status="done", result=result, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), error=None)
        for later in STAGES[STAGES.index(s) + 1:]:   # why: a re-run record/train/... makes everything after it stale
            if self.st(later).get("status") != "pending": self.state["stages"][later] = {"status": "pending", "invalidated_by": s}
        self.save(); self.out(f"[{s}] done"); return 0

    def tag_parent(self, **kw):
        try: self.parent().tags(kw)
        except Exception: pass

    def run(self, stages):
        for s in stages:
            rc = self.run_stage(s)
            if rc: return rc
        return 0

    # ---------------------------------------------------------------- stages
    def stage_record(self):
        a = self.a; demo_root = os.path.join(self.dir, "demos"); name = a.repo.split("/")[-1]; ds_root = os.path.join(demo_root, name); out = os.path.join(self.dir, "autoloop"); session = a.session or f"ll-{a.name}"; rc = 0
        dr = mod("jetson/demo_recorder.py", "demo_recorder_for_learn_loop")
        if a.adopt_dataset: ds_root = os.path.abspath(os.path.expanduser(a.adopt_dataset)); summary = {}
        else:
            argv = [sys.executable, os.path.join(TOOLS, "autoloop.py"), "--cycles", a.cycles, "--record-demos", "--demo-root", demo_root, "--demo-repo", a.repo, "--session", session, "--out-dir", out, "--radius-mm", a.radius_mm]
            if a.dry_run: argv += ["--dry-run", "--settle", "0", "--seed", a.seed]
            if a.stop_file: argv += ["--stop-file", a.stop_file]
            argv += list(a.autoloop_arg or []); os.makedirs(out, exist_ok=True)
            rc = self.sh(argv, os.path.join(self.dir, "record.log"))
            if rc not in (0, 2): raise StageError(f"autoloop exited {rc} (see record.log); repeat the command to record again")
            try: summary = json.load(open(os.path.join(out, f"autoloop_{session}.json")))
            except (OSError, ValueError) as e: raise StageError(f"autoloop wrote no session summary: {e}") from e
            if a.pull_from: self.pull(a.pull_from, ds_root)
        info_p = os.path.join(ds_root, "meta", "info.json")
        if not os.path.exists(info_p): raise StageError(f"no dataset at {ds_root} (meta/info.json missing); with the recorder on the Jetson use --pull-from")
        info = json.load(open(info_p)); h = dr.dataset_hash(ds_root); demos = summary.get("demos") or {}; kept = int(info.get("total_episodes") or 0)
        if demos.get("dataset_hash") and demos["dataset_hash"] != h: raise StageError(f"dataset hash changed between the recorder ({demos['dataset_hash'][:12]}) and now ({h[:12]}): {ds_root} was modified")
        if kept < a.min_kept: raise StageError(f"only {kept} kept episodes in {ds_root}, need --min-kept {a.min_kept}; repeat `record` (--force) to add more, the dataset is appended")
        base = {k: summary.get(k) for k in ("session", "cycles", "landed", "success_rate", "mean_err_mm", "median_err_mm", "end", "duration_s")}
        res = {"dataset_root": ds_root, "repo_id": a.repo, "dataset_hash": h, "episodes": kept, "frames": info.get("total_frames"), "fps": info.get("fps"), "rejected": demos.get("rejected"), "autoloop_rc": rc, "baseline": base,
               "dataset_run": self.find_run(EXP_DATASETS + ("-dry" if (summary.get("dry_run") or a.dry_run) else ""), f"params.dataset_hash = '{h}'"), "autoloop_summary": os.path.join(out, f"autoloop_{session}.json") if not a.adopt_dataset else None}
        p = self.parent(); p.params({"dataset_hash": h, "dataset_repo": a.repo, "record_session": base.get("session") or session}); p.metrics({"episodes_kept": kept, "scripted_success_rate": base.get("success_rate"), "scripted_median_err_mm": base.get("median_err_mm")})
        return res

    def pull(self, src, dst):
        r = subprocess.run(["rsync", "-a", "--delete", src.rstrip("/") + "/", dst.rstrip("/") + "/"], capture_output=True, text=True, timeout=3600)
        if r.returncode: raise StageError(f"rsync {src}: {r.stderr.strip()[-300:]}")

    def stage_train(self):
        a = self.a; rec = self.res("record"); info = json.load(open(os.path.join(rec["dataset_root"], "meta", "info.json"))); cfg = train_cfg(info, a)
        job = a.job or f"act_{a.name}"; out = os.path.join(self.dir, "train", job); log = os.path.join(self.dir, "train.log"); last = os.path.join(out, "checkpoints", "last"); steps_done = 0
        if a.force and os.path.exists(out): shutil.rmtree(out); self.state['ids'].pop('train_run', None)   # why: --force on train means train again from scratch (a new MLflow run too), not resume
        done_marker = os.path.join(out, "checkpoints", f"{cfg['steps']:06d}")
        resume = os.path.islink(last) and not os.path.exists(done_marker)
        if os.path.exists(out) and not os.path.islink(last): shutil.rmtree(out)   # why: lerobot-train refuses an existing output_dir without --resume; a half-made dir with no checkpoint has nothing to resume
        argv = resume_argv(out) if resume else train_argv(cfg, rec["repo_id"], rec["dataset_root"], out, job, a)
        t = self.track(EXP_TRAIN + self.sfx); rid = self.state["ids"].get("train_run"); parent = self.parent()
        if rid: run = t.attach(rid)
        else:
            run = t.start(job, tags={"host": "mac" if cfg["device"] == "mps" else "jetson" if cfg["device"] == "cuda" else "other", "family": "nutpick", "source": "learn-loop", "kind": "train", "dataset_hash": rec["dataset_hash"], "dataset_repo": rec["repo_id"],
                                     "dataset_run_id": rec.get("dataset_run") or "", "learn_loop": a.name, "learn_loop_run": getattr(parent, "id", "") or "", "record_session": (rec.get("baseline") or {}).get("session") or ""})
            if run.id: self.state["ids"]["train_run"] = run.id; self.save()
            run.params({"policy": cfg["policy"], "chunk_size": cfg["chunk_size"], "n_action_steps": cfg["chunk_size"], "batch_size": cfg["batch_size"], "steps": cfg["steps"], "device": cfg["device"], "dataset": rec["repo_id"], "dataset_hash": rec["dataset_hash"],
                        "episodes": cfg["episodes"], "frames": cfg["frames"], "fps": cfg["fps"], "images": ",".join(cfg["images"]), "dropped_features": ",".join(cfg["dropped_features"]), "warm_start": "scratch"})
        os.makedirs(os.path.dirname(out), exist_ok=True); t0 = time.time(); self.out(f"[train] {'RESUME' if resume else 'lerobot-train'} {cfg['policy']} {cfg['steps']} steps batch {cfg['batch_size']} on {cfg['device']}: eta {fmt_s(estimates(a)['train'])}")
        rc = 0 if os.path.exists(done_marker) else self.sh(argv, log); wall = time.time() - t0   # why: training finished but the stage did not get to record it; nothing to run
        ck = os.path.join(last, "pretrained_model"); steps_done = int(os.path.basename(os.path.realpath(last))) if os.path.islink(last) else 0
        pts, rate = parse_log(open(log).read()) if os.path.exists(log) else ([], None)
        if pts:
            keep = pts[:: max(1, len(pts) // 200)]
            for s, l in keep: run.metrics({"loss": l}, step=s)
            run.metrics({"final_loss": pts[-1][1]}, step=pts[-1][0])
        if rc != 0 or steps_done < cfg["steps"]:
            run.end("FAILED"); self.state["ids"]["train_run_failed"] = run.id; self.save()
            raise StageError(f"lerobot-train exit {rc}, reached step {steps_done} of {cfg['steps']} (see train.log); repeat the command to resume from the last checkpoint" if steps_done else f"lerobot-train exit {rc} before any checkpoint (see train.log)")
        run.metrics({"steps_done": steps_done, "wall_s": wall, "steps_per_s": rate if rate is not None else (steps_done / wall if wall > 0 else None)}); run.json({"argv": [str(x) for x in argv], "cfg": cfg}, "learn_loop_train.json")
        tc = os.path.join(ck, "train_config.json")
        if os.path.exists(tc): run.artifact(tc, "config")
        run.end("FINISHED"); self.state["ids"].pop("train_run_failed", None)
        return {"checkpoint": os.path.realpath(ck), "job": job, "steps": steps_done, "final_loss": pts[-1][1] if pts else None, "wall_s": round(wall, 1), "steps_per_s": rate, "device": cfg["device"], "train_run": self.state["ids"].get("train_run"), "images": cfg["images"],
                "chunk_size": cfg["chunk_size"], "policy": cfg["policy"], "resumed": resume, "cfg": {k: v for k, v in cfg.items() if k != "input_features"}}

    def policy_name(self): return (self.a.policy_name or "so101-nutpick") + self.sfx

    def stage_register(self):
        a = self.a; rec = self.res("record"); tr = self.res("train"); tool = mod("tools/so101_policy.py", "so101_policy_for_learn_loop"); name = self.policy_name()
        _, c = tool.client(); sha = tool.sha256_of(os.path.join(tr["checkpoint"], "model.safetensors")); mv = None
        try: mv = next((m for m in tool.versions_of(c, name) if (m.tags or {}).get("sha256") == sha and (m.tags or {}).get("dataset_hash") == rec["dataset_hash"]), None)   # why: a crash between register and state.json must not make a second version of the same bytes
        except tool.Fail: mv = None
        if mv is None:
            buf = io.StringIO(); err = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err): code = tool.main(["register", tr["checkpoint"], "--name", name, "--dataset", rec["repo_id"], "--note", f"learn-loop {a.name}: {rec['episodes']} kept episodes, {tr['steps']} steps, dataset {rec['dataset_hash'][:12]}"])
            if code: raise StageError(f"so101-policy register: {(err.getvalue() or buf.getvalue()).strip()[-300:]}")
            m = re.search(r"registered (\S+) v(\d+)\s+run (\w+)", buf.getvalue())
            if not m: raise StageError(f"could not read the registry answer: {buf.getvalue()[-200:]}")
            ver, reg_run = m.group(2), m.group(3)
        else: ver, reg_run = str(mv.version), mv.run_id
        parent = self.parent(); tags = {"dataset_hash": rec["dataset_hash"], "train_run_id": tr.get("train_run") or "", "learn_loop": a.name, "learn_loop_run": getattr(parent, "id", "") or "", "record_session": (rec.get("baseline") or {}).get("session") or "", "dataset_run_id": rec.get("dataset_run") or ""}
        for k, v in tags.items():
            if v: c.set_model_version_tag(name, ver, k, v); c.set_tag(reg_run, k, v)
        metrics = {"final_loss": tr.get("final_loss"), "steps": tr.get("steps"), "train_wall_s": tr.get("wall_s"), "episodes": rec["episodes"], "frames": rec.get("frames"), "scripted_success_rate": (rec.get("baseline") or {}).get("success_rate")}
        for k, v in metrics.items():
            if v is not None: c.log_metric(reg_run, k, float(v))
        if tr.get("train_run"): self.track(EXP_TRAIN + self.sfx).attach(tr["train_run"]).tags({"registry_name": name, "registry_version": ver, "registry_run_id": reg_run})
        parent.params({"policy_name": name, "policy_version": ver})
        return {"name": name, "version": ver, "registry_run": reg_run, "sha256": sha, "tags": tags}

    def stage_deploy(self):
        a = self.a; reg = self.res("register"); tool = mod("tools/so101_policy.py", "so101_policy_for_learn_loop"); host = a.host; spec = f"{reg['name']}@{reg['version']}"
        for argv in (["deploy", spec, "--host", host], ["status", reg["name"], "--host", host]):
            buf = io.StringIO(); err = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err): code = tool.main(argv)
            self.out("  " + (buf.getvalue() + err.getvalue()).strip().replace("\n", "\n  "))
            if code: raise StageError(f"so101-policy {argv[0]} exited {code}: {(err.getvalue() or buf.getvalue()).strip()[-300:]}")
        return {"spec": spec, "host": host, "path": f"~/so101/policies/{reg['name']}/current", "local_stand_in": os.environ.get("SO101_DEPLOY_LOCAL_ROOT") if a.dry_run else None}

    def lineage(self):
        rec = self.res("record"); tr = self.res("train"); reg = self.res("register"); p = self.parent()
        return {"policy_name": reg["name"], "policy_version": reg["version"], "registry_run_id": reg["registry_run"], "train_run_id": tr.get("train_run") or "", "dataset_hash": rec["dataset_hash"], "dataset_repo": rec["repo_id"],
                "dataset_run_id": rec.get("dataset_run") or "", "learn_loop": self.a.name, "learn_loop_run": getattr(p, "id", "") or "", "record_session": (rec.get("baseline") or {}).get("session") or ""}

    def stage_evaluate(self):
        a = self.a; reg = self.res("register"); self.res("deploy"); tr = self.res("train"); ev = mod("tools/learn_loop_eval.py", "learn_loop_eval_for_run")
        rt = (self.runtime or (dry_runtime if a.dry_run else real_runtime))(self, ev)
        ev_path = os.path.join(self.dir, "eval.json"); prev = json.load(open(ev_path)) if os.path.exists(ev_path) else {"trials": []}
        script = ev.rollout_script(reg["name"], tr.get("policy", "act"), tr.get("chunk_size", 100), a.trial_s, tr.get("images") or list(ev.ALL_CAMS), max_rel=a.max_rel_deg)
        t = self.track(EXP_LOOP + self.sfx); lin = self.lineage(); rid = self.state["ids"].get("eval_run")
        if rid: erun = t.attach(rid)
        else:
            erun = t.start(f"eval-{a.name}", tags={"kind": "eval-session", "mlflow.parentRunId": lin["learn_loop_run"], **lin} if lin["learn_loop_run"] else {"kind": "eval-session", **lin})
            erun.params({"trials": a.trials, "trial_s": a.trial_s, "radius_mm": a.radius_mm, "max_rel_deg": a.max_rel_deg, "target": f"{a.target[0]:g},{a.target[1]:g}", "policy": f"{reg['name']}@{reg['version']}"})
            if erun.id: self.state["ids"]["eval_run"] = erun.id; self.save()
        pe = ev.PolicyEval(rt["rn"], rt["al"], rt["gr"], rt["transport"], script, a.trial_s, track=t, lineage=lin, settle_s=a.settle, fresh_s=a.fresh_s); pe.trials = list(prev["trials"])
        first = len(prev["trials"]) + 1
        if first > a.trials: self.out(f"[evaluate] {len(prev['trials'])} trials already done"); out = {"trials": pe.trials, "end": "trials", "stop": None}
        else:
            self.out(f"[evaluate] trials {first}..{a.trials}, {a.trial_s:g} s each, eta {fmt_s((a.trials - first + 1) * (a.trial_s + TRIAL_OVERHEAD_S))}; stop file {rt['guard'].lim.stop_file}")
            out = pe.run(first, a.trials, parent_id=erun.id, on_trial=lambda r: json.dump({"trials": pe.trials, "policy": f"{reg['name']}@{reg['version']}"}, open(ev_path, "w"), indent=1, default=str))
        summ = ev.summarize(out["trials"], a.radius_mm); summ.update(end=out["end"], stop=out["stop"], policy=f"{reg['name']}@{reg['version']}", session=a.name)
        json.dump({"trials": out["trials"], "summary": summ, "policy": summ["policy"]}, open(ev_path, "w"), indent=1, default=str)
        erun.metrics({k: summ[k] for k in ("trials", "landed", "picked", "success_rate", "mean_err_mm", "median_err_mm", "median_placed_err_mm") if summ.get(k) is not None}); erun.json(summ, "eval_summary.json")
        if out["end"] != "trials":
            erun.tags({"end": out["end"]}); raise StageError(f"evaluation ended on a trip after {summ['trials']} of {a.trials} trials: {out['end']} {json.dumps((out['stop'] or {}).get('detail'))[:200]}; fix it and repeat the command to continue", code=2)
        erun.end("FINISHED"); self.state["ids"].pop("eval_run", None)
        return summ

    def baseline_pool(self, exclude_session=None):
        """Pooled success over the last K real so101-autoloop sessions in MLflow (dry runs have their own -dry experiment): {"sessions", "cycles", "landed", "success_rate"} or None."""
        c = self.mlflow_client()
        if c is None: return None
        try:
            e = c.get_experiment_by_name(EXP_AUTOLOOP + self.sfx)
            if not e: return None
            rs = c.search_runs([e.experiment_id], "tags.kind = 'session'", max_results=200, order_by=["attributes.start_time DESC"]); cyc = land = n = 0
            for r in rs:
                sess = r.data.tags.get("session")
                if sess == exclude_session or r.data.params.get("dry_run", "0") == "1" and not self.a.dry_run: continue
                k = r.data.metrics.get("cycles"); sr = r.data.metrics.get("success_rate")
                if not k or sr is None: continue
                cyc += int(k); land += int(round(sr * k)); n += 1
                if n >= self.a.baseline_sessions: break
            return {"sessions": n, "cycles": cyc, "landed": land, "success_rate": round(land / cyc, 3) if cyc else None} if cyc else None
        except Exception: return None

    def stage_report(self):
        a = self.a; ev = mod("tools/learn_loop_eval.py", "learn_loop_eval_for_report")
        if self.st("evaluate").get("status") != "done" and not a.partial: raise StageError("evaluate has not finished; run it, or `report --partial` for what exists")
        p = os.path.join(self.dir, "eval.json")
        if not os.path.exists(p): raise StageError("no eval.json; run evaluate first")
        e = json.load(open(p)); summ = ev.summarize(e["trials"], a.radius_mm); rec = self.res("record"); reg = self.res("register"); base = rec.get("baseline") or {}
        sess_n = int(base.get("cycles") or 0); sess_k = int(base.get("landed") or 0); lo, hi = ev.wilson(sess_k, sess_n)
        pool = self.baseline_pool(base.get("session")); plo, phi = ev.wilson(pool["landed"], pool["cycles"]) if pool else (None, None)
        def verdict(b_rate, blo, bhi):
            if b_rate is None or summ["success_rate"] is None: return "no baseline to compare with"
            if summ["ci95"][0] > bhi: return "policy better (intervals do not overlap)"
            if summ["ci95"][1] < blo: return "policy worse (intervals do not overlap)"
            return f"not distinguishable at {summ['trials']} trials against {sess_n if b_rate == base.get('success_rate') else pool['cycles']} baseline cycles"
        rpt = {"name": a.name, "policy": f"{reg['name']}@{reg['version']}", "dataset_hash": rec["dataset_hash"], "episodes": rec["episodes"], "policy_eval": summ, "partial": self.st("evaluate").get("status") != "done",
               "baseline_session": {"session": base.get("session"), "cycles": sess_n, "landed": sess_k, "success_rate": base.get("success_rate"), "ci95": [lo, hi], "median_err_mm": base.get("median_err_mm"), "verdict": verdict(base.get("success_rate"), lo, hi) if sess_n else "no baseline"},
               "baseline_pooled": ({**pool, "ci95": [plo, phi], "verdict": verdict(pool["success_rate"], plo, phi)} if pool else None), "stops": (e.get("summary") or {}).get("stop")}
        json.dump(rpt, open(os.path.join(self.dir, "report.json"), "w"), indent=1)
        md = render_report(rpt); open(os.path.join(self.dir, "report.md"), "w").write(md); self.out(md)
        p = self.parent(); p.metrics({"eval_success_rate": summ["success_rate"], "eval_median_err_mm": summ["median_err_mm"], "scripted_success_rate": base.get("success_rate"), "pooled_scripted_success_rate": pool["success_rate"] if pool else None,
                                      "delta_vs_session": (summ["success_rate"] - base["success_rate"]) if summ["success_rate"] is not None and base.get("success_rate") is not None else None})
        p.json(rpt, "report.json"); p.file("report.md", md)
        if not rpt["partial"]: p.end("FINISHED")
        return rpt


def pct(v): return "n/a" if v is None else f"{100 * v:.0f}%"
def mm(v): return "n/a" if v is None else f"{v:.1f} mm"


def render_report(r):
    pe = r["policy_eval"]; bs = r["baseline_session"]; bp = r.get("baseline_pooled")
    rows = [("policy " + r["policy"], f"{pe['landed']}/{pe['trials']}", pct(pe["success_rate"]), f"{pct(pe['ci95'][0])}-{pct(pe['ci95'][1])}", mm(pe["median_err_mm"]), mm(pe["median_placed_err_mm"]))]
    rows.append((f"scripted, this loop's session {bs.get('session') or ''}".strip(), f"{bs['landed']}/{bs['cycles']}", pct(bs["success_rate"]), f"{pct(bs['ci95'][0])}-{pct(bs['ci95'][1])}", mm(bs.get("median_err_mm")), "n/a"))
    if bp: rows.append((f"scripted, pooled last {bp['sessions']} sessions", f"{bp['landed']}/{bp['cycles']}", pct(bp["success_rate"]), f"{pct(bp['ci95'][0])}-{pct(bp['ci95'][1])}", "n/a", "n/a"))
    head = ("", "landed", "success", "95% CI", "median err", "median err (landed)")
    lines = [f"# learn-loop {r['name']}" + (" (PARTIAL)" if r.get("partial") else ""), "", f"dataset {r['dataset_hash'][:12]}, {r['episodes']} kept episodes; policy {r['policy']}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += ["", f"vs this session: {bs['verdict']}"] + ([f"vs pooled: {bp['verdict']}"] if bp else []) + ([f"stopped: {json.dumps(r['stops'])[:200]}"] if r.get("stops") else [])
    lines += ["", "The scripted pick gets one retry per cycle; the policy gets none. Success = the judge's `picked and landed within the radius`, the same rule for both."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- runtimes for evaluate
def eval_cfg(a, out_dir, session):
    return argparse.Namespace(cycles=a.trials, minutes=None, target=tuple(a.target), random_target=False, safe_box=tuple(a.safe_box), radius_mm=a.radius_mm, session=session, dry_run=bool(a.dry_run), max_retries=0, settle_s=a.settle, moved_mm=20.0,
                              spot_mm=(40.0, 50.0, 60.0), out_dir=out_dir, judge_wait_s=a.judge_wait, save_frames=not a.dry_run, start_near=tuple(a.start_near) if a.start_near else None, grasp_roll="off")


def limits(gr, a):
    kw = {"max_failures": a.max_consecutive_failures, "lost_after": a.lost_after}
    if a.stop_file: kw["stop_file"] = a.stop_file
    return gr.Limits(**kw)


def common(L, ev, ctx, pl, cc, table, transport, track, rng):
    al = mod("tools/autoloop.py", "autoloop_for_learn_loop"); gr = al.load_guardrails(); a = L.a; out_dir = os.path.join(L.dir, "eval"); os.makedirs(out_dir, exist_ok=True)
    guard = gr.Guard(cc, table, limits(gr, a), ctx); cfg = eval_cfg(a, out_dir, f"eval-{a.name}")
    rn = al.Runner(pl, cc, table, guard, ctx, cfg, track, rng, out=lambda l: print(l, flush=True))
    if cfg.start_near: rn.cur_in = [float(v) for v in cfg.start_near]
    return {"rn": rn, "al": al, "gr": gr, "guard": guard, "transport": transport}


def real_runtime(L, ev):
    """The Mac side of a real evaluation, built like autoloop.main's non-dry branch: the dashboard for state and scene, ssh for the Jetson."""
    import urllib.request
    al = mod("tools/autoloop.py", "autoloop_for_learn_loop"); gr = al.load_guardrails(); pl = al.load_pick_loop(); cc = pl.cam_calib(); a = L.a
    table = json.load(open(a.table or os.path.join(JETSON, "calib", "top_table.json")))
    if not (table.get("arm_fit") and table.get("arm_base_mm")): raise StageError("evaluate needs the arm fit in top_table.json (tools/jog_capture.py --fit)")
    get = lambda path, t: json.load(urllib.request.urlopen(pl.DASH + path, timeout=t))
    ctx = gr.Ctx(lambda: get("/api/state", gr.Limits().panel_s), lambda: get("/api/scene", 5), time.time, time.sleep, lambda d: print(json.dumps(d), flush=True), arm_jobs=lambda: al.jetson_arm_jobs(a.host))
    return common(L, ev, ctx, pl, cc, table, L.transport or ev.SshTransport(a.host), L.track(EXP_LOOP + L.sfx), random.Random(a.seed))


def dry_runtime(L, ev):
    """The same on fakes: virtual clock, FakeWorld + DryArm from autoloop.py, FakeRollout for the Jetson (policy success probability --dry-policy-p)."""
    ff = mod("tools/learn_loop_fakes.py", "learn_loop_fakes_for_run"); al = mod("tools/autoloop.py", "autoloop_for_learn_loop"); gr = al.load_guardrails(); pl = al.load_pick_loop(); cc = pl.cam_calib(); a = L.a
    tp = a.table or os.path.join(JETSON, "calib", "top_table.json"); tp = tp if os.path.exists(tp) else os.path.join(ROOT, "tests", "calib", "fixtures", "top_table_real.json"); table = json.load(open(tp)); rng = random.Random(a.seed)
    clock = ff.Clock(); world = al.FakeWorld(cc, table, gr.fk_mat_mm, nut_in=(6.0, 8.0), rng=rng); arm = al.DryArm(cc, table, world)
    pl.api, pl.run_plan, pl.find_nut, pl.fetch_top, pl.fetch_wrist = arm.api, arm.run_plan, world.find_nut, (lambda: b""), (lambda: None); pl.fetch_scene_state = lambda: (world.api_scene()["objects"], 0)
    ctx = gr.Ctx(world.api_state, world.api_scene, clock.now, clock.sleep, lambda d: print(json.dumps(d), flush=True))
    transport = L.transport or ff.FakeRollout(world, clock, a.target, p=a.dry_policy_p, rng=random.Random(a.seed + 1), script=tuple(a.dry_script or ()))
    return common(L, ev, ctx, pl, cc, table, transport, L.track(EXP_LOOP + L.sfx), rng)


# ---------------------------------------------------------------- command line
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="+", choices=("plan", "status", "all") + STAGES, help="one or more stages in order, or `all` (resumes at the first one not done)"); ap.add_argument("--name", required=True, help="the loop's name; its state is <out-dir>/<name>/state.json")
    ap.add_argument("--dry-real-record", action="store_true", help="with --dry-run: run the real autoloop --dry-run --record-demos for `record` (slower, a real LeRobot dataset)")
    ap.add_argument("--dry-run", action="store_true", help="fakes for the arm, the Jetson, lerobot-train and the dashboard; MLflow in <run dir>/mlflow.db unless --mlflow-uri"); ap.add_argument("--force", action="store_true", help="re-run the named stages even if done (the stages after them are invalidated)")
    ap.add_argument("--out-dir", default=None, help="default <repo>/outputs/learn_loop"); ap.add_argument("--mlflow-uri", default=None, help="default: $SO101_MLFLOW_URI / $MLFLOW_TRACKING_URI / the Mac's sqlite store; --dry-run uses a private one"); ap.add_argument("--seed", type=int, default=7)
    g = ap.add_argument_group("record"); g.add_argument("--cycles", type=int, default=30, help="autoloop cycles; each TRY the keep rule accepts becomes an episode"); g.add_argument("--repo", default="lastlight/so101-nut-pick"); g.add_argument("--session", default=None)
    g.add_argument("--min-kept", type=int, default=None, help="fail the record stage under this many kept episodes (default 10; 2 with --dry-run)"); g.add_argument("--adopt-dataset", default=None, metavar="PATH", help="use an existing dataset folder instead of recording")
    g.add_argument("--pull-from", default=None, metavar="HOST:PATH", help="rsync the dataset from here after recording (the recorder on the Jetson, phase 2)"); g.add_argument("--autoloop-arg", action="append", default=[], help="extra argument passed to autoloop.py (repeatable), e.g. --autoloop-arg=--random-target")
    g = ap.add_argument_group("train"); g.add_argument("--policy", choices=("act", "smolvla"), default="act"); g.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"), help="mps = the Mac (default, see docs/learn-loop.md); cuda = a CUDA box (SmolVLA needs one)")
    g.add_argument("--steps", type=int, default=20000); g.add_argument("--batch", type=int, default=8); g.add_argument("--chunk", type=int, default=100); g.add_argument("--workers", type=int, default=2); g.add_argument("--save-freq", type=int, default=2500); g.add_argument("--job", default=None)
    g = ap.add_argument_group("registry / deploy"); g.add_argument("--policy-name", default=None, help="registered model name (default so101-nutpick)"); g.add_argument("--host", default=os.environ.get("SO101_JETSON", "jetson"))
    g = ap.add_argument_group("evaluate"); g.add_argument("--trials", type=int, default=10); g.add_argument("--trial-s", type=float, default=25.0, help="seconds the policy may run per trial (the recorded episodes are ~15-25 s)"); g.add_argument("--target", type=float, nargs=2, default=[9.0, 6.0], metavar=("X", "Y"))
    g.add_argument("--safe-box", type=float, nargs=4, default=[6.0, 4.0, 13.0, 8.0], metavar=("X0", "Y0", "X1", "Y1")); g.add_argument("--radius-mm", type=float, default=15.0); g.add_argument("--max-rel-deg", type=float, default=2.0, help="lerobot --robot.max_relative_target, degrees per 30 Hz tick (the speed cap)")
    g.add_argument("--max-consecutive-failures", type=int, default=3); g.add_argument("--lost-after", type=int, default=3); g.add_argument("--stop-file", default=None, help="default ~/so101/outputs/run/autoloop.stop, the autoloop's"); g.add_argument("--settle", type=float, default=3.0)
    g.add_argument("--judge-wait", type=float, default=8.0); g.add_argument("--fresh-s", type=float, default=20.0, help="how long to wait for the cameras to come back after the rollout"); g.add_argument("--start-near", type=float, nargs=2, default=None, metavar=("X", "Y"))
    g.add_argument("--table", default=None); g.add_argument("--dry-policy-p", type=float, default=0.7, help="--dry-run: chance the fake policy places the nut"); g.add_argument("--dry-script", nargs="*", default=None, help="--dry-run: outcome per trial (ok miss drop escape die fail hang)")
    g = ap.add_argument_group("report"); g.add_argument("--baseline-sessions", type=int, default=5, help="how many recent real autoloop sessions the pooled baseline covers"); g.add_argument("--partial", action="store_true", help="report on an unfinished evaluation")
    return ap


def expand(stages):
    out = []
    for s in stages:
        out += list(STAGES) if s == "all" else [s]
    return out


def show_plan(L):
    a = L.a; est = estimates(a); tot = 0.0; L.out(f"loop {a.name}  ({'dry run' if a.dry_run else 'REAL hardware'})  state {L.state_path}")
    for s in STAGES:
        st = L.st(s); done = st.get("status") == "done"; tot += 0 if done else est[s]
        L.out(f"  {s:9s} {st.get('status', 'pending'):8s} eta {fmt_s(est[s]):>7s}  {st.get('error') or ''}")
    L.out(f"  remaining about {fmt_s(tot)}; train on {a.device} {a.steps} steps batch {a.batch} ({RATE.get(a.device, RATE['mps'])[0]} step/s at batch {RATE.get(a.device, RATE['mps'])[1]}); evaluate {a.trials} x {a.trial_s:g} s + {TRIAL_OVERHEAD_S:g} s overhead")


def main(argv=None, sys_=None, transport=None, runtime=None):
    a = build_parser().parse_args(argv)
    if a.min_kept is None: a.min_kept = 2 if a.dry_run else 10
    try: L = Loop(a, sys_=sys_, transport=transport, runtime=runtime)
    except StageError as e: print(f"learn_loop: {e}", file=sys.stderr); return 1
    stages = expand(a.stage)
    if "plan" in stages or "status" in stages: show_plan(L); return 0
    with L.envscope(): return L.run(stages)


if __name__ == "__main__": sys.exit(main())
