#!/usr/bin/env python3
"""learn_loop_fakes.py - everything `tools/learn_loop.py --dry-run` and tests/tools/test_learn_loop*.py stand in for the hardware with: a tiny LeRobot-shaped dataset,
a fake `autoloop --record-demos` and `lerobot-train` (FakeSys), a tiny checkpoint, and a fake Jetson rollout (FakeRollout) that moves the FakeWorld's nut and walks the
follower's read-back joints along a path the RolloutWatch can check. Nothing here opens a camera, a port, an ssh session or a GPU."""
import hashlib, json, os, random

IN = 25.4
HOVER = {"shoulder_pan": 15.0, "shoulder_lift": 17.0, "elbow_flex": 2.0, "wrist_flex": 80.0, "wrist_roll": 0.46}    # tip 11.2, 6.4 in, 39 mm up: inside the safe mat (fixture table)
ESCAPE = {"shoulder_pan": 90.0, "shoulder_lift": 17.0, "elbow_flex": 2.0, "wrist_flex": 80.0, "wrist_roll": 0.5}     # tip 17.4, -1.6 in: off the mat
FEATURES = {"action": {"dtype": "float32", "shape": [6], "names": [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]},
            "observation.state": {"dtype": "float32", "shape": [6], "names": [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]},
            "observation.environment_state": {"dtype": "float32", "shape": [5], "names": None},
            "observation.images.top": {"dtype": "video", "shape": [360, 640, 3], "names": ["height", "width", "channels"]},
            "observation.images.front": {"dtype": "video", "shape": [360, 640, 3], "names": ["height", "width", "channels"]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None}, "frame_index": {"dtype": "int64", "shape": [1], "names": None}, "episode_index": {"dtype": "int64", "shape": [1], "names": None}}


def make_dataset(root, episodes=3, fps=30, features=None, frames_per_ep=60):
    """A LeRobot-v3-shaped folder (meta/info.json, one data file, the lab sidecar). Enough for learn_loop's hash and feature reading; lerobot-train would not accept it."""
    os.makedirs(os.path.join(root, "meta", "lab"), exist_ok=True); os.makedirs(os.path.join(root, "data", "chunk-000"), exist_ok=True)
    json.dump({"codebase_version": "v3.0", "robot_type": "so_follower", "total_episodes": episodes, "total_frames": episodes * frames_per_ep, "fps": fps, "features": features or FEATURES}, open(os.path.join(root, "meta", "info.json"), "w"), indent=1)
    open(os.path.join(root, "data", "chunk-000", "file-000.parquet"), "wb").write(hashlib.sha256(f"{episodes}".encode()).digest() * 8)
    with open(os.path.join(root, "meta", "lab", "episodes.jsonl"), "w") as f:
        for i in range(episodes): f.write(json.dumps({"episode_index": i, "n_frames": frames_per_ep}) + "\n")
    return root


def make_ckpt(d, steps=1000, policy="act", chunk=100, images=("front", "top", "wrist"), dataset="lab/nut", seed=1):
    """A pretrained_model dir the registry accepts (train_config.json, config.json, model.safetensors) plus a processor json."""
    os.makedirs(d, exist_ok=True); feats = {"observation.state": {"type": "STATE", "shape": [6]}, **{f"observation.images.{c}": {"type": "VISUAL", "shape": [3, 360, 640]} for c in images}}
    json.dump({"type": policy, "chunk_size": chunk, "n_action_steps": chunk, "input_features": feats, "output_features": {"action": {"type": "ACTION", "shape": [6]}}}, open(os.path.join(d, "config.json"), "w"))
    json.dump({"policy": {"type": policy, "chunk_size": chunk, "n_action_steps": chunk, "optimizer_lr": 1e-5, "dim_model": 512, "use_vae": True, "device": "mps", "pretrained_path": None}, "batch_size": 8, "steps": steps, "seed": 1000, "dataset": {"repo_id": dataset}}, open(os.path.join(d, "train_config.json"), "w"))
    open(os.path.join(d, "model.safetensors"), "wb").write(bytes([(seed * 7 + i) % 256 for i in range(4096)])); json.dump({"steps": []}, open(os.path.join(d, "policy_preprocessor.json"), "w"))
    return d


def argval(argv, flag, default=None):
    for a in argv:
        if a.startswith(flag + "="): return a.split("=", 1)[1]
    for i, a in enumerate(argv[:-1]):
        if a == flag: return argv[i + 1]
    return default


class FakeSys:
    """Stands in for subprocess runs: `run(argv, log)` -> return code. Recognises tools/autoloop.py (writes a tiny kept-only dataset and the session summary) and
    lerobot-train (a tiny checkpoint, a log in LeRobot's format). `fail` = {"record": n, "train": n}: the next n calls of that kind exit 1 (train after writing one checkpoint)."""
    def __init__(self, kept=8, rejected=2, fail=None, success_rate=0.6, mlflow=None):
        self.kept, self.rejected, self.fail, self.sr, self.calls = kept, rejected, dict(fail or {}), success_rate, []

    def run(self, argv, log=None):
        self.calls.append(list(argv)); kind = "record" if any(str(a).endswith("autoloop.py") for a in argv) else "train" if any("lerobot-train" in str(a) for a in argv) else "other"
        if self.fail.get(kind, 0) > 0: self.fail[kind] -= 1; part = True
        else: part = False
        return getattr(self, "do_" + kind)(argv, log, part)

    def do_other(self, argv, log, part): return 0

    def do_record(self, argv, log, part):
        if part: return 1
        root = argval(argv, "--demo-root"); repo = argval(argv, "--demo-repo", "lab/nut"); out = argval(argv, "--out-dir"); session = argval(argv, "--session"); name = repo.split("/")[-1]
        make_dataset(os.path.join(root, name), episodes=self.kept); make_dataset(os.path.join(root, name + "-rejects"), episodes=self.rejected)
        cycles = int(argval(argv, "--cycles", 10)); os.makedirs(out, exist_ok=True)
        json.dump({"session": session, "cycles": cycles, "landed": int(round(self.sr * cycles)), "success_rate": self.sr, "mean_err_mm": 6.0, "median_err_mm": 5.5, "stops": [], "end": "cycles", "dry_run": "--dry-run" in argv,
                   "demos": {"kept": self.kept, "rejected": self.rejected, "aborted": 0, "repo_id": repo, "dataset_hash": None}}, open(os.path.join(out, f"autoloop_{session}.json"), "w"))
        if log: open(log, "a").write("fake autoloop done\n")
        self.log_dataset_run(os.path.join(root, name), repo)
        return 0

    def log_dataset_run(self, ds_root, repo):
        """What DemoSession.finish does at the end of a real session: a `dataset-version` run in so101-datasets-dry carrying the dataset hash (only when MLflow is configured)."""
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri: return
        import importlib.util
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        def ld(rel, nm):
            sp = importlib.util.spec_from_file_location(nm, os.path.join(here, rel)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
        h = ld("jetson/demo_recorder.py", "dr_for_fakes").dataset_hash(ds_root)
        with ld("jetson/labtrack.py", "lt_for_fakes").LabTrack("so101-datasets-dry", uri=uri).run(f"dataset-{repo.split('/')[-1]}-{h[:8]}", tags={"kind": "dataset-version"}) as r: r.params({"repo_id": repo, "dataset_hash": h})

    def do_train(self, argv, log, part):
        cp = argval(argv, "--config_path"); out = argval(argv, "--output_dir") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cp))))
        saved = json.load(open(cp)) if cp and os.path.exists(cp) else {}
        steps = int(argval(argv, "--steps", saved.get("steps", 1000))); save = int(argval(argv, "--save_freq", saved.get("_save_freq", 500))); chunk = int(argval(argv, "--policy.chunk_size", (saved.get("policy") or {}).get("chunk_size", 100)))
        feats = json.loads(argval(argv, "--policy.input_features", "{}") or "{}") or ({f"observation.images.{c}": 1 for c in saved.get("_images", [])}); images = [k.rsplit(".", 1)[-1] for k in feats if k.startswith("observation.images.")] or ["front", "top", "wrist"]
        policy = argval(argv, "--policy.type", "act"); ds = argval(argv, "--dataset.repo_id", "lab/nut")
        resume = argval(argv, "--resume") in ("true", "True"); start = 0
        if resume:
            last = os.path.join(out, "checkpoints", "last"); start = int(os.path.basename(os.path.realpath(last))) if os.path.islink(last) else 0
        lines = []
        for s in range(start + save, steps + 1, save):
            d = make_ckpt(os.path.join(out, "checkpoints", f"{s:06d}", "pretrained_model"), steps=steps, policy=policy, chunk=chunk, images=images, dataset=ds, seed=s % 5 + 1)
            tcp = os.path.join(d, "train_config.json"); tc = json.load(open(tcp)); tc.update(_save_freq=save, _images=images); json.dump(tc, open(tcp, "w"))
            lines.append(f"INFO 2026-11-11 10:00:00 ot_train.py:641 step:{s} smpl:{s * 8} ep:3 epch:0.22 loss:{max(0.05, 6.0 / (1 + s / 100)):.3f} grdn:10.0 lr:1.0e-05 updt_s:0.25 data_s:0.01 smp/s:30 l1_loss:0.1 kld_loss:0.001")
            link = os.path.join(out, "checkpoints", "last")
            if os.path.islink(link) or os.path.exists(link): os.remove(link)
            os.symlink(f"{s:06d}", link)
            if part: break   # a crash after the first checkpoint
        if log: open(log, "a").write("\n".join(lines) + "\n")
        return 1 if part else 0


class Clock:
    """Virtual time for dry runs and tests: advances only when something sleeps."""
    def __init__(self, t=1000.0): self.t = t
    def now(self): return self.t
    def sleep(self, s): self.t += s


class FakeRollout:
    """The Jetson side of a trial: transport.rollout(script, T, tick) / .stop(). Per rollout an outcome from `script` ("ok" | "miss" | "drop" | "escape" | "die" | "fail" | "mute" (no state published) | "hang"; default
    `default`, or a random "ok" with probability p): the nut moves in `world`, the follower's read-back joints walk HOVER (ESCAPE after `escape_at_s`), `tick(elapsed)` runs
    every poll_s on the virtual clock like the real transport polls. Keeps `scripts` (what would have been piped to ssh) and `stops`."""
    def __init__(self, world, clock, target_in, p=0.7, script=(), rng=None, poll_s=0.25, escape_at_s=3.0, noise_mm=3.0, on_tick=None):
        self.world, self.clock, self.target_mm = world, clock, [target_in[0] * IN, target_in[1] * IN]; self.p, self.script, self.rng, self.poll, self.escape_at = p, list(script), rng or random.Random(3), poll_s, escape_at_s
        self.noise, self.scripts, self.stops, self.n, self.history = noise_mm, [], 0, 0, []; self.ticks = 0; self.on_tick = on_tick   # on_tick(n_rollout, elapsed): a test's hook (e.g. touch the stop file)

    def stop(self): self.stops += 1; self.world.follower = {"pos": {}, "src": "busy", "t": self.clock.now()}

    def publish(self, joints): self.world.follower = {"pos": dict(joints), "src": "busy-tele", "mode": "deg", "t": self.clock.now()}

    def outcome(self):
        if self.n < len(self.script): return self.script[self.n]
        return "ok" if self.rng.random() < self.p else "miss"

    def rollout(self, script, duration_s, tick):
        self.scripts.append(script); want = self.outcome(); self.n += 1; el = 0.0; self.history.append(want)
        if want == "hang": self.clock.sleep(duration_s + 41.0); return {"rc": None, "elapsed_s": duration_s + 41.0, "trip": None, "overrun": True}
        while el < duration_s:
            self.clock.sleep(self.poll); el += self.poll; self.ticks += 1
            if want != "mute": self.publish(ESCAPE if (want == "escape" and el >= self.escape_at) else HOVER)
            if self.on_tick: self.on_tick(self.n, el)
            trip = tick(el)
            if trip is not None: self.stop(); return {"rc": None, "elapsed_s": round(el, 1), "trip": trip, "overrun": False}
            if want == "die" and el >= 2.0: return {"rc": 143, "elapsed_s": round(el, 1), "trip": None, "overrun": False}
            if want == "fail" and el >= 2.0: return {"rc": 1, "elapsed_s": round(el, 1), "trip": None, "overrun": False}
        px, py = self.target_mm
        if want == "ok": self.world.nut = [px + self.rng.gauss(0, self.noise / 2), py + self.rng.gauss(0, self.noise / 2)]
        elif want == "drop": self.world.nut = [px + 45.0, py + 30.0]
        self.world.follower = {"pos": {}, "src": "busy", "t": self.clock.now()}
        return {"rc": 0, "elapsed_s": round(el, 1), "trip": None, "overrun": False}
