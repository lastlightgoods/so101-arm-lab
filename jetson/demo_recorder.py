#!/usr/bin/env python3
"""demo_recorder.py - turns scripted picks into LeRobot v3.0 training episodes (docs/scripted-demos.md, plan docs 51/51A/51B).

PHASE 1: everything here runs against fakes. Nothing opens a camera, the servo port or the network.

    pick_seq.py --tick-sink  ->  /dev/shm/so101_ticks.jsonl  ->  TickBuffer   (per-tick action sent + state read before it, 40 Hz, time.monotonic())
    CameraSource.frame_at(cam, t)                                              (latest frame of each camera; FakeCameraSource in phase 1)
                   \\_______________ DemoRecorder: every 1/30 s boundary takes the newest tick and the newest frame of each camera ______/
                                      -> a spool on disk (raw frames) while the episode is open
                                      -> commit(keep)   LeRobotDataset (main repo)   + meta/lab/episodes.jsonl row
                                      -> commit(reject) LeRobotDataset (separate rejects repo), same features, same sidecar

The frame format is lerobot-record's own: features come from lerobot.utils.feature_utils.hw_to_dataset_features for a so_follower
(action and observation.state as `<motor>.pos`, observation.images.<cam>), plus observation.environment_state, constant within an episode.
LeRobot sets timestamp = frame_index / fps, so a slow session still "claims" 30 Hz: real sample times, tick ages and camera ages go to
meta/lab/frames/episode_NNNNNN.json, and an episode whose cadence or cameras misbehaved is reported unhealthy (the keep rule rejects it).

Idle gaps (the Mac thinking between two staged pick_seq runs) have no fresh tick (older than max_tick_age_s); those boundaries are DROPPED,
not recorded, so a policy does not learn to wait.

TODO(phase 2, hardware): a real CameraSource (lerobot OpenCVCamera async readers, opened ONCE per session so the locked exposure holds) that also
re-serves 15 fps JPEG to the dashboard, a resident process that calls pump() at 30 Hz and a control socket for open/end/commit. See docs/scripted-demos.md.
"""
import bisect, dataclasses, hashlib, importlib.util, json, math, os, shutil, signal, sys, tempfile, time
import numpy as np

FPS = 30
DEFAULT_VCODEC = "libsvtav1"   # lerobot-record's default; the real recorder on the Jetson (Linux) keeps it
TASK = "Pick up the nut and place it on the target"   # one fixed string; configuration never goes into the task (plan 4.2)
ROBOT_TYPE = "so_follower"
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAX_TICK_AGE_S = 0.060        # ticks come at ~40 Hz (25 ms apart plus the readback), so the plan's 25 ms would drop most frames; 60 ms is two ticks
CAM_FREEZE_S = 0.25           # a camera frame older than this at a sample = frozen (the threshold cam_health already uses)
OVER_BUDGET_MAX_PCT = 5.0     # reject an episode when more than this share of frames was sampled over one frame period late
BANDING_MAX = 8.0             # grey levels of row-mean spread on top/front (measured with the lock: 3.3 and 2.2); an own choice, see the doc
MIN_FRAMES = 30               # under one second is not an episode
BANDING_CAMS = ("top", "front")


def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def load_pick_seq(path=None):
    """jetson/pick_seq.py as a module (for ease_cmd and friends). It installs a SIGTERM handler at import; put the old one back."""
    old = signal.getsignal(signal.SIGTERM)
    try: return load_by_path(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_seq.py"), "pick_seq_for_demos")
    finally: signal.signal(signal.SIGTERM, old)


# ---------------------------------------------------------------- the camera source interface
class CameraSource:
    """What the recorder needs from the cameras. Phase 2 implements this on lerobot's OpenCVCamera async readers (one open for the whole session).
      names            -> ["top", "front", "wrist"]
      shapes           -> {"top": (360, 640, 3), ...}  (H, W, 3) uint8 RGB
      frame_at(cam, t) -> (frame, t_capture): the newest frame captured at or before monotonic time t, and when it was captured
      banding(cam)     -> row-mean spread in grey levels (docs/camera-settings.md) or None when not measured
    """
    names = ()
    shapes = {}
    def frame_at(self, cam, t): raise NotImplementedError   # TODO(phase 2): OpenCV tee
    def banding(self, cam): return None


class FakeCameraSource(CameraSource):
    """Synthetic cameras: a gradient whose phase moves with the capture time, captured at `cam_fps` with `latency_s` delay. `freeze` = [(cam, t0, t1)]
    holds the capture time of that camera at t0 during [t0, t1) (a frozen stream); `band` = {cam: banding value}."""
    def __init__(self, shapes=None, cam_fps=30.0, latency_s=0.010, freeze=(), band=None):
        self.shapes = dict(shapes or {"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}); self.names = tuple(self.shapes)
        self.cam_fps, self.latency, self.freeze, self.band = cam_fps, latency_s, list(freeze), dict(band or {})

    def frame_at(self, cam, t):
        tc = math.floor((t - self.latency) * self.cam_fps) / self.cam_fps
        for c, t0, t1 in self.freeze:
            if c == cam and t0 <= t < t1: tc = min(tc, math.floor((t0 - self.latency) * self.cam_fps) / self.cam_fps)
        h, w, _ = self.shapes[cam]; k = int(round(tc * self.cam_fps)) % 256; img = np.empty((h, w, 3), np.uint8)
        img[..., 0] = (np.arange(w)[None, :] * 255 // max(w - 1, 1) + k) % 256; img[..., 1] = (np.arange(h)[:, None] * 255 // max(h - 1, 1)); img[..., 2] = k
        return img, tc

    def banding(self, cam): return self.band.get(cam)


# ---------------------------------------------------------------- ticks (pick_seq --tick-sink)
class TickBuffer:
    """Ticks in time order. add() for fakes, poll_file() for the real sink file (incremental; handles pick_seq restarting the file)."""
    def __init__(self): self.ticks = []; self._t = []; self._pos = 0; self._ino = None; self._buf = b""

    def add(self, tick):
        if self._t and tick["t_mono"] < self._t[-1]: raise ValueError("ticks must arrive in time order")
        self.ticks.append(tick); self._t.append(tick["t_mono"])

    def at(self, t):
        """The newest tick with t_mono <= t, or None."""
        i = bisect.bisect_right(self._t, t); return self.ticks[i - 1] if i else None

    def poll_file(self, path):
        """Read the lines appended to `path` since the last call. A half-written last line waits for the next poll; garbage lines are skipped. Returns the number of ticks added."""
        try:
            st = os.stat(path)
            if self._ino != st.st_ino or st.st_size < self._pos: self._pos = 0; self._buf = b""; self._ino = st.st_ino   # a new file (pick_seq restarted, or the ring rotated)
            with open(path, "rb") as f:
                f.seek(self._pos); data = f.read(); self._pos += len(data)
        except OSError: return 0
        self._buf += data; *lines, self._buf = self._buf.split(b"\n"); n = 0
        for ln in lines:
            try: d = json.loads(ln)
            except ValueError: continue
            if "seq" not in d or "action" not in d or d.get("state") is None: continue
            if self._t and d["t_mono"] < self._t[-1]: continue
            self.add(d); n += 1
        return n


def synth_plan_ticks(buf, ps, plan, rows, max_deg_s, t, cur, correct_wrist_clamp=True):
    """Ticks pick_seq --tick-sink would write for `plan` on the fake arm, appended to `buf`. `ps` = the pick_seq module (the SAME ease functions
    the real arm sends); `rows` = the arm's per-waypoint answers (target / reached). The state of a tick is what the servos read before the command:
    the previous command plus the shortfall of this waypoint (reached - target) times the easing, so the last tick lands on `reached`.
    Returns (t_end, cur_end, wake_times): the virtual clock after the plan (including each settle), the arm pose to start the next plan from, and the times the
    recorder would have woken up (every tick, plus every 1/30 s of each settle)."""
    times = []; seq = len(buf.ticks)
    for wp, row in zip(plan["waypoints"], rows):
        tgt = {j: float(row["target"][j]) for j in ps.J}; got = {j: float(row["reached"][j]) for j in ps.J}
        secs = ps.move_secs(cur, tgt, float(wp.get("seconds", 1.5)), max_deg_s); n = ps.ease_n(secs); prev_state = dict(cur)
        for i in range(1, n + 1):
            act = ps.ease_cmd(cur, tgt, i, n); seq += 1
            buf.add({"seq": seq, "t_mono": t, "t_wall": t, "wp": wp.get("name"), "action": act, "state": {f"{j}.pos": prev_state[j] for j in ps.J}})
            times.append(t); e = (1 - math.cos(math.pi * i / n)) / 2
            prev_state = {j: cur[j] + (tgt[j] - cur[j]) * e + (got[j] - tgt[j]) * e for j in ps.J}; t += secs / n
        settle = float(wp.get("settle", 0.3)); tl = t - secs / n; times.extend(tl + k / FPS for k in range(1, int((t + settle - tl) * FPS))); t += settle; cur = got   # why: the resident recorder wakes every 1/30 s whether or not a tick arrives
    return t, cur, times


# ---------------------------------------------------------------- features
def build_features(shapes, env_len, motors=MOTORS):
    """The lerobot-record feature dict for a so_follower with cameras of the given (H, W, 3) shapes, plus observation.environment_state."""
    from lerobot.utils.feature_utils import hw_to_dataset_features, combine_feature_dicts
    mft = {f"{m}.pos": float for m in motors}
    f = combine_feature_dicts(hw_to_dataset_features(mft, "action"), hw_to_dataset_features({**mft, **{c: tuple(s) for c, s in shapes.items()}}, "observation"))
    from lerobot.utils.constants import OBS_ENV_STATE
    f[OBS_ENV_STATE] = {"dtype": "float32", "shape": (env_len,), "names": [f"env_{i}" for i in range(env_len)]}
    return f


def percentile(xs, p):
    if not xs: return None
    s = sorted(xs); k = (len(s) - 1) * p / 100.0; lo = int(math.floor(k)); hi = min(lo + 1, len(s) - 1); return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------- the sidecar (meta/lab/)
def sidecar_dir(root): return os.path.join(root, "meta", "lab")


def read_sidecar(root):
    p = os.path.join(sidecar_dir(root), "episodes.jsonl"); rows = []
    if not os.path.exists(p): return rows
    for ln in open(p):
        try: rows.append(json.loads(ln))
        except ValueError: continue
    return rows


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f: f.write(json.dumps(row, default=str) + "\n"); f.flush(); os.fsync(f.fileno())


def reconcile_sidecar(root, total_episodes):
    """Make meta/lab/episodes.jsonl agree with the dataset's episode count after a crash. A crash between save_episode and the sidecar write leaves an episode
    with no row: add a placeholder row for it. Rows beyond the count (cannot happen with this write order, but a hand-edit could) are dropped.
    Returns {"added": [indices], "dropped": [indices]}."""
    rows = read_sidecar(root); have = {r.get("episode_index") for r in rows}; added = [i for i in range(total_episodes) if i not in have]
    dropped = sorted(r["episode_index"] for r in rows if isinstance(r.get("episode_index"), int) and r["episode_index"] >= total_episodes)
    if dropped:
        keep = [r for r in rows if not (isinstance(r.get("episode_index"), int) and r["episode_index"] >= total_episodes)]
        p = os.path.join(sidecar_dir(root), "episodes.jsonl"); tmp = p + ".tmp"
        with open(tmp, "w") as f: f.write("".join(json.dumps(r, default=str) + "\n" for r in keep))
        os.replace(tmp, p)
    for i in added: append_jsonl(os.path.join(sidecar_dir(root), "episodes.jsonl"), {"episode_index": i, "reconciled": "no sidecar row (a crash between save_episode and the sidecar write?)"})
    return {"added": added, "dropped": dropped}


# ---------------------------------------------------------------- one episode in flight
@dataclasses.dataclass
class Episode:
    meta: dict
    t_open: float
    spool: str
    frames: list = dataclasses.field(default_factory=list)    # {action, state, imgs{cam: npy path}, t_sample, tick_seq, tick_age_s, cam_age_s{cam}, wp, late_s}
    dropped_idle: int = 0
    k: int = 0                     # next boundary index: t_open + k / fps
    t_end: float = None
    state: str = "open"            # open -> ended -> (committed | discarded)


class DemoRecorder:
    """Owns a main dataset (kept episodes) and a rejects dataset (everything the keep rule refuses), both LeRobotDataset v3.0, fps 30.
    Lifecycle per try: open(meta, t) -> pump(t) as time passes (30 Hz in the resident process; per tick time in the fakes) -> end(t) -> commit(keep, verdict) or abort(reason)."""
    def __init__(self, root, repo_id, cams, ticks, cfg, rejects_root=None, rejects_repo=None, fps=FPS, max_tick_age_s=MAX_TICK_AGE_S, calib_hashes=None, spool_dir=None,
                 banding_max=BANDING_MAX, vcodec=DEFAULT_VCODEC):
        self.vcodec = vcodec   # why: LeRobot's default SVT-AV1 aborts at teardown on macOS (libdispatch semaphore dispose, 2026-11-11); the Jetson keeps it, Mac tests and dry runs pass "h264"
        self.root, self.repo_id, self.cams, self.ticks, self.cfg, self.fps = root, repo_id, cams, ticks, cfg, fps
        self.rejects_root = rejects_root or root.rstrip("/") + "-rejects"; self.rejects_repo = rejects_repo or repo_id + "-rejects"
        self.max_tick_age, self.calib_hashes, self.banding_max = max_tick_age_s, calib_hashes or {}, banding_max
        self.spool_root = spool_dir or tempfile.mkdtemp(prefix="demo_spool_"); self._own_spool = spool_dir is None
        self.features = build_features(cams.shapes, len(cfg.env_vector())); self._ds = {}; self.ep = None; self.counts = {"kept": 0, "rejected": 0, "aborted": 0, "empty": 0}
        self.reconciled = {}

    # -- datasets
    def _encoder(self):
        from lerobot.configs.video import RGBEncoderConfig
        return RGBEncoderConfig(vcodec=self.vcodec)

    def dataset(self, kind):
        """The LeRobotDataset for "main" or "rejects" (created on first use, resumed when the folder already holds one)."""
        if kind in self._ds: return self._ds[kind]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root, repo = (self.root, self.repo_id) if kind == "main" else (self.rejects_root, self.rejects_repo)
        if os.path.exists(os.path.join(root, "meta", "info.json")):
            ds = LeRobotDataset.resume(repo, root=root, rgb_encoder=self._encoder()); user = lambda feats: {k: tuple(v["shape"]) for k, v in feats.items() if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
            have, want = user(ds.meta.features), user(self.features)
            if have != want: raise ValueError(f"{root}: existing dataset features {have} differ from this recorder's {want}")
            if ds.fps != self.fps: raise ValueError(f"{root}: existing dataset is {ds.fps} fps, recorder is {self.fps}")
        else:
            ds = LeRobotDataset.create(repo, self.fps, self.features, root=root, robot_type=ROBOT_TYPE, use_videos=True, rgb_encoder=self._encoder())
        self.reconciled[kind] = reconcile_sidecar(root, ds.meta.total_episodes); self._ds[kind] = ds; return ds

    def alive(self): return True   # phase 2: the resident process answers on its control socket

    # -- the episode
    def open(self, meta=None, t=None):
        if self.ep is not None and self.ep.state in ("open", "ended"): raise RuntimeError("an episode is already open; commit or abort it first")
        t = time.monotonic() if t is None else t
        self.ep = Episode(meta=dict(meta or {}), t_open=t, spool=tempfile.mkdtemp(prefix="ep_", dir=self.spool_root))

    def pump(self, now):
        """Sample every 1/fps boundary up to `now`. Boundaries with no fresh tick are dropped as idle."""
        ep = self.ep
        if ep is None or ep.state != "open": return 0
        n = 0; period = 1.0 / self.fps
        while ep.t_open + ep.k * period <= now + 1e-9:
            b = ep.t_open + ep.k * period; ep.k += 1; tk = self.ticks.at(b)
            if tk is None or b - tk["t_mono"] > self.max_tick_age: ep.dropped_idle += 1; continue
            k = len(ep.frames); imgs = {}; cam_age = {}
            for c in self.cams.names:
                img, tc = self.cams.frame_at(c, b); p = os.path.join(ep.spool, f"f{k:05d}_{c}.npy"); np.save(p, np.ascontiguousarray(img, dtype=np.uint8)); imgs[c] = p; cam_age[c] = b - tc
            ep.frames.append({"action": [float(tk["action"][f"{m}.pos"]) for m in MOTORS], "state": [float(tk["state"][f"{m}.pos"]) for m in MOTORS], "imgs": imgs,
                              "t_sample": b, "tick_seq": tk["seq"], "tick_t": tk["t_mono"], "tick_age_s": b - tk["t_mono"], "cam_age_s": cam_age, "wp": tk.get("wp"), "late_s": max(0.0, now - b)})
            n += 1
        return n

    def end(self, t=None):
        """Stop sampling; returns the health dict. The frames stay in the spool until commit() or abort()."""
        ep = self.ep
        if ep is None or ep.state != "open": raise RuntimeError("no open episode")
        t = time.monotonic() if t is None else t; self.pump(t); ep.t_end = t; ep.state = "ended"; return self.health()

    def health(self):
        """Recording health of the current episode: frames, dropped idle boundaries, cadence, camera ages, banding. ok is the keep rule's 4th condition."""
        ep = self.ep; fr = ep.frames; n = len(fr); period = 1.0 / self.fps
        late = [f["late_s"] for f in fr]; over = sum(1 for x in late if x > period); pct = round(100.0 * over / n, 2) if n else 0.0
        cam_max = {c: round(max((f["cam_age_s"][c] for f in fr), default=0.0), 4) for c in self.cams.names}
        band = {c: self.cams.banding(c) for c in self.cams.names}; reasons = []
        if n < MIN_FRAMES: reasons.append(f"too_few_frames:{n}")
        if pct > OVER_BUDGET_MAX_PCT: reasons.append(f"cadence:{pct}pct_over_budget")
        frozen = sorted(c for c, a in cam_max.items() if a > CAM_FREEZE_S)
        if frozen: reasons.append("camera_frozen:" + ",".join(frozen))
        bad = sorted(c for c in BANDING_CAMS if band.get(c) is not None and band[c] > self.banding_max)
        if bad: reasons.append("banding:" + ",".join(bad))
        ta = [f["tick_age_s"] for f in fr]
        return {"ok": not reasons, "reasons": reasons, "frames": n, "dropped_idle_frames": ep.dropped_idle, "over_budget_pct": pct, "cam_age_max_s": cam_max, "banding": band,
                "tick_age_mean_s": round(sum(ta) / n, 4) if n else None, "tick_age_p95_s": round(percentile(ta, 95), 4) if n else None, "tick_age_max_s": round(max(ta), 4) if n else None}

    def abort(self, reason="aborted"):
        """Drop the open or ended episode without saving it anywhere (a STOP, a trip mid-try, an interrupt). Returns True when there was one."""
        ep = self.ep
        if ep is None or ep.state not in ("open", "ended"): return False
        shutil.rmtree(ep.spool, ignore_errors=True); ep.state = "discarded"; self.counts["aborted"] += 1; self.last_abort = reason; return True

    def commit(self, keep, verdict=None):
        """Write the ended episode to the main dataset (keep) or the rejects dataset, add its sidecar row. Returns
        {"dataset": "main"|"rejects"|None, "episode_index", "frames", "health"}; an empty episode is written nowhere."""
        ep = self.ep
        if ep is None or ep.state != "ended": raise RuntimeError("end() the episode before commit()")
        h = self.health(); verdict = dict(verdict or {}); kind = "main" if keep else "rejects"
        if not ep.frames:
            shutil.rmtree(ep.spool, ignore_errors=True); ep.state = "discarded"; self.counts["empty"] += 1; return {"dataset": None, "episode_index": None, "frames": 0, "health": h}
        ds = self.dataset(kind); root = self.root if keep else self.rejects_root; env = np.asarray(self.cfg.env_vector(), np.float32); idx = ds.meta.total_episodes
        for f in ep.frames:
            fr = {"action": np.asarray(f["action"], np.float32), "observation.state": np.asarray(f["state"], np.float32), "observation.environment_state": env, "task": TASK}
            for c, p in f["imgs"].items(): fr[f"observation.images.{c}"] = np.load(p)
            ds.add_frame(fr)
        ds.save_episode()
        row = self.sidecar_row(idx, kind, ep, h, verdict)   # why after save_episode: a crash in between leaves an episode without a row, which reconcile_sidecar() repairs; the other order would leave a row for an episode that does not exist
        self.write_frame_log(root, idx, ep); append_jsonl(os.path.join(sidecar_dir(root), "episodes.jsonl"), row)
        shutil.rmtree(ep.spool, ignore_errors=True); ep.state = "committed"; self.counts["kept" if keep else "rejected"] += 1
        return {"dataset": kind, "episode_index": idx, "frames": len(ep.frames), "health": h, "row": row}

    def sidecar_row(self, idx, kind, ep, h, verdict):
        m = ep.meta
        return {"episode_index": idx, "dataset": kind, "kept": kind == "main", "task": TASK, "fps": self.fps, "n_frames": len(ep.frames),
                "configuration": self.cfg.sidecar_labels(self.calib_hashes), "scene": m.get("scene", {}), "teacher": m.get("teacher", {}),
                "quality": {"tier": verdict.get("tier"), **{k: v for k, v in h.items() if k != "reasons"}, "health_reasons": h["reasons"]},
                "verdict": {k: v for k, v in verdict.items() if k != "tier"}, "links": m.get("links", {}), "t_open_mono": ep.t_open, "t_end_mono": ep.t_end}

    def write_frame_log(self, root, idx, ep):
        d = os.path.join(sidecar_dir(root), "frames"); os.makedirs(d, exist_ok=True)
        cols = {k: [f[k] for f in ep.frames] for k in ("t_sample", "tick_seq", "tick_t", "tick_age_s", "wp")}
        cols["cam_age_s"] = {c: [f["cam_age_s"][c] for f in ep.frames] for c in self.cams.names}
        with open(os.path.join(d, f"episode_{idx:06d}.json"), "w") as f: json.dump(cols, f)

    def status(self):
        ep = self.ep
        return {"alive": True, "episode": None if ep is None else {"state": ep.state, "frames": len(ep.frames), "dropped_idle": ep.dropped_idle}, "counts": dict(self.counts)}

    def close(self):
        """Finalize both datasets (required, or the parquet footers are missing) and drop the spool. Idempotent."""
        if self.ep is not None and self.ep.state in ("open", "ended"): self.abort("closed")
        for ds in self._ds.values(): ds.finalize()
        self._ds.clear()
        if self._own_spool: shutil.rmtree(self.spool_root, ignore_errors=True)


def dataset_hash(root):
    """sha256 over every file of a dataset except raw images and the append-only links file, in sorted path order (path and content)."""
    h = hashlib.sha256()
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != "images")
        for name in sorted(fn):
            if name in ("links.jsonl",) or name.endswith(".tmp"): continue
            p = os.path.join(dp, name); h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
