"""jetson/demo_recorderd.py, the resident recorder: an episode recorded end to end through its control socket with a fake camd (fake ffmpeg), a tick file written live like
pick_seq --tick-sink, and the failure modes: abort, a handler that raises, a missing camera, hardware flags, a dead daemon. LeRobot video uses h264 here (SVT-AV1 aborts at
teardown on macOS). No ssh, no arm, no real camera."""
import importlib.util, json, os, shutil, sys, tempfile, threading, time
import pytest
from test_camd import FAKE_FFMPEG, PY, JETSON, load_mod, wait_until

SMALL = {"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.fixture(scope="module")
def mods():
    pytest.importorskip("lerobot"); pytest.importorskip("cv2")   # why: CI has no lerobot (requirements-dev.txt), DemoRecorder writes real LeRobot datasets
    camd = load_mod("camd", os.path.join(JETSON, "camd.py")); dcams = load_mod("demo_cams", os.path.join(JETSON, "demo_cams.py")); drd = load_mod("demo_recorderd", os.path.join(JETSON, "demo_recorderd.py"))
    return camd, dcams, drd


class Tickers:
    """What pick_seq --tick-sink does: one JSON line per ~25 ms with the command sent and the state read before it, on time.monotonic()."""
    def __init__(self, path): self.path, self.seq, self.stop = path, 0, False

    def run(self, secs, hz=40.0, header=True):
        with open(self.path, "w") as f:
            if header: f.write(json.dumps({"header": 1, "version": 1, "pid": 1, "t_mono": time.monotonic(), "t_wall": time.time()}) + "\n"); f.flush()
            t_end = time.monotonic() + secs
            while time.monotonic() < t_end and not self.stop:
                self.seq += 1; v = float(self.seq)
                f.write(json.dumps({"seq": self.seq, "t_mono": time.monotonic(), "t_wall": time.time(), "wp": "hover", "action": {f"{m}.pos": v + i for i, m in enumerate(MOTORS)}, "state": {f"{m}.pos": v - 0.5 + i for i, m in enumerate(MOTORS)}}) + "\n"); f.flush(); time.sleep(1 / hz)

    def start(self, secs):
        t = threading.Thread(target=self.run, args=(secs,), daemon=True); t.start(); return t


@pytest.fixture
def rig(mods, tmp_path):
    camd, dcams, drd = mods; d0 = tempfile.mkdtemp(prefix="rd"); made = {"servers": [], "daemons": [], "srcs": []}
    ff = os.path.join(d0, "ff.py"); open(ff, "w").write(FAKE_FFMPEG)

    def make(missing=(), busy_flags=(), cams=None):
        csock = os.path.join(d0, "c.sock"); dsock = os.path.join(d0, "d.sock"); ticks = os.path.join(d0, "ticks.jsonl")
        if cams is None:
            cc = {n: camd.Camera(n, (lambda n: None if n in missing else [PY, ff, n, "30"]), busy=os.path.join(d0, "nobusy"), retry_s=0.05) for n in SMALL}
            srv = camd.Server(cc, csock).start(); made["servers"].append(srv)
            cams = dcams.CamdCameraSource(csock, shapes=SMALL).start(); made["srcs"].append(cams)
            wait_until(lambda: len(cams.missing()) == len(missing))
        d = drd.Daemon(cams, ticks, spool=os.path.join(d0, "spool"), vcodec="h264", busy=tuple(busy_flags), calib_dir=os.path.join(d0, "calib"), log=lambda m: None); d.listen(dsock)
        th = threading.Thread(target=d.run, daemon=True); th.start(); made["daemons"].append((d, th))
        ctl = lambda op, **p: drd.ctl(op, p, dsock, timeout=60.0)[1]
        return d, ctl, Tickers(ticks), str(tmp_path / "data" / "nut")
    yield make
    for d, th in made["daemons"]: d.stopping = True; th.join(5)
    for c in made["srcs"]: c.close()
    for s in made["servers"]: s.stop()
    shutil.rmtree(d0, ignore_errors=True)


def configure(ctl, root, **kw): return ctl("configure", repo="lab/nut", root=root, lab={"lighting_tag": "lamp-a"}, vcodec="h264", **kw)


def test_one_episode_end_to_end_through_the_socket(rig):
    d, ctl, ticks, root = rig(); assert ctl("ping")["ok"] and not ctl("ping")["configured"]
    r = configure(ctl, root); assert r["ok"] and r["config_id"] == "stock-v1" and r["rejects_root"] == root + "-rejects"
    assert ctl("open", meta={"scene": {"start_in": [6, 8]}, "teacher": {"session": "t", "cycle": 1}})["ok"] and ctl("ping")["episode"]["state"] == "open"
    ticks.start(1.5).join(); time.sleep(0.1); h = ctl("end"); assert h["ok"]
    health = h["health"]; assert health["frames"] >= 30 and all(r.startswith("cadence:") for r in health["reasons"]), health   # why: cadence is wall-clock and the full suite loads the machine; cameras and ticks must be healthy. ~45 frames at 30 fps from 1.5 s of ticks
    c = ctl("commit", keep=True, verdict={"tier": "A", "keep": True}, meta={"teacher": {"n_attempts": 1}}); assert c["ok"] and c["dataset"] == "main" and c["episode_index"] == 0 and c["frames"] == health["frames"]
    assert ctl("link", kind="main", row={"episode_index": 0, "mlflow_cycle_run_id": "abc"})["ok"]
    f = ctl("finish"); assert f["ok"] and f["main"]["hash"] and len(f["main"]["rows"]) == 1 and f["rejects"]["hash"] is None and f["counts"]["kept"] == 1
    row = f["main"]["rows"][0]; assert row["teacher"] == {"session": "t", "cycle": 1, "n_attempts": 1} and row["configuration"]["lighting_tag"] == "lamp-a" and row["quality"]["tier"] == "A" and row["scene"]["start_in"] == [6, 8]
    assert "observation.images.wrist" in f["features"]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lab/nut", root=root); assert ds.num_episodes == 1 and ds.num_frames == health["frames"] and ds.fps == 30 and ds.meta.total_episodes == 1
    item = ds[0]; assert item["action"].shape == (6,) and float(item["action"][1] - item["action"][0]) == pytest.approx(1.0) and item["observation.images.wrist"].shape[-2:] == (48, 64)
    assert open(os.path.join(root, "meta", "lab", "links.jsonl")).read().count("abc") == 1


def test_a_rejected_try_goes_to_the_rejects_dataset(rig):
    d, ctl, ticks, root = rig(); configure(ctl, root); ctl("open", meta={}); ticks.start(1.4).join(); ctl("end")
    c = ctl("commit", keep=False, verdict={"discard_reason": "outside_radius"}); assert c["ok"] and c["dataset"] == "rejects"
    f = ctl("finish"); assert f["main"]["hash"] is None and len(f["rejects"]["rows"]) == 1 and f["rejects"]["rows"][0]["verdict"]["discard_reason"] == "outside_radius" and os.path.isdir(root + "-rejects")


def test_abort_saves_nothing_anywhere(rig):
    d, ctl, ticks, root = rig(); configure(ctl, root); ctl("open", meta={}); ticks.start(0.5).join()
    a = ctl("abort", reason="trip:run_plan"); assert a["ok"] and a["was_open"] and ctl("ping")["episode"] is None
    assert not ctl("abort")["was_open"]
    f = ctl("finish"); assert f["main"]["hash"] is None and f["rejects"]["hash"] is None and f["counts"]["aborted"] == 1 and not os.path.exists(os.path.join(root, "meta", "info.json"))
    assert os.listdir(d.spool) == []   # the raw frames of the aborted episode are gone


def test_a_missing_camera_still_opens_and_the_episode_fails_its_health_check(rig):
    d, ctl, ticks, root = rig(missing=("wrist",)); configure(ctl, root)
    o = ctl("open", meta={}); assert o["ok"] and o["cams_missing"] == ["wrist"]   # the arm is never made to wait for the recorder
    ticks.start(1.2).join(); h = ctl("end")["health"]
    assert not h["ok"] and any(r.startswith("camera_frozen:") and "wrist" in r for r in h["reasons"])
    ds = load_mod("demo_session_t", os.path.join(os.path.dirname(JETSON), "tools", "demo_session.py"))
    dec = ds.keep_decision({"picked": True, "landed_err_mm": 3.0, "retries": 0}, 15.0, None, True, h); assert not dec.keep and any("camera_frozen" in r for r in dec.reasons)   # the keep rule refuses it
    c = ctl("commit", keep=False, verdict=dec.as_json()); assert c["ok"] and c["dataset"] == "rejects"


def test_an_error_while_sampling_aborts_the_episode_and_the_daemon_keeps_running(rig, mods):
    d, ctl, ticks, root = rig(); configure(ctl, root); ctl("open", meta={}); orig = d.cams.frame_at
    def boom(cam, t): raise RuntimeError("decode exploded")
    d.cams.frame_at = boom; ticks.start(0.6).join(); time.sleep(0.2)
    assert ctl("ping")["ok"] and ctl("ping")["episode"] is None and "decode exploded" in ctl("status")["last_err"]   # the crash rejected the episode; the daemon and the arm are untouched
    d.cams.frame_at = orig; assert ctl("open", meta={})["ok"]   # and it records the next one
    ctl("abort")


def test_a_handler_that_raises_answers_an_error_instead_of_dying(rig):
    d, ctl, ticks, root = rig()
    assert not ctl("open", meta={})["ok"] and "not configured" in ctl("open", meta={})["err"]
    assert "unknown op" in ctl("frobnicate")["err"] and not ctl("end")["ok"] and not ctl("commit", keep=True)["ok"] and ctl("ping")["ok"]
    configure(ctl, root); ctl("open", meta={}); assert not configure(ctl, root)["ok"]   # a new session is refused while an episode is open


def test_hardware_flags_refuse_an_episode(rig, tmp_path):
    flag = str(tmp_path / "cams_busy"); open(flag, "w").close(); d, ctl, ticks, root = rig(busy_flags=(flag,)); configure(ctl, root)
    o = ctl("open", meta={}); assert not o["ok"] and o["err"] == "hardware_busy:cams_busy" and ctl("status")["busy"] == [flag]
    os.unlink(flag); assert ctl("open", meta={})["ok"]; ctl("abort")


def test_pick_seq_restarting_the_sink_file_does_not_lose_the_next_ticks(rig):
    d, ctl, ticks, root = rig(); configure(ctl, root); ctl("open", meta={}); ticks.start(0.6).join(); ticks.seq = 0; ticks.start(0.8).join()   # the second run truncates and rewrites the file (pick_seq opens it with "w" each run)
    h = ctl("end")["health"]; assert h["frames"] >= 30; ctl("abort")


def test_stale_spool_of_a_crashed_run_is_wiped_at_start(mods, tmp_path):
    camd, dcams, drd = mods; sp = tmp_path / "spool"; (sp / "ep_dead").mkdir(parents=True); (sp / "ep_dead" / "f00000_top.jpg").write_bytes(b"x")
    drd.Daemon(dcams.CamdCameraSource("/nonexistent", shapes=SMALL), str(tmp_path / "t"), spool=str(sp), calib_dir=str(tmp_path), log=lambda m: None); assert os.listdir(sp) == []


def test_a_dead_recorder_costs_the_episode_not_the_arm(rig, mods):
    ds = load_mod("demo_session_t2", os.path.join(os.path.dirname(JETSON), "tools", "demo_session.py")); camd, dcams, drd = mods
    d, ctl, ticks, root = rig(); sock = d.path; rec = ds.RemoteRecorder(lambda op, p, t=30.0: drd.ctl(op, p, sock, timeout=5.0)[1], root, "lab/nut", load_mod("lab_config_t", os.path.join(JETSON, "lab_config.py")).LabConfig(), vcodec="h264")
    rec.configure(); assert rec.alive(); rec.open({"teacher": {}}); d.stopping = True; time.sleep(0.5)   # the daemon dies in the middle of the try
    assert not rec.alive()
    h = rec.end(); assert not h["ok"] and h["reasons"][0].startswith("recorder_unreachable")
    c = rec.commit(False, {}); assert c["dataset"] is None and rec.abort("x") is False   # nothing raised into the runner
    with pytest.raises(ds.RecorderError): rec.open({})   # only open() raises, and DemoSession turns that into "this try is not recorded"


def test_the_recorder_process_never_touches_the_arm():
    for f in ("demo_recorderd.py", "demo_cams.py", "demo_recorder.py"):
        src = open(os.path.join(JETSON, f)).read()
        assert not any(w in src for w in ("send_action", "import lerobot", "from lerobot.robots", "from lerobot.motors", "import serial", "so101_follower", "/dev/so101", "FOLLOWER")), f
