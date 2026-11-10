"""jetson/cam_health.py without cameras: synthetic timestamp sequences through the statistics, a fake capture
source with scripted frame times through the capture loop and the report, and the exit codes at the thresholds."""
import importlib.util, json, os, threading
import numpy as np
import pytest
from jhelpers import JETSON

pytest.importorskip("cv2")


@pytest.fixture(scope="session")
def ch():
    spec = importlib.util.spec_from_file_location("cam_health", os.path.join(JETSON, "cam_health.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- a camera that exists only as a timetable
class FakeSource:
    """Frames at `fps` for `seconds` with optional extra gaps ({frame index: extra seconds}), duplicated frames every
    `dup_every` frames and `fail_first` failed reads at the start. now() is the time of the last frame handed out, so
    cam_health.capture()'s warm-up and deadline run on this clock, not the wall clock."""
    opened = []

    def __init__(self, cfg, fps=30.0, seconds=12.0, gaps=None, dup_every=0, fail_first=0, negotiated=None, frame_size=None, raise_on_open=None):
        if raise_on_open: raise raise_on_open
        self.cfg = cfg; self.neg = negotiated or {"width": cfg["width"], "height": cfg["height"], "fps": 30.0, "fourcc": cfg.get("fourcc") or "YUYV"}
        w, h = frame_size or (cfg["width"], cfg["height"])
        self.frames = []; t = 0.0; prev = None
        for k in range(int(seconds * fps) + 1):
            t += (gaps or {}).get(k, 0.0)
            if dup_every and k % dup_every == 0 and prev is not None: f = prev
            else: f = np.full((h, w, 3), k % 251, np.uint8); f[0, 0] = (k * 7) % 255
            self.frames.append((f, t)); prev = f; t += 1.0 / fps
        self.fail_first = fail_first; self.t_now = 0.0; self.i = 0; self.closed = False
        FakeSource.opened.append(self)

    def negotiated(self): return dict(self.neg)
    def now(self): return self.t_now
    def read(self):
        if self.fail_first > 0: self.fail_first -= 1; return None, self.t_now
        if self.i >= len(self.frames): self.t_now = self.frames[-1][1] + 1000.0; return None, self.t_now
        f, t = self.frames[self.i]; self.i += 1; self.t_now = t; return f, t
    def close(self): self.closed = True


def opener(**kw): return lambda cfg: FakeSource(cfg, **kw)


# ---------------------------------------------------------------- interval statistics on synthetic timestamps
def test_perfect_30fps_sequence(ch):
    ts = np.arange(0, 10.0001, 1 / 30)
    s = ch.interval_stats(ts)
    assert s["frames"] == 301 and s["fps"] == 30.0 and s["stalls"] == 0
    assert abs(s["p50_ms"] - 33.33) < 0.05 and abs(s["p95_ms"] - 33.33) < 0.05 and abs(s["max_ms"] - 33.33) < 0.05


def test_gaps_counted_as_stalls_and_lower_fps(ch):
    ts = list(np.arange(0, 5.0001, 1 / 30)); ts = [t + (0.15 if t > 2 else 0) + (0.3 if t > 4 else 0) for t in ts]
    s = ch.interval_stats(ts, stall_ms=100)
    assert s["stalls"] == 2 and abs(s["longest_stall_ms"] - 333.3) < 1 and s["max_ms"] == s["longest_stall_ms"]
    assert 27 < s["fps"] < 28                      # 150 frames over 5.45 s
    assert abs(s["p50_ms"] - 33.33) < 0.05 and s["p95_ms"] < 40


def test_stall_threshold_is_exclusive_and_configurable(ch):
    ts = [0, 0.1, 0.2, 0.3]                        # exactly 100 ms apart: not a stall at 100, a stall at 99
    assert ch.interval_stats(ts, stall_ms=100)["stalls"] == 0
    assert ch.interval_stats(ts, stall_ms=99)["stalls"] == 3


def test_too_few_frames(ch):
    s = ch.interval_stats([1.0])
    assert s["frames"] == 1 and s["fps"] == 0.0 and s["p50_ms"] is None and s["longest_stall_ms"] is None


def test_fourcc_decoding(ch):
    import cv2
    assert ch.fourcc_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
    assert ch.fourcc_str(cv2.VideoWriter_fourcc(*"YUYV")) == "YUYV"
    assert ch.fourcc_str(0) == "????" and ch.fourcc_str(None) == "????"


# ---------------------------------------------------------------- the capture loop on a fake source
def test_capture_discards_warmup_and_counts_duplicates_and_failed_reads(ch):
    src = FakeSource({"width": 64, "height": 48, "fourcc": None}, fps=30, seconds=12, dup_every=10, fail_first=3)
    cap = ch.capture(src, seconds=10.0, warmup=1.0)
    ts = cap["timestamps"]
    assert ts[0] >= 1.0 and ts[-1] >= 11.0 and len(ts) in (300, 301, 302)
    assert cap["failed_reads"] == 3 and cap["frame_size"] == (64, 48)
    assert 28 <= cap["duplicates"] <= 31            # one in ten frames repeats the previous one


def test_capture_near_duplicates_only_with_dup_mad(ch):
    src = FakeSource({"width": 64, "height": 48, "fourcc": None}, fps=30, seconds=3)
    assert ch.capture(src, 2.0, 0.5)["duplicates"] == 0           # consecutive frames differ by one grey level: not identical
    src = FakeSource({"width": 64, "height": 48, "fourcc": None}, fps=30, seconds=3)
    assert ch.capture(src, 2.0, 0.5, dup_mad=1.5)["duplicates"] > 50   # ...but near-identical under a 1.5 level tolerance


def test_capture_stops_when_source_dies(ch):
    src = FakeSource({"width": 8, "height": 8, "fourcc": None}, fps=30, seconds=2)   # frames run out after 2 s
    cap = ch.capture(src, seconds=10.0, warmup=0.5)
    assert len(cap["timestamps"]) < 50 and cap["failed_reads"] >= 1


# ---------------------------------------------------------------- report + verdicts + exit codes
def two_cams():
    return {"top": {"path": "/dev/fake0", "width": 640, "height": 360, "fps": 30, "fourcc": "MJPG"},
            "wrist": {"path": "/dev/fake1", "width": 640, "height": 480, "fps": 30, "fourcc": None}}


def test_healthy_report(ch):
    FakeSource.opened.clear()
    rep = ch.run(two_cams(), seconds=10.0, warmup=1.0, opener=opener(fps=30))
    assert rep["ok"] and ch.exit_code(rep) == 0
    for name, c in rep["cameras"].items():
        assert c["problems"] == [] and c["stats"]["fps"] == 30.0 and c["stats"]["stalls"] == 0 and c["duplicates"] == 0
        assert c["frame_size"] == (c["requested"]["width"], c["requested"]["height"])
    assert rep["cameras"]["top"]["negotiated"]["fourcc"] == "MJPG"
    assert len(FakeSource.opened) == 2 and all(s.closed for s in FakeSource.opened)
    assert "loadavg" in rep["cpu"]["before"]
    txt = ch.format_text(rep)
    assert "all cameras healthy" in txt and "top    640x360 MJPG" in txt and " OK" in txt


def test_cameras_are_opened_concurrently(ch):
    """All sources are open at once (one thread each), not one after the other: that is the point of the check."""
    started, release = [], threading.Event()
    class Blocking(FakeSource):
        def __init__(self, cfg):
            super().__init__(cfg, fps=30, seconds=2); started.append(cfg["path"]); release.wait(5)
    t = threading.Thread(target=lambda: ch.run(two_cams(), seconds=1.0, warmup=0.2, opener=Blocking), daemon=True); t.start()
    deadline = threading.Event()
    for _ in range(200):
        if len(started) == 2: break
        deadline.wait(0.01)
    assert len(started) == 2
    release.set(); t.join(5)


@pytest.mark.parametrize("fps,code", [(30.0, 0), (28.0, 0), (27.9, 1), (22.3, 1)])
def test_fps_threshold(ch, fps, code):
    rep = ch.run({"wrist": two_cams()["wrist"]}, 10.0, 1.0, opener(fps=fps))
    assert ch.exit_code(rep) == code
    if code: assert rep["cameras"]["wrist"]["problems"][0].startswith("fps ")


@pytest.mark.parametrize("gap_ms,code", [(90, 0), (200, 0), (251, 1), (600, 1)])
def test_stall_threshold(ch, gap_ms, code):
    rep = ch.run({"top": two_cams()["top"]}, 10.0, 1.0, opener(fps=30, gaps={150: gap_ms / 1000 - 1 / 30}))
    c = rep["cameras"]["top"]
    assert ch.exit_code(rep) == code
    assert c["stats"]["stalls"] == (1 if gap_ms > 100 else 0)
    assert abs(c["stats"]["longest_stall_ms"] - (gap_ms if gap_ms > 33.4 else 33.3)) < 1
    if code: assert any(p.startswith("stall ") for p in c["problems"])


def test_negotiated_size_mismatch_fails(ch):
    rep = ch.run({"top": two_cams()["top"]}, 5.0, 0.5, opener(frame_size=(640, 480), negotiated={"width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG"}))
    assert ch.exit_code(rep) == 1 and rep["cameras"]["top"]["problems"] == ["size 640x480 != requested 640x360"]


def test_frame_size_wins_over_driver_claim(ch):
    """The driver can report the requested size while delivering another; the frame's own shape is what counts."""
    rep = ch.run({"top": two_cams()["top"]}, 5.0, 0.5, opener(frame_size=(320, 180)))
    assert rep["cameras"]["top"]["problems"] == ["size 320x180 != requested 640x360"]


def test_fourcc_mismatch_is_reported(ch):
    rep = ch.run({"top": two_cams()["top"]}, 5.0, 0.5, opener(negotiated={"width": 640, "height": 360, "fps": 30.0, "fourcc": "YUYV"}))
    assert rep["cameras"]["top"]["problems"] == ["fourcc YUYV != requested MJPG"]


def test_open_failure_gives_exit_2_and_keeps_other_cameras(ch):
    def op(cfg): return FakeSource(cfg, raise_on_open=RuntimeError("cannot open /dev/fake1 (busy or unplugged)") if cfg["path"].endswith("1") else None)
    rep = ch.run(two_cams(), 5.0, 0.5, op)
    assert ch.exit_code(rep) == 2 and "busy or unplugged" in rep["cameras"]["wrist"]["error"]
    assert rep["cameras"]["top"]["problems"] == [] and rep["cameras"]["wrist"]["problems"][0].startswith("could not capture")
    assert "FAIL cannot open" in ch.format_text(rep)


def test_no_frames_at_all_is_a_problem(ch):
    rep = ch.run({"top": two_cams()["top"]}, 2.0, 0.5, opener(fps=30, seconds=0.2))   # the source dies in the warm-up
    assert rep["cameras"]["top"]["problems"] == ["no frames"] and ch.exit_code(rep) == 1


# ---------------------------------------------------------------- CLI
def test_main_json_and_busy_flag(ch, tmp_path, monkeypatch, capsys):
    flag = tmp_path / ".cams_busy"; seen = []
    monkeypatch.setattr(ch.time, "sleep", lambda s: None)            # BusyFlag's 1.5 s settle
    def op(cfg): seen.append(flag.exists()); return FakeSource(cfg, fps=30, seconds=4)
    code = ch.main(["3", "--warmup", "0.5", "--cams", "top,wrist", "--json", "--busy-flag", str(flag), "--device", "wrist=/dev/video9"], opener=op)
    rep = json.loads(capsys.readouterr().out)
    assert code == 0 and rep["ok"] and set(rep["cameras"]) == {"top", "wrist"}
    assert rep["cameras"]["wrist"]["path"] == "/dev/video9" and rep["seconds"] == 3.0
    assert seen == [True, True] and not flag.exists()                 # flag held while the devices were open, gone after


def test_main_no_busy_and_threshold_overrides(ch, tmp_path, capsys):
    flag = tmp_path / ".cams_busy"
    code = ch.main(["2", "--warmup", "0.2", "--cams", "wrist", "--no-busy", "--busy-flag", str(flag), "--min-fps", "20"], opener=opener(fps=22.3))
    out = capsys.readouterr().out
    assert code == 0 and "all cameras healthy" in out and not flag.exists()
    code = ch.main(["2", "--warmup", "0.2", "--cams", "wrist", "--no-busy", "--busy-flag", str(flag)], opener=opener(fps=22.3))
    assert code == 1 and "fps 22.3 < 28.0" in capsys.readouterr().out


def test_main_rejects_unknown_camera(ch, capsys):
    assert ch.main(["1", "--cams", "side", "--no-busy"], opener=opener()) == 2
    assert ch.main(["1", "--device", "nose=/dev/x", "--no-busy"], opener=opener()) == 2
