"""jetson/cam_lock.py against a fake v4l2-ctl and a fake frame source. No hardware: subprocess.run is replaced inside the
module by FakeV4L2, FrameSource by FakeFrames, and time.sleep by the fake's clock so the camera's delayed behaviour fires.
The fake models what was measured on the lab's C920s (046d:08e5) and the icSpring wrist camera:
  * a manual control is `inactive` while its auto flag is on: writes fail with EACCES ("Permission denied", rc 255),
    reads still return the value auto is holding;
  * --get-ctrl prints controls in class order, not request order, and menus as "3 (Aperture Priority Mode)";
  * C920 manual exposure snaps to 1/2^n s stops and the same stop may read back one unit low (78 -> 77);
  * ~1 s after auto_exposure goes 3 -> 1 the C920 forces gain to 255 (the white-out), clobbering a gain written before;
  * every new stream start in Manual Mode re-derives the C920 exposure (comes up at 312);
  * picture mean = light * exposure * 2^(gain/64) (C920) or light * exposure (wrist), clipped to 255; under auto the
    camera produces its own mid-grey whatever the light.
  * a control can refuse to take (write accepted, value unchanged), a device can be unplugged (no by-id symlink).
/dev/v4l/by-id is a temp dir of symlinks so the realpath step is exercised for real."""
import json, math, os, re, subprocess, sys, textwrap, importlib.util
import pytest
from jhelpers import JETSON

C920 = {"brightness": 128, "contrast": 128, "saturation": 128, "white_balance_automatic": 1, "gain": 109, "power_line_frequency": 2,
        "white_balance_temperature": 3366, "sharpness": 128, "backlight_compensation": 0, "auto_exposure": 3, "exposure_time_absolute": 83,
        "exposure_dynamic_framerate": 1, "pan_absolute": 0, "tilt_absolute": 0, "focus_absolute": 35, "focus_automatic_continuous": 1, "zoom_absolute": 100}
ICSPRING = {"brightness": 0, "contrast": 27, "saturation": 25, "hue": 0, "white_balance_automatic": 1, "gamma": 67, "power_line_frequency": 1,
            "white_balance_temperature": 4000, "sharpness": 4, "backlight_compensation": 0, "auto_exposure": 3, "exposure_time_absolute": 313}
GATES = {"focus_automatic_continuous": (1, ["focus_absolute"]), "auto_exposure": (3, ["exposure_time_absolute"]),
         "white_balance_automatic": (1, ["white_balance_temperature"])}
MENU = {"auto_exposure": {1: "Manual Mode", 3: "Aperture Priority Mode"}, "power_line_frequency": {0: "Disabled", 1: "50 Hz", 2: "60 Hz"}}
STOPS = [3, 5, 10, 20, 39, 78, 156, 312, 625, 1250, 2047]
TOP, FRONT, WRIST = ("usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0", "usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0",
                     "usb-icSpring_icspring_camera-video-index0")
NODES = {TOP: "video0", FRONT: "video4", WRIST: "video2"}
AUTO_MEAN = {TOP: 116.0, FRONT: 89.0, WRIST: 125.0}


class Result:
    def __init__(self, rc, out="", err=""): self.returncode, self.stdout, self.stderr = rc, out, err


class FakeV4L2:
    """State of three fake cameras plus a log of every set, keyed by /dev/videoN path."""
    def __init__(self, devdir):
        self.devdir = devdir
        self.state, self.light, self.clobber_pending, self.sticky, self.jitter = {}, {}, set(), {}, {}
        self.clip = {}                 # dev -> (fraction of pixels >= 250, brightest-decile local std); default: no clipping
        self.restart_quirk = True      # a new stream start re-derives C920 exposure
        self.log, self.calls, self.sleeps, self.streams = [], 0, [], []
        for by_id in NODES: self.replug(by_id)

    def dev(self, by_id): return os.path.join(self.devdir, NODES[by_id])
    def is_c920(self, dev): return dev != self.dev(WRIST)

    def replug(self, by_id):
        """Power cycle: every control back to what the camera boots with (auto on); light so that manual at auto's numbers
        reproduces auto's picture."""
        dev = self.dev(by_id); c920 = "icSpring" not in by_id
        self.state[dev] = dict(C920 if c920 else ICSPRING)
        s = self.state[dev]
        self.light[dev] = AUTO_MEAN[by_id] / ((s["exposure_time_absolute"] / 78) * 2 ** (s["gain"] / 64)) if c920 else AUTO_MEAN[by_id] / s["exposure_time_absolute"]

    def inactive(self, s, name):
        return any(s.get(flag) == on and name in gated for flag, (on, gated) in GATES.items())

    def mean(self, dev):
        s = self.state[dev]
        if s["auto_exposure"] == 3: return AUTO_MEAN[[b for b, n in NODES.items() if self.dev(b) == dev][0]]
        if self.is_c920(dev): m = self.light[dev] * (s["exposure_time_absolute"] / 78) * 2 ** (s["gain"] / 64)
        else: m = self.light[dev] * s["exposure_time_absolute"] * 2 ** (s["brightness"] / 64)   # the wrist has no gain: its brightness offset scales the picture
        return min(255.0, m)

    def sleep(self, secs):
        """The module's time.sleep: the camera's delayed gain clobber lands while the tool waits."""
        self.sleeps.append(secs)
        for dev in list(self.clobber_pending): self.state[dev]["gain"] = 255
        self.clobber_pending.clear()

    def stream_start(self, dev):
        self.streams.append(dev)
        s = self.state[dev]
        if self.restart_quirk and self.is_c920(dev) and s["auto_exposure"] == 1: s["exposure_time_absolute"] = 312

    def run(self, argv, **kw):
        self.calls += 1
        assert argv[0] == "v4l2-ctl" and argv[1] == "-d", argv
        dev, op = argv[2], argv[3]
        s = self.state.get(dev)
        if s is None or not os.path.exists(dev): return Result(1, "", f"Cannot open device {dev}, exiting.\n")
        if op.startswith("--get-ctrl="):
            names = op.split("=", 1)[1].split(","); self.log.append((dev, "get", names))
            out, err = [], []
            for n in sorted(names, key=lambda n: (n.startswith(("auto_exposure", "exposure", "focus", "pan", "tilt", "zoom")), n)):  # class order, not request order
                if n not in s: err.append(f"unknown control '{n}'"); continue
                v = s[n] + self.jitter.get((dev, n), 0)
                out.append(f"{n}: {v} ({MENU[n][v]})" if n in MENU else f"{n}: {v}")
            return Result(1 if err else 0, "\n".join(out) + "\n", "\n".join(err) + ("\n" if err else ""))
        if op.startswith("--set-ctrl="):
            name, val = op.split("=", 1)[1].split("="); val = int(val); self.log.append((dev, "set", (name, val)))
            if name not in s: return Result(1, "", f"unknown control '{name}'\n")
            if self.inactive(s, name): return Result(255, "", "VIDIOC_S_CTRL: failed: Permission denied\n")
            if name in self.sticky.get(dev, ()): return Result(0)
            if name == "auto_exposure" and self.is_c920(dev) and s[name] == 3 and val == 1: self.clobber_pending.add(dev)
            if name == "exposure_time_absolute" and self.is_c920(dev): val = min(STOPS, key=lambda st: abs(st - val))
            s[name] = val; return Result(0)
        if op == "--list-ctrls":
            self.log.append((dev, "list", None))
            lines = [f"{n:>31s} 0x00980900 (int)    : min=0 max=255 step=1 default={C920.get(n, 0)} value={v}" + ("  flags=inactive" if self.inactive(s, n) else "") for n, v in s.items()]
            return Result(0, "\n".join(lines) + "\n")
        raise AssertionError(f"unexpected v4l2-ctl call {argv}")

    def sets(self, by_id): return [p for d, op, p in self.log if d == self.dev(by_id) and op == "set"]


@pytest.fixture(scope="session")
def cam_lock():
    spec = importlib.util.spec_from_file_location("cam_lock", os.path.join(JETSON, "cam_lock.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


@pytest.fixture
def lab(cam_lock, tmp_path, monkeypatch):
    """Fake /dev tree (by-id symlinks -> videoN files), fake v4l2-ctl, fake frames, settings + busy flag under tmp."""
    dev = tmp_path / "dev"; by_id = dev / "by-id"; by_id.mkdir(parents=True)
    for name, node in NODES.items():
        (dev / node).write_text(""); (by_id / name).symlink_to(dev / node)
    fake = FakeV4L2(str(dev))

    class FakeFrames:
        opened = []
        def __init__(self, d, kind):
            if not os.path.exists(d): raise cam_lock.V4L2Error(f"cannot open {d}")
            self.dev = d; fake.stream_start(d); FakeFrames.opened.append((d, kind, os.path.exists(str(tmp_path / ".cams_busy"))))
        def pump(self, s): pass
        def clip_stats(self, skip=6, n=3): return fake.clip.get(self.dev, (0.0, 10.0))
        def mean(self, skip=6, n=3): return fake.mean(self.dev)
        def close(self): FakeFrames.opened.append((self.dev, "closed"))
    FakeFrames.opened = []

    monkeypatch.setattr(cam_lock, "BY_ID_DIR", str(by_id))
    monkeypatch.setattr(cam_lock, "BUSY_FLAG", str(tmp_path / ".cams_busy"))
    monkeypatch.setattr(cam_lock, "FrameSource", FakeFrames)
    monkeypatch.setattr(cam_lock, "TAKE_WAIT", 0)
    monkeypatch.setattr(cam_lock.subprocess, "run", fake.run)
    monkeypatch.setattr(cam_lock.time, "sleep", fake.sleep)
    for k in ("CAM_TOP", "CAM_FRONT", "CAM_WRIST"): monkeypatch.delenv(k, raising=False)
    settings = tmp_path / "calib" / "cam_settings.json"

    class Lab:
        def __init__(self): self.fake, self.frames, self.settings, self.dev, self.by_id, self.mod, self.tmp = fake, FakeFrames, settings, dev, by_id, cam_lock, tmp_path
        def run(self, *argv):
            return cam_lock.main([argv[0], "--file", str(settings), *argv[1:]])
        def capture(self): return self.run("capture", "--settle", "0")
        def unplug(self, name): (by_id / name).unlink()
        def data(self): return json.load(open(settings))
        def plan(self, name): return self.data()["cameras"][name]["controls"]
        def live(self, name): return fake.state[fake.dev(name)]
        def busy(self): return os.path.exists(str(tmp_path / ".cams_busy"))
    return Lab()


def order(seq, a, b):
    """True if control a is set before control b in a list of (name, value) sets."""
    names = [n for n, _ in seq]
    return a in names and b in names and names.index(a) < names.index(b)


# ---------- capture: file shape, ordering, brightness ----------
def test_capture_writes_one_block_per_camera_keyed_by_by_id(lab, capsys):
    assert lab.capture() == 0
    data = lab.data()
    assert set(data["cameras"]) == {TOP, FRONT, WRIST} and data["captured"] and data["band"] == [60, 190]
    top = data["cameras"][TOP]
    assert top["role"] == "top" and top["kind"] == "c920" and top["device_at_capture"].endswith("video0")
    c = top["controls"]
    assert c["focus_automatic_continuous"] == 0 and c["focus_absolute"] == 35
    assert c["auto_exposure"] == 1 and c["exposure_time_absolute"] == 78, "83 from auto snaps to the camera's 1/128 s stop"
    assert c["exposure_dynamic_framerate"] == 0 and c["white_balance_automatic"] == 0 and c["white_balance_temperature"] == 3366
    assert c["power_line_frequency"] == 2 and c["backlight_compensation"] == 0
    assert all(c[k] == 128 for k in ("brightness", "contrast", "saturation", "sharpness")), "defaults pinned, not altered"
    keys = list(c)                                     # file order is apply order
    assert keys.index("exposure_dynamic_framerate") < keys.index("auto_exposure"), "dynamic framerate off before manual exposure"
    for flag, (_, gated) in GATES.items():
        for g in gated: assert keys.index(flag) < keys.index(g), (flag, g, keys)
    w = data["cameras"][WRIST]
    assert w["controls"]["auto_exposure"] == 1 and w["controls"]["white_balance_automatic"] == 0
    assert w["controls"]["white_balance_temperature"] == 4000 and w["controls"]["power_line_frequency"] == 2, "icSpring ships at 50 Hz; US shop is 60 Hz"
    assert "focus_absolute" not in w["controls"] and "exposure_dynamic_framerate" not in w["controls"] and "gain" not in w["controls"]
    assert w["controls"]["brightness"] == 0 and w["controls"]["contrast"] == 27 and w["controls"]["gamma"] == 67, "the wrist keeps its own defaults, not the C920's"
    assert top["auto_seen"]["auto_exposure"] == 3 and top["auto_mean"] == 116.0, "what auto had chosen and produced is kept for the record"
    assert 100 <= top["target_mean"] <= 140 and abs(top["verified_mean"] - top["target_mean"]) <= 8
    out = capsys.readouterr().out
    assert "wrote" in out and "3 cameras" in out and "top: locked" in out and "picture mean" in out


def test_capture_locked_picture_matches_auto_brightness(lab):
    """The point of the exercise: each locked camera lands within 8 grey levels of what auto produced (clamped to 100-140)."""
    assert lab.capture() == 0
    for cam in (TOP, FRONT, WRIST):
        target = min(140, max(100, AUTO_MEAN[cam]))
        assert abs(lab.fake.mean(lab.fake.dev(cam)) - target) <= 8, cam
        assert lab.live(cam)["auto_exposure"] == 1
    assert lab.plan(FRONT)["gain"] > 109, "front auto sat at 89: the lock raises gain to reach the 100 floor"
    assert lab.plan(WRIST)["exposure_time_absolute"] == 313, "wrist auto was already mid-grey: nothing to tune"


def test_capture_takes_cameras_from_dashboard_and_gives_them_back(lab):
    assert lab.capture() == 0
    assert not lab.busy(), "busy flag removed afterwards"
    opened = [x for x in lab.frames.opened if x[1] != "closed"]
    assert all(busy for _d, _k, busy in opened), "every stream was opened with the dashboard held off"
    assert [k for _d, k, _b in opened] == ["c920", "c920", "icspring"]
    assert len([x for x in lab.frames.opened if x[1] == "closed"]) == 3


def test_capture_leaves_existing_busy_flag_alone(lab):
    open(lab.mod.BUSY_FLAG, "w").close()
    assert lab.capture() == 0 and lab.busy()


def test_capture_survives_gain_clobber_after_leaving_auto(lab):
    """The white-out: ~1 s after auto_exposure 3->1 the C920 forces gain 255. apply waits SETTLE and re-writes."""
    assert lab.capture() == 0
    assert lab.live(TOP)["gain"] != 255 and lab.live(FRONT)["gain"] != 255
    assert lab.fake.sleeps, "apply waited for the camera after the transition"
    assert all(lab.fake.mean(lab.fake.dev(c)) <= 190 for c in (TOP, FRONT))


def test_capture_re_enables_auto_first_when_cameras_are_locked(lab):
    """A second capture (new lighting session) must hand control back to auto before reading, or it would re-read the old lock."""
    assert lab.capture() == 0
    lab.fake.log.clear()
    assert lab.capture() == 0
    sets = lab.fake.sets(TOP)
    assert ("auto_exposure", 3) in sets and ("white_balance_automatic", 1) in sets and ("focus_automatic_continuous", 1) in sets
    d = lab.fake.dev(TOP)
    first_get = next(i for i, (dd, op, _) in enumerate(lab.fake.log) if dd == d and op == "get")
    assert any(i < first_get for i, (dd, op, p) in enumerate(lab.fake.log) if dd == d and op == "set" and p == ("auto_exposure", 3))


def test_capture_clamps_long_exposure_and_raises_gain(lab, capsys):
    lab.live(FRONT)["exposure_time_absolute"] = 500; lab.live(FRONT)["gain"] = 100
    lab.live(WRIST)["exposure_time_absolute"] = 600
    assert lab.capture() == 0
    f, w = lab.plan(FRONT), lab.plan(WRIST)
    assert f["exposure_time_absolute"] <= 312 and w["exposure_time_absolute"] <= 333
    data = lab.data()
    assert any("clamped to 312" in n for n in data["cameras"][FRONT]["notes"]) and data["cameras"][FRONT]["auto_seen"]["exposure_time_absolute"] == 500
    assert any("clamped" in n for n in data["cameras"][WRIST]["notes"])
    assert "clamped to 312" in capsys.readouterr().out
    assert lab.plan(TOP)["exposure_time_absolute"] == 78, "a short exposure is left alone"
    for cam in (FRONT, WRIST): assert 60 <= lab.fake.mean(lab.fake.dev(cam)) <= 190


def test_plan_gain_compensation_never_exceeds_255(lab):
    plan, notes = lab.mod.plan_from_auto("c920", dict(C920, exposure_time_absolute=2000, gain=240))
    assert plan["exposure_time_absolute"] == 312 and plan["gain"] == 255 and notes


def test_capture_tunes_gain_when_light_changed_since_auto_readback(lab):
    """Manual at auto's numbers comes out too dark (light dropped): the loop raises gain until the picture is mid-grey."""
    lab.fake.light[lab.fake.dev(TOP)] /= 3
    assert lab.capture() == 0
    assert lab.plan(TOP)["gain"] > 109 and abs(lab.fake.mean(lab.fake.dev(TOP)) - 116) <= 8


def test_capture_steps_exposure_down_when_gain_hits_zero(lab):
    lab.fake.light[lab.fake.dev(TOP)] *= 12
    assert lab.run("capture", "--settle", "0", "--min-exposure", "top=1") == 0, "floor lowered by hand: the stop-down path still works"
    assert lab.plan(TOP)["exposure_time_absolute"] < 78 and abs(lab.fake.mean(lab.fake.dev(TOP)) - 116) <= 8


def test_capture_refuses_and_rolls_back_a_whiteout(lab, capsys):
    """Even the shortest stop at gain 0 is blown out: that camera goes back to auto and gets no block, so check fails."""
    lab.fake.light[lab.fake.dev(FRONT)] = 1e5
    assert lab.capture() == 1
    out = capsys.readouterr().out
    assert "front: locked picture mean 255 is outside 60..190: ROLLED BACK TO AUTO" in out
    assert lab.live(FRONT)["auto_exposure"] == 3 and lab.live(FRONT)["white_balance_automatic"] == 1
    assert set(lab.data()["cameras"]) == {TOP, WRIST}, "top and wrist are still saved and locked"
    assert lab.run("check") == 1
    assert "front no settings captured" in capsys.readouterr().out
    assert lab.run("check", "--json") == 1
    r = json.loads(capsys.readouterr().out)
    assert r["cameras"]["front"]["status"] == "no settings captured" and r["cameras"]["top"]["ok"]


def test_capture_set_override_seeds_the_wrist_exposure(lab):
    assert lab.run("capture", "--settle", "0", "--set", "wrist:exposure_time_absolute=156") == 0
    d = lab.data()["cameras"][WRIST]
    assert any("overridden to 156" in n for n in d["notes"])
    assert 60 <= lab.fake.mean(lab.fake.dev(WRIST)) <= 190, "the loop still brings the picture to the target from there"
    assert lab.run("capture", "--settle", "0", "--set", "bogus") == 2
    assert lab.run("capture", "--settle", "0", "--set", "wrist:gain=3") == 1, "the wrist has no gain"


def test_capture_flags_light_starved_camera(lab):
    lab.fake.light[lab.fake.dev(FRONT)] /= 15
    assert lab.capture() == 0
    f = lab.plan(FRONT)
    assert f["exposure_time_absolute"] == 312 and f["gain"] >= 240, "went up to the longest 30 fps stop, then nearly all the gain"
    assert any("starved for light" in n for n in lab.data()["cameras"][FRONT]["notes"])
    assert 60 <= lab.fake.mean(lab.fake.dev(FRONT)) <= 190


# ---------- capture: flicker floor ----------
def exposure_of(lab, cam): return lab.plan(cam)["exposure_time_absolute"]


def test_c920_auto_at_38_is_lifted_to_the_flicker_floor_and_gain_pays_for_it(lab):
    for cam in (TOP, FRONT): lab.live(cam)["exposure_time_absolute"] = 38; lab.live(cam)["gain"] = 200
    lab.fake.light[lab.fake.dev(TOP)] = AUTO_MEAN[TOP] / ((38 / 78) * 2 ** (200 / 64))     # auto's numbers still give auto's picture
    assert lab.capture() == 0
    for cam in (TOP, FRONT):
        assert exposure_of(lab, cam) >= 77 and lab.plan(cam)["gain"] < 200, "longer exposure, so gain comes down"
    assert abs(lab.fake.mean(lab.fake.dev(TOP)) - 116) <= 8
    d = lab.data()["cameras"][TOP]
    assert d["exposure_floor"] == 78 and any("exposure floor 78" in n and "60 Hz" in n for n in d["notes"])
    assert any("shorter than the floor" in n for n in d["notes"])


def test_c920_gain_comes_right_down_when_auto_was_short_and_bright(lab):
    lab.live(TOP)["exposure_time_absolute"] = 38; lab.live(TOP)["gain"] = 80
    lab.fake.light[lab.fake.dev(TOP)] = AUTO_MEAN[TOP] / ((38 / 78) * 2 ** (80 / 64))
    assert lab.capture() == 0
    assert exposure_of(lab, TOP) == 78 and lab.plan(TOP)["gain"] <= 30 and abs(lab.fake.mean(lab.fake.dev(TOP)) - 116) <= 8


def test_mains_50_hz_floor_is_156_on_a_c920(lab):
    lab.live(TOP)["exposure_time_absolute"] = 38; lab.live(TOP)["gain"] = 200
    lab.fake.light[lab.fake.dev(TOP)] = AUTO_MEAN[TOP] / ((38 / 78) * 2 ** (200 / 64)) / 4   # dim enough that gain stays above zero
    assert lab.run("capture", "--settle", "0", "--mains-hz", "50") == 0
    assert exposure_of(lab, TOP) >= 156
    d = lab.data()
    assert d["mains_hz"] == 50 and d["cameras"][TOP]["exposure_floor"] == 156 and any("50 Hz" in n for n in d["cameras"][TOP]["notes"])
    assert exposure_of(lab, WRIST) == 313 and d["cameras"][WRIST]["exposure_floor"] is None, "the wrist has no floor unless asked"
    assert lab.run("capture", "--settle", "0", "--mains-hz", "50", "--flicker-floor", "wrist") == 0
    assert exposure_of(lab, WRIST) >= 120 and lab.data()["cameras"][WRIST]["exposure_floor"] == 120, "the wrist floor scales 100 * 60/50"
    with pytest.raises(SystemExit): lab.run("capture", "--mains-hz", "55")    # only 50 and 60 exist


def test_wrist_keeps_auto_exposure_by_default(lab):
    """A forced 100 clipped 46 % of the wrist picture (measured 11 Nov 2026): the wrist floor is opt-in."""
    lab.live(WRIST)["exposure_time_absolute"] = 35
    lab.fake.light[lab.fake.dev(WRIST)] = AUTO_MEAN[WRIST] / 35
    assert lab.capture() == 0
    w = lab.plan(WRIST)
    assert w["exposure_time_absolute"] == 35 and w["brightness"] == 0
    d = lab.data()["cameras"][WRIST]
    assert d["exposure_floor"] is None and not any("floor" in n for n in d["notes"])


def test_wrist_with_flicker_floor_flag_is_lifted_to_100_and_brightness_reaches_the_target(lab):
    lab.live(WRIST)["exposure_time_absolute"] = 35
    lab.fake.light[lab.fake.dev(WRIST)] = AUTO_MEAN[WRIST] / 35
    assert lab.run("capture", "--settle", "0", "--flicker-floor", "wrist") == 0
    w = lab.plan(WRIST)
    assert w["exposure_time_absolute"] >= 100 and w["brightness"] < 0, "longer exposure, darker by the brightness offset"
    assert abs(lab.fake.mean(lab.fake.dev(WRIST)) - 125) <= 8 and -128 <= w["brightness"] <= 127
    d = lab.data()["cameras"][WRIST]
    assert d["exposure_floor"] == 100 and any("exposure floor 100" in n for n in d["notes"]) and any("brightness" in n for n in d["notes"])


def test_min_exposure_override_replaces_the_floor(lab, capsys):
    for cam in (TOP, WRIST): lab.live(cam)["exposure_time_absolute"] = 38 if cam == TOP else 35
    lab.live(TOP)["gain"] = 200
    lab.fake.light[lab.fake.dev(TOP)] = AUTO_MEAN[TOP] / ((38 / 78) * 2 ** (200 / 64))
    lab.fake.light[lab.fake.dev(WRIST)] = AUTO_MEAN[WRIST] / 35
    assert lab.run("capture", "--settle", "0", "--min-exposure", "top=156", "--min-exposure", "wrist=200") == 0
    assert exposure_of(lab, TOP) >= 156 and exposure_of(lab, WRIST) >= 200
    d = lab.data()["cameras"]
    assert d[TOP]["exposure_floor"] == 156 and any("--min-exposure 156" in n for n in d[TOP]["notes"])
    assert d[WRIST]["exposure_floor"] == 200 and d[FRONT]["exposure_floor"] == 78
    assert lab.run("capture", "--settle", "0", "--flicker-floor", "nope") == 2
    assert lab.run("capture", "--settle", "0", "--min-exposure", "nope=5") == 2
    assert lab.run("capture", "--settle", "0", "--min-exposure", "top") == 2


def test_picture_that_cannot_reach_the_range_at_the_floor_rolls_back(lab, capsys):
    """Too bright even at gain 0 / brightness -128 on the shortest allowed exposure: back to auto, no block, never a shorter exposure."""
    lab.fake.light[lab.fake.dev(TOP)] *= 12
    lab.fake.light[lab.fake.dev(WRIST)] *= 40
    assert lab.run("capture", "--settle", "0", "--flicker-floor", "wrist") == 1
    out = capsys.readouterr().out
    assert "top: locked picture mean" in out and "ROLLED BACK TO AUTO" in out and "wrist: locked picture mean" in out
    assert set(lab.data()["cameras"]) == {FRONT}
    for cam in (TOP, WRIST): assert lab.live(cam)["auto_exposure"] == 3
    for cam, floor in ((TOP, 77), (WRIST, 100)):
        assert all(v >= floor for n, v in lab.fake.sets(cam) if n == "exposure_time_absolute" and v > 0), "the tuning loop never wrote a shorter exposure"


@pytest.mark.parametrize("hz,floors", [(60, {TOP: 77, FRONT: 77, WRIST: 100}), (50, {TOP: 156, FRONT: 156, WRIST: 120})])
@pytest.mark.parametrize("auto_exposure", [5, 20, 38, 60])
def test_no_saved_exposure_is_below_the_floor_for_any_kind(lab, hz, floors, auto_exposure):
    """Regression for the 11 Nov 2026 flicker bands: whatever auto chose, neither the saved file nor any write is shorter than the floor."""
    for cam in floors:
        lab.live(cam)["exposure_time_absolute"] = auto_exposure; lab.live(cam)["gain"] = 100
        lab.fake.light[lab.fake.dev(cam)] = AUTO_MEAN[cam] / (((auto_exposure / 78) * 2 ** (100 / 64)) if cam != WRIST else auto_exposure)
    lab.run("capture", "--settle", "0", "--mains-hz", str(hz), "--flicker-floor", "wrist")   # wrist floor is opt-in; some cases may roll back (rc 1); what is saved must still obey the floor
    for cam, floor in floors.items():
        sets = [v for n, v in lab.fake.sets(cam) if n == "exposure_time_absolute"]
        assert sets and all(v >= floor for v in sets), (cam, sets)
        if os.path.exists(lab.settings) and cam in lab.data()["cameras"]: assert lab.plan(cam)["exposure_time_absolute"] >= floor


@pytest.mark.parametrize("clip", [(0.46, 10.0), (0.0, 2.0)], ids=["46pct-saturated", "flat-bright-decile"])
def test_clipping_is_reported_and_exposure_is_not_raised_further(lab, clip):
    """Wrist picture too dark at auto's exposure but the sensor already clips: report it, leave the exposure where it is."""
    w = lab.fake.dev(WRIST); lab.fake.light[w] /= 3; lab.fake.clip[w] = clip
    assert lab.capture() in (0, 1)
    data = lab.data()["cameras"]
    if WRIST in data:
        assert lab.plan(WRIST)["exposure_time_absolute"] <= 313
        assert any("sensor clipping" in n for n in data[WRIST]["notes"])
    assert not any(v > 313 for n, v in lab.fake.sets(WRIST) if n == "exposure_time_absolute"), "never lengthened while clipping"
    assert not any("sensor clipping" in n for n in data[TOP]["notes"]), "clean cameras are not flagged"


def test_clipping_c920_stays_on_its_stop_and_is_flagged(lab):
    t = lab.fake.dev(TOP); lab.fake.light[t] /= 3; lab.fake.clip[t] = (0.2, 10.0)
    assert lab.capture() in (0, 1)
    assert not any(v > 78 for n, v in lab.fake.sets(TOP) if n == "exposure_time_absolute"), "the exposure stays on the floor stop; gain does the work"
    assert any("sensor clipping" in n for n in lab.data()["cameras"][TOP]["notes"])


# ---------- apply: ordering, inactive-while-auto, refusals ----------
def test_apply_switches_auto_off_before_writing_manual_values(lab, capsys):
    assert lab.capture() == 0
    for cam in NODES: lab.fake.replug(cam)                    # everything back on auto
    lab.fake.log.clear(); capsys.readouterr()
    assert lab.run("apply") == 0
    for cam in (TOP, FRONT):
        s = lab.fake.sets(cam)
        assert order(s, "focus_automatic_continuous", "focus_absolute") and order(s, "auto_exposure", "exposure_time_absolute")
        assert order(s, "white_balance_automatic", "white_balance_temperature") and order(s, "exposure_dynamic_framerate", "auto_exposure")
        live = lab.live(cam)
        assert live["auto_exposure"] == 1 and live["focus_automatic_continuous"] == 0 and live["white_balance_automatic"] == 0
        assert live["exposure_dynamic_framerate"] == 0 and live["power_line_frequency"] == 2 and live["gain"] != 255
    assert lab.live(WRIST)["power_line_frequency"] == 2 and lab.live(WRIST)["auto_exposure"] == 1
    assert all(not lab.mod.compare(lab.plan(c), lab.live(c)) for c in NODES), "every saved value took"
    assert capsys.readouterr().out.startswith("cams locked: top ok; front ok; wrist ok")


def test_apply_reports_control_rejected_while_inactive(lab, capsys, monkeypatch):
    """A file written in the wrong order (manual value before its auto flag) hits EACCES on the first pass; with a single
    write round apply must say which control and why. The normal retry round recovers, since the flag is off by then."""
    lab.capture()
    data = lab.data(); c = data["cameras"][TOP]["controls"]
    bad = {"focus_absolute": 100}; bad.update({k: v for k, v in c.items() if k != "focus_absolute"})
    data["cameras"][TOP]["controls"] = bad; json.dump(data, open(lab.settings, "w"))
    lab.fake.replug(TOP); capsys.readouterr()
    monkeypatch.setattr(lab.mod, "ROUNDS", 1)
    assert lab.run("apply") == 1
    out = capsys.readouterr().out
    assert "cams NOT locked (apply)" in out and "top focus_absolute=35 (want 100)" in out and "Permission denied" in out
    assert "front ok" in out and "wrist ok" in out
    monkeypatch.setattr(lab.mod, "ROUNDS", 3)
    assert lab.run("apply") == 0 and lab.live(TOP)["focus_absolute"] == 100


def test_apply_reports_control_that_will_not_take(lab, capsys):
    lab.capture(); lab.fake.replug(FRONT)
    lab.fake.sticky[lab.fake.dev(FRONT)] = {"exposure_dynamic_framerate"}
    assert lab.run("apply") == 1
    out = capsys.readouterr().out
    assert "front exposure_dynamic_framerate=1 (want 0)" in out and "top ok" in out
    assert lab.live(FRONT)["auto_exposure"] == 1, "the other controls were still applied"


def test_apply_verify_measures_and_rolls_back_a_whiteout(lab, capsys):
    """Lights changed after capture: apply --verify streams the camera, sees a white picture, hands it back to auto."""
    lab.capture(); capsys.readouterr()
    lab.fake.light[lab.fake.dev(TOP)] *= 50
    assert lab.run("apply", "--verify") == 1
    out = capsys.readouterr().out
    assert "top picture_mean=255 (want 60..190)" in out and "rolled back to auto" in out and "front ok" in out
    assert lab.live(TOP)["auto_exposure"] == 3 and lab.live(FRONT)["auto_exposure"] == 1
    assert not lab.busy() and lab.frames.opened, "streamed with the dashboard held off, flag removed afterwards"
    lab.fake.light[lab.fake.dev(TOP)] /= 50
    assert lab.run("apply", "--verify") == 0 and lab.live(TOP)["auto_exposure"] == 1


def test_apply_verify_fixes_the_stream_start_exposure_jump(lab):
    """Opening the stream makes the C920 come up at 312; the write during the stream puts it back and the picture is fine."""
    lab.capture()
    assert lab.run("apply", "--verify") == 0
    assert lab.live(TOP)["exposure_time_absolute"] == 78 and lab.fake.streams


# ---------- check ----------
def test_round_trip_capture_apply_check_and_replug(lab, capsys):
    assert lab.capture() == 0 and lab.run("check") == 0, "capture leaves the cameras locked"
    assert capsys.readouterr().out.splitlines()[-1] == "cams locked: top ok; front ok; wrist ok"
    lab.fake.replug(FRONT)                                   # USB re-plug: the camera forgets everything
    assert lab.run("check") == 1
    out = capsys.readouterr().out
    assert "cams NOT locked (check)" in out and "front" in out and "auto_exposure=3 (want 1)" in out and "top ok" in out
    assert lab.run("apply") == 0 and lab.run("check") == 0


def test_check_tolerates_the_c920_off_by_one_exposure_readback(lab):
    lab.capture()
    lab.fake.jitter[(lab.fake.dev(TOP), "exposure_time_absolute")] = -1       # 78 reads back as 77
    assert lab.run("check") == 0
    lab.fake.jitter[(lab.fake.dev(TOP), "exposure_time_absolute")] = -5
    assert lab.run("check") == 1


def test_check_is_one_read_per_camera(lab):
    lab.capture()
    lab.fake.calls = 0
    assert lab.run("check") == 0
    assert lab.fake.calls == 3


def test_check_json(lab, capsys):
    lab.capture()
    lab.live(WRIST)["white_balance_automatic"] = 1; capsys.readouterr()
    assert lab.run("check", "--json") == 1
    r = json.loads(capsys.readouterr().out)
    assert r["ok"] is False and r["cameras"]["top"]["ok"] is True and r["cameras"]["top"]["present"] is True
    w = r["cameras"]["wrist"]
    assert w["ok"] is False and w["diffs"] == {"white_balance_automatic": {"want": 0, "got": 1}} and w["by_id"] == WRIST
    assert w["device"].endswith("video2") and r["captured"]


def test_check_without_settings_file(lab, capsys):
    assert lab.run("check") == 1
    assert "no settings file" in capsys.readouterr().out
    assert lab.run("check", "--json") == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


# ---------- unplugged camera ----------
def test_unplugged_camera_is_reported_not_fatal_unless_strict(lab, capsys):
    lab.capture()
    lab.unplug(FRONT)
    assert lab.run("check") == 0
    assert "front unplugged" in capsys.readouterr().out
    assert lab.run("check", "--strict") == 1
    assert lab.run("apply") == 0 and lab.run("apply", "--strict") == 1
    assert lab.run("check", "--json") == 0
    r = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert r["cameras"]["front"] == {"role": "front", "by_id": FRONT, "device": None, "present": False, "ok": True, "diffs": {}, "status": "unplugged"}


def test_capture_skips_unplugged_camera(lab, capsys):
    lab.unplug(WRIST)
    assert lab.capture() == 0
    assert set(lab.data()["cameras"]) == {TOP, FRONT}
    assert "wrist: usb-icSpring_icspring_camera-video-index0 not plugged in" in capsys.readouterr().out
    assert lab.run("check") == 0, "an unplugged camera with no block is a note, not a failure"
    assert lab.run("capture", "--settle", "0", "--strict") == 1


# ---------- --only (udev passes the kernel name) ----------
@pytest.mark.parametrize("only", ["video4", "/dev/video4", "front", FRONT])
def test_only_limits_apply_to_one_camera(lab, only, monkeypatch, capsys):
    lab.capture()
    for cam in NODES: lab.fake.replug(cam)
    lab.fake.log.clear(); capsys.readouterr()
    if only.startswith("/dev") or only.startswith("video"):
        # the by-id symlink points into the fake /dev, so map the kernel name there
        monkeypatch.setattr(lab.mod.os.path, "realpath", lambda p, _r=os.path.realpath: _r(p.replace("/dev/", str(lab.dev) + "/", 1) if p.startswith("/dev/") else p))
    assert lab.run("apply", "--only", only) == 0
    assert lab.fake.sets(FRONT) and not lab.fake.sets(TOP) and not lab.fake.sets(WRIST)
    assert capsys.readouterr().out.strip() == "cams locked: front ok"
    assert lab.run("apply", "--only", "video9") == 1


# ---------- guard ----------
class StopGuard(Exception):
    pass


def run_guard(lab, ticks, interval="0", on_tick=None):
    """Run the guard loop for `ticks` idle sleeps (apply's own settle sleeps do not count); on_tick = {tick: callback}."""
    count = {"n": 0}; real = lab.fake.sleep
    def sleep(s):
        real(s)
        if s == 0.0:
            count["n"] += 1
            if on_tick and count["n"] in on_tick: on_tick[count["n"]]()
            if count["n"] >= ticks: raise StopGuard()
    lab.mod.time.sleep = sleep
    try:
        with pytest.raises(StopGuard): lab.run("guard", "--interval", interval)
    finally: lab.mod.time.sleep = real


def test_guard_relocks_after_stream_start_and_replug(lab, capsys):
    lab.capture(); capsys.readouterr()
    lab.live(TOP)["exposure_time_absolute"] = 312                 # lerobot opened the camera: the C920 re-derived exposure
    lab.fake.replug(WRIST)                                        # and the wrist was re-plugged
    run_guard(lab, ticks=2)
    out = capsys.readouterr().out
    assert "guard started" in out and "settings loaded" in out
    assert re.search(r"top: re-locked exposure_time_absolute 312->78", out) and re.search(r"wrist: re-locked .*auto_exposure 3->1", out)
    assert lab.live(TOP)["exposure_time_absolute"] == 78 and lab.live(WRIST)["auto_exposure"] == 1
    assert out.count("re-locked") == 2, "a quiet tick logs nothing"


def test_guard_reports_unplug_once_and_relocks_on_return(lab, capsys):
    lab.capture(); lab.unplug(FRONT); capsys.readouterr()
    def back(): (lab.by_id / FRONT).symlink_to(lab.dev / NODES[FRONT]); lab.fake.replug(FRONT)
    run_guard(lab, ticks=5, on_tick={3: back})
    out = capsys.readouterr().out
    assert out.count("front: unplugged") == 1, "a camera that stays unplugged is reported once, not every tick"
    assert "front: back on" in out and "front: re-locked" in out and lab.live(FRONT)["auto_exposure"] == 1


def test_guard_waits_for_settings_file(lab, capsys):
    run_guard(lab, ticks=2)
    assert capsys.readouterr().out.count("waiting for the settings file") == 1


def test_guard_is_single_instance(lab, capsys):
    lab.capture()
    import fcntl
    lock = open(str(lab.settings) + ".guard.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert lab.run("guard") == 0
    assert "guard already running" in capsys.readouterr().out
    lock.close()


# ---------- parser + env ----------
def test_get_ctrls_parses_reordered_output_and_menus(lab):
    vals = lab.mod.get_ctrls(lab.fake.dev(TOP), ["exposure_time_absolute", "auto_exposure", "power_line_frequency", "gain"])
    assert vals == {"exposure_time_absolute": 83, "auto_exposure": 3, "power_line_frequency": 2, "gain": 109}
    with pytest.raises(lab.mod.V4L2Error): lab.mod.get_ctrls(lab.fake.dev(TOP), ["nosuchctrl"])
    with pytest.raises(lab.mod.V4L2Error): lab.mod.get_ctrls(str(lab.dev / "video9"), ["gain"])


def test_env_sh_paths_override_defaults(lab, monkeypatch):
    monkeypatch.setenv("CAM_TOP", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_DEADBEEF-video-index0")
    assert lab.mod.cameras_from_env()["top"] == "usb-046d_HD_Pro_Webcam_C920_DEADBEEF-video-index0"
    assert lab.mod.cameras_from_env()["wrist"] == WRIST
    assert lab.mod.kind_of("usb-046d_HD_Pro_Webcam_C920_X-video-index0") == "c920" and lab.mod.kind_of(WRIST) == "icspring"
    assert lab.mod.kind_of("usb-other-video-index0") is None


def test_show_lists_live_values(lab, capsys):
    assert lab.run("show") == 0
    out = capsys.readouterr().out
    assert "top (" in out and re.search(r"exposure_time_absolute\s+83\s+inactive", out) and "wrist (" in out


def test_frame_source_needs_opencv_message(monkeypatch):
    """Under /usr/bin/python3 there is no cv2: the real FrameSource must say so instead of crashing."""
    import builtins
    real_import = builtins.__import__
    def no_cv2(name, *a, **k):
        if name == "cv2": raise ImportError("No module named cv2")
        return real_import(name, *a, **k)
    spec = importlib.util.spec_from_file_location("cam_lock_real", os.path.join(JETSON, "cam_lock.py"))
    real_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(real_mod)
    monkeypatch.setattr(builtins, "__import__", no_cv2)
    with pytest.raises(real_mod.V4L2Error, match="needs OpenCV"):
        real_mod.FrameSource("/dev/null", "c920")


# ---------- env.sh: so101-record refuses when the check fails, starts the guard when it passes ----------
ENV_SH = os.path.join(JETSON, "env.sh")


@pytest.fixture
def shell(tmp_path):
    """A fake $HOME/so101 so env.sh can be sourced on the Mac: empty venv activate, a stub cam_lock.py whose `check`
    passes only while ~/so101/.locked exists (every invocation is appended to calls.log), and a PATH shim lerobot-record
    that records its argv."""
    home = tmp_path / "home"; so = home / "so101"; (so / ".venv" / "bin").mkdir(parents=True)
    (so / ".venv" / "bin" / "activate").write_text("")
    calls = so / "calls.log"
    (so / "cam_lock.py").write_text(textwrap.dedent(f"""
        import os, sys
        open({str(calls)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
        if sys.argv[1] == "guard": sys.exit(0)
        assert sys.argv[1:3] == ["check", "--strict"], sys.argv
        ok = os.path.exists(os.path.expanduser("~/so101/.locked"))
        print("cams locked: top ok; front ok; wrist ok" if ok else "cams NOT locked (check): front auto_exposure=3 (want 1)")
        sys.exit(0 if ok else 1)
    """))
    bin_ = tmp_path / "bin"; bin_.mkdir(); called = tmp_path / "lerobot-record.argv"
    (bin_ / "lerobot-record").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{called}"\n'); (bin_ / "lerobot-record").chmod(0o755)
    env = dict(os.environ, HOME=str(home), PATH=f"{bin_}:{os.path.dirname(sys.executable)}:/usr/bin:/bin")

    class Shell:
        def __init__(self): self.home, self.so, self.called, self.calls = home, so, called, calls
        def run(self, cmd):
            r = subprocess.run(["bash", "-c", f"source {ENV_SH}; {cmd}"], env=env, capture_output=True, text=True, timeout=30)
            return r.returncode, r.stdout + r.stderr
        def lock(self): (so / ".locked").write_text("")
        def busy(self): return [p.name for p in so.iterdir() if p.name in (".cams_busy", ".arms_busy")]
        def invocations(self): return calls.read_text().splitlines() if calls.exists() else []
    return Shell()


@pytest.mark.parametrize("fn", ['so101-record cube "pick the cube" 2', "so101-record-more cube 2"])
def test_so101_record_refuses_when_cameras_are_not_locked(shell, fn):
    rc, out = shell.run(fn)
    assert rc != 0 and "cams NOT locked" in out and "REFUSING TO RECORD" in out and "so101-cams-lock" in out
    assert not shell.called.exists(), "lerobot-record was never started"
    assert shell.busy() == [], "no busy flags left behind"
    assert shell.invocations() == ["check --strict"], "no guard started either"


@pytest.mark.parametrize("fn, needle", [('so101-record cube "pick the cube" 2', "--dataset.repo_id=lastlight/cube"),
                                        ("so101-record-more cube 2", "--resume=true")])
def test_so101_record_runs_when_cameras_are_locked(shell, fn, needle):
    shell.lock()
    rc, out = shell.run(fn)
    assert rc == 0, out
    argv = shell.called.read_text().splitlines()
    assert needle in argv and "--dataset.num_episodes=2" in argv
    assert shell.busy() == [], "busy flags released after the run"
    assert shell.invocations()[:2] == ["check --strict", "guard"], "the guard is (re)started so the stream-start quirk is covered while lerobot streams"


def test_so101_stop_kills_every_motion_script():
    """The wall's KILL button runs so101-stop; every script that can move the arm must be in its pattern (9 Nov: goto.py and
    pick_seq.py were not, and TC's kill button did nothing during a scripted move)."""
    import os, re
    env = open(os.path.join(JETSON, "env.sh")).read()
    m = re.search(r"so101-stop\(\) \{(.*?)\n", env); assert m, "so101-stop missing"
    body = m.group(1)
    for script in ("goto", "pick_seq", "dance", "demo_wave", "nudge", "scripted_record", "scan_pose", "free_arm"):
        assert re.search(rf"\(.*\b{script}\b.*\)", body), f"{script}.py is not killed by so101-stop"
    for p in ("goto.py", "pick_seq.py"):
        assert "signal.SIGTERM" in open(os.path.join(JETSON, p)).read(), f"{p} must unwind cleanly on SIGTERM"
