#!/usr/bin/env python3
"""cam_lock.py · freeze the SO-101 cameras' exposure, white balance and focus so the arm entering the frame cannot change them.

    cam_lock.py capture [--settle S] [--set ROLE:CTRL=VAL ...]
                      per camera: take the camera from the dashboard, stream it in the mode recording uses, hand it back
                      to auto, let auto settle under today's light, read what it chose and the frame brightness it produced,
                      lock the camera, then tune gain (C920) or exposure (wrist) until the locked picture matches auto's
                      brightness. Writes what the cameras accepted to calib/cam_settings.json (one block per camera, keyed
                      by the /dev/v4l/by-id name). A camera whose locked picture ends outside BAND (60..190 mean) is put
                      back on auto and reported; it gets no block, so `check` fails until it is captured again.
                      Needs OpenCV: run with the venv python (`source ~/so101/env.sh` does that). The cameras are LOCKED afterwards.
                      Exposure is never locked below one light-flicker period: --mains-hz 60|50 (default 60), --min-exposure ROLE=N.
    cam_lock.py apply [--verify] [--wait S]
                      write the saved values (auto flag off first, then the manual value), verify the controls, exit 1 on a
                      miss. With --verify also stream each camera and measure the picture; outside BAND the camera is rolled
                      back to auto and apply fails. Plain apply is stdlib-only (udev rule, guard).
    cam_lock.py check [--json]     compare live values with the file; exit 0 only if every camera matches (< 1 s)
    cam_lock.py guard [--interval S]
                      stay resident: re-apply within a second whenever a camera drifts (stream start, re-plug), log each fix.
                      One instance per settings file (flock); a second one exits at once. This is the systemd user service.
    cam_lock.py show               dump the live values of the controls this tool cares about

Common options: --file PATH (default <this dir>/calib/cam_settings.json), --only DEV (one camera: role, by-id name or
/dev/videoN; the udev rule passes the kernel name), --strict (an unplugged camera is a failure instead of a note).

Why: the C920s run auto-exposure, auto white balance and continuous autofocus; the icSpring wrist camera runs auto-exposure
and auto white balance. With the arm moving through the frame the camera re-exposes on the arm, so image brightness becomes
a function of arm pose and the policy learns that instead of the scene.

What the hardware does (measured on the lab's C920s, USB id 046d:08e5, 9 Nov 2026):
  * a manual control is flagged `inactive` while its auto flag is on and the driver refuses writes (EACCES), so every
    auto flag is switched off before the value it gates is written (the file keeps that order);
  * manual exposure snaps to 1/2^n-second stops (3 5 10 20 39 78 156 312 625 1250 2047, in 100 us units) and the same
    stop reads back as 77 or 78, so exposure_time_absolute is compared with a tolerance of 2;
  * about a second after auto_exposure goes 3 -> 1 the camera forces gain to 255 (a white picture), clobbering a gain
    written in between, so `apply` waits SETTLE seconds after a transition and re-writes what moved (up to ROUNDS times);
  * at every stream start in Manual Mode the camera re-derives exposure (78 came up as 156, then 312); a write issued
    while the stream runs holds for that stream. lerobot opens the cameras itself, hence `guard`.
Flicker floor (measured 11 Nov 2026): the room lights flicker at twice the mains frequency (120 Hz on 60 Hz = 8.3 ms). An exposure
shorter than one flicker period bands the picture (auto chose 38 on the C920s and 35 on the wrist: 16 and 35 grey levels peak to
peak), so on a C920 capture never locks below a floor, whatever auto chose: the smallest stop >= one period (77/78 at 60 Hz,
156 at 50 Hz); gain is then tuned back down. The icSpring wrist has no gain and at >= 90 its sensor clips in this light (46 % of
pixels at 255), so its floor is OPT-IN: `--flicker-floor wrist` (100 units at 60 Hz, scaled by 60/mains_hz) or `--min-exposure
wrist=N`; brightness offset then does the darkening. `--mains-hz 50|60`. Every camera is checked for sensor clipping (> 5 % of
pixels >= 250, or the brightest decile's local std < 3): reported in the notes and the exposure is not raised further.
Recording runs at 30 fps, so a C920 exposure must stay <= 333 (one frame); the longest stop that fits is 312. If auto
picked more, capture starts from 312 with gain raised instead (~64 gain steps per stop) and the tuning loop takes it
from there; the wrist exposure is capped at 333 as well.

Cameras are found through /dev/v4l/by-id and resolved to /dev/videoN with realpath before v4l2-ctl is called. Device
paths come from CAM_TOP / CAM_FRONT / CAM_WRIST when set (env.sh), else the lab defaults below.
"""
import argparse, datetime, fcntl, json, math, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(HERE, "calib", "cam_settings.json")
BY_ID_DIR = "/dev/v4l/by-id"
V4L2_CTL = "v4l2-ctl"
PROC = "/proc"
BUSY_FLAG = os.path.expanduser("~/so101/.cams_busy")   # the Mac dashboard lets go of the cameras while this exists

# role -> by-id name; env.sh's CAM_* variables override (they hold the full /dev/v4l/by-id/... path)
DEFAULT_CAMERAS = {
    "top": "usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0",
    "front": "usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0",
    "wrist": "usb-icSpring_icspring_camera-video-index0",
}
MODES = {"c920": ("MJPG", 640, 360, 30), "icspring": (None, 640, 480, 30)}   # what SO101_CAMS in env.sh records with

EXPOSURE_MAX_30FPS = 333                                        # 100 us units: one frame at 30 fps
C920_STOPS = [3, 5, 10, 20, 39, 78, 156, 312, 625, 1250, 2047]  # what the C920 actually exposes at, 1/4096 s .. 1/5 s
EXPOSURE_CLAMP = 312                                            # longest stop inside a 30 fps frame
GAIN_STEPS_PER_STOP = 64                                        # first guess for the C920; the tuning loop refines it
MAINS_HZ = 60                                                   # US shop; 50 is allowed (--mains-hz)
FLOOR_TOL = 7                                                   # 100 us units: 78 (7.8 ms) counts as one 60 Hz period (8.3 ms)
WRIST_FLOOR_60 = 100                                            # icSpring floor at 60 Hz mains when opted in: 78..87 banded, 90..100 clean but clips
CLIP_LEVEL, CLIP_FRAC, CLIP_STD = 250, 0.05, 3.0               # sensor clipping: > 5 % of pixels >= 250, or brightest-decile local std < 3
BRIGHTNESS_RANGE = (-128, 127)                                  # icSpring brightness offset
BRIGHTNESS_STEPS_PER_STOP = 64                                  # first guess for the wrist brightness offset; the loop refines it
TOLERANCE = {"exposure_time_absolute": 2}                       # the same stop reads back 77 or 78
SETTLE = 1.5                                                    # s the C920 needs after leaving auto before a gain write sticks
ROUNDS = 3                                                      # write / verify rounds in apply
AUTO_FLAGS = ("focus_automatic_continuous", "auto_exposure", "white_balance_automatic")
TARGET_MIN, TARGET_MAX, TARGET_TOL = 100.0, 140.0, 8.0          # locked picture aims at auto's mean, clamped to mid-grey
BAND = (60.0, 190.0)                                            # outside this the lock is refused / rolled back to auto
TUNE_ROUNDS, TUNE_WAIT = 8, 0.5                                 # gain/exposure iterations and the pause before re-measuring
TAKE_WAIT, OPEN_RETRY = 2.0, 6.0                                # s for the dashboard's ffmpeg to let go, s to retry the open

# Per camera kind: the auto flags with their "auto on" / "auto off" values and the manual controls each flag gates, the
# fixed values we always want, and the controls we merely pin at whatever they are (defaults unless someone changed them).
KINDS = {
    "c920": {
        "autos": [  # (flag, auto-on value, auto-off value, manual controls it gates)
            ("focus_automatic_continuous", 1, 0, ["focus_absolute"]),
            ("auto_exposure", 3, 1, ["exposure_time_absolute"]),
            ("white_balance_automatic", 1, 0, ["white_balance_temperature"]),
        ],
        "pre": {"exposure_dynamic_framerate": 0},                # written before the auto flags go off
        "fixed": {"power_line_frequency": 2, "backlight_compensation": 0},
        "pin": ["gain", "brightness", "contrast", "saturation", "sharpness"],
        "stops": C920_STOPS,
    },
    "icspring": {
        "autos": [
            ("auto_exposure", 3, 1, ["exposure_time_absolute"]),
            ("white_balance_automatic", 1, 0, ["white_balance_temperature"]),
        ],
        "pre": {},
        "fixed": {"power_line_frequency": 2},
        "pin": ["brightness", "contrast", "saturation", "sharpness", "gamma", "hue"],
        "stops": None,
    },
}


def kind_of(by_id):
    n = by_id.lower()
    if "c920" in n: return "c920"
    if "icspring" in n: return "icspring"
    return None


def cameras_from_env():
    """role -> by-id name, honouring CAM_TOP / CAM_FRONT / CAM_WRIST from env.sh."""
    cams = {}
    for role, default in DEFAULT_CAMERAS.items():
        p = os.environ.get("CAM_" + role.upper())
        cams[role] = os.path.basename(p) if p else default
    return cams


def dev_path(by_id):
    """/dev/videoN behind a by-id name, or None when the camera is not plugged in."""
    link = os.path.join(BY_ID_DIR, by_id)
    if not os.path.exists(link): return None
    return os.path.realpath(link)


class V4L2Error(Exception):
    pass


def v4l2(dev, *args):
    r = subprocess.run([V4L2_CTL, "-d", dev, *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


_LINE = re.compile(r"^\s*(\w+)\s*:\s*(-?\d+)")


def get_ctrls(dev, names):
    """{name: int}. v4l2-ctl reorders its output by control class and prints menus as '3 (Aperture Priority Mode)'."""
    rc, out, err = v4l2(dev, "--get-ctrl=" + ",".join(names))
    vals = {}
    for line in (out + "\n" + err).splitlines():
        m = _LINE.match(line)
        if m and m.group(1) in names: vals[m.group(1)] = int(m.group(2))
    if rc != 0 and not vals: raise V4L2Error((err or out).strip().splitlines()[-1] if (err or out).strip() else f"v4l2-ctl rc={rc}")
    return vals


def set_ctrl(dev, name, value):
    """One control per call: a batch write fails as a whole when any member is still flagged inactive."""
    rc, out, err = v4l2(dev, f"--set-ctrl={name}={value}")
    if rc != 0: raise V4L2Error(f"{name}={value}: {(err or out).strip() or f'rc={rc}'}")


def list_ctrls(dev):
    """{name: {'value': int, 'inactive': bool, 'default': int|None}} from --list-ctrls (used by `show`)."""
    rc, out, err = v4l2(dev, "--list-ctrls")
    ctrls = {}
    for line in out.splitlines():
        m = re.match(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:(.*)$", line)
        if not m: continue
        rest = m.group(3)
        v = re.search(r"value=(-?\d+)", rest); d = re.search(r"default=(-?\d+)", rest)
        ctrls[m.group(1)] = {"value": int(v.group(1)) if v else None, "default": int(d.group(1)) if d else None,
                             "inactive": "inactive" in rest}
    return ctrls


def controls_of(kind):
    k = KINDS[kind]
    names = list(k["pre"])
    for flag, _on, _off, manual in k["autos"]: names += [flag] + manual
    names += list(k["fixed"]) + k["pin"]
    return names


def compare(controls, got):
    """{ctrl: {'want', 'got'}} for every saved control the camera does not hold (within TOLERANCE)."""
    return {n: {"want": int(v), "got": got.get(n)} for n, v in controls.items()
            if got.get(n) is None or abs(got[n] - int(v)) > TOLERANCE.get(n, 0)}


def log(msg):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def release_to_auto(dev, kind):
    """Rollback: hand the camera back to its auto algorithms (never raises)."""
    for flag, on, _off, _manual in KINDS[kind]["autos"]:
        try: set_ctrl(dev, flag, on)
        except V4L2Error: pass


# ---------------------------------------------------------------- frames (capture / apply --verify only)
class Cams:
    """Take the cameras from the Mac dashboard for the duration (its ffmpeg exits while BUSY_FLAG exists)."""
    def __enter__(self):
        self.had = os.path.exists(BUSY_FLAG)
        if not self.had:
            open(BUSY_FLAG, "w").close(); time.sleep(TAKE_WAIT)
        return self
    def __exit__(self, *exc):
        if not self.had:
            try: os.remove(BUSY_FLAG)
            except OSError: pass


class FrameSource:
    """Frames straight from the camera in the mode recording uses. Needs OpenCV (the Jetson venv has it)."""
    def __init__(self, dev, kind):
        try: import cv2
        except ImportError: raise V4L2Error("frame measurement needs OpenCV (cv2): run with the venv python (`source ~/so101/env.sh` first)") from None
        self.cv2 = cv2; fourcc, w, h, fps = MODES[kind]
        deadline = time.time() + OPEN_RETRY
        while True:
            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if self.cap.isOpened(): break
            if time.time() > deadline: raise V4L2Error(f"cannot open {dev} for frames (another program holds it?)")
            time.sleep(0.5)
        if fourcc: self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = self.cap.read()
        if not ok: self.cap.release(); raise V4L2Error(f"{dev} opened but gives no frames")
    def pump(self, seconds):
        """Keep frames flowing (auto only adapts on frames)."""
        t0 = time.time()
        while time.time() - t0 < seconds: self.cap.read()
    def mean(self, skip=6, n=3):
        """Mean grey level of the picture, after skipping frames still in flight from before the last control write."""
        vals = []
        for i in range(skip + n):
            ok, f = self.cap.read()
            if ok and i >= skip: vals.append(float(self.cv2.cvtColor(f, self.cv2.COLOR_BGR2GRAY).mean()))
        if not vals: raise V4L2Error("no frames")
        return sum(vals) / len(vals)
    def clip_stats(self, skip=6, n=3):
        """(fraction of pixels >= CLIP_LEVEL, local std of the brightest decile) averaged over n frames."""
        cv2 = self.cv2; fr, sd = [], []
        for i in range(skip + n):
            ok, f = self.cap.read()
            if not ok or i < skip: continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype("float32"); fr.append(float((g >= CLIP_LEVEL).mean()))
            hi = g >= max(float(sorted(g.ravel())[int(g.size * 0.9)]), 1.0)             # brightest decile
            sd.append(float((g - cv2.blur(g, (5, 5)))[hi].std()) if hi.any() else 0.0)  # why: clipped regions are flat
        if not fr: raise V4L2Error("no frames")
        return sum(fr) / len(fr), sum(sd) / len(sd)
    def close(self): self.cap.release()


def clipping(src):
    """(is the sensor clipping, description) from src.clip_stats(); (False, '') when the source cannot tell."""
    if not hasattr(src, "clip_stats"): return False, ""
    frac, std = src.clip_stats()
    return (frac > CLIP_FRAC or std < CLIP_STD), f"{frac * 100:.0f} % of pixels >= {CLIP_LEVEL}, brightest-decile local std {std:.1f}"


# ---------------------------------------------------------------- capture
def snap(value, stops):
    return min(stops, key=lambda s: abs(s - value)) if stops else value


def exposure_floor(kind, mains_hz=MAINS_HZ, override=None, opt_in=False):
    """(shortest exposure capture may lock, note saying why). One flicker period = 1/(2*mains_hz) s = 10000/(2*mains_hz) units.
    The wrist (no gain, clips at the period) has no floor, (0, None), unless `override` or `opt_in` asks for one."""
    period = 10000.0 / (2 * mains_hz); stops = KINDS[kind]["stops"]
    if not stops and override is None and not opt_in: return 0, None   # why: forced 100 clipped 46 % of the wrist picture
    if override is not None: raw, why = override, f"--min-exposure {override}"
    elif stops: raw, why = period, f"one flicker period at {mains_hz} Hz mains ({period / 10:.1f} ms)"
    else: raw, why = WRIST_FLOOR_60 * 60.0 / mains_hz, f"measured wrist floor for {mains_hz} Hz mains (--flicker-floor; may clip)"
    if stops: floor = min([s for s in stops if s >= raw - FLOOR_TOL] or [stops[-1]])   # why: manual exposure only exists at stops
    else: floor = int(round(raw))
    floor = min(floor, EXPOSURE_CLAMP if stops else EXPOSURE_MAX_30FPS)                 # why: never past one 30 fps frame
    return floor, f"exposure floor {floor} ({floor / 10:.1f} ms): {why}"


def plan_from_auto(kind, seen, overrides=None, floor=0, floor_note=None):
    """Turn what auto chose (`seen`, {control: value}) into the ordered manual settings and a list of notes.
    `floor` is the shortest exposure allowed (see exposure_floor); a shorter one is lifted and brightness compensated."""
    k = KINDS[kind]; plan = dict(k["pre"]); notes = []
    for flag, _on, off, manual in k["autos"]:
        plan[flag] = off
        for m in manual: plan[m] = seen[m]
    plan.update(k["fixed"])
    for m in k["pin"]: plan[m] = seen[m]
    for n, v in (overrides or {}).items():
        if n not in plan: raise ValueError(f"{n} is not a control this tool manages for a {kind}")
        plan[n] = int(v); notes.append(f"{n} overridden to {v} (--set)")
    e = plan["exposure_time_absolute"]
    if k["stops"]:
        s = snap(e, k["stops"])
        if s > EXPOSURE_MAX_30FPS:
            g = plan.get("gain", 0)
            ng = min(255, g + int(round(GAIN_STEPS_PER_STOP * math.log2(e / EXPOSURE_CLAMP))))
            plan["exposure_time_absolute"] = EXPOSURE_CLAMP; plan["gain"] = ng
            notes.append(f"auto exposure {e} ({e / 10:.1f} ms) exceeds the 30 fps ceiling of {EXPOSURE_MAX_30FPS}: clamped to "
                         f"{EXPOSURE_CLAMP} and gain raised {g} -> {ng} (~{GAIN_STEPS_PER_STOP} steps/stop); check the picture")
        else: plan["exposure_time_absolute"] = s
    elif e > EXPOSURE_MAX_30FPS:
        plan["exposure_time_absolute"] = EXPOSURE_MAX_30FPS
        notes.append(f"auto exposure {e} exceeds the 30 fps ceiling of {EXPOSURE_MAX_30FPS}: clamped")
    e = plan["exposure_time_absolute"]
    if floor_note: notes.append(floor_note)
    if e < floor:   # why: starting brightness stays near auto's: darken by the stops the longer exposure adds
        stops_up = math.log2(floor / max(e, 1)); plan["exposure_time_absolute"] = floor
        if k["stops"]:
            g = plan["gain"]; plan["gain"] = max(0, int(round(g - GAIN_STEPS_PER_STOP * stops_up)))
            notes.append(f"exposure {e} is shorter than the floor {floor}: raised to {floor}, gain {g} -> {plan['gain']}")
        else:
            b = plan["brightness"]; plan["brightness"] = int(min(BRIGHTNESS_RANGE[1], max(BRIGHTNESS_RANGE[0], round(b - BRIGHTNESS_STEPS_PER_STOP * stops_up))))
            notes.append(f"exposure {e} is shorter than the floor {floor}: raised to {floor}, brightness {b} -> {plan['brightness']}")
    return plan, notes


def tune(dev, kind, plan, src, target, floor=0):
    """Lock `plan` on a streaming camera, then nudge gain (C920, whole-stop exposure moves when gain runs out) or exposure
    (wrist; brightness offset once exposure would drop under `floor`) until the picture mean is within TARGET_TOL of `target`.
    Exposure never goes below `floor`. Returns (controls the camera holds, final mean, notes)."""
    notes = []; stops = KINDS[kind]["stops"]; spp = float(GAIN_STEPS_PER_STOP); spb = float(BRIGHTNESS_STEPS_PER_STOP)
    diffs = apply_block(dev, {"controls": plan})
    if diffs: raise V4L2Error("would not take: " + ", ".join(f"{n}={d['got']} (want {d['want']})" for n, d in diffs.items()))
    m = src.mean(); prev = None; clip, why = clipping(src); clipped = clip
    for _ in range(TUNE_ROUNDS):
        if abs(m - target) <= TARGET_TOL: break
        ratio = target / max(m, 1.0); changed = {}
        e = plan["exposure_time_absolute"]
        if stops:
            g = plan["gain"]
            if prev and prev["e"] == e and prev["g"] != g and prev["m"] > 0 and m > 0 and m != prev["m"]:
                est = (g - prev["g"]) / math.log2(m / prev["m"])         # measured gain steps per stop
                if 16 <= est <= 256: spp = est
            usable = [s for s in stops if floor <= s <= EXPOSURE_CLAMP] or [EXPOSURE_CLAMP]
            i = usable.index(e) if e in usable else len(usable) - 1
            ng = g + spp * math.log2(ratio)
            while ng > 255 and i + 1 < len(usable): i += 1; ng -= spp * math.log2(usable[i] / usable[i - 1])
            while ng < 0 and i > 0: i -= 1; ng += spp * math.log2(usable[i + 1] / usable[i])
            ng = int(round(min(255, max(0, ng))))
            if clip and usable[i] > e: i = usable.index(e) if e in usable else i   # why: clipping, longer exposure only whitens
            if usable[i] != e: changed["exposure_time_absolute"] = usable[i]
            if ng != g: changed["gain"] = ng
            prev = {"e": e, "g": g, "m": m}
        else:
            ne = int(round(min(EXPOSURE_MAX_30FPS, max(1, e * ratio)))); b = plan["brightness"]
            if prev and prev["e"] == e and prev["b"] != b and prev["m"] > 0 and m > 0 and m != prev["m"]:
                est = (b - prev["b"]) / math.log2(m / prev["m"])         # measured brightness steps per stop
                if 16 <= est <= 256: spb = est
            prev = {"e": e, "b": b, "m": m}
            if ne < floor:   # why: flicker banding; take the rest of the darkening from the brightness offset
                ne = floor; rest = ratio * e / floor
                nb = int(round(min(BRIGHTNESS_RANGE[1], max(BRIGHTNESS_RANGE[0], b + spb * math.log2(rest)))))
                if nb != b: changed["brightness"] = nb
            if clip and ne > e: ne = e     # why: clipping, longer exposure only whitens
            if ne != e: changed["exposure_time_absolute"] = ne
        if not changed: break
        for n, v in changed.items(): set_ctrl(dev, n, v); plan[n] = v
        time.sleep(TUNE_WAIT); m = src.mean(); clip, why = clipping(src); clipped = clipped or clip
    if clipped: notes.append(f"sensor clipping ({why}): exposure not raised further; use flicker-free light or an ND filter on the lens")
    held = get_ctrls(dev, list(plan)); diffs = compare(plan, held)
    if diffs: raise V4L2Error("drifted during tuning: " + ", ".join(f"{n}={d['got']} (want {d['want']})" for n, d in diffs.items()))
    controls = {n: held.get(n, v) for n, v in plan.items()}
    if abs(m - target) > TARGET_TOL: notes.append(f"picture mean {m:.0f} is off the target {target:.0f} (gain/exposure range exhausted): change the light")
    if controls.get("gain", 0) >= 240: notes.append(f"gain {controls['gain']} is near the top of the range: this camera is starved for light")
    return controls, m, notes


def parse_overrides(items):
    """--set ROLE:CTRL=VAL -> {role: {ctrl: val}}"""
    out = {}
    for it in items or []:
        m = re.match(r"^(\w+):(\w+)=(-?\d+)$", it)
        if not m: raise ValueError(f"--set wants ROLE:CTRL=VALUE, got {it!r}")
        out.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    return out


def parse_min_exposure(items):
    """--min-exposure ROLE=N -> {role: N}"""
    out = {}
    for it in items or []:
        m = re.match(r"^(\w+)=(\d+)$", it)
        if not m: raise ValueError(f"--min-exposure wants ROLE=N, got {it!r}")
        out[m.group(1)] = int(m.group(2))
    return out


def capture(cams, args):
    """Per camera: auto on, settle, read, lock, tune to auto's brightness, verify; write what the cameras hold."""
    try: overrides = parse_overrides(args.set); min_exp = parse_min_exposure(args.min_exposure)
    except ValueError as e: print(e); return 2
    mains_hz = getattr(args, "mains_hz", MAINS_HZ); opt_in = set(getattr(args, "flicker_floor", None) or [])
    out = {"captured": datetime.datetime.now().isoformat(timespec="seconds"), "host": os.uname().nodename,
           "note": "manual camera settings frozen for training; re-run `cam_lock.py capture` whenever the lighting setup changes (it locks too)",
           "exposure_rule": f"C920 exposure_time_absolute is 100 us units in 1/2^n s stops; recording at 30 fps needs <= {EXPOSURE_MAX_30FPS}, capture clamps at {EXPOSURE_CLAMP}",
           "mains_hz": mains_hz,
           "flicker_rule": f"capture never locks an exposure shorter than one light-flicker period ({10000 / (2 * mains_hz) / 10:.1f} ms at {mains_hz} Hz mains) (C920; the wrist only with --flicker-floor wrist or --min-exposure); see each block's exposure_floor",
           "band": list(BAND), "cameras": {}}
    for r in [*min_exp, *opt_in]:
        if r not in cams: print(f"--min-exposure: no camera role {r}"); return 2
    missing = []; failed = []
    with Cams():
        for role, by_id in cams.items():
            dev = dev_path(by_id)
            if dev is None: missing.append(role); print(f"{role}: {by_id} not plugged in"); continue
            kind = kind_of(by_id)
            if kind is None: failed.append(role); print(f"{role}: unknown camera kind for {by_id}"); continue
            src = None
            try:
                src = FrameSource(dev, kind)
                for flag, on, _off, _manual in KINDS[kind]["autos"]: set_ctrl(dev, flag, on)
                print(f"{role} ({dev}, {kind}): auto on, settling {args.settle:g} s ...", flush=True)
                src.pump(args.settle)
                seen = get_ctrls(dev, controls_of(kind)); m_auto = src.mean()
                target = min(TARGET_MAX, max(TARGET_MIN, m_auto))
                floor, floor_note = exposure_floor(kind, mains_hz, min_exp.get(role), role in opt_in)
                plan, notes = plan_from_auto(kind, seen, overrides.get(role), floor, floor_note)
                controls, m, more = tune(dev, kind, plan, src, target, floor)
                notes += more
                if not BAND[0] <= m <= BAND[1]:
                    release_to_auto(dev, kind); failed.append(role)
                    print(f"{role}: locked picture mean {m:.0f} is outside {BAND[0]:.0f}..{BAND[1]:.0f}: ROLLED BACK TO AUTO, not saved. Fix the light and capture again.")
                    for n in notes: print(f"  note: {n}")
                    continue
                out["cameras"][by_id] = {"role": role, "kind": kind, "device_at_capture": dev, "controls": controls, "auto_seen": seen,
                                         "exposure_floor": floor or None, "auto_mean": round(m_auto, 1), "target_mean": round(target, 1), "verified_mean": round(m, 1), "notes": notes}
                summary = " ".join(f"{n}={controls[n]}" for n in ("exposure_time_absolute", "gain", "white_balance_temperature", "focus_absolute") if n in controls)
                print(f"{role}: locked {summary}  picture mean {m:.0f} (auto gave {m_auto:.0f}, target {target:.0f})")
                for n in notes: print(f"  note: {n}")
            except (V4L2Error, ValueError) as e:
                failed.append(role); print(f"{role}: FAILED, back on auto: {e}")
                release_to_auto(dev, kind)
            finally:
                if src: src.close()
    if not out["cameras"]:
        print("no camera captured, file not written"); return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.file)), exist_ok=True)
    tmp = args.file + ".tmp"
    with open(tmp, "w") as f: json.dump(out, f, indent=2); f.write("\n")
    os.replace(tmp, args.file)
    print(f"wrote {args.file} ({len(out['cameras'])} camera{'s' if len(out['cameras']) != 1 else ''})")
    if failed: return 1
    return 1 if (missing and args.strict) else 0


# ---------------------------------------------------------------- apply / check / guard
def load(path):
    with open(path) as f: data = json.load(f)
    if "cameras" not in data or not isinstance(data["cameras"], dict): raise ValueError(f"{path}: no 'cameras' block")
    return data


def select(data, only):
    """Blocks to act on: all, or the one `only` names (role, by-id name or /dev/videoN / videoN)."""
    blocks = data["cameras"]
    if not only: return blocks
    if only in blocks: return {only: blocks[only]}
    for by_id, b in blocks.items():
        if b.get("role") == only: return {by_id: b}
    want = os.path.realpath(only if only.startswith("/") else os.path.join("/dev", only))
    for by_id, b in blocks.items():
        dev = dev_path(by_id)
        if dev is not None and os.path.realpath(dev) == want: return {by_id: b}
    return {}


def apply_block(dev, block):
    """Write the controls in file order (auto flag first, then what it gates), verify, re-write what moved.
    Returns {ctrl: {'want', 'got'[, 'error']}} for anything the camera would not hold."""
    controls = block["controls"]; names = list(controls)
    before = get_ctrls(dev, names)
    pending = set(names); errors = {}; diffs = {}
    for rnd in range(ROUNDS):
        for name in names:
            if name not in pending: continue
            try: set_ctrl(dev, name, int(controls[name])); errors.pop(name, None)
            except V4L2Error as e: errors[name] = str(e)
        transition = any(before.get(f) != int(controls[f]) for f in AUTO_FLAGS if f in controls)
        if rnd > 0 or transition: time.sleep(SETTLE)          # the C920 clobbers gain ~1 s after leaving auto
        got = get_ctrls(dev, names)
        diffs = compare(controls, got)
        if not diffs: break
        pending = set(diffs)
    for n, e in errors.items():
        if n in diffs: diffs[n]["error"] = e
    return diffs


def verify_block(dev, block):
    """apply --verify: lock while streaming in recording mode, measure the picture, roll back to auto if it is off."""
    kind = block.get("kind") or "c920"
    src = FrameSource(dev, kind)
    try:
        diffs = apply_block(dev, block)
        m = src.mean()
    finally: src.close()
    if diffs or not BAND[0] <= m <= BAND[1]:
        release_to_auto(dev, kind)
        diffs = dict(diffs); diffs["picture_mean"] = {"want": f"{BAND[0]:.0f}..{BAND[1]:.0f}", "got": round(m), "error": "rolled back to auto"}
    return diffs


def check_block(dev, block):
    controls = block["controls"]
    return compare(controls, get_ctrls(dev, list(controls)))


def wait_for(blocks, seconds):
    """Boot path: give the USB cameras up to `seconds` to show up before applying."""
    deadline = time.time() + seconds
    while time.time() < deadline and any(dev_path(b) is None for b in blocks): time.sleep(0.5)


def run_blocks(data, cams, args, fn, verb):
    blocks = select(data, args.only)
    if not blocks:
        print(f"no camera in {args.file} matches --only {args.only}"); return 1
    if getattr(args, "wait", 0): wait_for(blocks, args.wait)
    result = {"ok": True, "file": args.file, "captured": data.get("captured"), "cameras": {}}
    for by_id, block in blocks.items():
        role = block.get("role", by_id); dev = dev_path(by_id)
        entry = {"role": role, "by_id": by_id, "device": dev, "present": dev is not None, "ok": False, "diffs": {}}
        if dev is None:
            entry["ok"] = not args.strict; entry["status"] = "unplugged"
        else:
            try:
                entry["diffs"] = fn(dev, block); entry["ok"] = not entry["diffs"]
                entry["status"] = "locked" if entry["ok"] else "differs"
            except V4L2Error as e:
                entry["status"] = f"error: {e}"
        result["ok"] = result["ok"] and entry["ok"]
        result["cameras"][role] = entry
    if not args.only:   # a camera env.sh knows that was never captured (or was rolled back) is not locked either
        for role, by_id in cams.items():
            if by_id not in blocks:
                dev = dev_path(by_id)
                ok = dev is None and not args.strict
                result["cameras"][role] = {"role": role, "by_id": by_id, "device": dev, "present": dev is not None, "ok": ok, "diffs": {},
                                           "status": "unplugged" if dev is None else "no settings captured"}
                result["ok"] = result["ok"] and ok
    if getattr(args, "json", False):
        print(json.dumps(result)); return 0 if result["ok"] else 1
    parts = []
    for role, e in result["cameras"].items():
        if e["status"] == "locked": parts.append(f"{role} ok")
        elif e["status"] == "unplugged": parts.append(f"{role} unplugged")
        elif e["diffs"]: parts.append(f"{role} " + ", ".join(f"{n}={d['got']} (want {d['want']})" for n, d in e["diffs"].items()))
        else: parts.append(f"{role} {e['status']}")
    print(("cams locked: " if result["ok"] else f"cams NOT locked ({verb}): ") + "; ".join(parts))
    for role, e in result["cameras"].items():
        for n, d in e["diffs"].items():
            if d.get("error"): print(f"  {role} {n}: {d['error']}")
    return 0 if result["ok"] else 1


def guard(args):
    """Resident re-locker. One per settings file; exits 0 at once if another holds the lock."""
    lock_path = args.file + ".guard.lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    lock = open(lock_path, "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"guard already running for {args.file}"); return 0
    lock.write(str(os.getpid())); lock.flush()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log(f"guard started (pid {os.getpid()}, every {args.interval:g} s, {args.file})")
    data = None; mtime = None; said = set()
    while True:
        try: st = os.stat(args.file).st_mtime
        except OSError: st = None
        if st is None:
            if "nofile" not in said: log("waiting for the settings file (run `cam_lock.py capture`)"); said.add("nofile")
            time.sleep(args.interval); continue
        if st != mtime:
            try: data = load(args.file); mtime = st; said.discard("nofile"); log(f"settings loaded (captured {data.get('captured')}, {len(data['cameras'])} cameras)")
            except (OSError, ValueError) as e:
                if "badfile" not in said: log(f"cannot read {args.file}: {e}"); said.add("badfile")
                time.sleep(args.interval); continue
        for by_id, block in select(data, args.only).items():
            role = block.get("role", by_id); dev = dev_path(by_id)
            if dev is None:
                if ("unplugged", role) not in said: log(f"{role}: unplugged"); said.add(("unplugged", role))
                continue
            if ("unplugged", role) in said: log(f"{role}: back on {dev}"); said.discard(("unplugged", role))
            try:
                drift = check_block(dev, block)
                if drift:
                    left = apply_block(dev, block)
                    msg = ", ".join(f"{n} {d['got']}->{d['want']}" for n, d in drift.items())
                    log(f"{role}: re-locked {msg}" + ("" if not left else "; STILL OFF: " + ", ".join(f"{n}={d['got']}" for n, d in left.items())))
                said.discard(("error", role))
            except V4L2Error as e:
                if ("error", role) not in said: log(f"{role}: {e}"); said.add(("error", role))
        time.sleep(args.interval)


def show(cams, args):
    for role, by_id in cams.items():
        dev = dev_path(by_id)
        if dev is None: print(f"{role}: {by_id} unplugged"); continue
        kind = kind_of(by_id) or "c920"
        ctrls = list_ctrls(dev)
        print(f"{role} ({dev}):")
        for n in controls_of(kind):
            c = ctrls.get(n)
            if c is None: continue
            tail = ("  inactive" if c["inactive"] else "") + ("" if c["value"] == c["default"] else f"  (default {c['default']})")
            print(f"  {n:28s} {c['value']!s:>6}{tail}")
    return 0


# ---------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["capture", "apply", "check", "guard", "show"])
    ap.add_argument("--file", default=DEFAULT_FILE, help="settings json (default %(default)s)")
    ap.add_argument("--only", help="act on one camera: role, by-id name, /dev/videoN or videoN")
    ap.add_argument("--strict", action="store_true", help="an unplugged camera is a failure")
    ap.add_argument("--json", action="store_true", help="check/apply: machine-readable result")
    ap.add_argument("--verify", action="store_true", help="apply: also stream each camera, measure the picture, roll back to auto if it is off")
    ap.add_argument("--settle", type=float, default=4.0, help="capture: seconds to let auto settle (default %(default)s)")
    ap.add_argument("--set", action="append", metavar="ROLE:CTRL=VAL", help="capture: override one starting value, e.g. wrist:exposure_time_absolute=156")
    ap.add_argument("--mains-hz", type=int, choices=[50, 60], default=MAINS_HZ, help="capture: mains frequency; lights flicker at twice this, exposure is kept >= one flicker period (default %(default)s)")
    ap.add_argument("--min-exposure", action="append", metavar="ROLE=N", help="capture: override the shortest exposure allowed for one camera (100 us units), e.g. wrist=120")
    ap.add_argument("--flicker-floor", action="append", metavar="ROLE", help="capture: also apply the flicker floor to a camera that has none by default (the icSpring wrist; it may clip), e.g. wrist")
    ap.add_argument("--wait", type=float, default=0.0, help="apply: wait up to S seconds for the cameras to appear (boot)")
    ap.add_argument("--interval", type=float, default=1.0, help="guard: seconds between checks (default %(default)s)")
    args = ap.parse_args(argv)
    cams = cameras_from_env()
    if args.only and args.cmd in ("capture", "show"):
        cams = {r: b for r, b in cams.items() if args.only in (r, b)}
        if not cams: print(f"no camera named {args.only}"); return 1
    if args.cmd == "capture": return capture(cams, args)
    if args.cmd == "show": return show(cams, args)
    if args.cmd == "guard": return guard(args)
    try: data = load(args.file)
    except (OSError, ValueError) as e:
        if args.json: print(json.dumps({"ok": False, "error": str(e), "file": args.file}))
        else: print(f"cams NOT locked: no settings file ({e}); run `cam_lock.py capture` first")
        return 1
    if args.cmd == "apply" and args.verify:
        with Cams(): return run_blocks(data, cams, args, verify_block, "apply --verify")
    if args.cmd == "apply": return run_blocks(data, cams, args, apply_block, "apply")
    return run_blocks(data, cams, args, check_block, "check")


if __name__ == "__main__":
    sys.exit(main())
