"""Self-healing Feetech bus for LeRobot.

On this Mac the CH343 serial adapters occasionally stall (every packet times out, "There is no
status packet!") until the port is closed and reopened. LeRobot raises ConnectionError and the
record/teleop loop dies. This plugin wraps FeetechMotorsBus read/write methods: on ConnectionError
it closes and reopens the port, confirms a motor answers, and retries the call once.

Auto-imported by lerobot CLIs (package name starts with lerobot_robot_). Disable with
LEROBOT_BUSHEAL=0. Heals are logged at WARNING level and counted in bus._heal_count.
"""
import logging
import os
import time

from lerobot.motors.feetech import FeetechMotorsBus

logger = logging.getLogger("lerobot_robot_busheal")

MAX_HEAL_S = float(os.environ.get("LEROBOT_BUSHEAL_MAX_S", "10"))
CONNECT_MAX_S = float(os.environ.get("LEROBOT_BUSHEAL_CONNECT_S", "12"))
_WRAPPED = ("sync_read", "sync_write", "read", "write")


def _enabled() -> bool:
    return os.environ.get("LEROBOT_BUSHEAL", "1") != "0"


def heal(bus: FeetechMotorsBus, why: str) -> None:
    """Close + reopen the serial port until a motor answers, or give up after MAX_HEAL_S."""
    bus._healing = True
    t0 = time.perf_counter()
    first_id = next(iter(bus.ids), 1)
    attempt = 0
    try:
        while True:
            attempt += 1
            try:
                bus.port_handler.closePort()
            except Exception:  # nosec B110 - port may already be dead
                pass
            time.sleep(0.15)
            try:
                if bus.port_handler.openPort():
                    bus.set_timeout()
                    if bus.ping(first_id, num_retry=1) is not None:
                        dt = time.perf_counter() - t0
                        bus._heal_count = getattr(bus, "_heal_count", 0) + 1
                        logger.warning(
                            f"[busheal] {bus.port}: bus stalled during {why}; reopened on attempt {attempt} "
                            f"after {dt:.2f}s (heal #{bus._heal_count})"
                        )
                        return
            except Exception as e:  # keep trying until the deadline
                logger.debug(f"[busheal] attempt {attempt} failed: {e!r}")
            if time.perf_counter() - t0 > MAX_HEAL_S:
                raise ConnectionError(
                    f"[busheal] {bus.port}: could not revive the bus within {MAX_HEAL_S}s ({attempt} attempts)"
                )
            time.sleep(0.2)
    finally:
        bus._healing = False


def clear_overload(bus: FeetechMotorsBus, why: str) -> None:
    import re
    m = re.search(r"id_=(\d+)", why)
    ids = [int(m.group(1))] if m else list(bus.ids)
    table = bus.model_ctrl_table[next(iter(bus.motors.values())).model]
    addr, length = table["Torque_Enable"]
    paddr, plen = table["Present_Position"]; gaddr, glen = table["Goal_Position"]
    for id_ in ids:
        bus._write(addr, length, id_, 0, num_retry=2, raise_on_error=False)
        pos, comm, _ = bus._read(paddr, plen, id_, num_retry=2, raise_on_error=False)
        # why: the SCS SDK returns 0 (never None) for a timed-out read, which would park the gripper at position 0
        if bus._is_comm_success(comm):   # park the goal where the motor is, so re-enabling torque does not re-stall it
            bus._write(gaddr, glen, id_, int(pos), num_retry=2, raise_on_error=False)
        for val in (1, 0):
            bus._write(addr, length, id_, val, num_retry=2, raise_on_error=False)
            time.sleep(0.05)
    bus._overload_count = getattr(bus, "_overload_count", 0) + 1
    logger.warning(f"[busheal] {bus.port}: OVERLOAD on motor id {ids} cleared (torque cycled; #{bus._overload_count}). "
                   "The gripper is squeezing too hard; lower its torque limit.")



# --- joint-state publisher: every Present_Position read is mirrored to a small JSON file for the lab dashboard ---
import json as _json
STATE_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
_ROLE_HINTS = (("so101_follower", "follower"), ("so101_leader", "leader"), ("5C82107034", "follower"), ("5C82107128", "leader"))
_last_pub = {}
def _role(port: str) -> str:
    for hint, role in _ROLE_HINTS:
        if hint in port: return role
    return "arm"
def publish_state(bus, values: dict, normalized: bool) -> None:
    try:
        role = _role(str(bus.port)); now = time.time()
        if now - _last_pub.get(role, 0) < 0.04: return
        _last_pub[role] = now
        cal = {n: [c.range_min, c.range_max] for n, c in (bus.calibration or {}).items()}
        path = os.path.join(STATE_DIR, f"so101_{role}.json"); tmp = path + ".tmp"
        with open(tmp, "w") as f: _json.dump({"t": now, "role": role, "mode": "deg" if normalized else "raw", "src": "lerobot", "pos": {k: float(v) for k, v in values.items()}, "cal": cal}, f)
        os.replace(tmp, path)
    except Exception:  # nosec B110 - never let telemetry break the control loop
        pass

# --- servo health telemetry (docs/telemetry.md): temperature, load, current, voltage, error flags, goal vs actual -> /dev/shm/so101_<role>_tele.json ---
# Read-only: one sync READ of registers 40..70 for all six servos (~2.4 ms of wire time at 1 Mbaud), at most every LEROBOT_TELE_S (default 0.5 s),
# issued from the thread that just finished a Present_Position read, i.e. between two control-loop bus operations, never concurrently with one.
# LEROBOT_TELE_S=0 turns it off; LEROBOT_TELE_ROLES=follower,leader adds the leader. jetson/servo_telemetry.py does the decoding; it is looked up
# in $SO101_HOME (default ~/so101) and the plugin works without it.
TELE_S = float(os.environ.get("LEROBOT_TELE_S", "0.5"))
TELE_ROLES = tuple(r for r in os.environ.get("LEROBOT_TELE_ROLES", "follower").split(",") if r)
_tele_mod = []
_last_tele = {}
_tele_limits = set()
def _telemetry():
    if not _tele_mod:
        try:
            import sys
            home = os.environ.get("SO101_HOME") or os.path.expanduser("~/so101")
            if home not in sys.path: sys.path.append(home)
            import servo_telemetry
            _tele_mod.append(servo_telemetry)
        except Exception: _tele_mod.append(None)
    return _tele_mod[0]
def publish_telemetry(bus) -> None:
    try:
        if TELE_S <= 0: return
        role = _role(str(bus.port)); now = time.time()
        if role not in TELE_ROLES or now - _last_tele.get(role, 0) < TELE_S: return
        _last_tele[role] = now
        st = _telemetry()
        if st is None: return
        sample = st.read_sample(bus, now)
        if sample is None: return
        if role not in _tele_limits: sample["limits"] = st.read_limits(bus); _tele_limits.add(role)
        sample.update(role=role, src="lerobot", mode="deg")
        path = os.path.join(STATE_DIR, f"so101_{role}_tele.json"); tmp = path + ".tmp"
        with open(tmp, "w") as f: _json.dump(sample, f)
        os.replace(tmp, path)
    except Exception:  # nosec B110 - never let telemetry break the control loop
        pass

def _wrap(name: str) -> None:
    orig = getattr(FeetechMotorsBus, name)
    if getattr(orig, "_busheal", False):
        return

    def wrapper(self, *args, **kwargs):
        try:
            out = orig(self, *args, **kwargs)
            if name == "sync_read" and (args[0] if args else kwargs.get("data_name")) == "Present_Position" and isinstance(out, dict):
                publish_state(self, out, kwargs.get("normalize", True))
                publish_telemetry(self)
            return out
        except RuntimeError as e:
            # A servo that tripped overload answers every packet with an error status. Clear it
            # (torque off -> on) so the write goes through and the motor re-engages.
            if "Overload error" in str(e) and _enabled():
                clear_overload(self, str(e))
                return orig(self, *args, **kwargs)
            raise
        except ConnectionError:
            if not _enabled() or getattr(self, "_healing", False) or not self.port_handler.is_open:
                raise
            heal(self, name)
            return orig(self, *args, **kwargs)

    wrapper._busheal = True
    wrapper.__name__ = orig.__name__
    wrapper.__doc__ = orig.__doc__
    setattr(FeetechMotorsBus, name, wrapper)


for _n in _WRAPPED:
    _wrap(_n)
logger.info("[busheal] FeetechMotorsBus self-heal active for %s", ", ".join(_WRAPPED))


# --- connect(): retry the handshake a few times so a momentary stall at startup is not fatal ---
_orig_connect = FeetechMotorsBus.connect
if not getattr(_orig_connect, "_busheal", False):

    def _connect(self, handshake: bool = True):
        last = None
        t0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                return _orig_connect(self, handshake)
            except (ConnectionError, RuntimeError) as e:
                last = e
                if not _enabled() or "motor check failed" not in str(e) and not isinstance(e, ConnectionError):
                    raise
                try:
                    self.port_handler.closePort()
                except Exception:  # nosec B110
                    pass
                if time.perf_counter() - t0 > CONNECT_MAX_S:
                    logger.error(f"[busheal] {self.port}: motors still not answering after {CONNECT_MAX_S:.0f}s; is the arm powered?")
                    raise last from None
                if attempt == 1 or attempt % 5 == 0:
                    logger.warning(f"[busheal] {self.port}: motors not answering (attempt {attempt}), waiting for power/bus...")
                time.sleep(0.6)

    _connect._busheal = True
    _connect.__name__ = "connect"
    FeetechMotorsBus.connect = _connect


# --- gripper squeeze force cap -------------------------------------------------------------
# LeRobot's SOFollower.configure() writes Max_Torque_Limit=500 to the gripper on every connect.
# On this arm that stalls at ~540 current units against a bottle, above the 250 protection
# threshold, so the servo trips overload ~2 s into every grasp. Measured 6 Oct 2026:
#   limit 500 -> 541, 400 -> 370, 300 -> 240, 250 -> 187, 200 -> 142.
# Override with LEROBOT_GRIPPER_TORQUE (default 250).
GRIPPER_TORQUE = int(os.environ.get("LEROBOT_GRIPPER_TORQUE", "250"))
# current guard; measured stall current ~ 1.08 x torque limit on this gripper (250->187... 400->370, 500->541)
GRIPPER_P = int(os.environ.get("LEROBOT_GRIPPER_P", "0"))   # 0 = leave LeRobot's P gain (16)
GRIPPER_CURRENT = int(os.environ.get("LEROBOT_GRIPPER_CURRENT", str(max(250, int(GRIPPER_TORQUE * 1.35)))))
try:
    from lerobot.robots.so_follower.so_follower import SOFollower

    _orig_configure = SOFollower.configure
    if not getattr(_orig_configure, "_busheal", False):

        def _configure(self):
            _orig_configure(self)
            if "gripper" in self.bus.motors and _enabled():
                with self.bus.torque_disabled():
                    self.bus.write("Max_Torque_Limit", "gripper", GRIPPER_TORQUE, normalize=False)
                    self.bus.write("Torque_Limit", "gripper", GRIPPER_TORQUE, normalize=False)
                    # The STS3215 trips "overload" whenever it sits stalled at its torque limit for
                    # Protection_Time, whatever that limit is, so holding any object trips it in ~2 s.
                    # Protective_Torque=0 disables that timer (measured 6 Oct 2026: 8 s hold, no trip).
                    # The current guard (Protection_Current=250, stall current at limit 250 is ~187)
                    # and the 70 C temperature guard stay active.
                    self.bus.write("Protective_Torque", "gripper", 0, normalize=False)
                    self.bus.write("Protection_Time", "gripper", 254, normalize=False)
                    self.bus.write("Protection_Current", "gripper", GRIPPER_CURRENT, normalize=False)
                    if GRIPPER_P:
                        self.bus.write("P_Coefficient", "gripper", GRIPPER_P, normalize=False)
                logger.info(f"[busheal] gripper: torque limit {GRIPPER_TORQUE}/1000, current guard {GRIPPER_CURRENT}, P {GRIPPER_P or 16}, stall-timer trip disabled (LEROBOT_GRIPPER_TORQUE / _CURRENT / _P)")

        _configure._busheal = True
        _configure.__name__ = "configure"
        SOFollower.configure = _configure
except Exception as _e:  # pragma: no cover
    logger.warning(f"[busheal] gripper torque cap not installed: {_e!r}")


# --- voice announcements ("Recording episode 3", "Reset the environment", ...) -> /dev/shm/so101_say.jsonl.
# arm_state.py forwards them to the Mac dashboard, which speaks them with macOS `say`. spd-say has no audio over ssh.
try:
    import lerobot.utils.utils as _lu
    _orig_say = _lu.say
    def _say(text, blocking=False):
        try:
            with open(os.path.join(STATE_DIR, "so101_say.jsonl"), "a") as f: f.write(_json.dumps({"t": time.time(), "say": text}) + "\n")
        except OSError: pass
        if os.environ.get("LEROBOT_LOCAL_SAY") == "1": _orig_say(text, blocking)
    _say._busheal = True; _lu.say = _say
except Exception as _e:  # pragma: no cover
    logger.warning(f"[busheal] say() relay not installed: {_e!r}")


# --- stale camera frames must not end the session. LeRobot raises TimeoutError when a camera frame is older than
# 500 ms (seen 30 Oct 2026: overhead C920 at 692 ms while three cameras were being encoded on the CPU); the whole
# recording then exits, torque drops and the arm goes limp. Return the last frame instead, warn (rate-limited),
# and only raise once the camera has been silent for LEROBOT_CAM_STALE_S seconds (default 3), i.e. it is really gone.
try:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera as _Cam
    CAM_STALE_S = float(os.environ.get("LEROBOT_CAM_STALE_S", "3"))
    if not getattr(_Cam.read_latest, "_busheal", False):
        _orig_read_latest = _Cam.read_latest
        def _read_latest(self, max_age_ms: int = 500):
            try:
                return _orig_read_latest(self, max_age_ms)
            except TimeoutError:
                with self.frame_lock:
                    frame, ts = self.latest_frame, self.latest_timestamp
                age = time.perf_counter() - ts if ts is not None else CAM_STALE_S
                if frame is None or age > CAM_STALE_S:
                    raise
                now = time.monotonic()
                if now - getattr(self, "_stale_warned", 0) > 2:
                    self._stale_warned = now
                    self._stale_count = getattr(self, "_stale_count", 0) + 1
                    logger.warning(f"[busheal] {self}: frame {age*1e3:.0f} ms old, reusing it (#{self._stale_count}; raise after {CAM_STALE_S:.0f}s)")
                return frame
        _read_latest._busheal = True; _read_latest.__name__ = "read_latest"
        _Cam.read_latest = _read_latest
except Exception as _e:  # pragma: no cover
    logger.warning(f"[busheal] stale-frame tolerance not installed: {_e!r}")


# --- a camera that drops off USB must not end the session either. Seen 30 Oct 2026: the hub port of the wrist
# camera was "disabled by hub (EMI?)" and the camera re-enumerated a second later, but LeRobot's read thread
# gives up after 10 failed reads and the next read_latest() raises "read thread is not running" -> recording
# exits, torque drops, arm goes limp. Reopen the device instead (its /dev/v4l/by-id path comes back) and keep
# going; give up only after LEROBOT_CAM_REOPEN_S seconds (default 20). The stale-frame tolerance above covers
# the gap, so raise CAM_STALE_S with it (default 6 s here).
CAM_STALE_S = float(os.environ.get("LEROBOT_CAM_STALE_S", "6"))
CAM_REOPEN_S = float(os.environ.get("LEROBOT_CAM_REOPEN_S", "20"))
try:
    import cv2
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera as _Cam2
    from lerobot.utils.errors import DeviceNotConnectedError as _DNC
    def _cam_say(text):
        try:
            with open(os.path.join(STATE_DIR, "so101_say.jsonl"), "a") as f: f.write(_json.dumps({"t": time.time(), "say": text}) + "\n")
        except OSError: pass
    def _reopen(self, stop_event) -> bool:
        """Release the dead VideoCapture and reopen the same device until it answers or CAM_REOPEN_S passes."""
        t0 = time.monotonic(); attempt = 0
        try:
            if self.videocapture is not None: self.videocapture.release()
        except Exception: pass
        self.videocapture = None
        while not stop_event.is_set() and time.monotonic() - t0 < CAM_REOPEN_S:
            time.sleep(0.5); attempt += 1
            if isinstance(self.index_or_path, str) and not os.path.exists(self.index_or_path): continue   # not re-enumerated yet
            vc = None
            try:
                vc = cv2.VideoCapture(self.index_or_path, self.backend)
                if not vc.isOpened(): raise RuntimeError("not opened")
                self.videocapture = vc
                self._configure_capture_settings()
                ok, _ = vc.read()
                if not ok: raise RuntimeError("opened but no frame")
                dt = time.monotonic() - t0
                self._reopen_count = getattr(self, "_reopen_count", 0) + 1
                logger.warning(f"[busheal] {self}: device reopened after {dt:.1f}s, attempt {attempt} (reopen #{self._reopen_count})")
                _cam_say("camera reconnected")
                return True
            except Exception as e:
                logger.debug(f"[busheal] reopen attempt {attempt} failed: {e!r}")
                self.videocapture = None
                try:
                    if vc is not None: vc.release()
                except Exception: pass
        return False
    def _read_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None: raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")
        failure_count = 0
        while not stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frame = processed_frame; self.latest_timestamp = capture_time
                self.new_frame_event.set(); failure_count = 0
            except _DNC:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 3: logger.warning(f"Error reading frame in background thread for {self}: {e}")
                if failure_count >= 3:
                    logger.warning(f"[busheal] {self}: {failure_count} consecutive read failures, reopening the device")
                    if _reopen(self, stop_event): failure_count = 0
                    elif stop_event.is_set(): break
                    else: raise RuntimeError(f"{self} did not come back within {CAM_REOPEN_S:.0f}s.") from e
    if not getattr(_Cam2._read_loop, "_busheal", False):
        _read_loop._busheal = True; _Cam2._read_loop = _read_loop
except Exception as _e:  # pragma: no cover
    logger.warning(f"[busheal] camera reopen not installed: {_e!r}")


# --- pause / resume for lerobot-record (dashboard PAUSE button, key "p") -------------------------------------
# While paused the record loop keeps running so the follower stays live on the leader, but no frames are added
# to the dataset and the episode / reset clock stands still. Done without touching lerobot's source: the record
# script's `time.perf_counter` is swapped for a clock that stops while paused, LeRobotDataset.add_frame drops
# frames while paused, and the key listener gets a "p" toggle. Only active when lerobot-record is the running
# script (added 7 Nov 2026).
try:
    import sys as _sys, time as _rtime
    if "lerobot.scripts.lerobot_record" in _sys.modules and _enabled():
        import lerobot.utils.keyboard_input as _ki
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LRD
        _rec = _sys.modules["lerobot.scripts.lerobot_record"]
        # "lost" is the offset of the record clock behind the real one; "frozen" is the value it shows while paused.
        # The toggle runs on the key-listener thread while record_loop reads perf_counter() on the main thread, so each
        # state reads a single field besides "on" and the toggle writes that field *before* flipping "on" (8 Nov 2026).
        PAUSE = {"on": False, "lost": 0.0, "since": 0.0, "frozen": 0.0}

        class _PausableTime:
            """time module stand-in: perf_counter() does not advance while paused, everything else passes through."""
            def __getattr__(self, name): return getattr(_rtime, name)
            def perf_counter(self):
                if PAUSE["on"]: return PAUSE["frozen"]
                return _rtime.perf_counter() - PAUSE["lost"]
        _rec.time = _PausableTime()

        if not getattr(_LRD.add_frame, "_busheal", False):
            _orig_add_frame = _LRD.add_frame
            def _add_frame(self, frame):
                if PAUSE["on"]: return None
                return _orig_add_frame(self, frame)
            _add_frame._busheal = True; _LRD.add_frame = _add_frame

        # Which phase lerobot is in is known from its last announcement ("Recording episode 3", "Reset the environment").
        # Track it by wrapping whatever say() is current (the jsonl relay above), so the relay still gets every line.
        LAST_SAY = {"text": ""}
        _prev_say = _lu.say
        def _say_track(text, blocking=False):
            LAST_SAY["text"] = str(text); return _prev_say(text, blocking)
        _say_track._busheal = True; _lu.say = _say_track
        def _announce(text):
            # why: the plugin's own speech ("Paused", "Redo queued") must not overwrite the phase, or a second REDO press
            # or a pause during the reset would silently fall back to lerobot's cut-the-reset-short behaviour
            try: _prev_say(text)
            except Exception: pass

        def busheal_pause_toggle():
            now = _rtime.perf_counter()
            if PAUSE["on"]:
                PAUSE["lost"] = now - PAUSE["frozen"]; PAUSE["on"] = False
                print("\n__RESUMED__", flush=True); logger.info("[busheal] record resumed")
                _announce("Resumed")
            else:
                PAUSE["since"] = now; PAUSE["frozen"] = now - PAUSE["lost"]; PAUSE["on"] = True
                print("\n__PAUSED__", flush=True); logger.info("[busheal] record paused: arm live, frames and clock stopped")
                _announce("Paused")

        # REDO during the reset countdown: lerobot would cut the reset short and start re-recording at once, so there is
        # no time to put the scene back. Queue the re-record instead and let the reset run out; the episode is still
        # discarded (it is only saved after the reset). REDO during the episode behaves as before: end now, full reset, redo.
        if not getattr(_ki.apply_recording_control, "_busheal", False):
            _orig_apply = _ki.apply_recording_control
            def _apply_recording_control(control, events):
                if control == "left" and LAST_SAY["text"].startswith("Reset"):
                    events["rerecord_episode"] = True
                    print("Re-record queued: the reset runs to the end, then the episode is recorded again.", flush=True)
                    _announce("Redo queued. Finish the reset.")
                    return
                return _orig_apply(control, events)
            _apply_recording_control._busheal = True; _ki.apply_recording_control = _apply_recording_control

        if not getattr(_ki.create_key_listener, "_busheal", False):
            _orig_ckl = _ki.create_key_listener
            def _create_key_listener(dispatch, **kw):
                def _dispatch(name):
                    if str(name).lower() == "p": busheal_pause_toggle()
                    else: dispatch(name)
                return _orig_ckl(_dispatch, **kw)
            _create_key_listener._busheal = True; _ki.create_key_listener = _create_key_listener
        logger.info("[busheal] record pause installed (key p)")
except Exception as _e:  # pragma: no cover
    logger.warning(f"[busheal] record pause not installed: {_e!r}")
