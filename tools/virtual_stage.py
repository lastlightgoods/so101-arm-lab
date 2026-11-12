"""virtual_stage.py - the display-only "virtual arm" channel for the dashboard's 3D stage (docs/virtual-stage.md).

A FAKE run (tools/autoloop.py --dry-run --stage, tools/demo_session.py FakeRig) or a SIM run (sim/so101_simstate.py) publishes where its pretend arm is;
the dashboard draws it as a separate violet ghost arm under a big SIM / FAKE badge. The real arm is never touched: this channel is a file in /tmp that
dashboard/server.py reads into /api/state under its own key ("virtual"), and nothing on the path calls the arm, ssh or the Jetson.

Message (JSON): source SIM|FAKE (REAL is not accepted), session, t (sender clock, informational), seq, joints (the six LeRobot names; degrees, gripper 0-100),
joint_frame ("lerobot": the real arm's reading convention, the stage adds the arm's calibration offsets; "urdf": degrees from URDF zero, no offsets),
objects [{kind: nut|target, frame: mat_in|sim_base, x, y, z}]. Staleness is judged by the file's mtime (the server's own clock), never by `t`.
Stdlib only: the dashboard, the publishers and the simulator's interpreter all import this by path."""
import json, math, os, re, time

SOURCES = ("SIM", "FAKE")
UNVERIFIED = "UNVERIFIED"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAX_AGE_S = 2.5
DIR = os.environ.get("SO101_VIRTUAL_DIR") or "/tmp"   # the env var is for tests and for a second dashboard; the default is the shared /tmp
PREFIX = "so101_virtual_"
FRAME_OF = {"SIM": "urdf", "FAKE": "lerobot", UNVERIFIED: "lerobot"}   # why: sim_qpos_to_dataset_row is a pure rad->deg transform (no calibration offsets); DryArm's solver degrees get the table's offsets added before the kinematics, as the real arm's readings do
KINDS, FRAMES = ("nut", "target"), ("mat_in", "sim_base")
MAX_OBJECTS, MAX_SESSIONS = 8, 5
_SESSION_OK = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def clean_session(s):
    s = str(s or ""); return s if _SESSION_OK.match(s) else "unknown"


def path_for(session, directory=DIR): return os.path.join(directory, f"{PREFIX}{clean_session(session)}.json")


def _num(v):
    try: f = float(v)
    except (TypeError, ValueError): return None
    return f if math.isfinite(f) else None


def clean_objects(objs):
    out = []
    for o in (objs if isinstance(objs, (list, tuple)) else [])[:MAX_OBJECTS]:
        if not isinstance(o, dict) or o.get("kind") not in KINDS or o.get("frame") not in FRAMES: continue
        x, y, z = _num(o.get("x")), _num(o.get("y")), _num(o.get("z", 0.0))
        if x is None or y is None or z is None: continue
        out.append({"kind": o["kind"], "frame": o["frame"], "x": x, "y": y, "z": z})
    return out


def clean_joints(j):
    """The six joints as floats, or None when any is missing or not finite."""
    if not isinstance(j, dict): return None
    out = {k: _num(j.get(k)) for k in JOINTS}
    return out if all(v is not None for v in out.values()) else None


def make_message(source, session, joints, objects=None, seq=0, t=None):
    """A publisher's message. Raises ValueError for a source other than SIM / FAKE (REAL included) or joints that are not the six finite numbers."""
    if source not in SOURCES: raise ValueError(f"virtual source must be one of {SOURCES}, not {source!r}")
    j = clean_joints(joints)
    if j is None: raise ValueError("virtual joints need all of " + ", ".join(JOINTS) + " as finite numbers")
    return {"source": source, "session": clean_session(session), "t": time.time() if t is None else float(t), "seq": int(seq), "joints": j,
            "joint_frame": FRAME_OF[source], "objects": clean_objects(objects)}


def validate(msg):
    """What a reader may serve: a cleaned copy, or None when the joints are unusable. A missing or unknown source (REAL too) is relabelled
    UNVERIFIED, never trusted and never blank; joint_frame is recomputed from the label, not taken from the file."""
    if not isinstance(msg, dict): return None
    j = clean_joints(msg.get("joints"))
    if j is None: return None
    src = msg.get("source") if msg.get("source") in SOURCES else UNVERIFIED
    seq = msg.get("seq"); seq = seq if isinstance(seq, int) and not isinstance(seq, bool) else 0
    return {"source": src, "session": clean_session(msg.get("session")), "t": _num(msg.get("t")) or 0.0, "seq": seq, "joints": j,
            "joint_frame": FRAME_OF[src], "objects": clean_objects(msg.get("objects"))}


def write(msg, directory=DIR):
    """Atomic write of a message made by make_message. Returns the path."""
    p = path_for(msg["session"], directory); tmp = p + ".tmp"
    with open(tmp, "w") as f: f.write(json.dumps(msg))
    os.replace(tmp, p); return p


def read_fresh(directory=DIR, now=None, max_age=MAX_AGE_S):
    """Every virtual message whose file was written within max_age seconds (file mtime vs `now`), newest first, validated, at most MAX_SESSIONS.
    Each carries age_s. Unreadable, malformed or stale files are skipped."""
    now = time.time() if now is None else now; out = []
    try: names = [n for n in os.listdir(directory) if n.startswith(PREFIX) and n.endswith(".json")]
    except OSError: return out
    for n in names:
        p = os.path.join(directory, n)
        try:
            age = now - os.path.getmtime(p)
            if age > max_age: continue
            m = validate(json.load(open(p)))
        except (OSError, ValueError): continue
        if m is not None: m["age_s"] = round(max(0.0, age), 3); out.append(m)
    out.sort(key=lambda m: m["age_s"]); return out[:MAX_SESSIONS]


class NullSink:
    """The default sink: publishes nothing, waits for nothing."""
    active = False
    def frame(self, joints, objects=None): pass
    def wait(self, seconds): pass
    def close(self): pass


class VirtualSink:
    """Writes each frame to the session's file and (when speed > 0) sleeps seconds/speed so a virtual-clock run is watchable in real time.
    `writer` / `sleeper` / `directory` are injectable for tests."""
    active = True
    def __init__(self, source, session, speed=1.0, directory=DIR, sleeper=time.sleep, writer=None):
        if source not in SOURCES: raise ValueError(f"virtual source must be one of {SOURCES}, not {source!r}")
        self.source, self.session, self.speed, self.dir, self.sleep = source, clean_session(session), float(speed), directory, sleeper
        self.writer = writer or (lambda m: write(m, self.dir)); self.seq = 0; self.objects = []

    def frame(self, joints, objects=None):
        if objects is not None: self.objects = objects
        self.seq += 1
        try: self.writer(make_message(self.source, self.session, joints, self.objects, self.seq))
        except (OSError, ValueError): pass   # display only: a failed write never stops the run

    def wait(self, seconds):
        if self.speed > 0 and seconds > 0: self.sleep(seconds / self.speed)

    def close(self):
        try: os.remove(path_for(self.session, self.dir))
        except OSError: pass


def from_legacy_sim(sa):
    """The old /tmp/so101_sim_state.json (joints already in dataset degrees; obj / target in the sim base frame, metres) as a SIM message, or None."""
    if not isinstance(sa, dict): return None
    objs = [{"kind": k, "frame": "sim_base", "x": p[0], "y": p[1], "z": p[2]} for k, key in (("nut", "obj"), ("target", "target"))
            for p in [sa.get(key)] if isinstance(p, (list, tuple)) and len(p) >= 3]
    m = validate({"source": "SIM", "session": sa.get("session") or f"legacy-{sa.get('pid', 0)}", "t": sa.get("t"), "seq": sa.get("step", 0), "joints": sa.get("pos"), "objects": objs})
    if m: m["age_s"] = 0.0
    return m


def glide(sink, frm, goal, seconds, objects=None, hz=20.0):
    """Send `sink` the frames of a smoothstep move from pose `frm` (None: jump) to `goal` over `seconds` of virtual time, pacing with sink.wait. Returns the new pose."""
    if not sink.active: return dict(goal)
    a = frm or goal; n = max(1, int(round(seconds * hz))) if frm else 1
    for i in range(1, n + 1):
        u = i / n; e = u * u * (3 - 2 * u)
        sink.frame({k: a[k] + (goal[k] - a[k]) * e for k in goal}, objects); sink.wait(seconds / n)
    return dict(goal)
