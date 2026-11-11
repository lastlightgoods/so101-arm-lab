"""Fixtures for tools/pick_loop.py's closed loop and tools/wrist_check.py. Nothing reaches a dashboard, an ssh session or the Jetson:
  * `cc` / `wc` / `pl` load jetson/cam_calib.py, tools/wrist_check.py and tools/pick_loop.py by path; `table` is
    tests/calib/fixtures/top_table_real.json (the 9 Nov 2026 arm fit); `pl` gets a tmp ROOT holding that table, the repo's
    wrist_campose.json and labtrack.py, so the loop reads its calibration and writes outputs/run/ where it expects;
  * ShortArm answers /api/ik from the real kinematics and "runs" a plan by reporting reached joints whose FK tip sits `shortfall_mm`
    short of the command along the base->tip direction (what the real arm did on 10 Nov 2026: 5-8 mm) and `high_mm` above it;
  * Recorder stands in for labtrack and keeps params / metrics / artifact names in memory;
  * jaw_frame / frame_feed fake the wrist camera after the lift (a ring between the jaw tips, or not), overhead_frame paints dark blobs
    at mat points into a frame the size of the calibration's image for the whole-mat finder.
Joint solves use a damped Gauss-Newton on fk_tip (~1 ms) rather than cam_calib.ik_tip (0.5 s, searches for a vertical gripper): a fake
arm needs the kinematics, not the posture search."""
import contextlib, importlib.util, json, os, shutil, sys
import numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
JETSON = os.path.join(ROOT, "jetson")
FIXTURE_TABLE = os.path.join(ROOT, "tests", "calib", "fixtures", "top_table_real.json")
WRIST_POSE = os.path.join(JETSON, "calib", "wrist_campose.json")
PROBE_FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "wrist_probe_20261110.jpg")
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PAN_AXIS_MM = np.array([38.8353, 0.0])   # the pan axis sits 38.8 mm ahead of base_link (cam_calib.SO101_CHAIN)
SEED = {"shoulder_pan": 15.0, "shoulder_lift": 17.0, "elbow_flex": 2.0, "wrist_flex": 80.0, "wrist_roll": 0.46}   # the 10 Nov hover: gripper down over the nut
# the real hover of 10 Nov 2026 (probe_result.json): the nut commanded at mat (11.44, 6.26) in appeared at wrist pixel (350, 262), predicted (334.5, 255.8), depth 147 mm
PROBE_JOINTS = {"shoulder_pan": 14.901, "shoulder_lift": 17.451, "elbow_flex": 1.714, "wrist_flex": 80.440, "wrist_roll": 0.571}
PROBE_NUT_IN = (11.44, 6.26)
JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")   # a (headers-only) JPEG: find_nut is faked, the bytes just travel


def load(path, name=None):
    name = name or os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


@pytest.fixture(scope="session")
def cc():
    pytest.importorskip("cv2"); return load(os.path.join(JETSON, "cam_calib.py"), "cam_calib_for_pick_tests")


@pytest.fixture(scope="session")
def wc():
    pytest.importorskip("cv2"); return load(os.path.join(TOOLS, "wrist_check.py"))


@pytest.fixture(scope="session")
def table(): return json.load(open(FIXTURE_TABLE))


@pytest.fixture(scope="session")
def wpose(): return json.load(open(WRIST_POSE))


def tip_of(cc, table, joints):
    """FK tip (base mm) of command joints through the fit's zero offsets and tip offset: exactly what pick_loop.fk_mat measures."""
    fit = table["arm_fit"]; off = fit.get("joint_offsets_deg") or {}
    return cc.fk_tip({k: float(joints[k]) + float(off.get(k, 0.0)) for k in ARM}, tip_offset_mm=fit.get("tip_offset_mm"))[0]


def solve_joints(cc, table, seed, tip_mm, iters=10):
    """Command joints whose FK tip is tip_mm: minimum-norm Gauss-Newton on pan/lift/elbow/wrist_flex from `seed`, roll kept."""
    q = {k: float(seed[k]) for k in ARM}; names = ARM[:4]
    for _ in range(iters):
        t = tip_of(cc, table, q); e = np.asarray(tip_mm, np.float64) - t
        if np.linalg.norm(e) < 1e-4: break
        J = np.zeros((3, 4))
        for i, n in enumerate(names):
            q2 = dict(q); q2[n] += 0.01; J[:, i] = (tip_of(cc, table, q2) - t) / 0.01
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        for i, n in enumerate(names): q[n] += float(dq[i])
    return q


def mat_to_base(cc, table, x_in, y_in, h_mm=0.0):
    ax, ay = cc.table_to_arm(table, x_in * 25.4, y_in * 25.4); return np.array([ax, ay, float(table["arm_fit"]["table_z_mm"]) + h_mm])


class ShortArm:
    """The dashboard's /api/ik and the ssh'd pick_seq, faked on the real kinematics. Every arm waypoint is 'reached' `shortfall_mm`
    short along the base->tip direction and `high_mm` above the command (gripper-only waypoints and rest are reached exactly).
    The nut is found at `nut_in` (not at all when `found` is False: the near search misses and the loop falls back to the scene map
    `scene` / the whole-mat search); at landing it is `land_off_in` from where the loop placed it."""
    def __init__(self, cc, table, shortfall_mm=7.0, high_mm=0.0, nut_in=PROBE_NUT_IN, land_off_in=(0.1, -0.05), found=True, scene=None):
        self.cc, self.table, self.shortfall, self.high, self.nut_in, self.land_off, self.found, self.scene = cc, table, shortfall_mm, high_mm, nut_in, land_off_in, found, list(scene or [])
        self.calls, self.plans, self.rows, self.ik_calls, self.frames = [], [], [], [], 0

    def api(self, path):
        q = dict(p.split("=") for p in path.split("?")[1].split("&")); x, y, h = float(q["x_in"]), float(q["y_in"]), float(q.get("h_in", 0.0)); self.ik_calls.append((x, y, h))
        j = solve_joints(self.cc, self.table, SEED, mat_to_base(self.cc, self.table, x, y, h * 25.4)); j["wrist_roll"] = float(self.table["arm_fit"].get("wrist_roll_deg", 0.0))
        return {"ok": True, "joints_deg": {k: round(v, 3) for k, v in j.items()}}

    def reached(self, joints):
        tip = tip_of(self.cc, self.table, joints); d = tip[:2] - PAN_AXIS_MM; d /= np.linalg.norm(d)
        want = tip.copy(); want[:2] -= self.shortfall * d; want[2] += self.high
        return solve_joints(self.cc, self.table, joints, want)

    def run_plan(self, plan, max_deg_s):
        self.calls.append(([w["name"] for w in plan["waypoints"]], max_deg_s)); self.plans.append(plan); rows = []
        for i, w in enumerate(plan["waypoints"]):
            j = {k: float(w["joints"][k]) for k in ARM}; got = j if (w["gripper_only"] or w["name"] == "rest") else self.reached(j)
            rows.append({"wp": w["name"], "seconds": 1.0, "t": i, "target": {**j, "gripper": w["gripper"]}, "reached": {**{k: round(v, 3) for k, v in got.items()}, "gripper": w["gripper"]},
                         "max_err_deg": round(max(abs(j[k] - got[k]) for k in ARM), 2)})
        self.rows.append(rows); return rows

    def find_nut(self, cc, table, near_in, radius_px=70, jpeg=None):
        if jpeg is None: return (list(self.nut_in), [320, 180], 40) if self.found else None
        self.frames += 1; return [near_in[0] + self.land_off[0], near_in[1] + self.land_off[1]], [320, 180], 40


class RunRec:
    def __init__(self, name, tags): self.name, self.tags, self.params_, self.metrics_, self.files = name, tags or {}, {}, {}, []
    def params(self, d): self.params_.update({k: v for k, v in d.items() if v is not None})
    def metrics(self, d, step=None): self.metrics_.update({k: float(v) for k, v in d.items() if v is not None})   # labtrack drops None the same way
    def json(self, obj, name, subdir=None): self.files.append(f"{subdir}/{name}" if subdir else name)
    def file(self, name, data, subdir=None): self.files.append(f"{subdir}/{name}" if subdir else name)
    def artifact(self, path, subdir=None):
        if os.path.isfile(path): self.files.append(f"{subdir}/{os.path.basename(path)}" if subdir else os.path.basename(path))


class Recorder:
    def __init__(self): self.runs = []

    @contextlib.contextmanager
    def run(self, name, tags=None):
        r = RunRec(name, tags); self.runs.append(r); yield r


@pytest.fixture
def pl(tmp_path, monkeypatch, cc, wc):
    m = load(os.path.join(TOOLS, "pick_loop.py"), "pick_loop_closed")
    calib = tmp_path / "jetson" / "calib"; calib.mkdir(parents=True); shutil.copy(FIXTURE_TABLE, calib / "top_table.json"); shutil.copy(WRIST_POSE, calib / "wrist_campose.json")
    rec = Recorder(); monkeypatch.setattr(m, "ROOT", str(tmp_path)); monkeypatch.setattr(m, "cam_calib", lambda: cc); monkeypatch.setattr(m, "wrist_mod", lambda: wc); monkeypatch.setattr(m, "labtrack", lambda: rec)
    monkeypatch.setattr(m, "rec", rec, raising=False); monkeypatch.setattr(m.time, "sleep", lambda s: None); monkeypatch.setattr(m, "fetch_top", lambda: JPEG); monkeypatch.setattr(m, "fetch_wrist", lambda: None); monkeypatch.setattr(m, "follower_joints", lambda: None)   # why: hermetic: the arm cut-out never reads a live /api/state
    return m


def drive(pl, monkeypatch, capsys, arm, argv):
    """Run pick_loop.main() against `arm`; returns every stdout line parsed as JSON."""
    monkeypatch.setattr(pl, "find_nut", arm.find_nut); monkeypatch.setattr(pl, "api", arm.api); monkeypatch.setattr(pl, "run_plan", arm.run_plan); monkeypatch.setattr(pl, "fetch_scene_objects", lambda: list(arm.scene))
    monkeypatch.setattr(sys, "argv", ["pick_loop.py", *argv]); pl.main()
    return [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]


JAW_RING_UV = (340, 330)   # where the held nut's ring sat between the jaw tips in the 10 Nov diag frame (window 300-380 x 300-360)


def jaw_frame(ring=True, uv=JAW_RING_UV, r=15, jpeg=True):
    """A wrist frame after the lift: grey mat, two dark jaw bars either side of the window and, when `ring`, the nut's bright ring with a dark hole at `uv`."""
    import cv2
    img = np.full((480, 640, 3), 110, np.uint8); cv2.rectangle(img, (250, 290), (298, 480), (40, 40, 40), -1); cv2.rectangle(img, (382, 290), (430, 480), (40, 40, 40), -1)
    if ring: cv2.circle(img, tuple(int(v) for v in uv), r, (235, 235, 235), -1); cv2.circle(img, tuple(int(v) for v in uv), int(r * 0.4), (25, 25, 25), -1)
    return cv2.imencode(".jpg", img)[1].tobytes() if jpeg else img


def frame_feed(*frames):
    """A fetch_wrist stand-in handing out `frames` in order (the last one repeats); `.calls` counts the fetches."""
    def fetch():
        fetch.calls += 1; return frames[min(fetch.calls, len(frames)) - 1]
    fetch.calls = 0; return fetch


def overhead_frame(cc, table, blobs_in=(), r_px=3, bright=200):
    """PNG bytes of an overhead frame the size of the calibration's image: a light mat with a dark disc (radius r_px) at each mat point (inches)."""
    import cv2
    w, h = table.get("image_size") or [640, 360]; img = np.full((h, w, 3), bright, np.uint8)
    for x, y in blobs_in:
        u, v = cc.table_to_pixel(table, x * 25.4, y * 25.4); cv2.circle(img, (int(round(u)), int(round(v))), r_px, (20, 20, 20), -1)
    return cv2.imencode(".png", img)[1].tobytes()
