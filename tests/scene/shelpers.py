"""Fixtures for the mat scene detector: real overhead frames from 9 Nov 2026 plus the calibrations that match them.
Hermetic: no camera, no network."""
import importlib.util, json, os
import cv2, numpy as np, pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(HERE, "fixtures")
MODULE = os.path.join(ROOT, "dashboard", "scene_detect.py")
CAM_CALIB = os.path.join(ROOT, "jetson", "cam_calib.py")
REAL_TABLE = os.path.join(ROOT, "tests", "calib", "fixtures", "top_table_real.json")   # jetson/calib/top_table.json of 9 Nov 22:50 (camera pose + arm fit)
JOGGED_97 = {"shoulder_pan": 0.13, "shoulder_lift": 20.88, "elbow_flex": 13.58, "wrist_flex": 60.5, "wrist_roll": 1.1}   # follower joints with the tip on mat (9, 7) in


@pytest.fixture(scope="session")
def sd():
    spec = importlib.util.spec_from_file_location("scene_detect", MODULE); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def cc():
    """jetson/cam_calib.py (forward kinematics, arm_mask_px_from_joints)."""
    spec = importlib.util.spec_from_file_location("scene_cam_calib", CAM_CALIB); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def calib_real():
    return json.load(open(REAL_TABLE))


def pose_for_homography(calib, camera_matrix, n=6):
    """A camera_pose_est (rvec, tvec table -> camera, camera_matrix, no distortion) that agrees with the calibration's
    homography: table points on a grid are pushed through the inverse homography and solvePnP recovers the pose. Lets a
    homography-only fixture calibration be used with the 3D projection path, consistently with its own frames."""
    Hinv = np.linalg.inv(np.array(calib["homography_px_to_table"], np.float64))
    xs = np.linspace(40, 417, n); ys = np.linspace(40, 265, n)
    obj = np.array([[x, y, 0.0] for x in xs for y in ys]); q = np.hstack([obj[:, :2], np.ones((len(obj), 1))]) @ Hinv.T
    img = q[:, :2] / q[:, 2:3]; K = np.array(camera_matrix, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE); assert ok
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec); C = (-R.T @ tvec).ravel()
    return {"position_mm": C.tolist(), "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(), "camera_matrix": K.tolist(),
            "image_size": list(calib.get("image_size", [640, 360])), "source": "derived from the homography for the tests"}


@pytest.fixture(scope="session")
def calib_b_arm(calib_b, calib_real):
    """calib_b (matches new.jpg) with the real arm fit of the same mat layout and a camera pose consistent with calib_b's view."""
    c = json.loads(json.dumps(calib_b))
    c.update(arm_base_mm=calib_real["arm_base_mm"], arm_yaw_deg=calib_real["arm_yaw_deg"], arm_fit=calib_real["arm_fit"])
    c["camera_pose_est"] = pose_for_homography(calib_b, calib_real["camera_pose_est"]["camera_matrix"])
    return c


def _img(name):
    im = cv2.imread(os.path.join(FIX, name), cv2.IMREAD_COLOR)
    assert im is not None, name
    return im


@pytest.fixture(scope="session")
def calib_a():
    """16:30 calibration: markers 0/1 on the arm side; matches bg.jpg, obj.jpg, top_now.jpg."""
    return json.load(open(os.path.join(FIX, "calib_a.json")))


@pytest.fixture(scope="session")
def calib_b():
    """21:17 calibration: the mat turned round and the markers re-glued; matches new.jpg."""
    return json.load(open(os.path.join(FIX, "calib_b.json")))


@pytest.fixture(scope="session")
def frames():
    return {n: _img(n + ".jpg") for n in ("bg", "obj", "top_now", "new")}


@pytest.fixture
def scene_a(sd, calib_a, frames):
    s = sd.MatScene(calib_a); s.set_background(frames["bg"]); return s


@pytest.fixture
def scene_b(sd, calib_b, frames):
    s = sd.MatScene(calib_b); s.set_background(frames["new"]); return s


def paste_square(scene, raw_bg, x_mm, y_mm, size_mm=30.0, color=(40, 60, 230), angle_deg=0.0, length_mm=None):
    """A synthetic object: a coloured square of `size_mm` centred at (x_mm, y_mm) on the mat, drawn in mat space and
    un-warped into the raw camera frame through the inverse homography, composited over the clean frame."""
    hi = 4.0                                                                     # mm -> px of the drawing canvas
    W, H = int(round(scene.mat_w * hi)), int(round(scene.mat_h * hi))
    canvas = np.zeros((H, W, 3), np.uint8); mask = np.zeros((H, W), np.uint8)
    c = (x_mm * hi, (scene.mat_h - y_mm) * hi)
    box = cv2.boxPoints((c, ((size_mm if length_mm is None else length_mm) * hi, size_mm * hi), -angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(canvas, box, color); cv2.fillConvexPoly(mask, box, 255)
    A = np.array([[hi, 0, 0], [0, -hi, scene.mat_h * hi], [0, 0, 1]])            # mm -> canvas px
    M = np.linalg.inv(A @ scene.H)                                               # canvas px -> raw px
    h, w = raw_bg.shape[:2]
    col = cv2.warpPerspective(canvas, M, (w, h), flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)[:, :, None] / 255.0
    out = raw_bg.astype(np.float32) * (1 - m) + col.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)
