"""Synthetic camera for the calibration tests: renders a checkerboard or an ArUco table under a known camera
model (intrinsics, distortion, pose) so the tests can check what cam_calib recovers against ground truth."""
import math, os
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "jetson", "cam_calib.py")   # the tool under test; tests import this from synth, never from conftest

W, H = 640, 360
K_TRUE = np.array([[520.0, 0.0, 318.0], [0.0, 522.0, 182.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.12, 0.03, 0.0005, -0.0008, 0.0])


def rvec_from_euler(rx, ry, rz):
    """Rotation vector from small Euler angles (radians), applied x then y then z."""
    Rx = cv2.Rodrigues(np.array([rx, 0.0, 0.0]))[0]; Ry = cv2.Rodrigues(np.array([0.0, ry, 0.0]))[0]; Rz = cv2.Rodrigues(np.array([0.0, 0.0, rz]))[0]
    return cv2.Rodrigues(Rz @ Ry @ Rx)[0].ravel()


def distort_image(ideal, K, dist):
    """Warp an ideally-projected image through the lens model: distorted[p] = ideal[undistort(p)]."""
    h, w = ideal.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    grid = np.stack([xs, ys], -1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(grid, K, dist, P=K).reshape(h, w, 2).astype(np.float32)
    return cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def warp_plane_patch(canvas, patch, patch_corners_mm, rvec, tvec, K):
    """Project a planar patch (z=0 in the object frame) into the canvas with an ideal pinhole, darkest-wins."""
    obj = np.array([[x, y, 0.0] for x, y in patch_corners_mm], np.float64)
    img_pts, _ = cv2.projectPoints(obj, np.asarray(rvec, np.float64), np.asarray(tvec, np.float64), K, None)
    ph, pw = patch.shape[:2]
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(patch, Hm, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    np.minimum(canvas, warped, out=canvas)
    return canvas


def board_patch(cols, rows, square_px=60):
    """(cols+1)x(rows+1) squares with one white square of margin all round; inner corners at multiples of square_px."""
    n_c, n_r = cols + 3, rows + 3
    img = np.full((n_r * square_px, n_c * square_px), 255, np.uint8)
    for r in range(1, n_r - 1):
        for c in range(1, n_c - 1):
            if (r + c) % 2 == 0: img[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 0
    return img


def render_board(rvec, tvec, cols=9, rows=6, square_mm=25.0, K=K_TRUE, dist=DIST_TRUE, size=(W, H)):
    """Checkerboard whose first inner corner is the object-frame origin, posed by (rvec, tvec), seen by (K, dist)."""
    patch = board_patch(cols, rows)
    # patch pixel (0,0) is two squares before the first inner corner in both axes
    corners_mm = [(-2 * square_mm, -2 * square_mm), ((cols + 1) * square_mm, -2 * square_mm), ((cols + 1) * square_mm, (rows + 1) * square_mm), (-2 * square_mm, (rows + 1) * square_mm)]
    canvas = np.full((size[1], size[0]), 255, np.uint8)
    warp_plane_patch(canvas, patch, corners_mm, rvec, tvec, K)
    return distort_image(canvas, K, dist) if dist is not None and np.any(dist) else canvas


def board_poses(n=14, cols=9, rows=6, square_mm=25.0, seed=3):
    """Varied, fully-in-view board poses in front of a downward-looking camera."""
    rng = np.random.default_rng(seed); poses = []
    cx, cy = cols * square_mm / 2, rows * square_mm / 2
    while len(poses) < n:
        rx, ry, rz = math.radians(rng.uniform(-28, 28)), math.radians(rng.uniform(-28, 28)), math.radians(rng.uniform(-40, 40))
        z = rng.uniform(430, 720); rvec = rvec_from_euler(rx, ry, rz)
        R = cv2.Rodrigues(rvec)[0]; centre = R @ np.array([cx, cy, 0.0])
        # push the board toward the frame edges (in mm at this depth) so the lens model is observed where it matters
        half_w, half_h = z * (W / 2) / K_TRUE[0, 0], z * (H / 2) / K_TRUE[1, 1]
        tvec = np.array([rng.uniform(-half_w, half_w), rng.uniform(-half_h, half_h), z]) - centre
        pts, _ = cv2.projectPoints(np.array([[-2 * square_mm, -2 * square_mm, 0], [(cols + 1) * square_mm, -2 * square_mm, 0], [(cols + 1) * square_mm, (rows + 1) * square_mm, 0], [-2 * square_mm, (rows + 1) * square_mm, 0]], np.float64), rvec, tvec, K_TRUE, None)
        if pts[..., 0].min() > 4 and pts[..., 0].max() < W - 4 and pts[..., 1].min() > 4 and pts[..., 1].max() < H - 4: poses.append((rvec, tvec))
    return poses


def render_table(layout, rvec, tvec, K=K_TRUE, dist=DIST_TRUE, size=(W, H), dict_name="DICT_4X4_50", marker_px=240, angles_deg=None):
    """Table plane (z=0) with ArUco markers at the layout centres, seen by a camera at (rvec, tvec).
    angles_deg: optional {id: degrees} in-plane rotation of each marker about its centre (sloppy gluing)."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)); half = layout["marker_size_mm"] / 2
    canvas = np.full((size[1], size[0]), 255, np.uint8); quiet = marker_px // 6
    for mid, (x, y) in layout["markers"].items():
        m = cv2.aruco.generateImageMarker(d, int(mid), marker_px)
        patch = cv2.copyMakeBorder(m, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
        q = 2 * half * quiet / marker_px   # quiet zone in mm (marker_px pixels span the full marker_size, not half of it)
        a = math.radians((angles_deg or {}).get(int(mid), 0.0)); ca, sa = math.cos(a), math.sin(a)
        local = [(-half - q, half + q), (half + q, half + q), (half + q, -half - q), (-half - q, -half - q)]
        corners = [(x + u * ca - v * sa, y + u * sa + v * ca) for u, v in local]
        warp_plane_patch(canvas, patch, corners, rvec, tvec, K)
    return distort_image(canvas, K, dist) if dist is not None and np.any(dist) else canvas


def table_camera_pose(height_mm=620.0, tilt_deg=6.0, look_at=(200.0, 150.0)):
    """A camera `height_mm` above the table, looking roughly straight down at `look_at`, slightly tilted and
    rolled so the table +Y axis points up the image (operator stands at -Y)."""
    # object (table) -> camera: start with camera z pointing down at the table (R flips z and y)
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tilt = cv2.Rodrigues(np.array([math.radians(tilt_deg), math.radians(-tilt_deg / 2), math.radians(3.0)]))[0]
    R = tilt @ R0
    # place the camera so `look_at` sits on the optical axis `height_mm` away: P_cam = R P + t = (0, 0, h)
    tvec = np.array([0.0, 0.0, height_mm]) - R @ np.array([look_at[0], look_at[1], 0.0])
    return cv2.Rodrigues(R)[0].ravel(), tvec


def default_layout():
    return {"dictionary": "DICT_4X4_50", "marker_size_mm": 40.0,
            "markers": {0: [40.0, 40.0], 1: [360.0, 40.0], 2: [360.0, 260.0], 3: [40.0, 260.0], 4: [200.0, 30.0], 5: [200.0, 270.0]},
            "arm_base_mm": [200.0, 330.0], "arm_yaw_deg": -90.0}
