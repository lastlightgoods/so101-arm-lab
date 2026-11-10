"""Fixtures for the camera-calibration tool. Everything is synthetic: no camera, no Jetson, no network beyond localhost."""
import importlib.util, os, sys
import cv2, numpy as np, pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402
from synth import SCRIPT  # noqa: E402


@pytest.fixture(scope="session")
def cc():
    spec = importlib.util.spec_from_file_location("cam_calib", SCRIPT); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def board_images():
    """14 synthetic board views under the true camera model: list of (name, gray image)."""
    return [(f"board_{i:03d}.png", synth.render_board(r, t)) for i, (r, t) in enumerate(synth.board_poses(14))]


@pytest.fixture(scope="session")
def boards_dir(tmp_path_factory, board_images):
    d = tmp_path_factory.mktemp("boards")
    for name, im in board_images: cv2.imwrite(str(d / name), im)
    return d


@pytest.fixture(scope="session")
def true_intrinsics():
    return {"image_size": [synth.W, synth.H], "camera_matrix": synth.K_TRUE.tolist(), "dist_coeffs": synth.DIST_TRUE.tolist()}


@pytest.fixture(scope="session")
def table_scene():
    """A rendered ArUco table plus the ground truth that made it."""
    layout = synth.default_layout(); rvec, tvec = synth.table_camera_pose()
    img = synth.render_table(layout, rvec, tvec)
    return {"layout": layout, "rvec": rvec, "tvec": tvec, "img": img, "camera_position": (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()}
