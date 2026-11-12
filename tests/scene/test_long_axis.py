"""The angle convention of dashboard/scene_detect.py (11 Nov 2026): angle_deg is the LONG side's direction counter-clockwise from mat +X in
mat millimetres (X along the long edge, Y away from the arm), and long_axis_mat turns it into a unit vector in the frame of x_in / y_in. Proved
on synthetic bars drawn at known mat angles, un-warped into a camera frame and run through MatScene.detect, the whole code path of /api/scene."""
import math
import pytest
from shelpers import paste_square

ANGLES = [0, 20, 45, 70, 90, 110, 135, 160, 179]


def axis_err_deg(u, ang):
    """Angle between the axis u and the line at `ang` degrees (an axis has no sign), degrees."""
    d = abs(u[0] * math.cos(math.radians(ang)) + u[1] * math.sin(math.radians(ang))); return math.degrees(math.acos(min(1.0, d)))


@pytest.mark.parametrize("ang", ANGLES)
def test_a_bar_at_a_known_mat_angle_gives_that_axis(sd, scene_a, frames, ang):
    o = scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 12, angle_deg=ang, length_mm=40))["objects"]
    assert len(o) == 1 and o[0]["w_mm"] > 2.5 * o[0]["h_mm"]
    u = sd.long_axis_mat(o[0]); assert u is not None and abs(math.hypot(*u) - 1.0) < 1e-9
    assert axis_err_deg(u, ang) < 5.0, f"bar at {ang} deg: angle_deg {o[0]['angle_deg']} gave axis {u}"


@pytest.mark.parametrize("x,y", [(100, 100), (330, 220), (260, 80)])
def test_the_axis_does_not_depend_on_where_on_the_mat(sd, scene_a, frames, x, y):
    o = scene_a.detect(paste_square(scene_a, frames["bg"], x, y, 12, angle_deg=30, length_mm=40))["objects"][0]; assert axis_err_deg(sd.long_axis_mat(o), 30) < 5.0


def test_the_axis_is_in_inches_orientation_not_pixels(sd, scene_a, frames):
    """A bar rising to the upper right on the mat (+x, +y) is angle +30: the pixel rows run the other way, so a pixel-frame reading would say -30."""
    o = scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 12, angle_deg=30, length_mm=40))["objects"][0]; u = sd.long_axis_mat(o)
    assert u[0] * u[1] > 0 and 0.8 < abs(u[1] / u[0]) / math.tan(math.radians(30)) < 1.2


def test_round_and_nearly_round_objects_have_no_axis(sd, scene_a, frames):
    assert sd.long_axis_mat(scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 30))["objects"][0]) is None
    assert sd.long_axis_mat({"w_mm": 15.0, "h_mm": 12.0, "angle_deg": 10.0}) is None and sd.long_axis_mat({"w_mm": 15.0, "h_mm": 10.0, "angle_deg": 10.0}) is not None


def test_long_axis_rejects_junk(sd):
    for o in ({}, {"w_mm": 0, "h_mm": 0, "angle_deg": 0}, {"w_mm": "x", "h_mm": 1, "angle_deg": 0}, None): assert sd.long_axis_mat(o) is None
    u = sd.long_axis_mat({"w_mm": 20.0, "h_mm": 5.0, "angle_deg": 90.0}); assert abs(u[0]) < 1e-9 and abs(abs(u[1]) - 1) < 1e-9
