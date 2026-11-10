"""The kinematic arm mask: detect(arm_px=...) masks the arm where the follower's joints say it is, so the gripper's fingers
reaching over the mat are never reported as objects (the bug TC saw on the wall on 9 Nov). The arm_px structure comes from
cam_calib.arm_mask_px_from_joints; the frames are new.jpg (empty mat, calib_b's view) with synthetic blobs pasted in."""
import numpy as np, pytest
from shelpers import paste_square, JOGGED_97

WHITE = (250, 250, 250)
IN = 25.4


def dist_mm(o, x, y): return float(np.hypot(o["x_mm"] - x, o["y_mm"] - y))


def blob_at_px(scene, raw, u, v, size_mm, color=WHITE):
    """A square of size_mm on the mat whose centre lands on raw pixel (u, v)."""
    x, y = scene.px_to_mm(u, v); return paste_square(scene, raw, x, y, size_mm, color=color)


@pytest.fixture
def scene_arm(sd, calib_b_arm, frames):
    s = sd.MatScene(calib_b_arm); s.set_background(frames["new"]); return s


@pytest.fixture
def arm97(cc, calib_b_arm):
    return cc.arm_mask_px_from_joints(JOGGED_97, calib_b_arm)


# ---------------------------------------------------------------- the bug and the fix
def test_gripper_blob_is_an_object_without_the_mask_and_arm_with_it(scene_arm, arm97, frames):
    assert arm97["source"] == "pose" and arm97["names"][5] == "tip"
    u, v = arm97["points"][5]
    f = blob_at_px(scene_arm, frames["new"], u, v, 20)
    r0 = scene_arm.detect(f)                                                     # edge heuristic alone: the detached gripper blob is "an object"
    assert r0["arm_masked_by"] == "edge" and len(r0["objects"]) == 1 and np.hypot(r0["objects"][0]["px"][0] - u, r0["objects"][0]["px"][1] - v) < 4
    r = scene_arm.detect(f, arm_px=arm97)
    assert r["objects"] == [] and r["arm_in_view"] is True and r["arm_masked_by"] == "kinematics"
    assert 0.01 < r["arm_mask_frac"] < 0.5
    kin = scene_arm._last_kin; c, row = scene_arm.mm_to_warp(*scene_arm.px_to_mm(u, v))
    assert kin[int(row), int(c)] == 255 and scene_arm.last["arm"][int(row), int(c)] == 255, "the tip pixel is inside the kinematic mask, and the mask is the arm"
    assert scene_arm.last["fg"][int(row), int(c)] == 0


def test_same_blob_80_mm_from_the_tip_is_reported(scene_arm, arm97, frames):
    u, v = arm97["points"][5]; x, y = scene_arm.px_to_mm(u, v)
    assert abs(x - 9 * IN) < 4 and abs(y - 7 * IN) < 4, "the jogged pose puts the tip on mat (9, 7) in"
    f = paste_square(scene_arm, frames["new"], x - 80, y, 14, color=WHITE)        # 80 mm to the arm's side, clear of every link
    r = scene_arm.detect(f, arm_px=arm97)
    assert len(r["objects"]) == 1 and dist_mm(r["objects"][0], x - 80, y) < 4 and r["arm_masked_by"] == "kinematics" and r["arm_in_view"]
    f2 = blob_at_px(scene_arm, f, u, v, 20)                                       # both blobs: only the far one survives
    r2 = scene_arm.detect(f2, arm_px=arm97)
    assert len(r2["objects"]) == 1 and dist_mm(r2["objects"][0], x - 80, y) < 4


def test_blob_touching_the_mask_is_absorbed(scene_arm, arm97, frames):
    u, v = arm97["points"][5]; x, y = scene_arm.px_to_mm(u, v); r_mm = arm97["radii_px"][5] / 1.11
    f = paste_square(scene_arm, frames["new"], x - (r_mm + 8), y, 20, color=WHITE)   # overlaps the tip circle's rim
    assert scene_arm.detect(f, arm_px=arm97)["objects"] == []
    assert len(scene_arm.detect(f)["objects"]) == 1, "it is a real blob: the edge heuristic reports it"


def test_hex_nut_still_found_with_an_arm_far_from_it(scene_a, frames):
    """obj.jpg + calib_a: the arm parked at the top edge of that frame, the nut at (10.97, 6.94) in. An arm_px far from the
    nut (hand-made circles along the arm edge near x = 1..3 in) changes nothing about the nut."""
    far = {"points": [list(scene_a.mm_to_px(x, y)) for x, y in ((40, -60), (50, 0), (60, 40))], "radii_px": [50, 40, 30]}
    r = scene_a.detect(frames["obj"], arm_px=far)
    assert len(r["objects"]) == 1 and abs(r["objects"][0]["x_in"] - 10.97) * IN < 6 and abs(r["objects"][0]["y_in"] - 6.94) * IN < 6
    assert r["arm_masked_by"] == "kinematics" and r["arm_in_view"]
    r_edge = scene_a.detect(frames["obj"])
    assert r_edge["arm_masked_by"] == "edge" and r_edge["arm_mask_frac"] <= r["arm_mask_frac"], "the edge heuristic still runs underneath: the mask only grows"


def test_fallback_without_camera_pose_masks_the_gripper_too(sd, cc, calib_b_arm, frames):
    flat = {k: v for k, v in calib_b_arm.items() if k != "camera_pose_est"}
    s = sd.MatScene(flat); s.set_background(frames["new"])
    arm = cc.arm_mask_px_from_joints(JOGGED_97, flat)
    assert arm["source"] == "homography" and len(arm["points"]) == 7
    u, v = arm["points"][5]
    f = blob_at_px(s, frames["new"], u, v, 20)
    assert len(s.detect(f)["objects"]) == 1
    r = s.detect(f, arm_px=arm); assert r["objects"] == [] and r["arm_masked_by"] == "kinematics" and r["arm_in_view"]
    x, y = s.px_to_mm(u, v)
    r2 = s.detect(paste_square(s, frames["new"], x - 110, y, 14, color=WHITE), arm_px=arm)      # the inflated radii still leave the mat usable
    assert len(r2["objects"]) == 1 and dist_mm(r2["objects"][0], x - 110, y) < 4


# ---------------------------------------------------------------- the arm_px contract
def test_arm_px_validation_and_empty(scene_arm, frames):
    with pytest.raises(ValueError): scene_arm.detect(frames["new"], arm_px=[[1, 2]])
    with pytest.raises(ValueError): scene_arm.detect(frames["new"], arm_px={"points": [[1, 2]], "radii_px": []})
    r = scene_arm.detect(frames["new"], arm_px={"points": [], "radii_px": []})
    assert r["arm_masked_by"] == "edge" and r["objects"] == [] and not r["arm_in_view"], "no usable point: nothing is masked, the edge rule stands"
    r = scene_arm.detect(frames["new"], arm_px={"points": [[float("nan"), 1.0], [5000.0, 5000.0]], "radii_px": [10.0, 10.0]})
    assert r["arm_masked_by"] == "kinematics" and not r["arm_in_view"], "a point off the frame masks nothing on the mat"


def test_mask_is_circles_joined_by_thick_strokes(scene_arm):
    """Two circles 60 mm apart with radii 10 and 30 px: the mid-point between them is masked (the stroke is as thick as
    the larger radius), the far side of the small circle beyond its own radius is not."""
    a = scene_arm.mm_to_px(150, 150); b = scene_arm.mm_to_px(210, 150)
    kin = scene_arm.kinematic_mask({"points": [a, b], "radii_px": [10, 30]})
    assert kin.shape == (scene_arm.Hh, scene_arm.W) and kin.dtype == np.uint8
    def at(x, y):
        c, r = scene_arm.mm_to_warp(x, y); return kin[int(r), int(c)]
    assert at(180, 150) == 255 and at(180, 150 + 20) == 255 and at(150, 150) == 255 and at(210, 150) == 255
    assert at(150, 150 + 40) == 0 and at(150 - 40, 150) == 0 and at(300, 150) == 0
    assert scene_arm.kinematic_mask({"points": [[1, 1]], "radii_px": [0]}) is None


def test_no_background_reports_none(sd, calib_b_arm, frames, arm97):
    s = sd.MatScene(calib_b_arm)
    r = s.detect(frames["new"], arm_px=arm97)
    assert r["arm_masked_by"] == "none" and r["objects"] == [] and r["background"] == "missing"


def test_update_background_treats_the_masked_arm_as_busy(scene_arm, arm97, frames):
    u, v = arm97["points"][5]
    f = blob_at_px(scene_arm, frames["new"], u, v, 20)
    c, row = scene_arm.mm_to_warp(*scene_arm.px_to_mm(u, v)); c, row = int(c), int(row)
    fc, frow = scene_arm.mm_to_warp(100, 250); fc, frow = int(fc), int(frow)        # a free mat pixel, far from the arm
    before, free_before = scene_arm.bg[row, c].copy(), scene_arm.bg[frow, fc].copy()
    scene_arm.detect(f, arm_px=arm97)
    bright = np.clip(f.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    for _ in range(5): scene_arm.update_background(bright, alpha=0.5)             # reuses the last detect's mask
    assert np.allclose(scene_arm.bg[row, c], before), "the gripper is never learned into the background"
    assert scene_arm.bg[frow, fc].mean() > free_before.mean() * 1.1, "the free mat followed the light"
    assert scene_arm.detect(f, arm_px=arm97)["objects"] == []
