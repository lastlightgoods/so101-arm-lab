"""tools/pick_loop.py's whole-mat search cuts the arm out (10 Nov 2026): with no hint it took the parked arm's dark parts at mat (8.12, 3.85)
for the nut, closer to the mat centre than the real nut at (11.33, 5.81); first10b cycles 9-10 went there and found nothing in the jaws."""
import math
from pickhelpers import overhead_frame

REST = {"shoulder_pan": -0.7, "shoulder_lift": -100.2, "elbow_flex": 95.3, "wrist_flex": 71.9, "wrist_roll": 0.2}   # the parked arm, /api/state 10 Nov 2026 23:3x
GHOST_IN, NUT_IN = (8.12, 3.85), (11.33, 5.81)


def _masked(pl, cc, table, joints, pt_in):
    import numpy as np
    w, h = table.get("image_size") or [640, 360]; m = pl.arm_cut(cc, table, np.full((h, w), 255, np.uint8), joints)
    u, v = cc.table_to_pixel(table, pt_in[0] * 25.4, pt_in[1] * 25.4); return m[int(round(v)), int(round(u))] == 0


def test_the_parked_arm_covers_the_ghost_and_not_the_nut(pl, cc, table):
    assert _masked(pl, cc, table, REST, GHOST_IN), "the spot the search took for the nut is under the parked arm's silhouette"
    assert not _masked(pl, cc, table, REST, NUT_IN) and not _masked(pl, cc, table, REST, (9.0, 6.0)), "the nut and the target are clear of it"
    assert not _masked(pl, cc, table, None, GHOST_IN), "no joints: nothing cut"


def test_the_whole_mat_search_skips_the_arm_when_it_knows_the_joints(pl, cc, table):
    frame = overhead_frame(cc, table, [GHOST_IN, NUT_IN], r_px=4)
    old = pl.find_nut_mat(cc, table, None, jpeg=frame, markers_in=[], joints=None)
    assert old and math.hypot(old[0][0] - GHOST_IN[0], old[0][1] - GHOST_IN[1]) * 25.4 < 3.0, "without the cut-out: the ghost, nearer the mat centre"
    new = pl.find_nut_mat(cc, table, None, jpeg=frame, markers_in=[], joints=REST)
    assert new and math.hypot(new[0][0] - NUT_IN[0], new[0][1] - NUT_IN[1]) * 25.4 < 3.0, "with it: the nut"


def test_live_joints_are_read_from_the_dashboard_state(pl, monkeypatch):
    monkeypatch.undo()   # the fixture's stub off: the real follower_joints against a faked api
    import importlib.util, os
    from pickhelpers import TOOLS
    spec = importlib.util.spec_from_file_location("pick_loop_joints", os.path.join(TOOLS, "pick_loop.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.api = lambda path: {"arms": {"follower": {"pos": {**REST, "gripper": 0.5}}}} if path == "/api/state" else {}
    assert m.follower_joints() == REST
    m.api = lambda path: {"arms": {}}; assert m.follower_joints() is None
    def boom(path): raise OSError("dashboard down")
    m.api = boom; assert m.follower_joints() is None


# 11 Nov 2026 13:45: the NEAR search (find_nut) took a dark slot on the gripper for the nut, 14 mm from the real one; it had no arm cut-out
def test_the_near_search_skips_the_arm_when_it_knows_the_joints(pl, cc, table):
    frame = overhead_frame(cc, table, [GHOST_IN, NUT_IN], r_px=4); near = (9.2, 4.6)   # between them, the ghost (under the arm) is the nearer
    old = pl.find_nut(cc, table, near, radius_px=140, jpeg=frame, joints=None)
    assert old and math.hypot(old[0][0] - GHOST_IN[0], old[0][1] - GHOST_IN[1]) * 25.4 < 3.0, "without the cut-out: the ghost"
    new = pl.find_nut(cc, table, near, radius_px=140, jpeg=frame, joints=REST)
    assert new and math.hypot(new[0][0] - NUT_IN[0], new[0][1] - NUT_IN[1]) * 25.4 < 3.0, "with it: the nut"


def test_the_near_search_reads_the_live_joints_by_default_and_survives_a_missing_arm(pl, cc, table, monkeypatch):
    frame = overhead_frame(cc, table, [GHOST_IN, NUT_IN], r_px=4); near = (9.2, 4.6)
    monkeypatch.setattr(pl, "follower_joints", lambda: REST)
    new = pl.find_nut(cc, table, near, radius_px=140, jpeg=frame); assert new and math.hypot(new[0][0] - NUT_IN[0], new[0][1] - NUT_IN[1]) * 25.4 < 3.0
    monkeypatch.setattr(pl, "follower_joints", lambda: None)   # dashboard down: no cut, the search still answers
    assert pl.find_nut(cc, table, near, radius_px=140, jpeg=frame) is not None
    monkeypatch.setattr(cc, "arm_mask_px_from_joints", lambda *a, **k: 1 / 0)   # the arm mask itself fails: also no cut
    assert pl.find_nut(cc, table, near, radius_px=140, jpeg=frame, joints=REST) is not None
