"""The 3D stage follows /api/calib: the real 18x12 in mat and its ArUco markers sit where the measured arm pose puts them, and
the follower's joint offsets are applied. These tests let the three.js / urdf-loader modules load from their CDNs (Page(cdn=True);
software WebGL via SwiftShader), so they need the network and skip cleanly without it."""
import math
import os
import socket

import pytest

import apidata as fx
from uihelpers import FIRST_POLLS, Page

IN = 0.0254
SHOT = os.environ.get("SO101_STAGE_SHOT")   # path: save a screenshot of the calibrated stage (for a report)


def _cdn_reachable():
    for host in ("cdnjs.cloudflare.com", "cdn.jsdelivr.net"):
        try:
            socket.create_connection((host, 443), timeout=3).close()
        except OSError:
            return False
    return True


def mat_to_stage(mx_in, my_in, arm):
    """The contract's mat(in) -> arm(mm) conversion, then URDF (X, Y, Z) -> three.js (X, Z, -Y), metres."""
    bx, by = arm["base_mm"]
    yaw = math.radians(arm["yaw_deg"])
    dx, dy = mx_in * 25.4 - bx, my_in * 25.4 - by
    ax = dx * math.cos(yaw) + dy * math.sin(yaw)
    ay = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return [ax / 1000, arm["table_z_mm"] / 1000, -ay / 1000]


def world(page, name):
    """World position of a named scene object (parents' matrices refreshed first)."""
    return page.eval(f"(()=>{{const o=window.__scene.getObjectByName({name!r});if(!o)return null;o.updateWorldMatrix(true,false);"
                     f"const v=new window.__THREE.Vector3();o.getWorldPosition(v);return [v.x,v.y,v.z]}})()")


def close_to(a, b, tol=0.002):
    return a is not None and all(abs(x - y) <= tol for x, y in zip(a, b))


def joint(page, robot, name):
    return page.eval(f"(()=>{{const r=window.__robots&&window.__robots.{robot};if(!r)return null;const j=r.joints[{name!r}];"
                     f"return j.angle!==undefined?j.angle:j.jointValue[0]}})()")


def wait_joint(page, robot, name, expected, tol=1e-5, timeout=20):
    """apply() eases each joint toward its target by 30 % a frame; wait for it to settle on `expected` radians."""
    page.wait_until(f"(()=>{{const r=window.__robots&&window.__robots.{robot};if(!r)return false;const j=r.joints[{name!r}];"
                    f"const a=j.angle!==undefined?j.angle:j.jointValue[0];return Math.abs(a-({expected!r}))<{tol}}})()", timeout=timeout)
    return joint(page, robot, name)


@pytest.fixture
def stage(chrome, stub):
    """Opens the dashboard with the CDN allowed and waits until the stage (mat) and the URDF follower exist."""
    if not _cdn_reachable():
        pytest.skip("three.js CDN not reachable")
    pages = []

    def open_():
        p = Page(chrome, stub.url + "/", cdn=True)
        stub.wait_hits(FIRST_POLLS + ["/api/calib"])
        p.wait_until("window.calibSeq>=1&&!!window.__mat&&!!window.__scene", timeout=60)      # the module ran and built the mat
        p.wait_until("!!(window.__robots&&window.__robots.follower&&window.__robots.leader)", timeout=60)   # URDF + STL meshes in
        p.freeze_polling()
        pages.append(p)
        return p

    yield open_
    for p in pages:
        try:
            if not p.checked:
                p.check()
        finally:
            p.close()


def arms_state(follower=None, leader=None):
    s = fx.state()
    t = fx.now()
    if follower:
        s["arms"]["follower"] = {"role": "follower", "pos": follower, "src": "lerobot", "rx": t}
    if leader:
        s["arms"]["leader"] = {"role": "leader", "pos": leader, "src": "lerobot", "rx": t}
    return s


FOLLOWER = {"shoulder_pan": 12.0, "shoulder_lift": 20.0, "elbow_flex": -30.0, "wrist_flex": 15.0, "wrist_roll": 5.0, "gripper": 40.0}
LEADER = {"shoulder_pan": -8.0, "shoulder_lift": 33.0, "elbow_flex": 10.0, "wrist_flex": -22.0, "wrist_roll": 0.0, "gripper": 10.0}


def test_mat_and_markers_follow_the_calibration(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    arm = fx.CALIB_ARM
    pose = page.eval("window.__calibPose")
    assert pose["calibrated"] is True and pose["yaw_deg"] == 90.15 and pose["table_z_mm"] == 3.3

    # the four ArUco squares exist, named aruco_<id>, 2 in on a side, at the layout positions
    for mid, (mx, my) in fx.CALIB_MARKERS.items():
        o = page.eval(f"(()=>{{const o=window.__scene.getObjectByName('aruco_{mid}');if(!o)return null;o.geometry.computeBoundingBox();const b=o.geometry.boundingBox;"
                      f"return {{mesh:o.isMesh,w:b.max.x-b.min.x,h:b.max.y-b.min.y,u:o.userData}}}})()")
        assert o and o["mesh"], f"aruco_{mid} missing"
        assert abs(o["w"] - 2 * IN) < 1e-6 and abs(o["h"] - 2 * IN) < 1e-6
        assert o["u"]["aruco"] == int(mid) and o["u"]["mat_in"] == [mx, my]
        assert close_to(world(page, f"aruco_{mid}"), mat_to_stage(mx, my, arm)), (mid, world(page, f"aruco_{mid}"), mat_to_stage(mx, my, arm))

    # marker 0 at (16, 10) and the mat origin land where the conversion says (within 2 mm)
    assert close_to(world(page, "aruco_0"), mat_to_stage(16, 10, arm))
    assert close_to(world(page, "mat_frame"), mat_to_stage(0, 0, arm))
    # the mat plane itself: its near-left corner is the origin, its far-right corner is (18, 12), its centre (9, 6)
    corners = page.eval("(()=>{const m=window.__scene.getObjectByName('mat');m.updateWorldMatrix(true,false);const V=window.__THREE.Vector3;"
                        "const f=(x,y)=>{const v=m.localToWorld(new V(x,y,0));return [v.x,v.y,v.z]};"
                        f"return {{o:f(-{18*IN/2},-{12*IN/2}),far:f({18*IN/2},{12*IN/2}),c:f(0,0)}}}})()")
    assert close_to(corners["o"], mat_to_stage(0, 0, arm))
    assert close_to(corners["far"], mat_to_stage(18, 12, arm))
    assert close_to(corners["c"], mat_to_stage(9, 6, arm))
    assert close_to(page.eval("window.__matToWorld(16,10)"), mat_to_stage(16, 10, arm), 1e-6)
    # the mat's +X edge runs to the arm's right (+z) and +Y runs forward (+x): yaw 90.15 deg
    assert mat_to_stage(18, 0, arm)[2] > mat_to_stage(0, 0, arm)[2] + 0.45
    assert mat_to_stage(0, 12, arm)[0] > mat_to_stage(0, 0, arm)[0] + 0.30
    # the overhead camera frustum hangs over the mat centre at its old height
    top = page.eval("[window.__topPos.x,window.__topPos.y,window.__topPos.z]")
    c = mat_to_stage(9, 6, arm)
    assert abs(top[0] - c[0]) < 1e-6 and abs(top[2] - c[2]) < 1e-6 and top[1] == 0.5
    # handoff zone and robot are still there
    assert page.eval("!!window.__scene.getObjectByName('handoff')")
    assert page.eval("window.__scene.children.filter(o=>o.name==='mat_frame').length") == 1
    if SHOT:
        page.eval("(()=>{window.__ctl.autoRotate=false;window.__cam.position.set(0.9,0.75,0.6);window.__ctl.target.set(0.25,0.0,0);window.__ctl.update();return true})()")
        page.eval("(document.querySelector('#arm3d').scrollIntoView({block:'center'}),true)")
        page.eval("new Promise(r=>{let n=0;const f=()=>++n>12?r(true):requestAnimationFrame(f);f()})")   # a few rendered frames
        page.screenshot(SHOT)
    page.check()


def test_joint_offsets_are_applied_with_calibration(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    assert page.eval("window.calibOffsets") == fx.CALIB_ARM["joint_offsets_deg"]
    stub.routes["/api/state"] = arms_state(FOLLOWER, LEADER)
    page.eval("arms()")
    off = fx.CALIB_ARM["joint_offsets_deg"]
    for robot, pos in (("follower", FOLLOWER), ("leader", LEADER)):
        for j in ("shoulder_lift", "elbow_flex", "wrist_flex"):
            expected = (pos[j] + off[j]) * math.pi / 180
            assert abs(wait_joint(page, robot, j, expected) - expected) < 1e-5, (robot, j)
        for j in ("shoulder_pan", "wrist_roll"):                       # no offset measured for these: plain degrees
            expected = pos[j] * math.pi / 180
            assert abs(wait_joint(page, robot, j, expected) - expected) < 1e-5, (robot, j)
    # sanity: the offset really moved the joint off the naive value
    assert abs(joint(page, "follower", "shoulder_lift") - 20 * math.pi / 180) > 0.05
    page.check()


def test_without_calibration_the_stage_still_renders_and_no_offsets(stub, stage):
    assert stub.routes["/api/calib"] == fx.calib_none()
    page = stage()
    pose = page.eval("window.__calibPose")
    assert pose["calibrated"] is False
    assert page.eval("window.calib") is None and page.eval("window.calibOffsets") == {}
    # the stage is not blank: a WebGL canvas, the mat, a 1-inch-grid texture, the markers at the default layout, the robot
    assert page.eval("!!document.querySelector('#arm3d canvas')")
    assert page.eval("(()=>{const m=window.__scene.getObjectByName('mat');return m&&m.isMesh&&!!m.material.map&&m.visible})()")
    assert page.eval("[0,1,2,3].every(i=>!!window.__scene.getObjectByName('aruco_'+i))")
    assert page.eval("window.__robots.follower.visible") is True
    assert page.eval("Object.keys(window.__robots.follower.joints).length") >= 6
    # a fixed default pose keeps the mat in front of the arm: origin on the arm's left, +X edge to its right
    o, far = page.eval("window.__matToWorld(0,0)"), page.eval("window.__matToWorld(18,12)")
    assert o[2] < 0 < far[2] and far[0] > o[0] > 0
    assert close_to(world(page, "aruco_0"), page.eval("window.__matToWorld(16,10)"))
    # joints: plain degrees, no offsets
    stub.routes["/api/state"] = arms_state(FOLLOWER)
    page.eval("arms()")
    for j in ("shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan"):
        expected = FOLLOWER[j] * math.pi / 180
        assert abs(wait_joint(page, "follower", j, expected) - expected) < 1e-5, j
    page.check()


def test_calibration_arriving_later_rebuilds_the_mat(stub, stage):
    page = stage()
    before = world(page, "aruco_0")
    assert page.eval("window.__calibPose.calibrated") is False
    stub.routes["/api/calib"] = fx.calib()
    page.eval("calibPoll()")
    page.wait_until("window.__calibPose.calibrated===true")
    assert close_to(world(page, "aruco_0"), mat_to_stage(16, 10, fx.CALIB_ARM))
    assert not close_to(world(page, "aruco_0"), before, 0.01)
    assert page.eval("window.__scene.children.filter(o=>o.name==='mat_frame').length") == 1   # the old mat group is gone
    assert page.eval("window.calibOffsets") == fx.CALIB_ARM["joint_offsets_deg"]
    # a camera-only calibration (arm fit not done yet) keeps the fallback pose and zero offsets
    stub.routes["/api/calib"] = fx.calib(arm=False)
    page.eval("calibPoll()")
    page.wait_until("window.__calibPose.calibrated===false")
    assert page.eval("window.calibOffsets") == {}
    assert close_to(world(page, "aruco_0"), before)
    # a broken route is the same as no calibration
    stub.fail["/api/calib"] = 500
    page.eval("calibPoll()")
    assert page.eval("window.calib") is None and page.eval("window.__calibPose.calibrated") is False
    page.check()
