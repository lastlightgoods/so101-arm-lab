"""The WRIST CAM frustum on the 3D stage follows calib.camera.wrist (jetson/calib/wrist_campose.json served by /api/calib): it hangs
off the follower's gripper_link at position_mm, looks along optical_axis_gripper, is sized by hfov_deg at 4:3, falls back to the old
STL guess without a pose, and is re-attached when a 'calib' event brings a new pose. Reuses test_stage_calib's ``stage`` fixture
(three.js from its CDN, skipped offline)."""
import math

import apidata as fx
from test_stage_calib import stage  # noqa: F401  (the fixture; importing it registers it here)

OLD_POS = [-0.046, -0.022, -0.012]


def wrist_info(page):
    return page.eval("(()=>{const w=window.__wristCam;if(!w)return null;const V=window.__THREE.Vector3;const ax=new V(0,0,-1).applyQuaternion(w.quaternion),up=new V(0,1,0).applyQuaternion(w.quaternion);"
                     "const p=w.geometry.attributes.position.array;let mx=0,my=0,mz=0;for(let i=0;i<p.length;i+=3){mx=Math.max(mx,Math.abs(p[i]));my=Math.max(my,Math.abs(p[i+1]));mz=Math.max(mz,Math.abs(p[i+2]))}"
                     "return {pos:[w.position.x,w.position.y,w.position.z],axis:[ax.x,ax.y,ax.z],up:[up.x,up.y,up.z],parent:w.parent&&w.parent.name,name:w.name,pose:!!w.userData.pose,hfov:w.userData.hfov_deg,g:[mx,my,mz],"
                     "n:window.__robots.follower.links.gripper_link.children.filter(o=>o.name==='wrist_cam').length}})()")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_wrist_frustum_sits_at_the_hand_eye_pose(stub, stage):
    W = fx.wrist_pose()
    stub.routes["/api/calib"] = fx.calib(camera={**fx.calib()["camera"], "wrist": W})
    page = stage()
    page.wait_until("!!(window.__wristCam&&window.__wristCam.userData.pose)")
    w = wrist_info(page)
    assert w["parent"] == "gripper_link" and w["name"] == "wrist_cam" and w["n"] == 1
    assert all(abs(a - b / 1000) < 1e-9 for a, b in zip(w["pos"], W["position_mm"])), "local position = position_mm / 1000 in the URDF gripper_link frame"
    assert dot(w["axis"], W["rotation"]["optical_axis_gripper"]) > 0.99999, (w["axis"], W["rotation"]["optical_axis_gripper"])
    assert dot(w["up"], W["rotation"]["up_gripper"]) > 0.9999, "frustum +y = image up"
    assert abs(w["hfov"] - W["hfov_deg"]) < 1e-9 and abs(w["g"][0] / w["g"][2] - math.tan(math.radians(W["hfov_deg"] / 2))) < 1e-6 and abs(w["g"][1] / w["g"][0] - 3 / 4) < 1e-6, "half-width = len * tan(hfov/2), 4:3"
    # the quaternion and the axis/up vectors agree: dropping the quaternion gives the same frustum orientation
    W2 = fx.wrist_pose(); W2["rotation"] = {k: v for k, v in W2["rotation"].items() if k != "quaternion_xyzw"}
    stub.routes["/api/calib"] = fx.calib(camera={**fx.calib()["camera"], "wrist": W2})
    page.eval("calibPoll()")
    page.wait_until("!!(window.__wristCam&&window.__wristCam.userData.pose&&!window.__wristCam.userData.pose.rotation.quaternion_xyzw)")
    w2 = wrist_info(page)
    assert dot(w2["axis"], w["axis"]) > 0.99999 and dot(w2["up"], w["up"]) > 0.9999 and w2["n"] == 1
    page.check()


def test_wrist_frustum_falls_back_and_follows_calib_events(stub, stage):
    page = stage()                                                   # no calibration at all: the old STL guess
    page.wait_until("!!window.__wristCam")
    w = wrist_info(page)
    assert w["pose"] is False and all(abs(a - b) < 1e-9 for a, b in zip(w["pos"], OLD_POS)) and w["parent"] == "gripper_link"
    n = math.sqrt(0.28 ** 2 + 0.3 ** 2 + 0.91 ** 2)
    assert dot(w["axis"], [0.28 / n, 0.3 / n, -0.91 / n]) > 1 - 1e-9, "the old frustum looks toward the fingertips"
    # a calibration without a wrist pose keeps the guess
    stub.routes["/api/calib"] = fx.calib()
    page.eval("calibPoll()")
    page.wait_until("window.__calibPose.calibrated===true")
    assert wrist_info(page)["pose"] is False
    # the wrist pose arrives: the frustum moves (one frustum, on gripper_link)
    W = fx.wrist_pose(position_mm=[10.0, 50.0, -20.0], optical_axis=[0.0, -0.6, -0.8], up=[0.0, 0.8, -0.6], hfov_deg=70.0)
    stub.routes["/api/calib"] = fx.calib(camera={**fx.calib()["camera"], "wrist": W})
    page.eval("calibPoll()")
    page.wait_until("!!(window.__wristCam&&window.__wristCam.userData.pose)")
    w = wrist_info(page)
    assert all(abs(a - b) < 1e-9 for a, b in zip(w["pos"], [0.01, 0.05, -0.02])) and dot(w["axis"], [0.0, -0.6, -0.8]) > 0.99999 and w["n"] == 1 and abs(w["hfov"] - 70.0) < 1e-9
    # junk is ignored (fallback), and the pose gone again brings the guess back
    for junk in ({"position_mm": [1, 2], "rotation": {}}, {"position_mm": ["a", 0, 0], "rotation": {"optical_axis_gripper": [0, 0, -1]}}, "nope"):
        stub.routes["/api/calib"] = fx.calib(camera={**fx.calib()["camera"], "wrist": junk})
        page.eval("calibPoll()")
        page.wait_until("!!(window.__wristCam&&!window.__wristCam.userData.pose)")
        w = wrist_info(page)
        assert all(abs(a - b) < 1e-9 for a, b in zip(w["pos"], OLD_POS)) and w["n"] == 1, junk
    page.check()
