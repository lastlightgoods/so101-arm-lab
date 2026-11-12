"""The virtual ghost arm on the 3D stage (docs/virtual-stage.md): a SIM / FAKE stream from /api/state["virtual"] draws a violet wire arm under a fixed
badge and a floating label, the real arm still draws from arms.follower, both use one joint conversion, and nothing is ever POSTed. Needs the three.js
CDN like test_stage_calib (skips offline); SO101_STAGE_SHOT=/path.png saves a screenshot of the stage with the ghost."""
import math
import os
import time

import apidata as fx
from test_stage_calib import FOLLOWER, arms_state, stage, wait_joint  # noqa: F401  (stage: the fixture)

SHOT = os.environ.get("SO101_GHOST_SHOT")
GHOST_JOINTS = {"shoulder_pan": -25.0, "shoulder_lift": 35.0, "elbow_flex": 40.0, "wrist_flex": -10.0, "wrist_roll": 20.0, "gripper": 60.0}
OFFS = fx.CALIB_ARM["joint_offsets_deg"]


def vmsg(source="FAKE", session="demo", joints=None, seq=1, frame=None, objects=None):
    return {"source": source, "session": session, "t": time.time(), "seq": seq, "joints": dict(joints or GHOST_JOINTS), "joint_frame": frame or ("urdf" if source == "SIM" else "lerobot"),
            "objects": objects if objects is not None else [{"kind": "nut", "frame": "mat_in", "x": 6.0, "y": 8.0, "z": 0.0}, {"kind": "target", "frame": "mat_in", "x": 9.0, "y": 6.0, "z": 0.0}], "age_s": 0.1}


def feed(stub, page, virtual, follower=None):
    s = arms_state(follower)
    s["virtual"] = virtual; stub.routes["/api/state"] = s
    page.eval("arms()")


def ghost_joint(page, name):
    return page.eval(f"(()=>{{const r=window.__ghost;if(!r)return null;const j=r.joints[{name!r}];return j.angle!==undefined?j.angle:j.jointValue[0]}})()")


def wait_ghost(page, name, expected, tol=1e-5):
    page.wait_until(f"(()=>{{const r=window.__ghost;if(!r)return false;const j=r.joints[{name!r}];const a=j.angle!==undefined?j.angle:j.jointValue[0];return Math.abs(a-({expected!r}))<{tol}}})()", timeout=20)


def test_ghost_badge_label_and_objects_render_and_the_real_arm_is_still_drawn(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    page.wait_until("!!window.__ghost", timeout=60)
    assert page.eval("window.__ghost.visible") is False and page.eval("document.getElementById('virtbadge').style.display") == "none", "nothing virtual: nothing drawn"
    feed(stub, page, [vmsg("FAKE", "demo")], follower=FOLLOWER)
    page.wait_until("window.__ghost.visible===true", timeout=10)
    assert page.eval("document.getElementById('virtbadge').textContent").startswith("FAKE") and "NOT THE REAL ARM" in page.eval("document.getElementById('virtbadge').textContent")
    page.wait_until("window.__virtual.label==='FAKE · demo'", timeout=10)
    assert page.eval("window.__virtual.live") == 1
    assert page.eval("document.getElementById('arm3d').style.boxShadow") != "", "stage frame tint while live"
    assert page.eval("(()=>{const n=window.__scene.getObjectByName('ghost_nut'),t=window.__scene.getObjectByName('ghost_target');return n.visible&&t.visible&&t.material.isLineDashedMaterial})()") is True
    w = page.eval("(()=>{const n=window.__scene.getObjectByName('ghost_nut');return [n.position.x,n.position.z]})()")
    e = page.eval("window.__matToWorld(6,8)")
    assert abs(w[0] - e[0]) < 1e-6 and abs(w[1] - e[2]) < 1e-6, "the ghost nut sits at its mat position"
    # the real arm is still the orange follower driven by arms.follower, distinct from the ghost
    assert page.eval("window.__robots.follower.visible") is True and page.eval("window.__ghost!==window.__robots.follower")
    for j, deg in FOLLOWER.items():
        if j != "gripper": wait_joint(page, "follower", j, (deg + OFFS.get(j, 0.0)) * math.pi / 180)
    assert "REAL ARM ALSO LIVE" in page.eval("document.getElementById('virtbadge').textContent")
    if SHOT:   # a live stream for the picture: the page drops a stream that stops moving, so keep one flowing while the slow software renderer catches up
        n = [10]
        def live(path):
            n[0] += 1; st = arms_state(FOLLOWER); st["virtual"] = [vmsg("FAKE", "autoloop-demo", seq=n[0])]; return st
        stub.routes["/api/state"] = live
        page.eval("(setInterval(()=>arms(),150),true)")
        for j, deg in GHOST_JOINTS.items():
            if j != "gripper": wait_ghost(page, j, (deg + OFFS.get(j, 0.0)) * math.pi / 180)
        page.eval("(()=>{window.__ctl.autoRotate=false;window.__cam.position.set(0.7,0.5,0.9);window.__ctl.target.set(0.25,0.0,0.05);window.__ctl.update();return true})()")
        page.eval("(document.querySelector('#arm3d').scrollIntoView({block:'center'}),true)")
        page.eval("new Promise(r=>{let n=0;const f=()=>++n>4?r(true):requestAnimationFrame(f);f()})")
        page.screenshot(SHOT)
    # an unlabelled or REAL stream is shown as UNVERIFIED VIRTUAL, never blank
    feed(stub, page, [{**vmsg("FAKE", "x", seq=2), "source": "UNVERIFIED"}])
    page.wait_until("(window.__virtual.label||'').startsWith('UNVERIFIED VIRTUAL')", timeout=10)
    feed(stub, page, [{**vmsg(), "source": None, "seq": 3}])
    page.wait_until("document.getElementById('virtbadge').dataset.source==='UNVERIFIED VIRTUAL'&&window.__virtual.label==='UNVERIFIED VIRTUAL · demo'", timeout=10)
    # the stream going away clears everything
    feed(stub, page, [])
    page.wait_until("window.__ghost.visible===false", timeout=10)
    assert page.eval("document.getElementById('virtbadge').style.display") == "none" and page.eval("document.getElementById('arm3d').style.boxShadow") == ""
    assert stub.posts == [], "the virtual path never POSTs"
    page.check()


def test_the_ghost_and_the_real_arm_share_one_joint_conversion(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    page.wait_until("!!window.__ghost", timeout=60)
    same = {k: v for k, v in GHOST_JOINTS.items()}
    # a FAKE (lerobot-frame) ghost fed the follower's degrees ends on the follower's radians: same function, same offsets
    feed(stub, page, [vmsg("FAKE", joints=same)], follower=same)
    for j, deg in same.items():
        expected = page.eval(f"window.__jointTarget({j!r},{same!r},window.calibOffsets)")
        wait_ghost(page, j, expected)
        assert abs(wait_joint(page, "follower", j, expected) - expected) < 1e-5
        if j != "gripper": assert abs(expected - (deg + OFFS.get(j, 0.0)) * math.pi / 180) < 1e-9
    # a SIM (urdf-frame) ghost gets no offsets
    feed(stub, page, [vmsg("SIM", "sim-1", joints=same, seq=5)])
    for j in ("shoulder_lift", "elbow_flex", "wrist_flex"): wait_ghost(page, j, page.eval(f"window.__jointTarget({j!r},{same!r},{{}})"))
    assert stub.posts == []
    page.check()
