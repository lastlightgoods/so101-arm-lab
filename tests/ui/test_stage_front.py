"""The FRONT CAM frustum on the 3D stage follows camera.front.pose from /api/calib: placed in the mat group at position_mm, looking along
optical_axis, sized by hfov_deg, with a dashed ray to axis_hits_mm; without a pose it keeps the old static guess. Reuses
test_stage_calib's ``stage`` fixture (three.js from its CDN, software WebGL), so it needs the network and skips cleanly without it."""
import math

import apidata as fx
from test_stage_calib import stage, world  # noqa: F401  (stage: the fixture; importing it registers it here)

IN = 0.0254


def axis_dir(page, name):
    """World direction of a frustum's view axis (it looks down its local -z)."""
    return page.eval(f"(()=>{{const o=window.__scene.getObjectByName({name!r});o.updateWorldMatrix(true,false);"
                     f"const v=new window.__THREE.Vector3(0,0,-1).transformDirection(o.matrixWorld);return [v.x,v.y,v.z]}})()")


def expected_dir(page, axis):
    """A table-frame unit vector on the stage: the mat axes through __matToWorld (x, y in inches) plus z straight up."""
    o = page.eval("window.__matToWorld(0,0)")
    p = page.eval(f"window.__matToWorld({axis[0] / IN},{axis[1] / IN})")   # 1 m along the table-frame (x, y) components
    d = [p[0] - o[0], p[1] - o[1] + axis[2], p[2] - o[2]]
    n = math.sqrt(sum(c * c for c in d))
    return [c / n for c in d]


def test_front_cam_sits_at_the_measured_pose(stub, stage):
    stub.routes["/api/calib"] = fx.calib(front=fx.front())
    page = stage()
    assert page.eval("window.__calibPose.front.position_mm") == fx.FRONT_C
    w = world(page, "front_cam")
    e = page.eval(f"window.__matToWorld({fx.FRONT_C[0] / 25.4},{fx.FRONT_C[1] / 25.4})")
    assert abs(w[0] - e[0]) < 0.003 and abs(w[2] - e[2]) < 0.003 and abs(w[1] - (e[1] + fx.FRONT_C[2] / 1000)) < 0.003, (w, e)
    assert page.eval("window.__scene.getObjectByName('front_cam').parent.name") == "mat_frame"
    assert page.eval("window.__frontCam===window.__scene.getObjectByName('front_cam')")
    fp = page.eval("[window.__frontPos.x,window.__frontPos.y,window.__frontPos.z]")
    assert all(abs(a - b) < 1e-6 for a, b in zip(fp, w)), "label and ray follow the frustum"
    d, ed = axis_dir(page, "front_cam"), expected_dir(page, fx.FRONT_AXIS)
    assert all(abs(a - b) < 1e-4 for a, b in zip(d, ed)), (d, ed)
    assert d[1] < -0.7, "looking down at the mat at ~42 deg"
    # orange, sized by hfov_deg at 16:9
    g = page.eval("(()=>{const o=window.__scene.getObjectByName('front_cam'),p=o.geometry.attributes.position.array;let mx=0,my=0,mz=0;for(let i=0;i<p.length;i+=3){mx=Math.max(mx,Math.abs(p[i]));my=Math.max(my,Math.abs(p[i+1]));mz=Math.max(mz,Math.abs(p[i+2]))}return [mx,my,mz,o.material.color.getHex()]})()")
    assert abs(g[0] / g[2] - math.tan(math.radians(68.88 / 2))) < 1e-6 and abs(g[1] / g[0] - 9 / 16) < 1e-6 and g[3] == 0xffb347
    # the dashed ray runs from the camera to where the axis meets the mat
    ray = page.eval("(()=>{const r=window.__scene.getObjectByName('front_ray');if(!r||!r.visible)return null;const p=r.geometry.attributes.position.array;return [[p[0],p[1],p[2]],[p[3],p[4],p[5]]]})()")
    assert ray is not None
    hit = page.eval(f"window.__matToWorld({fx.FRONT_AIM[0] / 25.4},{fx.FRONT_AIM[1] / 25.4})")
    assert all(abs(a - b) < 1e-6 for a, b in zip(ray[0], w)) and all(abs(a - b) < 0.002 for a, b in zip(ray[1], hit)), (ray, hit)
    # the TOP cam is still at its own old guess (no camera.pose in this fixture)
    top = page.eval("[window.__topPos.x,window.__topPos.y,window.__topPos.z]")
    c = page.eval("window.__matToWorld(9,6)")
    assert abs(top[0] - c[0]) < 1e-6 and abs(top[2] - c[2]) < 1e-6 and top[1] == 0.5
    page.check()


def test_front_cam_keeps_the_static_placement_without_a_pose_and_moves_when_one_arrives(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    assert page.eval("window.__calibPose.front") is None
    w = world(page, "front_cam")
    assert all(abs(a - b) < 1e-6 for a, b in zip(w, [0.62, 0.07, 0.40])), w
    assert page.eval("window.__scene.getObjectByName('front_cam').parent===window.__scene")
    assert page.eval("!window.__scene.getObjectByName('front_ray').visible")
    assert page.eval("window.__scene.children.filter(o=>o.name==='front_cam').length") == 1
    # the pose arrives on a later poll: the calib event rebuilds the mat and the frustum moves into the mat group
    stub.routes["/api/calib"] = fx.calib(front=fx.front())
    page.eval("calibPoll()")
    page.wait_until("!!window.__calibPose.front")
    assert page.eval("window.__scene.getObjectByName('front_cam').parent.name") == "mat_frame"
    w2 = world(page, "front_cam")
    e = page.eval(f"window.__matToWorld({fx.FRONT_C[0] / 25.4},{fx.FRONT_C[1] / 25.4})")
    assert abs(w2[1] - (e[1] + 0.467)) < 0.003 and page.eval("window.__scene.getObjectByName('front_ray').visible")
    assert page.eval("window.__scene.getObjectByName('top_cam').parent===window.__scene"), "the top cam was not touched"
    # ...and goes back when the pose is gone again; only ever one front_cam in the scene
    stub.routes["/api/calib"] = fx.calib()
    page.eval("calibPoll()")
    page.wait_until("!window.__calibPose.front")
    assert page.eval("window.__scene.getObjectByName('front_cam').parent===window.__scene") and page.eval("!window.__scene.getObjectByName('front_ray').visible")
    assert page.eval("(()=>{let n=0;window.__scene.traverse(o=>{if(o.name==='front_cam')n++});return n})()") == 1
    page.check()
