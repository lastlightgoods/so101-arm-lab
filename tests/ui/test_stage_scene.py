"""The mat map: objects from /api/scene drawn on the 3D stage in the mat frame (obj_<id> footprints in their own colour),
fading out when they go, captions when the map cannot be trusted, the CAL chip's live / holding / check-mat parts, and the
SET BACKGROUND control on the TOP feed. The stage tests reuse test_stage_calib's ``stage`` fixture (three.js from its CDN,
skipped offline); the button and chip tests run on the plain page without any network."""
import math

import apidata as fx
from test_feed_overlay import show_top
from test_stage_calib import stage, world  # noqa: F401  (stage: the fixture; importing it registers it here)

NUT = dict(id=7, x_in=10.94, y_in=6.94, w_mm=12.0, h_mm=10.0, angle_deg=-90.0, color=[121, 126, 129])


def mesh_info(page, name):
    return page.eval(f"(()=>{{const o=window.__scene.getObjectByName({name!r});if(!o)return null;return {{mesh:!!o.isMesh,parent:o.parent&&o.parent.name,"
                     f"sx:o.scale.x,sy:o.scale.y,sz:o.scale.z,rz:o.rotation.z,hex:o.material.color.getHex(),want:o.userData.hex,op:o.material.opacity,z:o.position.z,"
                     f"id:o.userData.obj&&o.userData.obj.id}}}})()")


def channels(hex_):
    return [(hex_ >> 16) & 255, (hex_ >> 8) & 255, hex_ & 255]


def n_objects(page):
    return page.eval("window.__scene.getObjectByName('mat_frame').children.filter(o=>/^obj_/.test(o.name)).length")


def test_object_is_drawn_in_the_mat_frame_at_its_position_in_its_colour(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(**NUT)])
    page = stage()
    assert page.eval("scenePoll()") is True
    o = mesh_info(page, "obj_7")
    assert o and o["mesh"] and o["parent"] == "mat_frame" and o["id"] == 7
    assert abs(o["sx"] - 0.012) < 1e-9 and abs(o["sy"] - 0.010) < 1e-9 and o["sz"] == 1, "footprint w_mm x h_mm in metres"
    assert abs(o["rz"] + math.pi / 2) < 1e-9, "angle_deg about the mat normal (local z)"
    assert o["want"] == (121 << 16) | (126 << 8) | 129 and all(abs(a - b) <= 1 for a, b in zip(channels(o["hex"]), [121, 126, 129])), "mean RGB of the blob"
    assert 0.9 <= o["op"] <= 1 and 0.004 <= o["z"] <= 0.008, "sits on the mat surface"
    # world position = the mat point (via the stage's own mat -> world), within 2 mm in the plane, a little above the mat
    w, e = world(page, "obj_7"), page.eval("window.__matToWorld(10.94,6.94)")
    assert abs(w[0] - e[0]) < 0.002 and abs(w[2] - e[2]) < 0.002 and 0 < w[1] - e[1] < 0.012, (w, e)
    tag = page.eval("(()=>{const t=window.__scene.getObjectByName('objtag_7');return t&&t.isSprite&&t.parent.name==='mat_frame'})()")
    assert tag is True, "a small id label beside it"
    # the next poll moves the same mesh instead of making another
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(**{**NUT, "x_in": 4.0, "y_in": 3.0, "angle_deg": 30.0})])
    page.eval("scenePoll()")
    w2, e2 = world(page, "obj_7"), page.eval("window.__matToWorld(4,3)")
    assert abs(w2[0] - e2[0]) < 0.002 and abs(w2[2] - e2[2]) < 0.002 and abs(mesh_info(page, "obj_7")["rz"] - math.pi / 6) < 1e-9
    assert n_objects(page) == 1 and page.eval("window.__objs.size") == 1
    # two objects, one of them malformed: the good one is drawn, the bad one skipped without an exception
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(**NUT), fx.scene_obj(id=8, x_in=2, y_in=2, color=[255, 0, 0]), {"id": 9, "x_mm": None}])
    page.eval("scenePoll()")
    assert n_objects(page) == 2 and channels(mesh_info(page, "obj_8")["hex"])[0] == 255 and mesh_info(page, "obj_9") is None
    page.check()


def test_object_fades_after_one_empty_poll_and_vanishes_after_two(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(id=3)])
    page = stage()
    page.eval("scenePoll()")
    assert mesh_info(page, "obj_3")["op"] > 0.9
    stub.routes["/api/scene"] = fx.scene([])
    page.eval("scenePoll()")
    o = mesh_info(page, "obj_3"); assert o and o["op"] < 0.5, "one miss: still there, faded"
    page.eval("scenePoll()")
    assert mesh_info(page, "obj_3") is None and page.eval("!window.__scene.getObjectByName('objtag_3')") and page.eval("window.__objs.size") == 0
    # it comes back as a fresh, fully opaque mesh
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(id=3)])
    page.eval("scenePoll()")
    assert mesh_info(page, "obj_3")["op"] > 0.9
    # the scene going away (camera off, route broken) counts as absent too
    stub.routes["/api/scene"] = fx.scene_none()
    page.eval("scenePoll()"); page.eval("scenePoll()")
    assert mesh_info(page, "obj_3") is None and page.eval("window.sceneLast") is None
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(id=3)]); page.eval("scenePoll()")
    stub.fail["/api/scene"] = 500
    page.eval("scenePoll()"); page.eval("scenePoll()")
    assert mesh_info(page, "obj_3") is None and n_objects(page) == 0
    page.check()


def test_captions_say_why_the_map_is_empty(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    stub.routes["/api/scene"] = fx.scene([], background="missing")
    page = stage()
    page.eval("scenePoll()")
    assert page.text("#matcap") == "MAT MAP: NO BACKGROUND — tap SET BACKGROUND"
    assert page.eval("getComputedStyle(document.querySelector('#matcap')).display") != "none"
    assert page.eval("document.querySelector('#matcap').parentElement.id") == "arm3d"
    note = "mat moved 43 mm / 178 deg since calibration (markers re-glued?): re-run cam_calib.py table"
    stub.routes["/api/scene"] = fx.scene([], calib_ok=False, calib_note=note)
    page.eval("scenePoll()")
    assert page.text("#matcap") == "MAT MAP: " + note
    stub.routes["/api/scene"] = fx.scene([], holding=True, n_markers_seen=2)
    page.eval("scenePoll()")
    assert "HOLDING" in page.text("#matcap") and "2 markers" in page.text("#matcap")
    stub.routes["/api/scene"] = fx.scene([fx.scene_obj(id=1)])
    page.eval("scenePoll()")
    assert page.text("#matcap") == "" and page.eval("getComputedStyle(document.querySelector('#matcap')).display") == "none"
    stub.routes["/api/scene"] = fx.scene_none()
    page.eval("scenePoll()")
    assert page.text("#matcap") == ""
    page.check()


def test_a_live_camera_solve_refreshes_the_calibration_at_once(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    stub.routes["/api/scene"] = fx.scene([])
    page.eval("scenePoll()")
    seq, hits = page.eval("window.calibSeq"), stub.hits["/api/calib"]
    stub.routes["/api/scene"] = fx.scene([], calib_live=True, calib_live_at=1234.5, shift_px=10.6)
    page.eval("scenePoll()")
    page.wait_until(f"window.calibSeq>{seq}")
    assert stub.hits["/api/calib"] == hits + 1
    page.eval("scenePoll()")
    assert page.eval("window.calibSeq") == seq + 1 and stub.hits["/api/calib"] == hits + 1, "same solve: no second refresh"
    stub.routes["/api/scene"] = fx.scene([])                     # written back: live goes away, the stored solve is the new one
    page.eval("scenePoll()")
    page.wait_until(f"window.calibSeq>{seq + 1}")
    page.check()


def test_cal_chip_shows_live_holding_and_check_mat(page, stub):
    show_top(page, stub)
    assert page.text("#calchip") == "CAL 0.1 mm · ARM FIT 1.1 mm · ROLL 0.5°"
    stub.routes["/api/calib"] = fx.calib(live=True, live_at=1234.5, shift_px=10.6)
    page.eval("calibPoll()")
    assert page.text("#calchip") == "CAL 0.1 mm · ARM FIT 1.1 mm · ROLL 0.5° · LIVE 11 px"
    assert "re-solved live" in page.eval("document.querySelector('#calchip').title")
    stub.routes["/api/calib"] = fx.calib(holding=True, markers_seen=2)
    page.eval("calibPoll()")
    assert page.text("#calchip").endswith("· HOLDING")
    note = "mat moved 43 mm / 178 deg since calibration (markers re-glued?): re-run cam_calib.py table"
    stub.routes["/api/calib"] = fx.calib(calib_ok=False, calib_note=note)
    page.eval("calibPoll()")
    assert page.text("#calchip").endswith("· CHECK MAT") and note in page.eval("document.querySelector('#calchip').title")
    stub.routes["/api/calib"] = fx.calib(live_solve={"at": "2026-11-09 22:10:00", "shift_px": 10.6, "markers_seen": 4})
    page.eval("calibPoll()")
    assert page.text("#calchip") == "CAL 0.1 mm · ARM FIT 1.1 mm · ROLL 0.5°" and "re-solved from the markers 2026-11-09 22:10:00" in page.eval("document.querySelector('#calchip').title")
    page.check()


def test_top_cam_frustum_sits_at_the_estimated_camera_pose(stub, stage):
    """camera.pose (camera_pose_est passed through /api/calib): the TOP CAM frustum moves into the mat frame at position_mm, looks
    along optical_axis, is sized by hfov_deg at 16:9, says "est." while the source is nominal, and is not a mat-map object."""
    stub.routes["/api/calib"] = fx.calib(pose=fx.camera_pose(position_mm=[218, 162, 408], optical_axis=[0, 0, -1], hfov_deg=70.4))
    page = stage()
    assert page.eval("window.__calibPose.cam.position_mm") == [218, 162, 408]
    w, e = world(page, "top_cam"), page.eval("window.__matToWorld(218/25.4,162/25.4)")
    assert abs(w[0] - e[0]) < 0.003 and abs(w[2] - e[2]) < 0.003 and abs(w[1] - (e[1] + 0.408)) < 0.003, (w, e)
    assert page.eval("window.__scene.getObjectByName('top_cam').parent.name") == "mat_frame"
    top = page.eval("[window.__topPos.x,window.__topPos.y,window.__topPos.z]")
    assert all(abs(a - b) < 1e-6 for a, b in zip(top, w)), "ray and label follow the frustum"
    d = page.eval("(()=>{const o=window.__scene.getObjectByName('top_cam');o.updateWorldMatrix(true,false);const v=new window.__THREE.Vector3(0,0,-1).transformDirection(o.matrixWorld);return [v.x,v.y,v.z]})()")
    assert abs(d[0]) < 1e-6 and abs(d[1] + 1) < 1e-6 and abs(d[2]) < 1e-6, "optical axis [0,0,-1] in the table frame = straight down on the stage"
    g = page.eval("(()=>{const o=window.__scene.getObjectByName('top_cam'),p=o.geometry.attributes.position.array;let mx=0,my=0,mz=0;for(let i=0;i<p.length;i+=3){mx=Math.max(mx,Math.abs(p[i]));my=Math.max(my,Math.abs(p[i+1]));mz=Math.max(mz,Math.abs(p[i+2]))}return [mx,my,mz]})()")
    assert abs(g[0] / g[2] - math.tan(math.radians(70.4 / 2))) < 1e-6 and abs(g[1] / g[0] - 9 / 16) < 1e-6, "half-width = len * tan(hfov/2), 16:9"
    assert page.eval("window.__topLabelText") == "TOP CAM est."
    assert n_objects(page) == 0 and page.eval("window.__objs.size") == 0
    # a measured pose drops the "est."; no pose at all brings the old guess back (over the mat centre, 0.5 m up, outside the mat group)
    stub.routes["/api/calib"] = fx.calib(pose=fx.camera_pose(source="solvePnP from the 4 markers, lens height tape-measured"))
    page.eval("calibPoll()")
    page.wait_until("window.__topLabelText==='TOP CAM'")
    assert page.eval("window.__scene.getObjectByName('top_cam').parent.name") == "mat_frame"
    stub.routes["/api/calib"] = fx.calib()
    page.eval("calibPoll()")
    page.wait_until("!window.__calibPose.cam")
    top = page.eval("[window.__topPos.x,window.__topPos.y,window.__topPos.z]")
    c = page.eval("window.__matToWorld(9,6)")
    assert abs(top[0] - c[0]) < 1e-6 and abs(top[2] - c[2]) < 1e-6 and top[1] == 0.5
    assert page.eval("window.__scene.getObjectByName('top_cam').parent===window.__scene") and page.eval("window.__scene.children.filter(o=>o.name==='top_cam').length") == 1
    page.check()


def test_set_background_button_posts_and_confirms(page, stub):
    show_top(page, stub)
    r = page.eval("(()=>{const b=document.querySelector('#calbg'),r=b.getBoundingClientRect(),f=document.querySelector('#cam_top').parentElement.getBoundingClientRect();"
                  "return {w:r.width,h:r.height,inFeed:r.left>=f.left&&r.right<=f.right&&r.top>=f.top,chip:document.querySelector('#calchip').getBoundingClientRect().right<=r.left}})()")
    assert r["h"] >= 40 and r["w"] >= 100 and r["inFeed"] and r["chip"], f"touch sized, on the TOP feed, clear of the CAL chip: {r}"
    assert page.text("#calbg") == "SET BACKGROUND"
    page.click("#calbg")
    path, body = stub.wait_post(lambda p, b: p == "/api/scene/background")
    assert body == {"frames": 3}
    page.wait_until("document.querySelector('#calbg').textContent==='SAVED'")
    assert "ok" in page.cls("#calbg") and page.eval("document.querySelector('#calbg').disabled") is True
    assert "on" not in page.cls("#calread"), "the press was not taken as a tap-to-measure on the feed"
    page.wait_until("document.querySelector('#calbg').textContent==='SET BACKGROUND'&&!document.querySelector('#calbg').disabled", timeout=5)
    assert page.cls("#calbg") == "calbtn"
    # the server says no (or is down): FAILED, then back
    stub.fail["/api/scene/background"] = 500
    page.click("#calbg")
    page.wait_until("document.querySelector('#calbg').textContent==='FAILED'")
    assert "bad" in page.cls("#calbg") and len(stub.posts_to("/api/scene/background")) == 2
    page.wait_until("document.querySelector('#calbg').textContent==='SET BACKGROUND'", timeout=5)
    page.check()
