"""Servo temperature on the real follower in the 3D stage: each joint's STS3215 body is tinted by its live /api/tele temperature, with a deg C label; stale
telemetry shows the normal look and a "no telemetry" note; the virtual ghost is never tinted. Needs the three.js CDN (skips offline) like test_stage_calib.
SO101_SERVO_SHOT=/path.png saves a screenshot with a hot servo."""
import os
import time

import apidata as fx
from test_stage_calib import stage  # noqa: F401  (the fixture)
from test_virtual_ghost import feed, vmsg

SHOT = os.environ.get("SO101_SERVO_SHOT")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_COLOR = 0xD9D3C6


def tele(temps, age=0.4):
    t = time.time()
    rows = [[t - 1 + k * 0.5, 1.2, 12.0, [[temp, 12, 0.35, 12.0, 0, 0, 0, 0, 1, 0, 0] for temp in temps]] for k in range(2)]
    d = fx.tele_none()
    d.update(keys=["temp", "load", "cur", "volt", "pos", "goal", "sag", "vel", "on", "moving", "status"], joints=JOINTS, rows=rows, src="bus", age=age, n=2, t=rows[-1][0], now=t)
    return d


def poll(stub, page, payload):
    stub.routes["/api/tele"] = payload
    page.eval("(window.so101Tele.rows=[],window.so101Tele.since=0,teleTick(),true)")
    page.wait_until("window.so101Tele.rows.length>0||window.so101Tele.err", timeout=10)


def servo(page, j):
    return page.eval(f"(()=>{{const s=window.__servos&&window.__servos[{j!r}];if(!s)return null;return {{color:s.mesh.material.color.getHex(),tint:s.tint||null,label:s.label.visible,text:s.label.userData.text||''}}}})()")


def test_the_temperature_function_follows_the_lab_thresholds_and_loses_lightness_as_it_heats(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    f = lambda t: page.eval(f"window.__servoTempStyle({'null' if t is None else t})")  # noqa: E731
    assert f(49.9) is None and f(None) is None, "below 50 C: the normal look"
    assert f(50)["level"] == 1 and f(59.9)["level"] == 1 and f(60)["level"] == 2 and f(64.9)["level"] == 2
    assert f(65)["level"] == 3 and f(65)["pulse"] is False and f(69.9)["pulse"] is False and f(70)["pulse"] is True, "red from 65, slow pulse at the firmware cut"
    lum = lambda h: 0.2126 * (h >> 16 & 255) + 0.7152 * (h >> 8 & 255) + 0.0722 * (h & 255)  # noqa: E731
    ls = [lum(f(t)["hex"]) for t in (50, 55, 59.9, 60, 62.5, 64.9, 65, 70)]
    assert all(a >= b for a, b in zip(ls, ls[1:])) and ls[0] > 2 * ls[-1], ls   # lightness must fall, not only hue
    assert page.eval("document.getElementById('servoscale').textContent").count("60") and all(str(v) in page.eval("document.getElementById('servoticks').textContent") for v in (60, 65, 70))
    page.check()


def test_a_hot_servo_tints_its_own_mesh_and_shows_its_label_while_cool_ones_keep_the_normal_look(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    page.wait_until("!!window.__servos", timeout=30)
    poll(stub, page, tele([55, 66, 48, 61, 52, 40]))
    page.wait_until("window.__servoState&&window.__servoState.fresh===true", timeout=10)
    page.wait_until("(window.__servos.shoulder_lift.label.userData.text||'')==='66\\u00b0C'", timeout=10)
    lift, pan, elbow = servo(page, "shoulder_lift"), servo(page, "shoulder_pan"), servo(page, "elbow_flex")
    assert lift["tint"] == 0xC1121F and lift["color"] == 0xC1121F and lift["label"] and lift["text"] == "66°C"
    assert pan["tint"] is not None and pan["color"] != DEFAULT_COLOR and pan["text"] == "55°C"
    assert elbow["tint"] is None and elbow["color"] == DEFAULT_COLOR and elbow["text"] == "48°C", "48 C is cool: normal look, label still shows the number"
    assert servo(page, "gripper")["color"] == DEFAULT_COLOR
    # the motor sits in the joint's parent link: distinct meshes
    assert page.eval("new Set(Object.values(window.__servos).map(s=>s.mesh.uuid)).size") == 6
    assert page.eval("document.getElementById('servonote').textContent") == ""
    # load and current appear when the SERVOS panel is open
    page.click("#telebtn")
    page.wait_until("(window.__servos.shoulder_lift.label.userData.text||'').includes('A')", timeout=10)
    assert "%" in servo(page, "shoulder_lift")["text"]
    if SHOT:   # keep a fresh feed flowing while the slow software renderer catches up (stale telemetry rightly shows untinted)
        stub.routes["/api/tele"] = lambda path: tele([55, 66, 48, 61, 52, 40])
        page.eval("(setInterval(teleTick,400),true)")
        page.click("#telebtn")
        page.eval("(()=>{window.__ctl.autoRotate=false;window.__cam.position.set(0.7,0.5,0.9);window.__ctl.target.set(0.2,0.1,0.0);window.__ctl.update();document.querySelector('#arm3d').scrollIntoView({block:'center'});return true})()")
        page.eval("new Promise(r=>{let n=0;const f=()=>++n>4?r(true):requestAnimationFrame(f);f()})")
        page.wait_until("window.__servoState.fresh===true&&window.__servos.shoulder_lift.tint===0xC1121F", timeout=10)
        page.screenshot(SHOT)
    page.check()


def test_stale_telemetry_shows_the_normal_look_and_a_note_never_old_colours(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    page.wait_until("!!window.__servos", timeout=30)
    poll(stub, page, tele([66, 66, 66, 66, 66, 66]))
    page.wait_until("window.__servos.shoulder_lift.tint===0xC1121F", timeout=10)
    poll(stub, page, tele([66, 66, 66, 66, 66, 66], age=9.0))   # the Jetson stopped sending 9 s ago
    page.wait_until("window.__servoState.fresh===false", timeout=10)
    for j in JOINTS:
        s = servo(page, j)
        assert s["tint"] is None and s["color"] == DEFAULT_COLOR and s["label"] is False, j
    assert page.eval("document.getElementById('servonote').textContent") == "no telemetry"
    stub.routes["/api/tele"] = fx.tele_none()   # and no rows at all
    page.eval("(window.so101Tele.rows=[],teleTick(),true)")
    page.wait_until("window.__servoState.fresh===false", timeout=10)
    page.check()


def test_the_ghost_arm_is_never_tinted(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    page.wait_until("!!window.__servos&&!!window.__ghost", timeout=60)
    poll(stub, page, tele([72, 72, 72, 72, 72, 72]))
    feed(stub, page, [vmsg("FAKE", "demo")])
    page.wait_until("window.__ghost.visible===true&&window.__servos.shoulder_lift.tint===0xC1121F", timeout=10)
    cols = page.eval("(()=>{const c=new Set();window.__ghost.traverse(o=>{if(o.isMesh)c.add(o.material.color.getHex())});return [...c]})()")
    assert cols == [0xB26BFF], "every ghost mesh keeps the ghost colour while the real servos are red"
    assert page.eval("(()=>{let n=0;window.__ghost.traverse(o=>{if(o.isLineSegments&&o.material.color.getHex()!==0xb26bff)n++});return n})()") == 0
    assert stub.posts == []
    page.check()


def overlaps(a, b, gap=0):
    return a["x"] < b["x"] + b["w"] + gap and b["x"] < a["x"] + a["w"] + gap and a["y"] < b["y"] + b["h"] + gap and b["y"] < a["y"] + a["h"] + gap


def test_deoverlap_pushes_colliding_rects_apart_and_leaves_clear_ones(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    rects = [{"x": 100, "y": 100, "w": 60, "h": 24}, {"x": 110, "y": 105, "w": 60, "h": 24}, {"x": 120, "y": 108, "w": 60, "h": 24}, {"x": 400, "y": 300, "w": 60, "h": 24}]
    dy = page.eval(f"window.__deoverlap({rects!r})")
    moved = [{**r, "y": r["y"] + d} for r, d in zip(rects, dy)]
    assert dy[3] == 0, "a label that collides with nothing stays put"
    assert not any(overlaps(a, b) for i, a in enumerate(moved) for b in moved[i + 1:]), moved
    top = [{"x": 10, "y": 2, "w": 60, "h": 24}, {"x": 12, "y": 4, "w": 60, "h": 24}]   # up would leave the stage: the second goes down
    d2 = page.eval(f"window.__deoverlap({top!r})")
    assert min(r["y"] + d for r, d in zip(top, d2)) >= 0
    page.check()


def test_servo_labels_on_the_stage_do_not_overlap_in_screen_space(stub, stage):
    stub.routes["/api/calib"] = fx.calib()
    page = stage()
    page.wait_until("!!window.__servos", timeout=30)
    poll(stub, page, tele([52, 66, 48, 61, 40, 55]))
    page.wait_until("(window.__servoRects||[]).length===6", timeout=10)
    page.eval("(window.__ctl.autoRotate=false,window.__cam.position.set(0.7,0.5,0.9),window.__ctl.target.set(0.2,0.1,0),window.__ctl.update(),true)")   # the view of the collision screenshot
    page.wait_until("(window.__servoRects||[]).length===6", timeout=10)
    rects = page.eval("window.__servoRects")
    assert not any(overlaps(a, b) for i, a in enumerate(rects) for b in rects[i + 1:]), rects
    assert all(s["label"] and s["text"].endswith("°C") for s in (servo(page, j) for j in JOINTS)), "the number stays visible"
    page.check()
