"""What each mat-map blob IS, on the page: the 3D stage colours every object by its kind with a small text label (NUT / WING NUT / SPACER / TOY?), draws the
fitted ring as an open blue cylinder and each keep-out (the toy frogs) as a translucent red-orange cylinder labelled KEEP OUT + why, shows a legend, and the TOP feed
overlay draws the same labels and keep-out circles. The stage tests reuse test_stage_calib's CDN ``stage`` fixture (skipped offline); the feed test runs on the plain
page (the overlay lives in the classic script, so it needs no three.js)."""
import os

import apidata as fx
from test_feed_overlay import show_top, mat_to_display
from test_stage_calib import stage, world  # noqa: F401  (stage: the fixture)

KIND_HEX = {"nut": 0x4FD6FF, "wing nut": 0xFFD166, "spacer": 0x7CFF6B, "toy?": 0xC9C2B2, "keep-out": 0xFF5A3C, "ring wall": 0x2B8CFF}


def extras(page):
    return page.eval("window.__extras.map(o=>({name:o.name,sprite:!!o.isSprite,text:(o.userData&&o.userData.text)||null,open:o.isGroup?o.children[0].geometry.parameters.openEnded:null,"
                     "r:o.isGroup?o.children[0].geometry.parameters.radiusTop:null,h:o.isGroup?o.children[0].geometry.parameters.height:null,hex:o.isGroup?o.children[0].material.color.getHex():null,"
                     "op:o.isGroup?o.children[0].material.opacity:null,z:o.position.z,parent:o.parent&&o.parent.name}))")


def tag_text(page, id_):
    return page.eval(f"(()=>{{const t=window.__scene.getObjectByName('objtag_{id_}');return t?t.userData.text:null}})()")


def test_stage_draws_kinds_ring_and_keepout(stub, stage):
    stub.routes["/api/calib"] = fx.calib(); stub.routes["/api/scene"] = fx.labelled_scene()
    page = stage(); assert page.eval("scenePoll()") is True
    # colour by kind and a small text label
    for id_, kind, txt in ((11, "nut", "NUT"), (12, "wing nut", "WING NUT"), (13, "spacer", "SPACER"), (14, "toy?", "TOY?")):
        m = page.eval(f"window.__scene.getObjectByName('obj_{id_}').userData.hex"); assert m == KIND_HEX[kind], (kind, hex(m)); assert tag_text(page, id_) == txt, kind
    assert page.eval("window.__scene.getObjectByName('obj_15').userData.hex") == KIND_HEX["ring wall"] and tag_text(page, 15) is None, "a piece of the ring's wall: dim blue, no label"
    assert page.eval("window.__scene.getObjectByName('obj_20').userData.hex") == KIND_HEX["keep-out"] and tag_text(page, 20) is None, "the frog's own label is on its cylinder"
    assert page.eval("window.__scene.getObjectByName('obj_10').visible") is False, "the whole-ring blob is replaced by the open cylinder"
    # the ring: an open blue cylinder, 52 mm radius and 51 mm tall, at the fitted centre
    ex = {e["name"]: e for e in extras(page)}
    r = ex["ring"]; assert r["open"] is True and abs(r["r"] - 0.052) < 1e-9 and abs(r["h"] - 0.051) < 1e-9 and r["parent"] == "mat_frame" and abs(r["z"] - (0.0255 + 0.002)) < 1e-6
    assert r["hex"] == 0x2B8CFF and r["op"] < 0.6
    w, e = world(page, "ring"), page.eval("window.__matToWorld(7,8)"); assert abs(w[0] - e[0]) < 0.002 and abs(w[2] - e[2]) < 0.002
    # the keep-out: a closed, translucent red-orange cylinder of the server's radius and height, labelled KEEP OUT and why
    k = ex["keepout_20"]; assert k["open"] is False and abs(k["r"] - 0.0608) < 1e-9 and abs(k["h"] - 0.060) < 1e-9 and k["hex"] == 0xFF5A3C and 0.15 <= k["op"] <= 0.5
    w, e = world(page, "keepout_20"), page.eval("window.__matToWorld(12.5,3)"); assert abs(w[0] - e[0]) < 0.002 and abs(w[2] - e[2]) < 0.002
    assert ex["keepout_label_20"]["text"] == "KEEP OUT" and ex["keepout_why_20"]["text"] == "COLOURFUL" and ex["keepout_label_20"]["sprite"] is True
    assert page.eval("document.getElementById('scenelegend').classList.contains('on')") is True
    leg = page.eval("document.getElementById('scenelegend').innerText.toUpperCase()"); assert all(t in leg for t in ("NUT", "WING NUT", "SPACER", "TOY?", "RING", "KEEP OUT"))
    # an unchanged poll builds nothing new; the frog leaving removes its cylinder and labels, the ring stays
    n = len(extras(page)); page.eval("scenePoll()"); assert len(extras(page)) == n
    s = fx.labelled_scene(); s["objects"] = [o for o in s["objects"] if o["id"] != 20]; s["keepouts"] = []; stub.routes["/api/scene"] = s; page.eval("scenePoll()")
    assert sorted(e["name"] for e in extras(page)) == ["ring", "ring_label"] and page.eval("!window.__scene.getObjectByName('keepout_20')")
    # no scene at all for two polls: everything goes, the legend too
    stub.routes["/api/scene"] = fx.scene_none(); page.eval("scenePoll()"); page.eval("scenePoll()")
    assert extras(page) == [] and page.eval("document.getElementById('scenelegend').classList.contains('on')") is False
    page.check()


def test_stage_without_labels_is_the_old_picture(stub, stage):
    """An older server (no kind, no ring, no keepouts): measured colour, #id tags, no cylinders, no legend."""
    stub.routes["/api/calib"] = fx.calib(); stub.routes["/api/scene"] = fx.scene([fx.scene_obj(id=7)])
    page = stage(); page.eval("scenePoll()")
    assert tag_text(page, 7) == "#7" and extras(page) == [] and page.eval("document.getElementById('scenelegend').classList.contains('on')") is False
    page.check()


def test_top_feed_overlay_draws_labels_and_keepout_circles(page, stub):
    rect = show_top(page, stub); s = fx.labelled_scene(); page.eval(f"window.sceneLast={__import__('json').dumps(s)}; calDraw(); true")
    d = page.eval("window.CALDBG.scene"); assert d, "overlay state exposed"
    ks = d["keepouts"]; assert len(ks) == 1 and ks[0][3] == "KEEP OUT · COLOURFUL"
    ex, ey = mat_to_display(rect, 12.5, 3.0); assert abs(ks[0][0] - ex) < 1.5 and abs(ks[0][1] - ey) < 1.5
    # radius = 60.8 mm in display px: the distance between the centre and the point 60.8 mm further along mat X
    px, py = mat_to_display(rect, 12.5 + 60.8 / 25.4, 3.0); assert abs(ks[0][2] - ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5) < 1.5 and ks[0][2] > 10
    assert d["ring"] and abs(d["ring"][0] - mat_to_display(rect, 7.0, 8.0)[0]) < 1.5
    assert sorted(t for _, _, t in d["labels"]) == ["NUT", "SPACER", "TOY?", "WING NUT"]
    nut = next(l for l in d["labels"] if l[2] == "NUT"); nx, ny = mat_to_display(rect, 10.9, 7.1); assert abs(nut[0] - nx) < 1.5 and abs(nut[1] - ny) < 1.5
    # the keep-out circle is really painted (a red-orange pixel on its rim), and clearing the scene clears it
    def reddish(x, y): return page.eval(f"(()=>{{const d=document.querySelector('#calov').getContext('2d').getImageData({int(x) - 3},{int(y) - 3},6,6).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]>200&&d[i+1]<140&&d[i+3]>150)n++;return n}})()")
    assert reddish(ks[0][0] + ks[0][2], ks[0][1]) + reddish(ks[0][0], ks[0][1] + ks[0][2]) + reddish(ks[0][0] - ks[0][2], ks[0][1]) > 0
    page.eval("window.sceneLast=null; calDraw(); true"); assert page.eval("window.CALDBG.scene") is None
    page.check()


def test_screenshot_hook_for_the_stage(stub, stage):
    """SO101_LABELS_SHOT=/path.png saves a screenshot of the stage with the labelled scene (SO101_LABELS_SCENE=/scene.json: a labelled payload to draw) (used for the review picture); without it the test only checks the scene drew."""
    scene = fx.labelled_scene()
    if os.environ.get("SO101_LABELS_SCENE"): scene = fx.scene(**__import__("json").load(open(os.environ["SO101_LABELS_SCENE"])))   # a labelled /api/scene payload (objects, ring, keepouts) to draw instead
    stub.routes["/api/calib"] = fx.calib(); stub.routes["/api/scene"] = scene
    page = stage(); page.eval("scenePoll()"); assert any(e["name"].startswith("keepout_") for e in extras(page))
    if os.environ.get("SO101_LABELS_SHOT"):
        page.eval("(document.getElementById('arm3d').scrollIntoView({block:'center'}),true)"); page.eval("new Promise(r=>setTimeout(()=>r(true),1500))")   # let a few frames render
        page.screenshot(os.environ["SO101_LABELS_SHOT"])
    page.check()
