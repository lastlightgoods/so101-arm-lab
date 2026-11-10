"""FRONT-feed calibration overlay: the same mat grid / markers / arm base as the top feed, projected through the front camera's
pose-derived homography (camera.front.homography_px_to_table_mm), tap-to-measure on the mat surface confirmed via
/api/locate?cam=front, and a FRONT chip with the pose's reprojection error.

The front view is oblique (42 deg tilt), so nothing here assumes an orientation: every expectation is the fixture homography
(tests/ui/apidata.py, FRONT_*) applied in Python to the displayed, letterboxed #cam_front. Mirrors test_feed_overlay.py; the
TOP overlay's ids and behaviour are checked to be untouched alongside."""
import re
import time

import apidata as fx
from test_feed_overlay import IMG_W, IMG_H, TOL_IN, near, opaque_pixels_near, show_top


def show_front(page, stub, cal=None):
    """Bring the TOP and FRONT feeds up and load a calibration that carries camera.front (unless told otherwise)."""
    stub.routes["/api/cams"] = ["top", "front"]
    page.eval("cams()")
    stub.routes["/api/calib"] = fx.calib(front=fx.front()) if cal is None else cal
    page.eval("calibPoll()")
    page.wait_until("(()=>{const r=document.querySelector('#calov_front').getBoundingClientRect();return r.width>100&&r.height>100})()")
    page.eval("calOverlays.front.draw(); calDraw(); true")
    return page.eval("(()=>{const r=document.querySelector('#calov_front').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")


def to_display(rect, u, v):
    """Raw front-camera pixel -> css px inside the feed box: the contain letterbox only (the front feed is never rotated)."""
    _, _, w, h = rect
    s = min(w / IMG_W, h / IMG_H)
    dw, dh = IMG_W * s, IMG_H * s
    return (w - dw) / 2 + u * s, (h - dh) / 2 + v * s


def mat_to_pixel(x_in, y_in):
    return fx.h_apply(fx.FRONT_H_T2P, x_in * 25.4, y_in * 25.4)


def mat_to_display(rect, x_in, y_in):
    return to_display(rect, *mat_to_pixel(x_in, y_in))


def tol_at(rect, x, y):
    """Inches of mat covered by one display pixel (diagonal) at this spot of the oblique view, plus the top feed's allowance: synthetic
    clicks land on whole pixels, and far from the front camera a pixel is several mm of mat."""
    _, _, w, h = rect
    s = min(w / IMG_W, h / IMG_H)
    ox, oy = (w - IMG_W * s) / 2, (h - IMG_H * s) / 2
    p2t = fx._front_p2t()
    a = fx.h_apply(p2t, (x - ox) / s, (y - oy) / s)
    b = fx.h_apply(p2t, (x + 1 - ox) / s, (y + 1 - oy) / s)
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / 25.4 + TOL_IN


def click_front(page, rect, x, y):
    page.eval(f"(document.querySelector('#cam_front').dispatchEvent(new MouseEvent('click',{{bubbles:true,clientX:{rect[0] + x},clientY:{rect[1] + y}}})), true)")


def opaque_front(page, x, y, r=8):
    return page.eval(f"(()=>{{const c=document.querySelector('#calov_front'),d=c.getContext('2d').getImageData({x - r},{y - r},{2 * r},{2 * r}).data;"
                     f"let n=0;for(let i=3;i<d.length;i+=4)if(d[i]>200)n++;return n}})()")


def readout(page, sel="#calread_front"):
    txt = page.text(sel)
    return txt, {k: (float(a), float(b)) for k, a, b in re.findall(r"(MAT|ARM) (-?\d+\.\d\d), (-?\d+\.\d\d) in", txt)}


BLANK_FRONT = "(()=>{const c=document.querySelector('#calov_front'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;for(let i=3;i<d.length;i+=4)if(d[i])return false;return true})()"


def test_front_overlay_elements_and_chip(page, stub):
    show_front(page, stub)
    # its own canvas / chip / readout inside #cam_front's feed box, tagged data-cam=front; no SET BACKGROUND button there
    r = page.eval("(()=>{const f=document.querySelector('#cam_front').parentElement;return {cv:!!f.querySelector('#calov_front'),chip:!!f.querySelector('#calchip_front'),"
                  "read:!!f.querySelector('#calread_front'),btn:f.querySelectorAll('.calbtn').length,cam:document.querySelector('#calov_front').dataset.cam,"
                  "topBtn:document.querySelector('#cam_top').parentElement.querySelectorAll('#calbg').length}})()")
    assert r == {"cv": True, "chip": True, "read": True, "btn": 0, "cam": "front", "topBtn": 1}
    assert page.text("#calchip_front") == "FRONT · 1.9 px" and "off" not in page.cls("#calchip_front")
    assert "front camera 2026-11-09 22:58:43" in page.eval("document.querySelector('#calchip_front').title")
    assert page.text("#calchip") == "CAL 0.1 mm · ARM FIT 1.1 mm · ROLL 0.5°", "the TOP chip is what it always was"
    page.check()


def test_marker_zero_is_drawn_where_the_front_homography_puts_it(page, stub):
    rect = show_front(page, stub)
    u, v = mat_to_pixel(16.0, 10.0)
    assert abs(u - 341) < 2 and abs(v - 122) < 2                        # the fixture pose: marker 0 as the front camera sees it
    ex, ey = to_display(rect, u, v)
    dbg = page.eval("CALDBG_front.markers")
    assert set(dbg) == {"0", "1", "2", "3"}
    assert abs(dbg["0"][0] - ex) < 4 and abs(dbg["0"][1] - ey) < 4, (dbg["0"], (ex, ey))
    for mid, (mx, my) in fx.CALIB_MARKERS.items():
        px, py = mat_to_display(rect, mx, my)
        assert abs(dbg[mid][0] - px) < 4 and abs(dbg[mid][1] - py) < 4, mid
    assert opaque_front(page, round(ex), round(ey)) >= 20              # the 10 px square outline really is on the canvas there
    base = page.eval("CALDBG_front.base")
    bx, by = mat_to_display(rect, fx.CALIB_ARM["base_mm"][0] / 25.4, fx.CALIB_ARM["base_mm"][1] / 25.4)
    assert abs(base[0] - bx) < 4 and abs(base[1] - by) < 4
    mid = mat_to_display(rect, 9.5, 6.5)                               # the middle of a grid cell: only faint lines nearby at most
    assert opaque_front(page, round(mid[0]), round(mid[1]), r=3) == 0
    # the oblique view really is a different mapping from the top feed's: marker 0 lands elsewhere on its own feed
    top = page.eval("CALDBG.markers")
    assert set(top) == {"0", "1", "2", "3"}
    tu, tv = fx.h_apply(fx.h_inv(fx.CALIB_H), 16 * 25.4, 10 * 25.4)
    assert abs(tu - 93) < 4 and abs(tv - 276) < 4 and abs(tu - u) > 100
    page.check()


def test_tap_on_the_front_feed_measures_the_mat_point_under_it(page, stub):
    rect = show_front(page, stub)
    x, y = mat_to_display(rect, 9.0, 7.0)
    click_front(page, rect, x, y)
    txt, val = readout(page)
    assert "on" in page.cls("#calread_front")
    assert re.search(r"MAT \d\.\d\d, \d\.\d\d in · ARM -?\d+\.\d\d, -?\d+\.\d\d in", txt), txt
    tol = tol_at(rect, x, y)
    assert tol < 0.2, tol                                               # a display pixel is a few mm here, not centimetres
    assert near(val["MAT"], (9.0, 7.0), tol), (txt, tol)
    ax, ay = fx.table_to_arm(9.0 * 25.4, 7.0 * 25.4)
    assert near(val["ARM"], (ax / 25.4, ay / 25.4), tol), (txt, tol)
    tap = page.eval("CALDBG_front.tap")
    assert abs(tap[0] - x) < 1 and abs(tap[1] - y) < 1                 # crosshair under the finger
    assert opaque_front(page, round(x), round(y), r=22) >= 30
    assert "on" not in page.cls("#calread") and page.eval("calTap===null"), "the TOP overlay saw nothing"
    stub.wait_hits(["/api/locate"])                                     # confirmed with the server, through the front camera
    time.sleep(0.2)
    assert near(readout(page)[1]["MAT"], (9.0, 7.0), tol)
    # the server's answer wins when it differs; it was asked for cam=front with h_mm=0 (a recording stand-in for the callable route)
    seen = []
    real = fx.locate

    def rec(path):
        seen.append(path)
        return {"ok": True, "cam": "front", "pixel": [0, 0], "table_mm": [0, 0], "table_in": [1.23, 4.56], "arm_mm": [0, 0], "arm_in": [7.89, 0.12]}
    stub.routes["/api/locate"] = rec
    x2, y2 = mat_to_display(rect, 3.0, 2.0)
    click_front(page, rect, x2, y2)
    page.wait_until("document.querySelector('#calread_front').textContent.includes('1.23')")
    assert "7.89" in page.text("#calread_front")
    assert seen and "cam=front" in seen[-1] and "h_mm=0" in seen[-1], seen
    u2, v2 = mat_to_pixel(3.0, 2.0)
    q = dict(p.split("=") for p in seen[-1].split("?")[1].split("&"))
    assert abs(float(q["u"]) - u2) < 1.5 and abs(float(q["v"]) - v2) < 1.5, (q, (u2, v2))   # the raw front pixel went to the server
    stub.routes["/api/locate"] = real
    # a failing /api/locate leaves the client-side value in place
    stub.fail["/api/locate"] = 500
    click_front(page, rect, x, y)
    time.sleep(0.3)
    assert near(readout(page)[1]["MAT"], (9.0, 7.0), tol)
    del stub.fail["/api/locate"]
    # a tap on the readout clears the crosshair; a tap on the letterbox outside the image does nothing
    page.click("#calread_front")
    assert "on" not in page.cls("#calread_front") and page.eval("calOverlays.front.tap===null")
    _, _, w, h = rect
    s = min(w / IMG_W, h / IMG_H); ox, oy = (w - IMG_W * s) / 2, (h - IMG_H * s) / 2
    lx, ly = (ox / 2, h / 2) if ox > 4 else (w / 2, oy / 2)              # the bars are at the sides or top/bottom depending on the feed's shape
    assert ox > 4 or oy > 4
    click_front(page, rect, lx, ly)
    assert page.eval("calOverlays.front.tap===null") and "on" not in page.cls("#calread_front")
    page.check()


def test_front_uncalibrated_when_the_pose_is_absent(page, stub):
    """camera.front null (no front_campose.json yet): FRONT chip UNCALIBRATED, nothing drawn, taps ignored, while the TOP overlay works as usual."""
    rect = show_front(page, stub)
    x, y = mat_to_display(rect, 9.0, 7.0)
    click_front(page, rect, x, y)
    assert "on" in page.cls("#calread_front")
    stub.routes["/api/calib"] = fx.calib()                               # the top fit alone: camera.front = null
    page.eval("calibPoll()")
    assert page.text("#calchip_front") == "UNCALIBRATED" and "off" in page.cls("#calchip_front")
    assert "no front camera pose" in page.eval("document.querySelector('#calchip_front').title")
    assert "on" not in page.cls("#calread_front") and page.eval("calOverlays.front.tap===null")
    assert page.eval(BLANK_FRONT)
    click_front(page, rect, x, y)
    assert page.eval("calOverlays.front.tap===null") and "on" not in page.cls("#calread_front") and page.eval(BLANK_FRONT)
    assert stub.hits["/api/locate"] == 1                                 # only the calibrated tap asked the server
    assert page.text("#calchip") == "CAL 0.1 mm · ARM FIT 1.1 mm · ROLL 0.5°" and page.eval("Object.keys(CALDBG.markers).length") == 4
    # a singular front homography is the same as none; the whole calibration gone blanks both feeds
    stub.routes["/api/calib"] = fx.calib(front=fx.front(homography_px_to_table_mm=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    page.eval("calibPoll()")
    assert page.text("#calchip_front") == "UNCALIBRATED" and page.eval(BLANK_FRONT)
    stub.routes["/api/calib"] = fx.calib(ok=False)
    page.eval("calibPoll()")
    assert page.text("#calchip_front") == "UNCALIBRATED" and page.text("#calchip") == "UNCALIBRATED"
    # ...and it all comes back with the pose
    stub.routes["/api/calib"] = fx.calib(front=fx.front())
    page.eval("calibPoll()")
    assert page.text("#calchip_front") == "FRONT · 1.9 px" and page.eval("Object.keys(CALDBG_front.markers).length") == 4
    page.check()


def test_top_overlay_alone_is_unchanged_by_the_front_one(page, stub):
    """The original top-feed flow (test_feed_overlay.show_top: only the TOP feed live, no camera.front) still has its ids and draws its markers."""
    rect = show_top(page, stub)
    assert rect[2] > 100 and page.eval("Object.keys(CALDBG.markers).length") == 4
    assert page.eval("document.querySelector('#cam_front').parentElement.style.display") == "none"
    assert page.text("#calchip_front") == "UNCALIBRATED"
    assert page.eval("[typeof calDraw,typeof calClear,typeof calibPoll,typeof CALDBG,TOPFLIP,calTap]") == ["function", "function", "function", "object", False, None]
    page.check()
