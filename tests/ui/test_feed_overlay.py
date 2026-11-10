"""Top-feed calibration overlay: mat grid/markers projected through the homography, tap-to-measure readout, CAL chip.

The overlay maps raw 640x360 camera pixels onto the displayed #cam_top, which is letterboxed (and rotated 180 degrees only when TOPFLIP is on)
(object-fit:contain); the expectations here are computed independently in Python from the same fixture homography.
"""
import re
import time

import apidata as fx

IMG_W, IMG_H = 640, 360
TOL_IN = 0.04      # one display pixel is ~0.03 in on the 1600 px headless window (synthetic clicks land on whole pixels)


def show_top(page, stub, cal=None):
    """Bring the TOP feed up (the panel is hidden until /api/cams lists a camera) and load the calibration."""
    stub.routes["/api/cams"] = ["top"]
    page.eval("cams()")
    stub.routes["/api/calib"] = fx.calib() if cal is None else cal      # the stub's default route is the uncalibrated answer; these tests want the real fit unless told otherwise
    page.eval("calibPoll()")
    page.wait_until("(()=>{const r=document.querySelector('#calov').getBoundingClientRect();return r.width>100&&r.height>100})()")
    page.eval("calDraw(); true")
    return page.eval("(()=>{const r=document.querySelector('#calov').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")


TOPFLIP = False    # must match TOPFLIP in index.html: the top feed is shown as the camera delivers it (no 180 degree rotation since the mat was turned on 9 Nov)


def to_display(rect, u, v):
    """Raw camera pixel -> css px inside the feed box: the optional 180 degree rotation, then the contain letterbox."""
    _, _, w, h = rect
    s = min(w / IMG_W, h / IMG_H)
    dw, dh = IMG_W * s, IMG_H * s
    ox, oy = (w - dw) / 2, (h - dh) / 2
    if TOPFLIP:
        return ox + (IMG_W - u) * s, oy + (IMG_H - v) * s
    return ox + u * s, oy + v * s


def mat_to_pixel(x_in, y_in):
    return fx.h_apply(fx.h_inv(fx.CALIB_H), x_in * 25.4, y_in * 25.4)


def mat_to_display(rect, x_in, y_in):
    u, v = mat_to_pixel(x_in, y_in)
    return to_display(rect, u, v)


def click_feed(page, rect, x, y):
    page.eval(f"(document.querySelector('#cam_top').dispatchEvent(new MouseEvent('click',{{bubbles:true,clientX:{rect[0] + x},clientY:{rect[1] + y}}})), true)")


def opaque_pixels_near(page, x, y, r=8):
    """Count canvas pixels within +-r of (x, y) that are drawn at (near) full alpha: the marker squares and crosshair are
    alpha 1, the 1-inch grid is alpha .16, so the faint grid never satisfies this."""
    return page.eval(f"(()=>{{const c=document.querySelector('#calov'),d=c.getContext('2d').getImageData({x - r},{y - r},{2 * r},{2 * r}).data;"
                     f"let n=0;for(let i=3;i<d.length;i+=4)if(d[i]>200)n++;return n}})()")


def test_marker_zero_is_drawn_where_the_homography_puts_it(page, stub):
    rect = show_top(page, stub)
    u, v = mat_to_pixel(16.0, 10.0)
    assert abs(u - 93) < 4 and abs(v - 276) < 4            # the fixture frame: marker 0's centre as the camera saw it
    ex, ey = to_display(rect, u, v)
    dbg = page.eval("CALDBG.markers")
    assert set(dbg) == {"0", "1", "2", "3"}
    assert abs(dbg["0"][0] - ex) < 4 and abs(dbg["0"][1] - ey) < 4
    assert opaque_pixels_near(page, round(ex), round(ey)) >= 20       # the 10 px square outline really is on the canvas there
    mid = mat_to_display(rect, 9.5, 6.5)                               # the middle of a grid cell: only faint lines nearby at most
    assert opaque_pixels_near(page, round(mid[0]), round(mid[1]), r=4) == 0
    # marker 0 (mat x=16) sits on the LEFT of the raw frame: drawn on the left of the wall view, or on the right when the feed is flipped 180 degrees
    assert (dbg["0"][0] > dbg["1"][0]) if TOPFLIP else (dbg["0"][0] < dbg["1"][0])
    page.check()


def readout(page):
    """'MAT 9.02, 6.99 in · ARM 10.17, -0.00 in×' -> {'MAT': (9.02, 6.99), 'ARM': (10.17, -0.0)}"""
    txt = page.text("#calread")
    return txt, {k: (float(a), float(b)) for k, a, b in re.findall(r"(MAT|ARM) (-?\d+\.\d\d), (-?\d+\.\d\d) in", txt)}


def near(got, want, tol=TOL_IN):
    return all(abs(g - w) < tol for g, w in zip(got, want))


def test_tap_measures_mat_and_arm_inches(page, stub):
    rect = show_top(page, stub)
    x, y = mat_to_display(rect, 9.0, 7.0)
    click_feed(page, rect, x, y)
    txt, val = readout(page)
    assert "on" in page.cls("#calread")
    assert re.search(r"MAT \d\.\d\d, \d\.\d\d in · ARM -?\d+\.\d\d, -?\d+\.\d\d in", txt), txt     # two decimals, both frames
    assert near(val["MAT"], (9.0, 7.0)), txt
    ax, ay = fx.table_to_arm(9.0 * 25.4, 7.0 * 25.4)
    assert near(val["ARM"], (ax / 25.4, ay / 25.4)), txt
    tap = page.eval("CALDBG.tap")
    assert abs(tap[0] - x) < 1 and abs(tap[1] - y) < 1                 # crosshair drawn under the finger
    assert opaque_pixels_near(page, round(x), round(y), r=22) >= 30
    stub.wait_hits(["/api/locate"])                                     # confirmed with the server too...
    time.sleep(0.2)
    assert near(readout(page)[1]["MAT"], (9.0, 7.0))                    # ...which agrees, so the readout stands
    # the server's answer wins when it differs from the client-side estimate
    stub.routes["/api/locate"] = {"ok": True, "pixel": [0, 0], "table_mm": [0, 0], "table_in": [1.23, 4.56], "arm_mm": [0, 0], "arm_in": [7.89, 0.12]}
    x2, y2 = mat_to_display(rect, 3.0, 2.0)
    click_feed(page, rect, x2, y2)
    page.wait_until("document.querySelector('#calread').textContent.includes('1.23')")
    assert "7.89" in page.text("#calread")
    # a failing /api/locate leaves the client-side value in place
    stub.fail["/api/locate"] = 500
    click_feed(page, rect, x, y)
    time.sleep(0.3)
    assert near(readout(page)[1]["MAT"], (9.0, 7.0))
    del stub.fail["/api/locate"]
    # a tap on the chip clears the crosshair; a tap on the letterbox outside the image does nothing
    page.click("#calread")
    assert "on" not in page.cls("#calread")
    assert page.eval("calTap===null")
    click_feed(page, rect, 2, rect[3] / 2)
    assert page.eval("calTap===null") and "on" not in page.cls("#calread")
    page.check()


def test_status_chip_from_residuals(page, stub):
    show_top(page, stub)
    assert page.text("#calchip") == "CAL 0.1 mm · ARM FIT 1.1 mm · ROLL 0.5°"
    assert "off" not in page.cls("#calchip")
    stub.routes["/api/calib"] = fx.calib(arm=False)
    page.eval("calibPoll()")
    assert page.text("#calchip") == "CAL 0.1 mm · ARM UNFITTED"
    page.check()


def test_uncalibrated_draws_nothing_and_ignores_taps(page, stub):
    rect = show_top(page, stub)
    x, y = mat_to_display(rect, 9.0, 7.0)
    click_feed(page, rect, x, y)
    assert "on" in page.cls("#calread")
    stub.routes["/api/calib"] = fx.calib(ok=False)
    page.eval("calibPoll()")
    assert page.text("#calchip") == "UNCALIBRATED"
    assert "off" in page.cls("#calchip")
    assert "on" not in page.cls("#calread") and page.eval("calTap===null")      # the old crosshair is gone too
    blank = "(()=>{const c=document.querySelector('#calov'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;for(let i=3;i<d.length;i+=4)if(d[i])return false;return true})()"
    assert page.eval(blank)
    click_feed(page, rect, x, y)
    assert page.eval("calTap===null") and "on" not in page.cls("#calread")
    assert page.eval(blank)
    assert stub.hits["/api/locate"] == 1                                         # only the calibrated tap asked the server
    page.check()


def test_calib_route_garbage_then_recovers(page, stub):
    show_top(page, stub)
    stub.raw["/api/calib"] = b"{not json"
    page.eval("calibPoll()")
    assert page.text("#calchip") == "UNCALIBRATED"
    del stub.raw["/api/calib"]
    stub.fail["/api/calib"] = 500
    page.eval("calibPoll()")
    assert page.text("#calchip") == "UNCALIBRATED"
    del stub.fail["/api/calib"]
    stub.routes["/api/calib"] = {"ok": True, "homography_px_to_table_mm": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}   # singular
    page.eval("calibPoll()")
    assert page.text("#calchip") == "UNCALIBRATED"
    stub.routes["/api/calib"] = fx.calib()
    page.eval("calibPoll()")
    assert page.text("#calchip").startswith("CAL 0.1 mm")
    assert page.eval("Object.keys(CALDBG.markers).length") == 4
    page.check()


def test_lock_override_press_and_hold_still_works(page, stub):
    """The lock screen's 2 s press-and-hold override must not be disturbed by the feed's tap handler."""
    show_top(page, stub)
    stub.routes["/api/lock"] = {"locked": True, "status": "SCANNING", "progress": 0.2}
    page.eval("lockPoll()")
    assert "off" not in page.cls("#lockscreen")
    page.eval("(document.querySelector('#lockscreen').dispatchEvent(new PointerEvent('pointerdown',{bubbles:true})), true)")
    stub.wait_post(lambda p, b: p == "/api/unlock", timeout=3.5)
    assert page.text("#lockst") == "OVERRIDE ACCEPTED"
    assert page.eval("calTap===null")                     # the hold on the lock screen never reached the feed
    page.check()
