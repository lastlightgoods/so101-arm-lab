"""The servo telemetry panel: what /api/tele rows draw, the flags, the button, the empty and failing states."""
import os

import apidata as fx
import teledata


def open_panel(page):
    page.click("#telebtn")
    page.wait_until("document.querySelector('#telep').getClientRects().length>0")
    page.wait_until("window.so101Tele.rows.length>0")
    page.eval("(teleText(),teleDraw(),true)")


def drawn(page, sel):
    """True when the canvas differs from a blank one of the same size."""
    return page.eval(f"(()=>{{const c=document.querySelector({sel!r}),b=document.createElement('canvas');b.width=c.width;b.height=c.height;return c.width>0&&c.toDataURL()!==b.toDataURL()}})()")


def test_panel_hidden_until_asked_on_the_kiosk(page):
    assert page.eval("document.querySelector('#telep').getClientRects().length") == 0
    assert page.eval("document.querySelector('#telebtn').offsetParent!==null")


def test_empty_stream_says_no_telemetry(page, stub):
    stub.routes["/api/tele"] = fx.tele_none()
    page.click("#telebtn")
    page.wait_until("document.querySelector('#telep').getClientRects().length>0")
    page.eval("(teleTick().then(()=>{teleText();teleDraw()}),true)")
    page.wait_until("document.querySelector('#tele_chips').textContent.includes('NO TELEMETRY')")
    assert "nothing yet" in page.text("#tele_ev")
    page.check()


def test_rows_draw_every_chart_and_fill_the_table(page, stub):
    store, _, _ = teledata.build(history_s=120)
    stub.routes["/api/tele"] = lambda path: teledata.payload(store)
    page.eval("(teleTick(),true)")
    open_panel(page)
    for c in ("temp", "load", "cur", "bus", "volt", "pos"):
        assert drawn(page, f"#tc_{c}"), c
    rows = page.eval("[...document.querySelectorAll('#tele_t tbody tr')].map(r=>r.cells[0].textContent)")
    assert rows == ["PAN", "LIFT", "ELBOW", "WRIST", "ROLL", "GRIP"]
    assert "LIVE" in page.text("#tele_chips") and " V" in page.text("#tele_chips") and " A" in page.text("#tele_chips")
    assert page.eval("window.so101Tele.rows.length") > 500
    page.check()


def test_hot_servo_raises_flags_and_the_button_turns_red(page, stub):
    store, _, _ = teledata.build(history_s=420, hot="lift")
    stub.routes["/api/tele"] = lambda path: teledata.payload(store)
    page.eval("(teleTick(),true)")
    open_panel(page)
    assert page.eval("window.so101Tele.flags.some(f=>f.joint==='shoulder_lift'&&f.kind==='hot')")
    page.wait_until("document.querySelector('#telebtn').classList.contains('l2')")   # red: past 65 C, and the servo's own temperature bit is up
    assert "HOT" in page.text("#tele_t tbody tr:nth-child(2)")
    assert "LIFT" in page.text("#tele_ev") and "C, cut at 70 C" in page.text("#tele_ev")
    page.check()


def test_incremental_polling_asks_only_for_new_rows(page, stub):
    store, bus, tick = teledata.build(history_s=60)
    seen = []
    stub.routes["/api/tele"] = lambda path: (seen.append(path), teledata.payload(store, float(path.split("since=")[1])))[1]
    page.eval("(teleTick(),true)")
    page.wait_until("window.so101Tele.rows.length>100")
    n = page.eval("window.so101Tele.rows.length")
    tick()
    page.eval("(teleTick(),true)")
    page.wait_until(f"window.so101Tele.rows.length>{n}")
    assert page.eval("window.so101Tele.rows.length") - n <= 2   # one new row, not the whole history again
    assert float(seen[-1].split("since=")[1]) > 0
    page.check()


def test_joint_picker_switches_the_commanded_vs_actual_chart(page, stub):
    store, _, _ = teledata.build(history_s=120)
    stub.routes["/api/tele"] = lambda path: teledata.payload(store)
    page.eval("(teleTick(),true)")
    open_panel(page)
    page.click("#tele_pick button[data-j=elbow_flex]")
    assert page.eval("window.so101Tele.sel") == "elbow_flex"
    assert "sag" in page.text("#th_pos")
    page.click("#tele_win button[data-w='60']")
    assert page.eval("window.so101Tele.win") == 60
    page.check()


def test_broken_endpoint_does_not_throw(page, stub):
    stub.raw["/api/tele"] = b"{bad"
    page.eval("(teleTick(),true)")
    page.eval("teleTick()")
    assert page.eval("window.so101Tele.err") is True
    page.check()


def test_m_key_toggles_the_panel(page, stub):
    store, _, _ = teledata.build(history_s=60)
    stub.routes["/api/tele"] = lambda path: teledata.payload(store)
    page.key("m")
    page.wait_until("document.querySelector('#telep').getClientRects().length>0")
    page.key("m")
    page.wait_until("document.querySelector('#telep').getClientRects().length===0")
    page.check()


def test_panel_screenshot(page, stub):
    """SO101_TELE_SHOT=/path.png saves the panel (a hot shoulder lift, a brown-out dip) for the status report; otherwise just checks it opens."""
    shot = os.environ.get("SO101_TELE_SHOT")
    store, _, _ = teledata.build(history_s=420, hot="lift")
    stub.routes["/api/tele"] = lambda path: teledata.payload(store)
    page.eval("(teleTick(),true)")
    open_panel(page)
    page.eval("(window.scrollTo(0,0),true)")
    if shot:
        page.screenshot(shot)
    page.check()


def test_ipad_landscape_layout_shows_the_panel_inline(page, stub):
    """The tablet layout (landscape, under 1366 px wide) is what an iPad gets: the panel is part of the scrolling page under the arm row, not an overlay.
    An iframe 1180 x 820 gets that layout's media queries without resizing the browser. SO101_TELE_PAD_SHOT=/path.png saves what the iPad would show."""
    store, _, _ = teledata.build(history_s=300, hot="lift")
    stub.routes["/api/tele"] = lambda path: teledata.payload(store)
    page.eval("""(()=>{const f=document.createElement('iframe');f.id='pad';f.src='/';f.style.cssText='position:fixed;left:0;top:0;width:1180px;height:820px;z-index:99;border:0;background:#000';document.body.appendChild(f);return true})()""")
    page.wait_until("(()=>{const d=document.querySelector('#pad').contentDocument;return !!d&&!!d.querySelector('#telep')&&d.querySelector('#telep').getClientRects().length>0&&d.defaultView.so101Tele.rows.length>100})()", timeout=15)
    inline = page.eval("(()=>{const d=document.querySelector('#pad').contentDocument,p=d.querySelector('#telep'),cs=d.defaultView.getComputedStyle(p),r=p.getBoundingClientRect();return {pos:cs.position,w:Math.round(r.width),h:Math.round(r.height),tablet:d.defaultView.matchMedia('(orientation: landscape) and (max-width: 1366px)').matches}})()")
    assert inline["tablet"] and inline["pos"] != "fixed", inline
    assert 1100 <= inline["w"] <= 1180, f"the panel is {inline['w']} px wide in a 1180 px page"
    page.eval("(()=>{const d=document.querySelector('#pad').contentDocument;d.defaultView.teleTick().then(()=>{d.defaultView.teleText();d.defaultView.teleDraw()});return true})()")
    page.eval("(()=>{const d=document.querySelector('#pad').contentDocument;d.querySelector('#telep').scrollIntoView();return true})()")
    assert page.eval("(()=>{const d=document.querySelector('#pad').contentDocument;return [...d.querySelectorAll('#telep canvas.tchart')].every(c=>c.getClientRects().length>0&&c.clientWidth>200)})()")
    shot = os.environ.get("SO101_TELE_PAD_SHOT")
    if shot:
        page.screenshot(shot)
    page.check()
