"""Keyboard shortcuts: real key events through CDP Input, observed as POSTs at the stub."""
import time

import apidata as fx


def ctl(path, cmd):
    return lambda p, b: p == path and isinstance(b, dict) and b.get("cmd") == cmd


def test_p_pauses_while_recording(page, stub):
    stub.routes["/api/record"] = fx.record_running()
    page.eval("recPoll()")
    page.key("p")
    stub.wait_post(ctl("/api/record/ctl", "pause"))
    assert stub.posts_to("/api/arm/policy") == []
    page.check()


def test_p_is_policy_while_idle(page, stub):
    page.key("p")
    stub.wait_post(lambda p, b: p == "/api/arm/policy")
    assert stub.posts_to("/api/record/ctl") == []
    page.check()


def test_n_r_q_ignored_when_idle(page, stub):
    for k in "nrq":
        page.key(k)
    time.sleep(0.4)
    assert stub.posts == []
    page.check()


def test_n_r_q_drive_the_session_when_recording(page, stub):
    stub.routes["/api/record"] = fx.record_running()
    page.eval("recPoll()")
    for k, cmd in (("n", "next"), ("r", "rerecord"), ("q", "stop")):
        page.key(k)
        stub.wait_post(ctl("/api/record/ctl", cmd))
    page.check()


def test_c_opens_the_record_form(page, stub):
    page.key("c")
    page.wait_until("document.querySelector('#recmodal').classList.contains('on')")
    page.check()


def test_keys_ignored_while_typing_in_an_input(page, stub):
    page.eval("(()=>{const i=document.createElement('input');i.id='tt';document.body.appendChild(i);i.focus();return document.activeElement===i})()")
    for k in "ctsp":
        page.key(k)
    time.sleep(0.4)
    assert "on" not in page.cls("#recmodal")
    assert stub.posts == []
    assert page.value("#tt") == "ctsp"                       # the keystrokes went into the field
    page.eval("document.querySelector('#tt').remove()")
    page.check()


def test_keys_ignored_while_modal_open(page, stub):
    page.click("#recbtn")
    time.sleep(0.1)
    page.eval("document.activeElement&&document.activeElement.blur()")   # not typing: the modal guard alone must hold
    for k in "tsplc":
        page.key(k)
    time.sleep(0.4)
    assert stub.posts == []
    assert "on" in page.cls("#recmodal")
    page.click("#rf_cancel")
    page.check()


def test_mode_keys_post_arm_commands(page, stub):
    for k, mode in (("t", "teleop"), ("f", "free"), ("h", "hold"), ("s", "stop")):
        page.key(k)
        stub.wait_post(lambda p, b, m=mode: p == f"/api/arm/{m}")
    page.check()
