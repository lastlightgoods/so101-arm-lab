"""Record form (dataset preselect, edit mode) and the transport strip's phases."""
import re
import time

import apidata as fx


def test_dataset_list_sorted_newest_first(page):
    assert page.eval("[...document.querySelectorAll('#rf_dsl option')].map(o=>o.value)") == fx.NEWEST_FIRST
    assert page.eval("DSNEW.map(d=>d.episodes)") == [7, 12, 30]
    page.check()


def test_dataset_list_follows_mtime_changes(page, stub):
    s = fx.state()
    s["datasets"]["charlie"]["mtime"] = int(time.time())          # charlie just got a new batch
    stub.routes["/api/state"] = s
    page.eval("refresh()")
    assert page.eval("[...document.querySelectorAll('#rf_dsl option')].map(o=>o.value)") == ["charlie", "bravo", "alpha"]
    page.check()


def test_record_form_idle_preselects_newest_in_append_mode(page):
    page.eval("document.querySelector('#rf_name').value='stale-from-localstorage'")
    page.click("#recbtn")
    assert "on" in page.cls("#recmodal")
    assert page.value("#rf_name") == "bravo"
    assert page.eval("document.querySelector('input[name=rf_mode]:checked').value") == "append"
    assert page.text("#rf_last").startswith("last batch: bravo · 7 ep · ")
    assert page.text("#rf_go") == "● START RECORDING"
    assert page.eval("document.querySelector('#rf_name').disabled") is False
    assert page.eval("document.querySelector('#rf_editnote').style.display") == "none"
    page.click("#rf_cancel")
    assert "on" not in page.cls("#recmodal")
    page.check()


def test_record_form_edit_mode_keeps_session_name(page, stub):
    stub.routes["/api/record"] = fx.record_running(name="sessionB")
    page.eval("recPoll()")
    assert page.eval("recRunning") is True
    page.click("#recbtn")
    assert page.value("#rf_name") == "sessionB"              # not overwritten by the newest dataset (bravo)
    assert page.eval("document.querySelector('#rf_name').disabled") is True
    assert page.eval("[...document.querySelectorAll('input[name=rf_mode]')].every(r=>r.disabled)") is True
    assert page.text("#rf_go") == "✓ APPLY TO SESSION"
    assert page.eval("document.querySelector('#rf_editnote').style.display") == ""
    assert page.value("#rf_n") == "20" and page.value("#rf_ep") == "25" and page.value("#rf_rs") == "10"
    page.click("#rf_cancel")
    # back to idle: the form is editable again and preselects the newest dataset
    stub.routes["/api/record"] = fx.record()
    page.eval("recPoll()")
    page.click("#recbtn")
    assert page.value("#rf_name") == "bravo"
    assert page.eval("document.querySelector('#rf_name').disabled") is False
    page.click("#rf_cancel")
    page.check()


def test_strip_idle(page, stub):
    assert page.cls("#recstrip") == ""
    assert page.text("#recidletxt") == "TRANSPORT IDLE · C TO RECORD A DATASET"
    assert page.text("#recbtntxt") == "● RECORD"
    stub.routes["/api/record"] = fx.record(name="sessionB", episode=20)
    page.eval("recPoll()")
    assert page.text("#recidletxt") == "TRANSPORT IDLE · LAST SESSION sessionB · 20 EP"
    page.check()


def test_strip_starting(page, stub):
    stub.routes["/api/record"] = fx.record_running(phase="starting", elapsed=0)
    stub.routes["/api/record"]["ep_started"] = 0
    page.eval("recPoll()")
    assert page.text("#rectext") == "STARTING · sessionB"
    assert page.eval("document.querySelector('#recbar').style.width") == "0%"
    page.check()


def test_strip_running(page, stub):
    stub.routes["/api/record"] = fx.record_running(elapsed=10)
    page.eval("recPoll()")
    assert page.cls("#recstrip") == "on"
    assert re.fullmatch(r"RECORDING · ep 3/20 · 1[456]s", page.text("#rectext"))
    assert page.text("#recbtntxt") == "✎ EDIT"
    assert page.text("#recpause") == "‖ PAUSE"
    width = float(page.eval("document.querySelector('#recbar').style.width").rstrip("%"))
    assert 38 <= width <= 46
    assert page.eval("document.querySelector('#recidle').offsetParent") is None   # idle pill hidden while the strip is on
    page.check()


def test_strip_paused_freezes_countdown(page, stub):
    stub.routes["/api/record"] = fx.record_running(elapsed=10, paused_at_elapsed=6)
    page.eval("recPoll()")
    txt = page.text("#rectext")
    assert txt == "PAUSED · ep 3/20 · 19s"
    assert page.text("#recpause") == "▶ RESUME"
    assert "paused" in page.cls("#recstrip").split()
    time.sleep(1.1)
    page.eval("recPoll()")
    assert page.text("#rectext") == txt                      # clock did not move between polls
    assert page.eval("document.querySelector('#recbar').style.width") == "24%"
    page.check()


def test_strip_reset_phase(page, stub):
    stub.routes["/api/record"] = fx.record_running(phase="reset", elapsed=4)
    page.eval("recPoll()")
    assert re.fullmatch(r"RESET · ep 3/20 · [567]s", page.text("#rectext"))
    assert page.cls("#recstrip") == "on reset"
    page.check()


def test_strip_done_and_failed_banners(page, stub):
    stub.routes["/api/record"] = fx.record_done()
    page.eval("recPoll()")
    assert page.cls("#recstrip") == "on done"
    assert page.text("#rectext") == "DONE · sessionB · 20 episodes recorded"
    assert page.eval("document.querySelector('#recbar').style.width") == "100%"
    assert page.text("#recbtntxt") == "● RECORD"
    stub.routes["/api/record"] = fx.record_done(err="bus timeout on wrist_roll")
    page.eval("recPoll()")
    assert page.cls("#recstrip") == "on done"
    assert page.text("#rectext") == "RECORD FAILED · bus timeout on wrist_roll"
    page.check()


def test_done_banner_clears_after_ninety_seconds(page, stub):
    stub.routes["/api/record"] = fx.record_done()
    page.eval("recPoll()")
    assert page.cls("#recstrip") == "on done"
    page.eval("recDoneAt=Date.now()-91000")
    page.eval("recPoll()")
    assert page.cls("#recstrip") == ""
    assert page.text("#recidletxt") == "TRANSPORT IDLE · LAST SESSION sessionB · 20 EP"
    page.check()
