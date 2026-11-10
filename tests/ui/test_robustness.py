"""Broken backends: 500s, invalid JSON, slow routes. The page must never throw and the next poll must recover."""
import time

import pytest

import apidata as fx

POLLERS = {"/api/state": "refresh()", "/api/hist": "refresh()", "/api/record": "recPoll()", "/api/cams": "cams()",
           "/api/term": "term()", "/api/shell": "shell()", "/api/sim": "simPoll()", "/api/lock": "lockPoll()"}


@pytest.mark.parametrize("route", list(POLLERS))
def test_route_500_then_garbage_then_recovers(page, stub, route):
    fn = POLLERS[route]
    stub.fail[route] = 500
    stub.routes["/api/state"] = fx.state(gpu=66)
    page.eval(fn)
    page.eval("refresh()")
    if route != "/api/state":
        assert page.text("#v_gpu") == "66%"              # a broken side route never stalls the gauges
    del stub.fail[route]
    stub.raw[route] = b"{not json at all"
    stub.routes["/api/state"] = fx.state(gpu=67)
    page.eval(fn)
    page.eval("refresh()")
    if route != "/api/state":
        assert page.text("#v_gpu") == "67%"
    del stub.raw[route]
    stub.routes["/api/state"] = fx.state(gpu=77)
    page.eval(fn)
    page.eval("refresh()")
    assert page.text("#v_gpu") == "77%"                  # next poll recovers
    page.check()


def test_state_error_object_from_server(page, stub):
    """server.py answers {"error": ...} when it cannot build the state JSON."""
    stub.routes["/api/state"] = {"error": "TypeError: Object of type bytes is not JSON serializable"}
    page.eval("refresh()")
    page.eval("arms()")
    stub.routes["/api/state"] = fx.state(gpu=77)
    page.eval("refresh()")
    assert page.text("#v_gpu") == "77%"
    page.check()


def test_hist_non_array_json(page, stub):
    stub.routes["/api/hist"] = {"error": "no history"}
    page.eval("refresh()")
    time.sleep(0.2)                                       # let the (not awaited) hist fetch land
    page.eval("refresh()")
    assert page.text("#v_gpu") == "35%"
    page.check()


def test_slow_hist_does_not_stall_gauges(page, stub):
    stub.delay["/api/hist"] = 2.5
    stub.routes["/api/state"] = fx.state(gpu=77)
    page.eval("refresh(); true")                          # fire, do not await
    page.wait_until("document.querySelector('#v_gpu').textContent==='77%'", timeout=1.5)
    del stub.delay["/api/hist"]
    page.check()


def test_slow_hist_still_lands_and_draws(page, stub):
    stub.delay["/api/hist"] = 0.6
    stub.routes["/api/hist"] = fx.hist(150)
    page.eval("refresh(); true")
    page.wait_until("HIST.length===150", timeout=3)
    del stub.delay["/api/hist"]
    page.eval("refresh()")
    assert page.eval("(()=>{const c=document.querySelector('#g_gpu');const a=c.toDataURL();gauge('#g_gpu',0.77,heat(0.77));return a!==c.toDataURL()})()")
    page.check()


def test_live_page_survives_an_outage_and_recovers(live_page, stub):
    """Interval loops keep running through 500s on every route, then pick the data back up on their own."""
    for r in POLLERS:
        stub.fail[r] = 500
    time.sleep(3.5)                                       # one full refresh cycle + several arms()/lockPoll() ticks
    stub.fail.clear()
    stub.routes["/api/state"] = fx.state(gpu=88)
    live_page.wait_until("document.querySelector('#v_gpu').textContent==='88%'", timeout=8)
    live_page.check()
