"""spark(): the per-gauge history traces drawn inside the dial."""
import json

import apidata as fx

# draw a known gauge, remember it, run spark() on top, report whether any pixel changed
SPARK = ("(()=>{const c=document.querySelector('#g_gpu');gauge('#g_gpu',0.3,'#fff');const a=c.toDataURL();"
         "spark('#g_gpu',%s,%d,100,'#4fd6ff');return a!==c.toDataURL()})()")


def test_spark_empty_hist_draws_nothing(page):
    assert page.eval(SPARK % ("[]", 1)) is False
    assert page.eval(SPARK % ("null", 1)) is False
    page.check()


def test_spark_single_row_draws_nothing(page):
    assert page.eval(SPARK % (json.dumps(fx.hist(1)), 1)) is False
    page.check()


def test_spark_150_rows_draws(page):
    assert page.eval(SPARK % (json.dumps(fx.hist(150)), 1)) is True
    page.check()


def test_spark_null_emc_column(page):
    assert page.eval(SPARK % (json.dumps(fx.hist(150, emc=False)), 2)) is False   # every emc null: nothing to trace
    mixed = fx.hist(150)
    for r in mixed[::2]:
        r[2] = None
    assert page.eval(SPARK % (json.dumps(mixed), 2)) is True                      # nulls skipped, the rest traced
    page.check()


def test_spark_ignores_rows_older_than_five_minutes(page):
    old = [[r[0] - 600, *r[1:]] for r in fx.hist(150)]
    assert page.eval(SPARK % (json.dumps(old), 1)) is False
    page.check()


def test_refresh_draws_sparklines_when_online(page, stub):
    stub.routes["/api/hist"] = fx.hist(150)
    page.eval("refresh()")
    page.wait_until("HIST.length===150")
    page.eval("refresh()")
    # after refresh() every dial holds more than the bare gauge: repainting the bare gauge changes the pixels
    for g, frac in (("gpu", "0.35"), ("cpu", "22.5/100"), ("pwr", "7.9/25")):
        assert page.eval(f"(()=>{{const c=document.querySelector('#g_{g}');const a=c.toDataURL();"
                         f"gauge('#g_{g}',{frac},heat({frac}));return a!==c.toDataURL()}})()"), g
    page.check()


def test_refresh_draws_no_sparkline_when_offline(page, stub):
    stub.routes["/api/hist"] = fx.hist(150)
    page.eval("refresh()")
    page.wait_until("HIST.length===150")
    stub.routes["/api/state"] = fx.state(online=False)
    page.eval("refresh()")
    for g in ("gpu", "bus", "cpu", "ram", "tmp", "pwr"):
        assert page.eval(f"(()=>{{const c=document.querySelector('#g_{g}');const a=c.toDataURL();"
                         f"gauge('#g_{g}',0,heat(0));return a===c.toDataURL()}})()"), g   # bare empty dial, nothing else
    page.check()
