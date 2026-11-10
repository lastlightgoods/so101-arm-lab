"""The six Jetson gauges: values, the BUS gauge's three states, offline dashes."""
import apidata as fx

GAUGES = ["gpu", "bus", "cpu", "ram", "tmp", "pwr"]


def drawn(page, g):
    """True when the gauge canvas differs from a blank canvas of the same size."""
    return page.eval(f"(()=>{{const c=document.querySelector('#g_{g}'),b=document.createElement('canvas');"
                     f"b.width=c.width;b.height=c.height;return c.toDataURL()!==b.toDataURL()}})()")


def test_six_gauges_render_with_values(page):
    assert page.eval("document.querySelectorAll('.gauges canvas').length") == 6
    assert all(drawn(page, g) for g in GAUGES)
    assert page.text("#v_gpu") == "35%"
    assert page.text("#v_cpu") == "23%"       # Math.round(22.5)
    assert page.text("#v_ram") == "2.9G"      # 3000 MB / 1024
    assert page.text("#v_tmp") == "48°"
    assert page.text("#v_pwr") == "7.9W"
    assert page.text("#badge") == "IDLE · READY"
    assert page.text("#hosts") != ""
    page.check()


def test_bus_gauge_with_emc(page):
    assert page.text("#v_bus") == "42%"
    assert page.text("#l_bus") == "BUS · 665 MHz"
    page.check()


def test_bus_gauge_without_emc_needs_root(page, stub):
    stub.routes["/api/state"] = fx.state(emc=None, emc_mhz=0)
    page.eval("refresh()")
    assert page.text("#v_bus") == "--"
    assert page.text("#l_bus") == "BUS · NEEDS ROOT"
    assert page.text("#v_gpu") == "35%"       # the other gauges are unaffected
    assert drawn(page, "bus")                 # the dial still renders, at zero
    page.check()


def test_bus_gauge_emc_without_mhz(page, stub):
    stub.routes["/api/state"] = fx.state(emc=12, emc_mhz=0)
    page.eval("refresh()")
    assert page.text("#v_bus") == "12%"
    assert page.text("#l_bus") == "BUS"
    page.check()


def test_offline_shows_dashes_everywhere(page, stub):
    stub.routes["/api/state"] = fx.state(online=False)
    page.eval("refresh()")
    assert [page.text(f"#v_{g}") for g in GAUGES] == ["--"] * 6
    assert page.text("#l_bus") == "BUS"
    assert page.text("#badge") == "JETSON OFFLINE"
    assert page.cls("#badge") == "badge off"
    assert page.text("#hosts") == "" and page.text("#jmode") == ""
    assert all(drawn(page, g) for g in GAUGES)   # dials stay painted (empty), only the needles drop
    page.check()


def test_back_online_restores_values(page, stub):
    stub.routes["/api/state"] = fx.state(online=False)
    page.eval("refresh()")
    assert page.text("#v_gpu") == "--"
    stub.routes["/api/state"] = fx.state(gpu=61)
    page.eval("refresh()")
    assert page.text("#v_gpu") == "61%"
    assert page.text("#v_bus") == "42%"
    page.check()
