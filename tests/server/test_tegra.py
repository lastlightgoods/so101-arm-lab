"""parse_tegra() on real tegrastats lines, and tegra_loop() feeding state + the HIST ring buffer."""
import time
import pytest

# root tegrastats on the Orin Nano (JetPack 6): EMC_FREQ present, GR3D freq in brackets, VDD rails
ROOT_LINE = ("11-07-2026 21:14:02 RAM 4389/7620MB (lfb 10x4MB) SWAP 0/3810MB (cached 0MB) "
             "CPU [15%@1510,9%@1510,6%@1510,10%@1510,100%@1510,3%@1510] EMC_FREQ 7%@2133 GR3D_FREQ 42%@[611] "
             "NVDEC off NVJPG off NVJPG1 off VIC off OFA off APE 174 cpu@49.312C soc2@47.093C soc0@48.312C gpu@48.062C "
             "tj@49.312C soc1@47.906C VDD_IN 6281mW/6164mW VDD_CPU_GPU_CV 1964mW/1890mW VDD_SOC 1690mW/1680mW")
# unprivileged tegrastats: no EMC_FREQ, no GR3D clock, same temps/power
USER_LINE = ("11-07-2026 21:14:04 RAM 4390/7620MB (lfb 10x4MB) SWAP 0/3810MB (cached 0MB) "
             "CPU [12%@1510,off,6%@1510,10%@1510,100%@1510,3%@1510] GR3D_FREQ 0% "
             "cpu@49.0C soc2@47.0C soc0@48.0C gpu@51.5C tj@51.5C soc1@47.9C VDD_IN 5234mW/5234mW VDD_SOC 1690mW/1680mW")


def test_root_line_full_fields(srv):
    d = srv.parse_tegra(ROOT_LINE)
    assert (d["ram_used"], d["ram_total"], d["swap_used"], d["swap_total"]) == (4389, 7620, 0, 3810)
    assert d["cpu_cores"] == [15, 9, 6, 10, 100, 3] and d["cpu"] == pytest.approx(143 / 6) and d["cpu_mhz"] == 1510
    assert (d["gpu"], d["gpu_mhz"]) == (42, 611)
    assert (d["emc"], d["emc_mhz"]) == (7, 2133)
    assert d["temps"]["cpu"] == 49.312 and d["temps"]["gpu"] == 48.062 and d["temps"]["tj"] == 49.312
    assert d["temp"] == 49.312                         # max of tj / gpu / cpu
    assert d["power_mw"] == {"VDD_IN": 6281, "VDD_CPU_GPU_CV": 1964, "VDD_SOC": 1690}
    assert d["watts"] == pytest.approx(6.281)


def test_user_line_without_emc_or_gpu_clock(srv):
    d = srv.parse_tegra(USER_LINE)
    assert d["emc"] is None and d["emc_mhz"] == 0, "emc must be None (not 0) when tegrastats runs unprivileged"
    assert (d["gpu"], d["gpu_mhz"]) == (0, 0)
    assert d["cpu_cores"] == [12, 6, 10, 100, 3], "an 'off' core is skipped, not counted as 0"
    assert d["temp"] == 51.5 and d["watts"] == pytest.approx(5.234)


def test_gr3d_freq_without_brackets(srv):
    line = ROOT_LINE.replace("GR3D_FREQ 42%@[611]", "GR3D_FREQ 42%@611")
    assert srv.parse_tegra(line)["gpu_mhz"] == 611


def test_no_power_rails_gives_zero_watts(srv):
    line = ROOT_LINE.split(" VDD_IN")[0]
    d = srv.parse_tegra(line)
    assert d["watts"] == 0 and d["power_mw"] == {}


@pytest.mark.parametrize("junk", ["", "ssh: connect to host jetson port 22: No route to host", "RAM 1/2MB garbage"])
def test_non_tegra_lines_are_ignored(srv, junk):
    assert srv.parse_tegra(junk) is None


def test_tegra_loop_feeds_state_and_hist(srv, fake, run_loop, events):
    lines = [ROOT_LINE.replace("GR3D_FREQ 42%", f"GR3D_FREQ {g}%") + "\n" for g in range(160)]
    lines.insert(5, "not a tegrastats line\n")
    fake.on_popen("tegrastats", lines)
    t0 = time.time()
    run_loop(srv.tegra_loop, 1)                         # stream ends -> offline -> sleep(5) stops the loop
    argv, kw = fake.popens[0]
    assert argv == srv.SSH + [srv.TEGRA_CMD] and srv.JETSON in argv and kw.get("text")
    assert "sudo -n tegrastats" in srv.TEGRA_CMD and "|| tegrastats" in srv.TEGRA_CMD   # falls back when sudo -n is refused
    # HIST: one row per parsed line, 7 fields, capped at 150, newest last
    assert len(srv.HIST) == 150
    row = srv.HIST[-1]
    assert len(row) == 7
    assert row[0] == pytest.approx(round(t0), abs=2) and row[1] == 159 and row[2] == 7
    assert row[3] == round(143 / 6) and row[4] == round(100 * 4389 / 7620) and row[5] == 49.3 and row[6] == 6.28
    assert srv.HIST[0][1] == 10                          # rows 0..9 were evicted
    # state: last line applied, then the loop marked the Jetson offline when the stream ended
    assert srv.state["jetson"]["gpu"] == 159 and srv.state["jetson"]["online"] is False
    assert srv.state["jetson"]["last"] == pytest.approx(t0, abs=2)
    assert events() == ["Jetson offline", "Jetson online"]   # online announced once, not per line


def test_hist_emc_none_when_unprivileged(srv, fake, run_loop):
    fake.on_popen("tegrastats", [USER_LINE + "\n"])
    run_loop(srv.tegra_loop, 1)
    assert srv.HIST[0][2] is None
