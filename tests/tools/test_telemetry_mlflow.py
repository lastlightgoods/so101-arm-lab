"""Servo telemetry in MLflow: pick_seq's per-waypoint `tele` summaries become tele_* metrics on each pick run (pick_loop) and on each autoloop cycle
and session run, and a servo_telemetry.json artifact; with no telemetry in the rows nothing changes. Faked arm, recorder instead of MLflow."""
import importlib.util
import os
import sys

import pytest
from pickhelpers import ROOT, ShortArm, drive

JETSON = os.path.join(ROOT, "jetson")
TOOLS = os.path.join(ROOT, "tools")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def st():
    if JETSON not in sys.path:
        sys.path.insert(0, JETSON)
    import servo_telemetry   # a plain import: the one module object fake_servo_bus and the code under test share
    return servo_telemetry


@pytest.fixture(scope="module")
def fb(st):
    return _load(os.path.join(TOOLS, "fake_servo_bus.py"), "fake_servo_bus_mlflowtest")


class TeleRows:
    """Wraps run_plan so every waypoint row carries what pick_seq would attach: a Summary from the pretend servo bus driven to that waypoint."""
    def __init__(self, fb, run_plan, heat=None):
        self.bus = fb.FakeServoBus(seed=4); self.bus.script = False; self.fb, self.inner = fb, run_plan
        if heat: self.bus.extra_heat.update(heat)
    def __call__(self, plan, max_deg_s):
        rows = self.inner(plan, max_deg_s)
        return [{**r, "tele": self.fb.waypoint_tele(self.bus, {k: v for k, v in r["target"].items()}, 1.5)} for r in rows]


def test_each_pick_run_gets_tele_metrics_and_the_json(pl, cc, table, st, fb, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(pl, "_SERVO", [st])
    arm = ShortArm(cc, table, shortfall_mm=7.0); arm.run_plan = TeleRows(fb, arm.run_plan, heat={"shoulder_lift": 0.2})
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")])
    assert any(o.get("picked") for o in out)
    (run,) = pl.rec.runs
    m = run.metrics_
    for j in st.JOINTS:
        for k in ("temp_max_c", "temp_rise_c", "load_abs_max_pct", "cur_max_a", "sag_abs_max", "sag_hold_max"): assert f"tele_{k}_{j}" in m, (k, j)
    assert m["tele_samples"] > 10 and m["tele_total_a_max"] > 0.5 and 9 < m["tele_volt_min_v"] < 12.4
    assert m["tele_temp_rise_c_shoulder_lift"] > m["tele_temp_rise_c_wrist_roll"]
    assert "servo_telemetry.json" in run.files
    assert m["landing_err_mm"] >= 0 and "sag_grasp" in m, "the existing metrics are all still there"


def test_picks_without_telemetry_log_exactly_what_they_did_before(pl, cc, table, st, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(pl, "_SERVO", [st])
    arm = ShortArm(cc, table, shortfall_mm=7.0)
    drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")])
    (run,) = pl.rec.runs
    assert not [k for k in run.metrics_ if k.startswith("tele")] and "servo_telemetry.json" not in run.files


def test_a_malformed_tele_row_never_costs_the_pick(pl, cc, table, st, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(pl, "_SERVO", [st])
    arm = ShortArm(cc, table, shortfall_mm=7.0); inner = arm.run_plan
    arm.run_plan = lambda plan, v: [{**r, "tele": {"n": 3, "j": "garbage"}} for r in inner(plan, v)]
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")])
    assert any(o.get("picked") for o in out)
    assert not [k for k in pl.rec.runs[0].metrics_ if k.startswith("tele")]


def test_tele_of_merges_waypoints_and_ignores_none(pl, st, fb, monkeypatch):
    monkeypatch.setattr(pl, "_SERVO", [st])
    bus = fb.FakeServoBus(seed=1); bus.script = False
    a, b = fb.waypoint_tele(bus, {"shoulder_lift": -40}, 1.0), fb.waypoint_tele(bus, {"shoulder_lift": -10}, 1.0)
    t = pl.tele_of([{"wp": "a", "tele": a}, {"wp": "b"}, {"wp": "c", "tele": b}])
    assert t["n"] == a["n"] + b["n"] and pl.tele_of([{"wp": "a"}]) is None and pl.tele_of([]) is None


# ---------------------------------------------------------------- autoloop: one set per cycle, one per session
def test_autoloop_cycles_and_session_carry_tele_metrics(autorig, st, fb, pl, capsys):
    pl._SERVO[:] = [st]
    r = autorig(); bus = fb.FakeServoBus(seed=4); bus.script = False; bus.extra_heat["shoulder_lift"] = 0.2
    r.row_hook = lambda row: {**row, "tele": fb.waypoint_tele(bus, row["target"], 1.0)}   # the rig's run_plan wrapper is the supported seam for rows
    s, out = r.run(capsys, cycles=2, max_retries=0, session="tt")
    parent, *kids = r.track.runs
    assert len(kids) == 2
    for k in kids:
        assert k.metrics_["tele_samples"] > 10 and "tele_temp_max_c_shoulder_lift" in k.metrics_ and "servo_telemetry.json" in k.files and "cycle.json" in k.files
    assert parent.metrics_["tele_samples"] == sum(k.metrics_["tele_samples"] for k in kids)
    assert parent.metrics_["tele_temp_rise_c_shoulder_lift"] >= max(k.metrics_["tele_temp_rise_c_shoulder_lift"] for k in kids) - 1e-9
    assert parent.metrics_["tele_temp_max_c_shoulder_lift"] == max(k.metrics_["tele_temp_max_c_shoulder_lift"] for k in kids)
    assert "success_rate" in parent.metrics_


def test_autoloop_without_tele_is_unchanged(autorig, st, pl, capsys):
    pl._SERVO[:] = [st]
    r = autorig(); s, out = r.run(capsys, cycles=2, max_retries=0, session="nt")
    parent, *kids = r.track.runs
    assert all(not [k for k in x.metrics_ if k.startswith("tele")] and x.files == ["cycle.json"] for x in kids) and not [k for k in parent.metrics_ if k.startswith("tele")]


def test_dry_run_arm_attaches_synthetic_tele(al, gr, cc, table):
    """autoloop --dry-run exercises the whole path: DryArm rows carry tele from the pretend bus."""
    import random
    world = al.FakeWorld(cc, table, gr.fk_mat_mm, rng=random.Random(1)); arm = al.DryArm(cc, table, world)
    t = arm.fake_tele({"shoulder_pan": 10, "shoulder_lift": -40, "elbow_flex": 50, "wrist_flex": 60, "wrist_roll": 0, "gripper": 30}, 1.0)
    assert t and t["n"] >= 3 and "shoulder_lift" in t["j"]
