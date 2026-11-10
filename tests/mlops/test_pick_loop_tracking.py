"""tools/pick_loop.py's MLflow logging, driven with every hardware edge stubbed: the dashboard's /api/ik and overhead JPEG,
the ssh'd pick_seq rows, and cam_calib's pixel<->table maps. One run per pick in `so101-picks`; with tracking off the
stdout lines are byte-identical to the tracked run's."""
import json, os, sys
import pytest
from mlhelpers import JETSON, TOOLS, load, runs

pytest.importorskip("cv2")
JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")   # a (headers-only) JPEG: find_nut is stubbed, the bytes just travel


class FakeArm:
    """Scripted outcome per run: the nut is found where the previous run landed, pick_seq reports a little sag, the nut lands
    `land_off_in` from the target. `stop_after` runs it is lost."""
    def __init__(self, land_off_in=(0.1, -0.05), stop_after=None):
        self.land_off, self.stop_after, self.plans, self.frames, self.n = land_off_in, stop_after, [], 0, 0

    def find_nut(self, cc, table, near_in, radius_px=70, jpeg=None):
        if jpeg is not None: self.frames += 1
        if self.stop_after is not None and jpeg is not None and self.n >= self.stop_after: return None
        return [near_in[0] + (self.land_off[0] if jpeg is not None else 0.0), near_in[1] + (self.land_off[1] if jpeg is not None else 0.0)], [320, 180], 40

    def api(self, path):
        q = dict(p.split("=") for p in path.split("?")[1].split("&"))
        return {"ok": True, "joints_deg": {"shoulder_pan": float(q["x_in"]) * 3, "shoulder_lift": -40.0, "elbow_flex": 50.0, "wrist_flex": 60.0 + float(q["h_in"]), "wrist_roll": 0.5}}

    def run_plan(self, plan, max_deg_s):
        self.n += 1; self.plans.append(plan)
        return [{"wp": w["name"], "seconds": 1.0, "t": i, "target": w["joints"], "reached": {k: v - (0.0 if w["gripper_only"] else 1.5) for k, v in w["joints"].items()}, "max_err_deg": 0.0 if w["gripper_only"] else 1.5} for i, w in enumerate(plan["waypoints"])]


@pytest.fixture
def pl(tmp_path, monkeypatch):
    m = load(os.path.join(TOOLS, "pick_loop.py"))
    calib = tmp_path / "jetson" / "calib"; calib.mkdir(parents=True)
    json.dump({"units": "in"}, open(calib / "top_table.json", "w")); os.symlink(os.path.join(JETSON, "labtrack.py"), tmp_path / "jetson" / "labtrack.py")   # ROOT moves to tmp; the real tracker must still be found
    monkeypatch.setattr(m, "ROOT", str(tmp_path))
    monkeypatch.setattr(m, "cam_calib", lambda: object())
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m, "fetch_top", lambda: JPEG)
    return m


def drive(pl, monkeypatch, capsys, arm, argv):
    monkeypatch.setattr(pl, "find_nut", arm.find_nut); monkeypatch.setattr(pl, "api", arm.api); monkeypatch.setattr(pl, "run_plan", arm.run_plan)
    monkeypatch.setattr(sys, "argv", ["pick_loop.py", *argv]); pl.main()
    return [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]


def test_one_run_per_pick_with_params_metrics_tags_and_artifacts(pl, store, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SO101_MLFLOW_URI", store); state = tmp_path / "pick_state.json"; arm = FakeArm()
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "3", "--target", "9", "6", "--alt", "6", "5", "--state", str(state), "--gain", "0.5"])
    assert [o["run"] for o in out] == [1, 2, 3] and all(o["picked"] for o in out) and arm.frames == 3, out
    rs = runs(store, "so101-picks"); assert len(rs) == 3
    by_run = {int(r.data.tags["run"]): (r, p) for r, p in rs}
    r1, p1 = by_run[1]
    assert r1.info.status == "FINISHED" and r1.info.run_name.endswith("-r1") and r1.data.tags["host"] == "mac" and r1.data.tags["target_in"] == "6.00,5.00" and r1.data.tags["nut_in"] == "9.00,6.00"
    assert r1.data.params["target_in"] == "9.00,6.00" and r1.data.params["alt_in"] == "6.00,5.00" and r1.data.params["gain"] == "0.5" and r1.data.params["outcome_gain"] == "0.5" and r1.data.params["max_deg_s"] == "80.0"
    assert r1.data.params["state"] == "pick_state.json" and r1.data.params["runs"] == "3"
    m = r1.data.metrics
    assert m["picked"] == 1.0 and m["found_at_landing"] == 1.0 and m["landing_err_mm"] == pytest.approx(out[0]["err_mm"], abs=0.06) and m["before_mm"] > 0 and m["seconds"] >= 0
    assert {k for k in m if k.startswith("sag_")} == {"sag_hover_nut", "sag_grasp", "sag_close", "sag_lift", "sag_hover_tgt", "sag_place", "sag_open", "sag_lift2", "sag_rest"} and m["sag_grasp"] == 1.5 and m["sag_close"] == 0.0
    assert p1 == {"plan.json", "waypoints.json", "frames/landing.jpg", "state/pick_state.json"}
    assert by_run[3][0].data.tags["target_in"] == "9.00,6.00" and by_run[2][0].data.tags["target_in"] == "9.00,6.00", "even runs and the last run go to --target"


def test_a_lost_nut_still_closes_the_run(pl, store, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SO101_MLFLOW_URI", store); arm = FakeArm(stop_after=1)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "4", "--state", str(tmp_path / "s.json")])
    assert out[0]["picked"] is False and out[0]["landed_in"] is None and "stop" in out[1] and len(out) == 2
    (r, p), = runs(store, "so101-picks")
    assert r.info.status == "FINISHED" and r.data.metrics["picked"] == 0.0 and r.data.metrics["found_at_landing"] == 0.0 and "landing_err_mm" not in r.data.metrics and "frames/landing.jpg" in p


def test_ssh_failure_marks_the_run_failed_and_still_exits_like_before(pl, store, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SO101_MLFLOW_URI", store); arm = FakeArm()
    def dead(plan, max_deg_s): raise SystemExit("pick_seq produced no waypoint rows")
    monkeypatch.setattr(pl, "find_nut", arm.find_nut); monkeypatch.setattr(pl, "api", arm.api); monkeypatch.setattr(pl, "run_plan", dead)
    monkeypatch.setattr(sys, "argv", ["pick_loop.py", "--runs", "2", "--state", str(tmp_path / "s.json")])
    with pytest.raises(SystemExit, match="no waypoint rows"): pl.main()
    (r, p), = runs(store, "so101-picks"); assert r.info.status == "FAILED" and p == {"plan.json"} and r.data.params["gain"] == "0.7"


def test_output_lines_identical_with_tracking_off_and_dead_server(pl, store, tmp_path, monkeypatch, capsys):
    from mlhelpers import free_port
    outs = []
    for uri in ("", store, f"http://127.0.0.1:{free_port()}"):
        monkeypatch.setenv("SO101_MLFLOW_URI", uri); state = tmp_path / f"s{len(outs)}.json"
        lines = drive(pl, monkeypatch, capsys, FakeArm(), ["--runs", "2", "--state", str(state)])
        for l in lines: l.pop("t", None)   # the wall-clock stamp
        outs.append(lines)
    assert outs[0] == outs[1] == outs[2], "stdout does not change with tracking on, off, or failing"
    assert len(runs(store, "so101-picks")) == 2
