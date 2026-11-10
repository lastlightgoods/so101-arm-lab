"""jetson/cam_calib.py's MLflow logging: table, campose, armfit and handeye each leave one run in `so101-calibration` with the
written JSON attached and the same numbers the command printed; dry commands (check, handeye --check, locate) log nothing;
with tracking off the files and stdout are identical."""
import json, math, os, subprocess, sys
import pytest
from mlhelpers import JETSON, PY, ROOT, runs

cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(ROOT, "tests", "calib")); import synth  # noqa: E402  (the synthetic camera the calib tests use)
SCRIPT = os.path.join(JETSON, "cam_calib.py")
IN = 25.4


def run(uri, *args):
    env = dict(os.environ, SO101_MLFLOW_URI=uri)
    return subprocess.run([PY, SCRIPT, *map(str, args)], capture_output=True, text=True, timeout=600, env=env)


@pytest.fixture
def scene(tmp_path):
    """A rendered ArUco table with its layout and true intrinsics written under tmp_path, the pattern of tests/calib."""
    layout = synth.default_layout(); rvec, tvec = synth.table_camera_pose()
    d = tmp_path / "scene"; d.mkdir(); cv2.imwrite(str(d / "table.png"), synth.render_table(layout, rvec, tvec))
    json.dump({**layout, "markers": {str(k): v for k, v in layout["markers"].items()}}, open(tmp_path / "top_layout.json", "w"))
    json.dump({"image_size": [synth.W, synth.H], "camera_matrix": synth.K_TRUE.tolist(), "dist_coeffs": synth.DIST_TRUE.tolist()}, open(tmp_path / "top_intrinsics.json", "w"))
    return {"dir": d, "layout": layout, "rvec": rvec, "tvec": tvec}


def test_table_logs_one_run_with_residuals_and_the_files(store, tmp_path, scene):
    r = run(store, "--calib-dir", tmp_path, "table", "--source", scene["dir"], "--busy-flag", tmp_path / "flag"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); saved = json.load(open(out["saved"]))
    (m, paths), = runs(store, "so101-calibration")
    assert m.info.status == "FINISHED" and m.data.tags["kind"] == "table" and m.data.tags["camera"] == "top" and m.info.run_name.startswith("table-top-")
    assert m.data.params["markers"] == "0,1,2,3,4,5" and m.data.params["units"] == "mm" and m.data.params["camera"] == "top" and m.data.params["layout"] == str(tmp_path / "top_layout.json")
    assert m.data.params["intrinsics"] == str(tmp_path / "top_intrinsics.json") and m.data.params["image_size"] == "640x360" and m.data.params["out"] == out["saved"]
    for k in saved["residual_mm"]: assert m.data.metrics[f"residual_mm_{k}"] == saved["residual_mm"][k]
    assert m.data.metrics["residual_mm_max"] == saved["residual_mm_max"] == pytest.approx(out["residual_mm_max"], abs=0.006) and m.data.metrics["reproj_rms_px"] == saved["reproj_rms_px"]
    assert m.data.metrics["markers"] == 6 and m.data.metrics["points_used"] == 24 and m.data.metrics["camera_height_mm"] == saved["camera_position_mm"][2]
    assert paths == {"calib/top_table.json", "calib/top_table_check.png"}
    assert "labtrack" not in r.stderr and "tracking skipped" not in r.stderr


def test_table_without_intrinsics_logs_no_reproj_metric(store, tmp_path, scene):
    os.remove(tmp_path / "top_intrinsics.json")
    r = run(store, "--calib-dir", tmp_path, "table", "--source", scene["dir"]); assert r.returncode == 0, r.stderr
    (m, _), = runs(store, "so101-calibration"); assert m.data.params["intrinsics"] == "none" and "reproj_rms_px" not in m.data.metrics and "camera_height_mm" not in m.data.metrics and m.data.metrics["residual_mm_max"] < 1.5


def test_campose_armfit_and_check_locate(store, tmp_path, scene):
    r = run(store, "--calib-dir", tmp_path, "table", "--source", scene["dir"]); assert r.returncode == 0, r.stderr
    true_hfov = math.degrees(2 * math.atan((synth.W / 2) / synth.K_TRUE[0, 0]))
    r = run(store, "--calib-dir", tmp_path, "campose", "--source", scene["dir"], "--hfov", f"{true_hfov:.3f}"); assert r.returncode == 0, r.stderr
    cp = json.loads(r.stdout)
    # armfit: tip positions that the true transform (base at (200,330) facing -Y, from the layout) would give
    import importlib.util
    spec = importlib.util.spec_from_file_location("cam_calib", SCRIPT); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    truth = {"arm_base_mm": scene["layout"]["arm_base_mm"], "arm_yaw_deg": scene["layout"]["arm_yaw_deg"]}
    pairs = {"arm_units": "mm", "pairs": [{"marker": i, "arm": list(cc.table_to_arm(truth, *scene["layout"]["markers"][i]))} for i in range(4)]}
    pp = tmp_path / "pairs.json"; json.dump(pairs, open(pp, "w"))
    r = run(store, "--calib-dir", tmp_path, "armfit", "--pairs", pp); assert r.returncode == 0, r.stderr
    af = json.loads(r.stdout)
    # dry commands: nothing logged
    r = run(store, "--calib-dir", tmp_path, "check", "--source", scene["dir"]); assert r.returncode == 0, r.stderr
    r = run(store, "--calib-dir", tmp_path, "locate", "300", "180"); assert r.returncode == 0, r.stderr
    rs = {m.data.tags["kind"]: (m, p) for m, p in runs(store, "so101-calibration")}
    assert set(rs) == {"table", "campose", "armfit"}, "check and locate write nothing, so they log nothing"
    m, p = rs["campose"]
    assert m.data.metrics["hfov_deg"] == pytest.approx(true_hfov, abs=1e-3) and m.data.metrics["height_mm"] == pytest.approx(cp["height_mm"], abs=0.006) and m.data.metrics["tilt_deg"] == pytest.approx(cp["tilt_deg"], abs=0.06)
    assert m.data.metrics["reproj_rms_px"] == pytest.approx(cp["reproj_rms_px"], abs=0.006) and m.data.params["hfov_source"].startswith("nominal hfov") and m.data.params["units"] == "mm" and "lens_height" not in m.data.params   # None params are dropped
    assert p == {"calib/top_table.json"}, "campose writes into the table file when it exists"
    m, p = rs["armfit"]
    assert m.data.metrics["pairs"] == 4 and m.data.metrics["residual_mm_max"] == pytest.approx(af["residual_mm_max"], abs=0.006) and m.data.metrics["arm_yaw_deg"] == pytest.approx(af["arm_yaw_deg"], abs=0.006)
    assert m.data.metrics["arm_base_x_mm"] == pytest.approx(af["arm_base_mm"][0], abs=0.006) and {k for k in m.data.metrics if k.startswith("residual_mm_")} == {"residual_mm_0", "residual_mm_1", "residual_mm_2", "residual_mm_3", "residual_mm_max"}
    assert m.data.params["pairs_file"] == str(pp) and m.data.params["markers"] == "0,1,2,3,4,5" and p == {"calib/top_table.json", "calib/pairs.json"}


def test_tracking_off_or_dead_server_changes_nothing(tmp_path, scene):
    from mlhelpers import free_port
    outs = []
    for uri in ("", f"http://127.0.0.1:{free_port()}"):
        r = run(uri, "--calib-dir", tmp_path, "table", "--source", scene["dir"]); assert r.returncode == 0, r.stderr
        t = json.load(open(tmp_path / "top_table.json")); t.pop("created"); outs.append((r.stdout, t))
        assert ("tracking off" in r.stderr) == bool(uri)
    assert outs[0] == outs[1]


def test_handeye_logs_the_wrist_pose(store, tmp_path):
    """The synthetic wrist views come from tests/calib/test_handeye.py's generator; reuse it rather than re-deriving the geometry."""
    pytest.importorskip("scipy")
    import importlib.util
    spec = importlib.util.spec_from_file_location("cam_calib", SCRIPT); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    th = importlib.util.spec_from_file_location("calib_test_handeye", os.path.join(ROOT, "tests", "calib", "test_handeye.py")); th_mod = importlib.util.module_from_spec(th); th.loader.exec_module(th_mod)
    d = tmp_path; views = d / "wristcam"; views.mkdir()
    (d / "top_layout.json").write_text(json.dumps(th_mod.LAYOUT_FILE)); (d / "top_table.json").write_text(json.dumps(th_mod.CALIB))
    layout = cc.load_layout(str(d / "top_layout.json")); R_X, t_X = th_mod.true_X(cc)
    for i, j in enumerate(th_mod.JOINTS[:5]):
        cv2.imwrite(str(views / f"s{i + 1}.png"), th_mod.render_view(cc, R_X, t_X, j, layout)); (views / f"s{i + 1}.json").write_text(json.dumps({"name": f"s{i + 1}", "joints": {**j, "gripper": 0.0}}))
    r = run(store, "--calib-dir", d, "handeye", "--no-loo", "--only", "s1,s2,s3,s4,s5"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); pose = json.load(open(d / "wrist_campose.json"))
    r = run(store, "--calib-dir", d, "handeye", "--check", "s3"); assert r.returncode == 0, r.stderr
    (m, p), = runs(store, "so101-calibration")
    assert m.data.tags["kind"] == "handeye" and m.data.tags["camera"] == "wrist" and p == {"calib/wrist_campose.json"}, "--check logs nothing"
    assert m.data.params["views"] == "s1,s2,s3,s4,s5" and m.data.params["intrinsics"] == "none" and m.data.params["pose_sigma_deg"] == "1.0" and m.data.params["views_dir"] == str(d / "wristcam")
    for k in ("reproj_rms_px", "reproj_rms_px_corrected", "off_axis_mm", "tilt_from_gripper_axis_deg", "hfov_deg"): assert m.data.metrics[k] == pose[k], k
    assert m.data.metrics["views"] == 5 and m.data.metrics["markers"] == len(out["markers_used"]) and all(f"per_view_px_{v}" in m.data.metrics for v in out["views_used"]) and "leave_one_out_px_s1" not in m.data.metrics
