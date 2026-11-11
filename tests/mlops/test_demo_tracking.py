"""MLflow side of scripted-demo recording on a real SQLite store: the cycle runs carry the dataset link, a dataset-version run lands in its own experiment
(so101-datasets-dry for the dry run, like so101-autoloop-dry), and with no tracking server everything still works (jetson/labtrack.py never raises)."""
import json, os
import pytest

pytest.importorskip("lerobot")   # why: these tests write and read real LeRobot datasets; CI installs only requirements-dev.txt (no lerobot), so the file skips there
from mlhelpers import ROOT, TOOLS, runs, free_port, store  # noqa: F401


def run_cli(al, tmp_path, capsys, cycles=3):
    out = tmp_path / "out"; rc = al.main(["--dry-run", "--record-demos", "--cycles", str(cycles), "--seed", "2", "--session", "trk", "--out-dir", str(out), "--dry-miss", "0", "--settle", "0", "--stop-file", str(tmp_path / "none.stop")])
    return rc, out, json.load(open(out / "autoloop_trk.json"))


@pytest.fixture(scope="module")
def al():
    from mlhelpers import load
    return load(os.path.join(TOOLS, "autoloop.py"), "autoloop_for_demo_tracking")


def test_cycle_runs_link_the_dataset_and_a_version_run_lands_in_so101_datasets(al, store, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SO101_MLFLOW_URI", store); rc, out, s = run_cli(al, tmp_path, capsys); assert rc == 0 and s["demos"]["kept"] == 3
    cyc = {r.info.run_name: (r, p) for r, p in runs(store, "so101-autoloop-dry") if r.data.tags.get("kind") != "session"}
    assert {"autoloop-trk-c1", "autoloop-trk-c2", "autoloop-trk-c3"} <= set(cyc)
    r1, p1 = cyc["autoloop-trk-c1"]; assert r1.data.params["kept"] == "1" and r1.data.params["dataset_repo"] == "lastlight/so101-nut-pick" and r1.data.params["config_id"] == "stock-v1" and r1.data.params["tier"] == "A"
    assert r1.data.metrics["episode_index"] == 0.0 and r1.data.metrics["episode_frames"] > 30 and "episode_meta.json" in p1
    links = [json.loads(l) for l in open(out / "demos" / "trk" / "so101-nut-pick" / "meta" / "lab" / "links.jsonl")]
    assert [x["mlflow_cycle_run_id"] for x in links] == [cyc[f"autoloop-trk-c{i}"][0].info.run_id for i in (1, 2, 3)], "the run ids are written back once the cycle runs exist"
    assert all(x["mlflow_session_run_id"] for x in links)
    row = json.loads(open(out / "demos" / "trk" / "so101-nut-pick" / "meta" / "lab" / "episodes.jsonl").readline()); assert row["links"]["mlflow_cycle_run_name"] == "autoloop-trk-c1" and row["links"]["mlflow_session_run_id"] == links[0]["mlflow_session_run_id"]
    (v, paths), = runs(store, "so101-datasets-dry"); P = v.data.params
    assert v.data.tags["kind"] == "dataset-version" and P["episodes_total"] == "3" and P["config_stock-v1"] == "3" and P["fps"] == "30" and P["dataset_hash"] == s["demos"]["dataset_hash"] and P["repo_id"] == "lastlight/so101-nut-pick"
    assert v.data.metrics["kept"] == 3.0 and v.data.metrics["rejected"] == 0.0 and v.data.metrics["frames_total"] == s["demos"]["frames_total"] and len(P["features_hash"]) == 12
    assert {"dataset/info.json", "dataset/episodes.jsonl", "rejects_summary.json"} <= paths


def test_no_tracking_server_changes_nothing_but_the_logging(al, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SO101_MLFLOW_URI", f"http://127.0.0.1:{free_port()}"); rc, out, s = run_cli(al, tmp_path, capsys, cycles=2)
    assert rc == 0 and s["demos"]["kept"] == 2 and s["demos"]["dataset_hash"] and os.path.exists(out / "demos" / "trk" / "so101-nut-pick" / "meta" / "info.json")
    assert not os.path.exists(out / "demos" / "trk" / "so101-nut-pick" / "meta" / "lab" / "links.jsonl") or all(json.loads(l)["mlflow_cycle_run_id"] is None for l in open(out / "demos" / "trk" / "so101-nut-pick" / "meta" / "lab" / "links.jsonl"))
