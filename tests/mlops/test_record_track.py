"""jetson/record_track.py through a sourced jetson/env.sh: so101-record / so101-record-more leave one run in `so101-recordings`
with the dataset, task, camera lock state and episode count, and never change what the shell functions did before
(refusal when unlocked, guard started, busy flags released, lerobot-record's argv)."""
import json, os, time
import pytest
from mlhelpers import JETSON, load, runs


def test_so101_record_logs_a_run_around_lerobot_record(rshell):
    rshell.lock(); (rshell.so / "outputs" / "run").mkdir(parents=True)
    health = rshell.so / "outputs" / "run" / "cam_health.json"
    json.dump({"ok": True, "cameras": {"top": {"stats": {"fps": 29.9}}, "front": {"stats": {"fps": 30.0}}, "wrist": {"stats": {"fps": 29.8}}}}, open(health, "w"))
    rc, out = rshell.run('so101-record cube "pick the cube" 2')
    assert rc == 0, out
    assert "--dataset.repo_id=lastlight/cube" in rshell.argv() and "--dataset.num_episodes=2" in rshell.argv(), "lerobot-record ran exactly as before"
    assert rshell.busy() == [] and rshell.invocations()[:2] == ["check --strict", "guard"] and "check --json" in rshell.invocations()
    assert not rshell.state_file().exists(), "finish cleans up the hand-off file"
    (r, paths), = runs(rshell.uri, "so101-recordings")
    assert r.info.status == "FINISHED" and r.info.run_name.startswith("record-cube-") and r.data.tags["kind"] == "record" and r.data.tags["dataset"] == "cube" and r.data.tags["resume"] == "0"
    p = r.data.params
    assert p["repo_id"] == "lastlight/cube" and p["task"] == "pick the cube" and p["episodes_requested"] == "2" and p["resume"] == "False" and p["episodes_before"] == "0"
    assert p["cameras"] == "top,front,wrist" and p["fps"] == "30" and p["episode_time_s"] == "20" and p["reset_time_s"] == "10", "from SO101_CAMS / SO101_EP_S in env.sh"
    assert p["cam_top_exposure_time_absolute"] == "156" and p["cam_top_gain"] == "64" and p["cam_wrist_exposure_time_absolute"] == "313" and p["cam_settings_captured"] == "2026-11-09T23:06:42"
    m = r.data.metrics
    assert m["cams_locked"] == 1.0 and m["cam_health_ok"] == 1.0 and m["cam_health_fps_top"] == 29.9 and m["cam_health_age_s"] < 60
    assert m["episodes_total"] == 2.0 and m["episodes_new"] == 2.0 and m["rc"] == 0.0 and m["ok"] == 1.0 and 0 <= m["duration_s"] < 60
    assert paths == {"cameras/cam_settings.json", "cameras/cam_check.json", "cameras/cam_health.json", "dataset/info.json"}


def test_record_more_resumes_and_counts_new_episodes(rshell):
    rshell.lock(); root = rshell.home / ".cache" / "huggingface" / "lerobot" / "lastlight" / "cube" / "meta"; root.mkdir(parents=True)
    json.dump({"total_episodes": 5, "fps": 30}, open(root / "info.json", "w")); rshell.set_recorder(episodes=8)
    rc, out = rshell.run("so101-record-more cube 3"); assert rc == 0, out
    assert "--resume=true" in rshell.argv()
    (r, _), = runs(rshell.uri, "so101-recordings")
    assert r.data.tags["resume"] == "1" and r.data.params["episodes_before"] == "5" and r.data.params["root"] == str(root.parent) and r.data.params["task"].startswith("Pick up the lighter")
    assert r.data.metrics["episodes_total"] == 8.0 and r.data.metrics["episodes_new"] == 3.0


def test_a_failed_recording_is_a_failed_run(rshell):
    rshell.lock(); rshell.set_recorder(episodes=1, rc=3)
    rc, out = rshell.run('so101-record cube "pick the cube" 2'); assert rc == 0, out   # the function's status is the release, as before
    (r, _), = runs(rshell.uri, "so101-recordings"); assert r.info.status == "FAILED" and r.data.metrics["rc"] == 3.0 and r.data.metrics["ok"] == 0.0 and r.data.metrics["episodes_total"] == 1.0
    assert rshell.busy() == []


def test_stale_health_report_is_not_attached(rshell):
    rshell.lock(); (rshell.so / "outputs" / "run").mkdir(parents=True); health = rshell.so / "outputs" / "run" / "cam_health.json"
    json.dump({"ok": False}, open(health, "w")); old = time.time() - 3600; os.utime(health, (old, old))
    rc, out = rshell.run('so101-record cube "pick the cube" 1'); assert rc == 0, out
    (r, paths), = runs(rshell.uri, "so101-recordings"); assert "cam_health_ok" not in r.data.metrics and "cameras/cam_health.json" not in paths


def test_refusal_when_unlocked_logs_nothing(rshell):
    rc, out = rshell.run('so101-record cube "pick the cube" 2')
    assert rc != 0 and "REFUSING TO RECORD" in out and not rshell.called.exists() and runs(rshell.uri, "so101-recordings") == [] and not rshell.state_file().exists()


def test_dead_server_does_not_slow_or_fail_the_recording(rshell):
    from mlhelpers import free_port
    rshell.lock(); rshell.env["SO101_MLFLOW_URI"] = f"http://127.0.0.1:{free_port()}"; t0 = time.monotonic()
    rc, out = rshell.run('so101-record cube "pick the cube" 2')
    assert rc == 0 and time.monotonic() - t0 < 15, out
    assert "--dataset.num_episodes=2" in rshell.argv() and out.count("labtrack: tracking off") == 1 and '"tracking": false' in out and rshell.busy() == []
    assert not rshell.state_file().exists()


def test_no_record_track_on_the_box_is_fine(rshell):
    """A Jetson whose ~/so101 predates this file: so101-track is a no-op."""
    rshell.lock(); os.remove(rshell.so / "record_track.py")
    rc, out = rshell.run('so101-record cube "pick the cube" 2'); assert rc == 0 and "--dataset.num_episodes=2" in rshell.argv() and "record_track" not in out


def test_mlflow_uri_file_is_exported_by_env_sh(rshell):
    (rshell.so / ".mlflow_uri").write_text("http://mac.local:5050\n")
    del rshell.env["SO101_MLFLOW_URI"]
    rc, out = rshell.run('echo "$SO101_MLFLOW_URI"'); assert rc == 0 and out.strip() == "http://mac.local:5050"
    (rshell.so / ".mlflow_uri").write_text("")
    rc, out = rshell.run('echo "[$SO101_MLFLOW_URI]"'); assert out.strip() == "[]", "an empty file exports nothing"


def test_finish_without_start_and_cams_from_env(tmp_path, monkeypatch, capsys):
    rt = load(os.path.join(JETSON, "record_track.py"))
    assert rt.cams_from_env("{top: {type: opencv, index_or_path: /dev/x, width: 640, height: 360, fps: 30, fourcc: MJPG}, wrist: {type: opencv, fps: 30}}") == (["top", "wrist"], 30)
    assert rt.cams_from_env("") == ([], None)
    monkeypatch.setattr(rt, "STATE", str(tmp_path / "none.json"))
    assert rt.main(["finish", "--rc", "0"]) == 0 and json.loads(capsys.readouterr().out)["why"] == "no run was started"
    monkeypatch.setattr(rt, "cmd_start", lambda a: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert rt.main(["start", "--repo-id", "x/y"]) == 0 and "disk full" in capsys.readouterr().err, "bookkeeping never fails the shell"
