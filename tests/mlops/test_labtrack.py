"""jetson/labtrack.py: logs everything to a SQLite store, and is a fast, silent no-op whenever it cannot (bogus server, no
mlflow, a failure mid-run). Nothing here may take more than a few seconds or raise into the caller."""
import json, os, socket, subprocess, sys, time
import pytest
from mlhelpers import JETSON, PY, free_port, history, load, runs, sqlite_uri


# ---------------------------------------------------------------- URI resolution
def test_uri_resolution_order(lt, tmp_path, monkeypatch):
    assert lt.resolve_uri({"SO101_MLFLOW_URI": "http://a:1", "MLFLOW_TRACKING_URI": "http://b:1"}) == "http://a:1"
    assert lt.resolve_uri({"SO101_MLFLOW_URI": "", "MLFLOW_TRACKING_URI": "http://b:1"}) is None, "set-but-empty means off, whatever else is set"
    assert lt.resolve_uri({"MLFLOW_TRACKING_URI": "http://b:1"}) == "http://b:1"
    monkeypatch.setattr(lt, "MAC_DB", str(tmp_path / "nope" / "mlflow.db")); assert lt.resolve_uri({}) is None
    (tmp_path / "mlflow").mkdir(); monkeypatch.setattr(lt, "MAC_DB", str(tmp_path / "mlflow" / "mlflow.db"))
    assert lt.resolve_uri({}) == "sqlite:///" + str(tmp_path / "mlflow" / "mlflow.db"), "the Mac default only when the folder exists"


def test_artifact_root_sits_beside_the_db(lt, tmp_path):
    assert lt.artifact_root(sqlite_uri(tmp_path), "so101-picks") == str(tmp_path / "artifacts" / "so101-picks")
    assert lt.artifact_root("http://127.0.0.1:5050", "x") is None, "the server decides for http"


def test_key_sanitising(lt):
    assert lt.key("sag hover_nut") == "sag hover_nut" and lt.key("a/b.c-d") == "a/b.c-d" and lt.key("weird:key!") == "weird_key_"


# ---------------------------------------------------------------- the happy path on SQLite
def test_sqlite_run_logs_params_metrics_tags_and_artifacts(lt, store, tmp_path, capsys):
    t = lt.LabTrack("so101-test", store); assert t.enabled and t.why == ""
    f = tmp_path / "frame.jpg"; f.write_bytes(b"\xff\xd8jpeg")
    with t.run("r1", tags={"kind": "unit", "run": 3}) as r:
        assert r.enabled and r.id
        r.params({"target": "9,6", "gain": 0.7, "none_skipped": None, "sag hover": 1})
        r.metrics({"landing_err_mm": 12.3, "picked": True, "skip_none": None, "skip_text": "x", "skip_nan": float("nan")})
        for step, v in enumerate([0.5, 0.4, 0.3]): r.metrics({"loss": v}, step=step)
        r.artifact(str(f), "frames"); r.json({"waypoints": [1, 2]}, "plan.json"); r.file("note.txt", "hello", "notes"); r.tags({"later": "yes"})
        r.artifact(str(tmp_path / "missing.jpg"), "frames")   # silently skipped
    (run, paths), = runs(store, "so101-test")
    assert run.info.run_name == "r1" and run.info.status == "FINISHED"
    assert run.data.params == {"target": "9,6", "gain": "0.7", "sag hover": "1"}
    assert run.data.metrics == {"landing_err_mm": 12.3, "picked": 1.0, "loss": 0.3}
    assert history(store, run.info.run_id, "loss") == [(0, 0.5), (1, 0.4), (2, 0.3)]
    assert run.data.tags["kind"] == "unit" and run.data.tags["run"] == "3" and run.data.tags["later"] == "yes" and run.data.tags["host"] == socket.gethostname().split(".")[0]
    assert paths == {"frames/frame.jpg", "plan.json", "notes/note.txt"}
    assert run.info.artifact_uri.startswith(str(tmp_path / "mlflow" / "artifacts" / "so101-test")), run.info.artifact_uri
    assert capsys.readouterr().err == "", "nothing on stderr when everything works"


def test_exception_inside_the_block_marks_the_run_failed_and_propagates(lt, store):
    t = lt.LabTrack("so101-test", store)
    with pytest.raises(ValueError, match="boom"):
        with t.run("r2") as r: r.metrics({"a": 1}); raise ValueError("boom")
    (run, _), = runs(store, "so101-test"); assert run.info.status == "FAILED" and run.data.metrics == {"a": 1.0}


def test_attach_continues_a_run_from_another_process(lt, store):
    r = lt.LabTrack("so101-test", store).start("two-phase"); rid = r.id; r.params({"phase": "start"})
    r2 = lt.LabTrack("so101-test", store).attach(rid); r2.metrics({"episodes": 4}); r2.end("FINISHED")
    (run, _), = runs(store, "so101-test"); assert run.info.run_id == rid and run.info.status == "FINISHED" and run.data.params == {"phase": "start"} and run.data.metrics == {"episodes": 4.0}
    assert lt.LabTrack("so101-test", "").attach(rid).enabled is False, "attach on a disabled tracker is a no-op Run"


def test_same_experiment_is_reused_not_duplicated(lt, store):
    from mlflow import MlflowClient
    for i in range(2):
        with lt.LabTrack("so101-test", store).run(f"r{i}"): pass
    assert len(MlflowClient(tracking_uri=store).search_experiments()) == 2 and len(runs(store, "so101-test")) == 2   # Default + ours


# ---------------------------------------------------------------- degrading
def test_bogus_http_server_returns_fast_and_never_raises(lt, capsys):
    uri = f"http://127.0.0.1:{free_port()}"; t0 = time.monotonic()
    t = lt.LabTrack("so101-test", uri)
    with t.run("r", tags={"a": 1}) as r:
        r.params({"x": 1}); r.metrics({"y": 2}); r.json({"z": 3}, "z.json"); r.file("f.txt", b"ff"); r.tags({"t": 1})
    dt = time.monotonic() - t0
    assert dt < 3.0, f"a dead server must not hold the arm ({dt:.2f} s)"
    assert not t.enabled and not r.enabled and r.id is None and "not reachable" in t.why
    err = capsys.readouterr().err; assert err.count("labtrack:") == 1 and "tracking off" in err, err


def test_probe_timeout_path_is_bounded(lt, monkeypatch, capsys):
    """A black-holed address (packets dropped, no refusal) is the common case when the Mac is asleep: the 1 s probe bounds it."""
    def hang(addr, timeout=None): time.sleep(min(timeout, 0.3)); raise TimeoutError("timed out")
    monkeypatch.setattr(lt.socket, "create_connection", hang)
    t0 = time.monotonic(); t = lt.LabTrack("so101-test", "http://192.0.2.1:5050", probe_s=0.3)   # RFC 5737 documentation address; the connect is monkeypatched anyway
    with t.run("r") as r: r.metrics({"a": 1})
    assert time.monotonic() - t0 < 3.0 and not t.enabled and "within 0.3 s" in t.why
    assert capsys.readouterr().err.count("labtrack:") == 1


def test_no_uri_is_silent(lt, capsys):
    t = lt.LabTrack("so101-test", "")
    with t.run("r") as r: r.metrics({"a": 1})
    assert not t.enabled and t.why == "no tracking URI" and capsys.readouterr().err == "", "a machine with no server configured should not nag"


def test_missing_mlflow_module_is_a_noop(lt, store, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlflow", None)   # makes `import mlflow` raise ImportError
    t = lt.LabTrack("so101-test", store)
    with t.run("r") as r: r.params({"a": 1}); r.metrics({"b": 2}); r.json({}, "x.json")
    assert not t.enabled and "mlflow" in t.why and "Error" in t.why and capsys.readouterr().err.count("labtrack:") == 1


def test_failure_mid_run_logs_once_then_goes_silent(lt, store, capsys):
    t = lt.LabTrack("so101-test", store)
    with t.run("r") as r:
        r.metrics({"a": 1})
        def boom(*a, **k): raise RuntimeError("server went away")
        t._client.log_batch = boom
        r.metrics({"b": 2}); r.metrics({"c": 3}); r.params({"d": 4}); r.json({"e": 5}, "e.json")
    assert not t.enabled and "server went away" in t.why
    err = capsys.readouterr().err; assert err.count("labtrack:") == 1, err
    (run, _), = runs(store, "so101-test"); assert run.data.metrics == {"a": 1.0} and run.info.status == "RUNNING", "end_run was skipped too (tracking is off); the run stays open server-side, which is honest"


def test_keyboard_interrupt_is_not_swallowed(lt, store):
    t = lt.LabTrack("so101-test", store)
    def boom(*a, **k): raise KeyboardInterrupt()
    t._client.log_batch = boom
    with pytest.raises(KeyboardInterrupt):
        with t.run("r") as r: r.metrics({"a": 1})


# ---------------------------------------------------------------- CLI self-test
def test_selftest_cli(store, tmp_path):
    r = subprocess.run([PY, os.path.join(JETSON, "labtrack.py"), store], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    head, tail = (json.loads(l) for l in r.stdout.strip().splitlines())
    assert head["enabled"] and head["uri"] == store and tail["experiment"] == "so101-labtrack-selftest"
    (run, paths), = runs(store, "so101-labtrack-selftest"); assert run.info.run_id == tail["run_id"] and paths == {"hello.json"} and run.data.metrics == {"ok": 1.0}
    r = subprocess.run([PY, os.path.join(JETSON, "labtrack.py"), f"http://127.0.0.1:{free_port()}"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and json.loads(r.stdout.splitlines()[0])["enabled"] is False and "tracking off" in r.stderr
