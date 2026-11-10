"""Fixtures for the MLflow tracking layer (jetson/labtrack.py, jetson/record_track.py, the pick-loop and cam_calib loggers).
Everything logs to a temp SQLite store under tmp_path; nothing talks to the network (the "server down" cases use a closed
localhost port) and nothing touches TC's real ~/so101/mlflow store (tests/conftest.py blanks SO101_MLFLOW_URI)."""
import importlib.util, json, os, socket, subprocess, sys, textwrap
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JETSON = os.path.join(ROOT, "jetson")
TOOLS = os.path.join(ROOT, "tools")
PY = sys.executable
ENV_SH = os.path.join(JETSON, "env.sh")
pytest.importorskip("mlflow", reason="the tracking tests need mlflow (pip install mlflow)")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def load(path, name=None):
    """Import a script by path (the house pattern: nothing under jetson/ or tools/ is a package)."""
    name = name or os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def sqlite_uri(folder): return "sqlite:///" + os.path.join(str(folder), "mlflow.db")


def free_port():
    """A localhost port nothing listens on: connecting to it is refused at once (no network, no DNS)."""
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


def runs(uri, experiment):
    """[(run, {artifact paths})] for every run in `experiment`, newest first."""
    from mlflow import MlflowClient
    c = MlflowClient(tracking_uri=uri); e = c.get_experiment_by_name(experiment)
    if e is None: return []
    out = []
    for r in c.search_runs([e.experiment_id], order_by=["attributes.start_time DESC"]):
        paths = set()
        for a in c.list_artifacts(r.info.run_id):
            paths.update(f"{a.path}/{b.path.split('/')[-1]}" for b in c.list_artifacts(r.info.run_id, a.path)) if a.is_dir else paths.add(a.path)
        out.append((r, paths))
    return out


def history(uri, run_id, key):
    from mlflow import MlflowClient
    return [(m.step, m.value) for m in MlflowClient(tracking_uri=uri).get_metric_history(run_id, key)]


@pytest.fixture(scope="session")
def lt(): return load(os.path.join(JETSON, "labtrack.py"))


@pytest.fixture
def store(tmp_path):
    """A fresh SQLite tracking URI (the Mac's layout: mlflow.db + artifacts/ beside it)."""
    d = tmp_path / "mlflow"; d.mkdir(); return sqlite_uri(d)


class RecordShell:
    """A fake $HOME/so101 that lets jetson/env.sh be sourced on the Mac and `so101-record` run end to end without hardware:
    an empty venv activate, the REAL record_track.py + labtrack.py copied in (they are what is under test), a stub cam_lock.py
    (`check --strict` passes while ~/so101/.locked exists; `check --json` prints a fake report; `guard` is a no-op), a
    cam_settings.json with locked controls, and a PATH shim lerobot-record that records its argv and writes the dataset's
    meta/info.json like the real one would. SO101_MLFLOW_URI points at a temp SQLite store."""
    def __init__(self, tmp_path, uri):
        self.home = tmp_path / "home"; self.so = self.home / "so101"; (self.so / ".venv" / "bin").mkdir(parents=True)
        (self.so / ".venv" / "bin" / "activate").write_text("")
        for f in ("record_track.py", "labtrack.py"): (self.so / f).write_bytes(open(os.path.join(JETSON, f), "rb").read())
        self.calls = self.so / "calls.log"
        (self.so / "cam_lock.py").write_text(textwrap.dedent(f"""
            import json, os, sys
            open({str(self.calls)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
            if sys.argv[1] == "guard": sys.exit(0)
            ok = os.path.exists(os.path.expanduser("~/so101/.locked"))
            if sys.argv[1:] == ["check", "--json"]: print(json.dumps({{"ok": ok, "cameras": {{"top": {{"drift": {{}}}}, "front": {{"drift": {{}}}}, "wrist": {{"drift": {{}}}}}}}})); sys.exit(0 if ok else 1)
            assert sys.argv[1:3] == ["check", "--strict"], sys.argv
            print("cams locked: top ok; front ok; wrist ok" if ok else "cams NOT locked (check): front auto_exposure=3 (want 1)")
            sys.exit(0 if ok else 1)
        """))
        (self.so / "calib").mkdir()
        json.dump({"captured": "2026-11-09T23:06:42", "cameras": {"usb-top-video-index0": {"role": "top", "kind": "c920", "controls": {"auto_exposure": 1, "exposure_time_absolute": 156, "gain": 64, "focus_absolute": 30}},
                                                                   "usb-wrist-video-index0": {"role": "wrist", "kind": "icspring", "controls": {"auto_exposure": 1, "exposure_time_absolute": 313}}}},
                  open(self.so / "calib" / "cam_settings.json", "w"))
        self.bin = tmp_path / "bin"; self.bin.mkdir(); self.called = tmp_path / "lerobot-record.argv"
        self.set_recorder(episodes=2, rc=0)
        self.env = dict(os.environ, HOME=str(self.home), PATH=f"{self.bin}:{os.path.dirname(PY)}:/usr/bin:/bin", SO101_MLFLOW_URI=uri, SO101_EP_S="20")
        self.uri = uri

    def set_recorder(self, episodes, rc=0, sleep=0.0):
        """lerobot-record shim: records argv, writes meta/info.json with `episodes` total episodes into the dataset root, exits `rc`."""
        (self.bin / "lerobot-record").write_text(textwrap.dedent(f"""\
            #!/bin/sh
            printf "%s\\n" "$@" > "{self.called}"
            repo=$(printf "%s\\n" "$@" | sed -n 's/^--dataset.repo_id=//p'); root="$HOME/.cache/huggingface/lerobot/$repo"
            mkdir -p "$root/meta"; sleep {sleep}; printf '{{"total_episodes": {episodes}, "fps": 30, "codebase_version": "v3.0"}}' > "$root/meta/info.json"
            exit {rc}
            """)); (self.bin / "lerobot-record").chmod(0o755)

    def run(self, cmd):
        r = subprocess.run(["bash", "-c", f"source {ENV_SH}; {cmd}"], env=self.env, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def lock(self): (self.so / ".locked").write_text("")
    def busy(self): return [p.name for p in self.so.iterdir() if p.name in (".cams_busy", ".arms_busy")]
    def invocations(self): return self.calls.read_text().splitlines() if self.calls.exists() else []
    def argv(self): return self.called.read_text().splitlines() if self.called.exists() else []
    def state_file(self): return self.so / "outputs" / "run" / "record_track.json"


@pytest.fixture
def rshell(tmp_path, store): return RecordShell(tmp_path, store)
