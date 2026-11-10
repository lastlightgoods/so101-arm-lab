"""Fixtures for tools/so101_policy.py (the model registry and deployment path) and the `so101-policy-current` resolver in jetson/env.sh.

Nothing here reaches a network, a Jetson or the lab's MLflow store:
  * ``tool`` imports tools/so101_policy.py once (skipped when mlflow or rsync is missing, as on CI, see requirements-dev.txt);
  * ``lab`` points MLFLOW_TRACKING_URI at a fresh SQLite file under tmp_path with its own artifact root, builds a fake
    pretrained_model dir (train_config.json / config.json / model.safetensors of deterministic bytes), and sets
    SO101_DEPLOY_LOCAL_ROOT so the tool's ssh becomes ``bash -s`` with that directory as $HOME and rsync copies locally;
    SO101_JETSON is an unresolvable host and HF_HUB_OFFLINE=1 so a forgotten override could not go anywhere;
  * ``shell`` (for the env.sh tests) is a fake $HOME/so101 with an empty venv activate and a PATH shim ``lerobot-rollout``
    that records its argv instead of moving the arm.
"""
import contextlib, importlib.util, io, json, os, shutil, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "tools", "so101_policy.py")
JETSON_ENV = os.path.join(ROOT, "jetson", "env.sh")
BOGUS_JETSON = "so101-test-no-such-host.invalid"
TRAIN_CONFIG = {"policy": {"type": "act", "chunk_size": 100, "n_action_steps": 100, "optimizer_lr": 1e-05, "dim_model": 512, "use_vae": True, "device": "mps", "pretrained_path": None},
                "batch_size": 8, "steps": 5000, "seed": 1000, "dataset": {"repo_id": "lastlight/fake-set"}}


def make_ckpt(d, seed=1, kind="act", step=None, train_config=None):
    """A pretrained_model dir at d (or d/checkpoints/<step>/pretrained_model with step): the three files the tool requires plus a processor json."""
    if step: d = os.path.join(d, "checkpoints", step, "pretrained_model")
    os.makedirs(d, exist_ok=True)
    cfg = json.loads(json.dumps(train_config or TRAIN_CONFIG)); cfg["policy"]["type"] = kind
    if step: cfg["steps"] = int(step)
    json.dump(cfg, open(os.path.join(d, "train_config.json"), "w"))
    json.dump({"type": kind, "chunk_size": 100}, open(os.path.join(d, "config.json"), "w"))
    with open(os.path.join(d, "model.safetensors"), "wb") as f: f.write(bytes([(seed * 7 + i) % 256 for i in range(4096)]))
    json.dump({"steps": []}, open(os.path.join(d, "policy_preprocessor.json"), "w"))
    return d


@pytest.fixture(scope="session")
def tool():
    pytest.importorskip("mlflow", reason="the registry tests need mlflow (not in requirements-dev.txt)")
    if not shutil.which("rsync"): pytest.skip("rsync not installed")
    spec = importlib.util.spec_from_file_location("so101_policy_tool", TOOL)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


class Lab:
    def __init__(self, tool, tmp, ckpt):
        self.tool, self.tmp, self.ckpt = tool, str(tmp), ckpt
        self.jetson = os.path.join(self.tmp, "jetson_home"); self.policies = os.path.join(self.jetson, "so101", "policies")
    def run(self, *argv, rc=0):
        """Run a so101-policy command in-process; returns captured stdout + stderr. Asserts the exit code unless rc=None."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err): code = self.tool.main(list(argv))
        text = out.getvalue() + err.getvalue()
        if rc is not None: assert code == rc, f"exit {code} (wanted {rc}) for {argv}:\n{text}"
        return text
    def client(self):
        return self.tool.client()[1]
    def deployed(self):
        p = os.path.join(self.policies, "DEPLOYED.json")
        return json.load(open(p)) if os.path.exists(p) else None
    def current(self, name):
        p = os.path.join(self.policies, name, "current")
        return os.readlink(p) if os.path.islink(p) else None
    def versions(self, name):
        d = os.path.join(self.policies, name)
        return sorted(x for x in os.listdir(d) if x.startswith("v")) if os.path.isdir(d) else []
    def sha(self, path):
        return self.tool.sha256_of(path)


@pytest.fixture
def lab(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("SO101_MLFLOW_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SO101_DEPLOY_LOCAL_ROOT", str(tmp_path / "jetson_home"))
    monkeypatch.setenv("SO101_POLICIES_HUB", str(tmp_path / "hub"))
    monkeypatch.setenv("SO101_JETSON", BOGUS_JETSON)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))   # why: an empty Hub cache, so an offline lookup can only miss
    ckpt = make_ckpt(str(tmp_path / "train" / "act_fake"), seed=1, step="005000")
    os.symlink("005000", os.path.join(str(tmp_path / "train" / "act_fake"), "checkpoints", "last"))
    return Lab(tool, tmp_path, ckpt)


@pytest.fixture
def shell(tmp_path):
    """jetson/env.sh sourced against a fake $HOME: so101-policy-current, so101-run and so101-run-vla can run without an arm."""
    home = tmp_path / "home"; so = home / "so101"; (so / ".venv" / "bin").mkdir(parents=True); (so / ".venv" / "bin" / "activate").write_text("")
    bin_ = tmp_path / "bin"; bin_.mkdir(); argv_file = tmp_path / "lerobot-rollout.argv"
    (bin_ / "lerobot-rollout").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_file}"\n'); (bin_ / "lerobot-rollout").chmod(0o755)
    env = dict(os.environ, HOME=str(home), PATH=f"{bin_}:{os.path.dirname(sys.executable)}:/usr/bin:/bin")
    env.pop("SO101_POLICY", None); env.pop("SO101_VLA_POLICY", None)

    class Shell:
        def __init__(self): self.home, self.so, self.argv_file, self.env = home, so, argv_file, env
        def run(self, cmd, **extra):
            r = subprocess.run(["bash", "-c", f"source {JETSON_ENV}; {cmd}"], env={**env, **extra}, capture_output=True, text=True, timeout=30)
            return r.returncode, r.stdout + r.stderr
        def deploy(self, name, version, kind="act", write_deployed=True):
            """What so101-policy deploy leaves behind: <name>/v<version>/ with a checkpoint, current -> v<version>, DEPLOYED.json."""
            d = so / "policies" / name; make_ckpt(str(d / f"v{version}"), seed=version, kind=kind)
            cur = d / "current"
            if cur.is_symlink(): cur.unlink()
            cur.symlink_to(f"v{version}")
            if write_deployed: (so / "policies" / "DEPLOYED.json").write_text(json.dumps({"name": name, "version": version, "history": []}))
            return str(cur)
        def argv(self): return argv_file.read_text().splitlines() if argv_file.exists() else []
    return Shell()
