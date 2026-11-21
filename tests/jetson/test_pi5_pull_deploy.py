"""jetson/pi5_pull_deploy.sh (push-to-main deploys for the Pi) against a local origin repo, with fake curl (GitHub check runs),
pgrep, pip, systemctl and loginctl on PATH and a temp HOME. Runs under the Mac's bash 3.2. No network, no ssh."""
import json, os, subprocess
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "jetson" / "pi5_pull_deploy.sh"
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def git(cwd, *a):
    return subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True, text=True, env={**os.environ, **GIT_ENV}).stdout.strip()


def runs(*items):
    return json.dumps({"total_count": len(items), "check_runs": [dict(zip(("id", "name", "status", "conclusion"), i)) for i in items]})


CI = {"success": runs((1, "test", "completed", "success")), "pending": runs((1, "test", "in_progress", None)), "none": runs(),
      "failure": runs((1, "test", "completed", "failure"))}


class Pi:
    def __init__(self, tmp):
        self.tmp, self.origin, self.home, self.bin = tmp, tmp / "origin", tmp / "home", tmp / "bin"
        self.so = self.home / "so101"
        for d in (self.origin, self.so / ".venv" / "bin", self.bin, self.so / "calib"):
            d.mkdir(parents=True)
        git(self.origin, "init", "-q", "-b", "main")
        self.write({"jetson/goto.py": "print('v1')\n", "jetson/dance.py": "print('dance')\n", "jetson/env.sh": "so101-run() { lerobot-rollout --policy.device=cuda; }\n",
                    "jetson/calib/top_table.json": '{"repo": 1}\n', "jetson/99-so101-cams.rules": "rule\n",
                    "plugins/lerobot-robot-busheal/pyproject.toml": "[project]\nname = 'busheal'\n", "plugins/lerobot-robot-busheal/busheal/__init__.py": "X = 1\n",
                    "plugins/lerobot-camera-avf/avf.py": "mac only\n", "dashboard/server.py": "mac only\n"})
        self.commit("first")
        (self.so / "calib" / "top_table.json").write_text('{"pi": "own"}\n')
        self.ci("success"); self.procs("")
        self.tool("curl", 'echo "$*" >> "$FAKE_LOG/curl"; [ -f "$FAKE_LOG/curl_rc" ] && { echo "curl: (6) Could not resolve host" >&2; exit "$(cat "$FAKE_LOG/curl_rc")"; }; cat "$FAKE_LOG/ci.json"')
        self.tool("pgrep", 'grep -E -- "$2" "$FAKE_LOG/procs"')   # -af PATTERN: the script's real pattern decides what counts as an arm job
        self.tool("systemctl", 'echo "$*" >> "$FAKE_LOG/systemctl"; [ "$2" = is-active ] && echo active; exit 0')
        self.tool("loginctl", 'echo "$*" >> "$FAKE_LOG/loginctl"')
        self.tool("flock", "exit 0")   # Linux has it, the Mac does not: make both runs take the same path
        pip = self.so / ".venv" / "bin" / "pip"; pip.write_text('#!/bin/bash\necho "$*" >> "$FAKE_LOG/pip"\n'); pip.chmod(0o755)

    def tool(self, name, body):
        p = self.bin / name; p.write_text("#!/bin/bash\n" + body + "\n"); p.chmod(0o755)

    def write(self, files):
        for rel, text in files.items():
            p = self.origin / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)

    def commit(self, msg):
        git(self.origin, "add", "-A"); git(self.origin, "commit", "-q", "-m", msg); return git(self.origin, "rev-parse", "HEAD")

    def ci(self, state): (self.tmp / "ci.json").write_text(CI.get(state, state))
    def procs(self, text): (self.tmp / "procs").write_text(text)
    def calls(self, tool):
        p = self.tmp / tool; return p.read_text().splitlines() if p.exists() else []
    def log(self):
        p = self.so / "deploy.log"; return p.read_text().splitlines() if p.exists() else []
    def ref(self):
        p = self.so / ".deployed_ref"; return p.read_text().strip() if p.exists() else None

    def run(self, *args, **env):
        e = {**os.environ, **GIT_ENV, "HOME": str(self.home), "PATH": f"{self.bin}:{os.environ['PATH']}", "TMPDIR": str(self.tmp), "USER": "tc",
             "SO101_REPO_URL": str(self.origin), "FAKE_LOG": str(self.tmp), **env}
        for k in ("SO101_HOME", "SO101_SRC", "SO101_REQUIRE_CI"):
            if k not in env: e.pop(k, None)
        r = subprocess.run(["/bin/bash", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=60)
        return r.returncode, r.stdout + r.stderr


@pytest.fixture
def pi(tmp_path):
    return Pi(tmp_path)


def test_first_run_deploys_the_jetson_files_and_the_busheal_plugin_only(pi):
    rc, out = pi.run("run"); head = git(pi.origin, "rev-parse", "HEAD")
    assert rc == 0, out
    assert (pi.so / "goto.py").read_text() == "print('v1')\n" and (pi.so / "dance.py").exists()
    assert (pi.so / "plugins/lerobot-robot-busheal/busheal/__init__.py").exists()
    assert not (pi.so / "plugins/lerobot-camera-avf").exists() and not (pi.so / "dashboard").exists(), "Mac-only code stays off the Pi"
    assert not (pi.so / "99-so101-cams.rules").exists(), "Jetson system files are not deployed"
    assert (pi.so / "calib/top_table.json").read_text() == '{"pi": "own"}\n', "the Pi's calibration is never overwritten"
    assert "--policy.device=cpu" in (pi.so / "env.sh").read_text() and "cuda" not in (pi.so / "env.sh").read_text()
    assert pi.ref() == head and any(f"deployed {head[:7]} \"first\"" in line for line in pi.log()), pi.log()
    assert len(pi.calls("pip")) == 1 and "-e" in pi.calls("pip")[0], "a new pyproject.toml means a (re)install"
    assert len(pi.calls("curl")) == 1 and f"/commits/{head}/check-runs" in pi.calls("curl")[0], "CI asked about that exact commit"


def test_no_new_commit_is_a_quiet_no_op(pi):
    pi.run("run"); lines = pi.log(); curls = len(pi.calls("curl"))
    rc, out = pi.run("run")
    assert rc == 0 and "up to date at" in out
    assert pi.log() == lines and len(pi.calls("curl")) == curls, "no log line and no API call when nothing moved"


def test_a_changed_file_is_backed_up_then_replaced_and_upstream_deletions_stay(pi):
    pi.run("run")
    pi.write({"jetson/goto.py": "print('v2')\n"}); (pi.origin / "jetson/dance.py").unlink(); head = pi.commit("goto v2, dance gone")
    rc, out = pi.run("run")
    assert rc == 0, out
    assert (pi.so / "goto.py").read_text() == "print('v2')\n" and pi.ref() == head
    baks = list((pi.so / "deploy_bak").glob(f"*-{head[:7]}/goto.py"))
    assert len(baks) == 1 and baks[0].read_text() == "print('v1')\n"
    assert (pi.so / "dance.py").exists(), "a deploy never deletes"
    assert "1 file(s), backup deploy_bak/" in pi.log()[-1] and len(pi.calls("pip")) == 1, "no plugin reinstall for a script change"


@pytest.mark.parametrize("state, needle", [("pending", "CI not finished on"), ("none", "CI not finished on"), ("failure", "CI failure:failure on")])
def test_a_commit_without_green_ci_is_not_deployed(pi, state, needle):
    pi.run("run"); old = pi.ref()
    pi.write({"jetson/goto.py": "print('v2')\n"}); head = pi.commit("goto v2"); pi.ci(state)
    rc, out = pi.run("run")
    assert rc == 0 and pi.ref() == old and (pi.so / "goto.py").read_text() == "print('v1')\n"
    assert needle in pi.log()[-1] and head[:7] in pi.log()[-1], pi.log()


def test_pending_ci_deploys_once_it_turns_green_and_the_skip_is_logged_once(pi):
    pi.run("run"); pi.write({"jetson/goto.py": "print('v2')\n"}); head = pi.commit("goto v2"); pi.ci("pending")
    pi.run("run"); pi.run("run")
    assert sum("skip: CI not finished" in line for line in pi.log()) == 1, pi.log()
    pi.ci("success"); rc, out = pi.run("run")
    assert rc == 0 and pi.ref() == head and (pi.so / "goto.py").read_text() == "print('v2')\n"


def test_a_failed_verdict_is_cached_so_a_red_commit_does_not_spend_the_api_limit(pi):
    pi.run("run"); pi.write({"jetson/goto.py": "print('v2')\n"}); pi.commit("goto v2"); pi.ci("failure")
    pi.run("run"); n = len(pi.calls("curl")); pi.run("run"); pi.run("run")
    assert len(pi.calls("curl")) == n


def test_the_latest_run_of_the_named_check_decides(pi):
    pi.ci(runs((1, "test", "completed", "failure"), (7, "test", "completed", "success"), (9, "lint-other", "completed", "failure")))
    rc, out = pi.run("run")
    assert rc == 0 and pi.ref() == git(pi.origin, "rev-parse", "HEAD"), out


def test_an_unreachable_api_is_a_skip_not_a_deploy(pi):
    (pi.tmp / "curl_rc").write_text("6")
    rc, out = pi.run("run")
    assert rc == 0 and pi.ref() is None and not (pi.so / "goto.py").exists()
    assert "skip: CI unknown:api" in pi.log()[-1]


@pytest.mark.parametrize("busy, needle", [("flag", "arm busy flag"), ("proc", "arm job running: python /home/tc/so101/goto.py --read"),
                                          ("cwd", "arm job running: python3 -u pick_seq.py --serve"),
                                          ("lerobot", "arm job running: /home/tc/so101/.venv/bin/python /home/tc/so101/.venv/bin/lerobot-calibrate")])
def test_a_running_arm_job_holds_the_deploy_until_it_ends(pi, busy, needle):
    pi.run("run"); old = pi.ref()
    pi.write({"jetson/goto.py": "print('v2')\n"}); head = pi.commit("goto v2")
    if busy == "flag": (pi.so / ".arms_busy").touch()
    elif busy == "proc": pi.procs("4242 python /home/tc/so101/goto.py --read\n")
    elif busy == "cwd": pi.procs("4244 python3 -u pick_seq.py --serve\n")
    else: pi.procs("4243 /home/tc/so101/.venv/bin/python /home/tc/so101/.venv/bin/lerobot-calibrate --robot.type=so101_follower\n")
    rc, out = pi.run("run")
    assert rc == 0 and pi.ref() == old and needle in pi.log()[-1] and f"({head[:7]} waits)" in pi.log()[-1], pi.log()
    (pi.so / ".arms_busy").unlink(missing_ok=True); pi.procs("")
    rc, out = pi.run("run")
    assert pi.ref() == head and (pi.so / "goto.py").read_text() == "print('v2')\n"


def test_other_processes_do_not_count_as_arm_jobs(pi):
    pi.procs("100 python3 -m http.server 8000\n101 /bin/bash /home/tc/so101/pi5_pull_deploy.sh run\n102 vim goto.py.bak\n103 vim /home/tc/so101/goto.py\n104 less dance.py\n")
    rc, out = pi.run("run")
    assert rc == 0 and pi.ref(), out


def test_a_plugin_pyproject_change_reinstalls_it(pi):
    pi.run("run"); pi.write({"plugins/lerobot-robot-busheal/busheal/__init__.py": "X = 2\n"}); pi.commit("plugin code")
    pi.run("run"); assert len(pi.calls("pip")) == 1, "editable install: code changes need no reinstall"
    pi.write({"plugins/lerobot-robot-busheal/pyproject.toml": "[project]\nname = 'busheal'\nversion = '2'\n"}); pi.commit("plugin metadata")
    pi.run("run"); assert len(pi.calls("pip")) == 2 and "plugin reinstalled" in pi.log()[-1]


def test_dry_run_writes_nothing_in_so101(pi):
    before = sorted(p.name for p in pi.so.iterdir())
    rc, out = pi.run("run", "--dry-run")
    assert rc == 0 and "dry-run: would update goto.py" in out and "dry-run: would deploy" in out
    assert sorted(p.name for p in pi.so.iterdir()) == before and pi.ref() is None and pi.calls("pip") == []


def test_require_ci_0_deploys_without_asking_github(pi):
    rc, out = pi.run("run", SO101_REQUIRE_CI="0")
    assert rc == 0 and pi.ref() and pi.calls("curl") == []


def test_a_failed_fetch_is_logged_and_deploys_nothing(pi):
    rc, out = pi.run("run", SO101_REPO_URL=str(pi.tmp / "no-such-repo"))
    assert rc == 1 and pi.ref() is None and "skip: fetch of" in pi.log()[-1]


def test_install_writes_and_enables_the_user_timer_and_prints_the_linger_line(pi):
    rc, out = pi.run("install")
    d = pi.home / ".config/systemd/user"
    assert rc == 0, out
    assert f"ExecStart=/bin/bash {pi.so}/pi5_pull_deploy.sh run" in (d / "so101-pull-deploy.service").read_text()
    assert "OnUnitActiveSec=2min" in (d / "so101-pull-deploy.timer").read_text()
    assert "--user enable --now so101-pull-deploy.timer" in pi.calls("systemctl")
    assert "sudo loginctl enable-linger tc" in out and pi.calls("loginctl") == [], "install never runs sudo"
    rc, out = pi.run("uninstall")
    assert not (d / "so101-pull-deploy.timer").exists() and "--user disable --now so101-pull-deploy.timer" in pi.calls("systemctl")


def test_status_reports_ref_origin_arm_and_log(pi):
    pi.run("run"); rc, out = pi.run("status")
    assert rc == 0 and f"deployed:  {git(pi.origin, 'rev-parse', 'HEAD')}" in out and "arm: idle" in out and "timer: active" in out and "deployed " in out
