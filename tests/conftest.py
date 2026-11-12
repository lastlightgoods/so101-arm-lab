"""Root pytest shim: tracking off for every test by default. jetson/labtrack.py falls back to the Mac's real MLflow store
(~/so101/mlflow/mlflow.db) when no URI is set, so without this a CLI test run from the Mac would write into TC's real
experiments. Tests that exercise tracking set SO101_MLFLOW_URI themselves (monkeypatch or the subprocess env)."""
import os as _os
_os.environ["SO101_MLFLOW_URI"] = ""

# ---- no test reaches the real lab
# why: 2026-11-11 21:53 a full local suite run drove the REAL arm through one whole closed-loop pick (pick_seq --serve over ssh, tick sink on)
# and moved the tape ring: some test reached PickSession's default Popen (bound at definition time, so patching subprocess.Popen misses it).
# Every process spawn goes through Popen._execute_child: ssh/scp/rsync there is swapped for a command that fails the way ssh does in CI
# (exit 255), and urlopen to the live dashboard fails like a dashboard that is down. Each attempt is written to outputs/test_live_attempts.log.
import shlex as _shlex, subprocess as _sp, sys as _sys, time as _time, urllib.error as _ue, urllib.request as _ur
import pytest as _pytest

_LIVE_EXES = ("ssh", "scp", "rsync")
_DASH_URLS = tuple({_os.environ.get("SO101_DASH", "http://localhost:8765").rstrip("/"), "http://localhost:8765", "http://127.0.0.1:8765"})
_LOG = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs", "test_live_attempts.log")


def _live_argv(args):
    words = str(args).split() if isinstance(args, (str, bytes)) else [str(a) for a in args]
    if not words: return False
    if _os.path.basename(words[0]) == "rsync": return any(":" in w and not w.startswith(("/", ".", "-")) for w in words[1:])   # why: a LOCAL rsync (the deploy tests' stand-in for the Jetson, learn_loop's tests) is harmless; only host:path operands reach a machine
    if _os.path.basename(words[0]) in _LIVE_EXES: return True
    return _os.path.basename(words[0]) in ("sh", "bash", "zsh") and any(w in _LIVE_EXES for w in " ".join(words[1:]).replace(";", " ").replace("&&", " ").replace("|", " ").split())


def _note(nodeid, what):
    try:
        _os.makedirs(_os.path.dirname(_LOG), exist_ok=True)
        with open(_LOG, "a") as f: f.write(f"{_time.strftime('%Y-%m-%dT%H:%M:%S')} {nodeid} {what[:300]}\n")
    except OSError: pass


@_pytest.fixture(autouse=True)
def _no_live_lab(monkeypatch, request):
    real_exec, real_open = _sp.Popen._execute_child, _ur.urlopen
    def exec_child(self, args, *a, **k):
        if _live_argv(args):
            _note(request.node.nodeid, f"spawn {args!r}")
            fail = [_sys.executable, "-c", "import sys; sys.stderr.write('ssh: blocked in tests (tests/conftest.py)\\n'); sys.exit(255)"]
            shell = len(a) > 8 and bool(a[8])   # _execute_child(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, env, startupinfo, creationflags, shell, ...)
            args = _shlex.join(fail) if shell else fail
            if a and not shell: a = (None,) + tuple(a[1:])   # an explicit executable must not stay "ssh"
        return real_exec(self, args, *a, **k)
    def urlopen(url, *a, **k):
        u = url if isinstance(url, str) else getattr(url, "full_url", "")
        if u.startswith(_DASH_URLS):
            _note(request.node.nodeid, f"urlopen {u}"); raise _ue.URLError("live dashboard blocked in tests (tests/conftest.py)")
        return real_open(url, *a, **k)
    monkeypatch.setattr(_sp.Popen, "_execute_child", exec_child); monkeypatch.setattr(_ur, "urlopen", urlopen)
