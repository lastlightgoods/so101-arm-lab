"""tests/conftest.py's guard: no test reaches the real Jetson or the live dashboard (2026-11-11 21:53 a suite run drove the real arm)."""
import subprocess, urllib.error, urllib.request
import pytest


@pytest.mark.parametrize("argv, shell", [(["ssh", "-o", "ConnectTimeout=1", "jetson", "true"], False), ("ssh jetson true", True), (["bash", "-c", "cd /tmp && ssh jetson true"], False), (["scp", "a", "jetson:b"], False)])
def test_ssh_and_scp_fail_like_an_unreachable_host(argv, shell):
    r = subprocess.run(argv, shell=shell, capture_output=True, text=True, timeout=20); assert r.returncode == 255 and "blocked in tests" in r.stderr


def test_a_popen_bound_before_the_guard_is_blocked_too():
    popen = subprocess.Popen   # what PickSession(popen=subprocess.Popen) captured at import time
    p = popen(["ssh", "jetson", "pick_seq.py --serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True); out, err = p.communicate(timeout=20)
    assert p.returncode == 255 and "blocked in tests" in err


def test_other_commands_still_run():
    assert subprocess.run(["echo", "ok"], capture_output=True, text=True).stdout.strip() == "ok"


def test_the_live_dashboard_is_unreachable():
    with pytest.raises(urllib.error.URLError): urllib.request.urlopen("http://localhost:8765/api/state", timeout=2)
