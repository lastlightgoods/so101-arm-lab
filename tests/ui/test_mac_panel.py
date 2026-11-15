"""SO-101 Panel.app (REQ 007): the wall dashboard in its own native window so it never shares Chrome with the agents.
why: 2026-11-14 18:11 the Mac bogged down and the Chrome the agents use hung (the kiosk panel ran in that Chrome); TC rebooted.
These tests build the app into a temp dir (never ~/Applications, never launched: no window opens) and check the launcher."""
import os
import plistlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DASH = os.path.join(ROOT, "dashboard", "dash.sh")
MAC = os.path.join(ROOT, "dashboard", "mac")
SWIFT = os.path.join(MAC, "PanelApp.swift")
INDEX = os.path.join(ROOT, "dashboard", "index.html")


def _src(p):
    with open(p) as f:
        return f.read()


@pytest.mark.skipif(not shutil.which("zsh"), reason="zsh not installed")
def test_dash_sh_parses():
    r = subprocess.run(["zsh", "-n", DASH], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dash_sh_modes():
    s = _src(DASH)
    for mode in ("stop)", "app)", "safari)", "open|*)"):
        assert re.search(r"^\s+" + re.escape(mode), s, re.M), mode
    assert "SO101Panel" in s.split("stop)")[1].split(";;")[0], "stop must also quit the native panel"
    # a bare `so101-dash` stays the Chrome kiosk unless SO101_DASH_BROWSER says otherwise
    assert 'set -- "$SO101_DASH_BROWSER"' in s


def test_panel_app_source_keeps_the_panel_alive():
    s = _src(SWIFT)
    assert 'http://127.0.0.1:8765/' in s                         # same page as the kiosk
    assert "webViewWebContentProcessDidTerminate" in s            # page process killed or crashed -> reload, not a blank screen
    assert "didFailProvisionalNavigation" in s and "retrySoon" in s   # server not up yet -> keep retrying
    assert "frame.height > $0.frame.width" in s                   # prefers the vertical monitor
    assert "toggleFullScreen" in s
    assert "showReview" in s and 'keyEquivalent: "2"' in s   # Cmd-2 = TC Review (REQ 012), Cmd-1 = arm panel
    # the page calls alert(); a WKWebView without an alert panel delegate drops it silently
    if "alert(" in _src(INDEX):
        assert "runJavaScriptAlertPanelWithMessage" in s


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("swiftc"), reason="needs macOS + swiftc")
def test_panel_app_builds(tmp_path):
    r = subprocess.run([os.path.join(MAC, "build.sh"), str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    app = tmp_path / "SO-101 Panel.app"
    exe = app / "Contents" / "MacOS" / "SO101Panel"
    assert exe.is_file() and os.access(exe, os.X_OK)
    with open(app / "Contents" / "Info.plist", "rb") as f:
        info = plistlib.load(f)
    assert info["CFBundleExecutable"] == "SO101Panel"
    assert info["CFBundleIdentifier"] == "com.lastlight.so101-panel"
    assert info["NSAppTransportSecurity"]["NSAllowsLocalNetworking"] is True   # http://127.0.0.1 page + ttyd iframe
