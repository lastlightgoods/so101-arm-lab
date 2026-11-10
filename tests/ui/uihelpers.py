"""pytest glue: one headless Chrome per session, one stub backend + one fresh tab (node driver) per test."""
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apidata as fx  # noqa: E402
from stub_server import Stub  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
NODE = shutil.which("node")
FORBIDDEN = {8765, 8766, 9333}   # the live kiosk Chrome and dashboard: never bind, never connect
FIRST_POLLS = ["/api/state", "/api/hist", "/api/record", "/api/term", "/api/cams", "/api/shell", "/api/sim", "/api/lock"]


def free_port():
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if p not in FORBIDDEN:
            return p


def _killpg(proc):
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


@pytest.fixture(scope="session")
def chrome(tmp_path_factory):
    if not os.path.exists(CHROME):
        pytest.skip("Google Chrome not installed at " + CHROME)
    if not NODE:
        pytest.skip("node not on PATH")
    port = free_port()
    assert port not in FORBIDDEN
    busy = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True)
    assert busy.returncode != 0, f"port {port} unexpectedly in use"
    profile = tmp_path_factory.mktemp("chrome-profile")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                             "--no-first-run", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",   # software WebGL for the 3D-stage tests
                             "--no-default-browser-check", "--mute-audio",
                             "--hide-scrollbars", "--window-size=1600,1000", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.time() + 30
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None or time.time() > deadline:
                    _killpg(proc)
                    raise RuntimeError("headless Chrome did not come up") from None
                time.sleep(0.1)
        yield port
    finally:
        _killpg(proc)


class PageError(Exception):
    pass


class Page:
    """A tab of the headless Chrome, driven through driver.js over stdin/stdout JSON lines."""

    def __init__(self, chrome_port, url, hide="#gallery", cdn=False):
        # hide = the self-playing Tempest canvas: under --disable-gpu its shadow-blur frames keep the renderer's main
        # thread ~100% busy and make every fetch/eval wait ~100 ms; hidden from the first document, the suite is ~50x faster
        # cdn = let the three.js / urdf-loader modules load from their CDNs (3D-stage tests only; needs the network)
        self.proc = subprocess.Popen([NODE, os.path.join(HERE, "driver.js"), str(chrome_port), url, hide] + (["cdn"] if cdn else []),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.n = 0
        self.checked = False
        first = self._read(30)
        if not first.get("ready"):
            raise RuntimeError(f"driver failed to open the page: {first}")

    def _pump(self):
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put(None)

    def _read(self, timeout=15):
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("driver timed out") from None
        if line is None:
            raise RuntimeError("driver exited: " + (self.proc.stderr.read() or ""))
        return json.loads(line)

    def _cmd(self, **c):
        self.n += 1
        c["id"] = self.n
        self.proc.stdin.write(json.dumps(c) + "\n")
        self.proc.stdin.flush()
        r = self._read()
        if "error" in r:
            raise PageError(r["error"])
        return r

    # -- page access ---------------------------------------------------------
    def eval(self, expr):
        """Evaluate JS in the page; promises are awaited; the JSON value comes back (undefined -> None)."""
        return self._cmd(cmd="eval", expr=expr)["value"]

    def text(self, sel):
        return self.eval(f"document.querySelector({sel!r}).textContent")

    def cls(self, sel):
        return self.eval(f"document.querySelector({sel!r}).className")

    def value(self, sel):
        return self.eval(f"document.querySelector({sel!r}).value")

    def click(self, sel):
        self.eval(f"(document.querySelector({sel!r}).click(), true)")

    def key(self, k):
        self._cmd(cmd="key", key=k)

    def wait_until(self, expr, timeout=5.0, every=0.05):
        deadline = time.time() + timeout
        while True:
            v = self.eval(expr)
            if v:
                return v
            if time.time() > deadline:
                raise AssertionError(f"timed out waiting for: {expr}")
            time.sleep(every)

    def screenshot(self, path):
        """PNG of the viewport, written to path (for reports; nothing asserts on it)."""
        return self._cmd(cmd="shot", path=path)["value"]

    def freeze_polling(self):
        """Stop the page's own setInterval loops so each test drives refresh()/recPoll()/... explicitly."""
        self.eval("(()=>{for(let i=1;i<5000;i++)clearInterval(i);return true})()")

    # -- error bookkeeping ---------------------------------------------------
    def errors(self):
        return self._cmd(cmd="errors")

    def clear_errors(self):
        self._cmd(cmd="clear")

    def check(self):
        """Any uncaught page exception, or console.error/warn, fails the test."""
        self.checked = True
        e = self.errors()
        assert e["exceptions"] == [], "uncaught page exceptions:\n" + "\n".join(e["exceptions"])
        assert e["console"] == [], "console errors/warnings:\n" + "\n".join(e["console"])

    def close(self):
        try:
            self._cmd(cmd="close")
        except Exception:
            pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture
def stub():
    s = Stub()
    s.routes.update(fx.default_routes())
    s.start()
    yield s
    s.stop()


def _open(chrome, stub):
    p = Page(chrome, stub.url + "/")
    stub.wait_hits(FIRST_POLLS)                                              # the page's own first poll cycle
    p.wait_until("document.querySelector('#v_gpu').textContent!=='--'")      # ...has painted the gauges
    return p


@pytest.fixture
def page(chrome, stub):
    """Loaded dashboard after its first natural poll cycle, with the interval loops frozen for deterministic steps."""
    p = _open(chrome, stub)
    p.freeze_polling()
    yield p
    try:
        if not p.checked:
            p.check()
    finally:
        p.close()


@pytest.fixture
def live_page(chrome, stub):
    """Same, but the page keeps polling on its own timers."""
    p = _open(chrome, stub)
    yield p
    try:
        if not p.checked:
            p.check()
    finally:
        p.close()
