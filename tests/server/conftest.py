"""Hermetic fixtures for dashboard/server.py (the wall-dashboard backend).

How the module is imported without starting anything
----------------------------------------------------
server.py only starts its threads and the HTTP server under ``if __name__ == "__main__"``, so a plain import is inert.
Two pieces of import-time code still need guarding, which ``srv`` does before loading the file with importlib:
  * ``PORT = int(sys.argv[1])`` would choke on pytest's own argv, so argv is trimmed to the script name;
  * ``TRANSCRIPT = newest_transcript()`` globs ~/.claude for a Claude Code transcript, short-circuited by pointing
    SO101_TRANSCRIPT at a file that does not exist.
SO101_JETSON is set to an unresolvable host so that, even if a test forgot to script a call, nothing could reach the lab.

How the outside world is kept out
---------------------------------
* ``fake`` (autouse) replaces ``subprocess.run`` / ``subprocess.Popen`` for the duration of each test with a scripted
  fake: a test registers responses with ``fake.on(needle, stdout)``; any unscripted call raises AssertionError. The fake
  honours ``text=`` / ``errors=`` so a ``bytes`` response is decoded exactly the way the real module would decode it.
* ``HERE`` is redirected to ``tmp_path`` (events.log, index.html, roar.wav all resolve there), ``_scanpose`` is a no-op
  so the lock endpoints spawn no ssh thread, and all module-level state (state, REC, LOCK, HIST, cam_frames, SAY_Q...)
  is restored to its import-time snapshot before every test.
* Infinite loops (record_loop, tegra_loop, probe_loop) are driven for a bounded number of passes by ``run_loop``, which
  makes ``time.sleep`` raise after N calls; each pass through a loop body ends in exactly one sleep.
* ``http`` binds ``Srv`` on 127.0.0.1 port 0 in a daemon thread and shuts it down after the test. Nothing here talks to
  8765/8766/9333.
"""
import copy, importlib.util, json, os, queue, subprocess, sys, threading, time, urllib.error, urllib.request
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatewayhelpers as _g  # noqa: E402  (phone.py gateway fixtures: gw, upstream, dash_root, gateway; test files import its constants directly)
globals().update({_k: _v for _k, _v in vars(_g).items() if not _k.startswith("__")})

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER_PY = os.path.join(ROOT, "dashboard", "server.py")
BOGUS_JETSON = "so101-test-no-such-host.invalid"


@pytest.fixture(scope="session")
def srv():
    """The server module, imported once and inert (no threads, no sockets, no ssh)."""
    argv, env = sys.argv, dict(os.environ)
    sys.argv = ["server.py"]
    os.environ["SO101_TRANSCRIPT"] = os.path.join(ROOT, "tests", "server", "no-such-transcript.jsonl")
    os.environ["SO101_JETSON"] = BOGUS_JETSON
    os.environ["SO101_LOCK"] = "0"
    try:
        spec = importlib.util.spec_from_file_location("so101_dashboard_server", SERVER_PY)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    finally:
        sys.argv = argv; os.environ.clear(); os.environ.update(env)
    assert mod.JETSON == BOGUS_JETSON
    mod._TEST_SNAPSHOT = {k: copy.deepcopy(getattr(mod, k)) for k in ("state", "REC", "LOCK", "_roar", "_last_said", "_sim_cache")}
    return mod


class FakeSubprocess:
    """Scripted stand-in for subprocess.run / Popen. Rules are matched last-registered-first against the joined argv
    (substring) or via a callable(argv) -> bool. ``stdout`` may be str, bytes or callable(argv, kw) -> str|bytes."""
    def __init__(self): self.calls = []; self.popens = []; self.rules = []; self.popen_rules = []
    def on(self, needle, stdout="", returncode=0, stderr=""):
        self.rules.append((needle, stdout, returncode, stderr)); return self
    def on_popen(self, needle, lines):
        self.popen_rules.append((needle, lines)); return self
    @staticmethod
    def _hit(needle, argv):
        return needle(argv) if callable(needle) else needle in " ".join(map(str, argv))
    @staticmethod
    def _coerce(val, kw):
        text = kw.get("text") or kw.get("universal_newlines") or kw.get("encoding") or kw.get("errors")
        if text and isinstance(val, bytes): return val.decode(kw.get("encoding") or "utf-8", kw.get("errors") or "strict")
        if not text and isinstance(val, str): return val.encode()
        return val
    def run(self, argv, **kw):
        argv = list(map(str, argv)); self.calls.append((argv, kw))
        for needle, out, rc, err in reversed(self.rules):
            if self._hit(needle, argv):
                if callable(out): out = out(argv, kw)
                return subprocess.CompletedProcess(argv, rc, self._coerce(out, kw), self._coerce(err, kw))
        raise AssertionError("unscripted subprocess.run: " + " ".join(argv)[:300])
    def popen(self, argv, **kw):
        argv = list(map(str, argv)); self.popens.append((argv, kw))
        for needle, lines in reversed(self.popen_rules):
            if self._hit(needle, argv): return FakePopen(lines)
        raise AssertionError("unscripted subprocess.Popen: " + " ".join(argv)[:300])
    @property
    def ssh(self):
        """The remote command string of every ssh call, in order."""
        return [a[-1] for a, _ in self.calls if a and a[0] == "ssh"]


class FakePopen:
    def __init__(self, lines): self.stdout = iter(lines); self.returncode = 0
    def wait(self, timeout=None): return 0
    def poll(self): return 0
    def kill(self): pass


@pytest.fixture(autouse=True)
def fake(srv, monkeypatch, tmp_path):
    """Fresh module state + scripted subprocess for every test."""
    for k, v in srv._TEST_SNAPSHOT.items():
        live = getattr(srv, k); live.clear(); live.update(copy.deepcopy(v))
    srv.HIST.clear(); srv.cam_frames.clear(); srv.shell_calls.clear(); srv._calib_cache.clear()
    while not srv.SAY_Q.empty():
        try: srv.SAY_Q.get_nowait()
        except queue.Empty: break
    f = FakeSubprocess()
    monkeypatch.setattr(subprocess, "run", f.run); monkeypatch.setattr(subprocess, "Popen", f.popen)
    monkeypatch.setattr(srv, "HERE", str(tmp_path))
    monkeypatch.setattr(srv, "_scanpose", lambda rest: None)
    return f


@pytest.fixture
def logs(srv, monkeypatch):
    """Captures server.log(...) lines instead of printing them."""
    out = []
    monkeypatch.setattr(srv, "log", lambda *a: out.append(" ".join(str(x) for x in a)))
    return out


@pytest.fixture
def said(srv):
    """Everything queued for the speech loop (which is not running), as a list of strings."""
    def _drain():
        got = []
        while not srv.SAY_Q.empty(): got.append(srv.SAY_Q.get_nowait()[1])
        return got
    return _drain


class StopLoop(Exception): pass


@pytest.fixture
def run_loop(srv, monkeypatch):
    """Run an infinite loop function until time.sleep has been called ``sleeps`` times, then unwind it."""
    def _run(fn, sleeps=1, *args):
        n = {"left": sleeps}; me = threading.get_ident(); real_sleep = time.sleep
        def fake_sleep(s):
            if threading.get_ident() != me: return real_sleep(min(float(s), 0.01))   # why: the server's own threads sleep too; only the loop under test is counted (CI flake, 11 Nov)
            n["left"] -= 1
            if n["left"] <= 0: raise StopLoop
        monkeypatch.setattr(time, "sleep", fake_sleep)
        with pytest.raises(StopLoop): fn(*args)
        assert n["left"] == 0, f"loop ended early: {n['left']} sleeps unused"
    return _run


@pytest.fixture
def events(srv):
    """Event messages, newest first."""
    return lambda: [e["msg"] for e in srv.state["events"]]


class Client:
    def __init__(self, base): self.base = base
    def _req(self, path, data=None):
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET")
        if data is not None: req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as r: return r.status, r.headers, r.read()   # headers: case-insensitive mapping
        except urllib.error.HTTPError as e: return e.code, e.headers, e.read()
    def get(self, path): return self._req(path)
    def get_json(self, path):
        st, hdr, body = self._req(path); assert st == 200, (st, body[:200]); assert hdr.get("Content-Type") == "application/json"
        return json.loads(body.decode("utf-8")), body
    def post_json(self, path, obj=None, raw=None):
        st, hdr, body = self._req(path, raw if raw is not None else json.dumps(obj or {}).encode())
        assert st == 200, (st, body[:200]); return json.loads(body.decode("utf-8"))


@pytest.fixture
def http(srv):
    """A live server on 127.0.0.1:<ephemeral>, torn down after the test."""
    s = srv.Srv(("127.0.0.1", 0), srv.H)
    t = threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True); t.start()
    try: yield Client(f"http://127.0.0.1:{s.server_address[1]}")
    finally:
        s.shutdown(); s.server_close(); t.join(2)
