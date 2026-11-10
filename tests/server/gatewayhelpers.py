"""Fixtures for dashboard/phone.py (the LAN gateway) in both modes, with a stub upstream standing in for server.py.

* ``gw`` imports phone.py once with importlib; the module is inert on import (the server only starts under __main__) and
  nothing here reads or writes dashboard/.phone_token (the token is passed to ``make_handler`` directly).
* ``upstream`` is a threaded stub on 127.0.0.1:<ephemeral> that answers like server.py would: JSON echoes of the request
  (method, path, query, body, content type) under /api/, a JPEG-ish blob for /cam/<name>.jpg, a never-ending multipart
  stream for /cam/<name>, a 500 for /api/boom, and a marker body for /api/shell and /api/term so a test can prove the
  gateway never reached them. Every request is recorded in ``upstream.hits``.
* ``gateway`` is a factory: ``gateway("pad")`` / ``gateway("phone")`` binds a gateway on 127.0.0.1:<ephemeral> in front
  of the stub, serving a small fake dashboard tree (index.html, skins/x.css, urdf/so101.urdf, manifest, icon) from
  tmp_path, and returns a ``GwClient`` that speaks raw http.client (no redirect following, no path normalisation, so
  ``..`` really goes on the wire).
"""
import http.client, http.server, importlib.util, json, os, socketserver, threading, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHONE_PY = os.path.join(ROOT, "dashboard", "phone.py")
TOKEN = "unit-test-token-0123456789"
SECRET = b'["jetson shell transcript: must never cross the gateway"]'


@pytest.fixture(scope="session")
def gw():
    spec = importlib.util.spec_from_file_location("so101_phone_gateway", PHONE_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class _Upstream(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _reply(self, code, body, ctype, length=True):
        self.send_response(code); self.send_header("Content-Type", ctype)
        if length: self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self.server.hits.append(("GET", self.path, None, None))
        if self.path.startswith(("/api/shell", "/api/term")): return self._reply(200, SECRET, "application/json")
        if self.path.startswith("/api/boom"): return self._reply(500, b'{"ok": false, "err": "boom"}', "application/json")
        if self.path.startswith("/api/nolength"): return self._reply(200, b'{"nolen": true}', "application/json", length=False)   # server.py's /api/cams and /api/term send no Content-Length
        if self.path.startswith("/cam/") and ".jpg" in self.path: return self._reply(200, b"\xff\xd8JPEGBYTES\xff\xd9", "image/jpeg")
        if self.path.startswith("/cam/"):   # MJPEG: frames for ever, 40 ms apart
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            try:
                for i in range(10_000):
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 6\r\n\r\nFRAME" + str(i % 10).encode() + b"\r\n"); self.wfile.flush(); time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        if self.path.startswith("/api/"): return self._reply(200, json.dumps({"method": "GET", "path": self.path}).encode(), "application/json")
        self._reply(404, b"upstream: no such file", "text/plain")
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
        self.server.hits.append(("POST", self.path, body, self.headers.get("Content-Type")))
        if self.path.startswith(("/api/shell", "/api/term")): return self._reply(200, SECRET, "application/json")
        if self.path.startswith("/api/boom"): return self._reply(503, b"down", "text/plain")
        self._reply(200, json.dumps({"method": "POST", "path": self.path, "body": body.decode("utf-8", "replace"), "ctype": self.headers.get("Content-Type")}).encode(), "application/json")


class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
    def server_bind(self):
        socketserver.TCPServer.server_bind(self); self.server_name = "localhost"; self.server_port = self.server_address[1]


def _serve(srv):
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True); t.start(); return t


@pytest.fixture
def upstream():
    s = _Srv(("127.0.0.1", 0), _Upstream); s.hits = []; s.url = f"http://127.0.0.1:{s.server_address[1]}"; t = _serve(s)
    try: yield s
    finally: s.shutdown(); s.server_close(); t.join(2)


class GwClient:
    """Raw HTTP against the gateway. ``cookie=True`` sends the valid token cookie; ``path`` goes on the wire untouched."""
    def __init__(self, port, token): self.port = port; self.token = token
    def req(self, method, path, body=None, cookie=False, headers=None, read=True, timeout=5):
        h = dict(headers or {})
        if cookie: h["Cookie"] = f"so101k={self.token if cookie is True else cookie}"
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request(method, path, body=body, headers=h); r = c.getresponse()
        if not read: return r, c
        data = r.read(); c.close()
        return r.status, r.headers, data
    def get(self, path, **kw): return self.req("GET", path, **kw)
    def post(self, path, body=None, ctype="application/json", **kw):
        h = dict(kw.pop("headers", {}) or {})
        if ctype: h["Content-Type"] = ctype
        return self.req("POST", path, body=body, headers=h, **kw)


FAKE_TREE = {
    "index.html": "<!doctype html><title>SO-101 Lab</title><link rel=manifest href=/manifest.webmanifest><script src=skins/skins.js></script>FULL DASHBOARD",
    "phone.html": "<!doctype html><title>phone</title>PHONE PAGE",
    "skins/x.css": ":root{--acc:#ffb347}",
    "skins/skins.js": "window.skins=1",
    "urdf/so101.urdf": "<robot name='so101'/>",
    "urdf/assets/part.stl": "solid part\nendsolid part\n",
    "manifest.webmanifest": json.dumps({"name": "SO-101 Lab", "start_url": "/", "display": "fullscreen", "orientation": "landscape", "theme_color": "#050607", "background_color": "#050607", "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}]}),
    "icon.svg": "<svg xmlns='http://www.w3.org/2000/svg'/>",
    ".phone_token": "never-served\n",
    "server.py": "print('never served')\n",
}


@pytest.fixture
def dash_root(tmp_path):
    for rel, text in FAKE_TREE.items():
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    return str(tmp_path)


@pytest.fixture
def gateway(gw, upstream, dash_root):
    live = []
    def _start(mode="pad", token=TOKEN):
        H = gw.make_handler(token, upstream=upstream.url, page=os.path.join(dash_root, "phone.html"), mode=mode, root=dash_root)
        s = gw.Srv(("127.0.0.1", 0), H); t = _serve(s); live.append((s, t))
        return GwClient(s.server_address[1], token)
    try: yield _start
    finally:
        for s, t in live: s.shutdown(); s.server_close(); t.join(2)
