"""dashboard/phone.py: the LAN gateway in phone mode (unchanged allowlist) and pad mode (full dashboard for the iPad).
Everything runs against the stub upstream in gatewayhelpers.py; nothing here touches 8765 or 8770."""
import json, time
import pytest
from gatewayhelpers import SECRET, TOKEN


# ---------- auth: the same in both modes ----------
@pytest.mark.parametrize("mode", ["phone", "pad"])
def test_query_token_becomes_cookie_and_redirects(gateway, mode):
    c = gateway(mode)
    st, h, _ = c.get(f"/?k={TOKEN}")
    assert st == 303 and h["Location"] == "/"
    ck = h["Set-Cookie"]
    assert ck.startswith(f"so101k={TOKEN};") and "HttpOnly" in ck and "SameSite=Strict" in ck and "Path=/" in ck
    assert c.get("/", cookie=True)[0] == 200


@pytest.mark.parametrize("mode", ["phone", "pad"])
def test_unauthenticated_is_401_everywhere(gateway, mode, upstream):
    c = gateway(mode)
    for path in ("/", "/api/state", "/cam/top.jpg", "/skins/x.css", "/manifest.webmanifest"):
        assert c.get(path)[0] == 401, path
    assert c.get("/", cookie="wrong-token")[0] == 401 and c.get("/?k=nope")[0] == 401
    st, _, body = c.post("/api/arm/stop", b"", headers={"X-SO101": "1"})
    assert st == 401, "an unauthenticated write is refused before anything is forwarded"
    assert upstream.hits == []


@pytest.mark.parametrize("mode", ["phone", "pad"])
def test_shell_and_terminal_are_denied_even_with_a_valid_cookie(gateway, mode, upstream):
    c = gateway(mode)
    for path in ("/api/shell", "/api/shell?x=1", "/api/shell/more", "/api/term", "/api/terminal", "/api/ttyd", "/cam/ttyd", "/api/%73hell"):
        st, _, body = c.get(path, cookie=True)
        assert st == 404 and SECRET not in body, path
        st, _, body = c.post(path, b"{}", cookie=True, headers={"X-SO101": "1"})
        assert st == 404 and SECRET not in body, path
    assert c.get("/api/shell")[0] == 404, "denied before auth too: no 401 hint that the route exists"
    assert not [h for h in upstream.hits if "shell" in h[1] or "term" in h[1] or "ttyd" in h[1]]


@pytest.mark.parametrize("mode", ["phone", "pad"])
def test_dot_dot_is_denied_everywhere(gateway, mode, upstream):
    c = gateway(mode)
    for path in ("/skins/../server.py", "/skins/%2e%2e/server.py", "/api/../server.py", "/urdf/..%2fserver.py", "/..", "/api/state/.."):
        st, _, body = c.get(path, cookie=True)
        assert st == 404 and b"never served" not in body, path
    assert upstream.hits == []


# ---------- phone mode: the allowlist is unchanged ----------
def test_phone_mode_serves_phone_page_and_only_the_allowlist(gateway, upstream):
    c = gateway("phone")
    st, h, body = c.get("/", cookie=True)
    assert st == 200 and b"PHONE PAGE" in body and "default-src 'none'" in h["Content-Security-Policy"] and h["X-Frame-Options"] == "DENY"
    assert c.get("/index.html", cookie=True)[2] == body
    for ok in ("/api/state", "/cam/top.jpg", "/cam/wrist.jpg"):
        assert c.get(ok, cookie=True)[0] == 200, ok
    for no in ("/api/hist", "/api/record", "/cam/front.jpg", "/cam/top", "/skins/x.css", "/urdf/so101.urdf", "/manifest.webmanifest", "/api/locate?u=1&v=2"):
        assert c.get(no, cookie=True)[0] == 404, no
    assert [h[1] for h in upstream.hits] == ["/api/state", "/cam/top.jpg", "/cam/wrist.jpg"], "queries are not forwarded in phone mode"


def test_phone_mode_post_needs_the_header_and_only_stops(gateway, upstream):
    c = gateway("phone")
    assert c.post("/api/arm/stop", b"", cookie=True)[0] == 403
    assert c.post("/api/arm/teleop", b"", cookie=True, headers={"X-SO101": "1"})[0] == 404
    assert c.post("/api/record/ctl", b'{"cmd":"stop"}', cookie=True, headers={"X-SO101": "1"})[0] == 404
    st, _, body = c.post("/api/arm/stop", b"", cookie=True, headers={"X-SO101": "1"})
    assert st == 200 and json.loads(body)["path"] == "/api/arm/stop"
    assert [h[:2] for h in upstream.hits] == [("POST", "/api/arm/stop")]


# ---------- pad mode: the full dashboard ----------
def test_pad_serves_index_and_static_assets_with_the_right_types(gateway):
    c = gateway("pad")
    st, h, body = c.get("/", cookie=True)
    assert st == 200 and b"FULL DASHBOARD" in body and h["Content-Type"].startswith("text/html") and h["Cache-Control"] == "no-cache"
    csp = h["Content-Security-Policy"]
    for need in ("https://cdnjs.cloudflare.com", "https://cdn.jsdelivr.net", "'unsafe-inline'", "https://fonts.googleapis.com", "https://fonts.gstatic.com", "img-src 'self' blob: data:", "connect-src 'self'", "manifest-src 'self'"):
        assert need in csp, need
    assert h["X-Frame-Options"] == "DENY"
    assert c.get("/index.html", cookie=True)[2] == body
    for path, ctype in (("/skins/x.css", "text/css"), ("/skins/skins.js", "text/javascript"), ("/urdf/so101.urdf", "application/xml"), ("/urdf/assets/part.stl", "model/stl"), ("/icon.svg", "image/svg+xml")):
        st, h, b = c.get(path, cookie=True)
        assert st == 200 and h["Content-Type"] == ctype and h["Cache-Control"] == "public, max-age=3600" and len(b) == int(h["Content-Length"]), path
    assert c.get("/skins/x.css", cookie=True)[2] == b":root{--acc:#ffb347}"
    st, h, b = c.req("HEAD", "/skins/x.css", cookie=True)
    assert st == 200 and b == b"" and h["Content-Length"] == str(len(":root{--acc:#ffb347}"))


def test_pad_never_serves_the_rest_of_the_folder(gateway):
    c = gateway("pad")
    for path in ("/.phone_token", "/server.py", "/phone.html", "/skins/", "/skins/missing.css", "/urdf/", "/nope", "/events.log"):
        st, _, body = c.get(path, cookie=True)
        assert st == 404 and b"never" not in body, path


def test_pad_manifest_is_served_with_the_token_as_start_url(gateway):
    c = gateway("pad")
    st, h, body = c.get("/manifest.webmanifest", cookie=True)
    assert st == 200 and h["Content-Type"] == "application/manifest+json" and h["Cache-Control"] == "no-store"
    m = json.loads(body)
    assert m["name"] == "SO-101 Lab" and m["display"] == "fullscreen" and m["orientation"] == "landscape" and m["icons"][0]["src"] == "/icon.svg"
    assert m["start_url"] == f"/?k={TOKEN}", "the home-screen app has its own cookie jar: the start URL re-presents the token once"
    assert c.get("/manifest.webmanifest")[0] == 401


def test_pad_forwards_api_gets_with_query_status_and_type(gateway, upstream):
    c = gateway("pad")
    st, h, body = c.get("/api/locate?u=1.5&v=2&h_mm=0&cam=front", cookie=True)
    assert st == 200 and h["Content-Type"] == "application/json" and h["Cache-Control"] == "no-store" and json.loads(body)["path"] == "/api/locate?u=1.5&v=2&h_mm=0&cam=front"
    st, h, body = c.get(f"/api/state?k={TOKEN}&x=1")
    assert st == 200 and json.loads(body)["path"] == "/api/state?x=1", "the token never travels upstream"
    st, h, body = c.get("/api/boom", cookie=True)
    assert st == 500 and json.loads(body) == {"ok": False, "err": "boom"} and h["Content-Type"] == "application/json"
    st, h, body = c.get("/api/nolength", cookie=True)
    assert st == 200 and json.loads(body) == {"nolen": True}, "a body without Content-Length is streamed to the close"
    st, h, body = c.get("/cam/front.jpg?123", cookie=True)
    assert st == 200 and h["Content-Type"] == "image/jpeg" and body.startswith(b"\xff\xd8") and h["Cache-Control"] == "no-store"
    assert c.get("/api/nothing-here", cookie=True)[0] == 200, "pad mode forwards every /api path; the upstream decides"
    assert c.get("/x/api/state", cookie=True)[0] == 404


def test_pad_forwards_the_servo_telemetry_feed_and_phone_mode_does_not(gateway, upstream):
    """The telemetry panel on the iPad polls /api/tele?since=...; pad mode forwards it, the read-only phone page never needed it (docs/telemetry.md)."""
    st, h, body = gateway("pad").get("/api/tele?since=12.5", cookie=True)
    assert st == 200 and json.loads(body)["path"] == "/api/tele?since=12.5"
    assert gateway("phone").get("/api/tele", cookie=True)[0] == 404


def test_pad_forwards_posts_with_body_content_type_and_status(gateway, upstream):
    c = gateway("pad")
    p = json.dumps({"cmd": "pause"}).encode()
    st, h, body = c.post("/api/record/ctl", p, cookie=True)
    assert st == 200 and h["Content-Type"] == "application/json"
    r = json.loads(body); assert r == {"method": "POST", "path": "/api/record/ctl", "body": '{"cmd": "pause"}', "ctype": "application/json"}
    st, _, body = c.post("/api/arm/teleop", None, ctype=None, cookie=True)   # index.html posts the arm modes with no body at all
    assert st == 200 and json.loads(body)["body"] == ""
    st, h, body = c.post("/api/boom", b"x", cookie=True)
    assert st == 503 and body == b"down" and h["Content-Type"].startswith("text/plain")
    assert [h[:2] for h in upstream.hits] == [("POST", "/api/record/ctl"), ("POST", "/api/arm/teleop"), ("POST", "/api/boom")]
    assert upstream.hits[0][2] == p and upstream.hits[0][3] == "application/json"
    assert c.post("/cam/top.jpg", b"", cookie=True)[0] == 404 and c.post("/skins/x.css", b"", cookie=True)[0] == 404


def test_pad_post_refuses_a_foreign_origin_without_the_header(gateway, upstream):
    c = gateway("pad")
    assert c.post("/api/arm/stop", b"", cookie=True, headers={"Origin": "http://evil.example"})[0] == 403
    assert c.post("/api/arm/stop", b"", cookie=True, headers={"Origin": "http://evil.example", "X-SO101": "1"})[0] == 200
    assert c.post("/api/arm/stop", b"", cookie=True, headers={"Origin": f"http://127.0.0.1:{c.port}"})[0] == 200, "the page's own origin"
    assert len(upstream.hits) == 2


def test_pad_streams_mjpeg_without_buffering(gateway):
    c = gateway("pad")
    r, conn = c.get("/cam/top", cookie=True, read=False, timeout=5)
    assert r.status == 200 and r.headers["Content-Type"].startswith("multipart/x-mixed-replace") and r.headers.get("Content-Length") is None
    t0 = time.time(); got = b""
    while b"FRAME2" not in got and time.time() - t0 < 4: got += r.read1(4096) or b""
    conn.close()
    assert b"FRAME0" in got and b"FRAME2" in got and time.time() - t0 < 4, "frames arrive as they are produced; the gateway never waits for the end of the stream"


def test_pad_upstream_down_is_a_502_json(gw, dash_root):
    import os
    H = gw.make_handler(TOKEN, upstream="http://127.0.0.1:9", page=os.path.join(dash_root, "phone.html"), mode="pad", root=dash_root)   # port 9: nothing listens
    s = gw.Srv(("127.0.0.1", 0), H)
    import threading
    t = threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True); t.start()
    try:
        from gatewayhelpers import GwClient
        c = GwClient(s.server_address[1], TOKEN)
        st, h, body = c.get("/api/state", cookie=True)
        assert st == 502 and json.loads(body)["error"].startswith("dashboard server not running") and h["Content-Type"] == "application/json"
        assert c.get("/", cookie=True)[0] == 200, "the page itself still comes from disk"
    finally: s.shutdown(); s.server_close(); t.join(2)


def test_mode_and_port_defaults(gw, monkeypatch):
    assert gw.MODE == "phone" and gw.PORTS == {"phone": 8770, "pad": 8780}
    assert gw.static_path("/tmp/x", "/skins/a.css") == "/tmp/x/skins/a.css" and gw.static_path("/tmp/x", "/") == "/tmp/x/index.html"
    assert gw.static_path("/tmp/x", "/skins/../a.css") is None and gw.static_path("/tmp/x", "/server.py") is None and gw.static_path("/tmp/x", "/skins/") is None
    assert gw.denied("/api/shell") and gw.denied("/API/TERM") and gw.denied("/x/ttyd/") and gw.denied("/a/../b") and gw.denied("/api/%2e%2e/x") and not gw.denied("/api/state")
