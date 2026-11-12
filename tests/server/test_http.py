"""The HTTP surface (class H on Srv), bound to 127.0.0.1:<ephemeral>. Never touches /api/term (it probes port 8766)."""
import json, os, time
import pytest

CONTROL_SOUP = "\x1b[31mERR\x1b[0m\r\n\x00\x07\x1b[2K\rRecording episode 1 ━━━╸ 42%\t\x7f é 🦖"


def test_state_has_now_lock_and_derived_online(srv, http):
    srv.state["jetson"].update(online=True, last=time.time())
    s, _ = http.get_json("/api/state")
    assert s["now"] == pytest.approx(time.time(), abs=2) and s["lock"]["locked"] is False and s["jetson"]["online"] is True
    assert set(s) >= {"jetson", "mac", "training", "datasets", "policies", "procs", "events", "arms", "spotify", "lock", "sim_arm", "virtual", "now"}
    srv.state["jetson"]["last"] = time.time() - 13   # tegrastats went quiet 13 s ago -> offline even though the loop has not noticed
    s, _ = http.get_json("/api/state")
    assert s["jetson"]["online"] is False and srv.state["jetson"]["online"] is True, "derived in the response, state untouched"
    srv.state["jetson"]["last"] = time.time() - 11
    assert http.get_json("/api/state")[0]["jetson"]["online"] is True


def test_state_is_valid_json_with_control_chars_in_events(srv, http):
    srv.event(CONTROL_SOUP); srv.LOCK["status"] = CONTROL_SOUP
    s, body = http.get_json("/api/state")
    assert s["events"][0]["msg"] == CONTROL_SOUP and s["lock"]["status"] == CONTROL_SOUP
    assert all(b >= 0x20 for b in body), "no raw control bytes on the wire (browsers' JSON.parse would reject them)"


def test_hist_returns_the_ring(srv, http):
    srv.HIST.extend([[1762550000, 42, 7, 24, 58, 49.3, 6.28], [1762550002, 0, None, 3, 58, 48.0, 5.1]])
    h, _ = http.get_json("/api/hist")
    assert h == [[1762550000, 42, 7, 24, 58, 49.3, 6.28], [1762550002, 0, None, 3, 58, 48.0, 5.1]]
    assert http.get("/api/hist")[1]["Cache-Control"] == "no-store"


def test_hist_empty(http):
    assert http.get_json("/api/hist")[0] == []


def test_record_returns_rec_and_round_trips_log_escapes(srv, http):
    """Today's real 'Invalid control character' decode failure: the log tail carries ANSI / CR / NUL from the pty.
    json.dumps escapes every char < 0x20 as \\uXXXX, so whatever the tail contains, the body must parse back exactly."""
    srv.REC.update(running=True, name="cube", log=CONTROL_SOUP, err=CONTROL_SOUP[:40], phase="recording", episode=1, _nrec=1)
    r, body = http.get_json("/api/record")
    assert r["log"] == CONTROL_SOUP and r["err"] == CONTROL_SOUP[:40] and r["running"] is True and r["name"] == "cube"
    assert json.loads(body) == json.loads(json.dumps(srv.REC))
    assert all(b >= 0x20 for b in body) and b"\\u001b" in body and b"\\u0000" in body
    assert r == {**srv.REC}, "the whole REC dict is served, private _keys included"


def test_record_json_dumps_never_emits_what_loads_rejects(srv):
    for log in [CONTROL_SOUP, "".join(chr(i) for i in range(0x20)) * 3, "\udcff lone surrogate", "x" * 7000]:
        srv.REC["log"] = log[-1500:]
        assert json.loads(json.dumps(srv.REC)) == srv.REC


def test_record_ctl_endpoint(srv, http, fake, said):
    assert http.post_json("/api/record/ctl", {"cmd": "bogus"}) == {"ok": False, "err": "unknown command"}
    assert http.post_json("/api/record/ctl", {"cmd": "next"}) == {"ok": False, "err": "not recording"}
    srv.REC["running"] = True; fake.on("/tmp/so101_record_ctl", "")
    assert http.post_json("/api/record/ctl", {"cmd": "pause"}) == {"ok": True, "err": ""}
    assert fake.ssh == ["printf p > /tmp/so101_record_ctl"] and said() == []


def test_record_start_endpoint(srv, http, fake, events):
    fake.on("test -f", "no\n"); fake.on("record_ctl.py", "")
    assert http.post_json("/api/record", {"name": "cube", "task": "pick", "episodes": 3, "episode_s": 20, "reset_s": 5}) == {"ok": True}
    assert any('so101-record cube "pick" 3' in c for c in fake.ssh) and srv.REC["running"]
    assert http.post_json("/api/record", {"name": "cube", "task": "pick"}) == {"ok": False, "err": "already recording"}


def test_record_start_endpoint_bad_number_is_a_json_error_not_a_dropped_connection(srv, http, fake):
    assert http.post_json("/api/record", {"name": "cube", "task": "pick", "episodes": ""})["ok"] is False
    assert fake.calls == []


def test_record_apply_endpoint_requires_running(http):
    assert http.post_json("/api/record/apply", {"episodes": 5}) == {"ok": False, "err": "not recording"}


def test_post_malformed_body_and_unknown_endpoint(http):
    assert http.post_json("/api/record", raw=b"{not json") == {"ok": False, "err": "name and task are required"}
    assert http.post_json("/api/record", raw=b"") == {"ok": False, "err": "name and task are required"}
    assert http.post_json("/api/nothing", {}) == {"ok": False, "err": "no such endpoint"}


def test_lock_endpoints(srv, http, events):
    assert http.get_json("/api/lock")[0]["locked"] is False
    assert http.post_json("/api/lock") == {"ok": True}
    L, _ = http.get_json("/api/lock")
    assert L["locked"] is True and L["status"] == "NO SIGNAL" and L["progress"] == 0.0 and L["eye"] is None
    assert events()[0] == "DASHBOARD LOCKED · button"
    assert http.post_json("/api/unlock") == {"ok": True} and http.get_json("/api/lock")[0]["status"] == "UNLOCKED"
    assert events()[0] == "IDENTITY CONFIRMED · UNLOCKED · override key"


def test_arm_mode_guards_over_http(srv, http, fake, said, events):
    assert http.post_json("/api/arm/bogus?x=1") == {"ok": False, "err": "unknown mode"}
    srv.REC["running"] = True
    r = http.post_json("/api/arm/free")
    assert r["ok"] is False and "recording holds the arm" in r["err"] and said() == ["Stop the recording first."]
    assert fake.calls == [] and events() == []


def test_spotify_unknown_cmd(http, fake):
    assert http.post_json("/api/spotify/bogus") == {"ok": False, "err": "unknown"} and fake.calls == []


def test_cams_listing_and_stills(srv, http):
    assert http.get_json("/api/cams")[0] == []
    assert http.get("/cam/top.jpg")[0] == 404
    srv.cam_frames["top"] = (b"\xff\xd8JPEG\xff\xd9", time.time()); srv.cam_frames["wrist"] = (b"\xff\xd8old\xff\xd9", time.time() - 10)
    assert http.get_json("/api/cams")[0] == ["top"], "stale frames (> 3 s) are not live"
    st, hdr, body = http.get("/cam/top.jpg")
    assert st == 200 and hdr["Content-Type"] == "image/jpeg" and body == b"\xff\xd8JPEG\xff\xd9" and hdr["Content-Length"] == "8"
    assert http.get("/cam/wrist.jpg")[0] == 404


def test_root_serves_index_from_here(srv, http, tmp_path):
    (tmp_path / "index.html").write_text("<h1>wall</h1>")
    st, hdr, body = http.get("/")
    assert st == 200 and body == b"<h1>wall</h1>" and hdr["Content-Type"].startswith("text/html")
    assert http.get("/../server.py")[0] == 404, "SimpleHTTPRequestHandler stays inside HERE"


def test_gallery_lists_jpgs_under_here(srv, http, tmp_path):
    os.makedirs(tmp_path / "img" / "cube"); (tmp_path / "img" / "cube" / "ep01.jpg").write_bytes(b"x"); (tmp_path / "img" / "cube" / "note.txt").write_bytes(b"x")
    assert http.get_json("/api/gallery")[0] == ["img/cube/ep01.jpg"]


def test_shell_endpoint_serves_last_ten(srv, http):
    srv.shell_calls.extend({"id": f"c{i}", "t": "", "desc": "", "cmd": f"echo {i}", "out": "", "done": True} for i in range(14))
    s, _ = http.get_json("/api/shell")
    assert [c["id"] for c in s] == [f"c{i}" for i in range(4, 14)]


def test_json_responses_have_content_length_and_no_store(http):
    for path in ("/api/state", "/api/hist", "/api/record", "/api/lock"):
        st, hdr, body = http.get(path)
        assert st == 200 and hdr["Content-Length"] == str(len(body)) and hdr["Cache-Control"] == "no-store", path


def test_manifest_is_served_from_here_as_manifest_json(srv, http, tmp_path):
    assert http.get("/manifest.webmanifest")[0] == 404, "no file under HERE: 404, never a crash"
    (tmp_path / "manifest.webmanifest").write_text(json.dumps({"name": "SO-101 Lab", "display": "fullscreen", "orientation": "landscape", "start_url": "/"}))
    st, hdr, body = http.get("/manifest.webmanifest")
    assert st == 200 and hdr["Content-Type"] == "application/manifest+json" and hdr["Cache-Control"] == "no-cache" and hdr["Content-Length"] == str(len(body))
    assert json.loads(body) == {"name": "SO-101 Lab", "display": "fullscreen", "orientation": "landscape", "start_url": "/"}


def test_real_manifest_and_icons_exist_next_to_index(srv):
    d = os.path.dirname(os.path.abspath(srv.__file__))
    m = json.load(open(os.path.join(d, "manifest.webmanifest")))
    assert m["name"] == "SO-101 Lab" and m["display"] == "fullscreen" and m["orientation"] == "landscape" and m["start_url"] == "/"
    assert m["theme_color"] == m["background_color"] == "#050607", "the tempest --bg token"
    for ic in m["icons"]: assert os.path.isfile(os.path.join(d, ic["src"].lstrip("/"))), ic
    html = open(os.path.join(d, "index.html")).read()
    for need in ('<link rel="manifest" href="/manifest.webmanifest">', 'apple-mobile-web-app-capable" content="yes"', 'apple-mobile-web-app-status-bar-style" content="black-translucent"', "viewport-fit=cover", "user-scalable=no", "env(safe-area-inset-", "-webkit-touch-callout:none"):
        assert need in html, need


def test_server_bind_skips_fqdn_and_reports_port(srv):
    s = srv.Srv(("127.0.0.1", 0), srv.H)
    try: assert s.server_name == "localhost" and s.server_port == s.server_address[1] > 0 and s.daemon_threads
    finally: s.server_close()
