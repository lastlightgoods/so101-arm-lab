"""SO-101 LAN gateway for the dashboard: a phone page, or (pad mode) the full wall dashboard on an iPad.
    python phone.py [port]            phone mode (default, port 8770): serves phone.html, forwards a read-only allowlist
    python phone.py --pad [port]      pad mode (port 8780; or SO101_PHONE_MODE=pad): serves index.html + its assets, forwards /api and /cam
The kiosk server (server.py) stays on 127.0.0.1:8765; this gateway is the only thing that listens on the LAN.
Phone mode forwards only:
    GET  /api/state, /cam/top.jpg, /cam/wrist.jpg      (read-only status and camera stills)
    POST /api/arm/stop                                  (the one control: stop whatever the arm is doing)
Pad mode (docs/ipad-panel.md) forwards every GET/POST under /api/ and /cam/ with its body, query, status and content type,
streamed (the MJPEG /cam/<name> streams never end), and serves index.html, /skins, /urdf, /simroll, /img, the manifest and
icon from this folder.  Denied in every mode, authenticated or not: /api/shell*, /api/term*, anything ttyd/terminal, any
path with '..'.  So the Jetson shell stays Mac-only.
Auth: a token from SO101_PHONE_TOKEN or dashboard/.phone_token (made on first run, mode 600; delete the file to rotate).
Open the link once with ?k=<token>; the gateway swaps it for an HttpOnly cookie and redirects to a clean URL."""
import hmac, http.server, json, mimetypes, os, secrets, socket, socketserver, sys, urllib.error, urllib.request
from http.cookies import CookieError, SimpleCookie
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "phone.html")
TOKEN_FILE = os.path.join(HERE, ".phone_token")
UPSTREAM = os.environ.get("SO101_DASH_UPSTREAM", "http://127.0.0.1:8765")
BIND = os.environ.get("SO101_PHONE_BIND", "0.0.0.0")
MODE = "pad" if os.environ.get("SO101_PHONE_MODE", "phone").strip().lower() == "pad" else "phone"
PORTS = {"phone": 8770, "pad": 8780}
GET_ROUTES = {"/api/state", "/cam/top.jpg", "/cam/wrist.jpg"}
POST_ROUTES = {"/api/arm/stop"}
DENY_PREFIX = ("/api/shell", "/api/term")      # why: the Jetson shell and the ttyd terminal must never be reachable from the LAN
DENY_WORDS = ("ttyd", "terminal", "..")
STATIC_DIRS = ("/skins/", "/urdf/", "/simroll/", "/img/")                       # pad mode: the subtrees index.html loads from
STATIC_FILES = {"/index.html", "/manifest.webmanifest", "/icon.svg", "/icon.png"}  # ...and nothing else on disk (.phone_token, server.py, events.log stay private)
MIME = {".webmanifest": "application/manifest+json", ".stl": "model/stl", ".urdf": "application/xml", ".svg": "image/svg+xml", ".js": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8"}
COOKIE = "so101k"
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
# why: the full page imports three.js / urdf-loader as ES modules from cdnjs + jsdelivr, loads Google Fonts from the skins' @imports,
# shows Spotify album art from https, and draws blob:/data: textures; everything it talks to (api, cams, urdf) is same-origin
PAD_CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
           "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' data: https://fonts.gstatic.com; "
           "img-src 'self' blob: data: https:; connect-src 'self'; manifest-src 'self'; worker-src blob:; "
           "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

def load_token(path=TOKEN_FILE):
    t = os.environ.get("SO101_PHONE_TOKEN", "").strip()
    if t: return t
    try:
        with open(path) as f: t = f.read().strip()
    except FileNotFoundError: t = ""
    if len(t) < 16:
        t = secrets.token_urlsafe(18)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f: f.write(t + "\n")
    return t

def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("10.255.255.255", 1)); return s.getsockname()[0]  # no packet is sent; just picks the outbound interface
    except OSError: return "127.0.0.1"
    finally: s.close()

def denied(path):
    """True for the paths no mode ever forwards or serves, before and regardless of auth."""
    p = unquote(path).lower()
    return p.startswith(DENY_PREFIX) or any(w in p for w in DENY_WORDS)

def static_path(root, path):
    """Pad mode: the on-disk file for a static URL path, or None when it is not one of the allowed files / subtrees."""
    if path == "/": path = "/index.html"
    if path not in STATIC_FILES and not path.startswith(STATIC_DIRS): return None
    if path.endswith("/") or denied(path): return None
    full = os.path.normpath(os.path.join(root, unquote(path).lstrip("/")))
    return full if full.startswith(os.path.normpath(root) + os.sep) else None   # why: normpath (not realpath): dashboard/simroll is a symlink out of the tree on purpose

def make_handler(token, upstream=UPSTREAM, page=PAGE, mode=None, root=HERE):
    tok = token.encode(); mode = mode or MODE; pad = mode == "pad"
    class H(http.server.BaseHTTPRequestHandler):
        server_version = "so101-phone"; sys_version = ""
        def log_message(self, *a): pass
        def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=(), cache="no-store"):
            self.send_response(code)
            for k, v in (("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", cache),
                         ("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer"), *extra): self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD": self.wfile.write(body)
        def _auth(self):
            q = parse_qs(urlsplit(self.path).query).get("k", [""])[0]
            if q and hmac.compare_digest(q.encode(), tok): return "query"
            try: c = SimpleCookie(self.headers.get("Cookie", ""))
            except CookieError: return None
            v = c[COOKIE].value if COOKIE in c else ""
            return "cookie" if v and hmac.compare_digest(v.encode(), tok) else None
        def _deny(self): self._send(401, f"Not authorised. Open the link printed by `so101-dash {mode}` on the Mac.\n".encode())
        def _forward(self, method, path, body=None, ctype=None):
            hdr = {"Content-Type": ctype} if ctype else {}
            req = urllib.request.Request(upstream + path, method=method, data=(body or b"") if method == "POST" else None, headers=hdr)
            try:
                with urllib.request.urlopen(req, timeout=25 if method == "POST" else (30 if pad else 5)) as r: self._relay(r.status, r)   # pad: per-read timeout, so an idle MJPEG stream ends instead of pinning a thread
            except urllib.error.HTTPError as e: self._relay(e.code, e)
            except (urllib.error.URLError, OSError):
                self._send(502, json.dumps({"error": "dashboard server not running on the Mac"}).encode(), "application/json")
        def _relay(self, code, r):
            """Copy status + content type, then stream the body: a /cam/<name> MJPEG stream has no end, so it is never read whole."""
            ctype = r.headers.get("Content-Type", "application/octet-stream"); length = r.headers.get("Content-Length")
            if not pad or length is not None and int(length) <= 1 << 20: return self._send(code, r.read() or b"", ctype)
            self.send_response(code)
            for k, v in (("Content-Type", ctype), ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"), ("Connection", "close")): self.send_header(k, v)
            if length is not None: self.send_header("Content-Length", length)
            self.end_headers()
            if self.command == "HEAD": return
            try:
                while True:
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)   # why: read1 returns what has arrived (a frame), read would wait to fill 64 KB
                    if not chunk: break
                    self.wfile.write(chunk); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        def _static(self, path):
            full = static_path(root, path)
            if not full or not os.path.isfile(full): return self._send(404, b"not here\n")
            ext = os.path.splitext(full)[1].lower(); ctype = MIME.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
            if path == "/manifest.webmanifest": return self._manifest(full)
            html = ext == ".html"; extra = (("Content-Security-Policy", PAD_CSP), ("X-Frame-Options", "DENY")) if html else ()
            try:
                with open(full, "rb") as f: body = f.read()
            except OSError: return self._send(404, b"not here\n")
            self._send(200, body, ctype, extra, cache="no-cache" if html else "public, max-age=3600")   # why: assets cache normally; the page itself is revalidated so edits show up on the next open
        def _manifest(self, full):
            try:
                with open(full, "rb") as f: m = json.loads(f.read())
            except (OSError, ValueError): return self._send(404, b"not here\n")
            m["start_url"] = "/?" + urlencode({"k": token})   # why: an iOS home-screen web app does not share Safari's cookies; the start URL re-presents the token once and the gateway swaps it for a cookie in the app's own jar
            self._send(200, json.dumps(m).encode(), MIME[".webmanifest"], cache="no-store")
        def do_GET(self):
            path = urlsplit(self.path).path
            if denied(path): return self._send(404, b"not here\n")
            how = self._auth()
            if not how: return self._deny()
            if how == "query" and path in ("/", "/index.html", "/phone.html"):  # trade the link token for a cookie, then drop it from the address bar and history
                return self._send(303, extra=(("Location", "/"), ("Set-Cookie", f"{COOKIE}={token}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict")))
            if not pad:
                if path in ("/", "/index.html", "/phone.html"):
                    try:
                        with open(page, "rb") as f: body = f.read()
                    except OSError: return self._send(500, b"phone.html missing\n")
                    return self._send(200, body, "text/html; charset=utf-8", (("Content-Security-Policy", CSP), ("X-Frame-Options", "DENY")))
                if path in GET_ROUTES: return self._forward("GET", path)
                return self._send(404, b"not here\n")
            if path.startswith(("/api/", "/cam/")):
                q = urlsplit(self.path).query
                if how == "query": q = urlencode([(k, v) for k, v in parse_qs(q, keep_blank_values=True).items() if k != "k"], doseq=True)   # the token never travels upstream
                return self._forward("GET", path + ("?" + q if q else ""))
            self._static(path)
        do_HEAD = do_GET
        def do_POST(self):
            path = urlsplit(self.path).path
            if denied(path): return self._send(404, b"not here\n")
            if not self._auth(): return self._deny()
            if not pad:
                if path not in POST_ROUTES: return self._send(404, b"not here\n")
                if self.headers.get("X-SO101") != "1": return self._send(403, b"missing X-SO101 header\n")  # forces a CORS preflight for cross-site posts
                return self._forward("POST", path)
            if not path.startswith("/api/"): return self._send(404, b"not here\n")
            origin = self.headers.get("Origin")   # why: index.html's fetches send no custom header; a cross-site page always sends Origin, so one that does not match Host is refused (the SameSite cookie already keeps its cookie home)
            if origin and self.headers.get("X-SO101") != "1" and urlsplit(origin).netloc != self.headers.get("Host", ""): return self._send(403, b"cross-site post refused\n")
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1 << 20: return self._send(413, b"too big\n")
            self._forward("POST", path, self.rfile.read(n) if n else b"", self.headers.get("Content-Type"))
    return H

class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
    def server_bind(self):  # skip HTTPServer's getfqdn() reverse-DNS lookup (35 s stall on this Mac)
        socketserver.TCPServer.server_bind(self); self.server_name = "localhost"; self.server_port = self.server_address[1]

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--pad" in args: MODE = "pad"
    if "--phone" in args: MODE = "phone"
    nums = [a for a in args if a.isdigit()]
    port = int(nums[0]) if nums else PORTS[MODE]
    token = load_token()
    srv = Srv((BIND, port), make_handler(token, mode=MODE))
    what = "the full dashboard, /api and /cam forwarded (shell and terminal denied)" if MODE == "pad" else "a read-only allowlist"
    print(f"{MODE} page on http://{lan_ip()}:{port}/?k={token}   ({what} to {UPSTREAM})", flush=True)
    srv.serve_forever()
