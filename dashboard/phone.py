"""SO-101 phone gateway.  python phone.py [port]   ->  http://<mac-lan-ip>:8770/?k=<token>
A small LAN-facing front door for the phone page.  The kiosk server (server.py) stays on 127.0.0.1:8765;
this gateway serves phone.html and forwards only a short allowlist to it:
    GET  /api/state, /cam/top.jpg, /cam/wrist.jpg      (read-only status and camera stills)
    POST /api/arm/stop                                  (the one control: stop whatever the arm is doing)
Everything else is a 404, so the Jetson shell, the other arm modes, Spotify and the lock stay Mac-only.
Auth: a token from SO101_PHONE_TOKEN or dashboard/.phone_token (made on first run, mode 600).  Open the link once
with ?k=<token>; the gateway swaps it for an HttpOnly cookie and redirects to a clean URL."""
import hmac, http.server, json, os, secrets, socket, socketserver, sys, urllib.error, urllib.request
from http.cookies import CookieError, SimpleCookie
from urllib.parse import parse_qs, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "phone.html")
TOKEN_FILE = os.path.join(HERE, ".phone_token")
UPSTREAM = os.environ.get("SO101_DASH_UPSTREAM", "http://127.0.0.1:8765")
BIND = os.environ.get("SO101_PHONE_BIND", "0.0.0.0")
GET_ROUTES = {"/api/state", "/cam/top.jpg", "/cam/wrist.jpg"}
POST_ROUTES = {"/api/arm/stop"}
COOKIE = "so101k"
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

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

def make_handler(token, upstream=UPSTREAM, page=PAGE):
    tok = token.encode()
    class H(http.server.BaseHTTPRequestHandler):
        server_version = "so101-phone"; sys_version = ""
        def log_message(self, *a): pass
        def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=()):
            self.send_response(code)
            for k, v in (("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-store"),
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
        def _deny(self): self._send(401, b"Not authorised. Open the link printed by `so101-dash phone` on the Mac.\n")
        def _forward(self, method, path):
            req = urllib.request.Request(upstream + path, method=method, data=b"" if method == "POST" else None)
            try:
                with urllib.request.urlopen(req, timeout=25 if method == "POST" else 5) as r:
                    self._send(r.status, r.read(), r.headers.get("Content-Type", "application/octet-stream"))
            except urllib.error.HTTPError as e: self._send(e.code, e.read() or b"", e.headers.get("Content-Type", "text/plain"))
            except (urllib.error.URLError, OSError):
                self._send(502, json.dumps({"error": "dashboard server not running on the Mac"}).encode(), "application/json")
        def do_GET(self):
            how = self._auth(); path = urlsplit(self.path).path
            if not how: return self._deny()
            if path in ("/", "/index.html", "/phone.html"):
                if how == "query":  # trade the link token for a cookie, then drop it from the address bar and history
                    return self._send(303, extra=(("Location", "/"), ("Set-Cookie", f"{COOKIE}={token}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict")))
                try:
                    with open(page, "rb") as f: body = f.read()
                except OSError: return self._send(500, b"phone.html missing\n")
                return self._send(200, body, "text/html; charset=utf-8", (("Content-Security-Policy", CSP), ("X-Frame-Options", "DENY")))
            if path in GET_ROUTES: return self._forward("GET", path)
            self._send(404, b"not here\n")
        do_HEAD = do_GET
        def do_POST(self):
            if not self._auth(): return self._deny()
            path = urlsplit(self.path).path
            if path not in POST_ROUTES: return self._send(404, b"not here\n")
            if self.headers.get("X-SO101") != "1": return self._send(403, b"missing X-SO101 header\n")  # forces a CORS preflight for cross-site posts
            self._forward("POST", path)
    return H

class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
    def server_bind(self):  # skip HTTPServer's getfqdn() reverse-DNS lookup (35 s stall on this Mac)
        socketserver.TCPServer.server_bind(self); self.server_name = "localhost"; self.server_port = self.server_address[1]

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    token = load_token()
    srv = Srv((BIND, port), make_handler(token))
    print(f"phone page on http://{lan_ip()}:{port}/?k={token}   (forwarding a read-only allowlist to {UPSTREAM})", flush=True)
    srv.serve_forever()
