"""Stub backend for the dashboard UI tests.

Serves dashboard/ statically (index.html and friends) and answers every /api/* route from
mutable Python objects, so a test can flip the Jetson offline, start a recording or break a
route between two poll cycles. Threaded, binds 127.0.0.1:0 (never the live 8765/8766/9333).
"""
import http.server
import json
import os
import socketserver
import threading
import time
from collections import Counter

DASH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dashboard")


class Stub:
    def __init__(self):
        self.routes = {}      # route prefix -> object, JSON-encoded on every GET (or a callable(path) returning one)
        self.fail = {}        # route prefix -> HTTP status to answer with (HTML body, like a real server error)
        self.raw = {}         # route prefix -> raw bytes served as 200 application/json (for invalid JSON)
        self.delay = {}       # route prefix -> seconds to sleep before answering
        self.posts = []       # [(path, parsed body)] for every POST received
        self.hits = Counter()
        self.lock = threading.Lock()
        self.httpd = None
        self.thread = None

    # -- routing -------------------------------------------------------------
    def route_for(self, path):
        p = path.split("?")[0]
        keys = [k for k in self.routes if p.startswith(k)]
        return max(keys, key=len) if keys else None

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        stub = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=DASH, **k)

            def log_message(self, *a):
                pass

            def _answer(self, route, obj):
                with stub.lock:
                    stub.hits[route] += 1
                    d, fail, raw = stub.delay.get(route, 0), stub.fail.get(route), stub.raw.get(route)
                if d:
                    time.sleep(d)
                if fail:
                    body = b"<html><body><h1>stub failure</h1></body></html>"
                    self.send_response(fail)
                    self.send_header("Content-Type", "text/html")
                else:
                    body = raw if raw is not None else json.dumps(obj).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                route = stub.route_for(self.path)
                if route:
                    with stub.lock:
                        obj = stub.routes[route]
                    if callable(obj):                       # e.g. /api/locate: computed from the query string
                        obj = obj(self.path)
                    return self._answer(route, obj)
                if self.path == "/":
                    self.path = "/index.html"
                super().do_GET()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(n) if n else b""
                try:
                    parsed = json.loads(body or b"{}")
                except Exception:
                    parsed = body.decode("utf8", "replace")
                path = self.path.split("?")[0]
                with stub.lock:
                    stub.posts.append((path, parsed))
                route = stub.route_for(path) or path
                self._answer(route, {"ok": True})

        class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

            def server_bind(self):  # skip HTTPServer's getfqdn() reverse-DNS lookup (stalls on this Mac)
                socketserver.TCPServer.server_bind(self)
                self.server_name = "localhost"
                self.server_port = self.server_address[1]

        self.httpd = Srv(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    @property
    def port(self):
        return self.httpd.server_address[1]

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    # -- helpers for tests ---------------------------------------------------
    def posts_to(self, path):
        with self.lock:
            return [b for p, b in self.posts if p == path]

    def clear_posts(self):
        with self.lock:
            self.posts.clear()

    def wait_post(self, pred, timeout=3.0):
        """Return the first POST (path, body) satisfying pred, or raise after timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                for p in self.posts:
                    if pred(*p):
                        return p
            time.sleep(0.02)
        raise AssertionError(f"no matching POST within {timeout}s; got {self.posts}")

    def wait_hits(self, routes, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if all(self.hits[r] for r in routes):
                    return
            time.sleep(0.02)
        raise AssertionError(f"routes never polled: {[r for r in routes if not self.hits[r]]}")
