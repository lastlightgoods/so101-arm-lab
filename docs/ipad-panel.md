# The iPad panel: the wall dashboard as a touch control panel over the LAN

The wall dashboard (`dashboard/server.py` + `index.html`) listens only on `127.0.0.1:8765` on the Mac, on purpose: it
exposes a Jetson shell. `dashboard/phone.py` is the one thing that listens on the LAN. It has two modes:

| mode | command | port | what the browser gets |
|---|---|---|---|
| `phone` (default) | `so101-dash phone` or `python dashboard/phone.py [port]` | 8770 | `phone.html`, and only `/api/state`, two camera stills and `POST /api/arm/stop` |
| `pad` | `so101-dash pad` or `python dashboard/phone.py --pad [port]` (or `SO101_PHONE_MODE=pad`) | 8780 | the full `index.html` with every control, feed, skin and the 3D arm |

## Start it

```
so101-dash pad            # starts server.py if it is not running, then the gateway; prints the link once
# or, by hand, from the repo root:
~/so101/.venv/bin/python dashboard/phone.py --pad          # port 8780
~/so101/.venv/bin/python dashboard/phone.py --pad 8781     # another port
```

The gateway prints one line, for example

```
pad page on http://<mac-lan-ip>:8780/?k=Xy7...   (the full dashboard, /api and /cam forwarded (shell and terminal denied) to http://127.0.0.1:8765)
```

Open that link on the iPad once (same Wi-Fi). The gateway trades the `?k=` token for an `HttpOnly`, `SameSite=Strict`
cookie that lasts a year and redirects to the clean `/`. Without the cookie every path answers `401`. `so101-dash stop`
ends the server, the kiosk and the gateway together; the gateway's log is `/tmp/so101_pad.log`.

`SO101_PHONE_BIND` (default `0.0.0.0`) and `SO101_DASH_UPSTREAM` (default `http://127.0.0.1:8765`) are the only other knobs.

## Add to Home Screen (full screen, landscape, no Safari chrome)

1. On the iPad, in Safari, open the link above and let it redirect to `/`.
2. Tap the share button, then **Add to Home Screen**, keep the name **SO-101**, tap **Add**.
3. Open it from the home screen. It runs full screen (`display: fullscreen`, `apple-mobile-web-app-capable`,
   `black-translucent` status bar, `viewport-fit=cover` with safe-area padding), landscape.

A home-screen web app on iOS has its own cookie jar, separate from Safari's. The manifest the gateway serves therefore
carries `start_url: /?k=<token>`, so the first launch from the icon presents the token once more and the app gets its
own cookie. (The manifest is itself behind the cookie, so only an authenticated Safari ever sees it; `server.py` serves
the plain manifest with `start_url: /` for the kiosk.) The icon is `dashboard/icon.svg` / `icon.png`.

## What the pad gateway forwards, and what it never does

Forwarded to `server.py`, with the request body, content type and query string, and with the upstream status code,
content type and body streamed back (`Cache-Control: no-store`):

- every `GET` and `POST` under `/api/` (arm modes, STOP, record, calibration, scene, lock, Spotify, ...)
- every `GET` under `/cam/` (the polled `.jpg` stills and the MJPEG `/cam/<name>` streams, which are relayed frame by frame)

Served from `dashboard/` on the Mac, with normal caching (`public, max-age=3600`; the page itself is `no-cache`):
`/` and `/index.html`, `/skins/*`, `/urdf/*`, `/simroll/*`, `/img/*`, `/manifest.webmanifest`, `/icon.svg`, `/icon.png`.
Nothing else on disk is reachable (`.phone_token`, `server.py`, `events.log` all answer `404`).

Denied with `404` in both modes, authenticated or not: `/api/shell*`, `/api/term*`, any path containing `ttyd`,
`terminal` or `..` (also percent-encoded). The page copes: its shell panel shows `idle` and the terminal panel stays
hidden, exactly as when ttyd is down. Writes need the cookie; a cross-site page that posts with a foreign `Origin` is
refused (`403`) even if a cookie somehow came along.

The CSP on the page allows what it really needs and nothing more: inline scripts and styles, the three.js and
urdf-loader modules from cdnjs and jsdelivr, Google Fonts, `blob:`/`data:` images and https album art, `connect-src 'self'`.

## Touch and layout

On a landscape tablet viewport (`orientation: landscape` and width <= 1366 px: iPad Air 1180x820, iPad 1024x768) the
page stops being the fixed portrait kiosk stack and scrolls: feeds first, then the arm panel (modes, transport, 3D
stage, joints) with the training, datasets and policies tables three abreast under it, then status, gauges, the Claude
shell, the attract strip and the log. Buttons are at least 46 px tall, the key tags are hidden, and the **STOP** button
is pinned bottom-right, 168x72 px, above everything except the lock screen and the record dialog; it calls the same
`POST /api/arm/stop` as on the wall. The 3D stage uses three.js OrbitControls: one finger orbits, two fingers pinch to
zoom and pan; touching the stage pauses the auto-rotate, which resumes six seconds after the last finger lifts. Tap a
feed to measure a point on the mat, as on the wall. There are no hover-only controls.

The same rules apply to any Chrome window narrower than 1366 px in landscape, which is how the layout is checked
without an iPad (`http://127.0.0.1:8765` resized to 1180x820).

## Rotate or revoke the token

Delete `dashboard/.phone_token` and restart the gateway (`so101-dash pad` again): a new token is written (mode 600) and
printed, every old cookie and every old home-screen icon stops working (`401`). Re-open the new link on the iPad and
add the icon again. `SO101_PHONE_TOKEN=...` in the environment overrides the file (used by the tests and for a
one-off session).

## Tests

`tests/server/test_phone_gateway.py` (fixtures in `tests/server/gatewayhelpers.py`) runs both modes against a stub
upstream: the token/cookie handshake, 401 for unauthenticated writes, the shell/terminal/`..` denials with a valid
cookie, the unchanged phone allowlist, static files and their types, the manifest, GET/POST forwarding with body,
query, status and content type, the MJPEG relay, and the 502 when `server.py` is down. `tests/server/test_http.py`
covers `/manifest.webmanifest` on `server.py`.
