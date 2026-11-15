# The panel in its own window: SO-101 Panel.app

The wall dashboard used to run in a Chrome kiosk (`so101-dash`). On 14 Nov 2026 the Mac bogged down, the Chrome the
agents also use hung (no clicks, no approval prompts), and the Mac had to be rebooted. `SO-101 Panel.app` takes the panel
out of Chrome: one native macOS window with one WebKit view, full screen on the vertical monitor, sharing nothing with
any browser. A heavy or hung panel can only take itself down.

## Launch

```
so101-dash app        # starts server.py + the ttyd terminal if needed, builds the app if PanelApp.swift changed, opens it
so101-dash stop       # stops the server, the terminal, the gateway, the Chrome kiosk, the Safari panel window and the app
```

To make a plain `so101-dash` open the app, put `export SO101_DASH_BROWSER=app` in `env.sh` (or `safari`, or `chrome`).
The default is still the Chrome kiosk.

In the app: **Cmd-R** reloads the page, **Ctrl-Cmd-F** leaves or enters full screen, **Cmd-Q** quits. It is also in
`~/Applications`, so Spotlight finds "SO-101 Panel".

What the app does on its own:

- picks the portrait monitor (else any screen without the menu bar) and goes full screen there, like the kiosk did;
- keeps retrying every 3 s while `server.py` is not up, instead of showing an error page;
- reloads the page if its WebContent process crashes or macOS kills it for memory;
- keeps the page's `localStorage` (skin, selections) between launches;
- sends links that would open a new window to the default browser.

## Build

`dashboard/mac/build.sh [out-dir]` compiles `dashboard/mac/PanelApp.swift` with `swiftc` (Xcode command line tools
only, no Xcode project) into `~/Applications/SO-101 Panel.app`, with the panel icon and an ad-hoc signature.
`so101-dash app` runs it whenever the Swift source is newer than the built app. `tests/ui/test_mac_panel.py` builds it
into a temp folder and checks the launcher; it never opens a window.

## For agents: screenshots and memory

There is no DevTools port, so `dashboard/cdp.js` only drives the Chrome kiosk. For the app:

- `kill -USR1 $(pgrep -f "SO-101 Panel.app/Contents/MacOS/SO101Panel")` writes the page to `/tmp/so101-panel.png`
  (1440x2560 on the vertical monitor);
- `/tmp/so101-panel.webpid` holds the page's WebContent process id: `footprint $(cat /tmp/so101-panel.webpid)` gives its
  memory;
- Safari's Develop menu can attach Web Inspector to the app's page.

## Safari instead

`so101-dash safari` opens the panel in a new Safari window sized to monitor 2. macOS asks once whether the terminal may
control Safari (System Settings, Privacy & Security, Automation). Ctrl-Cmd-F makes the window full screen.

## Memory

Measured on 14 Nov 2026, live arm in teleop with all three cameras, sampled once a minute with
`footprint $(cat /tmp/so101-panel.webpid)` for the first 53 minutes after launch:

| | WebContent footprint |
|---|---|
| min / max | 568 / 750 MB |
| mean | 632 MB |
| first 10 min mean | 646 MB |
| last 10 min mean | 627 MB |

No upward trend. The page keeps bounded buffers: servo telemetry rows are trimmed to the last 11 minutes, the attract
game's particles, shots and foes are filtered every frame, mat-map meshes are disposed when the scene changes, the shell
panel shows the last 5 calls, and the camera stills are `no-store`. If the page process is ever killed for memory, the
app reloads it.
