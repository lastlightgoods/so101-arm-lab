#!/bin/zsh
# dash.sh start|stop|open|url|safari|app|pad [port]|phone [port]  — SO-101 lab dashboard: local server + Chrome kiosk on monitor 2;
# `app` opens it in its own native window instead (SO-101 Panel.app, mac/PanelApp.swift, full screen on the vertical monitor,
# no browser shared with the agents: REQ 007); `safari` opens it in a Safari window on monitor 2 (needs Automation permission once).
# SO101_DASH_BROWSER=app|safari|chrome picks what a bare `so101-dash` opens (default chrome). docs/mac-panel.md
# `pad` / `phone` also start the LAN gateway (phone.py) and print its one-time link (docs/ipad-panel.md)
PORT=8765; TPORT=8766; UDD=/tmp/so101-dash-chrome; PY=$HOME/so101/.venv/bin/python; DIR=$HOME/so101/dashboard
geom() { $PY - <<'PY'
import ctypes, ctypes.util
cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
class R(ctypes.Structure): _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("w", ctypes.c_double), ("h", ctypes.c_double)]
cg.CGDisplayBounds.restype = R; cg.CGMainDisplayID.restype = ctypes.c_uint32
ids = (ctypes.c_uint32 * 8)(); n = ctypes.c_uint32(); cg.CGGetActiveDisplayList(8, ids, ctypes.byref(n)); main = cg.CGMainDisplayID()
ext = [cg.CGDisplayBounds(ids[i]) for i in range(n.value) if ids[i] != main]
b = ext[0] if ext else cg.CGDisplayBounds(main); print(int(b.x), int(b.y), int(b.w), int(b.h))
PY
}
up() {   # ttyd (the Jetson shell panel) and server.py, each only if it is not already running
  pgrep -f "ttyd -p $TPORT" >/dev/null || { nohup ttyd -p $TPORT -i 127.0.0.1 -W -t fontSize=17 -t disableLeaveAlert=true -t titleFixed=jetson -t cursorBlink=true \
    -t 'theme={"background":"#07090d","foreground":"#e8edf5","cursor":"#38f0c8","selectionBackground":"#1d4a40"}' \
    ssh -t jetson > /tmp/so101_ttyd.log 2>&1 & echo "terminal (ttyd) started on 127.0.0.1:$TPORT"; }
  pgrep -f "dashboard/server.py" >/dev/null || { cd $DIR && nohup $PY $DIR/server.py $PORT > /tmp/so101_dash.log 2>&1 & echo "server started (pid $!), log /tmp/so101_dash.log"; sleep 1.5; }
}
APP="$HOME/Applications/SO-101 Panel.app"
{ [ -z "$1" ] || [ "$1" = start ]; } && [ -n "$SO101_DASH_BROWSER" ] && set -- "$SO101_DASH_BROWSER"
case "$1" in
  stop) pkill -f "dashboard/server.py"; pkill -f "dashboard/phone.py"; pkill -f "ttyd -p $TPORT"; pkill -f "user-data-dir=$UDD"
    pgrep -xq Safari && osascript -e 'on run {u}' -e 'tell application "Safari"' -e 'repeat with i from (count of windows) to 1 by -1' \
      -e 'try' -e 'if URL of current tab of window i starts with u then close window i' -e 'end try' -e 'end repeat' -e 'end tell' -e 'end run' \
      "http://127.0.0.1:$PORT" 2>/dev/null   # closes only Safari windows showing the panel
    pkill -f "$APP/Contents/MacOS/SO101Panel" 2>/dev/null
    echo stopped;;
  url)  echo "http://127.0.0.1:$PORT";;
  pad|phone) MODE=$1; GPORT="${2:-$([ "$MODE" = pad ] && echo 8780 || echo 8770)}"   # pad: the full dashboard for an iPad; phone: the small read-only page. Both need the server up.
    pgrep -f "dashboard/server.py" >/dev/null || { cd $DIR && nohup $PY $DIR/server.py $PORT > /tmp/so101_dash.log 2>&1 & echo "server started (pid $!), log /tmp/so101_dash.log"; sleep 1.5; }
    pkill -f "dashboard/phone.py --$MODE" 2>/dev/null; cd $DIR && nohup $PY $DIR/phone.py --$MODE $GPORT > /tmp/so101_$MODE.log 2>&1 &
    sleep 1; cat /tmp/so101_$MODE.log; echo "(gateway log /tmp/so101_$MODE.log · token in $DIR/.phone_token, delete it to rotate · so101-dash stop ends it)";;
  beat) bpm="${2:-100}"; secs="${3:-34}"; cd $DIR && [ -f beat_$bpm.wav ] || python3 beat.py $bpm $secs >/dev/null; afplay beat_$bpm.wav & echo "beat $bpm bpm";;
  safari) up; read X Y W H <<< "$(geom)"; echo "monitor 2: ${W}x${H} at $X,$Y"   # Safari has no DevTools port: cdp.js only drives the Chrome kiosk
    osascript -e 'on run {u, x1, y1, x2, y2}' -e 'tell application "Safari"' -e 'make new document with properties {URL:u}' \
      -e 'set bounds of front window to {x1 as integer, y1 as integer, x2 as integer, y2 as integer}' -e 'activate' -e 'end tell' -e 'end run' \
      "http://127.0.0.1:$PORT" $X $Y $((X+W)) $((Y+H)) >/dev/null && echo "Safari window open on monitor 2 (Ctrl-Cmd-F for full screen)"
    echo "http://127.0.0.1:$PORT";;
  app) up   # (re)builds the app when PanelApp.swift is newer than it; Cmd-R reloads, Ctrl-Cmd-F leaves full screen, Cmd-Q quits
    [ "$APP/Contents/MacOS/SO101Panel" -nt "$DIR/mac/PanelApp.swift" ] || "$DIR/mac/build.sh" >/dev/null || { echo "build failed: run $DIR/mac/build.sh"; exit 1; }
    pgrep -f "$APP/Contents/MacOS/SO101Panel" >/dev/null && { open "$APP"; echo "SO-101 Panel already running (brought to front)"; } \
      || { open "$APP" && echo "SO-101 Panel opening full screen on the vertical monitor"; }
    echo "http://127.0.0.1:$PORT";;
  open|*) up
    [ "$1" = "open" ] || { pkill -f "user-data-dir=$UDD" 2>/dev/null; read X Y W H <<< "$(geom)"; echo "monitor 2: ${W}x${H} at $X,$Y"
      open -na "Google Chrome" --args --user-data-dir=$UDD --no-first-run --remote-debugging-port=9333 "--remote-allow-origins=*" --kiosk --window-position=$X,$Y --window-size=$W,$H --app=http://127.0.0.1:$PORT; }
    echo "http://127.0.0.1:$PORT";;
esac
