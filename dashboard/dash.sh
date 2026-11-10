#!/bin/zsh
# dash.sh start|stop|open|url  — SO-101 lab dashboard: local server + Chrome kiosk on monitor 2
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
case "$1" in
  stop) pkill -f "dashboard/server.py"; pkill -f "ttyd -p $TPORT"; pkill -f "user-data-dir=$UDD"; echo stopped;;
  url)  echo "http://127.0.0.1:$PORT";;
  beat) bpm="${2:-100}"; secs="${3:-34}"; cd $DIR && [ -f beat_$bpm.wav ] || python3 beat.py $bpm $secs >/dev/null; afplay beat_$bpm.wav & echo "beat $bpm bpm";;
  open|*)
    pgrep -f "ttyd -p $TPORT" >/dev/null || { nohup ttyd -p $TPORT -i 127.0.0.1 -W -t fontSize=17 -t disableLeaveAlert=true -t titleFixed=jetson -t cursorBlink=true \
      -t 'theme={"background":"#07090d","foreground":"#e8edf5","cursor":"#38f0c8","selectionBackground":"#1d4a40"}' \
      ssh -t jetson > /tmp/so101_ttyd.log 2>&1 & echo "terminal (ttyd) started on 127.0.0.1:$TPORT"; }
    pgrep -f "dashboard/server.py" >/dev/null || { cd $DIR && nohup $PY $DIR/server.py $PORT > /tmp/so101_dash.log 2>&1 & echo "server started (pid $!), log /tmp/so101_dash.log"; sleep 1.5; }
    [ "$1" = "open" ] || { pkill -f "user-data-dir=$UDD" 2>/dev/null; read X Y W H <<< "$(geom)"; echo "monitor 2: ${W}x${H} at $X,$Y"
      open -na "Google Chrome" --args --user-data-dir=$UDD --no-first-run --remote-debugging-port=9333 "--remote-allow-origins=*" --kiosk --window-position=$X,$Y --window-size=$W,$H --app=http://127.0.0.1:$PORT; }
    echo "http://127.0.0.1:$PORT";;
esac
