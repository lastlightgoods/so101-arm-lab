"""SO-101 lab dashboard server.  python server.py [port]   ->  http://127.0.0.1:8765
Polls the Jetson over ssh (tegrastats stream + a slow status probe), the Mac via psutil, reads the
local datasets / training logs / checkpoints, and serves the gallery frames.  No cameras are opened here.
Read-only calibration endpoints: /api/calib, /api/locate?u=&v=&h_mm=&cam=top|front, /api/ik?x_in=&y_in=&h_in=&roll= (docs/camera-calibration.md).
Mat scene: /api/scene, /api/scene.png, POST /api/scene/background (docs/mat-scene.md); the overhead calibration heals itself when the camera moves,
and both camera poses (camera.pose, camera.front.pose) are re-solved from the markers with each lens's fixed field of view so the 3D stage follows the cameras."""
import glob, json, os, re, shlex, subprocess, threading, time, http.server, socketserver, sys
import psutil
import queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling modules (a no-op under `python server.py`; the tests import this file by path)
import telemetry_store   # servo health history + flags (docs/telemetry.md); /api/tele serves it
SAY_Q = queue.Queue(); _last_said = {"text": "", "t": 0.0}
def say(text):
    """Queue a spoken line; see say_loop. Never speaks in parallel."""
    SAY_Q.put((time.time(), str(text)))
def say_loop():
    while True:
        t, text = SAY_Q.get()
        if not SAY_Q.empty() and time.time() - t > 2.5: continue            # superseded while waiting
        if text == _last_said["text"] and time.time() - _last_said["t"] < 3: continue   # same line twice in a row
        _last_said.update(text=text, t=time.time())
        try: subprocess.run(say_argv(text), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        except Exception: pass


SAY_DEVICE = os.environ.get("SO101_SAY_DEVICE", "MacBook Pro Speakers")   # why: TC 2026-11-11 "put all voice from this project through the mac speakers" (the default output is often a Bluetooth speaker); "" = the system default


def say_argv(text):
    """macOS `say` on SAY_DEVICE only (no fallback to the default output: a missing device stays silent)."""
    return ["say"] + (["-a", SAY_DEVICE] if SAY_DEVICE else []) + ["-r", "190", str(text)]
HERE = os.path.dirname(os.path.abspath(__file__)); HOME = os.path.expanduser("~")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
JETSON = os.environ.get("SO101_JETSON", "jetson")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", JETSON]
state = {"jetson": {"online": False, "last": 0}, "mac": {}, "training": {}, "datasets": {}, "policies": {}, "procs": [], "events": [], "arms": {}, "spotify": {"running": False}, "policy": {}}   # policy: the Jetson's DEPLOYED.json (so101-policy deploy), served at /api/policy
lock = threading.RLock()
TELE = telemetry_store.TeleStore()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def event(msg):
    with lock:
        state["events"].insert(0, {"t": time.time(), "msg": msg}); del state["events"][40:]
    try:
        with open(os.path.join(HERE, "events.log"), "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError: pass
# ---------- tegrastats stream ----------
TS = re.compile(r"RAM (\d+)/(\d+)MB.*?SWAP (\d+)/(\d+)MB.*?CPU \[(.*?)\].*?GR3D_FREQ (\d+)%(?:@\[?(\d+))?")
EMC = re.compile(r"EMC_FREQ (\d+)%(?:@(\d+))?")  # memory-controller bus load; tegrastats only prints it when run as root
# root tegrastats adds EMC_FREQ (bus) + engine lines; fall back to plain tegrastats when passwordless sudo is not set up
TEGRA_CMD = "sudo -n tegrastats --interval 2000 2>/dev/null || tegrastats --interval 2000"
HIST = []  # rolling activity: [t, gpu%, emc% or None, cpu%, ram%, temp C, watts] per tegrastats line (2 s), last 5 min; served at /api/hist
def parse_tegra(line):
    m = TS.search(line)
    if not m: return None
    cores = [int(c.split("%")[0]) for c in m.group(5).split(",") if "%" in c]
    freqs = [int(c.split("@")[1]) for c in m.group(5).split(",") if "@" in c]
    d = {"ram_used": int(m.group(1)), "ram_total": int(m.group(2)), "swap_used": int(m.group(3)), "swap_total": int(m.group(4)),
         "cpu_cores": cores, "cpu": sum(cores) / max(1, len(cores)), "cpu_mhz": max(freqs) if freqs else 0, "gpu": int(m.group(6)),
         "gpu_mhz": int(m.group(7)) if m.group(7) else 0, "temps": {k: float(v) for k, v in re.findall(r"(\w+)@([\d.]+)C", line)},
         "power_mw": {k: int(v) for k, v in re.findall(r"(VDD_\w+) (\d+)mW", line)}}
    d["watts"] = d["power_mw"].get("VDD_IN", 0) / 1000
    e = EMC.search(line); d["emc"] = int(e.group(1)) if e else None; d["emc_mhz"] = int(e.group(2)) if e and e.group(2) else 0
    d["temp"] = max([d["temps"].get("tj", 0), d["temps"].get("gpu", 0), d["temps"].get("cpu", 0)])
    return d
def tegra_loop():
    while True:
        try:
            p = subprocess.Popen(SSH + [TEGRA_CMD], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                d = parse_tegra(line)
                if d:
                    with lock:
                        if not state["jetson"]["online"]: event("Jetson online")
                        state["jetson"].update(d); state["jetson"]["online"] = True; state["jetson"]["last"] = time.time()
                        HIST.append([round(time.time()), d["gpu"], d["emc"], round(d["cpu"]), round(100 * d["ram_used"] / max(1, d["ram_total"])), round(d["temp"], 1), round(d["watts"], 2)]); del HIST[:-150]
            p.wait()
        except Exception as e: log("tegra", e)
        with lock:
            if state["jetson"]["online"]: event("Jetson offline")
            state["jetson"]["online"] = False
        time.sleep(5)
# ---------- slow probes (Jetson + Mac): power mode, uptime, disk, lerobot processes, training logs, datasets, policies ----------
PROBE = r'''echo "###mode"; /usr/sbin/nvpmodel -q 2>&1 | head -1; echo "###uptime"; uptime -p; echo "###disk"; df -h / | tail -1
echo "###procs"; ps -eo pid,etime,args | grep -E "lerobot-(record|train|teleoperate|rollout|calibrate|replay)|so101/(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper)\\.py" | grep -v grep
for f in ~/so101/outputs/train/*/train.log ~/so101/outputs/train/*.log; do [ -f "$f" ] && echo "###trainlog $f" && tail -n 300 "$f"; done
for d in ~/so101/outputs/train/*/checkpoints; do [ -d "$d" ] && echo "###ckpts $d" && ls "$d"; done
for f in ~/.cache/huggingface/lerobot/lastlight/*/meta/info.json; do [ -f "$f" ] && echo "###dataset $f" && stat -c %Y "$f" 2>/dev/null && cat "$f"; done
for f in ~/so101/policies/*/train_config.json ~/so101/policies/*/current/train_config.json ~/so101/outputs/train/*/checkpoints/last/pretrained_model/train_config.json; do [ -f "$f" ] && echo "###policy $f" && stat -c %Y "$f" 2>/dev/null && cat "$f"; done
f=~/so101/policies/DEPLOYED.json; [ -f "$f" ] && echo "###deployed $f" && cat "$f"
echo "###end"'''
STEP = re.compile(r"step:\s*([\d.]+)([KM]?)"); LOSS = re.compile(r"loss:\s*([\d.]+)"); LR = re.compile(r"lr:\s*([\d.e+-]+)")
def num(s, suf): return float(s) * {"K": 1e3, "M": 1e6}.get(suf, 1)
def parse_probe(text, host):
    out = {"training": {}, "datasets": {}, "policies": {}, "procs": [], "mode": "", "uptime": "", "disk": ""}
    for block in text.split("###")[1:]:
        head, _, body = block.partition("\n"); tag, _, arg = head.partition(" ")
        if tag == "mode": out["mode"] = body.strip().replace("NV Power Mode: ", "")
        elif tag == "uptime": out["uptime"] = body.strip().replace("up ", "")
        elif tag == "disk":
            parts = body.split(); out["disk"] = {"used": parts[2], "total": parts[1], "pct": int(parts[4].rstrip("%"))} if len(parts) > 4 else ""
        elif tag == "procs":
            for l in body.strip().splitlines():
                p = l.split(None, 2)
                if len(p) == 3:
                    kind = re.search(r"lerobot-(\w+)", p[2]) or re.search(r"so101/(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper)\.py", p[2]); d, _, hms = p[1].rpartition("-"); parts = [int(x) for x in hms.split(":")]
                    secs = int(d or 0) * 86400 + sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
                    out["procs"].append({"host": host, "kind": kind.group(1) if kind else "?", "secs": secs, "args": p[2][:300]})
        elif tag == "trainlog":
            job = os.path.basename(arg)[:-4] if arg.endswith(".log") and os.path.basename(arg) != "train.log" else os.path.basename(os.path.dirname(arg)); pts = []
            for l in body.splitlines():
                s, lo = STEP.search(l), LOSS.search(l)
                if s and lo: pts.append([num(s.group(1), s.group(2)), float(lo.group(1))])
            lr = LR.findall(body)
            out["training"][job] = {"host": host, "points": pts, "lr": lr[-1] if lr else "", "updated": time.time()}
        elif tag == "ckpts":
            job = os.path.basename(os.path.dirname(arg)); steps = [int(x) for x in body.split() if x.isdigit()]
            t = out["training"].setdefault(job, {"host": host, "points": []}); t["ckpts"] = steps; t["last_ckpt"] = max(steps) if steps else 0
        elif tag == "dataset":
            try:
                mtime, _, js = body.partition("\n"); i = json.loads(js); name = os.path.basename(os.path.dirname(os.path.dirname(arg)))
                if "debug" in name: continue
                out["datasets"][name] = {"host": host, "mtime": int(mtime) if mtime.strip().isdigit() else 0, "episodes": i.get("total_episodes", 0), "frames": i.get("total_frames", 0), "fps": i.get("fps", 30),
                                         "hours": i.get("total_frames", 0) / max(1, i.get("fps", 30)) / 3600, "cams": [k.split(".")[-1] for k, v in i["features"].items() if v["dtype"] == "video"]}
            except Exception as e: log("dataset parse", arg, e)
        elif tag == "policy":
            try:
                mtime, _, js = body.partition("\n"); c = json.loads(js)
                name = arg.split("/policies/")[1].split("/")[0] if "/policies/" in arg else arg.split("/outputs/train/")[1].split("/")[0]
                out["policies"][name] = {"host": host, "type": c.get("policy", {}).get("type", "?"), "steps": c.get("steps", 0), "dataset": (c.get("dataset", {}) or {}).get("repo_id", "?"),
                                         "device": c.get("policy", {}).get("device", "?"), "mtime": int(float(mtime)) if mtime.strip().isdigit() else 0}
            except Exception as e: log("policy parse", arg, e)
        elif tag == "deployed":   # why: only present when the file exists, so a probe without it parses to the same dict as before
            try: d = json.loads(body); out["deployed"] = {k: d.get(k) for k in ("name", "version", "run_id", "sha256", "when", "by", "action", "path")} | {"host": host, "history": len(d.get("history") or [])}
            except Exception as e: log("deployed parse", arg, e)
    return out
def probe_loop():
    seen = {}
    while True:
        results = {}
        try: results["mac"] = parse_probe(subprocess.run(["bash", "-s"], input=PROBE.replace("/usr/sbin/nvpmodel -q 2>&1 | head -1", "echo mac").replace("uptime -p", "uptime").replace("stat -c %Y", "stat -f %m"), capture_output=True, text=True, timeout=30).stdout, "mac")
        except Exception as e: log("mac probe", e)
        try:
            r = subprocess.run(SSH + ["bash", "-s"], input=PROBE, capture_output=True, text=True, timeout=40)  # script via stdin: ssh would re-split a -c argument
            if r.returncode == 0: results["jetson"] = parse_probe(r.stdout, "jetson")
        except Exception as e: log("jetson probe", e)
        with lock:
            for k in ("training", "datasets", "policies"):
                merged = {}
                for host in ("mac", "jetson"):
                    for name, v in results.get(host, {}).get(k, {}).items(): merged[f"{name}@{host}" if name in merged else name] = v
                state[k] = merged
            state["procs"] = sum((results.get(h, {}).get("procs", []) for h in ("mac", "jetson")), [])
            if "jetson" in results: state["jetson"].update({"mode": results["jetson"]["mode"], "uptime": results["jetson"]["uptime"], "disk": results["jetson"]["disk"]}); state["policy"] = {**results["jetson"].get("deployed", {}), "probed": time.time()}   # why: cached here; /api/policy never opens its own ssh
            vm = psutil.virtual_memory(); du = psutil.disk_usage(HOME)
            state["mac"] = {"cpu": psutil.cpu_percent(interval=None), "ram_used": vm.used // 2**20, "ram_total": vm.total // 2**20, "disk_pct": du.percent, "disk_free_gb": du.free / 2**30}
            for name, d in state["datasets"].items():
                key = ("ds", name)
                if key in seen and seen[key] != d["episodes"]: event(f"{name}: {d['episodes']} episodes ({d['episodes'] - seen[key]:+d})")
                seen[key] = d["episodes"]
            for job, t in state["training"].items():
                key = ("ck", job); lc = t.get("last_ckpt", 0)
                if key in seen and seen[key] != lc and lc: event(f"{job}: checkpoint {lc:,} saved on {t['host']}")
                seen[key] = lc
            kinds = sorted({p["kind"] for p in state["procs"]})
            if seen.get("procs") != kinds:
                if kinds: event("running: " + ", ".join(f"lerobot-{k}" for k in kinds))
                elif "procs" in seen: event("all lerobot processes finished")
                seen["procs"] = kinds
        time.sleep(15)

# ---------- live cameras (Mac-side, icspring USB cams only, selected BY NAME via the avf plugin; FaceTime/iPhone are refused there) ----------
CAMS = {"top": {"native": "1920x1080", "width": 640, "height": 360}, "wrist": {"native": "640x480", "width": 640, "height": 480}}
cam_frames = {}   # name -> (jpeg bytes, time)
def cam_loop(name, spec):
    try:
        import cv2
        from lerobot_camera_avf import AVFCameraConfig, AVFCamera
    except Exception as e: log("cam", name, "plugin unavailable:", e); return
    while True:
        cam = None
        try:
            cam = AVFCamera(AVFCameraConfig(match="icspring", native=spec["native"], width=spec["width"], height=spec["height"], fps=30)); cam.connect()
            event(f"live feed: {name} camera"); fail = 0
            while True:
                try:
                    frame = cam.read_latest(max_age_ms=1500); fail = 0
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok: cam_frames[name] = (jpg.tobytes(), time.time())
                except Exception:
                    fail += 1
                    if fail > 50: raise
                time.sleep(1 / 15)
        except Exception as e:
            if not getattr(cam_loop, "warned", False): log("mac cam", name, "not available here:", str(e)[:90]); cam_loop.warned = True
        finally:
            try: cam and cam.disconnect()
            except Exception: pass
        cam_frames.pop(name, None); time.sleep(60)

# ---------- live cameras from the Jetson (ssh + ffmpeg -> MJPEG). Remote side yields while ~/so101/.cams_busy exists (set by so101-record/run). ----------
JETSON_CAMS = {"top": ("usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0", "mjpeg", "640x360"),      # Logitech C920 overhead (28 Oct)
               "front": ("usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0", "mjpeg", "640x360"),    # Logitech C920 facing the arm
               "wrist": ("usb-icSpring_icspring_camera-video-index0", "yuyv422", "640x480")}
def jetson_cam_loop(name, dev, fmt, size):
    cmd = (f"while [ -f ~/so101/.cams_busy ]; do sleep 1; done; "
           f"[ -f ~/so101/camd.py ] && {{ python3 ~/so101/camd.py tap {name}; rc=$?; [ $rc -eq 3 ] || exit $rc; }}; "   # why: camd (jetson/camd.py) owns the cameras when it runs and serves this same MJPEG; exit 3 = no camd, so the ffmpeg below runs exactly as before (11 Nov 2026, docs/scripted-demos.md)
           f"[ -e /dev/v4l/by-id/{dev} ] || exit 3; "
           f"ffmpeg -loglevel error -f v4l2 -input_format {fmt} -video_size {size} -framerate 15 -i /dev/v4l/by-id/{dev} -f mjpeg -q:v 6 - & p=$!; "
           "while kill -0 $p 2>/dev/null; do [ -f ~/so101/.cams_busy ] && kill $p; sleep 0.5; done")
    announced = False
    while True:
        try:
            p = subprocess.Popen(SSH + [cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            buf = b""
            while True:
                chunk = p.stdout.read(65536)
                if not chunk: break
                buf += chunk
                while True:
                    i = buf.find(b"\xff\xd8"); j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
                    if i < 0 or j < 0: buf = buf[-2:] if i < 0 else buf[i:]; break
                    cam_frames[name] = (buf[i:j + 2], time.time()); buf = buf[j + 2:]
                    if not announced: event(f"live feed: {name} camera (jetson)"); announced = True
            p.wait()
        except Exception as e: log("jetson cam", name, "error:", str(e)[:120])
        if cam_frames.pop(name, None) and announced: announced = False
        time.sleep(3)

# ---------- 🦖 handoff roar: gripper opens while the arm is extended (not at rest) -> play roar.wav on the Mac ----------
_roar = {"open": None, "last": 0}
def roar_check(pos):
    g = pos.get("gripper"); lift = pos.get("shoulder_lift")
    if g is None or lift is None: return
    was_open = _roar["open"]; now_open = g > 55 if was_open is not True else g > 35
    _roar["open"] = now_open
    if now_open and was_open is False and lift > -55 and time.time() - _roar["last"] > 8:
        _roar["last"] = time.time(); event("🦖 RAWR — handoff!")
        subprocess.Popen(["afplay", os.path.join(HERE, "roar.wav")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# ---------- joint states from the Jetson (arm_state.py streams JSON lines over ssh) ----------
def arm_loop():
    while True:
        try:
            p = subprocess.Popen(SSH + ["source ~/so101/.venv/bin/activate && exec python ~/so101/arm_state.py"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                try: d = json.loads(line)
                except ValueError: continue
                if d.get("role") == "say":
                    if time.time() - d.get("t", 0) < 30:
                        event("🔊 " + d["say"]); say(d["say"])
                    continue
                tele = d.pop("tele", None)   # servo health rides on the follower's line; it goes to its own store so /api/state stays small
                if tele and d.get("role") == "follower":
                    try:
                        for ev in TELE.add(tele, d.get("src")):
                            if ev["level"] >= 2: event(f"servo {ev['joint']}: {ev['text']}")   # only the red ones go to the log panel; the telemetry panel lists them all
                    except Exception as e: log("telemetry", repr(e))
                with lock: state.setdefault("arms", {})[d.get("role", "arm")] = {**d, "rx": time.time()}
                if d.get("role") == "follower": roar_check(d.get("pos") or {})
            p.wait()
        except Exception as e: log("arm stream", e)
        with lock: state["arms"] = {}
        time.sleep(5)

# ---------- Claude's shell: tail this Claude Code session's transcript for Bash tool calls + results ----------
def newest_transcript():
    """The most recently written Claude Code session transcript (any project); re-checked so a new session takes over."""
    if os.environ.get("SO101_TRANSCRIPT"): return os.environ["SO101_TRANSCRIPT"]
    fs = [f for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")) if "/memory/" not in f]
    for f in sorted(fs, key=os.path.getmtime, reverse=True)[:8]:   # newest session that is actually running shell commands
        try:
            with open(f, "rb") as fh: fh.seek(max(0, os.path.getsize(f) - 300_000)); tail = fh.read()
            if b'"name": "Bash"' in tail or b'"name":"Bash"' in tail: return f
        except Exception: pass
    return max(fs, key=os.path.getmtime) if fs else ""
TRANSCRIPT = newest_transcript()
shell_calls = []   # newest last: {id, t, desc, cmd, out, done}
def _ingest(line):
    try: d = json.loads(line)
    except ValueError: return
    c = (d.get("message") or {}).get("content")
    if not isinstance(c, list): return
    for it in c:
        if it.get("type") == "tool_use" and it.get("name") == "Bash":
            shell_calls.append({"id": it.get("id"), "t": d.get("timestamp", ""), "desc": it["input"].get("description", ""), "cmd": it["input"].get("command", "")[:600], "out": "", "done": False}); del shell_calls[:-14]
        elif it.get("type") == "tool_result":
            r = it.get("content"); txt = r if isinstance(r, str) else " ".join(x.get("text", "") for x in r if isinstance(x, dict)) if isinstance(r, list) else ""
            for call in shell_calls:
                if call["id"] == it.get("tool_use_id"): call["out"] = txt[-1500:]; call["done"] = True
def shell_loop():
    global TRANSCRIPT
    while True:
        TRANSCRIPT = newest_transcript()
        if not TRANSCRIPT or not os.path.exists(TRANSCRIPT): log("shell tail: no transcript found"); time.sleep(30); continue
        log("shell tail:", TRANSCRIPT.split("/")[-1][:12]); shell_calls.clear(); last_check = time.time()
        with open(TRANSCRIPT, "rb") as f:
            f.seek(max(0, os.path.getsize(TRANSCRIPT) - 400_000)); f.readline()
            for line in f: _ingest(line.decode("utf-8", "replace"))
            while True:
                line = f.readline()
                if line: _ingest(line.decode("utf-8", "replace"))
                else:
                    time.sleep(1.5)
                    if time.time() - last_check > 30:
                        last_check = time.time()
                        if newest_transcript() != TRANSCRIPT: break   # a newer session started: follow it

# ---------- spotify (Mac desktop app via AppleScript; never launches it) ----------
SPOT_AS = "\n".join([
    'if application "Spotify" is running then',
    ' tell application "Spotify"',
    '  set t to current track',
    '  return (player state as text) & "\\n" & (name of t) & "\\n" & (artist of t) & "\\n" & (album of t) & "\\n" & (artwork url of t) & "\\n" & (player position as text) & "\\n" & ((duration of t) / 1000 as text) & "\\n" & (sound volume as text)',
    ' end tell',
    'else',
    ' return "off"',
    'end if'])
def spotify_loop():
    while True:
        try:
            out = subprocess.run(["osascript", "-e", SPOT_AS], capture_output=True, text=True, timeout=4).stdout.strip()
            if out == "off" or not out: sp = {"running": False}
            else:
                st, track, artist, album, art, pos, dur, vol = (out.split("\n") + [""] * 8)[:8]
                sp = {"running": True, "state": st, "track": track, "artist": artist, "album": album, "art": art, "pos": float(pos.replace(",", ".") or 0), "dur": float(dur.replace(",", ".") or 0), "vol": int(float(vol or 0)), "t": time.time()}
            with lock: state["spotify"] = sp
        except Exception as e:
            with lock: state["spotify"] = {"running": False, "err": repr(e)[:80]}
        time.sleep(2.5)
def spotify_cmd(cmd):
    cmds = {"playpause": "playpause", "next": "next track", "previous": "previous track"}
    if cmd not in cmds: return {"ok": False, "err": "unknown"}
    r = subprocess.run(["osascript", "-e", 'if application "Spotify" is running then tell application "Spotify" to ' + cmds[cmd]], capture_output=True, text=True, timeout=4)
    return {"ok": r.returncode == 0, "err": r.stderr.strip()[:120]}
# ---------- arm modes (buttons on the ARM panel; each runs a so101-* command on the Jetson, detached) ----------
ARM_MODES = {
    "teleop": "so101-teleop",
    "free":   "so101-free",
    "hold":   "so101-hold",
    "policy": "so101-nudge; so101-run-async 90 $HOME/so101/policies/ft_lighter 90",
    "vla":    "so101-nudge; so101-run-vla 90 $HOME/so101/policies/smolvla_lighter_2500",
    "dance":  "so101-dance 100 45 1.0",
    "wave":   "so101-demo 1.0",
    "replay": "so101-replay-hub lerobot/svla_so101_pickplace $((RANDOM % 50))",
}
def arm_mode(mode):
    if mode == "stop":
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, "bash -lc 'source ~/so101/env.sh; so101-stop' 2>&1 | tail -2"], capture_output=True, text=True, timeout=20)
        subprocess.run(["pkill", "-f", "afplay .*beat_"], capture_output=True)
        with lock: state["events"].insert(0, {"t": time.time(), "msg": "ARM STOP (dashboard)"})
        return {"ok": True, "out": r.stdout.strip()[-200:]}
    if mode not in ARM_MODES: return {"ok": False, "err": "unknown mode"}
    cmd = ARM_MODES[mode]
    if mode in ("free", "hold") and REC["running"]:
        say("Stop the recording first."); return {"ok": False, "err": "recording holds the arm; press Q to stop it first"}
    if mode == "free": say("Arm released. Support it.")
    if mode == "hold": say("Holding.")
    remote = "mkdir -p ~/so101/outputs/run; nohup bash -lc 'source ~/so101/env.sh; " + cmd + "' > ~/so101/outputs/run/ui_" + mode + ".log 2>&1 &"
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, remote], capture_output=True, text=True, timeout=20)
    if mode == "dance":
        beat = os.path.join(HERE, "beat_100.wav")
        if os.path.exists(beat): threading.Timer(2.0, lambda: subprocess.Popen(["afplay", beat])).start()
    with lock: state["events"].insert(0, {"t": time.time(), "msg": "ARM MODE " + mode.upper() + " (dashboard)"})
    return {"ok": r.returncode == 0, "err": r.stderr.strip()[-200:]}

# ---------- retinal scan lock (theatrical: any eye in the WRIST camera unlocks; dashboard is localhost-only) ----------
LOCK = {"locked": os.environ.get("SO101_LOCK", "0") == "1", "progress": 0.0, "status": "NO SIGNAL", "eye": None, "w": 640, "h": 480, "t": 0}
_eye_cascade = None
def _scanpose(rest):
    try: subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, "bash -lc 'source ~/so101/env.sh; so101-scanpose %s' > ~/so101/outputs/run/ui_scanpose.log 2>&1" % ("rest" if rest else "")], timeout=40)
    except Exception as e: log("scanpose:", repr(e))
def set_lock(locked, why=""):
    with lock:
        LOCK.update(locked=locked, progress=0.0, status="NO SIGNAL" if locked else "UNLOCKED", eye=None)
        state["events"].insert(0, {"t": time.time(), "msg": ("DASHBOARD LOCKED" if locked else "IDENTITY CONFIRMED · UNLOCKED") + (" · " + why if why else "")})
    threading.Thread(target=_scanpose, args=(not locked,), daemon=True).start()
def _find_iris(gray):
    """Close-range iris finder: a dark disc (pupil+iris) ringed by something brighter (sclera/skin). Returns (x,y,r) or None."""
    h, w = gray.shape; g = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w / 2, param1=110, param2=28, minRadius=int(w * 0.05), maxRadius=int(w * 0.22))
    if circles is None: return None
    best = None
    for x, y, r in circles[0][:6]:
        x, y, r = int(x), int(y), int(r)
        if r < 6 or x - 2 * r < 0 or y - 2 * r < 0 or x + 2 * r >= w or y + 2 * r >= h: continue
        m = np.zeros_like(g, np.uint8); cv2.circle(m, (x, y), int(r * .55), 255, -1); inner = cv2.mean(g, m)[0]
        m[:] = 0; cv2.circle(m, (x, y), int(r * 1.9), 255, int(r * .9)); outer = cv2.mean(g, m)[0]
        score = outer - inner
        if score > 28 and (best is None or score > best[3]): best = (x, y, r, score)
    return best
def eye_loop():
    global cv2, np
    try: import cv2, numpy as np
    except Exception as e: log("eye: opencv unavailable", repr(e)); return
    if LOCK["locked"]: threading.Thread(target=_scanpose, args=(False,), daemon=True).start()
    seen = 0.0; last_t = 0; hold = 0
    while True:
        if not LOCK["locked"]: time.sleep(0.3); continue
        f = cam_frames.get("wrist")
        if not f or time.time() - f[1] > 3:
            with lock: LOCK.update(status="NO SIGNAL", eye=None, progress=0.0)
            time.sleep(0.3); continue
        if f[1] == last_t: time.sleep(0.03); continue
        last_t = f[1]
        try:
            img = cv2.imdecode(np.frombuffer(f[0], np.uint8), cv2.IMREAD_GRAYSCALE); h, w = img.shape[:2]
            hit = _find_iris(img)
        except Exception as e:
            log("eye detect:", repr(e)); time.sleep(0.5); continue
        now = time.time()
        if hit:
            x, y, r, score = hit; big = r >= w * 0.08
            seen = min(1.0, seen + (0.12 if big else 0.04)); hold = now
            with lock: LOCK.update(eye=[int(x - r), int(y - r), int(2 * r), int(2 * r)], w=w, h=h, status="SCANNING" if big else "MOVE CLOSER", progress=round(seen, 3), t=now)
            if seen >= 1.0:
                with lock: LOCK.update(status="CONFIRMED", progress=1.0)
                say("Identity confirmed. Welcome back.")
                time.sleep(1.8); set_lock(False, "retinal scan"); seen = 0.0
        else:
            seen = max(0.0, seen - 0.04) if now - hold < 0.8 else 0.0
            with lock: LOCK.update(eye=None, w=w, h=h, status="LOOK INTO THE WRIST CAMERA", progress=round(seen, 3), t=now)
        time.sleep(0.05)

# ---------- simulation status (datasets, eval results, running sim processes, rollout GIFs) ----------
SIM_DIR = os.path.expanduser("~/so101/sim"); _sim_cache = {"t": 0, "v": {}}
def sim_status():
    if time.time() - _sim_cache["t"] < 5: return _sim_cache["v"]
    out = {"datasets": [], "evals": [], "procs": [], "gifs": []}
    try:
        for info in sorted(glob.glob(os.path.join(SIM_DIR, "data", "*", "meta", "info.json"))):
            try: d = json.load(open(info)); out["datasets"].append({"name": info.split("/")[-3], "episodes": d.get("total_episodes"), "frames": d.get("total_frames")})
            except Exception: pass
        for lg in sorted(glob.glob(os.path.join(SIM_DIR, "eval_*.log")) + glob.glob(os.path.join(SIM_DIR, "analysis*.log")), key=os.path.getmtime):
            cur = None
            for line in open(lg, errors="replace"):
                m = re.search(r"=== (?:v2 )?checkpoint (\d+)|checkpoints/(\d+)/pretrained_model", line)
                if m: cur = (m.group(1) or m.group(2))
                m = re.search(r"SUCCESS (\d+)/(\d+) = (\d+)%", line)
                if m:
                    name = os.path.basename(lg).replace(".log", ""); ck = cur or (lambda mm: str(int(mm.group(1)) * 1000).zfill(6) if mm else None)(re.search(r"_(\d+)k\b", name))
                    entry = {"log": name, "ckpt": ck, "ok": int(m.group(1)), "n": int(m.group(2)), "pct": int(m.group(3))}
                    if not any(e["ckpt"] == ck and e["ok"] == entry["ok"] and e["n"] == entry["n"] for e in out["evals"]): out["evals"].append(entry)
        ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            if re.search(r"collect_sim\.py|eval_sim\.py|so101_sim", line) and "grep" not in line and "server.py" not in line:
                kind = "collect" if "collect_sim" in line else "eval" if "eval_sim" in line else "train" if "lerobot-train" in line else "sim"
                m = re.search(r"--episodes (\d+)", line); out["procs"].append({"kind": kind, "etime": line.split()[1], "episodes": int(m.group(1)) if m else None, "ckpt": (re.search(r"checkpoints/(\d+)", line) or [None, None])[1]})
        gifs = sorted(glob.glob(os.path.join(SIM_DIR, "rollouts", "**", "*.gif"), recursive=True), key=os.path.getmtime)[-12:]
        out["gifs"] = [{"url": "/simroll/" + os.path.relpath(g, os.path.join(SIM_DIR, "rollouts")), "ok": "_ok" in g, "run": g.split("/")[-2], "age": int(time.time() - os.path.getmtime(g))} for g in reversed(gifs)]
        out["active"] = bool(out["procs"]) or (out["gifs"] and out["gifs"][0]["age"] < 3 * 3600)
    except Exception as e: out["err"] = repr(e)[:120]
    _sim_cache.update(t=time.time(), v=out); return out

# ---------- recording from the dashboard (form -> lerobot-record on the edge host, progress parsed from its log) ----------
REC = {"running": False, "name": "", "task": "", "episodes": 0, "episode": 0, "phase": "", "started": 0, "ep_started": 0, "episode_s": 25, "reset_s": 10, "done": False, "err": "", "log": "", "done_total": 0, "episodes_target": 0, "session_episode": 0, "paused_at": 0}
def record_start(p):
    name = re.sub(r"[^A-Za-z0-9_\-]", "", str(p.get("name", "")))[:60]; task = str(p.get("task", "")).replace('"', "'").strip()[:200]
    try: n = max(1, min(200, int(p.get("episodes", 10)))); ep_s = max(5, min(120, int(p.get("episode_s", 25)))); rs = max(3, min(60, int(p.get("reset_s", 10)))); resume = bool(p.get("resume", True))
    except (TypeError, ValueError): return {"ok": False, "err": "episodes, episode_s and reset_s must be whole numbers"}   # why: an empty form field raised inside the request handler and the browser saw a dropped connection, not an error
    if not name or not task: return {"ok": False, "err": "name and task are required"}
    if REC["running"]: return {"ok": False, "err": "already recording"}
    if resume:   # append only works on a dataset that exists locally on the edge host; otherwise start it fresh
        chk = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, f"test -f ~/.cache/huggingface/lerobot/lastlight/{name}/meta/info.json && echo yes || echo no"], capture_output=True, text=True, timeout=15)
        if chk.stdout.strip() != "yes":
            resume = False
            with lock: state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {name}: no existing dataset, creating a new one"})
    fresh = not REC.get("_chain")
    with lock:
        REC.update(running=True, name=name, task=task, episodes=n, episode=0, phase="starting", started=time.time(), ep_started=0, episode_s=ep_s, reset_s=rs, done=False, err="", log="", _cal_noted=False, paused_at=0)
        if fresh: REC.update(done_total=0, episodes_target=n, chain_base=None, saved_total=0)
        REC["_chain"] = False
    fn = f'so101-record-more {name} {n} "{task}"' if resume else f'so101-record {name} "{task}" {n}'
    inner = f"source ~/so101/env.sh; export SO101_EP_S={ep_s} SO101_RESET_S={rs}; {fn}"
    remote = ("mkdir -p ~/so101/outputs/run; : > ~/so101/outputs/run/ui_record.log; rm -f /tmp/so101_record_ctl; "
              "nohup python3 ~/so101/record_ctl.py ~/so101/outputs/run/ui_record.log /tmp/so101_record_ctl -- bash -lc " + shlex.quote(inner) + " > /dev/null 2>&1 &")
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, remote], capture_output=True, text=True, timeout=20)
    with lock: state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {name} · {n} episodes · {ep_s}s (dashboard)"})
    if r.returncode != 0:
        with lock: REC.update(running=False, err=r.stderr.strip()[-200:])
        return {"ok": False, "err": r.stderr.strip()[-200:]}
    return {"ok": True}
def record_ctl(cmd):
    key = {"next": "n", "rerecord": "r", "stop": "q", "pause": "p"}.get(cmd)
    if not key: return {"ok": False, "err": "unknown command"}
    if not REC["running"]: return {"ok": False, "err": "not recording"}
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, f"printf {key} > /tmp/so101_record_ctl"], capture_output=True, text=True, timeout=15)
    with lock: state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {cmd.upper()} (dashboard)"})
    if r.returncode == 0 and cmd != "pause": say({"next": "Next.", "rerecord": "Redo. Reset first, then it records again.", "stop": "Stopping after this episode."}[cmd])   # pause/resume is spoken by the plugin
    return {"ok": r.returncode == 0, "err": r.stderr.strip()[-120:]}
def record_apply(p):
    """Change settings mid-session: stop cleanly at the episode boundary, then resume into the same dataset with the new values."""
    if not REC["running"]: return {"ok": False, "err": "not recording"}
    done_before = REC["done_total"] + max(0, REC["episode"] - 1)
    target = int(p.get("episodes", REC["episodes_target"]) or REC["episodes_target"]); remaining = max(1, target - done_before)
    name = REC["name"]
    record_ctl("stop")
    t0 = time.time()
    while REC["running"] and time.time() - t0 < 120: time.sleep(1)
    if REC["running"]: return {"ok": False, "err": "recorder did not stop"}
    newp = {"name": name, "task": p.get("task") or REC["task"], "episodes": remaining, "episode_s": p.get("episode_s", REC["episode_s"]), "reset_s": p.get("reset_s", REC["reset_s"]), "resume": True}
    with lock: REC["_chain"] = True
    res = record_start(newp)
    with lock: REC["done_total"] = done_before; REC["episodes_target"] = target
    return res
def record_reattach():
    """Server restarted while a recording runs on the edge host: pick it up from the lerobot-record command line."""
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, "pgrep -af 'bin/lerobot-record' | grep -v pgrep | head -1"], capture_output=True, text=True, timeout=15)
        cmd = r.stdout.strip()
        if not cmd: return False
        def arg(k, default=""):
            m = re.search(rf"--{re.escape(k)}=(\"[^\"]*\"|'[^']*'|\S+)", cmd); return m.group(1).strip("\"'") if m else default
        name = arg("dataset.repo_id").split("/")[-1]
        if not name: return False
        n = int(arg("dataset.num_episodes", "10") or 10); ep_s = int(float(arg("dataset.episode_time_s", "25") or 25)); rs = int(float(arg("dataset.reset_time_s", "10") or 10))
        with lock:
            REC.update(running=True, name=name, task=arg("dataset.single_task", REC.get("task") or ""), episodes=n, episode=0, phase="recording", started=time.time(), ep_started=0,
                       episode_s=ep_s, reset_s=rs, done=False, err="", log="", episodes_target=n, chain_base=None, saved_total=0, done_total=0)
        log(f"record: reattached to running session {name} ({n} eps, {ep_s}s/{rs}s)"); return True
    except Exception as e:
        log("record: reattach failed", repr(e)); return False
def record_loop():
    record_reattach()
    while True:
        if not REC["running"]: time.sleep(1); continue
        try:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", JETSON, f"tail -c 6000 ~/so101/outputs/run/ui_record.log 2>/dev/null; pgrep -f 'bin/lerobot-record' >/dev/null && echo __ALIVE__; python3 -c \"import json;print('__EPS__',json.load(open('/home/tc/.cache/huggingface/lerobot/lastlight/{REC['name']}/meta/info.json'))['total_episodes'])\" 2>/dev/null"], capture_output=True, text=True, errors="replace", timeout=15)   # why: tail -c can start mid-character (━ bars); a strict decode failed every poll, so a finished log was never noticed
            txt = r.stdout; alive = "__ALIVE__" in txt; txt = txt.replace("__ALIVE__", "")
            m_eps = re.search(r"__EPS__ (\d+)", txt); txt = re.sub(r"__EPS__ \d+\n?", "", txt)
            if m_eps:
                saved = int(m_eps.group(1))
                if REC.get("chain_base") is None: REC["chain_base"] = saved - 0
                REC["saved_total"] = saved - REC["chain_base"]
            eps = re.findall(r"Recording episode (\d+)", txt); phase = REC["phase"]; ep = REC["episode"]
            if eps:
                newep = int(eps[-1]); nrec = len(eps)   # a re-record repeats the same episode number, so count the lines too
                if newep != ep or nrec != REC.get("_nrec"): ep = newep; REC["ep_started"] = time.time(); REC["_nrec"] = nrec
                last_reset = txt.rfind("Reset the environment"); last_rec = txt.rfind("Recording episode")
                phase = "reset" if last_reset > last_rec else "recording"
            if "auto-ENTER" in txt and not REC.get("_cal_noted"):
                REC["_cal_noted"] = True
                with lock: state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {REC['name']}: motor calibration differed from the file; applied the file (auto-ENTER)"})
            # pause markers come from the busheal plugin; while paused the episode clock on the page stands still
            paused_now = txt.rfind("__PAUSED__") > txt.rfind("__RESUMED__")
            if paused_now and not REC.get("paused_at"):
                REC["paused_at"] = time.time()
                with lock: state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {REC['name']} paused"})
            elif not paused_now and REC.get("paused_at"):
                REC["ep_started"] += time.time() - REC["paused_at"]; REC["paused_at"] = 0
                with lock: state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {REC['name']} resumed"})
            if "__RECORD_EXIT__" in txt: alive = False
            if not alive and re.search(r"Traceback|Error", txt):
                lines = [l for l in txt.strip().splitlines() if l.strip() and "__RECORD_EXIT__" not in l and "__CTL__" not in l]
                # torchcodec prints a harmless load failure on every start (it falls back to pyav); never report that as the cause
                errl = next((l for l in reversed(lines) if re.search(r"Error|error:", l) and "torchcodec" not in l and "nvrtc" not in l), lines[-1] if "Traceback" in txt and lines else "")   # why: without a Traceback the only "Error" is the torchcodec noise, and the old fallback blamed a random last line on a clean exit
                REC["err"] = errl.strip()[:160]
            if not alive and (time.time() - REC["started"] > 20):
                with lock: REC.update(running=False, done=True, phase="done" if not REC["err"] else "error", log=txt[-1500:]); state["events"].insert(0, {"t": time.time(), "msg": f"RECORD {REC['name']} finished · {ep} episodes" if not REC["err"] else f"RECORD failed: {REC['err'][:80]}"})
                continue
            with lock: REC.update(episode=ep, phase=phase, log=txt[-1500:], done_total=REC.get("saved_total", REC["done_total"]))
        except Exception as e: log("record loop:", repr(e))
        time.sleep(2)
# ---------- calibration (read-only) ----------
# /api/calib, /api/locate and /api/ik expose the overhead-camera / arm calibration to the wall page. They read the repo
# copies of the Jetson's calib files (jetson/calib/top_table.json + top_layout.json, synced after each calibration) and
# never talk to the Jetson or move the arm: /api/ik is a dry-run calculator. /api/calib is pure JSON; /api/locate imports
# jetson/cam_calib.py (cv2 + numpy) and /api/ik also needs scipy, both lazily, so the dashboard still starts without them.
# Frames: mat = Fiskars 18x12 in mat, origin at the arm-side left corner, X along the 18 in edge, Y away from the arm
# (inches in the layout, mm in the files); arm = URDF base_link, X forward, Y to the arm's left, Z up.
# The FRONT camera (oblique, ~42 deg tilt) has only a pose: jetson/calib/front_campose.json from `cam_calib.py campose --name front`
# (rvec/tvec table -> camera, pinhole camera_matrix, no distortion). /api/calib derives the mat-surface homography from it
# (camera.front, pure Python) and /api/locate?cam=front maps a front pixel to the mat point under it ON THE SURFACE (h_mm must be 0).
REPO = os.path.dirname(HERE)
CALIB_DIR = os.environ.get("SO101_CALIB_DIR", os.path.join(REPO, "jetson", "calib"))
CAM_CALIB_PY = os.path.join(REPO, "jetson", "cam_calib.py")
MAT_IN = {"width_in": 18, "depth_in": 12}
MM_PER_IN = 25.4
NO_TABLE_ERR = "no table calibration yet (run so101-calib table)"
_calib_cache = {}   # path -> ((mtime_ns, size), parsed json); the files change rarely, but they do change
_calib_lock = threading.Lock(); _cam_calib_mod = []
def calib_paths(): return os.path.join(CALIB_DIR, "top_table.json"), os.path.join(CALIB_DIR, "top_layout.json")
def load_calib_json(path):
    """Parsed JSON, re-read whenever the file's mtime or size changes. Raises OSError / ValueError like json.load."""
    st = os.stat(path); key = (st.st_mtime_ns, st.st_size)
    hit = _calib_cache.get(path)
    if hit and hit[0] == key: return hit[1]
    with open(path) as f: data = json.load(f)
    _calib_cache[path] = (key, data); return data
def load_table():
    """The table calibration dict, or None when no table file exists yet."""
    try: return load_calib_json(calib_paths()[0])
    except FileNotFoundError: return None
def cam_calib():
    """jetson/cam_calib.py, imported on first use (it needs cv2 + numpy, which the dashboard otherwise does not)."""
    with _calib_lock:
        if not _cam_calib_mod:
            import importlib.util
            spec = importlib.util.spec_from_file_location("so101_cam_calib", CAM_CALIB_PY)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); _cam_calib_mod.append(mod)
        return _cam_calib_mod[0]
def _in(mm, nd=3): return round(float(mm) / MM_PER_IN, nd)
def camera_pose(est):
    """camera_pose_est from top_table.json (cam_calib.py campose: position_mm = camera centre in the table frame, z up;
    optical_axis / up_table unit vectors in the table frame; tilt_deg, axis_hits_mm, hfov_deg, image_size, reproj_rms_px,
    source, created) passed through as is, plus position_in. None when absent or not usable."""
    if not isinstance(est, dict): return None
    pos = est.get("position_mm")
    try:
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3): return None
        return {**est, "position_mm": [float(v) for v in pos], "position_in": [_in(v) for v in pos]}
    except (TypeError, ValueError): return None
def front_pose_path(): return os.path.join(CALIB_DIR, "front_campose.json")
def _rodrigues(r):
    """3x3 rotation from a Rodrigues vector (OpenCV convention), pure Python so /api/calib stays stdlib-only."""
    import math
    rx, ry, rz = (float(v) for v in r); th = math.sqrt(rx * rx + ry * ry + rz * rz)
    if th < 1e-12: return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = rx / th, ry / th, rz / th; c, s = math.cos(th), math.sin(th); C = 1.0 - c
    return [[c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s],
            [ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s],
            [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C]]
def _inv3(m):
    """Inverse of a 3x3 (lists), or None when singular."""
    import math
    (a, b, c), (d, e, f), (g, h, i) = m
    A, B, C = e * i - f * h, f * g - d * i, d * h - e * g; det = a * A + b * B + c * C
    if not det or not math.isfinite(det): return None
    s = 1.0 / det
    return [[A * s, (c * h - b * i) * s, (b * f - c * e) * s], [B * s, (a * i - c * g) * s, (c * d - a * f) * s], [C * s, (b * g - a * h) * s, (a * e - b * d) * s]]
def pose_homographies(est):
    """(table mm -> px, px -> table mm) 3x3 pair for the mat surface (z = 0) from a campose dict: [u v 1]^T ~ K [r1 r2 t] [x y 1]^T with
    R = Rodrigues(rvec), t = tvec (table -> camera, mm), K = camera_matrix. None when rvec/tvec/camera_matrix are missing or the matrix is singular."""
    import math
    try:
        R = _rodrigues(est["rvec"]); t = [float(v) for v in est["tvec"]]; K = [[float(v) for v in row] for row in est["camera_matrix"]]
        if len(t) != 3 or len(K) != 3 or any(len(row) != 3 for row in K): return None
        E = [[R[i][0], R[i][1], t[i]] for i in range(3)]
        H = [[sum(K[i][k] * E[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        Hi = _inv3(H)
        if Hi is None or not Hi[2][2]: return None
        w = abs(Hi[2][2]); Hi = [[v / w for v in row] for row in Hi]   # px -> table scaled so |[2][2]| = 1 like the top's; a POSITIVE scale keeps the sign of its third row = 1/depth (behind-the-camera test)
        if any(not math.isfinite(v) for row in H + Hi for v in row): return None
        return H, Hi
    except (KeyError, TypeError, ValueError, IndexError): return None
def camera_wrist():
    """camera.wrist for /api/calib: wrist_campose.json (cam_calib.py handeye: mount in gripper_link) passed through as is, plus
    position_in; None when the file is absent or unusable."""
    path = os.path.join(CALIB_DIR, "wrist_campose.json")
    try: w = load_calib_json(path)
    except FileNotFoundError: return None
    except (OSError, ValueError) as e: log("api/calib wrist:", repr(e)); return None
    pos = w.get("position_mm") if isinstance(w, dict) else None
    if not (isinstance(pos, (list, tuple)) and len(pos) == 3): return None
    try: pos = [float(v) for v in pos]
    except (TypeError, ValueError): return None
    return {**w, "position_mm": pos, "position_in": [round(v / 25.4, 4) for v in pos]}
def camera_front():
    """camera.front for /api/calib: the front camera's pose (front_campose.json passed through like camera_pose, or the live marker
    re-solve while it differs from the file: front_pose_step) plus the mat-surface homography pair derived from it, and the tracking
    state (pose_live / pose_at, live / live_at / moved_mm mirroring the top's, holding / markers_seen, live_solve). None when the file
    is absent or unusable."""
    path = front_pose_path()
    try: est = load_calib_json(path)
    except FileNotFoundError: return None
    except (OSError, ValueError) as e: log("api/calib front:", repr(e)); return None
    live = front_live(path); use = live["est"] if live else est
    pose = camera_pose(use); hh = pose_homographies(use) if pose else None
    if not hh: return None
    with _front_lock: holding, seen = bool(FRONT["holding"]), FRONT["markers_seen"]
    return {"pose": pose, "homography_px_to_table_mm": hh[1], "homography_table_to_px": hh[0], "reproj_rms_px": use.get("reproj_rms_px"), "created": est.get("created"),
            "pose_live": bool(live), "pose_at": live["at"] if live else None, "live": bool(live), "live_at": live["at"] if live else None, "moved_mm": live["moved_mm"] if live else None,
            "holding": holding, "markers_seen": seen, "live_solve": est.get("live_solve")}
def _table_to_arm(tab, x, y):
    """cam_calib.table_to_arm without cv2: arm mm from table mm with the table file's arm_base_mm / arm_yaw_deg, None without an arm pose."""
    import math
    base = (tab or {}).get("arm_base_mm")
    if base is None: return None
    yaw = math.radians(float(tab.get("arm_yaw_deg", 0.0))); dx, dy = x - float(base[0]), y - float(base[1])
    return (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))
def _q(q, key, default=None):
    """One float query parameter; ValueError when missing without a default, or not a finite number."""
    import math
    raw = q.get(key)
    if not raw or raw[0] == "":
        if default is None: raise ValueError(f"missing {key}")
        return float(default)
    v = float(raw[0])
    if not math.isfinite(v): raise ValueError(f"{key} must be a finite number")
    return v
def api_calib():
    tpath, lpath = calib_paths()
    try:
        tab = load_table()
        if tab is None: return {"ok": False, "err": NO_TABLE_ERR}
        try: layout = load_calib_json(lpath)
        except (OSError, ValueError): layout = {}
        markers_mm = tab.get("layout_markers") or {}
        ms_mm = tab.get("marker_size_mm")
        fit = tab.get("arm_fit"); arm = None
        if fit:
            base = fit.get("arm_base_mm") or tab.get("arm_base_mm") or [0.0, 0.0]; res = fit.get("residual_xy_mm") or []
            arm = {"base_mm": [round(float(v), 1) for v in base], "base_in": [_in(v, 2) for v in base], "yaw_deg": round(float(fit.get("arm_yaw_deg", tab.get("arm_yaw_deg", 0.0))), 2),
                   "table_z_mm": round(float(fit.get("table_z_mm", 0.0)), 1), "joint_offsets_deg": {k: round(float(v), 2) for k, v in (fit.get("joint_offsets_deg") or {}).items()},
                   "wrist_roll_deg": float(fit.get("wrist_roll_deg", 0.0)), "fitted": fit.get("fitted"), "residual_xy_mm_max": round(max(float(r) for r in res), 2) if res else None,
                   "points": len(fit.get("points") or []) or int(fit.get("n_points", 0))}
        live = scene_live(tpath)   # the marker re-solve while it differs from the file (docs/mat-scene.md, "self-healing")
        plive = scene_pose_live(tpath); pose = (camera_pose(plive["est"]) if plive else None) or camera_pose(tab.get("camera_pose_est"))   # the live camera pose while it differs from the file
        return {"ok": True, "units": tab.get("units", "in"), "mat": dict(MAT_IN), "image_size": tab.get("image_size"),
                "marker_size_in": _in(ms_mm) if ms_mm else layout.get("marker_size"),
                "layout_markers_in": {str(k): [_in(v[0]), _in(v[1])] for k, v in markers_mm.items()},
                "homography_px_to_table_mm": live["H"] if live else tab.get("homography_px_to_table"), "has_intrinsics": tab.get("rvec") is not None,
                "stored_homography": tab.get("homography_px_to_table"), "live": bool(live), "live_at": live["at"] if live else None, "shift_px": live["shift_px"] if live else None,
                "holding": bool(SCENE.get("holding")), "calib_ok": SCENE.get("calib_ok"), "calib_note": SCENE.get("calib_note"), "markers_seen": SCENE.get("markers_seen"), "live_solve": tab.get("live_solve"),
                "camera": {"residual_mm_max": round(float(tab["residual_mm_max"]), 2) if tab.get("residual_mm_max") is not None else None, "created": tab.get("created"),
                           "markers_used": tab.get("markers_used") or sorted(int(k) for k in markers_mm), "pose": pose,
                           "pose_live": bool(plive), "pose_at": plive["at"] if plive else None, "pose_moved_mm": plive["moved_mm"] if plive else None, "front": camera_front(), "wrist": camera_wrist()},
                "arm": arm, "files": {"table": os.path.abspath(tpath), "layout": os.path.abspath(lpath), "table_mtime": os.stat(tpath).st_mtime}}
    except Exception as e:
        log("api/calib:", repr(e)); return {"ok": False, "err": f"{type(e).__name__}: {e}"}
def api_locate(q):
    """Pixel (u, v) of the overhead camera (cam=top, the default), at h_mm above the mat -> mat mm/in and arm mm/in.
    cam=front: a front-camera pixel through the pose-derived homography, mat surface only (h_mm must be 0)."""
    try: u, v, h = _q(q, "u"), _q(q, "v"), _q(q, "h_mm", 0.0)
    except ValueError as e: return {"ok": False, "err": f"need u=<px>&v=<px> (numbers), optional h_mm: {e}"}
    cam = (q.get("cam") or ["top"])[0] or "top"
    if cam == "front": return locate_front(u, v, h)
    if cam != "top": return {"ok": False, "err": f"unknown cam {cam!r} (top or front)"}
    try:
        tab = load_table()
        if tab is None: return {"ok": False, "err": NO_TABLE_ERR}
        live = scene_live(calib_paths()[0])
        if live: tab = {**tab, "homography_px_to_table": live["H"]}   # the camera moved: the markers' re-solve, not the file (homography-only calibrations)
        cc = cam_calib(); x, y = cc.pixel_to_table(tab, u, v, h); arm = cc.table_to_arm(tab, x, y)
    except ImportError as e: return {"ok": False, "err": f"cam_calib unavailable: {e}"}
    except Exception as e: return {"ok": False, "err": str(e) or repr(e)}   # CalibError (behind the camera, height without intrinsics), bad file...
    return {"ok": True, "pixel": [u, v], "height_mm": h, "table_mm": [round(x, 2), round(y, 2)], "table_in": [_in(x), _in(y)],
            "arm_mm": [round(arm[0], 2), round(arm[1], 2)] if arm else None, "arm_in": [_in(arm[0]), _in(arm[1])] if arm else None}
def locate_front(u, v, h):
    """/api/locate?cam=front: the mat point under front pixel (u, v) ON THE SURFACE; arm frame from the table file when it has an arm pose."""
    if h: return {"ok": False, "err": "cam=front maps the mat surface only: h_mm must be 0 (the front camera has no height model)"}
    try:
        fr = camera_front()
        if fr is None: return {"ok": False, "err": "no front camera pose yet (run so101-calib campose --name front)"}
        H = fr["homography_px_to_table_mm"]; w = H[2][0] * u + H[2][1] * v + H[2][2]
        if w <= 1e-9: return {"ok": False, "err": "that pixel does not meet the mat surface in front of the camera"}
        x, y = (H[0][0] * u + H[0][1] * v + H[0][2]) / w, (H[1][0] * u + H[1][1] * v + H[1][2]) / w
        arm = _table_to_arm(load_table(), x, y)
    except Exception as e: return {"ok": False, "err": str(e) or repr(e)}
    return {"ok": True, "cam": "front", "pixel": [u, v], "height_mm": 0.0, "table_mm": [round(x, 2), round(y, 2)], "table_in": [_in(x), _in(y)],
            "arm_mm": [round(arm[0], 2), round(arm[1], 2)] if arm else None, "arm_in": [_in(arm[0]), _in(arm[1])] if arm else None}
def api_ik(q):
    """Dry-run inverse kinematics for a mat point (inches) h_in above the mat, with the fitted offsets. Moves nothing."""
    try:
        x_in, y_in, h_in = _q(q, "x_in"), _q(q, "y_in"), _q(q, "h_in", 0.0)
        roll = _q(q, "roll") if q.get("roll") and q["roll"][0] != "" else None
    except ValueError as e: return {"ok": False, "err": f"need x_in=<mat x>&y_in=<mat y> (inches), optional h_in, roll: {e}"}
    try:
        tab = load_table()
        if tab is None: return {"ok": False, "err": NO_TABLE_ERR}
        fit = tab.get("arm_fit")
        if not fit or not tab.get("arm_base_mm"): return {"ok": False, "err": "no arm fit yet (run tools/jog_capture.py --fit)"}
        cc = cam_calib()
        ax, ay = cc.table_to_arm(tab, x_in * MM_PER_IN, y_in * MM_PER_IN); z = float(fit.get("table_z_mm", 0.0)) + h_in * MM_PER_IN
        if roll is None: roll = float(fit.get("wrist_roll_deg", 0.0))   # why: the jaw tip is off the roll axis, so the roll the fit was jogged at is the one the fit is valid for
        res = cc.ik_tip([ax, ay, z], fit.get("joint_offsets_deg") or {}, roll_deg=roll)
    except ImportError as e: return {"ok": False, "err": f"cam_calib unavailable: {e}"}
    except Exception as e: return {"ok": False, "err": str(e) or repr(e)}   # CalibError: unreachable, or scipy missing
    return {"ok": True, "target_mm": [round(v, 2) for v in (ax, ay, z)], **res, "wrist_roll_deg": roll}
# ---------- mat scene: what is on the mat under the overhead camera + self-healing camera calibration (docs/mat-scene.md) ----------
# A background thread (scene_loop) decodes the newest top-camera JPEG that jetson_cam_loop left in cam_frames and runs
# dashboard/scene_detect.py's MatScene on it about twice a second; the result sits in SCENE and /api/scene serves it. No
# request handler ever waits for a detect. The MatScene is built from CALIB_DIR/top_table.json and rebuilt when that file
# changes; the empty-mat background is CALIB_DIR/top_mat_bg.png (POST /api/scene/background writes it, start-up loads it).
# Self-healing: MatScene re-fits the homography from the ArUco markers in every frame. When the re-fit is sound (calib_ok and
# at least 3 markers) and the markers sit more than SCENE_SHIFT_PX from where the stored solve puts them, that live solve is
# what /api/calib and /api/locate hand out, a LOG line says so (one per SCENE_NOTE_S), and it is written back into
# top_table.json (one write per SCENE_PERSIST_S) with the arm pose (arm_base_mm / arm_yaw_deg / arm_fit) and "created"
# untouched: the arm did not move when the camera stand did. Fewer than 3 markers: the last good solve stays ("holding").
# calib_ok false (the mat itself moved > 150 mm / 30 deg, or the markers are still off after the re-fit): nothing is served
# live or written, the note is exposed for the page. cv2 / numpy / scene_detect are imported lazily in the thread and the
# handlers, so the dashboard starts without them and every scene endpoint answers {"ok": false, "err": ...}.
SCENE_DETECT_PY = os.path.join(HERE, "scene_detect.py")
SCENE_BG_NAME = "top_mat_bg.png"
SCENE_PERIOD_S = 0.5            # detect period (the module costs 5-8 ms per frame)
SCENE_BG_EVERY = 10             # update_background (module defaults) every Nth detect
SCENE_SHIFT_PX = 3.0            # markers further than this from the stored solve -> live solve
SCENE_NOTE_S = 30.0             # LOG line rate limit
SCENE_PERSIST_S = 60.0          # write-back rate limit
SCENE_BG_SPACING_S = 0.5        # between the frames of a median background
SCENE_BG_MAX_FRAMES = 15
SCENE_FRAME_MAX_AGE_S = 3.0     # a top frame older than this means the camera is not live (same rule as /cam/top.jpg)
SCENE_MIN_HITS = 3              # an object must be seen in 3 consecutive detects (~1.5 s) before the map shows it: kills one-frame shadows and flicker
SCENE_JOINTS_MAX_AGE_S = 2.0    # follower joints older than this (or not in degrees) -> no kinematic arm mask, the edge heuristic alone
SCENE_LABELS_PY = os.path.join(REPO, "tools", "scene_labels.py")
SCENE_LABEL_EVERY_S = 0.45      # what-is-it labels (tools/scene_labels.py) are recomputed at most about twice a second; between runs /api/scene serves the cached ones
SCENE = {}
_scene_lock = threading.RLock(); _scene_mod = []
def scene_reset():
    """Forget every scene result and the live solve (start-up, and the tests between cases)."""
    SCENE.clear()
    SCENE.update(res=None, t=0.0, err="no detection yet", scene=None, table_key=None, n=0, live=None, holding=False, calib_ok=None, calib_note=None,
                 markers_seen=None, noted_at=0.0, persisted_at=0.0, writes=0, arm_px=None, arm_err=None,
                 pose=None, pose_H=None, pose_err=None, pose_noted_at=0.0, labels=None, labels_t=0.0, labels_err=None, label_runs=0, label_ms=None)   # the overhead camera's live pose (scene_pose), the live H it was solved from, the one-line reason it is not tracked
scene_reset()
def scene_joints(now):
    """The follower's latest joints in LeRobot degrees from the arm stream (state['arms']['follower'], what /api/state serves),
    or None: no follower, not in 'deg' mode (no motor calibration on the Jetson), or received more than SCENE_JOINTS_MAX_AGE_S ago."""
    with lock: fol = (state.get("arms") or {}).get("follower")
    if not isinstance(fol, dict) or fol.get("mode") != "deg": return None
    try:
        if now - float(fol.get("rx") or 0.0) > SCENE_JOINTS_MAX_AGE_S: return None
    except (TypeError, ValueError): return None
    pos = fol.get("pos")
    return dict(pos) if isinstance(pos, dict) and pos else None
def scene_arm_px(tab, sc, now):
    """The arm's silhouette for MatScene.detect(arm_px=...): cam_calib.arm_mask_px_from_joints on the fresh follower joints and
    the table calibration (the scene's current homography stands in for the file's, for the no-camera-pose fallback), or None
    when there are no fresh joints or the helper cannot (no arm pose in the file, cam_calib unavailable): one LOG line per
    distinct reason, and the detector falls back to its edge heuristic (docs/mat-scene.md, "Kinematic arm mask")."""
    pos = scene_joints(now)
    if pos is None: return None
    try:
        calib = dict(tab); calib["homography_px_to_table"] = sc.H.tolist()
        if SCENE.get("pose"): calib["camera_pose_est"] = SCENE["pose"]["est"]   # the served pose (live re-solve after a bump, scene_pose): the silhouette follows the camera
        arm = cam_calib().arm_mask_px_from_joints(pos, calib)
        SCENE["arm_err"] = None
        return arm if arm.get("points") else None
    except Exception as e:
        msg = f"arm mask skipped: {type(e).__name__}: {e}"
        if SCENE.get("arm_err") != msg: log("scene:", msg)
        SCENE["arm_err"] = msg; return None
def scene_mod():
    """dashboard/scene_detect.py, imported on first use (cv2 + numpy). ImportError without them."""
    with _scene_lock:
        if not _scene_mod:
            import importlib.util
            spec = importlib.util.spec_from_file_location("so101_scene_detect", SCENE_DETECT_PY)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); _scene_mod.append(mod)
        return _scene_mod[0]
def scene_bg_path(): return os.path.join(CALIB_DIR, SCENE_BG_NAME)
def _table_key(tpath):
    st = os.stat(tpath); return (st.st_mtime_ns, st.st_size)
_scene_labels_mod = []
def scene_labels_mod():
    """tools/scene_labels.py (light: numpy + tools/keepout.py, never tidy), imported on first use."""
    with _scene_lock:
        if not _scene_labels_mod:
            import importlib.util
            spec = importlib.util.spec_from_file_location("so101_scene_labels", SCENE_LABELS_PY)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); _scene_labels_mod.append(mod)
        return _scene_labels_mod[0]
def scene_label_step(res, frame, tab, now):
    """Labels for the blobs of one detect (kind, keep-out, the ring), cached in SCENE['labels'] and rate-limited to one run per SCENE_LABEL_EVERY_S. Never raises:
    a failure leaves the objects unlabelled and the reason in SCENE['labels_err']."""
    if now - SCENE["labels_t"] < SCENE_LABEL_EVERY_S and SCENE["labels"] is not None: return
    t0 = time.perf_counter()
    try:
        objs, ring = scene_labels_mod().label_objects(res.get("objects") or [], frame, cam_calib(), tab)
        kos = [{"id": o.get("id"), "x_in": o["x_in"], "y_in": o["y_in"], **o["keepout"]} for o in objs if o.get("keepout")]
        SCENE.update(labels={"by_id": {o.get("id"): {k: o[k] for k in ("kind", "keepout", "label_src")} for o in objs}, "ring": ring, "keepouts": kos}, labels_err=None)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if SCENE["labels_err"] != msg: log("scene: labels:", msg)
        SCENE.update(labels=None, labels_err=msg)
    SCENE.update(labels_t=now, label_runs=SCENE["label_runs"] + 1, label_ms=round((time.perf_counter() - t0) * 1000, 2))
def scene_get():
    """The MatScene for the current top_table.json, rebuilt when the file changes (our own write-back excepted: see
    scene_persist) with the saved background reloaded. Raises FileNotFoundError / ImportError / ValueError."""
    tpath = calib_paths()[0]; key = _table_key(tpath)
    with _scene_lock:
        if SCENE["scene"] is not None and SCENE["table_key"] == key: return SCENE["scene"]
        sd = scene_mod(); sc = sd.MatScene(load_calib_json(tpath), min_hits=SCENE_MIN_HITS)
        bg = scene_bg_path()
        if os.path.exists(bg):
            try: sc.load_background(bg)
            except Exception as e: log("scene: background", bg, "not loaded:", repr(e))
        SCENE.update(scene=sc, table_key=key, n=0, live=None, holding=False, calib_ok=None, calib_note=None, markers_seen=None, pose=None, pose_H=None)
        log("scene: built from", tpath, "(background %s)" % ("ok" if sc.bg is not None else "missing"))
        return sc
def scene_live(tpath):
    """The live solve to serve for this table file, or None (none, or it belongs to a table file that has since changed)."""
    live = SCENE.get("live")
    if not live: return None
    try: return live if SCENE.get("table_key") == _table_key(tpath) else None
    except OSError: return None
def scene_frame(now=None):
    """(BGR frame, frame time) decoded from the newest live top-camera JPEG, or (None, reason)."""
    now = time.time() if now is None else now
    f = cam_frames.get("top")
    if not f or now - f[1] > SCENE_FRAME_MAX_AGE_S: return None, "top camera not live"
    import cv2, numpy as np
    img = cv2.imdecode(np.frombuffer(f[0], np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None, "top frame did not decode"
    return img, f[1]
def _marker_shift_px(H_a, H_b, layout):
    """How far (max, px) the layout's marker centres land apart in the image under two px->mm homographies."""
    import numpy as np
    if not layout: return 0.0
    pts = np.hstack([np.array(list(layout.values()), np.float64), np.ones((len(layout), 1))])
    def proj(H):
        q = pts @ np.linalg.inv(np.array(H, np.float64)).T; return q[:, :2] / q[:, 2:3]
    return float(np.hypot(*(proj(H_a) - proj(H_b)).T).max())
def scene_persist(tab, live, now, pose=None):
    """Write the live solve into top_table.json. Everything else in the file stays (arm_base_mm, arm_yaw_deg, arm_fit, created,
    layout...); the per-marker residuals of the old solve go, residual_*_max become the re-fit's, and a live_solve block says
    when and why. The live camera pose (scene_pose), when there is one, goes in as camera_pose_est with the same write.
    Atomic (tmp + replace). The scene is NOT rebuilt for our own write: its tracks and mat-move reference stay."""
    tpath = calib_paths()[0]
    new = {k: v for k, v in tab.items() if k not in ("residual_mm", "residual_in")}
    new["homography_px_to_table"] = live["H"]
    if live.get("err_mm") is not None: new["residual_mm_max"] = round(float(live["err_mm"]), 4); new["residual_in_max"] = round(float(live["err_mm"]) / MM_PER_IN, 5)
    new["live_solve"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "shift_px": live["shift_px"], "markers_seen": live["markers"], "residual_mm_max": live.get("err_mm"),
                         "note": "homography re-solved from the markers by the dashboard after the camera moved; arm pose and 'created' are from the original calibration"}
    if pose:
        new["camera_pose_est"] = pose["est"]
        new["live_solve"]["camera_pose"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pose["at"])), "moved_mm": pose["moved_mm"], "markers_seen": pose["markers"]}
    tmp = tpath + ".tmp"
    with open(tmp, "w") as f: json.dump(new, f, indent=1)
    os.replace(tmp, tpath)
    key = _table_key(tpath); SCENE["table_key"] = key; SCENE["writes"] += 1; _calib_cache[tpath] = (key, new)
    log(f"scene: live camera solve written to {tpath} (markers moved {live['shift_px']} px)")
def scene_heal(sc, res, tab, now, frame=None):
    """After a detect: decide whether the markers' re-solve is served live, noted in the LOG and written back; with the frame,
    the overhead camera's pose is re-solved from the same markers (scene_pose) and travels with the write-back."""
    n = int(res.get("n_markers_seen") or 0); ok = bool(res.get("calib_ok")); err = res.get("calib_err_mm")
    SCENE.update(markers_seen=n, calib_ok=ok, calib_note=res.get("calib_note"), holding=n < 3)
    if n < 3 or not ok: return                                               # hold the last good solve; a moved mat is never written back
    H_live = sc.H.tolist(); H_stored = tab.get("homography_px_to_table")
    shift = _marker_shift_px(H_stored, H_live, sc.layout) if H_stored else 0.0
    if shift <= SCENE_SHIFT_PX: SCENE["live"] = None; SCENE["pose"] = None; return   # the file is right (or it is the solve + pose we wrote)
    live = SCENE["live"]
    if live is None or live["H"] != H_live: live = {"at": now}                # a new re-solve (MatScene only re-fits when the markers drift > 3 mm): its own time
    live.update(H=H_live, shift_px=round(shift, 1), markers=n, err_mm=err); SCENE["live"] = live
    if frame is not None:
        try: scene_pose(frame, tab, H_live, n, now)
        except Exception as e: log("scene: pose:", repr(e))
    if now - SCENE["noted_at"] >= SCENE_NOTE_S:
        SCENE["noted_at"] = now; event(f"overhead camera moved {shift:.0f} px, re-solved from {n} markers ({(err or 0):.1f} mm)")
    if now - SCENE["persisted_at"] >= SCENE_PERSIST_S:
        try: scene_persist(tab, live, now, SCENE["pose"]); SCENE["persisted_at"] = now
        except Exception as e: log("scene: write-back failed:", repr(e))
def scene_pose(frame, tab, H_live, n, now):
    """The overhead camera's pose from the markers of this frame with the stored pose's FIXED field of view (nominal_K at the
    frame's size), solved once per live homography. The served pose (SCENE['pose'], else the file's camera_pose_est) is only
    replaced when the camera centre moved more than POSE_MOVE_MM or the optical axis turned more than POSE_TURN_DEG, so noise
    never jitters the 3D stage; one event line per POSE_NOTE_S. No pose or no hfov_deg in the file: nothing is tracked, one
    LOG line says why (the FOV belongs to the lens and is pinned by `campose --lens-height`; it is never re-estimated here)."""
    stored = tab.get("camera_pose_est")
    if pose_hfov(stored) is None:
        pose_note(SCENE, "overhead camera pose has no hfov; run so101-calib campose --lens-height" if isinstance(stored, dict) else
                  "overhead camera has no pose; run so101-calib campose --lens-height"); return
    if SCENE["pose_H"] == H_live: return                                     # already solved for this re-fit
    try: est = pose_solve(frame, pose_layout(tab), stored)
    except Exception as e: pose_note(SCENE, f"overhead camera pose not re-solved: {type(e).__name__}: {e}"); return
    SCENE["pose_err"] = None; SCENE["pose_H"] = H_live
    served = SCENE["pose"]["est"] if SCENE["pose"] else stored
    moved, mm, deg = pose_moved(est, served)
    if not moved: return
    SCENE["pose"] = {"est": est, "at": now, "moved_mm": round(mm, 1), "turned_deg": round(deg, 2), "markers": n}
    if now - SCENE["pose_noted_at"] >= POSE_NOTE_S:
        SCENE["pose_noted_at"] = now; event(f"overhead camera pose re-solved: moved {mm:.0f} mm")
def scene_pose_live(tpath):
    """The overhead camera's live pose to serve for this table file, or None (none, or the file has since changed)."""
    live = SCENE.get("pose")
    if not live: return None
    try: return live if SCENE.get("table_key") == _table_key(tpath) else None
    except OSError: return None
def scene_step(now=None):
    """One detect on the newest top frame; updates SCENE. Returns the detect dict, or None with the reason in SCENE['err']."""
    now = time.time() if now is None else now
    try:
        tab = load_table()
        if tab is None: SCENE["err"] = NO_TABLE_ERR; return None
        frame, ft = scene_frame(now)
        if frame is None: SCENE["err"] = ft; return None
        sc = scene_get()
    except ImportError as e: SCENE["err"] = f"scene_detect unavailable: {e}"; return None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if SCENE["err"] != msg: log("scene:", msg)                           # once per distinct failure, not twice a second
        SCENE["err"] = msg; return None
    arm_px = scene_arm_px(tab, sc, now)
    with _scene_lock:
        res = sc.detect(frame, t=now, arm_px=arm_px)
        SCENE["n"] += 1; SCENE["arm_px"] = arm_px
        if SCENE["n"] % SCENE_BG_EVERY == 0: sc.update_background(frame)
        try: scene_heal(sc, res, tab, now, frame)
        except Exception as e: log("scene: heal:", repr(e))
        scene_label_step(res, frame, tab, now)
        SCENE.update(res=res, t=now, err=None)
    return res
def scene_loop():
    while True:
        try: scene_step()
        except Exception as e: log("scene loop:", repr(e))
        time.sleep(SCENE_PERIOD_S)
def api_scene():
    """The latest detect() dict plus ok / age_s / calibration-healing status; ok:false with the reason when there is none."""
    with _scene_lock: res, t, live, labels = SCENE["res"], SCENE["t"], SCENE["live"], SCENE["labels"]
    if res is None or not scene_frame_live():
        err = SCENE["err"] or "no detection yet"
        try:
            if load_table() is None: err = NO_TABLE_ERR
            elif not scene_frame_live(): err = "top camera not live"
        except Exception as e: err = f"{type(e).__name__}: {e}"
        return {"ok": False, "err": err, "age_s": round(time.time() - t, 2) if res is not None else None}
    by_id = (labels or {}).get("by_id", {})   # an object newer than the cached labels (< SCENE_LABEL_EVERY_S old) has no kind yet
    objects = [{**o, **by_id[o.get("id")]} if o.get("id") in by_id else {**o, "kind": None, "keepout": None, "label_src": None} for o in res.get("objects") or []]
    return {**res, "objects": objects, "ring": (labels or {}).get("ring"), "keepouts": (labels or {}).get("keepouts", []), "ok": True, "age_s": round(max(0.0, time.time() - t), 2), "background": res.get("background", "missing"),
            "calib_live": bool(live), "calib_live_at": live["at"] if live else None, "shift_px": live["shift_px"] if live else None, "holding": bool(SCENE["holding"])}
def scene_frame_live(now=None):
    f = cam_frames.get("top"); return bool(f) and (time.time() if now is None else now) - f[1] <= SCENE_FRAME_MAX_AGE_S
def api_scene_background(body):
    """POST /api/scene/background: learn the empty mat from the current top frame ({"frames": N} = median of N frames,
    SCENE_BG_SPACING_S apart, capped), save it beside the calibration and refresh the map at once."""
    try: n = max(1, min(SCENE_BG_MAX_FRAMES, int((body or {}).get("frames", 1) or 1)))
    except (TypeError, ValueError, AttributeError): return {"ok": False, "err": "frames must be a whole number"}
    try:
        if load_table() is None: return {"ok": False, "err": NO_TABLE_ERR}
        frames = []
        for i in range(n):
            if i: time.sleep(SCENE_BG_SPACING_S)
            frame, ft = scene_frame()
            if frame is None: return {"ok": False, "err": ft}
            frames.append(frame)
        sc = scene_get()
        with _scene_lock:
            sc.set_background(frames if n > 1 else frames[0])
            os.makedirs(CALIB_DIR, exist_ok=True); path = scene_bg_path(); sc.save_background(path)
            res = sc.detect(frames[-1]); SCENE.update(res=res, t=time.time(), err=None)
        event(f"mat background learned from {n} frame{'s' if n > 1 else ''} (dashboard)")
        return {"ok": True, "saved": path, "frames": n}
    except ImportError as e: return {"ok": False, "err": f"scene_detect unavailable: {e}"}
    except Exception as e: log("api/scene/background:", repr(e)); return {"ok": False, "err": f"{type(e).__name__}: {e}"}
def scene_png():
    """render() of the last detect (or the background) as PNG bytes; None when there is no scene yet."""
    try:
        with _scene_lock:
            sc = SCENE["scene"]
            if sc is None: return None
            import cv2
            ok, buf = cv2.imencode(".png", sc.render())
        return buf.tobytes() if ok else None
    except Exception as e: log("scene.png:", repr(e)); return None
# ---------- live camera poses: the 3D stage follows the cameras when their mounts move (docs/mat-scene.md, "Self-healing calibration") ----------
# Each camera's pose (position + orientation in the table frame) is re-solved from the ArUco markers with that lens's FIXED
# horizontal field of view, hfov_deg, pinned once by `cam_calib.py campose --lens-height` (a tape-measured lens height): the FOV
# belongs to the lens and does not change when the mount moves, the pose does. It is never re-estimated here, and a pose file
# without hfov_deg is not tracked (one LOG line says so). Overhead camera: scene_pose, inside the scene loop, whenever the live
# homography is accepted; served as camera.pose and written into top_table.json with the homography's write-back. Front camera:
# front_pose_loop (a sibling thread, ~1 Hz) on the newest front JPEG jetson_cam_loop left in cam_frames; served as camera.front.pose
# with the mat-surface homographies re-derived from it, written to front_campose.json at most once a minute. Both only replace the
# served pose when the camera moved more than POSE_MOVE_MM or turned more than POSE_TURN_DEG (hysteresis against solve noise), and
# hold the last pose with fewer than 3 markers in view (holding + markers_seen).
POSE_MOVE_MM = 5.0              # hysteresis: a re-solve replaces the served pose only when the camera centre moved more than this...
POSE_TURN_DEG = 1.0             # ...or the optical axis turned more than this
POSE_NOTE_S = 30.0              # event line rate limit, per camera
POSE_LIVE_SRC = "live marker re-solve, hfov from "
FRONT_POSE_PERIOD_S = 1.0       # front loop period (detect + solve ~5 ms)
FRONT_POSE_PERSIST_S = 60.0     # front_campose.json write-back rate limit
NO_FRONT_POSE_ERR = "no front camera pose yet (run so101-calib campose --name front --lens-height)"
FRONT = {}
_front_lock = threading.RLock()
def front_reset():
    """Forget the front camera's live pose and tracking state (start-up, and the tests between cases)."""
    FRONT.clear()
    FRONT.update(pose=None, file_key=None, holding=False, markers_seen=None, err="no front detection yet", pose_err=None, noted_at=0.0, persisted_at=0.0, writes=0, n=0, t=0.0)
front_reset()
def pose_hfov(est):
    """The stored pose's fixed horizontal field of view in degrees, or None (no pose / no usable hfov_deg)."""
    try:
        f = float(est.get("hfov_deg")); return f if 0.0 < f < 180.0 else None
    except (AttributeError, TypeError, ValueError): return None
def pose_layout(tab):
    """The marker layout for cam_calib.camera_pose_from_markers from the table calibration (ids as ints, positions in mm)."""
    return {"dictionary": tab.get("dictionary", "DICT_4X4_50"), "marker_size_mm": float(tab["marker_size_mm"]),
            "markers": {int(k): [float(v[0]), float(v[1])] for k, v in (tab.get("layout_markers") or {}).items()}}
def pose_solve(frame, layout, stored):
    """cam_calib.camera_pose_from_markers on this frame with the STORED pose's field of view (nominal_K at the frame's size), the
    result carrying hfov_deg, image_size, units_note and 'created' over and `source` marked as a live re-solve from the original."""
    import cv2
    cc = cam_calib(); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame; h, w = gray.shape[:2]
    hfov = pose_hfov(stored); est = cc.camera_pose_from_markers(gray, layout, cc.nominal_K(w, h, hfov))
    src = str(stored.get("source") or "unknown source")
    if src.startswith(POSE_LIVE_SRC): src = src[len(POSE_LIVE_SRC):]
    est.update(hfov_deg=hfov, image_size=[w, h], source=POSE_LIVE_SRC + src, units_note=stored.get("units_note", "position_mm in the table frame (z up from the mat surface)"), created=stored.get("created"))
    return est
def pose_delta(a, b):
    """(mm the camera centre moved, degrees the optical axis turned) between two poses; 180 deg when either has no axis."""
    import math
    mm = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a["position_mm"], b["position_mm"])))
    aa, ab = a.get("optical_axis"), b.get("optical_axis")
    try:
        na, nb = math.sqrt(sum(float(x) ** 2 for x in aa)), math.sqrt(sum(float(x) ** 2 for x in ab))
        d = sum(float(x) * float(y) for x, y in zip(aa, ab)) / (na * nb); deg = math.degrees(math.acos(max(-1.0, min(1.0, d))))
    except (TypeError, ValueError, ZeroDivisionError): deg = 180.0
    return mm, deg
def pose_moved(new, served):
    """(moved beyond the hysteresis?, mm, deg) of a new solve against the pose currently served."""
    mm, deg = pose_delta(new, served); return (mm > POSE_MOVE_MM or deg > POSE_TURN_DEG), mm, deg
def pose_note(st, msg):
    """One LOG line per distinct reason a camera's pose is not being tracked (st['pose_err'] remembers the last)."""
    if st.get("pose_err") != msg: log("pose:", msg)
    st["pose_err"] = msg
def front_frame(now=None):
    """(BGR frame, frame time) decoded from the newest live front-camera JPEG, or (None, reason)."""
    now = time.time() if now is None else now
    f = cam_frames.get("front")
    if not f or now - f[1] > SCENE_FRAME_MAX_AGE_S: return None, "front camera not live"
    import cv2, numpy as np
    img = cv2.imdecode(np.frombuffer(f[0], np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None, "front frame did not decode"
    return img, f[1]
def front_live(path):
    """The front camera's live pose to serve for this pose file, or None (none, or the file has since changed under it)."""
    live = FRONT.get("pose")
    if not live: return None
    try: return live if FRONT.get("file_key") == _table_key(path) else None
    except OSError: return None
def front_persist(live, now):
    """Write the live front pose over front_campose.json (atomic) with a live_solve block; the cache follows so nothing is re-read."""
    path = front_pose_path(); new = dict(live["est"])
    new["live_solve"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "moved_mm": live["moved_mm"], "markers_seen": live["markers"],
                         "note": "pose re-solved from the markers by the dashboard after the camera moved; hfov_deg (and 'created') are from the original campose --lens-height"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(new, f, indent=1)
    os.replace(tmp, path)
    key = _table_key(path); FRONT["file_key"] = key; FRONT["writes"] += 1; _calib_cache[path] = (key, new); live["persisted"] = True
    log(f"front pose: live camera pose written to {path} (moved {live['moved_mm']} mm)")
def front_pose_step(now=None):
    """One front-camera pose re-solve on the newest front frame; updates FRONT. Returns a small status dict, or None with the
    reason in FRONT['err'] (no file, no hfov, feed not live, no table for the layout, cv2 missing)."""
    now = time.time() if now is None else now; path = front_pose_path()
    try:
        try: stored = load_calib_json(path)
        except FileNotFoundError: FRONT["err"] = NO_FRONT_POSE_ERR; return None
        if pose_hfov(stored) is None:
            pose_note(FRONT, "front camera pose has no hfov; run so101-calib campose --name front --lens-height"); FRONT["err"] = FRONT["pose_err"]; return None
        frame, ft = front_frame(now)
        if frame is None: FRONT["err"] = ft; return None
        tab = load_table()
        if tab is None: FRONT["err"] = NO_TABLE_ERR; return None               # the marker layout lives in the table file
        layout = pose_layout(tab); cc = cam_calib(); import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        seen = cc.detect_markers(gray, layout["dictionary"]); n = sum(1 for i in seen if i in layout["markers"])
    except ImportError as e: FRONT["err"] = f"cam_calib unavailable: {e}"; return None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if FRONT["err"] != msg: log("front pose:", msg)
        FRONT["err"] = msg; return None
    with _front_lock:
        FRONT.update(markers_seen=n, holding=n < 3, t=now, n=FRONT["n"] + 1, err=None)
        if n < 3: return {"markers_seen": n, "holding": True, "moved": False}  # the arm or the overhead stand hides one or two: keep the last pose
        try: est = pose_solve(gray, layout, stored)
        except Exception as e: pose_note(FRONT, f"front camera pose not re-solved: {type(e).__name__}: {e}"); return None
        FRONT["pose_err"] = None
        live = front_live(path)
        if live is None or live.get("persisted"): FRONT["pose"] = live = None   # the file holds it now (our write), or the file changed under us
        served = live["est"] if live else stored
        moved, mm, deg = pose_moved(est, served)
        if moved:
            FRONT["pose"] = live = {"est": est, "at": now, "moved_mm": round(mm, 1), "turned_deg": round(deg, 2), "markers": n, "persisted": False}
            try: FRONT["file_key"] = _table_key(path)
            except OSError: FRONT["file_key"] = None
            if now - FRONT["noted_at"] >= POSE_NOTE_S: FRONT["noted_at"] = now; event(f"front camera pose re-solved: moved {mm:.0f} mm")
        if live and not live["persisted"] and now - FRONT["persisted_at"] >= FRONT_POSE_PERSIST_S:
            try: front_persist(live, now); FRONT["persisted_at"] = now
            except Exception as e: log("front pose: write-back failed:", repr(e))
        return {"markers_seen": n, "holding": False, "moved": moved, "moved_mm": round(mm, 1), "turned_deg": round(deg, 2)}
def front_pose_loop():
    while True:
        try: front_pose_step()
        except Exception as e: log("front pose loop:", repr(e))
        time.sleep(FRONT_POSE_PERIOD_S)
# ---------- http ----------
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=HERE, **k)
    def log_message(self, *a): pass
    def query(self):
        import urllib.parse
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query, keep_blank_values=True)
    def send_json(self, obj):
        body = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        try: body_in = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}") or b"{}")
        except Exception: body_in = {}
        if self.path.startswith("/api/record/ctl"): res = record_ctl(str(body_in.get("cmd", "")))
        elif self.path.startswith("/api/record/apply"): res = record_apply(body_in)
        elif self.path.startswith("/api/record"): res = record_start(body_in)
        elif self.path.startswith("/api/lock"): set_lock(True, "button"); res = {"ok": True}
        elif self.path.startswith("/api/unlock"): set_lock(False, "override key"); res = {"ok": True}
        elif self.path.startswith("/api/arm/"): res = arm_mode(self.path.split("/")[3].split("?")[0])
        elif self.path.startswith("/api/scene/background"): res = api_scene_background(body_in if isinstance(body_in, dict) else {})
        elif self.path.startswith("/api/spotify/"): res = spotify_cmd(self.path.split("/")[3].split("?")[0])
        else: res = {"ok": False, "err": "no such endpoint"}
        body = json.dumps(res).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/api/state"):
            sim_arm = None
            try:
                _sa = json.load(open("/tmp/so101_sim_state.json")); sim_arm = _sa if time.time() - _sa.get("t", 0) < 2.5 else None
            except Exception: pass
            try:
                with lock: body = json.dumps({**state, "lock": LOCK, "sim_arm": sim_arm, "now": time.time(), "jetson": {**state["jetson"], "online": state["jetson"]["online"] and time.time() - state["jetson"]["last"] < 12}}).encode()
            except Exception as e:
                log("state json error:", repr(e)); body = json.dumps({"error": repr(e)}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/tele"):
            try: since = float(self.query().get("since", ["0"])[0])
            except ValueError: since = 0.0
            self.send_json(TELE.api(since))
        elif self.path.startswith("/api/hist"):
            with lock: body = json.dumps(HIST).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/lock"):
            with lock: body = json.dumps(LOCK).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/sim"):
            body = json.dumps(sim_status()).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/record"):
            with lock: body = json.dumps(REC).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/cams"):
            live = [n for n, (_, t) in list(cam_frames.items()) if time.time() - t < 3]
            body = json.dumps(live).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/cam/") and ".jpg" in self.path:
            f = cam_frames.get(self.path[5:].split(".")[0])
            if f and time.time() - f[1] < 3:
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(f[0]))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(f[0])
            else: self.send_response(404); self.end_headers()
        elif self.path.startswith("/cam/"):
            name = self.path[5:].split(".")[0]
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = None
            try:
                while True:
                    f = cam_frames.get(name)
                    if f and f[1] != last:
                        last = f[1]; self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f[0])).encode() + b"\r\n\r\n" + f[0] + b"\r\n")
                    elif f is None and last is not None: break
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError): pass
        elif self.path.startswith("/api/term"):
            import socket
            try: socket.create_connection(("127.0.0.1", 8766), timeout=0.3).close(); up = True
            except OSError: up = False
            body = json.dumps({"up": up, "url": "http://127.0.0.1:8766/"}).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/shell"):
            body = json.dumps(shell_calls[-10:]).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/scene.png"):
            png = scene_png()
            if png: self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(png))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(png)
            else: self.send_json({"ok": False, "err": SCENE["err"] or "no scene yet"})
        elif self.path.startswith("/api/scene"): self.send_json(api_scene())
        elif self.path.startswith("/api/policy"):
            with lock: self.send_json(state["policy"])
        elif self.path.startswith("/api/calib"): self.send_json(api_calib())
        elif self.path.startswith("/api/locate"): self.send_json(api_locate(self.query()))
        elif self.path.startswith("/api/ik"): self.send_json(api_ik(self.query()))
        elif self.path.startswith("/manifest.webmanifest"):   # PWA manifest for the iPad panel (docs/ipad-panel.md); phone.py --pad serves the same file
            try:
                with open(os.path.join(HERE, "manifest.webmanifest"), "rb") as f: body = f.read()
            except OSError: self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "application/manifest+json"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-cache"); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/api/gallery"):
            imgs = sorted(os.path.relpath(p, HERE) for p in glob.glob(os.path.join(HERE, "img", "*", "*.jpg")))
            body = json.dumps(imgs).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        else:
            if self.path == "/": self.path = "/index.html"
            super().do_GET()
class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
    def server_bind(self):  # skip HTTPServer's getfqdn() reverse-DNS lookup (35 s stall on this Mac)
        socketserver.TCPServer.server_bind(self); self.server_name = "localhost"; self.server_port = self.server_address[1]
if __name__ == "__main__":
    try:
        for l in open(os.path.join(HERE, "events.log")).read().splitlines()[-25:]:
            state["events"].insert(0, {"t": time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S")), "msg": l[21:]})
    except Exception: pass
    threading.Thread(target=say_loop, daemon=True).start(); threading.Thread(target=spotify_loop, daemon=True).start(); threading.Thread(target=record_loop, daemon=True).start(); threading.Thread(target=eye_loop, daemon=True).start(); threading.Thread(target=tegra_loop, daemon=True).start(); threading.Thread(target=probe_loop, daemon=True).start(); threading.Thread(target=arm_loop, daemon=True).start(); threading.Thread(target=shell_loop, daemon=True).start(); threading.Thread(target=scene_loop, daemon=True).start(); threading.Thread(target=front_pose_loop, daemon=True).start()
    if os.environ.get("SO101_DASH_CAMS", "1") != "0":
        for n, spec in CAMS.items(): threading.Thread(target=cam_loop, args=(n, spec), daemon=True).start()
        for n, (dev, fmt, size) in JETSON_CAMS.items(): threading.Thread(target=jetson_cam_loop, args=(n, dev, fmt, size), daemon=True).start()
    log(f"dashboard on http://127.0.0.1:{PORT}  (jetson host: {JETSON})")
    Srv(("127.0.0.1", PORT), H).serve_forever()
