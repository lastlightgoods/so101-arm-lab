#!/usr/bin/env python3
"""film.py · multi-camera films of the SO-101 at work, with the command center split-screen.

  python tools/film.py live <seconds> [out.mp4] [--no-screen] [--audio file.wav]
      records all Jetson cameras at full quality (C920s 1280x720 MJPG, wrist 640x480) AND the kiosk screen on the Mac for
      <seconds>, then composes them. The dashboard's own camera streams pause meanwhile (.cams_busy).
  python tools/film.py dataset <name> [episode|all] [out.mp4] [--audio file.wav]
      composes the camera videos a recording or policy run already saved (~/.cache/huggingface/lerobot/lastlight/<name>);
      no screen capture (it was not recorded then).

Layout 1920x1080: command center (portrait kiosk) left 640 px; overhead 1280x720 top-right; front + wrist 640x360 each below.
Missing views become dark panels. Output: H.264, 30 fps, 1920x1080, in ~/so101/films/.
"""
import os, sys, json, glob, subprocess, time
HOME = os.path.expanduser("~"); FILMS = f"{HOME}/so101/films"; os.makedirs(FILMS, exist_ok=True)
JETSON = os.environ.get("SO101_JETSON", "jetson")
CAMS = {  # name: (jetson device by serial, ffmpeg input format, size, how to store on the Jetson)
    "top":   ("/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0", "mjpeg", "1280x720", "copy"),
    "front": ("/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0", "mjpeg", "1280x720", "copy"),
    "wrist": ("/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0", "yuyv422", "640x480", "x264"),
}
SCREEN_DEV = os.environ.get("SO101_SCREEN_DEV", "4")   # avfoundation "Capture screen 1" = the vertical kiosk monitor
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
def sh(cmd, **k): return subprocess.run(cmd, shell=True, text=True, capture_output=True, **k)
def label_png(txt, size=34):
    """Label as a transparent PNG (Homebrew ffmpeg has no drawtext filter)."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(FONT, size); im = Image.new("RGBA", (int(len(txt) * size * 0.72) + 40, size + 24), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=8, fill=(7, 9, 13, 150)); d.text((16, 10), txt, font=f, fill=(255, 255, 255, 255))
    path = f"/tmp/so101_label_{abs(hash(txt))}.png"; im.save(path); return path
def compose(inputs, out, audio=None):
    """inputs: dict view -> file path (or None). Views: screen, top, front, wrist."""
    args = []; filters = []; idx = {}; n_in = 0
    def add(path):
        nonlocal n_in; args.extend(["-i", path]); n_in += 1; return n_in - 1
    for v in ("screen", "top", "front", "wrist"):
        p = inputs.get(v)
        if p: idx[v] = add(p)
    def src(v, w, h, text):
        li = add(label_png(text if v in idx else text + " · no signal"))
        if v in idx:
            fit = f"scale=w={w:d}:h={h:d}:force_original_aspect_ratio=decrease,pad={w:d}:{h:d}:(ow-iw)/2:(oh-ih)/2:color=0x07090d"
            return f"[{idx[v]}:v]fps=30,{fit}[{v}_raw];[{v}_raw][{li}:v]overlay=24:24[{v}]"
        return f"color=c=0x0f131a:s={w}x{h}:r=30[{v}_raw];[{v}_raw][{li}:v]overlay=24:24[{v}]"
    filters += [src("screen", 640, 1080, "COMMAND CENTER"), src("top", 1280, 720, "OVERHEAD"), src("front", 640, 360, "FRONT"), src("wrist", 640, 360, "WRIST")]
    filters.append("[front][wrist]hstack=inputs=2:shortest=1[bottom];[top][bottom]vstack=inputs=2:shortest=1[right];[screen][right]hstack=inputs=2:shortest=1,format=yuv420p[v]")   # shortest: the dark placeholder sources are endless
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    if audio: cmd += ["-stream_loop", "-1", "-i", audio]
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]"]
    if audio: cmd += ["-map", f"{n_in}:a", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", "30", out]
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode: raise SystemExit("compose failed:\n" + r.stderr[-1500:])
    return out
def live(secs, out, screen=True, audio=None):
    tag = time.strftime("%Y%m%d_%H%M%S"); rdir = f"/tmp/so101_film_{tag}"; ldir = f"/tmp/so101_film_{tag}"; os.makedirs(ldir, exist_ok=True)
    jcmds = []
    for n, (dev, fmt, size, store) in CAMS.items():
        enc = "-c:v copy" if store == "copy" else "-c:v libx264 -preset ultrafast -crf 23"
        ext = "mkv" if store == "copy" else "mp4"
        jcmds.append(f"timeout -k 2 {secs + 8} ffmpeg -loglevel error -y -f v4l2 -input_format {fmt} -video_size {size} -framerate 30 -t {secs + 0.5} -i {dev} {enc} {rdir}/{n}.{ext} > {rdir}/{n}.log 2>&1 &")   # hard timeout: a camera the dashboard has not released yet must never hang the run
    remote = f"mkdir -p {rdir}; touch ~/so101/.cams_busy; for i in 1 2 3 4 5 6; do pgrep -f 'ffmpeg .* -f v4l2' >/dev/null || break; sleep 1; done; "   # wait until the dashboard's own ffmpeg streams have stopped + " ".join(jcmds) + " wait; rm -f ~/so101/.cams_busy; ls -la {rdir}"
    print(f"recording {secs}s on the Jetson ({', '.join(CAMS)}){' + kiosk screen' if screen else ''} ...", flush=True)
    scr = None
    if screen:
        scr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "avfoundation", "-framerate", "30", "-capture_cursor", "0", "-i", f"{SCREEN_DEV}:none", "-t", str(secs + 2.5), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", f"{ldir}/screen.mp4"], text=True, stderr=subprocess.PIPE)
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", JETSON, remote], text=True, capture_output=True)
    print(r.stdout.strip().splitlines()[-4:] if r.stdout else r.stderr[-300:], flush=True)
    files = {}
    for n, (_, _, _, store) in CAMS.items():
        ext = "mkv" if store == "copy" else "mp4"
        if subprocess.run(["scp", "-q", f"{JETSON}:{rdir}/{n}.{ext}", f"{ldir}/{n}.{ext}"]).returncode == 0 and os.path.getsize(f"{ldir}/{n}.{ext}") > 10000: files[n] = f"{ldir}/{n}.{ext}"
        else: print(f"  {n}: no recording (check {rdir}/{n}.log on the Jetson)")
    if scr:
        scr.wait(); err = scr.stderr.read() if scr.stderr else ""
        if os.path.exists(f"{ldir}/screen.mp4") and os.path.getsize(f"{ldir}/screen.mp4") > 10000: files["screen"] = f"{ldir}/screen.mp4"
        else: print("  screen: not captured. macOS needs Screen Recording permission for this terminal app (System Settings > Privacy & Security > Screen Recording).", err[-200:])
    subprocess.run(["ssh", JETSON, f"rm -rf {rdir}"])
    return compose(files, out, audio)
def dataset(name, episode, out, audio=None):
    root = f"{HOME}/.cache/huggingface/lerobot/lastlight/{name}"; info = json.load(open(f"{root}/meta/info.json"))
    files = {}
    for key in info["features"]:
        if not key.startswith("observation.images."): continue
        view = key.split(".")[-1]; view = "top" if view in ("top", "overhead") else view
        vids = sorted(glob.glob(f"{root}/videos/{key}/chunk-*/file-*.mp4"))
        if not vids: continue
        if episode == "all": files[view] = vids[0] if len(vids) == 1 else vids[0]
        else:
            import pandas as pd
            ep = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet"))]).iloc[int(episode)]
            ci, fi = int(ep[f"videos/{key}/chunk_index"]), int(ep[f"videos/{key}/file_index"]); t0, t1 = float(ep[f"videos/{key}/from_timestamp"]), float(ep[f"videos/{key}/to_timestamp"])
            cut = f"/tmp/so101_film_{name}_{episode}_{view}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t0), "-to", str(t1), "-i", f"{root}/videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4", "-c", "copy", cut], check=True); files[view] = cut
    if not files: raise SystemExit("no camera videos found in " + root)
    return compose(files, out, audio)
if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("live", "dataset"): print(__doc__); raise SystemExit(1)
    audio = None
    if "--audio" in a: i = a.index("--audio"); audio = a[i + 1]; del a[i:i + 2]
    screen = "--no-screen" not in a; a = [x for x in a if x != "--no-screen"]
    if a[0] == "live":
        secs = float(a[1]); out = a[2] if len(a) > 2 else f"{FILMS}/live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        print("film ->", live(secs, out, screen, audio))
    else:
        name = a[1]; ep = a[2] if len(a) > 2 else "all"; out = a[3] if len(a) > 3 else f"{FILMS}/{name}_ep{ep}.mp4"
        print("film ->", dataset(name, ep, out, audio))
