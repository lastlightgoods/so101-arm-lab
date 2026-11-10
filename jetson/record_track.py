#!/usr/bin/env python3
"""record_track.py - one MLflow run per so101-record session (experiment `so101-recordings`, docs/mlops/tracking.md).

env.sh calls it around lerobot-record (so101-record / so101-record-more):
    python3 ~/so101/record_track.py start --repo-id lastlight/<name> --task "<task>" --episodes N [--resume] [--root DIR]
    lerobot-record ...
    python3 ~/so101/record_track.py finish [--rc <lerobot-record exit code>]

start  logs the dataset repo id, task, episodes requested, fps and camera list (from $SO101_CAMS), every locked control from
       ~/so101/calib/cam_settings.json as cam_<role>_<control>, the live `cam_lock.py check --json` as an artifact, and the
       last cam_health JSON (~/so101/outputs/run/cam_health.json, written by so101-cams-health) when it is under 10 minutes old.
       The run id is kept in ~/so101/outputs/run/record_track.json for `finish`.
finish logs episodes_total (the dataset's meta/info.json), episodes_new, duration_s and the exit code, then ends the run
       (FINISHED on rc 0, FAILED otherwise).
Both exit 0 whatever happens (no server, no mlflow, no file): a recording must never be blocked by its bookkeeping.
"""
import argparse, glob, json, os, re, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SO101 = os.path.expanduser("~/so101")
STATE = os.path.join(SO101, "outputs", "run", "record_track.json")
HEALTH = os.path.join(SO101, "outputs", "run", "cam_health.json")
SETTINGS = os.path.join(SO101, "calib", "cam_settings.json")
CAM_LOCK = os.path.join(SO101, "cam_lock.py")
HEALTH_MAX_AGE_S = 600
EXPERIMENT = "so101-recordings"


def log(*a): print("record_track:", *a, file=sys.stderr, flush=True)


def labtrack():
    import importlib.util
    spec = importlib.util.spec_from_file_location("labtrack", os.path.join(HERE, "labtrack.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.LabTrack(EXPERIMENT)


def read_json(path):
    try: return json.load(open(path))
    except (OSError, ValueError): return None


def cams_from_env(spec=None):
    """Camera names and fps out of $SO101_CAMS ("{top: {type: opencv, ..., fps: 30}, ...}") without a YAML parser."""
    spec = os.environ.get("SO101_CAMS", "") if spec is None else spec
    names = re.findall(r"(\w+):\s*\{", spec); fps = re.findall(r"fps:\s*(\d+)", spec)
    return names, (int(fps[0]) if fps else None)


def dataset_root(repo_id, root=None): return root or os.path.join(os.path.expanduser("~/.cache/huggingface/lerobot"), repo_id)


def episodes_in(root):
    info = read_json(os.path.join(root, "meta", "info.json")) or {}
    return info.get("total_episodes"), info


def cmd_start(a):
    track = labtrack(); root = dataset_root(a.repo_id, a.root); before, info = episodes_in(root); names, fps = cams_from_env()
    name = a.repo_id.split("/")[-1]
    run = track.start(f"record-{name}-{time.strftime('%Y%m%d-%H%M%S')}", tags={"kind": "record", "dataset": name, "resume": int(bool(a.resume)), "host": socket.gethostname().split(".")[0]})
    params = {"repo_id": a.repo_id, "task": a.task, "episodes_requested": a.episodes, "resume": bool(a.resume), "root": root, "cameras": ",".join(names) or None, "fps": info.get("fps") or fps,
              "episode_time_s": os.environ.get("SO101_EP_S", "25"), "reset_time_s": os.environ.get("SO101_RESET_S", "10"), "episodes_before": before or 0}
    settings = read_json(SETTINGS)
    if settings:
        params["cam_settings_captured"] = settings.get("captured")
        for by_id, block in (settings.get("cameras") or {}).items():
            role = block.get("role") or by_id
            for ctrl, val in (block.get("controls") or {}).items(): params[f"cam_{role}_{ctrl}"] = val
    run.params(params)
    if run.enabled:   # why: the check and the file reads cost nothing when nothing is listening; skip them then
        if settings: run.artifact(SETTINGS, "cameras")
        if os.path.exists(CAM_LOCK):
            try:
                chk = subprocess.run([sys.executable, CAM_LOCK, "check", "--json"], capture_output=True, text=True, timeout=20)
                run.file("cam_check.json", chk.stdout or chk.stderr, "cameras"); run.metrics({"cams_locked": int(chk.returncode == 0)})
            except (OSError, subprocess.SubprocessError) as e: log("cam_lock check skipped:", e)
        for h in sorted(glob.glob(HEALTH.replace(".json", "*.json")), key=os.path.getmtime)[-1:]:
            age = time.time() - os.path.getmtime(h)
            if age <= HEALTH_MAX_AGE_S:
                run.artifact(h, "cameras"); rep = read_json(h) or {}
                run.metrics({"cam_health_ok": int(bool(rep.get("ok"))), **{f"cam_health_fps_{n}": (c.get("stats") or {}).get("fps") for n, c in (rep.get("cameras") or {}).items()}, "cam_health_age_s": age})
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump({"run_id": run.id, "uri": track.uri, "t0": time.time(), "repo_id": a.repo_id, "root": root, "episodes_before": before or 0, "requested": a.episodes}, open(STATE, "w"))
    print(json.dumps({"tracking": run.enabled, "run_id": run.id, "experiment": EXPERIMENT, "why": track.why}))


def cmd_finish(a):
    st = read_json(STATE)
    try: os.remove(STATE)
    except OSError: pass
    if not st or not st.get("run_id"): print(json.dumps({"tracking": False, "why": "no run was started"})); return
    track = labtrack(); run = track.attach(st["run_id"])
    total, _ = episodes_in(st["root"]); dur = time.time() - st["t0"]
    run.metrics({"episodes_total": total, "episodes_new": (total - st["episodes_before"]) if total is not None else None, "duration_s": dur, "rc": a.rc, "ok": int(a.rc == 0)})
    info = os.path.join(st["root"], "meta", "info.json")
    if os.path.exists(info): run.artifact(info, "dataset")
    run.end("FINISHED" if a.rc == 0 else "FAILED")
    print(json.dumps({"tracking": track.enabled, "run_id": st["run_id"], "episodes_total": total, "duration_s": round(dur, 1), "rc": a.rc}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("--repo-id", required=True); s.add_argument("--task", default=""); s.add_argument("--episodes", type=int, default=None)
    s.add_argument("--resume", action="store_true"); s.add_argument("--root"); s.set_defaults(fn=cmd_start)
    s = sub.add_parser("finish"); s.add_argument("--rc", type=int, default=0); s.set_defaults(fn=cmd_finish)
    a = ap.parse_args(argv)
    try: a.fn(a)
    except Exception as e: log(f"{a.cmd} failed: {type(e).__name__}: {e}")   # why: never let bookkeeping fail a recording
    return 0


if __name__ == "__main__": sys.exit(main())
