"""Scripted demonstration recorder: the follower performs pick -> carry -> hand-over from taught poses while cameras +
joint states are written as a LeRobot v3 dataset (same format lerobot-record produces, trains with lerobot-train).
Poses come from ~/so101/poses.json (so101-mark): 'rest', 'hand' and any number of 'spot*' grasp poses.
    python scripted_record.py <dataset-name> "<task>" <episodes> [--dry-run] [--fps 30]"""
import argparse, json, math, os, random, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("task"); ap.add_argument("episodes", type=int)
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--no-wait", action="store_true")
a = ap.parse_args()
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POSES = json.load(open(os.path.expanduser("~/so101/poses.json")))
spots = sorted(k for k in POSES if k.startswith("spot"))
for req in ("rest", "hand"):
    if req not in POSES: sys.exit(f"pose '{req}' missing: teleop there and run  so101-mark {req}")
if not spots: sys.exit("no spot* poses: teleop the gripper onto the lighter and run  so101-mark spot1  (then spot2 ...)")
OPEN, CLOSED = 75.0, 4.0
def with_grip(p, g): q = dict(p); q["gripper"] = g; return q
def lifted(p, d=22.0): q = dict(p); q["shoulder_lift"] = max(-100, q["shoulder_lift"] - d); q["elbow_flex"] = min(100, q["elbow_flex"] + d * 0.6); return q
def jitter(p, deg=2.0): return {k: (v + random.uniform(-deg, deg) if k != "gripper" else v) for k, v in p.items()}
def plan(spot):
    g = POSES[spot]; above = jitter(lifted(g)); hand = jitter(POSES["hand"], 3.0); rest = POSES["rest"]; s = random.uniform(0.85, 1.2)
    return [("approach", with_grip(above, OPEN), 2.2 * s), ("descend", with_grip(g, OPEN), 1.4 * s), ("grasp", with_grip(g, CLOSED), 0.7),
            ("lift", with_grip(above, CLOSED), 1.2 * s), ("carry", with_grip(hand, CLOSED), 2.4 * s), ("offer", with_grip(hand, CLOSED), 1.2),
            ("release", with_grip(hand, OPEN), 0.8), ("wait", with_grip(hand, OPEN), 0.8), ("home", with_grip(rest, 10.0), 2.4 * s)]
if a.dry_run:
    for i in range(a.episodes):
        sp = spots[i % len(spots)]; print(f"episode {i}: {sp}  " + "  ".join(f"{n}:{d:.1f}s" for n, _, d in plan(sp)))
    print("total per episode ~", round(sum(d for _, _, d in plan(spots[0])), 1), "s"); sys.exit()
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig
from lerobot.processor import make_default_processors
from lerobot.processor.factory import create_initial_features  # noqa
from lerobot.datasets.utils import aggregate_pipeline_dataset_features
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import ACTION, OBS_STR
cams = {"top": OpenCVCameraConfig(index_or_path=os.environ["CAM_TOP"], width=640, height=360, fps=30),
        "wrist": OpenCVCameraConfig(index_or_path=os.environ["CAM_WRIST"], width=640, height=480, fps=30)}
robot = SOFollower(SOFollowerConfig(port=os.environ.get("FOLLOWER", "/dev/so101_follower"), id=os.environ.get("FOLLOWER_ID", "ll_follower"), cameras=cams))
_t, _r, _o = make_default_processors()
features = combine_feature_dicts(
    aggregate_pipeline_dataset_features(pipeline=_t, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
    aggregate_pipeline_dataset_features(pipeline=_o, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True))
root = os.path.expanduser(f"~/.cache/huggingface/lerobot/lastlight/{a.name}")
if os.path.exists(root): ds = LeRobotDataset.resume(f"lastlight/{a.name}", root=root, rgb_encoder=RGBEncoderConfig(vcodec="h264"), encoder_threads=2, streaming_encoding=True, image_writer_processes=0, image_writer_threads=4)
else: ds = LeRobotDataset.create(f"lastlight/{a.name}", a.fps, root=root, robot_type=robot.name, features=features, use_videos=True, image_writer_processes=0, image_writer_threads=4, rgb_encoder=RGBEncoderConfig(vcodec="h264"), encoder_threads=2, streaming_encoding=True)
robot.connect(); print("robot + cameras connected; dataset at", root, flush=True)
def observe(): return robot.get_observation()
def ease(x): return 0.5 - 0.5 * math.cos(math.pi * x)
try:
    obs = observe(); cur = {j: float(obs[f"{j}.pos"]) for j in J}
    for ep in range(a.episodes):
        spot = spots[ep % len(spots)]
        print(f"\n=== EPISODE {ep + 1}/{a.episodes}: place the lighter at {spot.upper()} and press Enter (q + Enter = stop) ===", flush=True)
        if not a.no_wait and input().strip().lower() == "q": break
        steps = plan(spot); n_frames = 0; t_ep = time.perf_counter()
        for name, target, dur in steps:
            start = dict(cur); n = max(1, int(dur * a.fps))
            for i in range(1, n + 1):
                t0 = time.perf_counter(); s = ease(i / n)
                for j in J: cur[j] = start[j] + (target[j] - start[j]) * s
                obs = observe(); obs_p = _o(obs)
                act = {f"{j}.pos": cur[j] for j in J}; robot.send_action(act)
                frame = {**build_dataset_frame(ds.features, obs_p, prefix=OBS_STR), **build_dataset_frame(ds.features, act, prefix=ACTION), "task": a.task}
                ds.add_frame(frame); n_frames += 1
                time.sleep(max(0, 1 / a.fps - (time.perf_counter() - t0)))
            print(f"  {name:9s} ok", flush=True)
        ds.save_episode(); print(f"episode {ep + 1} saved: {n_frames} frames, {time.perf_counter() - t_ep:.1f}s", flush=True)
finally:
    ds.finalize(); robot.disconnect(); print("dataset finalized:", root, "episodes:", ds.num_episodes, flush=True)
