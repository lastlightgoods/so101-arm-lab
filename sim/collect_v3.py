"""v3 expert (so101 lab): collect_sim.py from aviadarn/so101-lerobot plus (1) DART-style hover noise (0.012 m) with a tight grasp
waypoint, and (2) re-grasp recovery: if the cube is not lifted after "lift", re-measure it and hover/descend/close again (<= 2 retries),
so the dataset contains the recovery behaviour the 76% policy was missing (see sim/ANALYSIS_v2_10k.md).

Original docstring:

Drives ``so101_nexus`` ``MuJoCoPickAndPlace-v1`` with a waypoint FSM solved through the
env's damped-least-squares IK, and writes the episodes as a LeRobot v3 dataset whose
keys and units match a real ``so101_follower`` (body joints in degrees, gripper 0-100).

Usage:
    python collect_sim.py --dry-run --episodes 5        # tune the expert, no dataset
    python collect_sim.py --episodes 50                  # record dataset to --root
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import so101_nexus.mujoco  # noqa: F401  (registers env ids)
from so101_nexus.config import SO101_JOINT_NAMES, PickAndPlaceConfig
from so101_nexus.kinematics import quat_multiply
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row
from so101_nexus.observations import JointPositions, OverheadCamera, WristCamera

ENV_ID = "MuJoCoPickAndPlace-v1"
IMG_W, IMG_H = 640, 480  # overridden by --img-w/--img-h
TASK = "Pick up the red cube and place it on the blue circle."

# Top-down tool orientation at shoulder_pan = 0: TCP local z points to -world z
# (measured empirically; see README). Rotated about world z by the pan angle.
DOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])

GRIPPER_OPEN_RAD = 1.5
GRIPPER_CLOSED_RAD = -0.17

# Expert tunables (metres). Tune with --dry-run.
HOVER_Z = 0.08          # height above the table for transport
GRASP_Z_OFFSET = 0.0    # TCP z relative to cube centre when grasping
PLACE_Z = 0.035         # TCP z when releasing over the disc
MAX_STEP_M = 0.006      # max TCP travel per control step (0.02 s) -> 0.3 m/s
GRIP_STEPS = 25         # control steps to wait for the jaw to close/open
SETTLE_STEPS = 15       # steps to hold still after release
STAGE_TIMEOUT = 60      # give up waiting for "reached" and advance after this many steps
IK_ORIENTATION_WEIGHT = 0.05
WAYPOINT_NOISE_M = 0.012   # hover waypoints (DART-style); grasp waypoints use GRASP_NOISE_M
GRASP_NOISE_M = 0.002
MAX_REGRASP = 2
MISS_PROB = 0.3            # fraction of episodes whose FIRST descent is deliberately 1-3 cm off, so the demo shows a recovery
MISS_OFFSET_M = (0.012, 0.03)
LIFT_OK_M = 0.02


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def top_down_quat(xy: np.ndarray) -> np.ndarray:
    """Tool-down orientation whose jaw axis is aligned with the arm's pan direction."""
    yaw = float(np.arctan2(xy[1], xy[0]))
    return quat_multiply(yaw_quat(yaw), DOWN_QUAT)


class Expert:
    """Waypoint FSM producing absolute joint targets (radians) each control step."""

    def __init__(self, env, rng: np.random.Generator):
        self.u = env.unwrapped
        self.rng = rng
        self.gripper = GRIPPER_OPEN_RAD
        self.plan: list[tuple[str, np.ndarray | None, float, int]] = []
        self.tcp_goal: np.ndarray | None = None

    def _noise(self) -> np.ndarray:
        return np.concatenate([self.rng.normal(0, WAYPOINT_NOISE_M, 2), [0.0]])

    def build_plan(self) -> None:
        obj = self.u._get_object_pose()[:3]
        tgt = self.u._get_target_pos()
        above_obj = obj + [0, 0, HOVER_Z] + self._noise()
        at_obj = obj + [0, 0, GRASP_Z_OFFSET] + self._noise() * (GRASP_NOISE_M / WAYPOINT_NOISE_M)
        self.deliberate_miss = bool(self.rng.random() < MISS_PROB)
        if self.deliberate_miss:
            ang = self.rng.uniform(0, 2 * np.pi); r = self.rng.uniform(*MISS_OFFSET_M)
            at_obj = at_obj + np.array([np.cos(ang) * r, np.sin(ang) * r, 0.0])
        above_tgt = np.array([tgt[0], tgt[1], HOVER_Z]) + self._noise()
        at_tgt = np.array([tgt[0], tgt[1], PLACE_Z]) + self._noise() * (GRASP_NOISE_M / WAYPOINT_NOISE_M)
        # (name, tcp target, gripper target, hold steps once reached)
        self.plan = [
            ("hover_obj", above_obj, GRIPPER_OPEN_RAD, 0),
            ("descend", at_obj, GRIPPER_OPEN_RAD, 0),
            ("close", None, GRIPPER_CLOSED_RAD, GRIP_STEPS),
            ("lift", above_obj, GRIPPER_CLOSED_RAD, 0),
            ("carry", above_tgt, GRIPPER_CLOSED_RAD, 0),
            ("lower", at_tgt, GRIPPER_CLOSED_RAD, 0),
            ("open", None, GRIPPER_OPEN_RAD, GRIP_STEPS),
            ("retreat", above_tgt, GRIPPER_OPEN_RAD, SETTLE_STEPS),
        ]
        self.stage = 0
        self.hold = 0
        self.stage_steps = 0
        self.tcp_goal = self.u._get_tcp_pose()[:3].copy()
        self.obj_z0 = float(obj[2])
        self.regrasps = 0

    @property
    def stage_name(self) -> str:
        return self.plan[self.stage][0] if self.stage < len(self.plan) else "done"

    def done(self) -> bool:
        return self.stage >= len(self.plan)

    def act(self) -> np.ndarray:
        """Return the 6-vector joint target (rad) for this control step."""
        name, target, grip, hold_steps = self.plan[self.stage]
        self.gripper = grip
        cur_tcp = self.u._get_tcp_pose()[:3]
        if target is None:
            reached = True
        else:
            delta = target - self.tcp_goal
            dist = float(np.linalg.norm(delta))
            if dist > MAX_STEP_M:
                self.tcp_goal = self.tcp_goal + delta / dist * MAX_STEP_M
            else:
                self.tcp_goal = target.copy()
            reached = float(np.linalg.norm(cur_tcp - target)) < 0.008
        self.stage_steps += 1
        if reached or self.stage_steps > STAGE_TIMEOUT:
            self.hold += 1
            if self.hold > hold_steps:
                if name.startswith("lift") and self.u._get_object_pose()[2] < self.obj_z0 + LIFT_OK_M and self.regrasps < MAX_REGRASP:
                    # Missed (or dropped): re-measure the cube and try again from a fresh hover. This is the recovery
                    # behaviour the policy must learn; the original expert never missed.
                    self.regrasps += 1
                    obj = self.u._get_object_pose()[:3]
                    above = obj + [0, 0, HOVER_Z] + self._noise() * 0.5
                    at = obj + [0, 0, GRASP_Z_OFFSET] + self._noise() * (GRASP_NOISE_M / WAYPOINT_NOISE_M)
                    retry = [("open_r", None, GRIPPER_OPEN_RAD, GRIP_STEPS // 2), ("hover_r", above, GRIPPER_OPEN_RAD, 0),
                             ("descend_r", at, GRIPPER_OPEN_RAD, 0), ("close_r", None, GRIPPER_CLOSED_RAD, GRIP_STEPS), ("lift_r", above, GRIPPER_CLOSED_RAD, 0)]
                    self.plan[self.stage + 1:self.stage + 1] = retry
                if name.startswith("lift"):
                    # The cube rarely sits exactly at the TCP after closing; shift the
                    # place waypoints by the measured grasp offset so the cube, not the
                    # tool, lands on the disc.
                    off = self.u._get_object_pose()[:2] - self.u._get_tcp_pose()[:2]
                    if np.linalg.norm(off) < 0.04:
                        for i in range(self.stage + 1, len(self.plan)):
                            n_, tgt_, g_, h_ = self.plan[i]
                            if tgt_ is not None:
                                tgt_ = tgt_.copy()
                                tgt_[:2] -= off
                                self.plan[i] = (n_, tgt_, g_, h_)
                self.stage += 1
                self.hold = 0
                self.stage_steps = 0
        quat = top_down_quat(self.tcp_goal[:2])
        return self.u._solve_ee_ik(self.tcp_goal, quat, self.gripper)


def make_env(seed: int, spawn_min: float = 0.10, spawn_max: float = 0.30, img_w: int = IMG_W, img_h: int = IMG_H,
             random_wrist_cam: bool = False, cube_mm: float = 25.0, goal_thresh: float = 0.025,
             overhead_margin: float = 0.10):
    # so101-nexus randomizes the wrist camera FOV (60-90 deg), pitch (-34..0 deg) and
    # position (+-1 cm) every episode. A real wrist camera is rigidly mounted, and that
    # jitter destroys the fine-alignment cue, so the default here is a fixed camera at
    # the centre of those ranges.
    if random_wrist_cam:
        wrist = WristCamera(width=img_w, height=img_h)
    else:
        wrist = WristCamera(width=img_w, height=img_h, fov_deg_range=(75.0, 75.0),
                            pitch_deg_range=(-17.2, -17.2), pos_x_noise=0.0, pos_y_noise=0.0, pos_z_noise=0.0)
    cfg = PickAndPlaceConfig(
        cube_side_length_mm=cube_mm,
        goal_thresh=goal_thresh,
        spawn_min_radius=spawn_min,
        spawn_max_radius=spawn_max,
        obs_mode="visual",
        observations=[
            JointPositions(),
            wrist,
            OverheadCamera(width=img_w, height=img_h),
        ],
        terminate_on_success=False,
    )
    env = gym.make(ENV_ID, config=cfg, render_mode="rgb_array", control_mode="pd_joint_pos")
    env.reset(seed=seed)
    # Package default is 0.01 (orientation almost ignored). 0.05 keeps the tool
    # vertical (down_cos 1.0) and converged fastest in a sweep over failing seeds.
    env.unwrapped.config.robot.ee_orientation_weight = IK_ORIENTATION_WEIGHT
    # Tighter overhead framing: so101-nexus pads the spawn box by a fixed 0.10 m; a smaller
    # margin gives more pixels per centimetre on the workspace (a real overhead webcam is
    # simply mounted closer).
    if overhead_margin != 0.10:
        from so101_nexus.camera_utils import compute_overhead_camera_params
        from so101_nexus.mujoco.base_env import _configure_free_camera

        u = env.unwrapped
        cam = u._overhead_cam_component
        params = compute_overhead_camera_params(
            spawn_center=u.config.spawn_center, spawn_max_radius=u.config.spawn_max_radius,
            margin=overhead_margin, fov_deg=cam.fov_deg, aspect=cam.width / cam.height,
        )
        _configure_free_camera(u._overhead_obs_cam, params)
    # _solve_ee_ik needs a scratch MjData; the env only allocates it for EE control modes.
    if env.unwrapped._ik_data is None:
        import mujoco

        env.unwrapped._ik_data = mujoco.MjData(env.unwrapped.model)
    return env


def run_episode(env, seed: int, rng, max_steps: int, on_frame=None, verbose=False) -> tuple[bool, int, str]:
    obs, info = env.reset(seed=seed)
    u = env.unwrapped
    expert = Expert(env, rng)
    expert.build_plan()
    success = False
    timeline: list[tuple[str, int]] = []
    last_stage = expert.stage_name
    grasped_ever = False
    for t in range(max_steps):
        action = expert.act()
        if on_frame is not None:
            on_frame(obs, action, info)
        obs, _r, _term, _trunc, info = env.step(action)
        success = bool(info.get("success", False))
        grasped_ever |= bool(info.get("is_grasped", 0.0))
        if expert.stage_name != last_stage:
            timeline.append((last_stage, expert.stage_steps if False else t))
            last_stage = expert.stage_name
        if expert.done():
            break
    if verbose:
        obj = u._get_object_pose()[:3]
        tgt = u._get_target_pos()
        print(
            f"    obj r={np.linalg.norm(obj[:2]):.3f} tgt r={np.linalg.norm(tgt[:2]):.3f} "
            f"grasped_ever={grasped_ever} final: dist={info['obj_to_target_dist']:.3f} "
            f"placed={info['is_obj_placed']} grasped={info['is_grasped']} static={info['is_obj_static']} "
            f"lift={info['lift_height']:.3f} obj_z={obj[2]:.3f} regrasps={expert.regrasps} miss={expert.deliberate_miss}\n    stages(end step): {timeline}"
        )
    return success, t + 1, expert.stage_name


def dataset_features(img_w: int = IMG_W, img_h: int = IMG_H) -> dict:
    names = [f"{j}.pos" for j in SO101_JOINT_NAMES]
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.wrist": {
            "dtype": "video", "shape": (img_h, img_w, 3), "names": ["height", "width", "channels"],
        },
        "observation.images.overhead": {
            "dtype": "video", "shape": (img_h, img_w, 3), "names": ["height", "width", "channels"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true", help="run the expert, write nothing")
    ap.add_argument("--repo-id", default="tclastlight/so101_sim_pick_v3")
    ap.add_argument("--root", default="data/so101_sim_pick_place")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gif", default="", help="write overhead GIF of the first episode here")
    ap.add_argument("--spawn-min", type=float, default=0.10)
    ap.add_argument("--spawn-max", type=float, default=0.30)
    ap.add_argument("--verbose", action="store_true", help="print per-episode diagnostics")
    ap.add_argument("--img-w", type=int, default=IMG_W)
    ap.add_argument("--img-h", type=int, default=IMG_H)
    ap.add_argument("--random-wrist-cam", action="store_true", help="restore so101-nexus wrist-camera randomization")
    ap.add_argument("--cube-mm", type=float, default=25.0, help="cube side length in mm")
    ap.add_argument("--goal-thresh", type=float, default=0.025, help="success radius around disc centre (m)")
    ap.add_argument("--overhead-margin", type=float, default=0.10, help="overhead camera padding around the spawn box (m)")
    ap.add_argument("--seeds", default="", help="comma-separated explicit episode seeds")
    args = ap.parse_args()

    env = make_env(args.seed, args.spawn_min, args.spawn_max, args.img_w, args.img_h, args.random_wrist_cam,
                   args.cube_mm, args.goal_thresh, args.overhead_margin)
    fps = int(round(1.0 / env.unwrapped.control_dt))
    rng = np.random.default_rng(args.seed)
    grip_lim = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))

    dataset = None
    if not args.dry_run:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(args.root)
        if root.exists():
            if not args.overwrite:
                print(f"{root} exists; pass --overwrite to replace it", file=sys.stderr)
                return 2
            shutil.rmtree(root)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=fps, features=dataset_features(args.img_w, args.img_h), root=root,
            robot_type="so101_follower", use_videos=True, image_writer_threads=4,
        )

    gif_frames: list[np.ndarray] = []
    n_ok = 0
    t0 = time.time()
    ep = 0
    attempt = 0
    explicit_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if explicit_seeds:
        args.episodes = len(explicit_seeds)
    while ep < args.episodes:
        if explicit_seeds:
            if attempt >= len(explicit_seeds):
                break
            seed = explicit_seeds[attempt]
        else:
            seed = args.seed * 100_000 + attempt
        attempt += 1

        def on_frame(obs, action, info, _first=(ep == 0)):
            if dataset is not None:
                dataset.add_frame({
                    "observation.state": sim_qpos_to_dataset_row(
                        obs["state"].astype(np.float64), gripper_limits_rad=grip_lim
                    ).astype(np.float32),
                    "action": sim_qpos_to_dataset_row(
                        np.asarray(action, dtype=np.float64), gripper_limits_rad=grip_lim
                    ).astype(np.float32),
                    "observation.images.wrist": obs["wrist_camera"],
                    "observation.images.overhead": obs["overhead_camera"],
                    "task": TASK,
                })
            if _first and args.gif and len(gif_frames) < 2000:
                gif_frames.append(obs["overhead_camera"][::2, ::2])

        ok, steps, stage = run_episode(env, seed, rng, args.max_steps, on_frame, args.verbose)
        status = "OK " if ok else "FAIL"
        print(f"[{status}] attempt {attempt:3d} seed {seed} steps {steps:4d} stage {stage}")
        if ok:
            n_ok += 1
            ep += 1
            if dataset is not None:
                # Serial encoding: the parallel encoder process pool gets killed under
                # macOS memory pressure and takes the whole recording down with it.
                dataset.save_episode(parallel_encoding=False)
        else:
            if dataset is not None:
                dataset.clear_episode_buffer()
            if ep == 0:
                gif_frames.clear()
            if explicit_seeds:
                ep += 1  # explicit seeds are a fixed list; don't retry
        if attempt >= args.episodes * 3:
            print("too many failures; stopping", file=sys.stderr)
            break

    print(f"success {n_ok}/{attempt} attempts, {ep} episodes kept, {time.time() - t0:.0f}s")
    if dataset is not None:
        dataset.finalize()
        print(f"dataset written to {args.root} ({dataset.num_episodes} episodes, fps={fps})")
    if args.gif and gif_frames:
        from PIL import Image

        ims = [Image.fromarray(f) for f in gif_frames[::3]]
        ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=60, loop=0)
        print(f"gif -> {args.gif}")
    env.close()
    return 0 if n_ok == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
