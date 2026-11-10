"""Rebuild a LeRobot v3 dataset with bad episodes dropped and idle tails cut.
    python tools/clean_dataset.py <src-folder> <dst-folder> "<task>" [--drop 0,4,7] [--tail 1.0] [--no-trim]
Tail cut = keep up to <tail> s after the arm's last motion (joints 0-4 moving > 0.4 deg/frame).
Head cut = keep only <head> s (default 0.5) before the leader first moves > 3 deg from its start (ACT freezes on idle starts)."""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("task")
ap.add_argument("--drop", default=""); ap.add_argument("--tail", type=float, default=1.0); ap.add_argument("--head", type=float, default=0.5); ap.add_argument("--no-trim", action="store_true"); ap.add_argument("--vcodec", default="h264")
a = ap.parse_args()
def main():
    import numpy as np, torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    cache = os.path.expanduser("~/.cache/huggingface/lerobot/lastlight")
    src = LeRobotDataset(f"lastlight/{a.src}", root=f"{cache}/{a.src}"); drop = {int(x) for x in a.drop.split(",") if x.strip()}
    dst_root = f"{cache}/{a.dst}"; assert not os.path.exists(dst_root), f"{dst_root} exists"
    META = ("timestamp", "frame_index", "episode_index", "index", "task_index")
    feats = {k: v for k, v in src.features.items() if k not in META}   # create() adds the bookkeeping features itself
    dst = LeRobotDataset.create(f"lastlight/{a.dst}", src.fps, root=dst_root, robot_type=src.meta.robot_type, features=feats, use_videos=True,
                                image_writer_threads=4, rgb_encoder=RGBEncoderConfig(vcodec=a.vcodec), streaming_encoding=False)
    img_keys = [k for k, v in src.features.items() if v["dtype"] == "video"]; kept = 0; total_in = 0; total_out = 0
    for e in range(src.meta.total_episodes):
        if e in drop: print(f"ep {e:2d}: dropped"); continue
        i0 = src.meta.episodes[e]["dataset_from_index"]; i1 = src.meta.episodes[e]["dataset_to_index"]; n = i1 - i0
        st = np.stack([src.hf_dataset[i]["observation.state"].numpy() for i in range(i0, i1)])
        keep = n; start = 0
        if not a.no_trim:
            ac = np.stack([src.hf_dataset[i]["action"].numpy() for i in range(i0, i1)])
            dev = np.abs(ac[:, :5] - ac[0, :5]).max(axis=1); first = np.where(dev > 3.0)[0]
            start = max(0, int(first[0] - a.head * src.fps)) if len(first) else 0
            vel = np.abs(np.diff(st[:, :5], axis=0)).max(axis=1); moving = np.where(vel > 0.4)[0]
            keep = min(n, int(((moving[-1] + 1) / src.fps + a.tail) * src.fps)) if len(moving) else n
        for i in range(i0 + start, i0 + keep):
            item = src[i]; frame = {"task": a.task}
            for k in feats:
                x = item[k]
                if k in img_keys: x = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)   # CHW float -> HWC uint8
                elif isinstance(x, torch.Tensor): x = x.numpy()
                frame[k] = x
            dst.add_frame(frame)
        dst.save_episode(parallel_encoding=False); kept += 1; total_in += n; total_out += keep - start
        print(f"ep {e:2d}: kept {start}..{keep}/{n} frames ({(keep - start) / src.fps:.1f}s, head cut {start / src.fps:.1f}s)", flush=True)
    dst.finalize(); print(f"done: {kept} episodes, {total_out}/{total_in} frames -> {dst_root}")
if __name__ == "__main__": main()
