"""Rebuild jar-pocket with idle tails cut (1 s after the arm's last motion) and bad episodes dropped."""
import os, glob, numpy as np, pandas as pd, torch, logging
logging.disable(logging.WARNING)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
SRC_ID="lastlight/jar-pocket_20261007_110305"; DST_ID="lastlight/jar-pocket-trim"
ROOT=os.path.expanduser("~/.cache/huggingface/lerobot/")
DROP={5,7,12,21}; TAIL_S=1.0; TASK="Place the jar in the tray pocket"
def main():
    src=LeRobotDataset(SRC_ID, root=ROOT+SRC_ID, video_backend="pyav")
    ep=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ROOT+SRC_ID+"/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ROOT+SRC_ID+"/data/**/*.parquet", recursive=True))])
    feats={k: src.features[k] for k in ("action","observation.state","observation.images.top","observation.images.wrist")}
    dst=LeRobotDataset.create(DST_ID, fps=src.fps, features=feats, root=ROOT+DST_ID, robot_type=src.meta.robot_type, use_videos=True, image_writer_threads=4)
    total=0
    for _,r in ep.iterrows():
        i=int(r.episode_index)
        if i in DROP: print(f"ep{i}: dropped", flush=True); continue
        g=df[df.episode_index==i].sort_values("frame_index"); s=np.stack(g["observation.state"].values)
        mv=np.abs(np.diff(s[:,:5],axis=0)).sum(1); moving=np.where(mv>0.5)[0]
        keep=min(len(g), int(((moving[-1]+1)/src.fps+TAIL_S)*src.fps)) if len(moving) else len(g)
        start=int(r.dataset_from_index)
        for k in range(keep):
            item=src[start+k]
            frame={"task":TASK, "action":item["action"].numpy(), "observation.state":item["observation.state"].numpy()}
            for cam in ("observation.images.top","observation.images.wrist"):
                img=item[cam]; frame[cam]=(img.permute(1,2,0)*255).round().clamp(0,255).to(torch.uint8).numpy()
            dst.add_frame(frame)
        dst.save_episode(parallel_encoding=False); total+=keep
        print(f"ep{i}: kept {keep}/{len(g)} frames ({keep/src.fps:.1f}s of {len(g)/src.fps:.1f}s)", flush=True)
    dst.finalize()
    print(f"DONE: {dst.meta.total_episodes} episodes, {total} frames -> {ROOT+DST_ID}", flush=True)

if __name__ == "__main__":
    main()
