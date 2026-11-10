"""Pull hero frames out of the recorded datasets for the dashboard gallery (top cam at 60%, wrist at 50%, top at end)."""
import glob, os, subprocess, sys, pandas as pd
ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/lastlight"); OUT = os.path.expanduser("~/so101/dashboard/img")
for ds in sys.argv[1:] or ["bottle-pick-clean", "jar-pocket-trim"]:
    os.makedirs(f"{OUT}/{ds}", exist_ok=True)
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/{ds}/meta/episodes/**/*.parquet", recursive=True)))
    for _, r in df.iterrows():
        ep = int(r["episode_index"])
        for cam, frac, tag in (("top", 0.6, "mid"), ("wrist", 0.5, "mid"), ("top", 0.97, "end")):
            k = f"videos/observation.images.{cam}"
            f0, f1 = r[f"{k}/from_timestamp"], r[f"{k}/to_timestamp"]
            vid = f"{ROOT}/{ds}/videos/observation.images.{cam}/chunk-{int(r[k+'/chunk_index']):03d}/file-{int(r[k+'/file_index']):03d}.mp4"
            out = f"{OUT}/{ds}/ep{ep:03d}_{cam}_{tag}.jpg"
            if os.path.exists(out): continue
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{f0 + (f1 - f0) * frac:.3f}", "-i", vid, "-frames:v", "1", "-q:v", "3", out], check=False)
    print(ds, len(glob.glob(f"{OUT}/{ds}/*.jpg")), "images")
