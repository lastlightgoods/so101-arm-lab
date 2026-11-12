# /// script
# requires-python = ">=3.10"
# dependencies = ["ultralytics>=8.3", "huggingface_hub", "opencv-python-headless"]
# ///
"""Fine-tune YOLO11n on the SO-101 mat dataset (private HF dataset) and push best.pt to a private HF model repo. Runs on a Hugging Face Job (never on the Mac).
Env: HF_TOKEN (secret), DATASET, MODEL_REPO, EPOCHS, IMGSZ, BATCH, RUN_NAME."""
import os, json, time, pathlib
from huggingface_hub import snapshot_download, HfApi
t0 = time.time()
DS, MR = os.environ.get("DATASET", "tclastlight/so101-mat-detect"), os.environ.get("MODEL_REPO", "tclastlight/so101-mat-detector")
EPOCHS, IMGSZ, BATCH, RUN = int(os.environ.get("EPOCHS", "150")), int(os.environ.get("IMGSZ", "960")), int(os.environ.get("BATCH", "16")), os.environ.get("RUN_NAME", "run1")
root = pathlib.Path(snapshot_download(DS, repo_type="dataset", local_dir="/tmp/ds"))
(root / "data_abs.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: nut\n  1: wing_nut\n  2: spacer\n  3: ring\n  4: frog\n  5: toy\n")
(root / "synth.yaml").write_text(f"path: {root}/synth_frog_test\ntrain: images\nval: images\nnames:\n  0: nut\n  1: wing_nut\n  2: spacer\n  3: ring\n  4: frog\n  5: toy\n")
import subprocess
subprocess.run('apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0', shell=True, check=False)   # ultralytics pulls full opencv-python, which needs libGL
from ultralytics import YOLO
m = YOLO("yolo11n.pt")
m.train(data=str(root / "data_abs.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=int(os.environ.get('PATIENCE','60')), project="/tmp/runs", name=RUN, exist_ok=True,
        degrees=0, scale=0.25, translate=0.1, fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=20, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, workers=4, seed=0, deterministic=True, plots=True)
best = pathlib.Path("/tmp/runs") / RUN / "weights" / "best.pt"
b = YOLO(str(best)); out = {}
for split in ("val", "test"):
    r = b.val(data=str(root / "data_abs.yaml"), split=split, imgsz=IMGSZ, conf=0.001, plots=False, project="/tmp/runs", name=f"{RUN}_{split}")
    out[split] = dict(map50=float(r.box.map50), map5095=float(r.box.map), per_class_map50={r.names[i]: float(v) for i, v in zip(r.box.ap_class_index, r.box.ap50)})
out["train_seconds"] = time.time() - t0
json.dump(out, open("/tmp/runs/eval_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
api = HfApi(); api.create_repo(MR, repo_type="model", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=str(best), path_in_repo=f"{RUN}/best.pt", repo_id=MR, repo_type="model")
api.upload_file(path_or_fileobj=str(best.parent / "last.pt"), path_in_repo=f"{RUN}/last.pt", repo_id=MR, repo_type="model")
for f in ("results.csv", "args.yaml", "results.png", "confusion_matrix.png"):
    p = pathlib.Path("/tmp/runs") / RUN / f
    if p.exists(): api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{RUN}/{f}", repo_id=MR, repo_type="model")
api.upload_file(path_or_fileobj="/tmp/runs/eval_summary.json", path_in_repo=f"{RUN}/eval_summary.json", repo_id=MR, repo_type="model")
print("DONE total seconds", time.time() - t0)
