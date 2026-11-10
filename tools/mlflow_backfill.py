#!/usr/bin/env python3
"""mlflow_backfill.py · import past LeRobot training runs into the local MLflow store, one MLflow run per training run.
    params  <- checkpoints/*/pretrained_model/train_config.json (policy type, chunk, lr, batch, steps, dataset, warm start)
    metrics <- outputs/train/<run>.log ("step: 2K ... loss: 0.123 ... lr: 1e-5" lines, same regexes as the dashboard)
    tags    <- host, task family, checkpoints present
Safe to re-run: a run that already exists (same name) is deleted and re-imported."""
import glob, json, os, re, time
import mlflow
HOME = os.path.expanduser("~"); TRAIN = f"{HOME}/so101/outputs/train"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{HOME}/so101/mlflow/mlflow.db"))
STEP = re.compile(r"step:\s*([\d.]+)([KM]?)"); LOSS = re.compile(r"loss:\s*([\d.]+)"); LR = re.compile(r"lr:\s*([\d.e+-]+)")
def mult(s): return {"K": 1e3, "M": 1e6}.get(s, 1)
def parse_log(path):
    pts = []
    for line in open(path, errors="ignore").read().replace("\r", "\n").splitlines():
        m = STEP.search(line); l = LOSS.search(line)
        if m and l:
            lr = LR.search(line); pts.append((int(float(m.group(1)) * mult(m.group(2))), float(l.group(1)), float(lr.group(1)) if lr else None))
    return pts
def params_for(run_dir):
    cfgs = sorted(glob.glob(f"{run_dir}/checkpoints/*/pretrained_model/train_config.json"))
    if not cfgs: return {}, []
    d = json.load(open(cfgs[-1])); p = d.get("policy", {})
    params = {"policy": p.get("type"), "chunk_size": p.get("chunk_size"), "n_action_steps": p.get("n_action_steps"), "lr": p.get("optimizer_lr"), "dim_model": p.get("dim_model"),
              "use_vae": p.get("use_vae"), "batch_size": d.get("batch_size"), "steps": d.get("steps"), "seed": d.get("seed"), "dataset": (d.get("dataset") or {}).get("repo_id"),
              "warm_start": (p.get("pretrained_path") or "scratch").replace(HOME + "/so101/outputs/train/", ""), "device": p.get("device")}
    return {k: v for k, v in params.items() if v is not None}, [os.path.basename(os.path.dirname(os.path.dirname(c))) for c in cfgs]
def family(name):
    for k in ("lighter", "sim_pick", "bottle", "jar"):
        if k in name: return k
    return "other"
def import_run(name, log_path, run_dir, host, extra_tags=None, extra_metrics=None):
    exp = mlflow.set_experiment("so101-act")
    for old in mlflow.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{name}'", output_format="list"): mlflow.delete_run(old.info.run_id)
    params, ckpts = params_for(run_dir) if run_dir and os.path.isdir(run_dir) else ({}, [])
    pts = parse_log(log_path) if log_path and os.path.exists(log_path) else []
    start = int(os.path.getmtime(log_path) * 1000) if log_path and os.path.exists(log_path) else int(time.time() * 1000)
    with mlflow.start_run(run_name=name, tags={"host": host, "family": family(name), "checkpoints": ",".join(ckpts) or "none", "source": "backfill", **(extra_tags or {})}) as r:
        if params: mlflow.log_params(params)
        if pts:
            mlflow.log_metrics({"loss": pts[0][1]}, step=pts[0][0])
            client = mlflow.MlflowClient(); from mlflow.entities import Metric
            ts = start; batch = []
            for s, loss, lr in pts:
                batch.append(Metric("loss", loss, ts, s))
                if lr is not None: batch.append(Metric("lr", lr, ts, s))
            for i in range(0, len(batch), 900): client.log_batch(r.info.run_id, metrics=batch[i:i + 900])
            mlflow.log_metric("final_loss", pts[-1][1], step=pts[-1][0])
        if extra_metrics: mlflow.log_metrics(extra_metrics)
        if log_path and os.path.exists(log_path): mlflow.log_artifact(log_path, "logs")
        for c in glob.glob(f"{run_dir}/checkpoints/*/pretrained_model/train_config.json")[-1:]: mlflow.log_artifact(c, "config")
    print(f"  {name:22s} {len(pts):4d} loss points  {len(params):2d} params  ckpts={','.join(ckpts) or '-'}  host={host}")
if __name__ == "__main__":
    print("tracking:", mlflow.get_tracking_uri())
    for log in sorted(glob.glob(f"{TRAIN}/*.log")):
        name = os.path.basename(log)[:-4]; import_run(name, log, f"{TRAIN}/{name}", "mac")
    for d in sorted(glob.glob(f"{TRAIN}/*/")):   # runs with checkpoints but no log on this machine (cloud / jetson)
        name = os.path.basename(d.rstrip("/"))
        if not os.path.exists(f"{TRAIN}/{name}.log"): import_run(name, None, d, "cloud-or-jetson")
    j = f"{HOME}/so101/mlflow/import/act_lighter_c50.log"
    if os.path.exists(j): import_run("act_lighter_c50", j, None, "jetson", {"note": "chunk-50 variant trained on the Orin Nano"})
    # measured rollout numbers worth keeping next to the runs (sim eval, 29-30 Oct 2026)
    client = mlflow.MlflowClient(); exp = mlflow.get_experiment_by_name("so101-act")
    for rn, metrics, note in (("act_sim_pick_v4b", {"sim_success_pooled": 0.84, "sim_success_seedset_a": 0.90, "sim_success_seedset_b": 0.78}, "eval_sim at 20k total steps (v4 10k warm start + 10k); staged on Jetson as act_sim_pick_v4_20k"),):
        for r in mlflow.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{rn}'", output_format="list"):
            for k, v in metrics.items(): client.log_metric(r.info.run_id, k, v)
            client.set_tag(r.info.run_id, "eval_note", note)
    print("done")
