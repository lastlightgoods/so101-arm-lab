#!/usr/bin/env python3
"""migrate_runs.py · copy experiments, runs, metric histories, artifacts and run-backed registered models from one MLflow store to
another through the public client API. Source is normally a COPY of the lab's SQLite file (sqlite:///.../mlflow.db, made by
migrate_sqlite.sh); target is the containerised server (http://127.0.0.1:5050) or, later, the AWS one.

    migrate_runs.py --source sqlite:///tmp/copy/mlflow.db --target http://127.0.0.1:5050 [--experiments so101-act ...] [--dry-run]

Run ids change (a client cannot create a run with a chosen id), so every copied run carries the tags `source_run_id` and
`migrated_from`; re-running skips runs the target already has. Not copied: deleted runs, traces, and prompt-registry entries
(registered models without a run behind them).
why the client API and not a SQL dump: SQLite -> Postgres needs a type-by-type translation, and the artifact URIs stored in every
run row would still point at Mac folders; written through the API, the target store is correct for its own artifact root."""
import argparse, os, sys, tempfile
import mlflow
from mlflow.entities import Metric, Param, ViewType
SRC_TAG, FROM_TAG = "source_run_id", "migrated_from"
def batched(seq, n):
    for i in range(0, len(seq), n): yield seq[i:i + n]
def copy_run(src, dst, run, dst_exp_id, origin, tmp):
    d = run.data
    # why: mlflow.log-model.history embeds the old artifact root; everything else (runName, user, source, the lab's host/family tags) carries over
    tags = {k: v for k, v in d.tags.items() if k != "mlflow.log-model.history"}; tags[SRC_TAG] = run.info.run_id; tags[FROM_TAG] = origin
    nid = dst.create_run(dst_exp_id, start_time=run.info.start_time, tags=tags, run_name=run.info.run_name).info.run_id
    for chunk in batched([Param(k, v) for k, v in d.params.items()], 100): dst.log_batch(nid, params=chunk)   # why: 100 params / 1000 metrics per log_batch call
    metrics = [Metric(m.key, m.value, m.timestamp, m.step) for key in d.metrics for m in src.get_metric_history(run.info.run_id, key)]
    for chunk in batched(metrics, 1000): dst.log_batch(nid, metrics=chunk)
    n_files = 0
    if src.list_artifacts(run.info.run_id):
        local = src.download_artifacts(run.info.run_id, "", tmp); dst.log_artifacts(nid, local)
        n_files = sum(len(f) for _, _, f in os.walk(local))
    dst.set_terminated(nid, status=run.info.status, end_time=run.info.end_time)
    return nid, len(d.params), len(metrics), n_files
def copy_models(src, dst, run_map, dry):
    for m in src.search_registered_models():
        versions = src.search_model_versions(f"name = '{m.name}'")
        if not versions or any(v.run_id is None for v in versions):
            print(f"  model {m.name}: skipped ({'no versions' if not versions else 'not backed by runs: prompt or external source'})"); continue
        if dry: print(f"  model {m.name}: {len(versions)} versions would be copied"); continue
        if _exists(dst, m.name): print(f"  model {m.name}: already on target, skipped"); continue
        dst.create_registered_model(m.name, tags=m.tags, description=m.description)
        new_by_old = {}
        for v in sorted(versions, key=lambda v: int(v.version)):
            if v.run_id not in run_map: print(f"    v{v.version}: run {v.run_id[:8]} not migrated, skipped"); continue
            old_root = src.get_run(v.run_id).info.artifact_uri; new_root = dst.get_run(run_map[v.run_id]).info.artifact_uri
            source = new_root + v.source[len(old_root):] if v.source.startswith(old_root) else v.source   # why: the model files moved with the run's artifacts
            nv = dst.create_model_version(m.name, source, run_id=run_map[v.run_id], tags=v.tags, description=v.description)
            new_by_old[v.version] = nv.version
        for alias, old_v in (m.aliases or {}).items():
            if old_v in new_by_old: dst.set_registered_model_alias(m.name, alias, new_by_old[old_v])
        print(f"  model {m.name}: {len(new_by_old)} versions, {len(m.aliases or {})} aliases")
def _exists(client, name):
    try: client.get_registered_model(name); return True
    except mlflow.exceptions.MlflowException: return False
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="tracking URI to read, e.g. sqlite:////tmp/copy/mlflow.db")
    ap.add_argument("--target", required=True, help="tracking URI to write, e.g. http://127.0.0.1:5050")
    ap.add_argument("--experiments", nargs="*", help="experiment names to copy (default: every active experiment with runs)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied, write nothing")
    a = ap.parse_args()
    src, dst = mlflow.MlflowClient(tracking_uri=a.source), mlflow.MlflowClient(tracking_uri=a.target)
    origin = a.source.split("/")[-1] or a.source
    total = {"experiments": 0, "runs": 0, "skipped": 0, "params": 0, "metrics": 0, "files": 0}; run_map = {}
    for e in src.search_experiments(view_type=ViewType.ACTIVE_ONLY):
        if a.experiments and e.name not in a.experiments: continue
        runs = src.search_runs([e.experiment_id], run_view_type=ViewType.ACTIVE_ONLY, max_results=50000)
        if not runs: print(f"experiment {e.name}: no runs, skipped"); continue
        de = dst.get_experiment_by_name(e.name)
        have = {r.data.tags.get(SRC_TAG): r.info.run_id for r in dst.search_runs([de.experiment_id], run_view_type=ViewType.ALL, max_results=50000)} if de else {}
        print(f"experiment {e.name}: {len(runs)} runs on source, {len([h for h in have if h])} already migrated on target")
        total["experiments"] += 1
        if de is None and not a.dry_run: de_id = dst.create_experiment(e.name, tags=e.tags)
        else: de_id = de.experiment_id if de else None
        for run in sorted(runs, key=lambda r: r.info.start_time or 0):
            if run.info.run_id in have: run_map[run.info.run_id] = have[run.info.run_id]; total["skipped"] += 1; continue
            if a.dry_run: print(f"  {run.info.run_name:24s} {run.info.run_id[:8]}  {len(run.data.params):3d} params  {len(run.data.metrics):3d} metric keys"); continue
            with tempfile.TemporaryDirectory() as tmp: nid, n_p, n_m, n_f = copy_run(src, dst, run, de_id, origin, tmp)
            run_map[run.info.run_id] = nid; total["runs"] += 1; total["params"] += n_p; total["metrics"] += n_m; total["files"] += n_f
            print(f"  {run.info.run_name:24s} {run.info.run_id[:8]} -> {nid[:8]}  {n_p:3d} params {n_m:5d} metric points {n_f:3d} artifact files")
    copy_models(src, dst, run_map, a.dry_run)
    print(("dry run: " if a.dry_run else "done: ") + ", ".join(f"{v} {k}" for k, v in total.items()))
    return 0
if __name__ == "__main__": sys.exit(main())
