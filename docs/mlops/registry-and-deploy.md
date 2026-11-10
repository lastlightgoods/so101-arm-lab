# Model registry and deployment

How a trained SO-101 policy gets from a checkpoint folder to the arm, and back if it is worse: `tools/so101_policy.py`
(`so101-policy` in `env.sh`) on top of the MLflow Model Registry that `tools/mlflow_ui.sh` already serves. Every version the
Jetson has ever run is a numbered directory on the Jetson, a numbered version in the registry, and an entry in one JSON file.

```
train ──► register ──► candidate ──► deploy ──► jetson ──► rollback
  │           │            │            │          │           │
  │           │            │            │          │           └ previous entry in DEPLOYED.json's history; one command
  │           │            │            │          └ alias on the version that is live; version tagged deployed_at / deployed_to / deployed_by
  │           │            │            └ rsync to ~/so101/policies/<name>/v<N>/, current -> v<N>, DEPLOYED.json, sha256 checked
  │           │            └ alias on the newest registered version (what you mean to try next)
  │           └ one MLflow run (params, tags, the checkpoint as an artifact) + one model version
  └ so101-train / so101-finetune on the Mac or Jetson, or an HF Jobs run that lands on the Hub
```

## Where things live

| Place | What |
| --- | --- |
| `~/so101/mlflow/mlflow.db` (Mac) | the registry and the `so101-registry` experiment, same SQLite file as `so101-act`; `MLFLOW_TRACKING_URI` overrides (an `http://` server works too). This tool does not read `SO101_MLFLOW_URI` or `~/so101/.mlflow_uri`, which the loggers in `tracking.md` use; to send the registry through the server, set `MLFLOW_TRACKING_URI` to the same URL |
| `~/so101/mlflow/artifacts/so101-registry/<run>/artifacts/model/` (Mac) | the registered checkpoint, byte for byte (`SO101_MLFLOW_ARTIFACTS` overrides the root) |
| `~/so101/policies_hub/<owner>__<repo>/` (Mac) | Hub snapshots pulled by `register --from-hf`; `policies_hub/registry/<name>/v<N>/` is the download cache when the store is remote |
| `~/so101/policies/<name>/v<N>/` (Jetson) | one deployed version: `config.json`, `model.safetensors`, `train_config.json`, processors. The last 3 versions of each model stay |
| `~/so101/policies/<name>/current -> v<N>` (Jetson) | what `so101-run` / `so101-run-async` / `so101-run-vla` load by default |
| `~/so101/policies/DEPLOYED.json` (Jetson) | the latest deploy (name, version, run id, sha256 of the weights, when, by whom, action) and the history of every deploy and rollback |

The older hand-copied folders (`~/so101/policies/act_lighter`, `smolvla_lighter`, ...) are untouched and remain the fallback.

## Commands

All on the Mac, from a shell that sourced `env.sh`. Nothing here needs the Jetson except `deploy`, `rollback` and `status`,
which talk to it over `ssh`/`rsync` (host `$SO101_JETSON`, default `jetson`).

```sh
# 1. register a checkpoint (any pretrained_model dir: train_config.json + config.json + model.safetensors)
so101-policy register ~/so101/outputs/train/act_lighter/checkpoints/last/pretrained_model --name so101-lighter \
    --dataset lastlight/lighter-handoff-clean --note "25k steps, chunk 100"
#    -> run so101-lighter:register in experiment so101-registry (params: policy, chunk_size, n_action_steps, lr, dim_model, use_vae,
#       batch_size, steps, seed, dataset, warm_start, device, the same keys mlflow_backfill.py writes; tags: dataset, host, git_commit,
#       checkpoint_step, sha256, source_dir, hf_repo/hf_job/note when given), the checkpoint under artifacts/model/,
#       model version N with the same tags, alias candidate -> N

# 1b. register a model that an HF Jobs run pushed to the Hub (snapshot_download into ~/so101/policies_hub/)
so101-policy register --from-hf tclastlight/act_sim_pick_v3_2026-10-21_08-27-49 --name so101-sim-pick --hf-job 6ad877870a35baab53a1b3d6 --step 012500
#    a job repo holds every checkpoint; --step narrows the download and picks that one, otherwise the newest is registered.
#    HF_HUB_OFFLINE=1 is honoured (cache only, no network). Cloud checkpoints still need `python jetson/prep_ckpt.py <dir>` before LeRobot 0.6.1 loads them.

# 2. look
so101-policy list                 # every model: versions, aliases, origin (dir or Hub repo + job), which one is deployed and when
so101-policy list so101-lighter

# 3. deploy (by version number or alias; a bare name means the newest version)
so101-policy deploy so101-lighter@candidate --dry-run    # prints the plan, touches nothing
so101-policy deploy so101-lighter@candidate              # rsync, current -> vN, DEPLOYED.json, sha256 verified, alias jetson -> N, version tagged
so101-policy deploy so101-lighter@3                      # a specific version; re-running it is a no-op ("already deployed")
so101-policy deploy so101-lighter@3 --dest ~/so101/policies/lighter-test   # somewhere else on the Jetson (DEPLOYED.json records the path)

# 4. run it on the Jetson: nothing to type, so101-run finds current
so101-run 60              # prints "policy: so101-lighter v3 (~/so101/policies/so101-lighter/current)" first

# 5. worse than before? one command
so101-policy rollback so101-lighter                      # current -> the previous version in the history, alias jetson follows, history records "rollback"

# 6. what is where
so101-policy status                                      # DEPLOYED.json, each model's current on the Jetson vs the registry's jetson alias, the so101-run default
```

`status` exits 1 on any MISMATCH (alias moved by hand in the UI, weights missing on the Jetson, a registry `jetson` alias for a model that is not on the Jetson).

## What the Jetson side does with it

`jetson/env.sh`:

- `so101-policy-current [name] [type]` prints `~/so101/policies/<name>/current` when that symlink exists and has weights. The name comes
  from the argument, then `$SO101_POLICY`, then `DEPLOYED.json`. With a type (`act` or `smolvla`, read from `config.json`) only a matching
  checkpoint counts, so an ACT deployment never ends up in the SmolVLA runner.
- `so101-run` and `so101-run-async` default to `so101-policy-current "" act`, `so101-run-vla` to `so101-policy-current "$SO101_VLA_POLICY" smolvla`.
  When nothing fits they fall back to the old fixed paths (`~/so101/policies/act_lighter`, `~/so101/policies/smolvla_lighter`). An explicit
  checkpoint argument still wins. Each prints a `policy: <name> v<N> (<path>)` line before the cameras are taken.
- The dashboard's POLICY / VLA buttons still pass their own fixed paths (`dashboard/server.py`, `ARM_MODES`); switch them to a bare
  `so101-run-async 90` once a registry deployment is the one you want on the button.

`dashboard/server.py` reads `DEPLOYED.json` inside the existing 15 s ssh probe (section `###deployed`), caches it in `state["policy"]`
and serves it at `GET /api/policy` (no ssh per request; an unreachable Jetson keeps the last answer). `~/so101/policies/<name>/current/train_config.json`
is also in the probe's policy glob, so a deployed model shows in the POLICIES panel like the hand-copied ones; `index.html` adds `v<N>` to that
row and `JETSON · <name> v<N>` to the panel header.

## Rollback and retention in detail

`DEPLOYED.json` keeps one history list for all models (newest last, capped at 50). `rollback <name>` takes the last entry for that name and
redeploys the most recent earlier entry with a different version, so after `1, 2, 3` it goes to 2, and a second rollback goes to 3 again
(the "previous" of 2 is now 3). After each deploy the tool keeps `v<N>` for the three most recent distinct versions in that model's history
(the new one first) and `rm -rf`s the rest; a pruned version can still be deployed again, rsync just copies it back. The sha256 of
`model.safetensors` is computed on the Mac before the copy and on the Jetson after it; the alias moves only when they agree.

## How a Hub model enters the registry

`register --from-hf <owner>/<repo>` calls `huggingface_hub.snapshot_download` into `~/so101/policies_hub/<owner>__<repo>/` (with
`--step` only `checkpoints/<step>/*` plus the top-level json/safetensors/md files) and registers the chosen `pretrained_model`
directory exactly like a local one, with `hf_repo` (and `--hf-job`) as tags on the run and the version. A repo with the checkpoint at
its root (what `policy.push_to_hub=true` writes) is registered as is. The `.cache` folder the Hub client leaves next to the files is
not part of the checkpoint and is never copied to the Jetson.

## Tests

`tests/tools/` (29 tests, about 30 s): a temp SQLite tracking URI with its own artifact root, a fake checkpoint, and
`SO101_DEPLOY_LOCAL_ROOT` pointing at a temp directory so `ssh` becomes `bash -s` with that `$HOME` and `rsync` copies locally;
register/list/deploy/rollback/status run end to end. The Hub path uses a fake `snapshot_download` under `HF_HUB_OFFLINE=1`, plus one
test that lets the real client miss an empty offline cache. `test_jetson_env_policy.py` sources `jetson/env.sh` against a fake `$HOME`
with a `lerobot-rollout` shim. `tests/server/test_policy_api.py` covers the probe section and `/api/policy` with the scripted fake ssh.
The tools tests skip themselves when `mlflow` or `rsync` is missing (CI installs only `requirements-dev.txt`).

## Not done yet

- The dashboard buttons do not resolve `current` (see above); the registry alias is the source of truth, the button strings are not.
- `register` does not run `jetson/prep_ckpt.py`; cloud checkpoints written by a newer LeRobot still need it before the Jetson loads them.
- No rollout logging yet: success per episode as a nested run under the deployed version is the next step in `docs/mlflow_field_guide.html`.
