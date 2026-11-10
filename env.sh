# source ~/so101/env.sh
source ~/so101/.venv/bin/activate
export FOLLOWER=/dev/tty.usbmodem5C821070341   # board serial 5C82107034
export LEADER=/dev/tty.usbmodem5C821071281     # board serial 5C82107128
export FOLLOWER_ID=ll_follower
export LEADER_ID=ll_leader
alias so101-calib-follower='lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID'
alias so101-calib-leader='lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID'
alias so101-teleop='lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID'
alias so101-teleop-robust='python ~/so101/tools/teleop_robust.py'
export SO101_CAMS='{top: {type: avf, match: icspring, native: 1920x1080, width: 640, height: 360, fps: 30}, wrist: {type: avf, match: icspring, native: 640x480, width: 640, height: 480, fps: 30}}'
# so101-record <dataset-name> "<task description>" [num_episodes]   -> local dataset, no upload
so101-record() {
  local name="$1" task="$2" n="${3:-10}"
  lerobot-record \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID \
    --robot.cameras="$SO101_CAMS" \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID \
    --display_data=true \
    --dataset.repo_id=lastlight/$name --dataset.single_task="$task" \
    --dataset.num_episodes=$n --dataset.episode_time_s=40 --dataset.reset_time_s=12 --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false
}
# so101-record-more <dataset-folder-name> <num_more_episodes>   -> resume an existing local dataset
so101-record-more() { local ds="lastlight/$1" n="${2:-10}"
  lerobot-record --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="$SO101_CAMS" \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID --display_data=true \
    --resume=true --dataset.repo_id=$ds --dataset.root=$HOME/.cache/huggingface/lerobot/$ds \
    --dataset.single_task="Pick up the bottle and place it in the bin" --dataset.num_episodes=$n \
    --dataset.episode_time_s=40 --dataset.reset_time_s=12 --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false; }
# so101-run [seconds] [checkpoint-dir]  -> run the trained policy (ACT temporal ensembling, ~29 Hz). Async alt: --inference.type=rtc
so101-run() { local dur="${1:-60}" ckpt="${2:-$HOME/so101/outputs/train/act_bottle_pick/checkpoints/last/pretrained_model}"
  lerobot-rollout --strategy.type=base --inference.type=sync --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 --policy.path="$ckpt" --policy.device=mps \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="$SO101_CAMS" \
    --task="Pick up the bottle and place it in the bin" --duration=$dur --display_data=false; }
# gripper force/speed: PWM cap (speed follows it). 250 = gentle but slow; 400 = keeps up with the trigger.
export LEROBOT_GRIPPER_TORQUE=500
export LEROBOT_GRIPPER_CURRENT=700
export LEROBOT_GRIPPER_P=32
# so101-train <dataset-folder-name> [job-name] [steps]  -> ACT on the Mac GPU; writes train.log so the dashboard's loss chart follows it
so101-train() { local ds="$1" job="${2:-act_$1}" steps="${3:-40000}"
  lerobot-train --dataset.repo_id=lastlight/$ds --policy.type=act --output_dir=$HOME/so101/outputs/train/$job --job_name=$job \
    --policy.device=mps --policy.push_to_hub=false --wandb.enable=false --steps=$steps --save_freq=5000 2>&1 | tee -a $HOME/so101/outputs/train/$job.log; }
# lab dashboard on monitor 2:  so101-dash | so101-dash stop   (http://127.0.0.1:8765)
alias so101-dash='$HOME/so101/dashboard/dash.sh'
# MLflow tracking server (docs/mlops/tracking.md): so101-mlflow start [--lan] | stop | status | uri. Mac tools log straight into the
# SQLite store by default; ~/so101/.mlflow_uri (one line, the server URL) makes them log through the server instead, like the Jetson.
alias so101-mlflow='$HOME/so101/tools/mlflow_ui.sh'
[ -s ~/so101/.mlflow_uri ] && export SO101_MLFLOW_URI="$(head -n1 ~/so101/.mlflow_uri)"
# why: the container server (docs/mlops/colima.md) keeps artifacts in S3; mlflow >= 3.17 clients would otherwise move them via presigned http://s3:9000 URLs only the containers can reach
export MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=${MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD:-false} MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=${MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD:-false}
# so101-finetune <dataset> <checkpoint-dir> [job] [steps]  -> continue training from a pretrained_model dir (ACT config comes from it).
so101-finetune() { local ds="$1" ckpt="$2" job="${3:-ft_$1}" steps="${4:-8000}"
  lerobot-train --dataset.repo_id=lastlight/$ds --policy.path="$ckpt" --output_dir=$HOME/so101/outputs/train/$job --job_name=$job \
    --policy.device=mps --policy.push_to_hub=false --wandb.enable=false --steps=$steps --save_freq=2000 2>&1 | tee -a $HOME/so101/outputs/train/$job.log; }
# ---- simulation (MuJoCo SO-101 via so101-nexus + sim/so101-lerobot scripts; Apache-2.0) ----
export MUJOCO_GL=glfw
SIM=$HOME/so101/sim/so101-lerobot
# so101-sim-collect [episodes] [name] : scripted-expert pick-and-place demos -> sim/data/<name> (LeRobot v3, so101_follower units)
so101-sim-collect() { local n="${1:-300}" name="${2:-so101_sim_pick_v1}"; python $SIM/collect_sim.py --episodes $n --seed 11 --img-w 320 --img-h 240 --overhead-margin 0.04 --spawn-min 0.04 --spawn-max 0.17 --repo-id tclastlight/$name --root $HOME/so101/sim/data/$name --gif $HOME/so101/sim/data/$name.gif --overwrite 2>&1 | grep -vE "warn|Warn|deprecat" | tee -a $HOME/so101/sim/data/$name.collect.log; }
# so101-sim-train [name] [job] [steps] : ACT on the sim dataset (mps), log beside the run like so101-train
so101-sim-train() { local name="${1:-so101_sim_pick_v1}" job="${2:-act_sim_$1}" steps="${3:-20000}"; mkdir -p $HOME/so101/outputs/train
  lerobot-train --dataset.repo_id=tclastlight/$name --dataset.root=$HOME/so101/sim/data/$name --policy.type=act --policy.device=mps --policy.push_to_hub=false \
    --output_dir=$HOME/so101/outputs/train/$job --job_name=$job --steps=$steps --batch_size=8 --save_freq=2500 --log_freq=100 --num_workers=2 --wandb.enable=false 2>&1 | tee -a $HOME/so101/outputs/train/$job.log; }
# so101-sim-eval <pretrained_model dir> [episodes] : success rate on unseen layouts (+ GIFs in sim/rollouts)
so101-sim-eval() { local ckpt="$1" n="${2:-50}"; mkdir -p $HOME/so101/sim/rollouts; (cd $SIM && python eval_sim.py --policy "$ckpt" --episodes $n --seed 9100 --goal-thresh 0.04 --overhead-margin 0.04 --gif-dir $HOME/so101/sim/rollouts 2>&1 | grep -vE "warn|Warn|deprecat"); }
# so101-policy register|list|deploy|rollback|status : model registry (MLflow) -> Jetson deployment, docs/mlops/registry-and-deploy.md
so101-policy() { $HOME/so101/.venv/bin/python $HOME/so101/tools/so101_policy.py "$@"; }
# so101-film live <seconds> [out.mp4] | dataset <name> [episode|all] [out.mp4]   (--audio file.wav, --no-screen)
# multi-camera film: command center (kiosk screen) left, overhead + front + wrist right, 1920x1080 -> ~/so101/films/
so101-film() { $HOME/so101/.venv/bin/python $HOME/so101/tools/film.py "$@"; }
