# Jetson version of env.sh — source ~/so101/env.sh   (mirrors the Mac env.sh, 14 Oct 2026)
source ~/so101/.venv/bin/activate
export FOLLOWER=/dev/so101_follower   # udev symlink, board serial 5C82107034
export LEADER=/dev/so101_leader       # udev symlink, board serial 5C82107128
export FOLLOWER_ID=ll_follower
export LEADER_ID=ll_leader
# Linux: cameras by stable /dev/v4l/by-id path (fill in after `ls -l /dev/v4l/by-id/`)
export CAM_TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0     # Logitech C920 overhead (28 Oct), MJPG 640x360
export CAM_WRIST=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0           # icSpring wrist, 640x480 YUYV, left side of the arm
export CAM_FRONT=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0   # Logitech C920 facing the arm from the far side (28 Oct), MJPG 640x360
export SO101_CAMS="{top: {type: opencv, index_or_path: $CAM_TOP, width: 640, height: 360, fps: 30, fourcc: MJPG}, front: {type: opencv, index_or_path: $CAM_FRONT, width: 640, height: 360, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $CAM_WRIST, width: 640, height: 480, fps: 30}}"   # 3 cameras for recording
export SO101_CAMS_2="{top: {type: opencv, index_or_path: $CAM_TOP, width: 640, height: 360, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $CAM_WRIST, width: 640, height: 480, fps: 30}}"   # 2 cameras for policies trained before the front camera existed
so101-calib-follower() { so101-arms-take; lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID; so101-arms-release; }
so101-calib-leader()   { so101-arms-take; lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID; so101-arms-release; }
so101-teleop() { so101-arms-take; lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID; so101-arms-release; }
alias so101-cams='python -c "import cv2,os;[print(p, os.environ[p], cv2.VideoCapture(os.environ[p]).isOpened()) for p in (\"CAM_TOP\",\"CAM_WRIST\")]"'
# busy flags: the Mac dashboard releases the cameras while ~/so101/.cams_busy exists and the arm serial ports
# while ~/so101/.arms_busy exists (its joint-state poller then forwards what the busheal plugin publishes instead)
so101-cams-take()    { touch ~/so101/.arms_busy ~/so101/.cams_busy; sleep 1.5; }
so101-cams-release() { rm -f ~/so101/.cams_busy ~/so101/.arms_busy; }
so101-arms-take()    { touch ~/so101/.arms_busy; sleep 1; }
so101-arms-release() { rm -f ~/so101/.arms_busy; }
# camera lock (docs/camera-settings.md): exposure / white balance / focus frozen to ~/so101/calib/cam_settings.json so the arm
# entering the frame cannot re-expose the picture. Re-applied by the udev rule + user service on boot/re-plug.
so101-cams-capture() { python3 ~/so101/cam_lock.py capture "$@"; }            # new lighting session: let auto choose, lock, tune the picture to match (needs the venv python: this shell)
so101-cams-lock()    { python3 ~/so101/cam_lock.py apply --verify "$@"; }     # re-apply the saved values and measure the picture (rolls back to auto if it is off)
so101-cams-check()   { python3 ~/so101/cam_lock.py check "$@"; }              # exit 0 only if all three match the file
so101-cams-guard()   { mkdir -p ~/so101/outputs/run; (nohup python3 ~/so101/cam_lock.py guard >> ~/so101/outputs/run/cam_lock_guard.log 2>&1 &) }   # resident re-locker (no-op if one runs)
so101-cams-locked-or-bail() { python3 ~/so101/cam_lock.py check --strict && { so101-cams-guard; return 0; }
  echo "REFUSING TO RECORD: cameras are not locked (see the line above). Run so101-cams-lock, or so101-cams-capture for a new lighting session, then retry." >&2; return 1; }
# so101-record <dataset-name> "<task description>" [num_episodes]   -> local dataset, no upload
so101-record() {
  local name="$1" task="$2" n="${3:-10}"; so101-cams-locked-or-bail || return 1; so101-cams-take
  lerobot-record \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID \
    --robot.cameras="$SO101_CAMS" \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID \
    --display_data=false --play_sounds=true \
    --dataset.repo_id=lastlight/$name --dataset.single_task="$task" \
    --dataset.num_episodes=$n --dataset.episode_time_s=${SO101_EP_S:-25} --dataset.reset_time_s=${SO101_RESET_S:-10} --dataset.streaming_encoding=true --dataset.encoder_threads=2 --dataset.rgb_encoder.vcodec=h264 \
    --dataset.push_to_hub=false; so101-cams-release
}
# so101-record-more <dataset-folder-name> <num_more_episodes>   -> resume an existing local dataset
so101-record-more() { local ds="lastlight/$1" n="${2:-10}" task="${3:-${SO101_TASK:-Pick up the lighter and hand it to the person}}"; so101-cams-locked-or-bail || return 1; so101-cams-take
  lerobot-record --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="$SO101_CAMS" \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=$LEADER_ID --display_data=false --play_sounds=true \
    --resume=true --dataset.repo_id=$ds --dataset.root=$HOME/.cache/huggingface/lerobot/$ds \
    --dataset.single_task="$task" --dataset.num_episodes=$n \
    --dataset.episode_time_s=${SO101_EP_S:-25} --dataset.reset_time_s=${SO101_RESET_S:-10} --dataset.streaming_encoding=true --dataset.encoder_threads=2 --dataset.rgb_encoder.vcodec=h264 \
    --dataset.push_to_hub=false; so101-cams-release; }
# NOTE (ssh recording): no viewer window and no arrow-key shortcuts; episodes end on the timer. SO101_EP_S / SO101_RESET_S override.
# so101-train <dataset-folder-name> [job-name] [steps]  -> ACT on the Orin GPU (run so101-perf first)
so101-train() { local ds="$1" job="${2:-act_$1}" steps="${3:-40000}"
  lerobot-train --dataset.repo_id=lastlight/$ds --policy.type=act --output_dir=$HOME/so101/outputs/train/$job --job_name=$job \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --steps=$steps --save_freq=5000 --batch_size=8 --num_workers=2 \
    2>&1 | tee -a $HOME/so101/outputs/train/$job.log; }   # train.log feeds the Mac dashboard's loss chart
# ---- deployed policies (so101-policy on the Mac, docs/mlops/registry-and-deploy.md): ~/so101/policies/<name>/v<N>/ + current -> v<N>, DEPLOYED.json ----
# so101-policy-current [name] [type] : path of the deployed checkpoint (~/so101/policies/<name>/current); name defaults to $SO101_POLICY, then DEPLOYED.json.
#   With a type (act|smolvla) only a matching checkpoint counts. Prints nothing and returns 1 when nothing fitting is deployed.
so101-policy-current() { local n="${1:-$SO101_POLICY}" p t
  [ -z "$n" ] && n=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/so101/policies/DEPLOYED.json")))["name"])' 2>/dev/null)
  p="$HOME/so101/policies/$n/current"; { [ -n "$n" ] && [ -L "$p" ] && [ -f "$p/model.safetensors" ]; } || return 1
  if [ -n "$2" ]; then t=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("type",""))' "$p/config.json" 2>/dev/null); [ "$t" = "$2" ] || return 1; fi; echo "$p"; }
# so101-policy-banner <ckpt> : one line saying which policy (and deployed version) is about to run
so101-policy-banner() { if [ -L "$1" ]; then echo "policy: $(basename "$(dirname "$1")") $(readlink "$1") ($1)"; else echo "policy: $1 (not a registry deployment)"; fi; }
# so101-run [seconds] [checkpoint-dir]  -> run a trained policy (ACT temporal ensembling). Async alt: --inference.type=rtc
# Mac-trained policies copied to ~/so101/policies/act_jar_pocket and ~/so101/policies/act_bottle_pick
# so101-run [seconds] [checkpoint-dir] ["task text"]   (SO101_TASK env also works)
# default checkpoint: the deployed ACT policy (so101-policy-current "" act) when one exists, else ~/so101/policies/act_lighter as before
so101-run() { local dur="${1:-60}" ckpt="${2:-$(so101-policy-current "" act)}" task="${3:-${SO101_TASK:-Pick up the lighter and hand it to the person}}"; ckpt="${ckpt:-$HOME/so101/policies/act_lighter}"; so101-policy-banner "$ckpt"; so101-cams-take
  lerobot-rollout --strategy.type=base --inference.type=sync --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 --policy.path="$ckpt" --policy.device=cuda \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="${SO101_RUN_CAMS:-$SO101_CAMS_2}" \
    --task="$task" --duration=$dur --display_data=false; so101-cams-release; }
# gripper force/speed (busheal plugin applies these on connect): PWM cap, speed follows it.
export LEROBOT_GRIPPER_TORQUE=500
export LEROBOT_GRIPPER_CURRENT=700
export LEROBOT_GRIPPER_P=32
# ---- maintenance & performance ----
# so101-update : OS + JetPack packages (NVIDIA apt repo), then pip-upgrade lerobot in the venv. Reboot if it says so.
so101-update() { sudo apt-get update && sudo apt-get dist-upgrade -y && sudo apt-get autoremove -y && sudo pip3 install -U jetson-stats; \
  [ -f /var/run/reboot-required ] && echo ">>> REBOOT REQUIRED: sudo reboot" || echo "update done"; }
# lerobot itself is pinned at 0.6.1 (Python 3.10 + patch, see jetson_setup.sh); to upgrade, bump the pin there and rerun section 4.
# so101-perf   : max power mode + lock clocks (fan gets loud; use for training/rollout).  so101-quiet : back to balanced.
alias so101-perf='sudo nvpmodel -m 2 >/dev/null; sudo jetson_clocks; sudo nvpmodel -q | head -2'
alias so101-quiet='sudo jetson_clocks --restore 2>/dev/null; sudo nvpmodel -m 1 >/dev/null; sudo nvpmodel -q | head -2'
# so101-health : temps, clocks, GPU/CPU load, memory (press q to leave jtop)
alias so101-health='jtop'
alias so101-stats='tegrastats --interval 1000 | head -3'
# so101-stop : stop whatever lerobot job is running (teleop/record/rollout) and clear the dashboard busy flags
so101-stop() { pkill -TERM -f "lerobot-(teleoperate|record|rollout|calibrate|replay)"; pkill -INT -f "so101/(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper).py"; sleep 2; rm -f ~/so101/.arms_busy ~/so101/.cams_busy; echo stopped; }
# demos:  so101-replay <dataset-folder> [episode]  -> follower re-performs a recorded episode (open-loop, no cameras)
#         so101-demo [speed]                       -> scripted wave routine (jetson/demo_wave.py); stop teleop first (so101-stop)
so101-replay() { local ds="${1:-jar-pocket-trim}" ep="${2:-0}"; so101-arms-take
  lerobot-replay --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID \
    --dataset.repo_id=lastlight/$ds --dataset.root=$HOME/.cache/huggingface/lerobot/lastlight/$ds --dataset.episode=$ep; so101-arms-release; }
so101-demo() { so101-arms-take; python ~/so101/demo_wave.py --speed "${1:-1.0}"; so101-arms-release; }
# scripted demonstrations:  so101-mark <rest|hand|spot1..>  (while teleoping: saves the follower pose)
#                           so101-scripted <dataset> "<task>" <episodes>   -> arm performs pick+handoff from the marked poses, recording a dataset
so101-mark() { (cd ~/so101 && python ~/so101/mark_pose.py "$@"); }
so101-scripted() { so101-cams-take; python ~/so101/scripted_record.py "$@"; so101-cams-release; }
# so101-run-async [s] [ckpt] [queue_threshold] ["task"] : rollout with background inference (rtc engine + actasync shim).
# The loop holds 30 Hz regardless of inference time; a fresh chunk replaces the queue whenever it drops to the threshold
# (chunk 100 -> 90, chunk 50 -> 40 = re-plan every ~10 steps). Sync inference only managed 13 Hz on the Jetson (15 Oct).
so101-run-async() { local dur="${1:-60}" ckpt="${2:-$(so101-policy-current "" act)}" qt="${3:-90}" task="${4:-${SO101_TASK:-Pick up the lighter and hand it to the person}}"; ckpt="${ckpt:-$HOME/so101/policies/act_lighter}"; so101-policy-banner "$ckpt"; so101-cams-take
  lerobot-rollout --strategy.type=base --inference.type=rtc --inference.queue_threshold=$qt --policy.path="$ckpt" --policy.device=cuda \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="${SO101_RUN_CAMS:-$SO101_CAMS_2}" \
    --task="$task" --duration=$dur --display_data=false; so101-cams-release; }
# so101-nudge [lift_deg] : ease the follower ~18 deg off the rest pose (demos have idle starts; ACT freezes from exact rest).
so101-nudge() { so101-arms-take; python ~/so101/nudge.py "$@"; so101-arms-release; }
# so101-finetune <dataset> <checkpoint-dir> [job] [steps] : continue from a pretrained_model dir (cuda, batch 4, 1 worker; ~1.9 step/s).
so101-finetune() { local ds="$1" ckpt="$2" job="${3:-ft_$1}" steps="${4:-8000}"; mkdir -p $HOME/so101/outputs/train
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train --dataset.repo_id=lastlight/$ds --policy.path="$ckpt" --output_dir=$HOME/so101/outputs/train/$job --job_name=$job \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --steps=$steps --save_freq=2000 --batch_size=4 --num_workers=1 2>&1 | tee -a $HOME/so101/outputs/train/$job.log; }
# so101-replay-hub [repo_id] [episode] : replay a Hub dataset recorded on ANOTHER SO-10x on our follower.
# Those datasets store joints normalized -100..100, so the robot runs with use_degrees=false (maps to OUR calibration range).
# Default: lerobot/svla_so101_pickplace (50 eps, "pink lego brick into the transparent box", SmolVLA paper). Clear the mat first.
so101-replay-hub() { local ds="${1:-lerobot/svla_so101_pickplace}" ep="${2:-0}"; so101-arms-take
  lerobot-replay --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.use_degrees=false \
    --dataset.repo_id=$ds --dataset.root=$HOME/.cache/huggingface/lerobot/$ds --dataset.episode=$ep; so101-arms-release; }
# so101-dance [bpm] [seconds] [energy 0-1.3] : the arm dances to a tempo (safe pose above the mat); play the beat on the Mac: so101-dash beat <bpm>
so101-dance() { so101-arms-take; python ~/so101/dance.py --bpm "${1:-100}" --seconds "${2:-30}" --energy "${3:-1.0}"; so101-arms-release; }
# so101-run-vla [s] [ckpt] ["task"] : run a SmolVLA checkpoint (cameras renamed top->camera1, wrist->camera2 as in training), async rtc inference.
# default checkpoint: the deployed SmolVLA policy ($SO101_VLA_POLICY or DEPLOYED.json, type smolvla) when one exists, else ~/so101/policies/smolvla_lighter
so101-run-vla() { export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; local dur="${1:-60}" ckpt="${2:-$(so101-policy-current "$SO101_VLA_POLICY" smolvla)}" task="${3:-${SO101_TASK:-Pick up the lighter and hand it to the person}}"; ckpt="${ckpt:-$HOME/so101/policies/smolvla_lighter}"; so101-policy-banner "$ckpt"; so101-cams-take
  lerobot-rollout --strategy.type=base --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.queue_threshold=40 --policy.path="$ckpt" --policy.device=cuda \
    --rename_map="{\"observation.images.top\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}" \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="${SO101_RUN_CAMS:-$SO101_CAMS_2}" \
    --task="$task" --duration=$dur --display_data=false; so101-cams-release; }
# so101-scanpose [rest] : wrist camera up for the dashboard retinal-scan lock (holds torque); "rest" eases back down.
so101-scanpose() { if pgrep -f "lerobot-(teleoperate|record|rollout|calibrate|replay)|so101/(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper).py" >/dev/null; then echo "arm busy, skipping"; return 1; fi; so101-arms-take; python ~/so101/scan_pose.py "$@" 2>&1 | grep -vE "warn|Warn"; so101-arms-release; }
export SO101_SCAN_POSE="90,-30,0,-40,0,8"   # pan,lift,elbow,wrist,roll,grip for the retinal-scan pose (pan + = right, from where TC stands)
# so101-scanpose-teach : torque off, pose the arm by hand, Enter -> saved as SO101_SCAN_POSE (the retinal-scan position).
so101-scanpose-teach() { so101-arms-take; python ~/so101/scanpose_teach.py 2>&1 | grep -vE "warn|Warn"; so101-arms-release; source ~/so101/env.sh; }
# so101-calib-gripper [follower|leader] : voice-guided gripper range only (close 6 s, open 6 s); writes range_min/max in the calibration json
so101-calib-gripper() { so101-arms-take; python ~/so101/calib_gripper.py "${1:-follower}" 2>&1 | grep -vE "warn|Warn"; so101-arms-release; }
# so101-free : let go of the follower (torque off) so it can be moved by hand; stops teleop/policy first. so101-hold re-engages it where it is.
so101-free() { pkill -TERM -f "lerobot-(teleoperate|rollout|replay)"; pkill -INT -f "so101/(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper).py"; sleep 1.5; so101-arms-take; python3 ~/so101/free_arm.py; so101-arms-release; }
so101-hold() { so101-arms-take; python3 ~/so101/free_arm.py hold; so101-arms-release; }
# so101-calib : overhead-camera calibration tool (checkerboard intrinsics, ArUco table map). `so101-calib --help`
alias so101-calib='python ~/so101/cam_calib.py'
# ---- camera health (docs/camera-health.md) ----
# so101-cams-health [seconds] : all three cameras open together at the recording settings; fps, frame-interval p50/p95/max,
#   duplicates, stalls, negotiated size/fourcc, CPU load. Exit 1 on fps < 28, a stall > 250 ms or a wrong size. `--json` for machines.
so101-cams-health() { so101-cams-take; python3 ~/so101/cam_health.py --no-busy "$@"; local rc=$?; so101-cams-release; return $rc; }
# so101-cams-drift set|check : reference frames in ~/so101/calib/ref_<cam>.jpg; `check` reports the frame shift per camera and, for
#   the top camera, the ArUco markers against top_table.json (camera moved vs mat moved). Arm at rest. Exit 1 when something moved.
so101-cams-drift() { so101-cams-take; python3 ~/so101/cam_drift.py --no-busy "$@"; local rc=$?; so101-cams-release; return $rc; }
