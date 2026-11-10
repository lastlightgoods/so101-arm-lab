# SO-ARM101 on this Mac — lerobot 0.6.1, Python 3.12, venv ~/so101/.venv

    source ~/so101/.venv/bin/activate

Port mapping (confirmed by wiggle test 5 Oct 2026; both must be on USB-A-to-C):
    FOLLOWER = /dev/tty.usbmodem5C821070341   (board serial 5C82107034)
    LEADER   = /dev/tty.usbmodem5C821071281   (board serial 5C82107128)
    `source ~/so101/env.sh` sets $FOLLOWER/$LEADER and aliases
    so101-calib-follower, so101-calib-leader, so101-teleop.
Both arms: ids 1-6, STS3215 fw 3.15, 1 Mbaud. Motor setup (step 2) is DONE.

## 0. Status 5 Oct 2026 (evening)
- Board 5C82107128 (/dev/tty.usbmodem5C821071281): all 6 motors answer, ids 1-6,
  STS3215 fw 3.15, 1 Mbaud. LeRobot bus.connect() works. Motor setup NOT needed.
- Board 5C82107034 dropped off USB after switching to a USB-C-to-C cable.
  Waveshare boards often lack CC pull-downs: use USB-A-to-C for it.
- FeetechMotorsBus.scan_port() returns {} on these boards (broadcast ping
  collides). Not a fault; connect()/ping per id work. Use tools/ping_arm.py instead.

## 0b. Old checklist if a board stops answering
- Servo power brick plugged into each board (follower 12 V, leader 5-7.4 V per kit).
- 3-pin cable from motor 1 (shoulder pan) into the board.
- Waveshare board: both jumpers on channel B (USB).
- Rescan:
    python -c "from lerobot.motors.feetech import FeetechMotorsBus as F; \
      [print(p, F.scan_port(p)) for p in ['/dev/tty.usbmodem5C821070341','/dev/tty.usbmodem5C821071281']]"
  Expect {1000000: [1,2,3,4,5,6]} on each port.

## 1. Which port is which
    lerobot-find-port        # unplug one USB when asked

## 2. Motor IDs (ONLY if scan shows ids != 1..6, e.g. a self-built kit)
    lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER>
    lerobot-setup-motors --teleop.type=so101_leader --teleop.port=<LEADER>

## 3. Calibrate (once; files land in ~/.cache/huggingface/lerobot/calibration)
    lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER> --robot.id=ll_follower
    lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER>  --teleop.id=ll_leader
  Middle every joint, Enter, then sweep each joint to both limits.

## 4. Teleoperate
    lerobot-teleoperate \
      --robot.type=so101_follower --robot.port=<FOLLOWER> --robot.id=ll_follower \
      --teleop.type=so101_leader  --teleop.port=<LEADER>  --teleop.id=ll_leader

## 5. Cameras (two icSpring UVC cams are on the hub; one is the CNC cam)
    lerobot-find-cameras opencv
  then add to teleop/record:
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --display_data=true
Training on this M1 Max: --policy.device=mps

## Camera wall on monitor 2
    ~/so101/tools/cams.sh start   # cams_avf.py: selects icSpring cams by AVFoundation device object, never by index
    ~/so101/tools/cams.sh stop
    ~/so101/tools/cams.sh start   # cams_avf.py: selects icSpring cams by AVFoundation device object, never by index
Monitor 2 is 2560x1440 at CoreGraphics x=1512,y=-75 (hardcoded in cams.py).

## ROOT CAUSE of bus dropouts (found 5 Oct 2026 22:20)
Both servo boards AND both icSpring cameras hang off one USB 2.0 hub behind the
Anker dock (ioreg: USB2.1 Hub > USB2.0 HUB > {2x USB Single Serial, 2x icspring}).
Opening a camera stream (cams.py, LightBurn, Chrome, lerobot-find-cameras...)
stalls the CH343 serial device within ~0.3 s and it stays dead until the port is
closed and reopened (flush does not help). Reproduced 2/2 with a 60 Hz read test.
FIX: put the servo boards on a different USB controller than the cameras/dock
(a different port on the MacBook). Verify with:
    ioreg -p IOUSB -w0 | grep -E "XHCI|Serial|icspring|HUB|Hub"
  -> the two "USB Single Serial" lines must sit under a different AppleT6000USBXHCI@
     than the icspring cameras.
WORKAROUND meanwhile: `so101-teleop-robust` (tools/teleop_robust.py) reopens a stalled bus
instead of exiting. Calibration still needs the cams closed (tools/cams.sh stop).

## Cameras for lerobot-record: `type: avf` (plugin ~/so101/plugins/lerobot-camera-avf, pip -e installed)
Selects cameras by NAME + native format via AVFoundation. Never by index. FaceTime/iPhone refused.
    python -m lerobot_camera_avf      # lists devices without opening any
    --robot.cameras="{top: {type: avf, match: icspring, native: 1920x1080, width: 640, height: 360, fps: 30}, wrist: {type: avf, match: icspring, native: 640x480, width: 640, height: 480, fps: 30}}"
lerobot-record/teleoperate/calibrate auto-import it (package name prefix lerobot_camera_).

## Bus self-heal plugin: ~/so101/plugins/lerobot-robot-busheal (pip -e installed)
Auto-imported by lerobot-record/teleoperate/calibrate/replay. Wraps FeetechMotorsBus
sync_read/sync_write/read/write: on ConnectionError ("no status packet") it closes and
reopens the port, pings a motor, retries once. Logs "[busheal] ... reopened" at WARNING.
Disable: LEROBOT_BUSHEAL=0.  Give-up time: LEROBOT_BUSHEAL_MAX_S (default 10 s).
The stock `so101-teleop` is now as robust as `so101-teleop-robust`.

## Gripper "quits mid-grasp" = STS3215 overload trip (solved 6 Oct 2026)
Symptom: follower gripper goes limp/weak a couple of seconds into every grasp; later every
packet to id 6 answers "Overload error" and disconnect fails. Cause: the STS3215 trips overload
when it sits stalled at its torque limit for Protection_Time (2 s), at ANY limit value, and
LeRobot's configure() sets gripper Max_Torque_Limit=500 whose stall current (541) also exceeds
Protection_Current=250. Measured stall current vs limit: 500->541 400->370 300->240 250->187.
Fix (busheal plugin, applied after LeRobot's configure on every connect):
  Max_Torque_Limit=Torque_Limit=250, Protective_Torque=0 (disables the stall timer),
  Protection_Time=254. Current guard (250) and 70 C temperature guard stay on.
  Env: LEROBOT_GRIPPER_TORQUE=250. Pull test: holds the taped bottle firmly.
The plugin also clears a tripped overload flag (parks goal at present position, cycles torque)
and logs "[busheal] OVERLOAD ... cleared".
Streaming encoding: without --dataset.streaming_encoding=true the teleop loop freezes ~12 s
after each episode while video encodes (follower appears locked). Now on in so101-record.

## Datasets & training (7 Oct 2026)
- lastlight/bottle-pick_20261006_233211: 50 raw episodes (jars into the 4-pocket tray; task string
  still says "bottle"/"bin"). 10 dropped (0,8,9,10,12,13,14,27,38,39: hand in frame or jar still in
  the gripper at the 25 s cutoff) -> lastlight/bottle-pick-clean: 40 episodes, 28,288 frames.
- Training: ACT, mps, 50k steps, batch 8, ckpt every 5k -> ~/so101/outputs/train/act_bottle_pick
  log /tmp/so101_train.log. Resume:  lerobot-train --config_path=outputs/train/act_bottle_pick/checkpoints/last/pretrained_model/train_config.json --resume=true
- Run the policy:  lerobot-rollout --strategy.type=base --policy.path=outputs/train/act_bottle_pick/checkpoints/last/pretrained_model \
    --policy.device=mps --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=$FOLLOWER_ID --robot.cameras="$SO101_CAMS" \
    --task="Pick up the bottle and place it in the bin" --duration=60
- Next session: record with key presses from your own terminal (so101-record-more), reset 12 s.

## Running the policy (7 Oct 2026)
`so101-run [seconds]` = lerobot-rollout with --inference.type=rtc (async background inference; needs the
lerobot_policy_actasync shim in ~/so101/plugins so ACT accepts the rtc call shape). Loop holds 30 Hz with
no slow-step warnings. Sync mode: normal step 5 ms, chunk refresh +29 ms (2x ResNet18 7.5 ms, transformer
14 ms) -> one 40 ms frame per chunk. fp16 on mps is slower (85 ms); CPU 103 ms.
Behaviour (50k ACT, 40 eps): reaches then hovers. Likely scene mismatch: training frames always show the
leader arm + hand in the overhead view. Next data session: leader out of the top camera's view, hands out
of frame during episodes, record with key presses (so101-record-more), then retrain (100k steps).
Gripper speed: torque cap is a PWM cap so it also sets speed. 250 lagged the trigger during recording;
env.sh now sets LEROBOT_GRIPPER_TORQUE=400, LEROBOT_GRIPPER_CURRENT=540 (stall current at 400 ~370).

## jar-pocket dataset (7 Oct 2026) and second training run (8 Oct)
- lastlight/jar-pocket_20261007_110305: 22 raw episodes, all hit the 40 s ceiling (key presses did not
  register), median 11 s idle tail. Overhead camera now framed tight on the tray; leader base visible
  bottom-left in every frame (keep it there at rollout).
- tools/trim_dataset.py -> lastlight/jar-pocket-trim: 18 episodes, 16,295 frames; each cut 1 s after the arm's
  last motion; dropped 5, 7, 12 (unfinished at cutoff) and 21 (nothing happened). Script needs the
  __main__ guard (macOS spawn re-imports the script for the video encoder).
- Training: ACT 40k steps, mps -> outputs/train/act_jar_pocket, log /tmp/so101_train2.log.
  Run it:  so101-run 60 ~/so101/outputs/train/act_jar_pocket/checkpoints/last/pretrained_model
- Gripper now: torque 500, current guard 700, P 32 (env.sh).

## Lab dashboard (14 Oct 2026) — ~/so101/dashboard
    so101-dash          # start server + Chrome kiosk on monitor 2 (portrait 1440x2560 auto-detected)
    so101-dash stop     # close both       so101-dash open -> server only, http://127.0.0.1:8765
Shows: mission status (which lerobot process runs where), Jetson vitals via ssh tegrastats (GPU/CPU/RAM/temp/W, power mode),
rotating hero frames from the datasets (img/, regenerate with `python dashboard/extract_frames.py [dataset]`), ACT loss curves
from outputs/train/<job>/train.log on Mac + Jetson (so101-train now tees one), dataset + policy tables, event log (events.log).
No cameras are opened by the dashboard. Server: dashboard/server.py (stdlib + psutil); page: dashboard/index.html; log /tmp/so101_dash.log.
Gotcha fixed: HTTPServer's getfqdn() stalls 35 s on this Mac -> server_bind overridden.
Dashboard additions (14 Oct, evening): LIVE FEEDS from the Jetson cameras (ssh+ffmpeg MJPEG, polled as single frames by the page;
the Jetson side yields the cameras while ~/so101/.cams_busy exists -> so101-record/so101-run on the Jetson set/clear it) and an
embedded JETSON TERMINAL in the middle (ttyd on 127.0.0.1:8766 running `ssh -t jetson`; click into it on monitor 2 and type).
Jetson python: Python 3.10 + lerobot 0.6.1 via jetson/py310_compat.py (rewrites PEP 695 syntax + typing.Self imports) and
jetson/so101_nvidia_preload.py (.pth preload of libcudss/libcusparseLt, RTLD_LOCAL — RTLD_GLOBAL crashed cv2). Servos 1-6 on both
arms answer from the Jetson (/dev/so101_follower, /dev/so101_leader). Training on both machines at once = two separate runs
(no cross-device DDP across MPS+CUDA); the dashboard overlays both loss curves.
ARM panel (14 Oct late): joint table + virtual arm (follower solid, leader orange ghost). Data path: Jetson `~/so101/arm_state.py`
streamed over ssh by the dashboard server. Idle -> it reads both arms itself (read-only, 15 Hz). While a lerobot process runs
(or ~/so101/.arms_busy exists, set by the so101-* functions) it releases the ports and forwards /dev/shm/so101_<role>.json,
which the busheal plugin now writes on every Present_Position read. Kinematic conventions live in index.html `ARM` (link mm,
sign/offset per joint) — tune there if the ghost disagrees with the real arm.
3D arm (14 Oct late): the ARM panel renders the real SO-101 as a wireframe with three.js + urdf-loader (CDN importmap in index.html).
Model = TheRobotStudio/SO-ARM100 `so101_new_calib.urdf` + 13 STL meshes, downloaded into dashboard/urdf/ (16 MB). Its joint
zero equals LeRobot's calibrated zero, so degrees -> radians go straight in (gripper % -> URDF -0.17..1.75 rad). Follower green,
leader orange ghost, camera orbits slowly. Headless screenshots need --use-angle=swiftshader to render WebGL.
Voice (15 Oct): lerobot's say() on the Jetson is patched by the busheal plugin to append to /dev/shm/so101_say.jsonl; arm_state.py
forwards it; the Mac dashboard server speaks it (macOS `say`) and logs it. Jetson record functions run with --play_sounds=true.
LEROBOT_LOCAL_SAY=1 also keeps spd-say on the Jetson. Scripted recorder (jetson/scripted_record.py, so101-mark) is parked/untested.

## Lighter handoff (15 Oct 2026)
Recorded 30 teleop episodes on the Jetson -> `lighter-handoff_20261014_234858` (raw, Mac + Jetson). Reviewed contact sheets:
kept 16 -> `lighter-handoff-clean` (7,349 frames; built with ~/so101/tools/clean_dataset.py, now the general drop+trim tool).
Dropped 3,4,6,7,8,14,15,17,18,20,21,23,24,25: every grasp with the lighter INSIDE the wooden tray failed, far-right-edge spots
failed, and 4/17/21 handed off on the mat instead of at the top edge. Jetson record loop ran ~23 Hz (462 frames per 20 s
episode) while the dataset is stamped 30 fps -> playback/policy timing is ~1.3x compressed; consistent between record and
rollout on the Jetson, but keep in mind. Next: record ~20 more (mid-mat spots, hand at the top edge, no tray), then train.
Jetson training gotchas (15 Oct): pip's nvidia-cublas-cu12/nvidia-cuda-nvrtc-cu12 (pulled in by cudss/cusparselt) shadow the
Jetson's own cuBLAS -> CUBLAS_STATUS_ALLOC_FAILED; uninstalled (now in jetson_setup.sh). An "NVML_SUCCESS INTERNAL ASSERT" in
CUDACachingAllocator on the Jetson = out of memory (8 GB shared): use --batch_size=4 --num_workers=1 and pause the dashboard
camera streams (touch ~/so101/.cams_busy) while training. Second run act_lighter_c50 (chunk_size 50) runs on the Jetson in
parallel with the Mac's act_lighter (chunk 100) as an A/B. Dashboard theme is now "hi-fi" (amber/cyan, Michroma/Rajdhani).

## 15 Oct 2026 evening: first lighter-handoff tests on the Jetson
- Both 25k ACT runs finished: Mac `act_lighter` (chunk 100) final loss 0.001, Jetson `act_lighter_c50` (chunk 50) 0.002.
  Both staged on the Jetson in `~/so101/policies/act_lighter` and `~/so101/policies/act_lighter_c50`.
- `so101-run` (sync inference, temporal ensemble) only reaches **13 Hz on the Jetson** (30 Hz target) -> arm moves at
  <half speed; chunk-100 model grabbed the lighter but it slipped, chunk-50 model missed and froze.
- Fix: `so101-run-async [s] [ckpt] [queue_threshold]` on the Jetson = `--inference.type=rtc` + actasync shim.
  Holds 30 Hz, inference latency 4-5 steps (~150 ms). Threshold 90 for chunk 100, 40 for chunk 50.
  Rollout logs: `~/so101/outputs/run/*.log`.
- **Root cause of the freeze (15 Oct 18:30):** 13 of 16 clean episodes start with 1-5.6 s idle at the rest pose (waiting for
  the "start" voice cue). From the rest pose the ACT model therefore predicts "stay" and never moves (chunk 100 = 3.3 s).
  Even after `so101-nudge` (jetson/nudge.py, eases the arm ~18 deg off rest, holds torque) it returns to rest.
  Fix: `tools/clean_dataset.py --head 0.4` (new option, cuts everything earlier than 0.4 s before the leader first moves > 3 deg)
  -> `lighter-handoff-clean2`, then `so101-finetune lighter-handoff-clean2 outputs/train/act_lighter/checkpoints/last/pretrained_model`.
  When recording more demos: move the leader immediately after the start cue, no pause.
- 16 Oct 19:10: dashboard header is now a turntable (spinning vinyl + tonearm drops on the record whenever a lerobot
  process runs; REC lamp blinks during record; NOW PLAYING line). Compact layout tightened to fit 2560 px with the live
  feed visible (~2519 px). Recording session 16 Oct 18:59 added 2 episodes (16,17) to lighter-handoff-clean on the Jetson
  before the user said stop. ft_lighter (8k fine-tune on clean2) finished 15 Oct 20:02 and is staged at
  jetson:~/so101/policies/ft_lighter — NOT yet tested: `so101-nudge; so101-run-async 60 ~/so101/policies/ft_lighter 90`.
- 16 Oct 21:45: SmolVLA fine-tune submitted to HF Jobs (user tclastlight, a100-large $2.50/h, 10k steps batch 32,
  timeout 4h): job 6ad2ff91c66288145226580c (attempt 1 failed: dataset repo needs a v3.0 git tag; attempt 2 failed: smolvla_base wants observation.images.camera1..3 -> use --rename_map top->camera1, wrist->camera2 and --policy.empty_cameras=1; same rename needed at rollout) -> model repo tclastlight/smolvla_lighter_2026-10-17_04-54-40 (private),
  dataset tclastlight/lighter-handoff-clean2 (private, 16 eps, idle-trimmed). Monitor: `hf jobs logs <id>`.
  The 2 episodes recorded 16 Oct 18:59 (eps 16,17) were killed mid-write: data/chunk-000/file-001.parquet is truncated on
  Jetson + Mac copies of lighter-handoff-clean (meta says 18 eps). Repair or re-record before using that set again.
- Demo: `so101-replay-hub [repo] [ep]` on the Jetson replays Hub SO-10x datasets (joints normalized -100..100 -> robot
  runs with use_degrees=false). lerobot/svla_so101_pickplace (50 eps, lego brick into box) is cached on both machines
  (meta+data only; episode videos download on demand).
- Fun: `so101-dance [bpm] [seconds] [energy]` on the Jetson (jetson/dance.py, sinusoids around a safe pose, chomps on 2 and 4)
  + `so101-dash beat <bpm>` on the Mac (dashboard/beat.py synthesizes a kick/snare/hat loop, afplay). Dashboard deck shows
  DANCE / DEMO_WAVE / NUDGE / REPLAY as activity now (server.py probe).
- SmolVLA job attempt 3 (6ad2ff91c66288145226580c) is TRAINING as of 16 Oct 21:56 PDT (10k steps, batch 32, A100).
- 16 Oct 22:20: dashboard center strip (was the Ken Burns gallery) is now a self-playing Tempest web (canvas, index.html
  `startTempest`; arrow keys + space play it when the kiosk has focus; spawns faster while a lerobot process runs).
  Gallery code preserved in dashboard/index.pre_tempest.html.
- Checkpoints trained in the cloud image (newer lerobot) carry config keys our 0.6.1 doesn't know (`dtype`): rollout fails
  with "The fields `dtype` are not valid for SmolVLAConfig". Fix applied to policies/smolvla_lighter_2500/config.json on
  the Jetson (original kept as config.orig.json); repeat for every new checkpoint (strip keys not in the dataclass).
  Run: `so101-nudge; so101-run-vla 60 ~/so101/policies/smolvla_lighter_2500`. Dashboard: ARM panel mode buttons
  (POST /api/arm/<teleop|policy|vla|dance|wave|replay|stop>), Spotify on the deck (POST /api/spotify/<playpause|next|previous>).
- SmolVLA on the Jetson (16 Oct 22:50): needs `pip install 'lerobot[smolvla]'` extras (transformers 5.5.4, num2words),
  `jetson/prep_vla_ckpt.py <pretrained_model>` on every checkpoint (drops `dtype`, device=cpu, load_vlm_weights=false),
  and a one-line patch in site-packages lerobot/policies/pretrained.py `_load_as_safetensor` -> device="cpu"
  (loading straight to cuda OOMs = NVML_SUCCESS assert; original kept as pretrained.py.orig; REAPPLY after any lerobot
  reinstall). Standalone load then takes ~55 s and 1.2 GB GPU. Desktop (gnome-shell+Xorg+gnome-software) holds ~0.7 GB:
  `sudo systemctl set-default multi-user.target && sudo reboot` would free it (user decision).
- SmolVLA memory on the Jetson, final recipe (16 Oct 23:05): config via prep_vla_ckpt.py (load_vlm_weights=false), pretrained.py
  stages weights on CPU, and smolvlm_with_expert.py builds the VLM in bf16 when not loading Hub weights (the standard
  path is bf16 too). Result: 0.93 GB GPU, ~45 s load. Both patches are in jetson_setup.sh. policies/smolvla_lighter_5000 copied.
- 16 Oct 23:20 Retinal-scan lock: dashboard starts LOCKED (SO101_LOCK=0 to disable); the server watches the WRIST camera
  for a close-up iris (HoughCircles dark disc + bright surround; OpenCV 5 has no Haar cascades), fills a progress ring,
  says "Identity confirmed" and unlocks. Theater, not security (any eye). LOCK button under the clock; hold U 2 s to
  override. On lock the arm takes `so101-scanpose` (wrist cam up, SO101_SCAN_POSE to tune), on unlock `so101-scanpose rest`.
- !! Follower bus is flaky tonight: LeRobot handshake intermittently reports motor 6 (gripper) / 2 missing; raw pings show
  garbled model numbers; leader port threw "device reports readiness to read but returned no data". Likely a loose
  servo cable or USB after the long teleop session. nudge/dance/scan_pose now retry connect 4x. Reseat cables if it persists.
- Cloud SmolVLA job 6ad2ff91 COMPLETED (10k steps, 61 min, ~$2.55): final model in tclastlight/smolvla_lighter_2026-10-17_04-54-40
  (also tags 002500/005000/007500). 5000 is prepped on the Jetson; the final one still needs download + prep_vla_ckpt + copy.
- 18 Oct: dashboard arm row = 3/5 tall ARM stage (table, 24x18" mat with grid + centre squares, TOP CAM frustum + ray,
  wrist-cam frustum on the gripper link, HANDOFF zone) + 2/5 column with loss / datasets / policies (fixed 922 px row in
  compact mode, page = 2560 exactly). urdf-loader re-applies URDF materials after load (lit -> black): frame() re-wireifies.
  Kiosk Chrome now runs with --remote-debugging-port=9333; `node dashboard/cdp.js shot out.png` / `eval "<js>"` screenshots
  and queries the REAL kiosk page (use this instead of the flaky Chrome extension / headless Chrome).
- 18 Oct: 3D follower is now solid (MeshStandardMaterial warm white + amber edges, hemisphere + 2 directional lights,
  STL normals computed); leader ghost hidden unless teleop/record runs or the LEADER button is on.
- 18 Oct: wrist-cam frustum pose derived from TheRobotStudio Optional/Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl (mm) vs
  the URDF's wrist_roll_follower_so101_v1.stl (m): camera plate centroid (2.3, 72, -4) mm in the STL frame, normal
  (0,-0.4,0.9) -> gripper_link frame (visual rpy -pi flips y,z): position (0.0023,-0.069,-0.0022) m, optical axis
  (0,0.4,-0.9), ~24 deg toward the fingertips. Live wrist frame agrees (fingertip enters bottom-centre). STLs in /tmp/so101_mount.
- 18 Oct (later): user says the wrist cam is side-mounted; frustum moved to gripper_link (-0.046,-0.022,-0.012), axis
  (-0.28,-0.3,0.91) (static-jaw side, looking down the fingers and slightly inward). Flip x if it is mirrored.

## 19 Oct 2026: template + simulation
- Template repo: github.com/lastlightgoods/robot-lab-template (private, GitHub template): machine.toml + bin/lab.py
  (record/clean/train/finetune/run/prep-ckpt/push-dataset), tools, edge/trainer env examples, plugins, dashboard, PLAYBOOK.md.
- Simulation: `pip install mujoco so101-nexus` (Apache-2.0, MuJoCo Menagerie robotstudio_so101) + clone of
  github.com/aviadarn/so101-lerobot in sim/ (collect_sim.py scripted IK expert -> LeRobot v3 dataset in so101_follower
  units: degrees + gripper 0-100, cameras `wrist` + `overhead` 320x240, 50 Hz; eval_sim.py = success on unseen seeds).
  Dry run: 5/5 expert successes, ~3 s/episode on the Mac (MUJOCO_GL=glfw). Commands: so101-sim-collect / -train / -eval.
  Their best: ACT 600 demos, 20k steps -> 90% in sim. Generating so101_sim_pick_v1 (300 eps) now.
- Sim results v1 (300 demos, ACT 20k on the Mac, 3.8 step/s): success on 50 unseen layouts = 20% @10k, 20% @15k, 30% @20k
  (loss 0.002; failures mostly never grasp). Matches the wrapper author's v3 curve (300 demos: 42% @40k, 68% @60k).
  Their recipe to 88-92%: 600 demos + warm start from the best checkpoint, 20k steps, 320x240, tighter overhead crop.
  Running now: so101_sim_pick_v2 (600 eps, seed 11) -> warm start from act_sim_pick_v1/020000 -> act_sim_pick_v2 (20k)
  -> auto-eval 10k/15k/20k into sim/eval_v2.log (GIFs in sim/rollouts/v2_*). Note: running evals while collecting
  slows collection ~5x (CPU contention).
- 21 Oct 2026: Jetson moved to a new LAN address (ssh config + dashboard updated; `ssh-keygen -R` was needed after the move).
- Sim results v2 (600 demos, warm start from v1/20k): 76% @10k, 52% @15k, 56% @20k on 50 unseen layouts -> best =
  act_sim_pick_v2/checkpoints/010000, copied to jetson:~/so101/policies/act_sim_pick_v2_10k for a sim-to-real try
  (needs a ~25 mm red cube and a blue disc target on the mat; cameras named `wrist` + `overhead` in the sim set, our real
  set uses `top` -> use --rename_map at rollout). Success peaks before the end of training: pick by rollout, not loss.
- 21 Oct: dashboard SIM view (Tempest strip -> rollout GIFs + eval table + status; /api/sim) and the 3D stage follows the
  simulator: sim/so101_simstate.py (installed as .venv site-packages/so101_simstate.py + so101_simstate.pth) wraps the
  so101_nexus env step/reset and writes /tmp/so101_sim_state.json (joints in dataset units, obj/target/tcp, success);
  server adds `sim_arm` to /api/state; the stage shows SIMULATION, the red cube, blue disc and tool point while fresh.
  Reinstall: cp sim/so101_simstate.py .venv/lib/python3.12/site-packages/ && echo "import so101_simstate" > .../so101_simstate.pth
- Failure analysis of v2/10k (sim/ANALYSIS_v2_10k.md): 11/12 failures never grasp (descent lands 1-3 cm off, closes on air,
  "phantom-carries"); 1 drop on the longest carry; far band (r >= 0.29 m) fails 38% vs 21%; a 600-step budget rescues none.
  Fix implemented in sim/collect_v3.py: DART hover noise 0.012 m, tight grasp waypoint, 30% deliberate 1-3 cm first-descent
  misses + re-grasp recovery (<= 2). Dry run: 16/23 kept, recoveries in 8 of 23. Generating so101_sim_pick_v3 (600 eps) ->
  warm start from v2/10k, 15k steps on the Mac -> eval every 2.5k (sim/eval_v3m.log).
- Inference tuning of v2/10k (sim/INFERENCE_TUNING.md): default full-chunk open loop 76%; n_action_steps 50 -> 58%, 25 -> 58%,
  10 -> 16%, temporal ensemble 0.01 -> 56%. Re-planning more often HURTS this ACT policy. Our Jetson so101-run-async uses
  queue_threshold 90 (re-plan every ~10 steps = the 16% regime): for sim-trained ACT use --inference.queue_threshold=10
  --inference.rtc.execution_horizon=10; A/B the lighter model the same way before assuming the opposite.
- Round 3 analysis (sim/ANALYSIS_v3m.md): recoveries transferred (7 baseline failures rescued) but 12 new failures: policy
  re-opens good grasps (imitates `open_r`) and places just outside 4 cm; only 73/600 v3 episodes contain a recovery while
  ~18% teach "off-centre is fine". Round 4 = v2 (600) + the 73 recovery episodes (sim/v3_recovery_episodes.json, two
  gripper closes), built by sim/build_v4.py (delete_episodes + merge_datasets) -> so101_sim_pick_v4; warm start from v2/10k,
  10k steps, eval every 2.5k (sim/eval_v4.log). MultiLeRobotDataset is deactivated in 0.6.1, hence the merge.
- Round 4 (v2 + 73 recovery eps = 673, warm start v2/10k): 48% @2.5k, 72% @5k, 70% @7.5k, **78% @10k** (seed 9100) -> best so far,
  still rising; continuing 10k more (act_sim_pick_v4b, eval every 2.5k) and confirming v4/10k vs v2/10k on seed 9200.
  Cloud: long-from-scratch 44/56/46/52/54/62% at 5k..30k (worse than warm starts); cloud v3 warm start 72% @2.5k so far.
- Second seed set (9200): v4/10k 72% vs v2/10k 64% -> pooled over 100 layouts v4 = 75%, v2 = 70%. The recovery minority
  is a real +5. v4/10k is the sim-to-real candidate (copying to jetson:~/so101/policies/act_sim_pick_v4_10k).
- Cloud rounds (sim/CLOUD_ROUNDS.md, ~$10.9): warm start at batch 32 degraded (72% @2.5k then 58-68%); from scratch
  reached 62% @30k when the 3 h cap hit. Neither beat the Mac's 76%/78%. Lesson: warm starts at batch 32 need a lower LR;
  the Mac's batch-8 warm starts win. jetson/prep_ckpt.py (generalized: drops config keys unknown to the local policy
  dataclass, any policy type) now exists; sim/cloud_eval_driver.py scores downloaded checkpoints.
- Round 4 continuation (act_sim_pick_v4b = v4 steps 10k-20k): 58/62/60/**90%** at 12.5k/15k/17.5k/20k (seed 9100).
  Confirming v4b/010000 on seed 9200; copied to jetson:~/so101/policies/act_sim_pick_v4_20k.
- CONFIRMED: v4b/010000 (= round 4 step 20k) 90% (seed 9100) + 78% (seed 9200) = **84% pooled over 100 unseen layouts**.
  Best sim policy. On the Jetson as policies/act_sim_pick_v4_20k. Sim-to-real next: red ~25 mm cube + blue 5 cm disc on the
  mat, `--rename_map '{"observation.images.top":"observation.images.overhead"}'`, open-loop chunks (queue_threshold 10).
- 28 Oct: wrist camera is on the arm's LEFT as you face it from the front of the mat (user): dashboard frustum at
  gripper_link (-0.046,-0.022,-0.012), axis (-0.28,-0.3,0.91); props/camera_angles.html matches. `so101-calib-gripper
  follower|leader` = voice-guided gripper range (close 6 s / open 6 s); first run saw no movement, nothing changed.
- 28 Oct cameras: overhead is now a Logitech C920 (serial 9D46D6AF, MJPG 640x360; frame is rotated ~30° to the mat, square it
  up), a second C920 (B6A1F66F) faces the arm from the far side at table height (recommend raising to ~400 mm / 45° down),
  wrist icSpring unchanged. env.sh: CAM_TOP/CAM_FRONT/CAM_WRIST, SO101_CAMS = 3 cams (top, front, wrist) for recording,
  SO101_CAMS_2 = top+wrist for pre-existing policies (so101-run-* default to it; SO101_RUN_CAMS overrides). Dashboard shows
  three feeds and a FRONT CAM frustum. Both C920s stream 30 fps together. Dashboard no longer locks on server restart
  (SO101_LOCK=1 to opt in); the auto-lock had been raising the arm on every restart.
- 29 Oct: grippers recalibrated with so101-calib-gripper (voice relay key is "say"; CALIB_HOLD=10 for longer windows):
  follower 1395-2865 (was 1413-2910), leader 1384-2622 (was 1372-2629). Fingers move freely with torque off.
- Teleop broke after the gripper recalibration: the JSON no longer matched the servo EEPROM (Min/Max limits, homing), so
  SOLeader.connect() -> calibrate() -> input() -> EOFError when launched without a TTY. Fixed with bus.write_calibration()
  on both arms; calib_gripper.py now does that itself. Smoke test: both arms connect. At disconnect the follower logged
  "Failed to write Torque_Enable on id 2" (shoulder_lift) once: that servo's cable is the one to reseat if it recurs.
