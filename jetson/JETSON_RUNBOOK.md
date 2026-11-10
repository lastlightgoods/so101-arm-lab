# SO-101 on the Jetson Orin Nano (original dev kit) — runbook, 6 Oct 2026

Goal: the Jetson runs the arms, cameras, teleop, record and train. The Mac just SSHes in.
Target OS: JetPack 6.2.1 SD image, then apt to 6.2.2 (Ubuntu 22.04, Python 3.10, CUDA 12.6,
torch 2.11 CUDA wheels from pypi.jetson-ai-lab.io/jp6/cu126). JetPack 7 has no torch wheels
for Orin yet, so stay on 6.2.x.

Files here:  jp513-orin-nano-sd.zip (bridge, ~? GB)   jp621-orin-nano-sd.zip (11.7 GB)
             jetson_setup.sh (run on the Jetson)       env.sh (Jetson version)
Need: one microSD, 64 GB+ UHS-1 (128 GB better); the Mac's built-in SD slot; Balena Etcher.

## A. Check the firmware first (decides whether step B is needed)
1. microSD NOT inserted. DisplayPort monitor + USB keyboard, 19 V supply in.
2. Tap Esc repeatedly after the NVIDIA splash -> UEFI menu. Read the firmware version.
   36.x or newer -> skip to C.   35.x -> do B.

## B. Firmware bridge via JetPack 5.1.3 (original kits ship with 35.x)
1. Etcher: write jp513-orin-nano-sd.zip to the microSD (Etcher reads the zip directly).
2. Boot it, finish the Ubuntu wizard, connect to wifi/ethernet.
3. Wait for the "bootloader update scheduled" notification, or check:
       sudo systemctl status nv-l4t-bootloader-config     # should be inactive / success
4. sudo reboot  -> watch the firmware update on screen; it boots 5.1.3 again.
5. sudo nvbootctrl dump-slots-info                       # confirm new version
   sudo apt update && sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater
   sudo reboot   -> QSPI update runs. After it finishes, POWER OFF.
   (The 5.1.3 card will not boot any more; that is expected.)

## C. JetPack 6.2.1  (DONE 14 Oct 2026: firmware was already 36.4.3, so B was skipped)
1. Etcher is broken on this Mac; wrote the image with dd from Terminal.app (needs a tty for sudo):
       unzip jp621-orin-nano-sd.zip; diskutil list external physical; diskutil unmountDisk /dev/diskN
       sudo dd if=~/so101/jetson/jp621-orin-nano-sd.img of=/dev/rdiskN bs=4m status=progress && sync && diskutil eject /dev/diskN
   If the card shows in the reader but no diskN appears: macOS accessory security holds it "Limited" -> click Allow
   (System Settings > Privacy & Security > Allow accessories to connect).
2. Booted, Ubuntu wizard -> user `tc`, hostname `tc-nano`, a fixed LAN IP (set a DHCP reservation) (uncheck Chromium in the wizard; it hangs).
3. From the Mac:  ssh-copy-id tc@<jetson-ip>   -> ~/.ssh/config alias `jetson` (user tc) points at that IP.

## D. LeRobot on the Jetson
Already copied from the Mac (14 Oct): ~/so101/{jetson_setup.sh,env.sh,plugins/,policies/act_jar_pocket,policies/act_bottle_pick},
~/.cache/huggingface/lerobot/calibration, datasets bottle-pick-clean + jar-pocket-trim.
Run the installer (sudo prompts for the password; passwordless sudo deliberately NOT set up):
    ssh -t jetson bash ~/so101/jetson_setup.sh        # ~20-40 min: apt dist-upgrade to 6.2.2, torch CUDA, lerobot, plugins, jtop
    ssh jetson sudo reboot
Then plug in both servo boards (USB-A on the Jetson is fine) and both cameras:
    ls -l /dev/so101_*            # follower + leader symlinks from the udev rule
    ls -l /dev/v4l/by-id/         # put the two icspring ...-video-index0 paths into ~/so101/env.sh (CAM_TOP / CAM_WRIST)
env.sh is sourced automatically by ~/.bashrc on the Jetson.  Commands (same names as the Mac):
    so101-teleop | so101-record <name> "<task>" [n] | so101-record-more <ds> [n] | so101-train <ds> | so101-run [s] [ckpt]

## E. Keeping it updated and fast
    so101-update     # apt dist-upgrade (JetPack point releases come through NVIDIA's apt repo) + pip -U lerobot; reboot if told
    so101-perf       # MAXN_SUPER power mode + locked clocks for training/rollout (fan loud)
    so101-quiet      # back to 25W balanced mode (the default set by the installer)
    so101-health     # jtop: temps, clocks, GPU/CPU load, memory
Ubuntu's unattended-upgrades handles security patches automatically; run so101-update monthly for the rest.
The Mac dashboard (`so101-dash`, ~/so101/dashboard) watches the Jetson over ssh: vitals, running jobs, train.log loss curves,
live camera feeds, and an embedded terminal (ttyd -> ssh jetson) in the middle of monitor 2.
Do not jump to JetPack 7 (no torch wheels for Orin yet) — stay on 6.2.x.

## Notes
- Power: use the 19 V brick; USB-C PD from a laptop charger is not enough under load.
- Put the Jetson in MAXN / 25 W mode for training:  sudo nvpmodel -m 0 && sudo jetson_clocks
- 8 GB RAM is shared with the GPU; ACT trains fine, SmolVLA/Pi0 will not fit.
- The camera-open bus stall was a macOS/CH343 issue; it may not exist on Linux, but the
  busheal plugin is installed there too so it does no harm.
