#!/bin/bash
# Run ON THE RASPBERRY PI 5 (or via `ssh -t tc@tc-pi5.local bash ~/so101/pi5_setup.sh`) after first boot of
# Raspberry Pi OS Lite 64-bit (Trixie, Python 3.13). The Pi runs SCRIPTED arm jobs only (pick_seq, goto, dance, nudge,
# free_arm, scan_pose, calib_gripper, arm_state ...): no policies, no training, no dataset recording, no cameras.
# So this installs lerobot 0.6.1 with the feetech extra only (CPU torch 2.11, same version as the Mac and the Jetson),
# the busheal robot plugin (gripper torque/current on connect), and the same stable servo-board names as the Jetson.
# Expects ~/so101 to hold the repo's jetson/ files and plugins/ (same layout as the Jetson). Idempotent: safe to re-run.
# Adding dataset recording later (scripted_record.py): pip install 'lerobot[feetech,dataset]==0.6.1' + ffmpeg + cameras.
set -euo pipefail
echo "== $(tr -d '\0' </proc/device-tree/model 2>/dev/null) / $(. /etc/os-release; echo "$PRETTY_NAME") / python $(python3 -V | cut -d' ' -f2) =="
sudo true   # not `sudo -v`: with sudoers verifypw=all it still wants a password while %sudo has no NOPASSWD
# -- 1. OS up to date + build/runtime packages --
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip python3-dev build-essential git usbutils curl rsync htop
sudo apt-get autoremove -y
# -- 2. stable names for the two Waveshare CH343 servo boards (by USB serial), as on the Jetson --
sudo tee /etc/udev/rules.d/99-so101.rules >/dev/null <<'RULES'
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5C82107034", SYMLINK+="so101_follower", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5C82107128", SYMLINK+="so101_leader",   MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666"
RULES
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout "$USER" || true
# -- 3. python env: CPU torch 2.11 from the PyTorch CPU index, then lerobot 0.6.1 (feetech only) --
mkdir -p ~/so101 && cd ~/so101
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'lerobot[feetech]==0.6.1'
# re-pin torch in case lerobot's resolver swapped it
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
# -- 4. the busheal robot plugin from the Mac (actasync is a policy shim and lerobot-camera-avf is macOS-only: both skipped) --
[ -d ~/so101/plugins/lerobot-robot-busheal ] && pip install -e ~/so101/plugins/lerobot-robot-busheal
python - <<'PY'
import torch; print("torch", torch.__version__, "threads", torch.get_num_threads())
import lerobot, lerobot.motors.feetech, lerobot.robots.so_follower; print("lerobot", lerobot.__version__, "feetech + so_follower import ok")
PY
mkdir -p ~/.cache/huggingface/lerobot/calibration
# -- 5. env.sh: the Jetson's (FOLLOWER/LEADER/ids and the so101-* job wrappers), with --policy.device=cpu --
[ -f ~/so101/env.sh ] && sed -i 's/--policy\.device=cuda/--policy.device=cpu/g' ~/so101/env.sh
grep -q 'so101/env.sh' ~/.bashrc || echo '[ -f ~/so101/env.sh ] && source ~/so101/env.sh' >> ~/.bashrc
echo "== devices =="; ls -l /dev/so101_* 2>/dev/null || echo "servo boards not plugged in yet"
# -- 6. passwordless sudo was only for first-boot setup (the Jetson deliberately has none); remove it --
sudo rm -f /etc/sudoers.d/90-tc-setup
echo "DONE. Reboot once (new kernel/firmware + dialout group). sudo now asks for tc's password."
