#!/bin/bash
# Run ON THE JETSON (or via `ssh -t jetson bash ~/so101/jetson_setup.sh`) after JetPack 6.2.x first boot.
# Installs LeRobot + torch(CUDA) for the SO-101 arms, stable device names, plugins, monitoring, power mode.
# sudo will ask for your password.  Idempotent: safe to re-run after a failure.
set -euo pipefail
echo "== JetPack / L4T =="; head -1 /etc/nv_tegra_release || true
sudo -v
# -- 1. bring the OS and JetPack fully up to date (6.2.1 -> 6.2.2+) --
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo apt-get install -y python3.10-venv python3-pip git ffmpeg libgl1 v4l-utils usbutils curl rsync htop
sudo apt-get autoremove -y
# -- 2. stable names for the two Waveshare CH343 servo boards (by USB serial) --
sudo tee /etc/udev/rules.d/99-so101.rules >/dev/null <<'RULES'
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5C82107034", SYMLINK+="so101_follower", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5C82107128", SYMLINK+="so101_leader",   MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666"
RULES
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout,video "$USER" || true
# -- 3. monitoring (jtop) and power mode: 25W balanced by default; `so101-perf` switches to MAXN_SUPER + locked clocks --
sudo pip3 install -U jetson-stats
echo n | sudo nvpmodel -m 1 >/dev/null 2>&1 || true
sudo nvpmodel -q | head -2 || true
# -- 4. python env (Python 3.10: NVIDIA ships CUDA torch wheels only for cp310, so lerobot 0.6.1 is installed with a 3.10 syntax patch) --
mkdir -p ~/so101 && cd ~/so101
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
# CUDA torch for JetPack 6 (CUDA 12.6) from NVIDIA's Jetson index; same 2.11 as the Mac. cudss/cusparselt are runtime deps of that wheel.
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.10.0 torchcodec==0.10.0 nvidia-cudss-cu12 nvidia-cusparselt-cu12 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
# cudss/cusparselt drag in pip's x86/SBSA cuBLAS + nvrtc, which torch then preloads instead of the Jetson's own -> CUBLAS_STATUS_ALLOC_FAILED. Remove them.
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 || true
# preload those two libs at interpreter start (torch's loader does not find them in site-packages/nvidia on its own)
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
cp ~/so101/so101_nvidia_preload.py "$SP/" && echo "import so101_nvidia_preload" > "$SP/so101_nvidia_preload.pth"
pip install --ignore-requires-python 'lerobot[feetech,dataset,core-scripts,training]==0.6.1'
# re-pin torch in case lerobot's resolver swapped it for a CPU wheel
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
python ~/so101/py310_compat.py
# SmolVLA on the 8 GB Jetson: extras + stage weights on CPU when loading (loading straight to cuda OOMs; see NEXT_STEPS)
pip install --ignore-requires-python 'lerobot[smolvla]==0.6.1' 'num2words<0.6.0,>=0.5.14' 2>&1 | tail -1
python - <<'PYP'
p=next(__import__("glob").iglob("$HOME/so101/.venv/lib/python3.*/site-packages/lerobot/policies/pretrained.py".replace("$HOME",__import__("os").path.expanduser("~"))))
s=open(p).read(); a="model, model_file, strict=strict, device=resolve_safetensors_device(map_location)"
if a in s: open(p,"w").write(s.replace(a,'model, model_file, strict=strict, device="cpu"  # so101: stage weights on CPU')); print("pretrained.py patched")
PYP
python - <<'PYP'
import glob, os
p=next(glob.iglob(os.path.expanduser("~/so101/.venv/lib/python3.*/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py")))
s=open(p).read(); a="            self.vlm = SmolVLMForConditionalGeneration(config=config)\n"
if a in s: open(p,"w").write(s.replace(a,"            self.vlm = SmolVLMForConditionalGeneration(config=config).to(torch.bfloat16)  # so101: bf16 VLM, halves memory\n")); print("smolvlm_with_expert.py patched")
PYP
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import lerobot, lerobot.motors.feetech, lerobot.policies.act.modeling_act; print("lerobot", lerobot.__version__, "imports ok")
PY
# -- 5. plugins copied from the Mac (avf camera plugin is macOS-only, skipped) --
[ -d ~/so101/plugins/lerobot-robot-busheal ]    && pip install -e ~/so101/plugins/lerobot-robot-busheal
[ -d ~/so101/plugins/lerobot-policy-actasync ]  && pip install -e ~/so101/plugins/lerobot-policy-actasync
pip install mlflow-skinny   # the client only (no server/UI/pandas): labtrack.py logs calibration + recording runs to the Mac (docs/mlops/tracking.md); optional
mkdir -p ~/.cache/huggingface/lerobot/calibration
grep -q 'so101/env.sh' ~/.bashrc || echo '[ -f ~/so101/env.sh ] && source ~/so101/env.sh' >> ~/.bashrc
echo "== devices =="; ls -l /dev/so101_* 2>/dev/null || echo "servo boards not plugged in yet"
ls -l /dev/v4l/by-id/ 2>/dev/null || echo "no cameras yet"
echo "DONE. Reboot once (new kernel/firmware + dialout group), then open a terminal: env.sh loads automatically."
