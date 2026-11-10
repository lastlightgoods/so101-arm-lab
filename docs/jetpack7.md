# JetPack 7.2 for the lab's Orin Nano Super (8 GB, microSD) — 10 Nov 2026

Today: JetPack 6.2.1 / L4T R36.4.7, UEFI 36.4.4, Ubuntu 22.04.5, kernel 5.15.148, Python 3.10, torch 2.11.0+CUDA 12.6, LeRobot 0.6.1. Current JetPack 7 is 7.2.1 (Jetson Linux 39.2.1, 12 Aug 2026); 7.2.0 shipped 2 Jun 2026.

## 1. Support, versions, install path, microSD, Super, firmware

Supported: "Adds support for the Jetson Orin product family within JetPack 7 releases"; "JetPack 7.2 ... is now available for Jetson Orin Nano Developer Kit" (NVIDIA staff, https://forums.developer.nvidia.com/t/372060 and /t/372151). r39.2 release notes: "This release supports the NVIDIA Jetson Thor and Jetson Orin platforms" (https://docs.nvidia.com/jetson/archives/r39.2/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.pdf).

Versions (https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2 and archive-7.2.1): Ubuntu 24.04, kernel 6.8, CUDA 13.2.1 (13.2.2 in 7.2.1), cuDNN 9.20.0, TensorRT 10.16.2, VPI 4.1.3. Ubuntu 24.04 python3 is 3.12.3 (packages.ubuntu.com/noble/python3).

Not an apt upgrade: NVIDIA lists only ISO, SDK Manager, manual flash; apt is used afterwards for `nvidia-jetpack` (/t/372060). "On Jetson Orin Nano Developer Kit, Jetson ISO is supported for JetPack 7.2 and later" and is "the recommended first-time setup path" (https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_bsp.html). The ISO goes on a USB stick (Balena Etcher from a Mac is fine); SDK Manager needs an Ubuntu x86 host.

microSD: "There is no SD card image for Jetson Orin Nano Developer Kit in this release" (/t/372060), but the ISO installs "directly on target storage ... such as a microSD card or a NVMe SSD" (setup_bsp.html); quick_start.html says "64GB UHS-1 or larger", NVMe "optional, but recommended".

Firmware: "JetPack 7.2 and later installations require JetPack 6.x-generation Jetson UEFI/QSPI firmware" (setup_bsp.html; threshold 36.0, /t/372151). Your 36.4.4 qualifies. The installer then flashes new QSPI firmware: release note 6266271, "allow Capsule update of the QSPI image ... press 'y' when prompted ... Skipping this step causes installation issues due to incompatibility of new ISO images with older QSPI images"; the prompt "only waits 30 seconds" (quick_start.html). Users report UEFI 39.2.0 afterwards (/t/372151 #57).

Super mode: 7.2.0 release note 6279443, "units ... updated to JetPack 7.2 using ISO install continue to use the same profile that was set before update ... will not default to 'Super' mode". 7.2.1: "ISO now flashes the Jetson Orin Nano Developer Kit with Super Mode flashing configuration by default" (archive-7.2.1). Staff: "I tried the rel-36.5 super config to rel-39.2 and I still saw super config" (/t/372151 #22). Default 25 W, MAXN SUPER selectable (quick_start.html). Verify `/etc/nv_boot_control.conf` contains `jetson-orin-nano-devkit-super`.

## 2. PyTorch and LeRobot

NVIDIA staff (AastaLLL, /t/372773): "Orin can use SBSA package now so upstream package should work", showing `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132` giving torch 2.12.0+cu132 with CUDA available. CUDA 13.2 blog: "This release also supports NVIDIA Jetson Orin devices on the same CUDA SBSA toolkit" (https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/). Caveat: upstream wheels warn "Found GPU0 Orin ... CC 8.7 ... except {8.7}"; NVIDIA says filter it. Performance on sm_87 is unverified.

Wheels listed directly today: download.pytorch.org/whl/cu132 aarch64 cp310-cp313 has torch 2.12.0 to 2.15.0; /whl/cu130 has 2.9.0 to 2.14.1. https://pypi.jetson-ai-lab.io has only jp6/cu126, cu128, cu129 and sbsa/cu130 (torch 2.9.0 to 2.11.0, cp312); no jp7 index (jetson-containers issues #1711, #1736 agree). The docs.nvidia.com "PyTorch for Jetson" page has no JetPack 7 entry.

LeRobot 0.6.1 (pyproject tag v0.6.1, PyPI): `requires-python = ">=3.12"`, `torch>=2.7,<2.12.0`, `torchvision>=0.22.0,<0.27.0`, `numpy>=2.0.0,<2.3.0`. Python 3.12 retires py310_compat.py, but the torch pin excludes every cu132 wheel; use torch 2.11.0 from `/whl/cu130` or `sbsa/cu130` (minor-version compatibility, docs.nvidia.com/deploy/cuda-compatibility). LeRobot 0.7.0 allows `torch<2.16`. The venv is rebuilt from scratch either way.

## 3. Kernel 6.8 and the hardware

(a) Wi-Fi: NVIDIA's out-of-tree tree at tag jetson_39.2_GA still carries drivers/net/wireless/realtek/rtl8822ce, built by default, MODULE_NAME 88x2ce, and its Tegra block defines no CONFIG_CONCURRENT_MODE in 36.4.4 or 39.2 (gitlab.com/nvidia/nv-tegra/linux-nv-oot). Expect the same vendor driver and the same no-AP+STA behaviour. Mainline 6.8 rtw88 `rtw_8822ce` (10ec:c822) advertises 1 station + 1 AP, max 2 interfaces, 1 channel, 8822C only, since "wifi: rtw88: 8822c: add iface combination" (Apr 2023, kernel 6.4; rtw88/main.c v6.8). Whether NVIDIA's 39.2 kernel enables RTW88_8822CE is unverified; a third-party JetPack 7 image says it is kept off (github.com/wendylabsinc/WendyOS-Builder PR 276). Check `lspci -k` and `iw list`.

(b) UVC: between 5.15 and 6.8, "Check for INACTIVE in uvc_ctrl_is_accessible()" (6.3) rejects writes to exposure_time_absolute while auto_exposure is not Manual, and honours a new auto_exposure value in the same `v4l2-ctl -c` call; "Use standard names for menus" and "Implement mask for V4L2_CTRL_TYPE_MENU" (6.3) change menu text only (GitHub commit log, uvc_ctrl.c). Control names and the master/slave mapping are unchanged; no C920 (046d:08e5) quirk in either kernel. cam_lock risk is ordering only: set auto_exposure=1 first.

(c) CH343: WCH says CH342/343/9102 are CDC-ACM compliant and use the stock cdc-acm driver (github.com/WCHSoftGroup/ch343ser_linux README). cdc-acm is in both kernels; ch341.c lists only 1a86:5523/7522/7523 in 6.8. Expect /dev/ttyACM* as today.

(d) Containers: staff say Orin Nano on 39.2 "can now run mainstream Arm64 'arm64-SBSA' containers and binaries" including upstream vLLM (/t/372151). MIG is "Jetson Thor T5000 ... technology preview"; the SBSA flashing change is Thor-only (/t/372060).

## 4. TensorRT and CUDA

"By default, TensorRT engines are compatible only with the version of TensorRT used to build them"; "hardware compatibility is not supported on NVIDIA DriveOS or NVIDIA JetPack" (https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html). TensorRT 10.3 to 10.16.2 means any engine is rebuilt. I found no NVIDIA sentence requiring CUDA 12 binaries to be rebuilt; in practice jp6 wheels link L4T CUDA 12.6/cuDNN 9.3 libraries absent on 7.2. The lab is PyTorch eager with no engines, so only the venv is affected.

## 5. Backup and rollback

From the Mac: `diskutil list`, `diskutil unmountDisk /dev/diskN`, `sudo dd if=/dev/rdiskN of=orin-jp621.img bs=4m status=progress`, then zstd/gzip it (195 GB is empty). Restore with dd, Raspberry Pi Imager or Balena Etcher. Full 233 GiB (250 GB) card: 104 min at 40 MB/s, 69 min at 60, 46 min at 90 (UHS-I reader ceiling); writing back is slower.

A second card is cleaner but not a free rollback. NVIDIA (wichiu, /t/379510): downgrade "is not an in-place downgrade ... Use SDK Manager to perform a full JetPack 6.2 flash. This restores both the QSPI boot firmware and the operating system image ... Simply replacing the microSD card with a JetPack 6.2 image is not the documented rollback path because it does not restore the QSPI firmware." Devkit guide: "The firmware must be compatible with the JetPack release you boot" (update_firmware.html). No NVIDIA page says whether a 36.x card boots on 39.x UEFI; one user on 39.x firmware had a 6.2.1 card boot but oem-config never finish, while a 6.2 card worked (/t/387254, single report, unverified). Plan on an x86 host (Ubuntu, or SDK Manager on Windows per staff in /t/387254) for any return.

## 6. Recommendation and test order

Wait, or run 7.2.1 on a second card as a parallel lab only after accepting the QSPI update is one-way without an x86 host. Gains: Python 3.12 (drop the shim), SBSA containers. Costs: full reinstall, torch held at 2.11+cu130 for LeRobot 0.6.1, same Wi-Fi driver, install bugs still reported in Oct 2026. First, dd-image the current card. Then: (1) `nv_boot_control.conf` shows -super, `nvpmodel -q` offers 25 W and MAXN SUPER; (2) servo bus: ttyACM enumerates, 1 Mbaud ping of every STS3215; (3) cameras: both C920 and the wrist cam enumerate, cam_lock applies, `v4l2-ctl -C auto_exposure,exposure_time_absolute,gain` reads back; (4) Wi-Fi hotspot: `iw list`, start AP, note station behaviour; (5) LeRobot teleop; (6) record one episode; (7) short train, `torch.cuda.is_available()`, no CC 8.7 warning flood; (8) dashboard ssh probes (new host key, paths on 24.04).

## Owner's claims checked

- "JetPack 7.2 brings JetPack 7 to the Orin family": confirmed (/t/372060).
- "MIG and SBSA are Thor-only": MIG yes; SBSA no, the SBSA CUDA toolkit and arm64-SBSA containers cover Orin (CUDA 13.2 blog; /t/372151); only SBSA flashing is Thor-specific.
- "JetPack 7.1 added T4000": confirmed, with Jetson Linux 38.4 (archive-7.1; /t/356852).
- Unverified: JetPack 6 card on 39.x UEFI; rtw88 in NVIDIA's 39.2 kernel config; cu132 wheel speed on sm_87; any NVIDIA "rebuild CUDA 12 binaries" statement.
