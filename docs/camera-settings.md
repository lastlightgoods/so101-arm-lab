# Camera settings lock

`jetson/cam_lock.py` (on the Jetson `~/so101/cam_lock.py`) freezes exposure, white balance and focus on the three
recording cameras so the arm entering the frame cannot change the picture. Settings live in
`~/so101/calib/cam_settings.json` (repo copy `jetson/calib/cam_settings.json`), one block per camera keyed by its
`/dev/v4l/by-id` name.

## Why

The two Logitech C920s (top, front) run auto-exposure, auto white balance and continuous autofocus; the icSpring wrist
camera runs auto-exposure and auto white balance. When the arm moves through the frame the camera re-exposes on the arm,
so image brightness becomes a function of arm pose. A policy trained on that learns the brightness cue instead of the
scene, and it breaks the moment the lighting or the arm's path differs.

The lighting rule for data collection: **vary lighting between sessions deliberately, re-capture at the start of each
session, keep the settings fixed within a session.**

## What is frozen

| camera | switched off | frozen at auto's value | forced |
|---|---|---|---|
| C920 top, front | `focus_automatic_continuous`, `auto_exposure` (1 = Manual), `white_balance_automatic`, `exposure_dynamic_framerate` | `focus_absolute`, `exposure_time_absolute`, `white_balance_temperature`, `gain` (then tuned, see below) | `power_line_frequency=2` (60 Hz), `backlight_compensation=0` |
| icSpring wrist | `auto_exposure` (1), `white_balance_automatic` | `exposure_time_absolute` (then tuned), `white_balance_temperature` | `power_line_frequency=2` (it ships at 50 Hz) |

Brightness, contrast, saturation, sharpness (and gamma, hue on the wrist) are pinned at whatever each camera has, which
is its own defaults unless somebody changed them. The wrist keeps the icSpring defaults (brightness 0 in -128..127),
never the C920's.

Exposure units on the C920 are 100 us. Recording runs at 30 fps, so exposure must stay at or below 333 (one frame); the
camera only exposes in 1/2^n-second stops (78 = 1/128 s, 156, 312 ...), and 312 is the longest stop that fits. If auto
picked more, capture starts from 312 with gain raised instead and says so. The wrist exposure is capped at 333 too.

## What this C920 revision does (measured 9 Nov 2026, USB id 046d:08e5)

These are the reasons the tool is more than three `v4l2-ctl` lines:

1. A manual control is flagged `inactive` while its auto flag is on, and writes to it fail with EACCES. The auto flag is
   switched off first; the file keeps that order.
2. Manual exposure snaps to the stop table above and the same stop reads back as 77 or 78. `check` allows a tolerance of
   2 on `exposure_time_absolute`.
3. About a second after `auto_exposure` goes 3 -> 1 the camera **forces gain to 255**, clobbering a gain written in
   between. That is the white-out seen on the first attempt. `apply` waits 1.5 s after a transition and re-writes what
   moved.
4. At **every stream start** in Manual Mode the camera re-derives its exposure (78 came up as 156, then 312). A write
   issued while the stream runs holds for that stream. lerobot opens the cameras itself at the start of a recording, so a
   one-shot apply before recording is not enough: the `guard` re-applies within about a second of any drift.
5. The top C920 re-enumerates on USB now and then (it was USB device 34 on a bus where the others are 8 and 9) and comes
   back with factory defaults, i.e. auto on. The udev rule and the guard both catch that.

## Flicker bands: keep the exposure at one light-flicker period (11 Nov 2026)

The room lights flicker at 120 Hz (8.3 ms). The first capture locked the C920s at exposure 38 (3.8 ms) and the wrist at 35,
shorter than one flicker period, so bright and dark bands rolled through every frame: measured as the frame-to-frame
spread of row means over 16 frames, top 16, front 5, wrist 35 grey levels peak-to-peak. The mat map read the bands as
objects on an empty mat. The C920 only exposes in 1/2^n-second stops (77/78 = 1/128 s, 156, 312), none an exact multiple
of 8.3 ms; 77 is the closest and cut the banding to 2 to 3 levels. The wrist's own exposure scale is not linear: 78 to
87 still banded (8 to 20 levels), 90 to 100 did not (about 4). Settings now: top exposure 77 gain 0 (mean 114), front 77
gain 40 (mean 118), wrist exposure 100 brightness -70 (mean 120); banding 3.3 / 2.2 / 4.1 levels. The old file is kept
on the Jetson as `calib/cam_settings.before_flicker_fix.json`.
Correction the same evening: the wrist went back to exposure 35, brightness 0. At one flicker period its sensor
saturates in this light (46 % of pixels at 255 at exposure 100, brightness 0); the -70 brightness offset only greyed the
clipped areas, which looked like glare. The icSpring has no gain control, so here no exposure is both band-free and
unclipped (35 -> 33 grey levels of banding, 50 -> 8 with 19 % clipped, 65 -> 24 with 31 % clipped; auto exposure with
60 Hz anti-flicker banded worst, 45). The wrist keeps its bands until the room light is flicker-free (a DC-driven LED
lamp fixes all three cameras) or its lens gets a neutral-density filter so a full-period exposure does not clip.
Watch out: on the front C920 a gain write sometimes reads back a different number for a few seconds (set 0 read 109,
set 30 read 229); `cam_lock.py check` settled on the written value within 5 s here.

## The icSpring wrist camera looks green

Measured 10 Nov 2026 on a grey patch of the mat, centre of the frame: with the captured values (white balance 4000, saturation 25)
the green channel sat 21 % above red and 29 % above blue. The module has a fixed green bias that its controls cannot remove:
`white_balance_temperature` only trades red against blue (3000 K: R 142 / B 187; 6500 K: R 208 / B 81; green stayed at 166 to 178
throughout), and `hue` only rotates the cast. `saturation` is the one control that shrinks it: 18 -> G/R 1.14, 12 -> 1.10, 8 -> 1.06,
4 -> 1.02. The file now pins the wrist at white balance 4000, hue 0, saturation 8 (G/R 1.06, G/B 1.08 on the same patch); go to 4 if
a neutral picture matters more than object colour. Changing it: edit `calib/cam_settings.json`, `so101-cams-lock`, `so101-cams-check`.
Stopping the guard to experiment (`systemctl --user stop so101-cam-lock`) lets the C920s drift at the next stream start; start it
again and `so101-cams-check` within a few seconds.

## Commands (`source ~/so101/env.sh` first)

| command | what it does |
|---|---|
| `so101-cams-capture` | New lighting session. Per camera: takes it from the dashboard, streams it in the mode recording uses, hands it back to auto, lets auto settle (4 s), reads what it chose and the picture's mean grey, locks the camera, then tunes gain (C920) or exposure (wrist) until the locked picture matches auto's mean (clamped to 100..140, within 8). Writes the file. A camera whose locked picture ends outside 60..190 is **put back on auto and not saved**; fix the light and run it again. Takes about 15 s per camera; the dashboard feeds pause meanwhile. Needs the venv python (this shell has it). |
| `so101-cams-lock` | `apply --verify`: re-apply the saved values, stream each camera, measure the picture, roll back to auto if it is outside 60..190. |
| `so101-cams-check` | Compare live values with the file, one line, exit 0 only if everything matches. `--json` for machines. |
| `so101-cams-guard` | Start the resident re-locker in the background if none is running (the systemd user unit normally does this). |
| `python3 ~/so101/cam_lock.py show` | Dump the live values of the controls the tool manages, with `inactive` flags. |
| `python3 ~/so101/cam_lock.py capture --set wrist:exposure_time_absolute=156` | Seed one starting value by hand before tuning. |

`so101-record` and `so101-record-more` run `cam_lock.py check --strict` first and refuse to start, with a clear message,
if any camera is unplugged, uncaptured or drifted. When the check passes they also make sure a guard is running.

An unplugged camera is reported, not fatal, unless `--strict`.

## Re-capturing when the lighting changes

```
source ~/so101/env.sh
so101-cams-capture          # arm at rest, lights as they will be for the session
so101-cams-check            # cams locked: top ok; front ok; wrist ok
```

Then record. The guard picks up the new file by its modification time; nothing to restart. Keep the picture means it
prints in the session notes (they are also in the json: `auto_mean`, `target_mean`, `verified_mean`).

## Install on the Jetson (once)

Files are copied with the rest of `jetson/` (`scp jetson/cam_lock.py jetson/99-so101-cams.rules jetson/so101-cam-lock.service jetson:~/so101/`).

udev rule, re-applies the lock 3 s after a camera appears (boot or re-plug), needs sudo:

```
sudo install -m 644 ~/so101/99-so101-cams.rules /etc/udev/rules.d/99-so101-cams.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux --action=add
```

Guard as a systemd user unit (no sudo), plus lingering so it runs at boot without a login (sudo):

```
mkdir -p ~/.config/systemd/user && cp ~/so101/so101-cam-lock.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now so101-cam-lock.service
sudo loginctl enable-linger tc
```

## Verify

```
so101-cams-check                                   # one line, exit 0
python3 ~/so101/cam_lock.py check --json | python3 -m json.tool
v4l2-ctl -d /dev/video0 --list-ctrls | grep -E "auto|absolute|gain"   # auto flags 0 / Manual Mode, no `inactive`
journalctl --user -u so101-cam-lock -n 20          # guard log: one line per re-lock, e.g. after lerobot opens a camera
tail ~/so101/outputs/run/cam_lock_guard.log        # same, when started by so101-record instead of systemd
```

Pull one frame per camera from the Mac dashboard (`http://localhost:8765/cam/top.jpg`, `front.jpg`, `wrist.jpg`) and
check that the mean grey is near the `verified_mean` in the json and does not move when the arm is waved through the
frame.

## Tests

`tests/jetson/test_cam_lock.py` runs the tool against a fake `v4l2-ctl` and a fake frame source that reproduce the
behaviours above (inactive-while-auto, stop quantisation, the gain clobber, the stream-start jump, a control that will
not take, an unplugged camera, a white-out), plus the `so101-record` refusal through a sourced `env.sh`.
