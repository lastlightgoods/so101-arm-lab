# The Logitech C920 that un-locks itself: gain 255, re-derived exposure, and how we keep it frozen for LeRobot

If you record SO-101 datasets with a Logitech C920 and have tried to pin its exposure so the arm
moving through the frame stops changing the picture, this is for you. We spent an evening on it and
measured some firmware behaviour worth knowing before you write the three `v4l2-ctl` lines everyone
writes first.

## Why lock at all

Out of the box a C920 runs auto exposure, auto white balance and continuous autofocus. When the arm
enters the frame the camera re-exposes on the arm, so brightness becomes a function of arm pose, and a
policy can learn the brightness cue instead of the scene. Our rule: vary lighting between sessions
deliberately, re-capture settings at the start of each session, keep them fixed within a session.

## What this revision does (USB id 046d:08e5, measured 9 Nov 2026 on a Jetson Orin Nano)

The controls, by their v4l2 names: `auto_exposure` (3 = Aperture Priority, 1 = Manual Mode),
`exposure_time_absolute`, `exposure_dynamic_framerate`, `gain`, `white_balance_automatic`,
`white_balance_temperature`, `focus_automatic_continuous`, `focus_absolute`, `power_line_frequency`,
`backlight_compensation`.

1. **Manual controls are `inactive` while their auto flag is on.** A write to `exposure_time_absolute`
   or `gain` in that state fails with EACCES. Set `auto_exposure=1` first, then the manual values.

2. **Exposure snaps to a stop table.** `exposure_time_absolute` is in 100 us units and the camera only
   exposes in 1/2^n-second stops: 78 (1/128 s), 156, 312. The same stop reads back as 77 or 78, so
   verify with a tolerance of 2. At 30 fps exposure must stay at or below 333, so 312 is the longest
   usable stop; if auto wanted more, start at 312 and raise gain.

3. **About one second after `auto_exposure` goes 3 -> 1, the camera forces `gain` to 255.** This one
   cost us the evening. The sequence we saw: write `auto_exposure=1`, write `gain=42`, read back 42,
   and roughly a second later read back 255 and a white picture. Any gain written inside that window
   is clobbered. We now wait 1.5 s after the transition and rewrite whatever moved.

4. **At every stream start in Manual Mode the camera re-derives its exposure.** We had 78 locked;
   opening the stream brought it up as 156, the next open 312. A write issued while the stream runs
   holds for that stream. LeRobot opens the cameras itself when `lerobot-record` starts, so a one-shot
   apply before recording is not enough.

5. **The camera occasionally re-enumerates on USB** and comes back with factory defaults, auto on.

## The fix, in three layers

**Capture once per lighting session.** `cam_lock.py capture` streams each camera in the exact mode
recording uses (640x360 MJPG 30 fps), hands it to auto, waits 4 s, reads what auto chose and the
picture's mean grey, locks, then tunes `gain` until the locked picture matches auto's mean within 8
grey levels (clamped to 100-140). A locked picture outside 60-190 is put back on auto and not saved.
The result is one JSON block per camera keyed by its `/dev/v4l/by-id` name. For us on 9 Nov: top C920
exposure 38, gain 42, white balance 3377 K, focus 30; front C920 exposure 38, gain 85, white balance
3388 K, focus 35. Auto had chosen exposure 83 and gain 109 on both. We also force
`power_line_frequency=2` and `backlight_compensation=0`, and switch off `exposure_dynamic_framerate` so
the camera cannot halve the frame rate to buy light.

**A resident guard.** `cam_lock.py guard` runs as a systemd user unit (`Restart=always`, with
`loginctl enable-linger` so it starts at boot). It checks every second and re-applies the saved values
within about a second of any drift, logging each fix. That catches the stream-start jump.

**A udev rule for re-plugs.** On `ACTION=="add"` for the C920's capture node, a transient
`systemd-run --on-active=3` unit re-applies the file 3 s later, because udev kills `RUN+=` programs
after a few seconds and the camera is not always ready to take controls the instant its node exists.

And a guard rail: our `so101-record` wrapper runs `cam_lock.py check --strict` first and refuses to
start if any camera is unplugged, uncaptured or drifted. Recording on auto is the expensive mistake,
so the tool does not let it happen quietly.

## How to tell it worked

Pull a frame with the arm out of the picture and one with the arm waved through it; the mean grey
should not move (ours stays at 112.6 on the top camera). `v4l2-ctl -d /dev/video0 --list-ctrls | grep
-E "auto|absolute|gain"` should show the auto flags off and no `inactive` anywhere.
`journalctl --user -u so101-cam-lock -n 20` shows one line per re-lock, typically right after LeRobot
opens a camera. The whole tool is covered by tests against a fake `v4l2-ctl` that reproduces every
behaviour above, so it can be worked on without the camera.

## Where the code is

`jetson/cam_lock.py`, `jetson/99-so101-cams.rules`, `jetson/so101-cam-lock.service` and
`docs/camera-settings.md` in the so101-arm-lab repository at github.com/lastlightgoods/so101-arm-lab.
The repo will be public soon. If your C920 revision behaves differently, we would like the numbers.
