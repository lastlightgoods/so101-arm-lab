# What the arm is doing on its own, and how

This page explains the session runner ([`autoloop.md`](autoloop.md)) in one read, with a recording and the numbers from the
first attended ten-cycle session on 10 Nov 2026. The same text is on the site as
[Practising alone](https://so101-arm-lab.c-thomas-clark-pdx.workers.dev/notes/practising-alone).

[![Camera recordings from the overhead C920, the front C920 and the wrist camera at 3x speed: the arm picks a hex nut and places it on a target on the mat. Poster frame of the video.](../site/public/video/autoloop-poster.jpg)](../site/public/video/autoloop-cycle7.mp4)

*Camera recordings from the overhead C920, the front C920 and the wrist camera, 3x speed, about 95 s of session first10b
(the end of cycle 6, then cycle 7). Click for the MP4 (2.4 MB); GitHub does not play repo videos inline.*

## What I am trying to do

I want the arm to practise by itself: try a pick, judge the result, reset the mat, try again. A person resetting the nut between
every attempt is the slow part of learning from a real arm, so the first milestone is a session where nobody touches the mat.

The milestone is called "10 in a row, attended". The task is to pick a hex nut and place it on a target at mat (9, 6) in. The
bar is at least 9 of 10 cycles landing within 15 mm of the target, with zero guardrail trips. I stand at the wall STOP the whole
time.

## One cycle, step by step

1. **Guard.** Before anything moves the runner checks the stop file, that the dashboard answers, that the overhead camera sees
   at least 3 of its 4 markers, and that no LeRobot job is holding the arm.
2. **Find the nut.** The overhead camera's mat map gives the nut's position in mat inches. If the map misses it, a search over
   the whole mat image takes over.
3. **Closed-loop pick** (`closed_loop_pick` in `tools/pick_loop.py`). The nut position goes through inverse kinematics. The
   arm is commanded to a hover point, the joints it actually reached are read back, forward kinematics (`fk_mat`) turns them into
   a position on the mat, and the difference is the correction. That happens twice: once at the hover, once at the grasp height
   with the jaws open. If the tip is still more than 6 mm above the mat, the grasp is lowered. The jaws open to 35 and close in
   two steps, so the moving jaw meets the nut slowly. The numbers behind this are in [`pick-loop.md`](pick-loop.md): an
   open-loop hover came up 5 to 8 mm short, and after two corrections the grasp pose was within 0.1 mm of the command.
4. **Jaw check** (`jaw_check`). After the lift, one wrist-camera frame is searched for the nut's bright ring with a dark hole
   inside a window between the jaw tips. Not found: open, go back, grasp once more. Not found again: record a failed pick.
5. **Place and judge.** The arm carries the nut to the target and places it. After 1 s the overhead camera's map gives where the
   nut is. The cycle landed if the nut moved more than 20 mm from where it started and ended within 15 mm of the target.
6. **Reset.** The same closed-loop pick takes the nut from the target and drops it at a random spot in a safe box, x 6 to 13 in
   and y 4 to 8 in, at least 40 mm from other objects, 50 mm from marker centres and 60 mm from where it was. The next cycle
   starts from somewhere new.

Every cycle is logged to MLflow as a run in the experiment `so101-autoloop`, with the landing error, the retries and the stage
detail of each pick.

## Guardrails

The runner never talks to the arm directly. Every motion goes through one script that applies the arm's own limits, and
[`guardrails.py`](../tools/guardrails.py) decides before each step whether to go on. Any trip ends the session.

| Guardrail | What it stops |
| --- | --- |
| workspace | A planned waypoint outside the mat (shrunk by 20 mm), above 4 in, or more than 10 mm below the mat plane. The plan is never run. |
| motion watchdog | A waypoint where commanded and reached joints differ by more than 8 degrees. |
| perception | Fewer than 3 markers seen for 10 s, calibration not ok, or no scene from the overhead camera. |
| panel liveness | The dashboard does not answer within 3 s, or reports the Jetson offline. |
| arm busy | A LeRobot job (teleop, record, rollout) already holds the arm. |
| stop file | A clean stop between cycles or before any motion. |
| caps | Cycles, minutes, and 3 failed cycles in a row. |

The wall STOP cuts the arm independently of all of this.

## What happened on 10 Nov 2026

Session `first10b`: 10 cycles, target (9, 6) in, 15 mm radius, 1067 s in total.

| Cycle | Landed | Error from target (mm) | Retries | Cycle time (s) |
| --- | --- | --- | --- | --- |
| 1 | yes | 5.1 | 0 | 84.53 |
| 2 | yes | 5.7 | 0 | 84.71 |
| 3 | yes | 3.4 | 0 | 84.69 |
| 4 | yes | 7.9 | 0 | 101.71 |
| 5 | no | 4.2 (not picked, moved 1.7 mm) | 1 | 127.51 |
| 6 | no | 20.6 (not picked) | 1 | 127.86 |
| 7 | yes | 4.4 | 0 | 84.43 |
| 8 | no | none recorded | 1 | 85.09 |
| 9 | no | none recorded | 1 | 117.43 |
| 10 | no | 62.8 (not picked) | 1 | 163.83 |

5 of 10 landed (success rate 0.5), mean error 14.3 mm, median 5.4 mm, 5 retries. The runner stopped after cycle 10 with a
`consecutive_failures` trip (3 failed cycles in a row, cycles 8 to 10). That is below the bar of 9 of 10 with zero trips, so the
milestone is not met. The earlier session `first10` ran 4 cycles, landed 3 (mean error 9.2 mm) and stopped on a `run_plan` trip
in cycle 4 because a LeRobot teleop job had started and was holding the arm; the arm-busy check exists because of that.

Landed cycles were all within 8 mm of the target.

## What went wrong, from the stage logs

Every pick logs its stages to MLflow. Reading them after the session changed what I thought had happened.

- **The fixed jaw came down on the nut.** The SO-101 has one moving jaw and one fixed jaw, the thumb. The point I aimed was the fixed jaw's tip, where the jaws meet when they are closed, so the thumb landed on the nut. In some cycles it pushed the nut forward; in others it came down on top of it. Fixed after the session: the pick now aims the fixed jaw 10 mm behind the nut, along the direction the jaws open, so the nut sits inside the open gap ([`pick-loop.md`](pick-loop.md), "The grasp aim").
- **Cycle 4's reset failed and nothing noticed.** The wrist camera found no nut between the jaws on both tries, so the reset put nothing down and the nut stayed on the target. The runner does not check a reset, so it assumed the nut was at the reset spot.
- **Cycles 5 and 6 picked the nut but scored as misses.** The jaw check saw the nut held after every grasp. The nut started on or next to the target, so carrying it back moved it less than 20 mm, and the judge's rule that a picked nut moves more than 20 mm called that a miss.
- **In cycles 8 to 10 the map lost the nut.** After cycle 8's place the overhead map reported no objects at all, although the jaw check had seen the nut held. In cycle 9 the map sent the arm to two spots where the wrist camera found nothing between the jaws. The nut had sat on the target through cycles 4 to 6, and the map's background absorbs anything that stays still for about 8 minutes, so the likely cause is the nut being learned into the background there. I have not confirmed that.

## About the video

I started the recording partway into the session. The dashboard dropped the wrist stream after 35 s and the overhead stream after
95 s; the front stream ran on. The clip is the first 95 s of the three recordings at 3x. The wrist camera shows the end of
cycle 6; the overhead and front cameras then cover most of cycle 7, a landed cycle. I estimated the cycle boundaries from the
cycle durations in the log, so they can be a few seconds off. The nut is a small dot in the overhead view. The wrist loop
([MP4](../site/public/video/autoloop-wrist-loop.mp4)) is 5 s of the wrist recording at 3x.

## Where the code is

`tools/autoloop.py`, `tools/guardrails.py`, `tools/pick_loop.py` (`closed_loop_pick`, `jaw_check`, `fk_mat`,
`find_nut_anywhere`), and the docs [`autoloop.md`](autoloop.md), [`pick-loop.md`](pick-loop.md) and
[`camera-settings.md`](camera-settings.md).

## What is next

- Judge every reset with the overhead map, and start the next cycle from where the nut really is.
- When the nut already sits on the target at the start of a cycle, reset it first instead of scoring a pick that cannot move it.
- Keep the map from learning a still nut into the background during a session, and test a learned object detector on the same camera frames.
- Run the ten cycles again with me at the STOP.
