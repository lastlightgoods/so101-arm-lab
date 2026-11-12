# Virtual arm on the 3D stage (SIM / FAKE ghost)

Fake-arm runs and simulator runs show up on the dashboard's 3D stage as a separate violet wire ghost arm, under a big
**SIM** or **FAKE** badge, a floating label with the session id and a violet frame around the stage. The real arm is
orange and always draws from the real follower stream; it is never hidden and never driven by this channel.

Display only. The channel is one way: a publisher process writes a JSON file in /tmp, `dashboard/server.py` reads it
into `/api/state` under the key `virtual`, the page draws it. Nothing on that path calls the arm, ssh or the Jetson,
and `state["arms"]` (what feeds the camera's arm mask) is never touched.

## Watch a run

    # a fake run (virtual clock, real kinematics), paced in real time
    .venv/bin/python tools/autoloop.py --dry-run --stage --cycles 3 --session demo
    # faster / unpaced: --stage-speed 2   --stage-speed 0

    # a simulator run: nothing to add. sim/so101_simstate.py is loaded into the sim's interpreter
    # (so101_simstate.pth) and publishes on every step; the ghost appears while the sim runs.

Open the dashboard. The badge reads FAKE or SIM; two seconds and a half after the publisher stops (or the run ends)
the ghost, badge and frame go away. `autoloop --stage` needs `--dry-run`.

## Message

`/tmp/so101_virtual_<session>.json`, written atomically (`tools/virtual_stage.py`):

| field | |
|---|---|
| `source` | `SIM` or `FAKE`. `REAL` is refused by the writer; a file with any other or no source is served as `UNVERIFIED` and labelled "UNVERIFIED VIRTUAL" |
| `session` | `[A-Za-z0-9_.-]{1,40}` |
| `t`, `seq` | sender clock (informational), frame counter |
| `joints` | the six LeRobot names, degrees; gripper 0 to 100 |
| `joint_frame` | `lerobot` (FAKE) or `urdf` (SIM), see below |
| `objects` | `[{kind: nut or target, frame: mat_in or sim_base, x, y, z}]`; mat_in is mat inches, sim_base is metres (x forward, y left, z up) |

Staleness is judged by the file's mtime on the dashboard machine (2.5 s), never by `t`, so a skewed or dead sender
cannot keep a ghost alive. The page also drops a stream whose `seq` has not moved for 2.5 s. `SO101_VIRTUAL_DIR`
moves the directory (tests).

## Joint conventions (the calibration offsets)

The stage converts joint degrees to URDF radians in one function (`jointTarget` in `dashboard/index.html`) for the
real follower, the leader and the ghost. The real arm's readings are LeRobot-zeroed, so the stage adds the measured
`joint_offsets_deg` to bring them to URDF zero.

* FAKE: `DryArm` degrees are the same convention as the real arm (autoloop adds the table's offsets before the
  kinematics), so the ghost gets the same offsets.
* SIM: `sim_qpos_to_dataset_row` is a pure radians-to-degrees transform with synthetic calibration; the sim's zero is
  the URDF zero. The ghost gets no offsets (`joint_frame: urdf`). Not yet compared against a live sim.

The old `/tmp/so101_sim_state.json` and `sim_arm` in `/api/state` remain for one release; the page draws them through
the ghost like any other SIM stream.

## Servo temperature on the real arm

The real follower's six STS3215 bodies are tinted by the live temperature from the telemetry panel's `/api/tele` fetch
(the motor of each joint is the sts3215 mesh in the joint's parent link). Below 50 C the normal look; 50-60 light amber
to amber; 60-65 orange (60 = the unattended start gate); 65 and up red (the overnight stop trip); at the firmware cut
(`temp_cut_c`, 70) red with a slow pulse. Every servo carries a label with the number in C (load and current too on
hover or while the SERVOS panel is open), and the stage legend shows the 60 / 65 / 70 marks. Telemetry older than 5 s
(or none) shows the normal look and "no telemetry". The ghost is never tinted. Thresholds come from `TD.marks` / `TD.limits`.
