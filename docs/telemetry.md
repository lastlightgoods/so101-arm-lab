# Servo telemetry: what the servos report, the live panel, MLflow, and how it shares the bus

The Feetech STS3215 (and the STS3250 of the later special-edition arm) report their own temperature, load, current, supply voltage,
speed, error bits and commanded-versus-actual position over the same serial bus the arm is driven on. Before this, the lab read only
`Present_Position`. Now the dashboard shows the rest live, and every pick and autoloop cycle logs it to MLflow so sag, heat and current
compare across sessions. Everything here only **reads**: no register of any servo is ever written by this code.

| piece | file | runs on |
|---|---|---|
| register list, one-packet block read, decode, flags, per-run summaries | `jetson/servo_telemetry.py` | Jetson (and loaded by path on the Mac) |
| idle streaming next to the position poll | `jetson/arm_state.py` | Jetson |
| while `lerobot-record` / teleop own the bus | `plugins/lerobot-robot-busheal` (`publish_telemetry`) | Jetson, inside the lerobot process |
| while a pick runs | `jetson/pick_seq.py` (`Tele`) | Jetson |
| history, flags, `/api/tele` | `dashboard/telemetry_store.py`, `dashboard/server.py` | Mac |
| the panel (M key, SERVOS button, inline on the iPad) | `dashboard/index.html` | browser |
| MLflow metrics | `tools/pick_loop.py` (`tele_of`), `tools/autoloop.py` | Mac |
| read-only hardware check | `jetson/verify_telemetry.py` | Jetson |
| pretend arm and demo dashboard | `tools/fake_servo_bus.py`, `tools/telemetry_demo.py` | Mac |

## Why the joints strip chart "did nothing"

The strip chart is the Mission Control skin's attract panel (`dashboard/skins/missioncontrol.js`, six lanes, last 60 s). It was fed
correctly when the arm was idle; it looked dead for three separate reasons:

1. **The scale hid everything.** Every lane was a fixed +-120 deg (+-180 deg for the wrist roll) over about 100 px, so 1 deg was 0.4 px.
   The 3 deg shoulder-lift sag between two waypoints drew a 1 px step; a resting arm drew six ruler-straight lines. (Checked in the UI
   harness: a 3 deg step moved the LIFT pen about 1 px on the old code; the lane now autoscales to the pen's last 60 s, never narrower
   than 8 deg, and 3 deg moves it a third of the lane. `tests/ui/test_strip_chart.py`.)
2. **During every pick and autoloop cycle it had no data at all.** `pick_seq.py` holds `~/so101/.arms_busy`, so `arm_state.py` lets
   go of the port and forwards `/dev/shm/so101_follower.json` instead. Only the busheal plugin writes that file, and LeRobot loads
   plugins only inside its own command-line entry points (`register_third_party_plugins()` in `lerobot-record`, `-teleoperate`, ...);
   `pick_seq.py` imports `lerobot.robots.so_follower` directly, so the plugin never loaded and no file was written. `arm_state.py`
   then emitted `{"src": "busy", "pos": {}}` and the chart drew nothing, with no message. That is exactly when the arm moves.
   `pick_seq.py` now writes `/dev/shm/so101_follower_tele.json` (positions included) and `arm_state.py` borrows the positions from it.
3. **It never said why.** With no reading for 2.5 s the paper now shows `NO JOINT FEED` and the reason (busy without a state file,
   unplugged, bus error, no stream from the Jetson).

The live dashboard was checked read-only on 10 Nov 2026 (`GET /api/state`): the follower line was `src: idle-poll`, six joints in degrees,
fresh `rx`, so the data does reach the browser when the arm is idle. The page samples it every 100 ms and redraws at 30 fps, so it was updating.
To confirm reason 2 on the real arm: run any `pick_seq.py` plan and, on the Jetson, `ls -l /dev/shm/so101_follower.json` (old code: absent or stale).

## What a servo can report

Source: the Feetech STS/SMS control table (`http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0`) as
transcribed in LeRobot 0.6.1, `lerobot/motors/feetech/tables.py` (`STS_SMS_SERIES_CONTROL_TABLE`; sign-magnitude bits in
`STS_SMS_SERIES_ENCODINGS_TABLE`). The `sts3215` and `sts3250` entries of `MODEL_CONTROL_TABLE` are the same table object, so every
address below applies to both; they differ only in model number (777 and 2825) and in what the motor can do (STS3250 datasheet: 4.2 A
stall at 12 V against 2.7 A, 16 kg.cm rated, docs/stronger-arm.md). `tests/jetson/test_servo_telemetry.py` checks every address here
against the installed LeRobot table.

Live registers (all in addresses 40 to 70, read in **one** packet for all six servos):

| register | addr | bytes | meaning and unit | read back on TC's arm |
|---|---|---|---|---|
| `Torque_Enable` | 40 | 1 | 1 torque on, 0 free | seen (LeRobot reads and writes it on every connect) |
| `Acceleration` | 41 | 1 | commanded acceleration ramp | to verify |
| `Goal_Position` | 42 | 2 | **commanded** position, steps, sign bit 15 | seen (busheal `clear_overload` reads position and parks the goal) |
| `Goal_Time` / `Goal_Velocity` | 44 / 46 | 2 / 2 | commanded move time / speed | to verify |
| `Torque_Limit` | 48 | 2 | active torque cap, permille | seen (busheal writes it for the gripper) |
| `Lock` | 55 | 1 | EEPROM write lock | to verify |
| `Present_Position` | 56 | 2 | steps, 4096 per turn | seen (everything uses it) |
| `Present_Velocity` | 58 | 2 | steps/s, sign bit 15 | to verify |
| `Present_Load` | 60 | 2 | 0.1 % of rated PWM duty, sign bit 10 (a torque proxy, not newton-metres) | to verify |
| `Present_Voltage` | 62 | 1 | 0.1 V | to verify |
| `Present_Temperature` | 63 | 1 | degC | to verify |
| `Status` | 65 | 1 | error bits: 0 voltage, 1 angle sensor, 2 temperature, 3 current, 5 overload (bit 4 undocumented) | to verify |
| `Moving` | 66 | 1 | 1 while moving to a goal | to verify |
| `Present_Current` | 69 | 2 | counts, 6.5 mA each on the Feetech sheet | seen on the gripper (busheal's 6 Oct stall measurements are in this unit); **scale unverified** |

Limits and identity (read once per connect, addresses 0 to 38): `Model_Number` (3, seen at every connect), `Firmware_Major/Minor_Version`
(0, 1; seen), `Max_Temperature_Limit` (13, the **overheat cut**, factory 70 degC), `Max_Voltage_Limit` (14), `Min_Voltage_Limit` (15),
`Max_Torque_Limit` (16; seen), `Unloading_Condition` (19), `LED_Alarm_Condition` (20), `Protection_Current` (28; seen), `Protective_Torque` (34; seen),
`Protection_Time` (35; seen), `Overload_Torque` (36), `Over_Current_Protection_Time` (38). The panel replaces its 70 degC mark with the value
read from the servos. Also readable but not polled (configuration, no health value): `ID`, `Baud_Rate`, `Return_Delay_Time`,
`Response_Status_Level`, position limits, the PID terms, dead zones, `Angular_Resolution`, `Homing_Offset`, `Operating_Mode`, the factory block (80 to 86),
`Goal_Position_2` (71).

"seen" means the repo already reads or writes that register on the real arm (cited); it does **not** mean this code has run on the arm. Nothing was
run on hardware in building this: every row marked "to verify" is unproven on TC's servos until `jetson/verify_telemetry.py` says so.

## What the panel shows

Press **M** (or the **SERVOS** button in the arm panel's 3D view) on the wall dashboard for a full-window view; on the iPad (landscape, scrolling layout)
the panel is inline under the arm row. Time window 1, 5 or 10 minutes.

- **Temperature** per joint over time with the cut (70 degC, or what the servos report) and a 60 degC warning mark.
- **Load** (signed % of torque) and **current** (A) per joint; the shoulder lift is drawn thick.
- **Bus**: total current of all six servos against the **5 A** continuous mark of the Waveshare board (Robonine load test, docs/stronger-arm.md),
  and supply voltage (the lowest of the six) with 11 V and 10 V marks for brown-out watching on the 12 V 5 A supply.
- **Commanded vs actual** for the picked joint over the last 20 s (dashed goal, solid actual) and, below it, the **sag** (goal minus actual) over the
  chosen window with the joint's usual holding sag in the heading.
- **Error flags as they happen**: every time a flag first goes up (hot, brown-out, bus over rating, a servo's own error bit, sag above its usual) a line is
  logged with its time; red ones also go to the main LOG panel and `events.log`.
- A per-joint table (temperature, load, current, sag, one-minute peaks, flag chips) and a header strip (link state and source, supply, bus current, hottest, flags).
- The SERVOS button itself turns amber or blinks red while any flag is up, so a hot servo shows even with the panel closed.

Flags: temperature amber from 60 degC, red from 65 degC (both pulled lower if a servo reports a lower cut); supply amber under 11 V, red under 10 V;
bus amber from 4 A, red from 5 A; sag: the last 20 holding samples average more than 1.5 times the joint's median holding sag (and over 1 deg more),
with at least 60 holding samples of history. These are lab choices (constants at the top of `servo_telemetry.py`), not servo settings.

## Polling rate and how it shares the bus

Wire time at 1 Mbaud is 10 us per byte. One `sync_read` of `Present_Position` for six servos is 14 + 6 x (6 + 2) = 62 bytes, 0.62 ms (this is
what the control loop does every frame). The telemetry block (addresses 40 to 70) is 14 + 6 x (6 + 31) = 236 bytes, **2.36 ms**, once per sample for
all six servos. (Computed, not measured: `jetson/verify_telemetry.py` times both on the real bus and prints them; expect more from USB-serial latency.)
Decoding on the Mac took 0.06 to 0.12 ms per sample in the fake-arm tests (`dt_ms` in each sample).

| who has the bus | position reads | telemetry | bus time used |
|---|---|---|---|
| idle (`arm_state.py`, nothing else running) | 15 Hz | 5 Hz (every 3rd tick) | about 0.9 % + 1.2 % = 2 % |
| `lerobot-record` / teleop at 30 fps (busheal plugin) | the loop's own | 2 Hz (`LEROBOT_TELE_S=0.5`) | 4.7 ms per second = 0.5 % extra; one control frame in 15 runs 2.4 ms longer inside a 33 ms frame |
| a pick (`pick_seq.py`, 40 steps/s) | its own | 4 Hz while moving (`SO101_TELE_S=0.25`), plus one read after each waypoint settles | 9.4 ms per second = about 1 %, taken out of the move loop's own sleep so a move lasts exactly as long (`tests/jetson/test_pick_seq_telemetry.py`) |

How it never collides with motion: the bus has one master and no locking, so telemetry is only ever issued by whoever already owns the bus, from the
thread that just finished its own read. `arm_state.py` reads only when **no** lerobot process runs and `~/so101/.arms_busy` is absent; the moment either appears it
closes its port (torque untouched) and stops reading. While busy it forwards the small `/dev/shm/so101_<role>_tele.json` file the owner writes (busheal for
`lerobot-record`, `pick_seq.py` for a pick), which costs the bus nothing. The block read is its own packet; it never touches the Present_Position reader's state beyond
the SDK's `_setup_sync_reader`, which LeRobot re-sets before every read. A failed or timed-out telemetry read returns `None`, is not retried and never triggers
busheal's heal; any exception is swallowed (`tests/busheal/test_busheal_telemetry.py`). Off switches: `LEROBOT_TELE_S=0`, `SO101_TELE_S=0`, `SO101_TELE_EVERY=0`.
The leader gets no telemetry unless `LEROBOT_TELE_ROLES=follower,leader` / `SO101_TELE_LEADER=1`.

Past rates: the idle 5 Hz is what the page can use (the dashboard polls at 1 Hz and draws 5 Hz rows); going to 15 Hz idle would cost 7 % of the bus for no
visible gain, and in a recording anything above 4 Hz starts to be a measurable share of the control frame. Temperature and voltage change over seconds;
load, current and sag change per frame, which is why a pick is sampled at 4 Hz plus a forced read at each waypoint's end, and why the per-pick MLflow numbers are
peaks (max load, max current, max sag), not averages.

## MLflow

Every pick run (`so101-picks`) and autoloop cycle run (`so101-autoloop`) gets, when the Jetson sent telemetry, `tele_samples`, `tele_total_a_max`,
`tele_volt_min_v`, `tele_error_flags` and, for each joint `J`: `tele_temp_max_c_J`, `tele_temp_rise_c_J`, `tele_load_abs_max_pct_J`, `tele_cur_max_a_J`,
`tele_sag_abs_max_J` (any time) and `tele_sag_hold_max_J` (only while holding, not moving: the gravity sag), plus a `servo_telemetry.json` artifact. The autoloop
session (parent) run gets the same over all its cycles. With no telemetry in the rows (old Jetson files, `SO101_TELE_S=0`) nothing is logged and nothing else changes.
`python tools/autoloop.py --dry-run` exercises the whole path on the pretend arm (note: it writes to the `so101-autoloop-dry` experiment in the Mac store).

## Checking it on the real arm (read-only)

`jetson/verify_telemetry.py` (deploy it first) holds `.arms_busy`, opens the follower bus the way `arm_state.py` does (torque untouched, nothing written), and:

1. reads **every** register in the list above, per servo, and prints `reads back` / `NO ANSWER` next to what the repo already proved ("seen") and what it did not ("to verify");
2. checks the one-packet block against single-register reads of the same servos;
3. times the block read against a `Present_Position` read (the real numbers for the table above);
4. with `--meter-volts V`: the servos' voltage against a multimeter on the supply; with `--meter-amps A`: the sum of the six currents against a clamp meter or the bench
   supply's display at the same moment, and prints the corrected mA-per-count if the 6.5 is wrong;
5. with `--hand-move 10` (after `so101-free`): torque is off, move joints by hand for 10 s; speed, moving flag and position must respond (untouched joints are skipped);
6. checks `Status` reads 0 at rest.

Not provable without a fault: the error bits. The first real overload or over-temperature trip is that check (busheal logs `OVERLOAD`; the panel's flag list
should show it at the same moment). Load and current under motion are confirmed by the first `pick_seq.py` run: its `tele` rows should show the lift's load and current
rise on the way to `grasp` against an idle sample.

## Fake arm and demo

`tools/fake_servo_bus.py` is a pretend six-servo bus (gravity load, current, heating, supply sag, injectable faults) whose block reads go through the real decoder.
`python tools/telemetry_demo.py [port] [--hot lift]` serves the real dashboard page with a fake `/api` (the UI-test stub) and `/api/tele` from the real store fed by that bus,
on a port you choose (not 8765, 8766 or 9333). It opens no serial port, camera or other network port. `tests/ui/test_telemetry_panel.py` drives the same data in headless Chrome
(`SO101_TELE_SHOT=/path.png` saves a screenshot).
