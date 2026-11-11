# Stronger SO-101: motors, designs, materials, lab impact, staged plan

Research memo, 11 Nov 2026 (Bend, OR). Read-only; nothing bought. Every number below is from a page fetched during this research (URL beside it); "not stated" means the fetched page had no figure. Figures seen only in search snippets are listed under "Could not verify".

## 1. Motors

### 1a. Feetech bus servos (same 1 Mbaud half-duplex TTL bus, same 45 x 25 x 35 mm body)

| Servo | Stall torque | No-load speed | Voltage | Stall / idle current | Weight | Case, gears | Feedback | LeRobot 0.6.1 table | Price seen |
|---|---|---|---|---|---|---|---|---|---|
| STS3215 7.4 V (C001) | 19.5 kg.cm @ 7.4 V; README says 16.5 kg.cm at 6 V | 52 rpm | 4-7.4 V (Waveshare) | 2.5 A / 150 mA (Waveshare) | 55 g (Core Electronics) | PA+GF, steel | load, pos, speed, V, I, temp | `sts3215` (model no. 777) | $15 overseas (Feetech); $13.89 (SO-ARM100 BOM); $16.99-21.99 (Waveshare) |
| STS3215 12 V (C018) | 30 kg.cm @ 12 V, rated 10 kg.cm | 0.222 s/60 deg = 45 rpm | 6-12.6 V | 2.7 A / 180 mA | 55 +/- 1 g | PA+GF, steel | same | `sts3215` | as above |
| STS3235 (C001) | 30 kg.cm @ 12 V, rated 10 | 0.222 s/60 deg | 6-12 V | 2.7 A / 190 mA | 70.5 +/- 1 g | aluminium, steel | load, pos, speed, V, temp | **not in 0.6.1**; PR #4898 (open) adds model no. 2057 | not stated on Feetech page |
| STS3250 (C001/C002) | 50 kg.cm @ 12 V, rated 16 kg.cm; measured 48 peak, 25 sustained before protection | 0.133 s/60 deg = 75 rpm | 6-12 V | 4.2 A / 280 mA | 74.5 +/- 1 g | aluminium, steel, coreless motor, 25T 5.9 mm spline | load, pos, speed, V, temp | `sts3250` (model no. 2825) | not stated on Feetech page; "~$55" in LeRobotDepot motor guide |
| ST3215-HS (Waveshare) | 20 kg.cm @ 12 V | 106 rpm | 6-12.6 V | 2.7 A | not stated | not stated | not stated | would ping as whatever model no. it reports; unverified | $21.99 (Waveshare, search snippet) |
| HLS3930M / HLS3950M (Robonine test) | 35 / 50 kg.cm; HLS3950M measured 63 kg.cm peak at 6.0 A | not stated | 12 V bench; 9-12.6 V (author comment) | 6.0 A peak (3950M) | 70.5 / 74.5 g | aluminium | current loop, direct torque command | **not in any LeRobot table** | not stated |

Sources: Feetech STS3215 7.4 V https://www.feetechrc.com/en/2020-05-13_56655.html ; STS3215 12 V https://www.feetechrc.com/525603.html ; Waveshare ST3215 https://www.waveshare.com/st3215-servo.htm ; Core Electronics https://core-electronics.com.au/feetech-sts3215-smart-servo.html ; STS3235 https://www.feetechrc.com/12v-30kg-metal-shell-metal-tooth-iron-core-motor-magnetic-coding-double-shaft-ttl-series-steering-gear.html ; STS3250 https://www.feetechrc.com/en/562636.html ; Robonine STS3250 evaluation https://robonine.com/feetech-sts3250-smart-actuator-evaluation-of-accuracy-torque-and-backlash/ ; Robonine five-servo comparison https://robonine.com/smart-serial-servo-motor-comparison-robotics/ ; LeRobotDepot https://github.com/maximilienroberti/lerobotdepot ; LeRobot v0.6.1 table https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/motors/feetech/tables.py ; PR #4898 https://github.com/huggingface/lerobot/pull/4898

Notes on fit and protocol:
- LeRobot 0.6.1 `tables.py` ships exactly `sts3215` (777), `sts3250` (2825), `sm8512bl` (11272), `scs0009` (1284) plus generic `sts_series`/`sms_series`/`scs_series`, all sharing one STS/SMS control table and 4096-count resolution. LeRobot 0.7.0 (released 2026-11-05) moved motor definitions into the `rustypot` dependency; its hardware doc still lists only those four Feetech models.
- `so_follower.py` in 0.6.1 hard-codes all six motors as `"sts3215"`, and `MotorsBus._assert_motors_exist()` compares the pinged model number with the table entry and raises `RuntimeError` on mismatch. So an STS3250 on ID 2 or 3 works only after the motor dict says `"sts3250"` (edit, or a plugin robot class as the LeKiwi/PincOpen post and the Robonine plugin do).
- STS3250 vs STS3215 mechanically: identical nominal A/B/C dimensions (45.22 x 24.72 x 35 mm vs 45.2 x 24.7 x 35 mm) and a 25T horn. Robonine's author calls the mounting pattern and horn compatible, "near drop-in", but "slightly deeper"; the HF LeKiwi post says it is "slightly wider" and needed a hammer to seat in PLA. Plan on reprinting the shoulder/elbow motor shells.
- Thermal: STS3215 rose to 48 C and held for an hour at 1 kg x 15 cm, then hit 71 C and overheated after ~90 min of +/-90 deg oscillation (https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/). STS3250 went 40 to 70 C in 8 min at 2 kg x 100 mm, with torque disabled at 70 C; in the 50 %-load comparison it was the thermal outlier (+25 C internal) and the 100 % test ended with a burnt motor.

### 1b. Non-Feetech options LeRobot already drives

| Motor / arm | Torque | Price | Bus / board | LeRobot |
|---|---|---|---|---|
| Dynamixel XL430-W250-T | 1.5 N.m (15.3 kg.cm) @ 12 V, 1.4 A, 61 rpm, 57.2 g | $50 (Koch v1.1 BOM) | TTL multidrop, U2D2 / OpenRB | `xl430-w250` in 0.6.1 dynamixel table; Koch v1.1 shoulders |
| Dynamixel XM430-W350-T | 4.1 N.m (41.8 kg.cm) @ 12 V, 2.3 A, 46 rpm, 82 g, current feedback | $310.39 (RobotShop, in stock) | TTL/RS-485 | `xm430-w350` in 0.6.1 table |
| Dynamixel XL330-M288 / M077 | not fetched | $24 each (Koch BOM) | TTL, 5 V | Koch elbow/wrist/gripper, OMX |
| Damiao / RobStride CAN (Seeed reBot B601) | not stated per motor; arm payload 1.5 kg (DM) / 2.5 kg (RS) | $1,499 / $1,699 follower (LeRobotDepot) | CAN; `pip install -e ".[rebot]"`; RobStride needs SocketCAN | upstream `rebot_b601` in 0.7.0 docs |

Sources: https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/ ; https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/ ; https://www.robotshop.com/products/robotis-dynamixel-xm430-w350-t-smart-servo-actuator ; https://github.com/jess-moss/koch-v1-1 ; https://github.com/Seeed-Projects/reBot-DevArm ; https://huggingface.co/docs/lerobot/en/rebot_b601

### 1c. What the shoulder lift and elbow actually need

Inputs: link masses and joint offsets from `so101_new_calib.urdf` (https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf): upper arm 0.103 kg, lower arm 0.104 kg, wrist 0.079 kg, gripper 0.087 kg + jaw 0.012 kg; joint-to-joint distances 116, 135, 64 and 98 mm (shoulder-lift axis to gripper frame = 413 mm, longer than the ~330 mm quoted in the brief; Robonine measures 441 mm horizontal reach from the base). Assumptions: arm fully horizontal (worst case), each link's COM at mid-link, payload at the gripper frame, static only, g = 9.81.

| Payload | Shoulder-lift static torque | Elbow static torque |
|---|---|---|
| 0 g | 0.82 N.m = 8.3 kg.cm | 0.44 N.m = 4.5 kg.cm |
| 100 g | 1.22 N.m = 12.5 kg.cm | 0.73 N.m = 7.4 kg.cm |
| 250 g | 1.83 N.m = 18.7 kg.cm | 1.17 N.m = 11.9 kg.cm |
| 500 g | 2.84 N.m = 29.0 kg.cm | 1.90 N.m = 19.3 kg.cm |

Reading: with 250 g the shoulder sits at 18.7 kg.cm, i.e. 62 % of the 12 V STS3215 stall figure and above its 10 kg.cm *rated* torque, which is why the stock arm sags and heats. On an STS3250 the same load is 37 % of stall and right at its 16 kg.cm rated torque (Robonine measured 25 kg.cm sustained before protection). The elbow is comfortable on an STS3250 at any payload here, and borderline on a 12 V STS3215 at 250 g once acceleration is added. Caveat: the URDF masses appear to include the servos (links weigh ~100 g vs 55 g per servo) but the file does not say so; STS3250s add ~20 g each at the shoulder and elbow.

## 2. Existing stronger / compatible designs

| Design | What changed | Payload / reach | Cost | LeRobot status | Link |
|---|---|---|---|---|---|
| SO-ARM102 (Robonine) | PET-CF topology-optimised links; follower 2x STS3235 (pan, wrist flex), 2x STS3250 (shoulder lift, elbow), 2x ST-3215-C049 1:191 (wrist roll, gripper); 85 mm two-jaw parallel gripper; link lengths 85/119/139/62/132 mm | 300 g rated (SO-101: 150 g); 483 mm; 3.9 mm tip deflection vs 7.55 mm SO-101 | $389 kit (unassembled, ships late Oct 2026 from China); DIY BOM $342 + $46 filament | plugin `lerobot_robot_so_arm102[feetech]==0.1.0a1`, pinned to `lerobot==0.6.1`, type `so_arm102_follower`, leader stays `so101_leader` | https://robonine.com/shop/so-arm102-robotic-arm-kit/ , https://github.com/roboninecom/SO-ARM102 |
| SO-101 + STS3250 on joints 1-4 (LeKiwi/PincOpen blog) | servo swap only, no new prints; P-gain 16 -> 14, heavy joints P = 10 and accel 200 to stop brown-outs | not stated | not stated | config edit on 0.5.2; "LeRobot 0.6 auto-discovers plugins" | https://huggingface.co/blog/zuoxingdong/mobile-manipulation-lekiwi-pincopen |
| Koch v1.1 | Dynamixel XL430 x2 + XL330 x4 follower | not stated | $278 follower + $199 leader | native | https://github.com/jess-moss/koch-v1-1 |
| Moss v1 | Feetech, 5 DoF | not stated | $288 pair | listed | LeRobotDepot |
| LeKiwi | SO-101 arm on a 3-wheel base | not stated | $482 (12 V build) | native | LeRobotDepot |
| AM-ARM200 | 4x STS3215 + 3x STS3095, 6 DoF | 1 kg advertised | ~$388 pair | community fork | LeRobotDepot |
| OpenArm 2.0 (Enactic) | Damiao CAN, 7 DoF bimanual | not stated | from $6,500 | native, `[damiao]` extra | https://huggingface.co/docs/lerobot/en/openarm |
| Seeed reBot B601-DM / -RS | Damiao / RobStride CAN, 6 DoF + parallel gripper | 1.5 kg, 767 mm, 4.5 kg, 24 V / 2.5 kg, 754 mm, 6.7 kg, 48 V | $1,499 / $1,699 follower | upstream | https://github.com/Seeed-Projects/reBot-DevArm |
| Trossen WidowX AI | integrated actuators, Ethernet, 24 V 15 A | 1.5 kg, 0.769 m, 4 kg, joints 0-2 effort 27 N.m | EUR 18,000 ex VAT at a reseller (4-6 week delivery); Trossen store not reachable | native | https://docs.trossenrobotics.com/trossen_arm/main/specifications/wxai.html , https://www.generationrobots.com/en/404333-widowx-ai.html |

Giraffe and ARX do not appear in LeRobotDepot and were not researched further. "SO-100/101 Pro" is a vendor label, not a distinct open design in any fetched source.

## 3. Materials on the P2S (hardened nozzle, AMS 2 Pro)

| Filament | Bending modulus XY | HDT 0.45 MPa | Drying | Nozzle / plate | AMS 2 Pro | MSRP (1 kg) | Source |
|---|---|---|---|---|---|---|---|
| PLA Basic | 2750 MPa (Young's 2580) | 57 C; Vicat 57 C; Tg 60 C | oven 50 C 8 h | any; 35-45 C bed | yes | - | TDS V3.0 https://wiki.bambulab.com/filament-acc/abs-asa-pc/bambu_pla_basic_technical_data_sheet.pdf |
| PLA Tough+ | 2140 MPa | 61 C | per wiki (PLA: 55 C 8 h oven / 45 C 12 h AMS 2 Pro) | any; "compatible with all printers, AMS, build plate, and nozzles" | yes | $20.99 ($13.64 at 10+) | https://us.store.bambulab.com/products/pla-tough-upgrade |
| PETG Basic | 1950 MPa (Young's 2780) | 71 C (68 C @ 1.8 MPa); Vicat 69 C | 65 C 8 h oven or AMS 2 Pro | 230-260 C; 65-75 C bed; any nozzle | yes | $13.99 ($9.09 at 10+) | TDS V3.0 https://store.bblcdn.com/s1/default/cb94589bf7994fdcbfa833badefae9cd/Bambu_PETG_Basic_Technical_Data_Sheet.pdf , https://us.store.bambulab.com/products/petg-basic |
| PETG-CF | not fetched (TDS blocked) | not stated | "dry thoroughly" | hardened steel required, 0.6 mm recommended; 0.2 mm no; 60-80 C bed | AMS 2 Pro / HT / AMS yes, AMS lite no | $31.99 ($20.79 at 10+) | https://us.store.bambulab.com/products/petg-cf |
| PLA-CF | not fetched | not stated | dry before use | stainless not recommended; 0.2 mm no | "All AMS Series Compatible" | $31.99 ($20.79 at 10+) | https://us.store.bambulab.com/products/pla-cf |
| PAHT-CF | 4230 MPa dry XY (3640 wet) | 194 C | 80 C 12 h | 0.6 mm recommended, stainless not recommended; enclosure required; glue stick, 90-110 C bed | AMS 2 Pro / HT / AMS yes, AMS lite no | page shows $29.99-$49.99; MSRP line not captured | https://us.store.bambulab.com/products/paht-cf |
| PA6-CF | 5460 MPa dry XY (3560 wet) | 186 C | high; "highly moisture-absorbent" | hardened, 0.2 mm no | "All AMS Series NOT Compatible" (external spool) | page shows $25.79-$42.99 | https://us.store.bambulab.com/products/pa6-cf |
| PPA-CF | not fetched | - | - | P2S FAQ lists it as supported | - | - | https://wiki.bambulab.com/en/p2s/manual/p2s-faq |
| TPU for AMS (68D) | - | - | dry before use | 230 C; 0.2 mm no | all AMS yes | page shows $20.99-$34.99 | https://us.store.bambulab.com/products/tpu-for-ams |

Bambu's rules that matter here (https://wiki.bambulab.com/en/general/filament-guide-material-table): for any CF/GF filament "always use a hardened steel nozzle", 0.6 mm first choice, 0.4 mm second (max ~100 mm/s); PLA parts "at about 50 C or higher ... may slowly degrade, soften, and deform". AMS 2 Pro store page (https://us.store.bambulab.com/products/ams-2-pro): supported "Bambu PLA-CF/PAHT-CF/PETG-CF ... and TPU for AMS"; not supported "TPE, generic TPU ... Bambu PET-CF/TPU 95A, and other filament that contains carbon fiber or glass fiber"; max drying 65 C. The P2S ships with a hardened steel nozzle, 300 C hotend, 110 C bed (P2S FAQ above); the AMS 2 Pro has the hardened extruder path implied by the CF support list, but the wiki warns abrasive filament still wears the AMS PTFE over time. TPU: the TPU guide table (https://wiki.bambulab.com/en/knowledge-sharing/tpu-printing-guide) marks TPU 95A HF as not usable in AMS, AMS lite and AMS 2 Pro, usable on H2/X/P-series printers via external spool; the P2S FAQ says print TPU from the spool holder. No official Bambu page listing third-party TPU brands was found; owner reports of Overture TPU 95A on the P2S via the standalone spool holder are in a third-party article (https://bambuhub.net/articles/overture-tpu-on-bambu), not a Bambu compatibility page.

Servo heat vs plastic: the STS3215 plateaued at 48 C and reached 71 C in the Robonine tests; the STS3250 reached 70 C in 8 min at high load. PLA Tough+ (HDT 61 C, Vicat ~62 C) is therefore marginal for the shells that clamp a hot servo; PETG Basic (71 C) is better; PETG-CF and PAHT-CF (194 C) are safe.

SO-ARM100 print guidance (https://github.com/TheRobotStudio/SO-ARM100): PLA+, 0.4 mm nozzle at 0.2 mm layers (or 0.6 mm at 0.4 mm), 15 % infill, supports everywhere except slopes over 45 deg and never in horizontal screw holes. The repo does not label parts as load-bearing; from the kinematics the base, shoulder motor holder, upper arm, lower arm and the servo shells carry the bending load, the wrist/gripper parts carry the pinch load.

Recommendation per part class (P2S Combo): base, shoulder holder, upper and lower arm: PETG-CF via AMS 2 Pro (dry first, 0.6 mm or 0.4 mm hardened nozzle); if deflection still matters, PAHT-CF external-dried at 80 C 12 h. Servo shells / motor mounts (hot zone): PETG-CF or PAHT-CF, not PLA. Gripper body and jaws: PETG-CF; fingertip pads: TPU 95A from the external spool holder (brand unverified), or Bambu TPU for AMS if the softer pad is not needed. Leader arm and non-structural covers: PLA Tough+ or PETG Basic (already in the cart).

## 4. What changes in the lab

- Power: two STS3250 stall at 4.2 A each plus four STS3215 at 2.7 A is 19.2 A worst case at 12 V; in practice Robonine measured 1.26 A average / 3.28 A peak for the SO-ARM102 follower under 300 g on a 12 V 5 A supply. The SO-ARM100 README already calls for "12 V 5 A+" for 12 V servos. Budget a 12 V 8-10 A supply (price not researched) and keep peaks down with the LeKiwi post's lower P-gain / acceleration settings.
- Bus board: Waveshare Bus Servo Adapter (A) is $4.99, 9-12.6 V, current rating "not stated" (https://www.waveshare.com/bus-servo-adapter-a.htm). Robonine's load test (https://robonine.com/testing-waveshare-serial-bus-servo-driver-boards-under-load/) rates it "up to about 5 A continuous"; the screw-terminal MOSFET hit 78-80 C at 7 A and ~115 C at 10 A, and some rev 1.1 boards lost their TVS diode. Use the barrel jack path, or the Bus Servo Driver HAT (5-7 A sustained).
- LeRobot 0.6.1: STS3250 is known (model 2825) but the follower class must declare it; calibration files keep the same 4096 resolution. STS3235 needs PR #4898 or the Robonine plugin. Firmware: no change needed for the bus; confirm each new servo's baud (1 Mbaud default) and ID with `lerobot-setup-motors`.
- Kinematics: a servo-only swap keeps link lengths, so the URDF, dashboard model and base pose stay valid but joint zero offsets must be redone (new horns, new calibration). Any reprinted link, servo shell or gripper changes the URDF (lengths and masses), the dashboard 3D model, the jog-and-fit base pose and joint offsets, and, if the wrist/gripper link moves the icSpring camera, the hand-eye calibration.
- Gripper: new jaws mean new `RANGE_0_100` calibration limits and new grasp thresholds; the parallel gripper has its own comparison doc.
- Data: the HF post claims "older datasets stay valid" for a pure motor swap (same control table and resolution). That holds for action/state encoding only; dynamics, reach and gripper geometry differ once prints change, so treat the rebuilt arm as a new robot ID with its own datasets and policies.

## 5. Recommendation (staged)

| Stage | What | Verified cost | What it buys |
|---|---|---|---|
| 1 | Reprint load-bearing links and servo shells in PETG-CF (or PAHT-CF), install the parallel gripper (see the separate gripper doc), recalibrate | PETG-CF $31.99/kg, PETG Basic $13.99, PLA Tough+ $20.99 (Bambu US store); gripper hardware per other doc | Robonine's stiffer frame halved tip deflection (7.55 -> 3.9 mm); a two-jaw parallel gripper is what fixes the 10 mm nut slipping; fewer thermal creep problems at the servo mounts |
| 2 | STS3250 on shoulder lift (ID 2) and elbow (ID 3); 12 V 8-10 A supply; declare `sts3250` in the follower config; lower P-gain/accel on those joints | STS3250: no fetched vendor price (LeRobotDepot "~$55" each); supply not priced | shoulder torque margin goes from 62 % to 37 % of stall at 250 g; 75 rpm vs 45 rpm; this is the stage that makes autonomous pick-and-place episodes repeatable without overheating stops |
| 3 | Remaining joints: STS3235 on pan and wrist flex (needs PR #4898 or plugin) or STS3250 throughout; keep STS3215 on roll and gripper | not fetched | aluminium cases and lower backlash (0.44 deg vs 0.77 deg measured) on the joints that set tip placement; needed for sub-millimetre nut pickup |
| 4 | Only if payload above ~300 g is really needed: SO-ARM102 kit ($389) as a drop-in LeRobot arm, or a CAN-class arm (reBot B601-DM $1,499, 1.5 kg) | as listed | more payload and reach, but new URDF, new datasets, new power and bus hardware |

Stages 1-2 are the sensible target for the self-learning pick-and-place plan: the stiff frame and parallel gripper fix the nut pick, and the two STS3250s give the thermal headroom that long unattended episode collection needs.

## Could not verify

- Feetech official prices for STS3235/STS3250 (not on Feetech pages). Seller prices seen only in search snippets, pages rate-limited: ArmDojo STS3250 $84, OpenELAB $73.99, WowRobo $43 (sold out), eBay STS3235 $65.54.
- Dynamixel XL430-W250-T current USD price (Robotis and RobotShop pages unreachable; Koch BOM $50 used).
- Trossen WidowX AI store price (store unreachable; only a EUR reseller figure).
- SO-101 arm weight (Hiwonder page unreachable; snippet 697 g leader / 704 g follower).
- Bambu TDS for PETG-CF, PLA-CF and PPA-CF (CDN/reseller PDFs blocked), so their moduli/HDT are missing above.
- Whether STS3250 bodies fit the stock SO-101 shoulder/elbow shells without reprinting (two sources say tight/"slightly deeper").
- ST3215-HS and HLS-series (HLS3625 7.4 V 25 kg.cm, HLS3930M, HLS3950M) protocol compatibility with LeRobot's FeetechMotorsBus: no table entries, no community report found; HLS3625 figures exist only in search snippets.
- A Bambu-published list of third-party TPU 95A brands for the P2S.
