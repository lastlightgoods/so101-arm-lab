#!/usr/bin/env python3
"""verify_telemetry.py - which servo registers actually read back on THIS arm (docs/telemetry.md, "to verify").

Read-only. It opens the follower bus the way arm_state.py does (torque untouched, never written, nothing changed on any servo), reads every
register in servo_telemetry.REGISTERS one servo at a time, then times the one-packet block read the dashboard uses, then checks the block against
the single reads and, with a meter, the voltage and current scale.

  python verify_telemetry.py                         # the register sweep + timing, arm at rest (about 15 s)
  python verify_telemetry.py --meter-volts 12.04     # also: the servo voltage reading vs a multimeter on the supply
  python verify_telemetry.py --meter-amps 0.62       # also: the current scale (6.5 mA per count?) vs a clamp meter / bench supply at the same moment
  python verify_telemetry.py --hand-move 10          # also: torque OFF in FREE mode first (so101-free), then move the joints by hand for 10 s:
                                                     #       speed, moving flag and position must respond
  python verify_telemetry.py --json out.json         # the full result as JSON (default: print the table only)

Holds ~/so101/.arms_busy while it runs, like pick_seq.py, so arm_state.py lets go of the port; refuses to run while a lerobot job has the arm.
Exit status 0 = every polled register read on every servo and the block matched the single reads; 1 = something did not (the table says what).
What this cannot prove without a fault: the Status error bits (they read 0 at rest); the first real overload or over-temperature trip is the check.
"""
import argparse, json, os, statistics, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import servo_telemetry as st

BUSY = os.path.expanduser("~/so101/.arms_busy")


def single_reads(bus, names=None):
    """{joint: {register name: {"ok": bool, "value": int|None, "addr", "n"}}} reading each register on its own, one servo at a time."""
    out = {}
    for n in names or list(bus.motors):
        rows = {}
        for name, addr, size, kind, meaning, polled, status in st.REGISTERS:
            if size not in (1, 2): rows[name] = {"ok": False, "value": None, "addr": addr, "n": size, "why": "size"}; continue
            try: v, comm, err = bus._read(addr, size, bus.motors[n].id, raise_on_error=False)
            except Exception as e: rows[name] = {"ok": False, "value": None, "addr": addr, "n": size, "why": type(e).__name__}; continue
            ok = bus._is_comm_success(comm); rows[name] = {"ok": ok, "value": int(v) if ok else None, "addr": addr, "n": size, "err_bits": int(err)}
        out[n] = rows
    return out


def time_block(bus, n=100):
    """Milliseconds per one-packet block read and per Present_Position sync read (what the control loop does), n of each."""
    ids = [m.id for m in bus.motors.values()]; blk, pos = [], []
    for _ in range(n):
        t = time.perf_counter(); st.read_blocks(bus, ids); blk.append((time.perf_counter() - t) * 1000)
    for _ in range(n):
        t = time.perf_counter(); bus.sync_read("Present_Position", normalize=False); pos.append((time.perf_counter() - t) * 1000)
    q = lambda xs: {"mean": round(statistics.mean(xs), 2), "p95": round(sorted(xs)[int(len(xs) * 0.95) - 1], 2), "max": round(max(xs), 2)}
    return {"block_ms": q(blk), "position_ms": q(pos), "n": n}


def idle_samples(bus, seconds, hz=10.0, sleep=time.sleep):
    out = []
    for _ in range(max(1, int(seconds * hz))):
        s = st.read_sample(bus)
        if s: out.append(s)
        sleep(1.0 / hz)
    return out


def hand_move(bus, seconds, hz=20.0, sleep=time.sleep):
    """Torque must be off (so101-free). Per joint: the most the position, speed and moving flag responded while you moved it by hand."""
    rng, vmax, moved = {}, {}, {}
    for _ in range(max(1, int(seconds * hz))):
        s = st.read_sample(bus)
        for n, d in ((s or {}).get("j") or {}).items():
            if d.get("pos") is not None: lo, hi = rng.get(n, (d["pos"], d["pos"])); rng[n] = (min(lo, d["pos"]), max(hi, d["pos"]))
            vmax[n] = max(vmax.get(n, 0.0), abs(d["vel_dps"])); moved[n] = moved.get(n, 0) or d["moving"]
        sleep(1.0 / hz)
    return {n: {"pos_range": round(rng[n][1] - rng[n][0], 1), "vel_max_dps": vmax[n], "moving_flag_seen": bool(moved[n]), "responds": rng[n][1] - rng[n][0] > 3 and vmax[n] > 5 and bool(moved[n])} for n in rng}


def evaluate(singles, block_samples, timing, meter_volts=None, meter_amps=None, hand=None):
    """The verdicts: per register 'reads' / 'no answer', block vs single reads, meter checks, hand-move response. Returns (rows, ok)."""
    rows, ok = [], True
    for name, addr, size, kind, meaning, polled, status in st.REGISTERS:
        got = [d[name]["ok"] for d in singles.values()]; good = all(got) and bool(got)
        rows.append({"register": name, "addr": addr, "bytes": size, "kind": kind, "polled": polled, "was": status, "reads_on": f"{sum(got)}/{len(got)} servos", "verdict": "reads back" if good else "NO ANSWER"})
        if polled and not good: ok = False
    last = block_samples[-1] if block_samples else None
    checks = []
    if not last: checks.append(("block read", False, "the block read returned nothing")); ok = False
    else:
        for n, d in last["j"].items():   # the block against single reads of the same servo (same second, so tolerances not equalities)
            s1 = singles.get(n, {})
            if not s1: continue
            tv, vv = s1["Present_Temperature"]["value"], s1["Present_Voltage"]["value"]
            good = tv is not None and abs(tv - d["temp_c"]) <= 2 and vv is not None and abs(vv / 10.0 - d["volt_v"]) <= 0.3
            checks.append((f"block vs single reads: {n}", good, f"temp {d['temp_c']} vs {tv}, volt {d['volt_v']} vs {None if vv is None else vv / 10.0}")); ok = ok and good
        if meter_volts is not None:
            v = statistics.median(s["volt_min"] for s in block_samples); good = abs(v - meter_volts) <= 0.4
            checks.append(("supply voltage vs meter", good, f"servos say {v:.1f} V, meter {meter_volts:.2f} V")); ok = ok and good
        if meter_amps is not None:
            a = statistics.median(s["total_a"] for s in block_samples); good = meter_amps > 0 and abs(a - meter_amps) <= max(0.15, 0.25 * meter_amps)
            scale = round(st.MA_PER_COUNT * meter_amps / a, 2) if a > 0 else None
            checks.append(("current scale vs meter", good, f"sum of servo currents {a:.2f} A at {st.MA_PER_COUNT} mA/count, meter {meter_amps:.2f} A" + (f"; suggested scale {scale} mA/count" if scale else ""))); ok = ok and good
        stat = {n: d["status"] for n, d in last["j"].items()}; checks.append(("status bits at rest", not any(stat.values()), f"Status registers: {stat}")); ok = ok and not any(stat.values())
    if hand:
        for n, h in hand.items():   # a joint nobody touched is skipped, not failed
            touched = h["pos_range"] >= 1.0
            checks.append((f"hand move: {n}", h["responds"] if touched else None, f"moved {h['pos_range']} deg, peak {h['vel_max_dps']} deg/s, moving flag {'seen' if h['moving_flag_seen'] else 'never'}" + ("" if touched else " (not moved by hand: skipped)")))
            ok = ok and (h["responds"] or not touched)
    return rows, checks, ok


def print_report(rows, checks, timing, ok):
    print(f"{'register':32} {'addr':>4} {'n':>2}  {'was':9} {'verdict':10} reads on")
    for r in rows: print(f"{r['register']:32} {r['addr']:>4} {r['bytes']:>2}  {r['was']:9} {r['verdict']:10} {r['reads_on']}{'  (polled)' if r['polled'] else ''}")
    print()
    for name, good, detail in checks: print(f"  {'SKIP' if good is None else 'PASS' if good else 'FAIL'}  {name}: {detail}")
    if timing: print(f"\n  timing over {timing['n']} reads: block {timing['block_ms']}  vs  Present_Position {timing['position_ms']}   (ms; 'mean p95 max')")
    print("\nALL POLLED REGISTERS VERIFIED" if ok else "\nSOMETHING DID NOT VERIFY (see FAIL / NO ANSWER above)")


def connect(port, model="sts3215"):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    motors = {n: Motor(i, model, MotorNormMode.DEGREES) for i, n in enumerate(st.JOINTS, 1)}
    bus = FeetechMotorsBus(port, motors); bus.connect(handshake=False); return bus


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=os.environ.get("FOLLOWER", "/dev/so101_follower")); ap.add_argument("--model", default="sts3215", choices=sorted(st.MODELS), help="motor model (sts3250 for the special-edition arm)")
    ap.add_argument("--seconds", type=float, default=8.0, help="idle sampling time"); ap.add_argument("--meter-volts", type=float); ap.add_argument("--meter-amps", type=float)
    ap.add_argument("--hand-move", type=float, default=0.0, metavar="SECONDS"); ap.add_argument("--json", default=None); ap.add_argument("--timing-reads", type=int, default=100)
    a = ap.parse_args(argv)
    if subprocess.run(["pgrep", "-f", "lerobot-(record|teleoperate|rollout|calibrate|replay)"], capture_output=True).returncode == 0: raise SystemExit("a lerobot job holds the arm; so101-stop first")
    made = not os.path.exists(BUSY)
    if made: open(BUSY, "w").close(); time.sleep(1.2)
    try:
        bus = connect(a.port, a.model)
        try:
            singles = single_reads(bus); timing = time_block(bus, a.timing_reads); samples = idle_samples(bus, a.seconds); hand = hand_move(bus, a.hand_move) if a.hand_move else None; limits = st.read_limits(bus)
        finally:
            try: bus.disconnect(disable_torque=False)
            except Exception: pass
    finally:
        if made:
            try: os.remove(BUSY)
            except FileNotFoundError: pass
    rows, checks, ok = evaluate(singles, samples, timing, a.meter_volts, a.meter_amps, hand); print_report(rows, checks, timing, ok)
    if a.json: json.dump({"ok": ok, "registers": rows, "checks": [{"name": n, "pass": g, "detail": d} for n, g, d in checks], "timing": timing, "limits": limits}, open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__": main()
