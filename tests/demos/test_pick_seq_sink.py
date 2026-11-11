"""jetson/pick_seq.py --tick-sink: OFF by default, and with it off the commands sent to the servos are byte for byte what main sent before the flag existed.
The reference is the pick_seq.py of main (commit 1db8807) BEFORE the change, frozen as pick_seq_reference.py (REFERENCE_MOVE is its move() verbatim), and both run on the
same machine at test time. No robot, no sleeping, no /dev/shm."""
import json, math, os
import pytest
from demohelpers import FakeRobot, PLAN, commands_hash, load, run_pick_seq

# why: sha256 goldens of the float reprs differ between macOS and glibc libm (math.cos in the easing), so the pre-change code is frozen and compared at test time, exact equality
@pytest.fixture
def ref_main(tmp_path, monkeypatch, capsys):
    """(hash, count) of the commands the frozen pre-change pick_seq.py sends for main() on PLAN from the fake start pose, --correct 0.8."""
    ref = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_seq_reference.py"), "pick_seq_reference")
    (tmp_path / "ref").mkdir(); rob, _ = run_pick_seq(ref, tmp_path / "ref", monkeypatch, capsys)
    return commands_hash(rob.sent), len(rob.sent)


def REFERENCE_MOVE(robot, cur, tgt, seconds, max_deg_s, J, sleep):
    biggest = max(abs(tgt[j] - cur[j]) for j in J); secs = max(seconds, biggest / max_deg_s); n = max(12, int(secs * 40))
    for i in range(1, n + 1):
        e = (1 - math.cos(math.pi * i / n)) / 2
        robot.send_action({f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * e for j in J}); sleep(secs / n)
    return secs


def move_cases(J):
    return [({j: 0.0 for j in J}, {j: float(i * 7 - 20) for i, j in enumerate(J)}, 1.5, 80.0), ({j: 1.0 for j in J}, {j: 1.0 for j in J}, 0.3, 80.0), ({j: -30.0 for j in J}, {j: 50.0 for j in J}, 0.5, 50.0)]


def test_move_without_a_sink_sends_exactly_what_main_sent(ps, monkeypatch):
    sleeps = []; monkeypatch.setattr(ps.time, "sleep", sleeps.append); new = []; ref_sleeps = []
    for cur, tgt, sec, v in move_cases(ps.J):
        a, b = FakeRobot(ps.J), FakeRobot(ps.J); s1 = ps.move(a, cur, tgt, sec, v); s2 = REFERENCE_MOVE(b, cur, tgt, sec, v, ps.J, ref_sleeps.append)
        assert a.sent == b.sent and s1 == s2; new += a.sent
    assert sleeps == ref_sleeps, "the pacing is unchanged too"


def test_main_without_the_flag_sends_the_pre_change_command_sequence(ps, ref_main, tmp_path, monkeypatch, capsys):
    rob, rows = run_pick_seq(ps, tmp_path, monkeypatch, capsys)
    assert (commands_hash(rob.sent), len(rob.sent)) == ref_main
    assert [r.get("wp") for r in rows if "wp" in r] == [w["name"] for w in PLAN["waypoints"]] and rows[-1].get("done") is True
    assert rob.reads == 1 + len(PLAN["waypoints"]), "flag off: the state is read only at the start and after each waypoint, as before"


def test_the_flag_does_not_change_a_single_command(ps, ref_main, tmp_path, monkeypatch, capsys):
    sink = tmp_path / "ticks.jsonl"; rob, _ = run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink", str(sink)])
    assert (commands_hash(rob.sent), len(rob.sent)) == ref_main, "with the sink ON the servos still get exactly the pre-change commands"


def test_off_by_default_writes_nothing_even_with_the_env_var_set(ps, tmp_path, monkeypatch, capsys):
    p = tmp_path / "env_ticks.jsonl"; monkeypatch.setenv("SO101_TICK_SINK", str(p)); created = []
    monkeypatch.setattr(ps, "TickSink", lambda path, *a, **k: created.append(path))
    run_pick_seq(ps, tmp_path, monkeypatch, capsys); assert created == [] and not p.exists()


def test_the_path_is_overridable_flag_value_then_env_then_default(ps, tmp_path, monkeypatch, capsys):
    paths = []
    class Spy:
        def __init__(self, path, *a, **k): paths.append(path)
        def before(self, robot): return None
        def after(self, *a): pass
        def close(self): pass
    monkeypatch.setattr(ps, "TickSink", Spy); monkeypatch.delenv("SO101_TICK_SINK", raising=False)
    run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink", "/x/flag.jsonl"])
    monkeypatch.setenv("SO101_TICK_SINK", "/x/env.jsonl"); run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink"])
    monkeypatch.delenv("SO101_TICK_SINK"); run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink"])
    assert paths == ["/x/flag.jsonl", "/x/env.jsonl", ps.DEFAULT_TICK_SINK == "/dev/shm/so101_ticks.jsonl" and "/dev/shm/so101_ticks.jsonl"]


def read_lines(p): return [json.loads(l) for l in open(p)]


def test_ring_file_format_one_line_per_tick(ps, tmp_path, monkeypatch, capsys):
    sink = tmp_path / "ticks.jsonl"; rob, _ = run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink", str(sink)]); lines = read_lines(sink)
    head, ticks = lines[0], lines[1:]
    assert head["header"] == 1 and head["version"] == 1 and len(ticks) == len(rob.sent) == 124
    assert [t["seq"] for t in ticks] == list(range(1, 125)) and all(set(t) == {"seq", "t_mono", "t_wall", "wp", "action", "state"} for t in ticks)
    assert [t["action"] for t in ticks] == rob.sent, "action = what send_action returned"
    names = [w["name"] for w in PLAN["waypoints"]]; assert [t["wp"] for t in ticks][0] == names[0] and [t["wp"] for t in ticks][-1] == names[-1]
    assert all(b["t_mono"] >= a["t_mono"] for a, b in zip(ticks, ticks[1:]))
    # the state of tick i is the servo readback BEFORE command i: the fake servos follow the commands, so it is command i-1
    assert [t["state"] for t in ticks[1:]] == [t["action"] for t in ticks[:-1]]
    assert ticks[0]["state"] == {f"{j}.pos": v for j, v in {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 70.0, "wrist_roll": 0.5, "gripper": 0.0}.items()}


def test_the_sink_costs_one_extra_state_read_per_tick(ps, tmp_path, monkeypatch, capsys):
    rob, _ = run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink", str(tmp_path / "t.jsonl")]); assert rob.reads == 1 + len(PLAN["waypoints"]) + 124


def test_a_dead_sink_path_never_stops_the_arm(ps, ref_main, tmp_path, monkeypatch, capsys):
    rob, rows = run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink", str(tmp_path / "no" / "such" / "dir" / "t.jsonl")])
    assert (commands_hash(rob.sent), len(rob.sent)) == ref_main and rows[-1].get("done") is True


def test_one_failed_state_read_turns_the_sink_off_and_the_moves_go_on(ps, ref_main, tmp_path, monkeypatch, capsys):
    class Flaky(FakeRobot):
        def get_observation(self):
            if len(self.sent) == 30 and not getattr(self, "failed_once", False): self.failed_once = True; self.reads += 1; raise OSError("bus read failed once")   # the sink's read before tick 31
            return super().get_observation()
    rob = Flaky(ps.J, {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 70.0, "wrist_roll": 0.5, "gripper": 0.0})
    sink = tmp_path / "t.jsonl"; rob, rows = run_pick_seq(ps, tmp_path, monkeypatch, capsys, argv=["--tick-sink", str(sink)], robot=rob)
    assert (commands_hash(rob.sent), len(rob.sent)) == ref_main and rows[-1].get("done") is True
    assert len([t for t in read_lines(sink) if "seq" in t]) == 30, "the sink stopped at the failure and never raised into the motion code"


def test_the_sink_survives_a_write_error_midway(ps, tmp_path, monkeypatch, capsys):
    sink = ps.TickSink(str(tmp_path / "t.jsonl")); r = FakeRobot(ps.J)
    sink.after(ps.read(r), {f"{j}.pos": 1.0 for j in ps.J}, "a"); sink.f.close()   # the file is closed under it: the next write raises ValueError inside
    sink.after(None, {f"{j}.pos": 2.0 for j in ps.J}, "a"); assert sink.f is None and sink.err
    sink.after(None, {f"{j}.pos": 3.0 for j in ps.J}, "a"); sink.close()   # and later calls are no-ops


def test_a_crash_leaves_a_readable_file_with_at_most_one_torn_line(ps, dr, tmp_path):
    sink = ps.TickSink(str(tmp_path / "t.jsonl")); r = FakeRobot(ps.J)
    for i in range(5): sink.after(ps.read(r), {f"{j}.pos": float(i) for j in ps.J}, "a")
    sink.f.write('{"seq": 6, "t_mono": 1.0, "wp"'); sink.f.flush()   # killed mid-write
    buf = dr.TickBuffer(); assert buf.poll_file(str(tmp_path / "t.jsonl")) == 5 and [t["seq"] for t in buf.ticks] == [1, 2, 3, 4, 5]


def test_the_recorder_reads_what_pick_seq_wrote_incrementally(ps, dr, tmp_path):
    p = str(tmp_path / "t.jsonl"); sink = ps.TickSink(p); r = FakeRobot(ps.J); buf = dr.TickBuffer(); assert buf.poll_file(p) == 0
    for i in range(3): sink.after(ps.read(r), {f"{j}.pos": float(i) for j in ps.J}, "a")
    assert buf.poll_file(p) == 3
    for i in range(2): sink.after(ps.read(r), {f"{j}.pos": float(i) for j in ps.J}, "b")
    assert buf.poll_file(p) == 2 and len(buf.ticks) == 5 and buf.at(buf.ticks[3]["t_mono"])["seq"] == 4
    ps.TickSink(p)   # pick_seq restarts the file (the next staged run): the reader starts over without losing what it has
    assert buf.poll_file(p) == 0 and len(buf.ticks) == 5


def test_the_ring_rotates_at_max_lines_and_keeps_one_generation(ps, tmp_path):
    p = str(tmp_path / "t.jsonl"); sink = ps.TickSink(p, max_lines=10); r = FakeRobot(ps.J)
    for i in range(25): sink.after(ps.read(r), {f"{j}.pos": float(i) for j in ps.J}, "a")
    sink.close(); assert os.path.exists(p + ".1") and len([l for l in read_lines(p) if "seq" in l]) <= 10
    seqs = [l["seq"] for l in read_lines(p + ".1") + read_lines(p) if "seq" in l]; assert seqs[-1] == 25


def test_ease_helpers_are_the_ones_move_uses(ps):
    cur = {j: 0.0 for j in ps.J}; tgt = {j: 10.0 for j in ps.J}; n = ps.ease_n(1.0); assert n == 40 and ps.ease_n(0.1) == 12
    assert ps.move_secs(cur, tgt, 0.5, 80.0) == 0.5 and ps.move_secs(cur, tgt, 0.1, 20.0) == 0.5
    r = FakeRobot(ps.J)
    import time as _t
    old = _t.sleep; _t.sleep = lambda s: None
    try: ps.move(r, cur, tgt, 1.0, 80.0)
    finally: _t.sleep = old
    assert r.sent == [ps.ease_cmd(cur, tgt, i, n) for i in range(1, n + 1)]
