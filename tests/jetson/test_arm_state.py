"""arm_state.py with every external edge faked: ports are temp files, the bus is FakeBus, /dev/shm is a temp dir."""
import json
from jhelpers import FakeBus


def deg_cal():
    from lerobot.motors import MotorCalibration
    return {"shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=100, range_max=3000),
            "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=200, range_max=2500)}


# ---------- busy detection ----------
def test_busy_flag_closes_ports_and_forwards_fresh_shm_state(arm):
    m = arm.mod
    for role in ("follower", "leader"): m.buses[role] = FakeBus(f"/dev/{role}", {})
    arm.mark_busy()
    fresh = {"t": arm.clock.now - 0.5, "role": "follower", "src": "lerobot", "pos": {"gripper": 42.0}}
    (arm.shm / "so101_follower.json").write_text(json.dumps(fresh))
    t0 = arm.clock.now
    out = arm.run()
    assert m.buses == {}
    assert all(("disconnect", False) in b.calls for b in FakeBus.instances), "ports must be released without touching torque"
    assert arm.by_role("follower") == [fresh]
    assert arm.by_role("leader") == [{"t": t0, "role": "leader", "src": "busy", "pos": {}}]
    assert not any(d.get("src") == "idle-poll" for d in out)


def test_busy_stale_shm_state_gives_empty_pos(arm):
    arm.mark_busy()
    (arm.shm / "so101_leader.json").write_text(json.dumps({"t": arm.clock.now - 2.5, "role": "leader", "pos": {"x": 1}}))
    (arm.shm / "so101_follower.json").write_text("{not json")
    arm.run()
    for role in ("follower", "leader"):
        (d,) = arm.by_role(role); assert d["src"] == "busy" and d["pos"] == {}


def test_lerobot_process_counts_as_busy(arm, monkeypatch):
    monkeypatch.setattr(arm.mod, "lerobot_running", lambda: True)
    arm.run()
    assert {d["src"] for d in arm.out} == {"busy"}
    assert FakeBus.instances == []


def test_lerobot_running_uses_pgrep(arm_state_module, monkeypatch):
    m = arm_state_module; seen = {}
    class R:
        def __init__(self, rc): self.returncode = rc
    def fake_run(args, **kw): seen["args"] = args; return R(seen.get("rc", 1))
    monkeypatch.setattr(m.subprocess, "run", fake_run)
    seen["rc"] = 0; assert m.lerobot_running() is True
    seen["rc"] = 1; assert m.lerobot_running() is False
    assert seen["args"][:2] == ["pgrep", "-f"] and "lerobot-(record" in seen["args"][2]
    def boom(*a, **k): raise FileNotFoundError("pgrep")
    monkeypatch.setattr(m.subprocess, "run", boom)
    assert m.lerobot_running() is False


def test_busy_check_runs_at_most_every_half_second(arm, monkeypatch):
    checks = []
    monkeypatch.setattr(arm.mod, "lerobot_running", lambda: checks.append(arm.clock.now) or False)
    arm.run(sleeps=30)   # 30 ticks at 1/15 s = 2 s of fake time
    assert len(checks) == 4, checks   # t=0, 0.53, 1.07, 1.6 (never two checks closer than 0.5 s)
    assert all(b - a > 0.5 for a, b in zip(checks, checks[1:]))


# ---------- idle polling ----------
def test_idle_poll_with_calibration_emits_degrees(arm):
    m = arm.mod; m.cals["follower"] = deg_cal()
    arm.run()
    (f,) = arm.by_role("follower")
    bus = next(b for b in FakeBus.instances if b.port == str(arm.ports["follower"]))
    assert bus.calls[0] == ("connect", False) and bus.calls[1] == ("sync_read", "Present_Position", True)
    assert bus.calibration is m.cals["follower"]
    assert f["src"] == "idle-poll" and f["mode"] == "deg"
    assert f["pos"] == {"shoulder_pan": 1.0, "gripper": 2.0} and all(isinstance(v, float) for v in f["pos"].values())
    assert f["cal"] == {"shoulder_pan": [100, 3000], "gripper": [200, 2500]}
    assert set(m.buses) == {"follower", "leader"}
    (l,) = arm.by_role("leader"); assert l["mode"] == "raw" and l["cal"] == {}


def test_idle_poll_without_calibration_reads_raw(arm):
    arm.run()
    bus = FakeBus.instances[0]
    assert ("sync_read", "Present_Position", False) in bus.calls
    assert all(d["mode"] == "raw" and d["src"] == "idle-poll" for d in arm.out)


def test_bus_is_reused_across_ticks(arm):
    arm.run(sleeps=3)
    assert len(FakeBus.instances) == 2
    assert sum(c[0] == "connect" for b in FakeBus.instances for c in b.calls) == 2


def test_unplugged_port_is_reported_not_opened(arm):
    arm.ports["leader"].unlink(); t0 = arm.clock.now
    arm.run()
    (l,) = arm.by_role("leader"); assert l == {"t": t0, "role": "leader", "src": "unplugged", "pos": {}}
    assert [b.port for b in FakeBus.instances] == [str(arm.ports["follower"])]
    assert "leader" not in arm.mod.buses


def test_read_error_emits_error_closes_bus_and_backs_off(arm):
    FakeBus.fail = RuntimeError("Port /dev/x: read timeout " + "x" * 200)
    arm.run(sleeps=1)   # the 1 s backoff is the first sleep
    (f,) = arm.by_role("follower")
    assert f["src"] == "error" and f["pos"] == {} and f["err"].startswith("Port /dev/x") and len(f["err"]) == 100
    assert arm.clock.sleeps == [1]
    assert ("disconnect", False) in FakeBus.instances[0].calls and arm.mod.buses == {}


def test_bus_reconnects_after_error(arm):
    FakeBus.fail = RuntimeError("boom")
    arm.run(sleeps=3)   # tick 1: follower err (sleep 1), leader err (sleep 1), tick sleep -> stop
    FakeBus.fail = None; arm.out.clear()
    arm.run(sleeps=1)
    assert {d["src"] for d in arm.out} == {"idle-poll"}
    assert len(FakeBus.instances) == 4


def test_busy_then_idle_reopens_ports(arm):
    arm.mark_busy(); arm.run(sleeps=1)
    assert FakeBus.instances == []
    (arm.tmp / ".arms_busy").unlink(); arm.out.clear()
    arm.run(sleeps=1)
    assert {d["src"] for d in arm.out} == {"idle-poll"}


# ---------- say relay ----------
def test_forward_say_tails_incrementally(arm):
    m = arm.mod; say = arm.shm / "so101_say.jsonl"
    with open(say, "a") as f: f.write(json.dumps({"t": 1, "say": "one"}) + "\n" + json.dumps({"t": 2, "say": "two"}) + "\n")
    m.forward_say(); m.forward_say()
    assert [d["say"] for d in arm.out] == ["one", "two"]
    assert all(d["role"] == "say" for d in arm.out)
    with open(say, "a") as f: f.write(json.dumps({"t": 3, "say": "three"}) + "\n")
    m.forward_say()
    assert [d["say"] for d in arm.out] == ["one", "two", "three"]


def test_forward_say_handles_truncation_and_missing_file(arm):
    m = arm.mod; say = arm.shm / "so101_say.jsonl"
    say.write_text(json.dumps({"t": 1, "say": "a long first message"}) + "\n"); m.forward_say()
    say.write_text(json.dumps({"t": 2, "say": "b"}) + "\n"); m.forward_say()      # rewritten shorter: size < say_pos
    assert [d["say"] for d in arm.out] == ["a long first message", "b"]
    say.unlink(); m.forward_say()
    assert m.say_pos == 0
    say.write_text(json.dumps({"t": 3, "say": "c"}) + "\n"); m.forward_say()
    assert [d["say"] for d in arm.out][-1] == "c"


def test_forward_say_skips_bad_lines(arm):
    m = arm.mod; say = arm.shm / "so101_say.jsonl"
    say.write_text("{not json\n" + json.dumps({"t": 1, "say": "ok"}) + "\n" + "42\n" + '"str"\n' + "[1,2]\n" + json.dumps({"t": 2, "say": "ok2"}) + "\n")
    m.forward_say()
    assert [d["say"] for d in arm.out] == ["ok", "ok2"]


def test_forward_say_waits_for_a_partial_line(arm):
    """The plugin appends a whole line at once, but a reader can still land mid-write; half a line must not be lost."""
    m = arm.mod; say = arm.shm / "so101_say.jsonl"
    line = json.dumps({"t": 1, "say": "Recording episode 3"}) + "\n"
    say.write_text(line[:15]); m.forward_say()
    assert arm.out == []
    with open(say, "a") as f: f.write(line[15:])
    m.forward_say()
    assert [d["say"] for d in arm.out] == ["Recording episode 3"]


def test_main_loop_forwards_say_each_tick(arm):
    say = arm.shm / "so101_say.jsonl"; say.write_text(json.dumps({"t": 1, "say": "hi"}) + "\n")
    arm.run()
    assert [d for d in arm.out if d.get("role") == "say"] == [{"role": "say", "t": 1, "say": "hi"}]
