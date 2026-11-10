"""Smaller load-bearing pieces: the event ring, speech queue, roar trigger, transcript ingest, arm-mode guards."""
import json, os, queue, time
import pytest


def test_event_ring_keeps_40_newest_first_and_appends_to_log(srv, tmp_path):
    for i in range(45): srv.event(f"e{i}")
    msgs = [e["msg"] for e in srv.state["events"]]
    assert len(msgs) == 40 and msgs[0] == "e44" and msgs[-1] == "e5"
    assert all(abs(e["t"] - time.time()) < 2 for e in srv.state["events"])
    lines = (tmp_path / "events.log").read_text().splitlines()
    assert len(lines) == 45 and lines[-1].endswith("  e44") and len(lines[0].split("  ")[0]) == 19   # "YYYY-mm-dd HH:MM:SS  msg"


def test_event_log_write_failure_is_swallowed(srv, monkeypatch):
    monkeypatch.setattr(srv, "HERE", "/nonexistent/dir/for/so101/tests")
    srv.event("still recorded in memory")
    assert srv.state["events"][0]["msg"] == "still recorded in memory"


def test_say_queues_text_with_timestamp(srv, said):
    srv.say("Next."); srv.say(42)
    assert said() == ["Next.", "42"]


class _Drained(Exception): pass


class _Q(queue.Queue):
    """A queue whose get() ends say_loop once everything queued has been consumed."""
    def get(self, *a, **k):
        if self.empty(): raise _Drained
        return super().get(*a, **k)


def test_say_loop_dedups_and_drops_superseded(srv, fake, monkeypatch):
    fake.on(lambda a: a[0] == "say", "")
    q = _Q(); monkeypatch.setattr(srv, "SAY_Q", q)
    q.put((time.time() - 10, "stale: superseded while others waited")); q.put((time.time(), "Holding.")); q.put((time.time(), "Holding.")); q.put((time.time(), "Next."))
    with pytest.raises(_Drained): srv.say_loop()
    assert [a for a, _ in fake.calls] == [["say", "-r", "190", "Holding."], ["say", "-r", "190", "Next."]]
    assert all(k.get("timeout") == 20 for _, k in fake.calls) and srv._last_said["text"] == "Next."


def test_roar_only_on_open_while_extended_and_rate_limited(srv, fake, events):
    fake.on_popen("afplay", [])
    srv.roar_check({"gripper": 10, "shoulder_lift": -80})        # at rest, closed
    srv.roar_check({"gripper": 70, "shoulder_lift": -80})        # opens at rest: no roar
    assert fake.popens == []
    srv.roar_check({"gripper": 10, "shoulder_lift": 0})          # closes while extended
    srv.roar_check({"gripper": 60, "shoulder_lift": 0})          # opens while extended -> roar
    assert len(fake.popens) == 1 and fake.popens[0][0] == ["afplay", os.path.join(srv.HERE, "roar.wav")] and events()[0] == "🦖 RAWR — handoff!"
    srv.roar_check({"gripper": 10, "shoulder_lift": 0}); srv.roar_check({"gripper": 60, "shoulder_lift": 0})
    assert len(fake.popens) == 1, "8 s rate limit"
    srv.roar_check({"gripper": 40, "shoulder_lift": 0})          # hysteresis: still open above 35 once open
    assert srv._roar["open"] is True
    srv.roar_check({"gripper": None, "shoulder_lift": 0}); srv.roar_check({})
    assert len(fake.popens) == 1


def test_ingest_pairs_bash_calls_with_results_and_caps_at_14(srv):
    def use(i): return json.dumps({"timestamp": f"t{i}", "message": {"content": [{"type": "tool_use", "name": "Bash", "id": f"id{i}", "input": {"command": f"echo {i}", "description": f"d{i}"}}]}})
    def res(i, txt): return json.dumps({"message": {"content": [{"type": "tool_result", "tool_use_id": f"id{i}", "content": txt}]}})
    for i in range(20): srv._ingest(use(i))
    assert [c["id"] for c in srv.shell_calls] == [f"id{i}" for i in range(6, 20)]
    srv._ingest(res(19, "x" * 2000)); srv._ingest(res(18, [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])); srv._ingest(res(0, "gone"))
    assert srv.shell_calls[-1]["out"] == "x" * 1500 and srv.shell_calls[-1]["done"] is True
    assert srv.shell_calls[-2]["out"] == "a b" and srv.shell_calls[-3]["done"] is False
    srv._ingest("not json"); srv._ingest(json.dumps({"message": {"content": "string content"}}))
    srv._ingest(json.dumps({"message": {"content": [{"type": "tool_use", "name": "Read", "id": "r", "input": {}}]}}))
    assert len(srv.shell_calls) == 14 and srv.shell_calls[0]["cmd"] == "echo 6" and srv.shell_calls[0]["desc"] == "d6"


def test_arm_mode_unknown_and_recording_guard(srv, fake, said, events):
    assert srv.arm_mode("fly") == {"ok": False, "err": "unknown mode"}
    srv.REC["running"] = True
    for mode in ("free", "hold"):
        r = srv.arm_mode(mode)
        assert r["ok"] is False and "press Q" in r["err"]
    assert said() == ["Stop the recording first."] * 2 and fake.calls == [] and events() == []
    assert set(srv.ARM_MODES) == {"teleop", "free", "hold", "policy", "vla", "dance", "wave", "replay"}


def test_arm_mode_runs_detached_on_the_edge_host(srv, fake, said, events):
    fake.on("so101-hold", "", 0)
    assert srv.arm_mode("hold") == {"ok": True, "err": ""}
    cmd = fake.ssh[0]
    assert cmd.startswith("mkdir -p ~/so101/outputs/run; nohup bash -lc 'source ~/so101/env.sh; so101-hold' > ~/so101/outputs/run/ui_hold.log 2>&1 &")
    assert fake.calls[0][0][:4] == ["ssh", "-o", "ConnectTimeout=5", srv.JETSON]
    assert said() == ["Holding."] and events() == ["ARM MODE HOLD (dashboard)"]


def test_arm_stop_kills_beat_and_logs_event(srv, fake, events):
    fake.on("so101-stop", "stopped\n"); fake.on(lambda a: a[0] == "pkill", "")
    assert srv.arm_mode("stop") == {"ok": True, "out": "stopped"}
    assert fake.calls[1][0] == ["pkill", "-f", "afplay .*beat_"] and events() == ["ARM STOP (dashboard)"]


def test_spotify_cmd_unknown_makes_no_call(srv, fake):
    assert srv.spotify_cmd("stop") == {"ok": False, "err": "unknown"} and fake.calls == []


def test_set_lock_resets_scan_state(srv, events):
    srv.LOCK.update(progress=0.7, eye=[1, 2, 3, 4], status="SCANNING")
    srv.set_lock(True, "button")
    assert srv.LOCK["locked"] and srv.LOCK["progress"] == 0.0 and srv.LOCK["eye"] is None and srv.LOCK["status"] == "NO SIGNAL"
    srv.set_lock(False)
    assert srv.LOCK["status"] == "UNLOCKED" and events()[0] == "IDENTITY CONFIRMED · UNLOCKED"


def test_module_is_inert_on_import(srv):
    """Import-time guards: the port came from the trimmed argv, the transcript glob was bypassed, nothing is bound."""
    assert srv.PORT == 8765 and srv.TRANSCRIPT.endswith("no-such-transcript.jsonl") and srv.LOCK["locked"] is False
    assert srv.state["jetson"] == {"online": False, "last": 0} and srv.HIST == [] and srv.REC["running"] is False
