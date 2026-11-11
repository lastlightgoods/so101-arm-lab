"""dashboard/telemetry_store.py (history, flags, events), the arm stream's `tele` going into it, and /api/tele."""
import json

import pytest
from teleserverhelpers import Clock, modules, sample

st, fb, ts = modules()


@pytest.fixture
def store():
    clock = Clock()
    return ts.TeleStore(st, window_s=60, clock=clock), clock, fb.FakeServoBus(seed=1)


def feed(store, clock, bus, n=1, step=0.2, **tweak):
    ev = []
    for _ in range(n):
        bus.advance(step); clock.advance(step); ev += store.add(sample(st, bus, **tweak), "idle-poll")
    return ev


# ---------- history ----------
def test_rows_are_compact_and_stamped_with_this_machines_clock(store):
    s, clock, bus = store
    smp = sample(st, bus); smp["t"] = 99.0   # the Jetson's clock is wildly off
    s.add(smp, "idle-poll")
    api = s.api()
    assert api["keys"] == list(ts.KEYS) and api["joints"] == list(st.JOINTS)
    (row,) = api["rows"]
    assert row[0] == clock.t and len(row[3]) == 6 and len(row[3][1]) == len(ts.KEYS)
    assert api["now"] == clock.t and api["src"] == "idle-poll" and api["n"] == 1 and api["age"] == 0


def test_since_returns_only_newer_rows_and_the_window_is_pruned(store):
    s, clock, bus = store
    feed(s, clock, bus, n=10)
    t5 = s.api()["rows"][4][0]
    assert [r[0] for r in s.api(since=t5)["rows"]] == [r[0] for r in s.api()["rows"][5:]]
    feed(s, clock, bus, n=int(100 / 0.2))      # 100 s more: only the last 60 s survive
    rows = s.api()["rows"]
    assert rows[-1][0] - rows[0][0] <= 60.01 and len(rows) <= 302


def test_empty_store_and_sample_without_joints(store):
    s, clock, bus = store
    assert s.api()["rows"] == [] and s.api()["age"] is None and s.api()["flags"] == []
    assert s.add({"j": {}}) == [] and s.api()["n"] == 0


def test_missing_joint_gives_a_null_cell(store):
    s, clock, bus = store
    bus.silent = {3}; feed(s, clock, bus)
    assert s.api()["rows"][0][3][2] is None


# ---------- flags and events ----------
def test_temperature_flags_rise_once_per_level(store):
    s, clock, bus = store
    assert feed(s, clock, bus, n=3) == [] and s.api()["flags"] == []
    ev = feed(s, clock, bus, shoulder_lift={"temp_c": 61})
    assert [(e["joint"], e["kind"], e["level"]) for e in ev] == [("shoulder_lift", "hot", 1)]
    assert feed(s, clock, bus, n=5, shoulder_lift={"temp_c": 62}) == [], "an unchanged flag is not an event again"
    ev = feed(s, clock, bus, shoulder_lift={"temp_c": 66})
    assert [(e["kind"], e["level"]) for e in ev] == [("hot", 2)]
    assert {f["kind"] for f in s.api()["flags"]} == {"hot"}
    feed(s, clock, bus, shoulder_lift={"temp_c": 40})
    assert s.api()["flags"] == []
    assert len(feed(s, clock, bus, shoulder_lift={"temp_c": 61})) == 1, "it cooled and heated again: a new event"
    assert [e["level"] for e in s.api()["events"]] == [1, 2, 1]


def test_hot_flag_uses_the_cut_read_from_the_servo(store):
    s, clock, bus = store
    smp = sample(st, bus, gripper={"temp_c": 52}); smp["limits"] = {"gripper": {"temp_cut_c": 60}}
    ev = s.add(smp)
    assert ev and ev[0]["joint"] == "gripper" and "cut at 60 C" in ev[0]["text"] and s.api()["limits"]["gripper"]["temp_cut_c"] == 60


def test_servo_error_bits_are_red_events(store):
    s, clock, bus = store
    ev = feed(s, clock, bus, wrist_flex={"status": 0b100000, "err": ["overload"]})
    assert [(e["joint"], e["kind"], e["level"]) for e in ev] == [("wrist_flex", "err:overload", 2)]


def test_brownout_and_board_rating(store):
    s, clock, bus = store
    ev = feed(s, clock, bus, shoulder_pan={"volt_v": 10.5})
    assert [(e["joint"], e["kind"], e["level"]) for e in ev] == [("bus", "volt", 1)]
    ev = feed(s, clock, bus, shoulder_pan={"volt_v": 9.7})
    assert ("bus", "volt", 2) in [(e["joint"], e["kind"], e["level"]) for e in ev] and "brown-out" in ev[0]["text"]
    smp = sample(st, bus); smp["total_a"] = 5.4
    assert [e["kind"] for e in s.add(smp)] == ["bus"]
    assert any(f["kind"] == "bus" and f["level"] == 2 for f in s.api()["flags"])


def test_sag_above_usual_raises_after_enough_history(store):
    s, clock, bus = store
    holding = {"shoulder_lift": {"sag": 1.0, "moving": 0, "on": 1}}
    feed(s, clock, bus, n=80, **holding)
    assert not [f for f in s.api()["flags"] if f["kind"] == "sag"]
    ev = feed(s, clock, bus, n=25, shoulder_lift={"sag": 4.0, "moving": 0, "on": 1})
    assert [e["kind"] for e in ev] == ["sag"] and ev[0]["level"] == 1
    # a joint that is moving does not feed the holding-sag history (tracking lag is not sag)
    s2 = ts.TeleStore(st, clock=Clock()); b2 = fb.FakeServoBus(seed=2)
    for _ in range(100): b2.advance(0.2); s2.add(sample(st, b2, shoulder_lift={"sag": 9.0, "moving": 1, "on": 1}))
    assert len(s2.sag_hist["shoulder_lift"]) == 0


def test_events_are_capped(store):
    s, clock, bus = store
    for i in range(ts.MAX_EVENTS + 20):
        feed(s, clock, bus, shoulder_lift={"temp_c": 61 + (i % 2) * 5})
        feed(s, clock, bus, shoulder_lift={"temp_c": 30})
    assert len(s.events) == ts.MAX_EVENTS and len(s.api()["events"]) == 30


# ---------- the arm stream -> store, and /api/tele ----------
def arm_lines(tele_samples):
    out = []
    for i, smp in enumerate(tele_samples):
        out.append(json.dumps({"t": 1.0 + i, "role": "follower", "mode": "deg", "src": "idle-poll", "pos": {"shoulder_pan": 1.5}, "cal": {}, "tele": smp}) + "\n")
        out.append(json.dumps({"t": 1.0 + i, "role": "leader", "src": "idle-poll", "pos": {"shoulder_pan": 2.5}}) + "\n")
    return out


@pytest.fixture
def server_store(srv, monkeypatch):
    clock = Clock(time_now())
    t = ts.TeleStore(st, clock=clock)
    monkeypatch.setattr(srv, "TELE", t)
    return t


def time_now():
    import time
    return time.time()


def test_arm_loop_moves_tele_into_the_store_not_the_state(srv, fake, run_loop, server_store, events):
    bus = fb.FakeServoBus(seed=1); bus.advance(5)
    hot = sample(st, bus, shoulder_lift={"temp_c": 67})
    fake.on_popen("arm_state", arm_lines([sample(st, bus), hot]))
    run_loop(srv.arm_loop, 1)
    assert server_store.api()["n"] == 2 and server_store.api()["src"] == "idle-poll"
    # (arm_loop clears state["arms"] when the stream ends, as it always did; the stream's own lines never carried tele into it)
    assert any("shoulder_lift" in e["msg"] and "C, cut at 70 C" in e["msg"] for e in srv.state["events"]), "a red flag also goes to the dashboard log"


def test_arm_loop_survives_a_lines_without_telemetry_and_a_bad_one(srv, fake, run_loop, server_store):
    lines = [json.dumps({"t": 1.0, "role": "follower", "src": "idle-poll", "pos": {"shoulder_pan": 1.0}}) + "\n",
             json.dumps({"t": 2.0, "role": "follower", "src": "idle-poll", "pos": {}, "tele": {"j": "garbage"}}) + "\n",
             json.dumps({"t": 3.0, "role": "leader", "src": "idle-poll", "pos": {}, "tele": {"j": {"x": {}}}}) + "\n"]
    fake.on_popen("arm_state", lines)
    run_loop(srv.arm_loop, 1)
    assert server_store.api()["n"] == 0


def test_api_tele_serves_rows_flags_and_since(srv, http, server_store):
    bus = fb.FakeServoBus(seed=1); bus.advance(5); clock = server_store.clock
    for _ in range(5):
        bus.advance(0.2); clock.advance(0.2); server_store.add(sample(st, bus, shoulder_lift={"temp_c": 66}), "lerobot")
    body, _ = http.get_json("/api/tele")
    assert len(body["rows"]) == 5 and body["src"] == "lerobot" and body["flags"][0]["kind"] == "hot" and body["marks"]["bus_a"] == 5.0
    t3 = body["rows"][2][0]
    body2, _ = http.get_json(f"/api/tele?since={t3}")
    assert len(body2["rows"]) == 2
    assert http.get_json("/api/tele?since=notanumber")[0]["rows"] and http.get("/api/tele")[1]["Cache-Control"] == "no-store"


def test_api_tele_is_empty_before_any_telemetry(http, server_store):
    body, _ = http.get_json("/api/tele")
    assert body["rows"] == [] and body["n"] == 0 and body["age"] is None
