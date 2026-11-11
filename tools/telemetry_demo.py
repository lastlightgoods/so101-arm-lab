#!/usr/bin/env python3
"""telemetry_demo.py - the dashboard with a pretend arm, to look at the servo telemetry panel without hardware or the live dashboard.

    python tools/telemetry_demo.py [port]      # default 8799; refuses the live dashboard's ports (8765, 8766, 9333)
    python tools/telemetry_demo.py 8799 --hot lift   # the shoulder lift runs hot, the supply sags, a servo flags an overload: the panel's warning states

Serves dashboard/ (the real index.html) through the UI-test stub (tests/ui/stub_server.py: every /api route answers canned data) and answers
/api/tele from the real TeleStore, fed by tools/fake_servo_bus.py through the real servo_telemetry decoder. Ten minutes of arm history are
generated at start; after that the pretend arm keeps picking in real time. Opens no serial port, no camera and no network port but its own.
Press Ctrl-C to stop. Open http://127.0.0.1:<port>/ and press M (or the SERVOS button in the arm panel).
"""
import argparse, os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests", "ui"), os.path.join(ROOT, "dashboard")]
import fake_servo_bus as fb, servo_telemetry as st, telemetry_store as ts   # noqa: E402
import apidata, stub_server   # noqa: E402

FORBIDDEN = {8765, 8766, 9333}
HOT = {"lift": "shoulder_lift", "elbow": "elbow_flex", "grip": "gripper", "pan": "shoulder_pan"}


def build(history_s=600, hot=None, seed=3, hz=5.0):
    """(store, bus, tick) : a TeleStore holding `history_s` seconds of picking, and tick() to advance the pretend arm by 1/hz of real time."""
    store = ts.TeleStore(st); bus = fb.FakeServoBus(seed=seed); bus.advance(3.0)   # a warm-up so the first sample is not the arm's start-up jolt
    now = time.time(); bus.t = now - history_s
    if hot:
        j = HOT.get(hot, hot); bus.extra_heat[j] = 0.0
    n = int(history_s * hz)
    for k in range(n):
        if hot and k == int(n * 0.35): bus.extra_heat[HOT.get(hot, hot)] = 0.3   # starts running hot a third of the way in
        if hot and k == int(n * 0.72): bus.extra_drop_v = 1.7                      # a brown-out
        if hot and k == int(n * 0.80): bus.extra_drop_v = 0.0
        bus.advance(1.0 / hz); store.add(st.read_sample(bus, now=bus.t), "idle-poll", at=bus.t)
    store.add(st.read_sample(bus, now=bus.t), "idle-poll"); store.add(st.read_sample(bus, now=bus.t), "idle-poll")   # the first rows of the live stream

    def tick():
        bus.t = max(bus.t, time.time() - 1.0 / hz); bus.advance(1.0 / hz); store.add(st.read_sample(bus, now=bus.t), "idle-poll")
    return store, bus, tick


def serve(port, hot=None, hz=5.0):
    store, bus, tick = build(hot=hot, hz=hz)
    stub = stub_server.Stub(); stub.routes.update(apidata.default_routes())
    stub.routes["/api/tele"] = lambda path: store.api(float((path.split("since=")[1].split("&")[0]) if "since=" in path else 0))
    def api_state(path):
        last = store.last or {}; pos = {n: d["pos"] for n, d in (last.get("j") or {}).items() if d.get("pos") is not None}
        return {**apidata.state(), "arms": {"follower": {"role": "follower", "src": "idle-poll", "mode": "deg", "pos": pos, "rx": time.time()}}}
    stub.routes["/api/state"] = api_state
    stub.start(port)
    def pump():
        while True: time.sleep(1.0 / hz); tick()
    threading.Thread(target=pump, daemon=True).start()
    return stub, store


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", type=int, nargs="?", default=8799); ap.add_argument("--hot", default=None, help="joint that runs hot (lift, elbow, grip, pan or a full joint name)")
    a = ap.parse_args()
    if a.port in FORBIDDEN: raise SystemExit(f"port {a.port} is the live dashboard's; pick another")
    stub, _ = serve(a.port, a.hot)
    print(f"telemetry demo on http://127.0.0.1:{a.port}/  (pretend arm; Ctrl-C to stop)", flush=True)
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt: stub.stop()
