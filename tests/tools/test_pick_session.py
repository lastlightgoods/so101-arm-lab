"""The persistent pick_seq session (12 Nov 2026): jetson/pick_seq.py --serve and tools/pick_loop.py PickSession. Fakes only: a FakeRobot in process, and for the pipe
tests a real subprocess that runs pick_seq.serve on a FakeRobot (no ssh, no serial port, no motion)."""
import json, os, signal, subprocess, sys, textwrap, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "demos"))
from demohelpers import FakeRobot, PLAN, commands_hash, load, run_pick_seq   # noqa: E402

START = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 70.0, "wrist_roll": 0.5, "gripper": 0.0}
PLAN2 = {"waypoints": [{"name": "carry", "joints": {"shoulder_pan": -20.0, "shoulder_lift": -30.0, "elbow_flex": 50.0, "wrist_flex": 60.0, "wrist_roll": 0.5}, "gripper": 10, "seconds": 0.8, "settle": 0.1},
                       {"name": "open", "joints": {}, "gripper": 35, "seconds": 0.3, "settle": 0.1, "gripper_only": True}]}


@pytest.fixture(scope="module")
def ps():
    sys.path.insert(0, os.path.join(ROOT, "jetson")); return load(os.path.join(ROOT, "jetson", "pick_seq.py"), "pick_seq_for_serve_tests")


@pytest.fixture(scope="module")
def pl(): return load(os.path.join(ROOT, "tools", "pick_loop.py"), "pick_loop_for_session_tests")


class Reader:
    """A scripted stdin: str lines, then EOF; TimeoutError / KeyboardInterrupt entries are raised."""
    def __init__(self, *items): self.items = list(items)
    def next(self, timeout):
        if not self.items: return None
        x = self.items.pop(0)
        if isinstance(x, BaseException) or (isinstance(x, type) and issubclass(x, BaseException)): raise x
        return x if isinstance(x, str) else json.dumps(x)


def serve_with(ps, tmp_path, monkeypatch, capsys, reader, robot=None, busy_exists=False, argv=()):
    rob = robot or FakeRobot(ps.J, START); busy = tmp_path / "arms_busy"
    if busy_exists: busy.write_text("")
    monkeypatch.setattr(ps, "connect", lambda: rob); monkeypatch.setattr(ps, "lerobot_running", lambda: False); monkeypatch.setattr(ps, "BUSY", str(busy)); monkeypatch.setattr(ps.time, "sleep", lambda s: None)
    monkeypatch.setattr(ps, "LineReader", lambda fd=0: reader); monkeypatch.setattr(sys, "argv", ["pick_seq.py", "--serve", *argv]); ps.main(); lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    return rob, lines, busy


def rows_of(lines): return [{k: v for k, v in r.items() if k != "t"} for r in lines if "wp" in r]


def plan_msg(plan, correct=0.8, max_deg_s=80.0): return {"waypoints": plan["waypoints"], "max_deg_s": max_deg_s, "correct": correct}


def test_serve_two_plans_in_one_connection_equal_two_main_runs(ps, tmp_path, monkeypatch, capsys):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir(); (tmp_path / "s").mkdir()
    ra, la = run_pick_seq(ps, tmp_path / "a", monkeypatch, capsys, plan=PLAN)
    rb, lb = run_pick_seq(ps, tmp_path / "b", monkeypatch, capsys, plan=PLAN2, start=dict(ra.pos))   # main's second run starts where the first ended, as the one arm does
    connects = []; rob = FakeRobot(ps.J, START)
    monkeypatch.setattr(ps, "connect", lambda: connects.append(1) or rob)
    rob, lines, _ = serve_with(ps, tmp_path / "s", monkeypatch, capsys, Reader(plan_msg(PLAN), plan_msg(PLAN2)), robot=rob)
    assert rob.sent, lines
    assert rob.sent == ra.sent + rb.sent and commands_hash(rob.sent) == commands_hash(ra.sent + rb.sent), "same commands, exact"
    assert rows_of(lines) == rows_of(la) + rows_of(lb), "same rows"
    assert [l for l in lines if "plan_done" in l and "total_s" in l][-1]["plan_done"] == 2 and lines[-1] == {"bye": "eof"} and rob.disconnected
    assert rob.reads == ra.reads + rb.reads, "same state reads"


def test_serve_defaults_and_per_plan_options(ps, tmp_path, monkeypatch, capsys):
    (tmp_path / "a").mkdir(); (tmp_path / "s").mkdir()
    ra, _ = run_pick_seq(ps, tmp_path / "a", monkeypatch, capsys, argv=["--max-deg-s", "30"])
    rob, _, _ = serve_with(ps, tmp_path / "s", monkeypatch, capsys, Reader(plan_msg(PLAN, 0.8, 30.0)))
    assert rob.sent == ra.sent, "max_deg_s and correct in the message act like the flags"


def test_serve_writes_one_tick_sink_per_plan_like_main(ps, tmp_path, monkeypatch, capsys):
    sink = tmp_path / "t.jsonl"; rob, _, _ = serve_with(ps, tmp_path, monkeypatch, capsys, Reader(plan_msg(PLAN), plan_msg(PLAN2)), argv=["--tick-sink", str(sink)])
    lines = [json.loads(l) for l in open(sink)]; assert lines[0]["header"] == 1 and [t["seq"] for t in lines[1:]][0] == 1 and len(lines) - 1 == len(rob.sent) - 124 > 0   # the second plan's file restarts at seq 1 and holds only its own ticks


@pytest.mark.parametrize("ender", [Reader(), Reader({"quit": True}), Reader(TimeoutError), Reader(KeyboardInterrupt)], ids=["eof", "quit", "idle", "term"])
def test_every_way_out_disconnects_and_removes_the_busy_file(ps, tmp_path, monkeypatch, capsys, ender):
    ender.items.insert(0, plan_msg(PLAN2)); rob, lines, busy = serve_with(ps, tmp_path, monkeypatch, capsys, ender)
    assert rob.disconnected and not busy.exists() and len(rob.sent) > 0 and "bye" in lines[-1]


def test_a_busy_file_that_was_not_ours_stays(ps, tmp_path, monkeypatch, capsys):
    rob, _, busy = serve_with(ps, tmp_path, monkeypatch, capsys, Reader({"quit": True}), busy_exists=True); assert rob.disconnected and busy.exists()


def test_the_busy_file_is_held_for_the_whole_session(ps, tmp_path, monkeypatch, capsys):
    seen = []
    class Spy(Reader):
        def next(self, t): seen.append(os.path.exists(tmp_path / "arms_busy")); return super().next(t)
    serve_with(ps, tmp_path, monkeypatch, capsys, Spy(plan_msg(PLAN2), plan_msg(PLAN2))); assert seen == [True, True, True]


def test_term_in_the_middle_of_a_move_stops_and_holds(ps, tmp_path, monkeypatch, capsys):
    rob = FakeRobot(ps.J, START); real = rob.send_action
    def send(a):
        if len(rob.sent) == 10: raise KeyboardInterrupt   # the SIGTERM handler's exception, delivered mid-move
        return real(a)
    rob.send_action = send; rob, lines, busy = serve_with(ps, tmp_path, monkeypatch, capsys, Reader(plan_msg(PLAN)), robot=rob)
    assert len(rob.sent) == 10 and {"stopped": True} in lines and rob.disconnected and not busy.exists() and not [l for l in lines if "plan_done" in l]


def test_refuses_while_a_lerobot_job_holds_the_arm(ps, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ps, "connect", lambda: pytest.fail("must not connect")); monkeypatch.setattr(ps, "lerobot_running", lambda: True); monkeypatch.setattr(ps, "BUSY", str(tmp_path / "b"))
    monkeypatch.setattr(sys, "argv", ["pick_seq.py", "--serve"])
    with pytest.raises(SystemExit, match="lerobot job"): ps.main()
    assert not (tmp_path / "b").exists()


BAD = ["not json", "[1, 2]", '"x"', "{}", '{"quit": false}', '{"quit": true, "x": 1}', '{"cmd": "rm -rf /"}', '{"waypoints": [], "eval": "1"}', '{"waypoints": "abc"}', '{"waypoints": [1]}', '{"waypoints": [{"joints": {"shoulder_pan": "x"}}]}',
       '{"waypoints": [{"joints": []}]}', '{"waypoints": [{"joints": {}, "seconds": null}]}', '{"waypoints": [{"joints": {}}], "max_deg_s": 0}', '{"waypoints": [{"joints": {}}], "max_deg_s": "fast"}', '{"waypoints": [{"joints": {}}], "correct": NaN}', "\x00"]


def test_malformed_input_prints_an_error_does_not_move_and_the_session_goes_on(ps, tmp_path, monkeypatch, capsys):
    rob, lines, _ = serve_with(ps, tmp_path, monkeypatch, capsys, Reader(*BAD, plan_msg(PLAN2)))
    errs = [l for l in lines if "error" in l]; assert len(errs) == len(BAD) and not any(l.get("fatal") for l in errs)
    ref = FakeRobot(ps.J, START); (tmp_path / "r").mkdir(); r2, _ = run_pick_seq(ps, tmp_path / "r", monkeypatch, capsys, plan=PLAN2, robot=ref, argv=["--correct", "0"])
    assert rob.sent and rob.sent == r2.sent, "only the valid plan moved (correct 0 in the message, as in --correct 0)"


def test_malformed_input_alone_sends_nothing(ps, tmp_path, monkeypatch, capsys):
    rob, lines, busy = serve_with(ps, tmp_path, monkeypatch, capsys, Reader(*BAD)); assert rob.sent == [] and rob.reads == 0 and rob.disconnected and not busy.exists()


def test_a_bus_fault_mid_plan_ends_the_session_cleanly(ps, tmp_path, monkeypatch, capsys):
    rob = FakeRobot(ps.J, START); real = rob.send_action
    def send(a):
        if len(rob.sent) == 5: raise OSError("bus")
        return real(a)
    rob.send_action = send; rob, lines, busy = serve_with(ps, tmp_path, monkeypatch, capsys, Reader(plan_msg(PLAN), plan_msg(PLAN2)), robot=rob)
    assert any(l.get("fatal") for l in lines) and rob.disconnected and not busy.exists() and len(rob.sent) == 5


def test_line_reader_on_a_real_pipe(ps):
    r, w = os.pipe(); lr = ps.LineReader(r); os.write(w, b'{"a": 1}\n{"b"'); assert lr.next(1) == '{"a": 1}'
    with pytest.raises(TimeoutError): lr.next(0.05)
    os.write(w, b': 2}\n'); assert lr.next(1) == '{"b": 2}'
    os.write(w, b"tail"); os.close(w); assert lr.next(1) == "tail" and lr.next(1) is None and lr.next(1) is None; os.close(r)


# ---------------------------------------------------------------- the real pipe: a subprocess running serve() on a FakeRobot
SERVER = textwrap.dedent('''
    import os, sys, time, json
    sys.path.insert(0, {jetson!r}); sys.path.insert(0, {demos!r})
    import pick_seq as ps
    from demohelpers import FakeRobot
    mode, counter, busy = sys.argv[1:4]; real_sleep = time.sleep
    START = {start!r}
    def connect():
        open(counter, "a").write("c\\n"); return FakeRobot(ps.J, START)
    ps.connect = connect; ps.lerobot_running = lambda: False; ps.BUSY = busy; ps.time.sleep = lambda s: None
    if mode == "eof_after_first":
        orig, st = ps.LineReader.next, {{"n": 0}}
        def nxt(self, t):
            st["n"] += 1; return orig(self, t) if st["n"] == 1 else None
        ps.LineReader.next = nxt
    if mode in ("die_midplan", "slowmove", "stoponplan"):
        orig_move, calls = ps.move, {{"n": 0}}
        def mv(*a, **k):
            calls["n"] += 1
            if mode == "slowmove": real_sleep(30)
            if mode == "stoponplan": raise KeyboardInterrupt
            if mode == "die_midplan" and calls["n"] == 2: os._exit(1)
            return orig_move(*a, **k)
        ps.move = mv
    sys.argv = ["pick_seq.py", "--serve"] + sys.argv[4:]; ps.main()
''')


@pytest.fixture
def server(tmp_path):
    f = tmp_path / "fake_server.py"; f.write_text(SERVER.format(jetson=os.path.join(ROOT, "jetson"), demos=os.path.join(ROOT, "tests", "demos"), start=START))
    counter, busy = tmp_path / "connects", tmp_path / "arms_busy"
    def argv(mode="ok", *extra): return [sys.executable, str(f), mode, str(counter), str(busy), *extra]
    argv.connects = lambda: len(counter.read_text().split()) if counter.exists() else 0; argv.busy = busy; return argv


def direct_rows(ps, plans, tmp_path, monkeypatch, capsys):
    out, pos = [], dict(START)
    for i, p in enumerate(plans):
        (tmp_path / f"d{i}").mkdir(); rob, ls = run_pick_seq(ps, tmp_path / f"d{i}", monkeypatch, capsys, plan=p, start=pos, argv=["--correct", "0"]); pos = dict(rob.pos); out += rows_of(ls)
    return out


def test_pick_session_runs_four_stages_in_one_connection_and_returns_main_s_rows(ps, pl, server, tmp_path, monkeypatch, capsys):
    plans = [PLAN, PLAN2, PLAN, PLAN2]; want = direct_rows(ps, plans, tmp_path, monkeypatch, capsys)
    s = pl.PickSession(argv=server()); got = []
    try:
        for p in plans: got += [{k: v for k, v in r.items() if k != "t"} for r in pl.run_plan(p, 80.0, session=s)]
    finally: s.close()
    assert got == want and server.connects() == 1 and s.starts == 1 and s.proc is None and not server.busy.exists()
    # the old path: one pick_seq start (one connect) per stage
    n = []; rob = FakeRobot(ps.J, START); rob.disconnect = lambda: n.append(1)   # why: run_pick_seq installs its own connect; one disconnect = one connect
    for i, p in enumerate(plans): (tmp_path / f"o{i}").mkdir(); run_pick_seq(ps, tmp_path / f"o{i}", monkeypatch, capsys, plan=p, robot=rob)
    assert (len(n), server.connects()) == (4, 1), "connects for a 4-stage pick: 4 -> 1"


def test_a_session_that_ends_between_picks_is_restarted_not_replayed(ps, pl, server):
    s = pl.PickSession(argv=server("eof_after_first"))
    try:
        r1 = pl.run_plan(PLAN2, 80.0, session=s); r2 = pl.run_plan(PLAN2, 80.0, session=s)
    finally: s.close()
    assert len(r1) == len(r2) == 2 and s.starts == 2 and server.connects() == 2 and not s.broken


def test_a_dead_session_falls_back_once_to_the_old_path(pl, monkeypatch):
    calls = []; monkeypatch.setattr(pl, "run_plan_ssh", lambda plan, v: calls.append(plan) or [{"wp": "x"}])
    s = pl.PickSession(argv=[sys.executable, "-c", "import sys; sys.exit(3)"])
    assert pl.run_plan(PLAN2, 80.0, session=s) == [{"wp": "x"}] and len(calls) == 1 and s.broken and s.starts == 1
    assert pl.run_plan(PLAN2, 80.0, session=s) == [{"wp": "x"}] and len(calls) == 2 and s.starts == 1, "a broken session is not respawned for every stage"


def test_a_broken_pipe_falls_back_once(pl, monkeypatch):
    calls = []; monkeypatch.setattr(pl, "run_plan_ssh", lambda plan, v: calls.append(plan) or [{"wp": "x"}])
    class P:
        stdin = type("W", (), {"write": lambda self, b: (_ for _ in ()).throw(BrokenPipeError()), "flush": lambda self: None, "close": lambda self: None})(); stdout = None
        def poll(self): return None
        def wait(self, timeout=None): return 0
    s = pl.PickSession(popen=lambda *a, **k: P()); assert pl.run_plan(PLAN2, 80.0, session=s) == [{"wp": "x"}] and len(calls) == 1


def test_a_session_that_dies_mid_plan_is_not_replayed(pl, server, monkeypatch):
    monkeypatch.setattr(pl, "run_plan_ssh", lambda *a: pytest.fail("a half-run plan must not be replayed"))
    s = pl.PickSession(argv=server("die_midplan"))
    with pytest.raises(SystemExit, match="not replaying"): pl.run_plan(PLAN, 80.0, session=s)
    s.close()


def test_stop_means_stop_no_fallback(pl, server, monkeypatch):
    monkeypatch.setattr(pl, "run_plan_ssh", lambda *a: pytest.fail("no replay after so101-stop"))
    s = pl.PickSession(argv=server("stoponplan"))
    with pytest.raises(SystemExit, match="stopped"): pl.run_plan(PLAN, 80.0, session=s)
    assert s.proc is None and not server.busy.exists()


def test_a_plan_times_out_and_the_session_is_closed(pl, server, monkeypatch):
    monkeypatch.setattr(pl, "run_plan_ssh", lambda *a: pytest.fail("no replay after a timeout"))
    s = pl.PickSession(argv=server("slowmove"), timeout=1.5)
    with pytest.raises(subprocess.TimeoutExpired): pl.run_plan(PLAN2, 80.0, session=s)
    assert s.proc is None


def test_the_server_refuses_a_rejected_plan_and_the_session_survives(pl, server):
    s = pl.PickSession(argv=server())
    try:
        with pytest.raises(SystemExit, match="refused"): pl.run_plan({"waypoints": []}, 80.0, session=s)
        assert len(pl.run_plan(PLAN2, 80.0, session=s)) == 2 and s.starts == 1
    finally: s.close()


def test_a_real_sigterm_stops_the_server_and_it_removes_the_busy_file(server):
    p = subprocess.Popen(server("slowmove"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdin.write(json.dumps(plan_msg(PLAN)) + "\n"); p.stdin.flush()
    first = [json.loads(p.stdout.readline()) for _ in range(2)]; assert first[0] == {"ready": True} and "start" in first[1]   # the arm is "in the move" (the 30 s sleep)
    p.send_signal(signal.SIGTERM); rest = [json.loads(l) for l in p.stdout]; assert p.wait(10) == 0 and {"stopped": True} in rest and not server.busy.exists()


def test_a_real_idle_timeout_and_eof_exit(server):
    p = subprocess.Popen(server("ok", "--idle-s", "0.4"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True); out = [json.loads(l) for l in p.stdout]
    assert p.wait(10) == 0 and out[-1] == {"bye": "idle"} and not server.busy.exists()
    p = subprocess.Popen(server(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True); p.stdin.close(); out = [json.loads(l) for l in p.stdout]
    assert p.wait(10) == 0 and out[-1] == {"bye": "eof"} and not server.busy.exists()


def test_close_sends_quit(pl, server):
    s = pl.PickSession(argv=server()); pl.run_plan(PLAN2, 80.0, session=s); proc = s.proc; s.close(); assert proc.poll() == 0 and s.proc is None; s.close()   # idempotent


# ---------------------------------------------------------------- the callers
def test_no_session_means_the_old_path(pl, monkeypatch):
    calls = []; monkeypatch.setattr(pl, "run_plan_ssh", lambda plan, v: calls.append(1) or [{"wp": "x"}]); monkeypatch.setattr(pl, "SESSION", None)
    assert pl.run_plan(PLAN2, 80.0) == [{"wp": "x"}] and calls == [1]


def test_main_closes_the_session_on_any_exception(pl, monkeypatch):
    made = []
    class S:
        broken = False
        def close(self): made.append("closed")
    monkeypatch.setattr(pl, "PickSession", S); monkeypatch.setattr(pl, "SESSION", None); monkeypatch.setattr(sys, "argv", ["pick_loop.py"]); monkeypatch.setattr(pl, "_main", lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError): pl.main()
    assert made == ["closed"] and pl.SESSION is None
    monkeypatch.setattr(sys, "argv", ["pick_loop.py", "--no-session"]); monkeypatch.setattr(pl, "_main", lambda a: setattr(pl, "seen", pl.SESSION))
    pl.main(); assert pl.seen is None


def test_the_session_default_is_on_and_no_session_turns_it_off(pl):
    assert pl.parser().parse_args([]).session is True and pl.parser().parse_args(["--no-session"]).session is False and pl.parser().parse_args(["--session"]).session is True


def test_the_guard_sees_every_plan_and_row_through_a_session(pl, server, monkeypatch):
    al = load(os.path.join(ROOT, "tools", "autoloop.py"), "autoloop_for_session_tests"); seen = {"plans": [], "rows": []}
    class G:
        def check_plan(self, plan): seen["plans"].append([w["name"] for w in plan["waypoints"]])
        def check_waypoint(self, row): seen["rows"].append(row["wp"])
        def trip(self, *a): raise AssertionError(a)
    r = object.__new__(al.Runner); r.pl, r.guard, r.demo, r.tele_rows, r.in_motion = pl, G(), None, [], False
    monkeypatch.setattr(pl, "SESSION", pl.PickSession(argv=server()))
    try:
        with r.guarded():
            for p in (PLAN, PLAN2, PLAN2): pl.run_plan(p, 80.0)
    finally: pl.close_session()
    assert seen["plans"] == [["hover", "grasp", "close", "lift"], ["carry", "open"], ["carry", "open"]] and seen["rows"] == ["hover", "grasp", "close", "lift", "carry", "open", "carry", "open"] and server.connects() == 1


def test_autoloop_runner_closes_the_session_at_the_end(monkeypatch):
    al = load(os.path.join(ROOT, "tools", "autoloop.py"), "autoloop_for_session_close"); closed = []
    r = object.__new__(al.Runner); r.pl = type("P", (), {"close_session": staticmethod(lambda: closed.append(1))}); monkeypatch.setattr(al.Runner, "_run", lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): r.run()
    assert closed == [1]
    assert al.build_parser().parse_args([]).pick_session is True and al.build_parser().parse_args(["--no-pick-session"]).pick_session is False
