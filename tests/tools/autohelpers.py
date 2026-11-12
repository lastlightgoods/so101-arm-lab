"""Fixtures for tools/autoloop.py and tools/guardrails.py. Nothing reaches a dashboard, an ssh session or the arm:
  * `al` / `gr` load tools/autoloop.py and tools/guardrails.py by path;
  * `autorig` builds a Rig around the `pl` fixture of pickhelpers.py: pick_loop's api / run_plan / find_nut are the ShortArm (real kinematics,
    reaches 7 mm short) and autoloop.FakeWorld (the dashboard's /api/state and /api/scene plus a nut that a pick moves), labtrack is the
    Recorder, and time is a Clock that only advances when the code sleeps, so a "10 s" watchdog costs no wall time;
  * Rig.run(...) runs a whole session and returns (summary, every stdout line parsed as JSON)."""
import json, os, random, sys
import pytest
from pickhelpers import Recorder, ShortArm, load, FIXTURE_TABLE, TOOLS  # noqa: F401

SAFE = (6.0, 4.0, 13.0, 8.0)   # the approved safe box (pick_loop.SAFE_BOX)


@pytest.fixture(scope="session")
def al(): return load(os.path.join(TOOLS, "autoloop.py"), "autoloop_under_test")


@pytest.fixture(scope="session")
def gr(): return load(os.path.join(TOOLS, "guardrails.py"), "guardrails_under_test")


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def now(self): return self.t
    def sleep(self, s): self.t += s


class Rig:
    def __init__(self, al, gr, pl, cc, table, tmp_path, monkeypatch, shortfall_mm=7.0, nut_in=(6.0, 8.0), script=(), seed=7):
        self.al, self.gr, self.pl, self.cc, self.table, self.tmp = al, gr, pl, cc, table, tmp_path
        self.clock = Clock(); self.rng = random.Random(seed); self.notes = []
        self.arm = ShortArm(cc, table, shortfall_mm=shortfall_mm); self.world = al.FakeWorld(cc, table, gr.fk_mat_mm, nut_in=nut_in, rng=random.Random(seed + 1), script=script)
        self.row_hook = None   # fn(row) -> row, to inject a bad pick_seq row
        def run_plan(plan, max_deg_s):
            rows = self.arm.run_plan(plan, max_deg_s); self.world.on_plan(plan)
            return [self.row_hook(r) for r in rows] if self.row_hook else rows
        monkeypatch.setattr(pl, "api", self.arm.api); monkeypatch.setattr(pl, "run_plan", run_plan); monkeypatch.setattr(pl, "find_nut", self.world.find_nut); monkeypatch.setattr(pl, "fetch_wrist", lambda: None); monkeypatch.setattr(pl, "fetch_scene_objects", lambda: self.world.api_scene()["objects"]); monkeypatch.setattr(pl, "fetch_scene_state", lambda: (self.world.api_scene()["objects"], 0))   # why: hermetic: the jaw check and the finder must never reach a live dashboard
        self.stop_file = str(tmp_path / "autoloop.stop"); self.limits = {}
        self.ctx = gr.Ctx(self.state, self.scene, self.clock.now, self.clock.sleep, self.notes.append)

    def state(self): return self.world.api_state()
    def scene(self): return self.world.api_scene()

    def cfg(self, **kw):
        d = dict(cycles=10, minutes=None, target=(9.0, 6.0), random_target=False, safe_box=SAFE, radius_mm=15.0, session="t", dry_run=True, max_retries=1, settle_s=1.0, moved_mm=20.0,
                 spot_mm=(40.0, 50.0, 60.0), out_dir=str(self.tmp / "out"), judge_wait_s=4.0, save_frames=False); d.update(kw)
        import argparse; return argparse.Namespace(**d)

    def run(self, capsys, **kw):
        lim = self.gr.Limits(stop_file=self.stop_file, **{**self.limits, **{k: kw.pop(k) for k in list(kw) if k in self.gr.Limits.__dataclass_fields__}})
        guard = self.gr.Guard(self.cc, self.table, lim, self.ctx); track = Recorder(); self.track = track; demo = kw.pop("demo", None)
        torque_off = kw.pop("torque_off", None); self.runner = self.al.Runner(self.pl, self.cc, self.table, guard, self.ctx, self.cfg(**kw), track, self.rng, out=print, demo=demo, torque_off=torque_off); self.guard = guard
        summary = self.runner.run()
        out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
        return summary, out


@pytest.fixture
def autorig(al, gr, pl, cc, table, tmp_path, monkeypatch):
    return lambda **kw: Rig(al, gr, pl, cc, table, tmp_path, monkeypatch, **kw)


def events(out, kind): return [o for o in out if o.get("event") == kind]
