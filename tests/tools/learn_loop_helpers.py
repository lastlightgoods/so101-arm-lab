"""Shared by tests/tools/test_learn_loop*.py: tools/learn_loop.py and its fakes loaded by path, a `run` that calls learn_loop.main in-process on a tmp out-dir, and the `lrt` fixture that
builds an evaluate runtime on the ShortArm / FakeWorld rig of autohelpers.py (real kinematics, the autoloop's Runner and judge) with a FakeRollout standing in for the Jetson."""
import json, os, random, shutil
import pytest
from pickhelpers import load, TOOLS, Recorder

mlflow = pytest.importorskip("mlflow", reason="the registry stages need mlflow (not in requirements-dev.txt)")
if not shutil.which("rsync"): pytest.skip("rsync not installed", allow_module_level=True)
LL = load(os.path.join(TOOLS, "learn_loop.py"), "learn_loop_under_test"); FF = load(os.path.join(TOOLS, "learn_loop_fakes.py"), "learn_loop_fakes_under_test"); EV = load(os.path.join(TOOLS, "learn_loop_eval.py"), "learn_loop_eval_under_test")
PRE = ("record", "train", "register", "deploy")
DEFAULT = ("--dry-run", "--where", "local", "--cycles", "6", "--trials", "3", "--steps", "1000", "--save-freq", "500")


def run(tmp, *argv, sys_=None, transport=None, runtime=None, name="t1", extra=DEFAULT):
    return LL.main([*argv, "--name", name, "--out-dir", str(tmp / "ll"), *extra], sys_=sys_ or FF.FakeSys(), transport=transport, runtime=runtime)


def state(tmp, name="t1"): return json.load(open(tmp / "ll" / name / "state.json"))


def client(tmp, name="t1"): return mlflow.MlflowClient(tracking_uri="sqlite:///" + str(tmp / "ll" / name / "mlflow.db"))


def events(capsys):
    """Every JSON line the session printed (trial, stop, trip, stage ...)."""
    out = []
    for line in capsys.readouterr().out.splitlines():
        try: out.append(json.loads(line))
        except ValueError: pass
    return out


@pytest.fixture
def lrt(autorig, al, gr):
    """make(script=(), p=0.7, on_tick=None, escape_at_s=3.0, **rig kwargs) -> a runtime factory for Loop (.rig, .transport, .moves attached). Every plan the arm runs is logged in .moves."""
    def make(script=(), p=0.7, on_tick=None, escape_at_s=3.0, **rigkw):
        rig = autorig(nut_in=(6.0, 8.0), **rigkw); tr = FF.FakeRollout(rig.world, rig.clock, (9.0, 6.0), p=p, script=script, rng=random.Random(5), on_tick=on_tick, escape_at_s=escape_at_s); moves = []
        orig = rig.arm.run_plan
        def logged(plan, max_deg_s):
            moves.append([w["name"] for w in plan["waypoints"]]); return orig(plan, max_deg_s)
        rig.arm.run_plan = logged
        def factory(L, ev):
            a = L.a; lim = gr.Limits(stop_file=a.stop_file, max_failures=a.max_consecutive_failures, lost_after=a.lost_after); guard = gr.Guard(rig.cc, rig.table, lim, rig.ctx)
            rn = al.Runner(rig.pl, rig.cc, rig.table, guard, rig.ctx, rig.cfg(settle_s=0.0, judge_wait_s=0.0), Recorder(), rig.rng, out=print); rig.rn = rn
            return {"rn": rn, "al": al, "gr": gr, "guard": guard, "transport": tr}
        factory.rig, factory.transport, factory.moves = rig, tr, moves
        return factory
    return make
