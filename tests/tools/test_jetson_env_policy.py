"""jetson/env.sh: so101-policy-current resolves the deployed checkpoint (~/so101/policies/<name>/current via DEPLOYED.json or
$SO101_POLICY, optionally filtered by policy type), and so101-run / so101-run-async / so101-run-vla default to it, print which
version runs, and fall back to the old fixed paths when nothing is deployed. Sourced against a fake $HOME; lerobot-rollout is a shim."""
import os
import pytest


def test_policy_current_resolves_from_deployed_json(shell):
    cur = shell.deploy("so101-fake", 2)
    rc, out = shell.run("so101-policy-current")
    assert rc == 0 and out.strip() == cur
    assert shell.run("so101-policy-current so101-fake act")[0] == 0
    rc, out = shell.run("so101-policy-current so101-fake smolvla"); assert rc == 1 and out.strip() == ""   # type filter: an ACT deployment is not a VLA one
    rc, out = shell.run("so101-policy-current so101-other"); assert rc == 1 and out.strip() == ""


def test_policy_current_env_override_and_nothing_deployed(shell):
    rc, out = shell.run("so101-policy-current"); assert rc == 1 and out.strip() == "", "no DEPLOYED.json: empty and exit 1"
    shell.deploy("so101-a", 1); shell.deploy("so101-b", 3)   # DEPLOYED.json now names so101-b
    assert shell.run("so101-policy-current")[1].strip().endswith("/so101-b/current")
    assert shell.run("so101-policy-current", SO101_POLICY="so101-a")[1].strip().endswith("/so101-a/current")
    os.unlink(os.path.join(str(shell.so), "policies", "so101-a", "v1", "model.safetensors"))
    assert shell.run("so101-policy-current", SO101_POLICY="so101-a")[0] == 1, "a current without weights does not count"


@pytest.mark.parametrize("fn", ["so101-run 5", "so101-run-async 5"])
def test_run_defaults_to_the_deployed_act_policy_and_says_so(shell, fn):
    cur = shell.deploy("so101-fake", 2)
    rc, out = shell.run(fn)
    assert rc == 0, out
    assert "policy: so101-fake v2 (" + cur + ")" in out
    assert f"--policy.path={cur}" in shell.argv() and "--duration=5" in shell.argv()
    assert not (shell.so / ".cams_busy").exists() and not (shell.so / ".arms_busy").exists()


def test_run_falls_back_to_the_old_default_and_an_explicit_checkpoint_still_wins(shell):
    rc, out = shell.run("so101-run 5")
    assert rc == 0 and f"--policy.path={shell.home}/so101/policies/act_lighter" in shell.argv() and "policy: " in out and "not a registry deployment" in out
    shell.deploy("so101-fake", 2)
    rc, out = shell.run("so101-run 5 /somewhere/else")
    assert rc == 0 and "--policy.path=/somewhere/else" in shell.argv() and "policy: /somewhere/else" in out


def test_run_vla_only_takes_a_smolvla_deployment(shell):
    shell.deploy("so101-fake", 2)   # the deployed model is ACT: so101-run-vla must not pick it
    rc, out = shell.run("so101-run-vla 5")
    assert rc == 0 and f"--policy.path={shell.home}/so101/policies/smolvla_lighter" in shell.argv()
    vla = shell.deploy("so101-vla", 1, kind="smolvla")   # DEPLOYED.json now names the VLA model
    rc, out = shell.run("so101-run-vla 5")
    assert rc == 0 and f"--policy.path={vla}" in shell.argv() and "policy: so101-vla v1" in out
    shell.deploy("so101-fake", 3)   # back to an ACT deployment on top; SO101_VLA_POLICY still finds the VLA one
    rc, out = shell.run("so101-run-vla 5", SO101_VLA_POLICY="so101-vla")
    assert rc == 0 and f"--policy.path={vla}" in shell.argv()
    rc, out = shell.run("so101-run 5"); assert f"--policy.path={shell.home}/so101/policies/so101-fake/current" in shell.argv() and "policy: so101-fake v3" in out
