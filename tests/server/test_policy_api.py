"""/api/policy: the Jetson's ~/so101/policies/DEPLOYED.json (written by tools/so101_policy.py deploy) rides along in the slow
ssh probe (###deployed section), is cached in state["policy"], and is served without any ssh of its own."""
from test_probe import JETSON_PROBE, MAC_PROBE

DEPLOYED = {"name": "so101-lighter", "version": 3, "run_id": "cb5e60b2c7bb4a46a45dfd340bb4eb46", "sha256": "b9ee64de5f00d1a27efbcd0cc6da35b1f7e710cf236f495a78848c02aa0c3760",
            "when": "2026-11-10T01:15:22-0800", "by": "tc", "from_host": "mac", "action": "rollback", "path": "~/so101/policies/so101-lighter/current",
            "history": [{"name": "so101-lighter", "version": 2}, {"name": "so101-lighter", "version": 3}]}
import json as _json
WITH_DEPLOYED = JETSON_PROBE.replace("###end", "###policy /home/tc/so101/policies/so101-lighter/current/train_config.json\n1762540002\n"
                                     '{"policy": {"type": "act", "device": "cuda"}, "steps": 25000, "dataset": {"repo_id": "lastlight/lighter"}}\n'
                                     "###deployed /home/tc/so101/policies/DEPLOYED.json\n" + _json.dumps(DEPLOYED, indent=1) + "\n###end")


def test_probe_script_reads_deployed_json_and_current_checkpoints(srv):
    assert "~/so101/policies/DEPLOYED.json" in srv.PROBE and '###deployed' in srv.PROBE and "~/so101/policies/*/current/train_config.json" in srv.PROBE
    assert srv.PROBE.rstrip().endswith('echo "###end"'), "DEPLOYED.json comes before ###end so a missing file cannot truncate the probe"


def test_parse_probe_deployed_section(srv, logs):
    out = srv.parse_probe(WITH_DEPLOYED, "jetson")
    assert out["deployed"] == {"name": "so101-lighter", "version": 3, "run_id": DEPLOYED["run_id"], "sha256": DEPLOYED["sha256"], "when": DEPLOYED["when"], "by": "tc",
                               "action": "rollback", "path": "~/so101/policies/so101-lighter/current", "host": "jetson", "history": 2}
    assert out["policies"]["so101-lighter"] == {"host": "jetson", "type": "act", "steps": 25000, "dataset": "lastlight/lighter", "device": "cuda", "mtime": 1762540002}, "<name>/current/train_config.json lists as <name>"
    assert "deployed" not in srv.parse_probe(JETSON_PROBE, "jetson"), "no section, no key (older Jetsons and the Mac)"
    bad = srv.parse_probe("###deployed /x/DEPLOYED.json\n{nope\n###end\n", "jetson")
    assert "deployed" not in bad and any(l.startswith("deployed parse") for l in logs)


def test_probe_loop_caches_it_and_api_policy_serves_it_without_ssh(srv, fake, run_loop, http, logs):
    assert http.get_json("/api/policy")[0] == {}, "nothing probed yet"
    fake.on(lambda a: a[0] == "bash", MAC_PROBE)
    fake.on(lambda a: a[0] == "ssh" and a[-2:] == ["bash", "-s"], WITH_DEPLOYED)
    run_loop(srv.probe_loop, 1)
    n = len(fake.calls)
    j, _ = http.get_json("/api/policy")
    assert j["name"] == "so101-lighter" and j["version"] == 3 and j["host"] == "jetson" and j["action"] == "rollback" and j["history"] == 2 and j["probed"] > 0
    assert len(fake.calls) == n, "the GET opened no ssh"
    s, _ = http.get_json("/api/state"); assert s["policy"]["name"] == "so101-lighter" and "so101-lighter@jetson" not in s["policies"] and s["policies"]["so101-lighter"]["host"] == "jetson"


def test_probe_without_deployed_file_clears_the_cache_and_ssh_failure_keeps_it(srv, fake, run_loop, http, logs):
    fake.on(lambda a: a[0] == "bash", MAC_PROBE)
    probes = iter([WITH_DEPLOYED, JETSON_PROBE])
    fake.on(lambda a: a[0] == "ssh" and a[-2:] == ["bash", "-s"], lambda argv, kw: next(probes))
    run_loop(srv.probe_loop, 2)
    assert http.get_json("/api/policy")[0].keys() == {"probed"}, "DEPLOYED.json gone on the Jetson: only the probe time remains"
    srv.state["policy"] = {"name": "so101-lighter", "version": 3, "probed": 1.0}
    fake.on(lambda a: a[0] == "ssh", "", returncode=255, stderr="ssh: connect to host: timed out")
    run_loop(srv.probe_loop, 1)
    assert http.get_json("/api/policy")[0]["name"] == "so101-lighter", "an unreachable Jetson keeps the last answer"
