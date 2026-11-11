"""The plugin's servo-health mirror: after a Present_Position sync_read it adds ONE block read (at most every LEROBOT_TELE_S) and writes so101_<role>_tele.json."""
import json
import os
import sys

import pytest

from busheal_fakes import FakeSyncReader, make_bus

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "jetson"))
import servo_telemetry as st  # noqa: E402  (the same file the plugin looks up in $SO101_HOME)


class BlockReader(FakeSyncReader):
    """FakeSyncReader that also answers block reads: data_dict[id] is the 31-byte live block (or the EEPROM limits block) of a servo at rest."""
    def __init__(self, sim):
        super().__init__(sim)
        self.data_dict = {}
        self.packets = []

    def txRxPacket(self):
        comm = super().txRxPacket()
        self.packets.append((self.start_address, self.data_length))
        if self.start_address == st.BLOCK[0]:
            self.data_dict = {i: list(st.encode_block(pos=2000 + i, goal=2003 + i, volt=11.8, temp=40 + i, cur=120, load=150)) for i in self.ids}
        elif self.start_address == st.LIMITS_BLOCK[0]:
            lim = bytearray(st.LIMITS_BLOCK[1]); lim[0] = 70; lim[1] = 140
            self.data_dict = {i: list(lim) for i in self.ids}
        return comm

    def isAvailable(self, id_, addr, length):
        return id_ in self.data_dict


@pytest.fixture
def tele(plugin, monkeypatch):
    plugin._tele_mod[:] = [st]
    monkeypatch.setattr(plugin, "TELE_S", 0.5)
    bus, sim = make_bus()
    reader = BlockReader(sim); sim.reader = reader; bus.sync_reader = reader
    return bus, sim, reader


def files(state_dir):
    return sorted(p.name for p in state_dir.glob("so101_*.json"))


def test_position_read_also_publishes_telemetry(plugin, tele, state_dir, clock):
    bus, sim, reader = tele
    out = bus.sync_read("Present_Position", normalize=False)
    assert out["gripper"] == 2006                                   # the control loop's own read is untouched
    assert files(state_dir) == ["so101_follower.json", "so101_follower_tele.json"]
    d = json.loads((state_dir / "so101_follower_tele.json").read_text())
    assert d["role"] == "follower" and d["src"] == "lerobot" and set(d["j"]) == set(st.JOINTS)
    assert d["j"]["shoulder_lift"]["temp_c"] == 42 and d["j"]["shoulder_lift"]["volt_v"] == 11.8 and d["j"]["gripper"]["cur_a"] == pytest.approx(0.78)
    assert d["limits"]["gripper"]["temp_cut_c"] == 70
    assert [p for p in reader.packets if p[0] == st.BLOCK[0]] == [st.BLOCK], "one packet, registers 40..70, for all six servos"


def test_limits_are_read_once_and_telemetry_is_rate_limited(plugin, tele, state_dir, clock):
    bus, sim, reader = tele
    extra = lambda: [p for p in reader.packets if p != (56, 2)]   # everything but the control loop's own position reads
    bus.sync_read("Present_Position", normalize=False)
    assert extra() == [st.BLOCK, st.LIMITS_BLOCK]
    clock.advance(0.1); bus.sync_read("Present_Position", normalize=False)
    assert len(extra()) == 2, "inside LEROBOT_TELE_S: no extra bus traffic at all"
    clock.advance(0.6); bus.sync_read("Present_Position", normalize=False)
    assert extra() == [st.BLOCK, st.LIMITS_BLOCK, st.BLOCK], "one more block read, and no second limits read"
    d = json.loads((state_dir / "so101_follower_tele.json").read_text())
    assert "limits" not in d


def test_only_the_follower_by_default(plugin, tele, state_dir, clock):
    bus, sim, reader = tele
    bus.port = "/dev/tty.so101_leader"; bus.port_handler = sim.port
    bus.sync_read("Present_Position", normalize=False)
    assert files(state_dir) == ["so101_leader.json"] and not [p for p in reader.packets if p[0] == st.BLOCK[0]]


def test_switch_off(plugin, tele, state_dir, clock, monkeypatch):
    bus, sim, reader = tele
    monkeypatch.setattr(plugin, "TELE_S", 0)
    bus.sync_read("Present_Position", normalize=False)
    assert files(state_dir) == ["so101_follower.json"] and reader.packets == [(56, 2)]


def test_other_reads_never_trigger_it(plugin, tele, state_dir, clock):
    bus, sim, reader = tele
    bus.sync_read("Present_Velocity", normalize=False)
    bus.read("Present_Position", "gripper", normalize=False)
    assert files(state_dir) == [] and not [p for p in reader.packets if p[0] == st.BLOCK[0]]


def test_a_dead_telemetry_read_does_not_touch_the_control_reads(plugin, tele, state_dir, clock, monkeypatch):
    bus, sim, reader = tele
    monkeypatch.setattr(st, "read_sample", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = bus.sync_read("Present_Position", normalize=False)
    assert out["gripper"] == 2006 and files(state_dir) == ["so101_follower.json"]


def test_unwritable_state_dir_is_swallowed(plugin, tele, state_dir, clock, monkeypatch):
    bus, _, _ = tele
    monkeypatch.setattr(plugin, "STATE_DIR", str(state_dir / "missing"))
    assert bus.sync_read("Present_Position", normalize=False)["gripper"] == 2006


def test_without_the_module_the_plugin_runs_as_before(plugin, tele, state_dir, clock):
    bus, _, reader = tele
    plugin._tele_mod[:] = [None]
    bus.sync_read("Present_Position", normalize=False)
    assert files(state_dir) == ["so101_follower.json"]


def test_module_is_found_through_so101_home(plugin, state_dir, monkeypatch):
    """The plugin lives in the venv, the decoder in ~/so101: $SO101_HOME says where, and a missing file means telemetry off, not an error."""
    jet = os.path.join(ROOT, "jetson")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != jet])   # restored by monkeypatch
    monkeypatch.delitem(sys.modules, "servo_telemetry")                   # restored by monkeypatch
    plugin._tele_mod.clear(); monkeypatch.setenv("SO101_HOME", str(state_dir / "nowhere"))
    assert plugin._telemetry() is None
    plugin._tele_mod.clear(); monkeypatch.setenv("SO101_HOME", jet)
    assert plugin._telemetry().__name__ == "servo_telemetry"
