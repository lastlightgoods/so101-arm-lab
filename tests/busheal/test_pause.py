"""Record pause (key p) and REDO-during-reset, installed because lerobot_record was imported first (conftest)."""
import sys
import time

import pytest

import lerobot.utils.keyboard_input as ki
import lerobot.utils.utils as lu
from lerobot.datasets.lerobot_dataset import LeRobotDataset

rec = sys.modules["lerobot.scripts.lerobot_record"]
T0 = 1000.0   # Clock start


def events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


class FakeListener:
    """Stand-in for the real create_key_listener: captures the dispatch instead of grabbing the keyboard."""

    def __init__(self):
        self.dispatch = None
        self.kwargs = None

    def __call__(self, dispatch, **kw):
        self.dispatch, self.kwargs = dispatch, kw
        return self

    def press(self, *names):
        for n in names:
            self.dispatch(n)

    def stop(self):
        pass


@pytest.fixture
def listener(plugin, monkeypatch):
    fl = FakeListener()
    monkeypatch.setattr(plugin, "_orig_ckl", fl)
    return fl


# --- installation ---------------------------------------------------------------------------------------------------

def test_pause_section_installed(plugin):
    assert plugin.PAUSE["on"] is False
    assert isinstance(rec.time, plugin._PausableTime)
    assert LeRobotDataset.add_frame._busheal
    assert ki.create_key_listener._busheal and ki.apply_recording_control._busheal
    assert lu.say is plugin._say_track


# --- the pausable clock ---------------------------------------------------------------------------------------------

def test_pausable_time_passes_other_attributes_through(plugin, clock):
    assert rec.time.sleep is time.sleep                 # the patched one: proves it is a live lookup
    assert rec.time.monotonic() == clock.now
    assert rec.time.time() == clock.time()
    assert rec.time.perf_counter_ns is time.perf_counter_ns


def test_perf_counter_freezes_while_paused_and_is_continuous(plugin, clock):
    pt = rec.time.perf_counter
    assert pt() == T0
    clock.advance(5)
    plugin.busheal_pause_toggle()                       # pause at 1005
    assert pt() == 1005.0
    clock.advance(10)
    assert pt() == 1005.0, "the record clock stands still while paused"
    plugin.busheal_pause_toggle()                       # resume at 1015
    assert pt() == 1005.0, "no jump on resume"
    clock.advance(2)
    assert pt() == 1007.0
    plugin.busheal_pause_toggle()
    clock.advance(3)
    plugin.busheal_pause_toggle()
    clock.advance(1)
    assert pt() == 1008.0, "lost time accumulates across pauses (13 s of 21 s)"
    assert time.perf_counter() == 1021.0, "the real clock is untouched"


def test_perf_counter_is_consistent_at_every_step_of_a_toggle(plugin, clock, monkeypatch):
    """The key thread toggles while the record thread reads perf_counter(). Observe the clock after each single
    write to PAUSE: it must never show a value other than the one before/after the toggle (no transient jump)."""
    pt = rec.time.perf_counter
    seen = []

    class Observing(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            seen.append((k, pt()))

    monkeypatch.setattr(plugin, "PAUSE", Observing(plugin.PAUSE))
    clock.advance(5)
    plugin.busheal_pause_toggle()                       # pause at 1005: clock must read 1005 throughout
    assert [v for _, v in seen] == [1005.0] * len(seen) and seen
    seen.clear()
    clock.advance(7)
    plugin.busheal_pause_toggle()                       # resume at 1012: clock must read 1005 throughout
    assert [v for _, v in seen] == [1005.0] * len(seen) and seen


def test_pause_toggle_prints_markers_and_speaks(plugin, clock, capsys, say_lines):
    plugin.busheal_pause_toggle()
    assert plugin.PAUSE["on"] is True
    assert "\n__PAUSED__\n" in capsys.readouterr().out
    plugin.busheal_pause_toggle()
    assert plugin.PAUSE["on"] is False
    assert "\n__RESUMED__\n" in capsys.readouterr().out
    assert [l["say"] for l in say_lines()] == ["Paused", "Resumed"]


def test_pause_toggle_survives_a_broken_say(plugin, clock, monkeypatch):
    def boom(text, blocking=False):
        raise OSError("no speaker")

    monkeypatch.setattr(plugin, "_prev_say", boom)
    monkeypatch.setattr(lu, "say", boom)
    plugin.busheal_pause_toggle()
    assert plugin.PAUSE["on"] is True


# --- frames ---------------------------------------------------------------------------------------------------------

def test_add_frame_dropped_while_paused(plugin, monkeypatch):
    added = []
    monkeypatch.setattr(plugin, "_orig_add_frame", lambda self, frame: added.append(frame))
    ds = object.__new__(LeRobotDataset)
    ds.add_frame({"n": 1})
    plugin.PAUSE["on"] = True
    assert ds.add_frame({"n": 2}) is None
    plugin.PAUSE["on"] = False
    ds.add_frame({"n": 3})
    assert added == [{"n": 1}, {"n": 3}]


# --- keys -----------------------------------------------------------------------------------------------------------

def test_key_p_toggles_and_other_keys_pass_through(plugin, clock, listener):
    got = []
    out = ki.create_key_listener(got.append, controls_help="hint")
    assert out is listener and listener.kwargs == {"controls_help": "hint"}
    listener.press("p")
    assert plugin.PAUSE["on"] is True
    listener.press("P")
    assert plugin.PAUSE["on"] is False
    listener.press("n", "left", "esc")
    assert got == ["n", "left", "esc"], "p never reaches lerobot's dispatch, everything else does"


def test_init_keyboard_listener_end_to_end(plugin, clock, listener, capsys):
    out, ev = ki.init_keyboard_listener()
    assert out is listener
    listener.press("p")
    assert plugin.PAUSE["on"] is True and ev == events()
    listener.press("n")
    assert ev["exit_early"] is True
    listener.press("p")
    assert plugin.PAUSE["on"] is False


# --- REDO during the reset -------------------------------------------------------------------------------------------

def test_redo_during_reset_only_queues_the_rerecord(plugin, clock, capsys, say_lines):
    lu.log_say("Reset the environment", play_sounds=True)
    ev = events()
    ki.apply_recording_control("left", ev)
    assert ev == {"exit_early": False, "rerecord_episode": True, "stop_recording": False}
    assert "Re-record queued" in capsys.readouterr().out
    assert [l["say"] for l in say_lines()] == ["Reset the environment", "Redo queued. Finish the reset."]


def test_redo_during_episode_is_stock_lerobot(plugin, clock):
    lu.log_say("Recording episode 4", play_sounds=True)
    ev = events()
    ki.apply_recording_control("left", ev)
    assert ev == {"exit_early": True, "rerecord_episode": True, "stop_recording": False}


def test_redo_with_no_announcement_yet_is_stock(plugin, clock):
    ev = events()
    ki.apply_recording_control("left", ev)
    assert ev["exit_early"] and ev["rerecord_episode"]


def test_next_and_stop_are_unchanged_during_reset(plugin, clock, capsys):
    lu.log_say("Reset the environment", play_sounds=True)
    ev = events()
    ki.apply_recording_control("right", ev)
    assert ev == {"exit_early": True, "rerecord_episode": False, "stop_recording": False}
    ev = events()
    ki.apply_recording_control("esc", ev)
    assert ev == {"exit_early": True, "rerecord_episode": False, "stop_recording": True}


def test_redo_pressed_twice_during_reset_stays_queued(plugin, clock, capsys):
    # the plugin's own "Redo queued" announcement must not count as leaving the reset phase
    lu.log_say("Reset the environment", play_sounds=True)
    ev = events()
    ki.apply_recording_control("left", ev)
    ki.apply_recording_control("left", ev)
    assert ev["rerecord_episode"] is True and ev["exit_early"] is False


def test_pause_during_reset_keeps_redo_queued(plugin, clock, capsys):
    lu.log_say("Reset the environment", play_sounds=True)
    plugin.busheal_pause_toggle()
    plugin.busheal_pause_toggle()
    ev = events()
    ki.apply_recording_control("left", ev)
    assert ev["rerecord_episode"] is True and ev["exit_early"] is False


def test_rerecord_announcement_ends_the_reset_phase(plugin, clock):
    # after the reset lerobot says "Re-record episode" / "Recording episode N": REDO is back to stock
    lu.log_say("Reset the environment", play_sounds=True)
    lu.log_say("Re-record episode", play_sounds=True)
    ev = events()
    ki.apply_recording_control("left", ev)
    assert ev["exit_early"] is True


def test_record_flow_discards_episode_when_redo_is_queued_during_reset():
    """The wrapper relies on lerobot 0.6.1's record() checking rerecord_episode after the reset loop and clearing
    the buffer; pin that so a lerobot upgrade that moves the check is caught here."""
    import inspect

    src = inspect.getsource(rec.record)
    i_reset = src.index('log_say("Reset the environment"')
    i_check = src.index('if events["rerecord_episode"]:')
    i_clear = src.index("dataset.clear_episode_buffer()")
    assert i_reset < i_check < i_clear
