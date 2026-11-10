"""say() relay to so101_say.jsonl, and the later _say_track wrapper that must sit on top of it."""
import lerobot.utils.utils as lu


def test_say_is_tracked_wrapper_over_the_relay(plugin):
    # ordering subtlety: the pause section wraps whatever `say` was current, which must be the relay
    assert lu.say is plugin._say_track
    assert plugin._prev_say is plugin._say
    assert plugin._say._busheal and plugin._say_track._busheal


def test_say_relays_to_jsonl_and_tracks_text(plugin, say_lines, local_say, clock):
    lu.say("Recording episode 3")
    lines = say_lines()
    assert len(lines) == 1 and lines[0]["say"] == "Recording episode 3" and lines[0]["t"] == clock.time()
    assert plugin.LAST_SAY["text"] == "Recording episode 3"
    assert local_say.calls == [], "no local speech unless LEROBOT_LOCAL_SAY=1 (no audio over ssh)"


def test_say_local_when_env_set(plugin, say_lines, local_say, monkeypatch, clock):
    monkeypatch.setenv("LEROBOT_LOCAL_SAY", "1")
    lu.say("Reset the environment", blocking=True)
    assert local_say.calls == [("Reset the environment", True)]
    assert len(say_lines()) == 1


def test_log_say_reaches_relay_as_record_uses_it(plugin, say_lines, clock):
    lu.log_say("Reset the environment", play_sounds=True)
    assert [l["say"] for l in say_lines()] == ["Reset the environment"]
    assert plugin.LAST_SAY["text"] == "Reset the environment"


def test_log_say_silent_leaves_last_say_untouched(plugin, say_lines, clock):
    lu.log_say("Reset the environment", play_sounds=False)
    assert say_lines() == [] and plugin.LAST_SAY["text"] == ""


def test_relay_survives_unwritable_state_dir(plugin, state_dir, monkeypatch, clock):
    monkeypatch.setattr(plugin, "STATE_DIR", str(state_dir / "missing"))
    lu.say("still fine")
    assert plugin.LAST_SAY["text"] == "still fine"
