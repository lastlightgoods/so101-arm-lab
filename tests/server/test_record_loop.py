"""record_loop(): progress parsed from the ui_record.log tail that record_ctl.py writes on the edge host.
Each poll is one ssh call returning the log tail (+ __ALIVE__ while lerobot-record runs, + '__EPS__ n' from info.json)."""
import time
import pytest

ALIVE = "\n__ALIVE__\n"
TORCHCODEC = "RuntimeError: Could not load libtorchcodec. Likely causes: FFmpeg is not properly installed (falls back to pyav)\n"


@pytest.fixture
def poll(srv, fake, run_loop, monkeypatch):
    """Drive record_loop over successive log tails (str or bytes); one loop pass per tail. REC must be running."""
    monkeypatch.setattr(srv, "record_reattach", lambda: False)
    def _go(*tails, started_ago=0):
        it = iter(tails); fake.on("tail -c 6000", lambda argv, kw: next(it))
        srv.REC.update(running=True, name=srv.REC["name"] or "cube", started=time.time() - started_ago)
        run_loop(srv.record_loop, len(tails))
    return _go


def test_tail_command_targets_this_dataset(srv, poll, fake):
    srv.REC["name"] = "cube"
    poll("Recording episode 1" + ALIVE)
    cmd = fake.ssh[-1]
    assert "tail -c 6000 ~/so101/outputs/run/ui_record.log" in cmd and "pgrep -f 'bin/lerobot-record'" in cmd
    assert "lerobot/lastlight/cube/meta/info.json" in cmd and fake.calls[-1][0][:2] == ["ssh", "-o"]


def test_recording_episode_sets_phase_and_clock(srv, poll):
    t0 = time.time()
    poll("Recording episode 1" + ALIVE)
    assert srv.REC["episode"] == 1 and srv.REC["phase"] == "recording" and srv.REC["running"] is True
    assert srv.REC["ep_started"] == pytest.approx(t0, abs=1)
    assert srv.REC["log"].strip() == "Recording episode 1", "__ALIVE__ is stripped from the log shown on the page"


def test_same_episode_seen_again_keeps_clock(srv, poll):
    poll("Recording episode 1" + ALIVE)
    srv.REC["ep_started"] = 1000.0
    poll("Recording episode 1" + ALIVE)
    assert srv.REC["ep_started"] == 1000.0


def test_reset_phase_after_episode(srv, poll):
    poll("Recording episode 1\nReset the environment" + ALIVE)
    assert srv.REC["phase"] == "reset" and srv.REC["episode"] == 1
    poll("Recording episode 1\nReset the environment\nRecording episode 2" + ALIVE)
    assert srv.REC["phase"] == "recording" and srv.REC["episode"] == 2


def test_rerecord_repeats_episode_number_but_resets_clock(srv, poll):
    poll("Recording episode 3" + ALIVE)
    srv.REC["ep_started"] = 1000.0
    poll("Recording episode 3\nReset the environment\n__CTL__ r\nRecording episode 3" + ALIVE)   # same number, one more line
    assert srv.REC["episode"] == 3 and srv.REC["phase"] == "recording"
    assert srv.REC["ep_started"] == pytest.approx(time.time(), abs=1) and srv.REC["_nrec"] == 2


def test_pause_and_resume_shift_episode_clock(srv, poll, events):
    poll("Recording episode 2" + ALIVE)
    poll("Recording episode 2\n__PAUSED__" + ALIVE)
    assert srv.REC["paused_at"] == pytest.approx(time.time(), abs=1) and events()[0] == "RECORD cube paused"
    # pretend the pause lasted 5 s, then busheal prints __RESUMED__
    srv.REC["paused_at"] = time.time() - 5; srv.REC["ep_started"] = 1000.0
    poll("Recording episode 2\n__PAUSED__\n__RESUMED__" + ALIVE)
    assert srv.REC["paused_at"] == 0 and srv.REC["ep_started"] == pytest.approx(1005.0, abs=0.5)
    assert events()[0] == "RECORD cube resumed"
    poll("Recording episode 2\n__PAUSED__\n__RESUMED__" + ALIVE)
    assert events()[0] == "RECORD cube resumed" and len([e for e in events() if "resumed" in e]) == 1, "no repeat on the next poll"


def test_second_pause_after_resume_is_detected(srv, poll, events):
    poll("__PAUSED__\n__RESUMED__\n__PAUSED__" + ALIVE)
    assert srv.REC["paused_at"] > 0 and events()[0] == "RECORD cube paused"


def test_auto_enter_marker_announced_once(srv, poll, events):
    line = "__CTL__ auto-ENTER: calibration mismatch, applying the calibration file\n"
    poll(line + ALIVE, line + "Recording episode 1" + ALIVE, line + "Recording episode 2" + ALIVE)
    assert [e for e in events() if "auto-ENTER" in e] == ["RECORD cube: motor calibration differed from the file; applied the file (auto-ENTER)"]
    assert srv.REC["_cal_noted"] is True


def test_eps_marker_tracks_saved_episodes_relative_to_chain_base(srv, poll):
    poll("Recording episode 1" + ALIVE + "__EPS__ 12\n")
    assert srv.REC["chain_base"] == 12 and srv.REC["saved_total"] == 0 and srv.REC["done_total"] == 0
    assert "__EPS__" not in srv.REC["log"]
    poll("Recording episode 3" + ALIVE + "__EPS__ 14\n")
    assert srv.REC["saved_total"] == 2 and srv.REC["done_total"] == 2


def test_process_gone_within_grace_period_is_not_an_exit(srv, poll):
    poll("", started_ago=5)                            # no __ALIVE__ yet: lerobot-record is still importing torch
    assert srv.REC["running"] is True and srv.REC["done"] is False


def test_clean_exit_after_grace_marks_done(srv, poll, events):
    poll("Recording episode 1\nReset the environment\nRecording episode 2\n__RECORD_EXIT__ 0\n", started_ago=30)
    assert srv.REC["running"] is False and srv.REC["done"] is True and srv.REC["phase"] == "done" and srv.REC["err"] == ""
    assert events()[0] == "RECORD cube finished · 2 episodes"
    assert "__RECORD_EXIT__" in srv.REC["log"]


def test_exit_marker_beats_a_stale_alive(srv, poll):
    poll("Recording episode 1\n__RECORD_EXIT__ 0" + ALIVE, started_ago=30)   # pgrep raced the exit
    assert srv.REC["running"] is False and srv.REC["phase"] == "done"


def test_traceback_picks_cause_line_never_torchcodec_or_nvrtc(srv, poll, events):
    tail = (TORCHCODEC + "Recording episode 1\nTraceback (most recent call last):\n  File \"record.py\", line 1, in <module>\n"
            "ConnectionError: Camera top_cam not found\n"
            "nvrtc: error: invalid value for --gpu-architecture (-arch)\n__RECORD_EXIT__ 1\n")
    poll(tail, started_ago=30)
    assert srv.REC["err"] == "ConnectionError: Camera top_cam not found" and srv.REC["phase"] == "error"
    assert events()[0] == "RECORD failed: ConnectionError: Camera top_cam not found"


def test_traceback_without_error_word_falls_back_to_last_line(srv, poll):
    poll("Recording episode 1\nTraceback (most recent call last):\n  File \"x\"\nKeyboardInterrupt\n__RECORD_EXIT__ 130\n", started_ago=30)
    assert srv.REC["err"] == "KeyboardInterrupt" and srv.REC["phase"] == "error"


def test_clean_exit_with_torchcodec_noise_is_still_done(srv, poll, events):
    """torchcodec prints a harmless 'RuntimeError: Could not load libtorchcodec' on every start; on a short session it is
    still inside the 6000-byte tail at exit. That must not turn a clean exit into 'RECORD failed: Recording episode 2'."""
    poll(TORCHCODEC + "Recording episode 1\nReset the environment\nRecording episode 2\n__RECORD_EXIT__ 0\n", started_ago=30)
    assert srv.REC["err"] == "" and srv.REC["phase"] == "done"
    assert events()[0] == "RECORD cube finished · 2 episodes"


def test_tail_cut_inside_a_multibyte_char_does_not_stall_the_loop(srv, poll, logs):
    """tail -c 6000 can start mid-character (lerobot prints ━ progress bars). A strict decode would raise on every poll
    and, once the log stops growing at exit, the recording would never be marked finished."""
    tail = "━━━".encode()[1:] + b"\nRecording episode 1\n__RECORD_EXIT__ 0\n"
    poll(tail, started_ago=30)
    assert not [l for l in logs if "UnicodeDecodeError" in l], logs
    assert srv.REC["running"] is False and srv.REC["phase"] == "done"


def test_ssh_failure_is_logged_and_loop_continues(srv, fake, run_loop, monkeypatch, logs):
    monkeypatch.setattr(srv, "record_reattach", lambda: False)
    fake.on("tail -c 6000", lambda argv, kw: (_ for _ in ()).throw(TimeoutError("ssh hung")))
    srv.REC.update(running=True, name="cube", started=time.time())
    run_loop(srv.record_loop, 1)
    assert any("record loop:" in l and "ssh hung" in l for l in logs) and srv.REC["running"] is True


def test_idle_loop_sleeps_without_ssh(srv, fake, run_loop, monkeypatch):
    monkeypatch.setattr(srv, "record_reattach", lambda: False)
    run_loop(srv.record_loop, 3)
    assert fake.calls == []
