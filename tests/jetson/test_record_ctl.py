"""record_ctl.py as a black box: run it as a subprocess around tests/jetson/fake_record.py and read its log."""
import errno, os, resource, signal, stat, subprocess, time
import pytest
from jhelpers import PY, RECORD_CTL, FAKE_RECORD, wait_for


@pytest.fixture
def ctl(tmp_path):
    procs = []
    def start(*child_args, fifo=None, premake_fifo=True):
        log = tmp_path / "record.log"; fifo = fifo or tmp_path / "ctl.fifo"
        if premake_fifo: os.mkfifo(fifo)
        p = subprocess.Popen([PY, RECORD_CTL, str(log), str(fifo), "--", PY, FAKE_RECORD, *map(str, child_args)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(p); return p, str(log), str(fifo)
    yield start
    for p in procs:
        if p.poll() is None: p.kill()
        p.wait(timeout=5)


def send_keys(fifo, data, timeout=5.0):
    """Write to the FIFO once the wrapper holds it open (it opens O_RDWR, so a non-blocking writer open succeeds)."""
    deadline = time.time() + timeout
    while True:
        try: fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK); break
        except OSError as e:
            if e.errno != errno.ENXIO or time.time() > deadline: raise
            time.sleep(0.02)
    try: os.write(fd, data)
    finally: os.close(fd)


def finish(p, log, timeout=5.0):
    p.wait(timeout=timeout); assert p.returncode == 0, p.stderr.read().decode()
    return open(log, "rb").read()


# ---------- output + exit code ----------
@pytest.mark.parametrize("code", [0, 3])
def test_child_output_and_exit_code_land_in_log(ctl, code):
    p, log, _ = ctl("exit", code)
    data = finish(p, log)
    assert b"fake lerobot-record starting" in data
    assert data.rstrip().endswith(f"__RECORD_EXIT__ {code}".encode())


def test_child_killed_by_signal_is_reported_negative(ctl):
    p, log, _ = ctl("killed")
    data = finish(p, log)
    assert b"about to die" in data
    assert data.rstrip().endswith(f"__RECORD_EXIT__ {-signal.SIGKILL}".encode())


# ---------- control keys via the FIFO ----------
def test_control_keys_reach_child_and_are_logged(ctl):
    p, log, fifo = ctl("keys")
    wait_for(log, "READY")
    send_keys(fifo, b"xn")            # x is not a control key: dropped, never reaches the child
    wait_for(log, "GOT n")
    for k in b"rpq":
        send_keys(fifo, bytes([k])); wait_for(log, f"GOT {chr(k)}")
    data = finish(p, log)
    for k in "nrpq": assert f"\n__CTL__ {k}\n".encode() in data
    assert b"__CTL__ x" not in data and b"GOT x" not in data
    assert data.rstrip().endswith(b"__RECORD_EXIT__ 0")


def test_fifo_created_when_missing(ctl, tmp_path):
    fifo = tmp_path / "nested.fifo"
    p, log, _ = ctl("exit", 0, fifo=fifo, premake_fifo=False)
    finish(p, log)
    assert stat.S_ISFIFO(os.stat(fifo).st_mode)


def test_idle_fifo_with_no_writer_does_not_spin(ctl):
    """O_RDWR keeps the FIFO from reading EOF forever once the dashboard's `printf k > fifo` writer has come and gone;
    with O_RDONLY, Linux select() returns immediately from then on and the loop burns a core. (macOS reports that EOF
    only once, so this test cannot fail for that reason here; it still bounds the wrapper's idle CPU on any host.)"""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    p, log, fifo = ctl("wait")
    wait_for(log, "WAITING")
    send_keys(fifo, b"x")             # a writer connects and closes; from here on the FIFO has no writer
    time.sleep(1.0)
    p.send_signal(signal.SIGTERM); finish(p, log)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    assert cpu < 0.5, f"wrapper+child used {cpu:.2f}s CPU while idle for 1s (spinning select loop?)"


# ---------- calibration prompt ----------
def test_calibration_prompt_is_auto_answered(ctl):
    p, log, _ = ctl("prompt")
    data = finish(p, log)
    assert b"__CTL__ auto-ENTER" in data
    assert b"GOT_ENTER" in data, "child never received the newline"
    assert data.count(b"__CTL__ auto-ENTER") == 1, "the echoed newline must not trigger a second ENTER"
    assert data.rstrip().endswith(b"__RECORD_EXIT__ 0")


def test_calibration_prompt_split_across_reads_is_still_answered(ctl):
    p, log, _ = ctl("split-prompt")
    data = finish(p, log, timeout=8)
    assert b"GOT_ENTER" in data, "prompt arrived in two chunks and the wrapper missed it"
    assert data.count(b"__CTL__ auto-ENTER") == 1
    assert data.rstrip().endswith(b"__RECORD_EXIT__ 0")


# ---------- signals ----------
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_to_wrapper_is_forwarded_to_child(ctl, sig):
    p, log, _ = ctl("wait")
    wait_for(log, "WAITING")
    p.send_signal(sig)
    data = finish(p, log)
    assert f"CHILD_GOT_{sig.name}".encode() in data
    assert data.rstrip().endswith(b"__RECORD_EXIT__ 0")
