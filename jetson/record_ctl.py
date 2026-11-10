#!/usr/bin/env python3
"""record_ctl.py · run a shell command (lerobot-record via so101-record*) inside a pseudo-terminal so LeRobot's terminal
key listener is active, and forward control keys written to a FIFO:  n = end episode now, r = re-record, q = stop cleanly.
    python record_ctl.py <log> <fifo> -- <command...>
The FIFO is created if missing; the child's output goes to <log>; the log ends with "__RECORD_EXIT__ <code>"."""
import os, sys, pty, select, signal, time, errno
log_path, fifo = sys.argv[1], sys.argv[2]; cmd = sys.argv[sys.argv.index("--") + 1:]
if not os.path.exists(fifo): os.mkfifo(fifo)
pid, master = pty.fork()
if pid == 0:
    os.execvp(cmd[0], cmd)   # child: stdin/stdout are the pty slave -> TerminalKeyListener engages
log = open(log_path, "ab", buffering=0); ctl = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)   # O_RDWR: a FIFO with no writer reads EOF forever and select() spins at 100% CPU
def forward(sig, _):
    try: os.kill(pid, sig)
    except Exception: pass
signal.signal(signal.SIGTERM, forward); signal.signal(signal.SIGINT, forward)
alive = True; tail = b""
while alive:
    r, _, _ = select.select([master, ctl], [], [], 0.5)
    if master in r:
        try: data = os.read(master, 65536)
        except OSError as e:
            if e.errno == errno.EIO: data = b""
            else: raise
        if not data: alive = False
        else:
            log.write(data)
            # lerobot stops at an input() when the motors' stored calibration differs from the calibration file. Nobody can
            # type into this pty from the dashboard, so answer ENTER = "use the calibration file" (seen 7 Nov 2026, ll_follower).
            # why: the prompt can straddle two reads, so match against the unfinished line carried over from the previous read.
            if b"Press ENTER to use provided calibration" in tail + data: os.write(master, b"\n"); log.write(b"\n__CTL__ auto-ENTER: calibration mismatch, applying the calibration file\n"); tail = b""
            else: tail = (tail + data).rsplit(b"\n", 1)[-1][-256:]
    if ctl in r:
        try: keys = os.read(ctl, 64)
        except OSError: keys = b""
        for k in keys.decode(errors="ignore"):
            if k in "nrqp": os.write(master, k.encode()); log.write(f"\n__CTL__ {k}\n".encode())
    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid: alive = False; code = os.waitstatus_to_exitcode(status); break
    except ChildProcessError: alive = False; code = -1; break
else:   # why: pty EOF usually lands before the child is reapable, so WNOHANG missed it and the log said -1 for a clean exit
    try: code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
    except ChildProcessError: code = -1
time.sleep(0.2)
try: log.write(os.read(master, 65536))
except OSError: pass
log.write(f"\n__RECORD_EXIT__ {code}\n".encode()); log.close()
