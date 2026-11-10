"""Stand-in for lerobot-record, driven by record_ctl.py inside a pty. Mode is argv[1]:
    exit <n>     print a line, exit with code n
    killed       print a line, then SIGKILL itself (killed-by-signal exit)
    keys         cbreak mode (like lerobot's key listener): echo each stdin char as "GOT <c>", exit on q
    prompt       print the calibration prompt, block on input(), then print GOT_ENTER
    split-prompt same, but the prompt is written in two flushes 0.3 s apart; NO_ENTER if nothing arrives in 2 s
    wait         sleep until SIGTERM/SIGINT; print which signal arrived and exit 0"""
import os, select, signal, sys, time

PROMPT = "Press ENTER to use provided calibration file associated with the id ll_follower, or type 'c' and press ENTER to run calibration: "


def say(s): sys.stdout.write(s + "\n"); sys.stdout.flush()


def await_enter(timeout):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r and sys.stdin.readline(): say("GOT_ENTER")
    else: say("NO_ENTER"); sys.exit(3)


mode = sys.argv[1]
if mode == "exit":
    say("fake lerobot-record starting"); sys.exit(int(sys.argv[2]))
elif mode == "killed":
    say("about to die"); os.kill(os.getpid(), signal.SIGKILL); time.sleep(5)
elif mode == "keys":
    import tty
    tty.setcbreak(sys.stdin.fileno()); say("READY")
    while True:
        c = sys.stdin.read(1)
        if not c: sys.exit(2)
        say(f"GOT {c}")
        if c == "q": sys.exit(0)
elif mode == "prompt":
    sys.stdout.write(PROMPT); sys.stdout.flush(); await_enter(5)
elif mode == "split-prompt":
    sys.stdout.write(PROMPT[:22]); sys.stdout.flush(); time.sleep(0.3)
    sys.stdout.write(PROMPT[22:]); sys.stdout.flush(); await_enter(2)
elif mode == "wait":
    def h(sig, _): say(f"CHILD_GOT_{signal.Signals(sig).name}"); sys.exit(0)
    signal.signal(signal.SIGTERM, h); signal.signal(signal.SIGINT, h); say("WAITING")
    while True: time.sleep(0.05)
