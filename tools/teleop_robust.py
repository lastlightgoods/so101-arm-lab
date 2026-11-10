"""Leader->follower teleop that survives a stalled serial bus by reopening it.
Ctrl-C to stop (follower torque is released on exit)."""
import time, sys
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
FOLLOWER = "/dev/tty.usbmodem5C821070341"; LEADER = "/dev/tty.usbmodem5C821071281"
HZ = 60
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
robot  = SO101Follower(SO101FollowerConfig(port=FOLLOWER, id="ll_follower"))
leader = SO101Leader(SO101LeaderConfig(port=LEADER, id="ll_leader"))
def reopen(dev, name):
    """Close the dead port hard, then reconnect until it answers."""
    try: dev.bus.port_handler.closePort()
    except Exception: pass
    for attempt in range(1, 200):
        try:
            dev.connect(); log(f"{name} bus reopened (attempt {attempt})"); return
        except Exception as e:
            try: dev.bus.port_handler.closePort()
            except Exception: pass
            if attempt % 10 == 1: log(f"{name} reconnect failed: {str(e)[:80]}")
            time.sleep(0.3)
    log(f"{name}: giving up"); sys.exit(1)
robot.connect(); leader.connect(); log("teleop running at", HZ, "Hz. Ctrl-C to stop.")
stalls = 0; n = 0; t_last = time.time()
try:
    while True:
        t0 = time.perf_counter()
        try: action = leader.get_action()
        except Exception as e:
            stalls += 1; log("LEADER bus stalled:", str(e)[:70]); reopen(leader, "leader"); continue
        try: robot.send_action(action)
        except Exception as e:
            stalls += 1; log("FOLLOWER bus stalled:", str(e)[:70]); reopen(robot, "follower"); continue
        n += 1
        if time.time() - t_last > 30:
            log(f"ok: {n} frames, {stalls} stalls so far"); t_last = time.time()
        dt = 1 / HZ - (time.perf_counter() - t0)
        if dt > 0: time.sleep(dt)
except KeyboardInterrupt:
    log("stopping")
finally:
    for dev, name in ((robot, "follower"), (leader, "leader")):
        try: dev.disconnect()
        except Exception as e: log(f"{name} disconnect: {str(e)[:60]} (power-cycle releases torque)")
