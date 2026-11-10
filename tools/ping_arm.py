"""Ping ids 1-6 on every usbmodem port without LeRobot's broadcast ping."""
import glob, serial, sys, time
def pkt(i, instr, params=b""):
    body = bytes([i, len(params)+2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
for p in sys.argv[1:] or sorted(glob.glob("/dev/tty.usbmodem*")):
    try: s = serial.Serial(p, 1000000, timeout=0.15)
    except Exception as e: print(p, "open failed:", e); continue
    found = {}
    for i in range(1, 7):
        s.reset_input_buffer(); s.write(pkt(i, 0x02, bytes([0x38, 2]))); time.sleep(0.03); r = s.read(16)
        if len(r) >= 8: found[i] = int.from_bytes(r[5:7], "little")
    s.close()
    print(p, "->", found if found else "no motors")
