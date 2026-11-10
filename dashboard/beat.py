"""Write a simple kick/hat/snare loop as beat_<bpm>.wav (16-bit mono 44.1 kHz).  python beat.py 100 32"""
import sys, math, struct, wave, random
bpm = float(sys.argv[1]) if len(sys.argv) > 1 else 100; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 32; sr = 44100
beat = 60 / bpm; n = int(secs * sr); out = [0.0] * n
def add(start, dur, f):
    s0 = int(start * sr)
    for i in range(int(dur * sr)):
        if s0 + i < n: out[s0 + i] += f(i / sr)
random.seed(1)
t = 0.0; k = 0
while t < secs:
    add(t, 0.25, lambda x: 0.9 * math.sin(2 * math.pi * (55 + 90 * math.exp(-x * 30)) * x) * math.exp(-x * 9))          # kick every beat
    if k % 2 == 1: add(t, 0.18, lambda x: 0.5 * (random.random() * 2 - 1) * math.exp(-x * 18))                             # snare on 2 and 4
    for h in (0, 0.5): add(t + h * beat, 0.05, lambda x: 0.22 * (random.random() * 2 - 1) * math.exp(-x * 60))            # hats on 8ths
    t += beat; k += 1
peak = max(abs(v) for v in out) or 1
with wave.open(f"beat_{int(bpm)}.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(b"".join(struct.pack("<h", int(32000 * v / peak)) for v in out))
print(f"beat_{int(bpm)}.wav")
