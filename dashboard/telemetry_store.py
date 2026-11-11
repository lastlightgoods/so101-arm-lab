"""telemetry_store.py - the dashboard's memory of servo telemetry (docs/telemetry.md).

server.py feeds it every telemetry sample the arm stream delivers (jetson/arm_state.py puts it on the follower's line as "tele"); the page reads it
back at /api/tele?since=<t>. It keeps the last WINDOW_S seconds of rows, works out which flags are up (hot servo, brown-out, bus over the board's
rating, servo error bits, a joint sagging more than it usually does) and records each flag the moment it first goes up, so the page can list
"error flags as they happen" even when nobody was looking. Pure Python, no dashboard imports: the tests drive it directly.
"""
import collections, importlib.util, os, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW_S = 600           # ten minutes of rows (5 Hz idle, 2-4 Hz while the arm works)
MAX_EVENTS = 100
KEYS = ("temp", "load", "cur", "volt", "pos", "goal", "sag", "vel", "on", "moving", "status")   # per-joint order inside a row


def load_servo_telemetry():
    """jetson/servo_telemetry.py by path (the same file the Jetson runs): the thresholds and the sag rule live there once."""
    spec = importlib.util.spec_from_file_location("servo_telemetry_for_dash", os.path.join(os.path.dirname(HERE), "jetson", "servo_telemetry.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class TeleStore:
    def __init__(self, st=None, window_s=WINDOW_S, clock=time.time):
        self.st = st or load_servo_telemetry(); self.window_s = window_s; self.clock = clock; self.lock = threading.Lock()
        self.rows = collections.deque(); self.events = collections.deque(maxlen=MAX_EVENTS); self.limits = {}; self.last = None; self.src = None; self.rx = 0.0
        self.sag_hist = {n: collections.deque(maxlen=900) for n in self.st.JOINTS}   # |sag| while holding, per joint
        self.active = {}      # (joint, kind) -> level currently up; an event is written when a key appears or its level rises
        self.n = 0

    # -- feeding ---------------------------------------------------------------------------------------
    def add(self, sample, src=None, at=None):
        """Store one sample ({"j": {joint: {...}}, "total_a", "volt_min", ...}); returns the events it raised (so the server can log them).
        Rows are stamped with this machine's clock (`at`, default now), not the Jetson's, so a skewed Jetson clock cannot shift the charts."""
        st = self.st; t = float(self.clock() if at is None else at); j = sample.get("j") if isinstance(sample, dict) else None
        j = {n: d for n, d in j.items() if isinstance(d, dict)} if isinstance(j, dict) else {}
        if not j: return []   # a malformed line from the stream is dropped whole, never half stored
        with self.lock:
            row = [round(t, 2), sample.get("total_a"), sample.get("volt_min"), [self._cells(j.get(n)) for n in st.JOINTS]]   # first: a bad value raises before anything is stored
            if isinstance(sample.get("limits"), dict) and sample["limits"]: self.limits = sample["limits"]
            self.src = src or sample.get("src") or self.src; self.rx = self.clock(); self.n += 1
            self.rows.append(row); cut = t - self.window_s
            while self.rows and self.rows[0][0] < cut: self.rows.popleft()
            self.last = {**sample, "j": j}
            for n, d in j.items():
                if d.get("sag") is not None and d.get("on") and not d.get("moving"): self.sag_hist[n].append(d["sag"])
            return self._flags(t, self.last)

    @staticmethod
    def _cells(d):
        if not d: return None
        r = lambda v, k=1: None if v is None else round(v, k)
        return [d.get("temp_c"), r(d.get("load_pct")), r(d.get("cur_a"), 2), r(d.get("volt_v")), r(d.get("pos")), r(d.get("goal")), r(d.get("sag"), 2), r(d.get("vel_dps")),
                d.get("on"), d.get("moving"), d.get("status")]

    # -- flags -----------------------------------------------------------------------------------------
    def current_flags(self, sample=None):
        """[{"joint", "kind", "level" (1 amber, 2 red), "text"}] for the newest sample (or `sample`)."""
        st = self.st; s = sample or self.last
        if not s: return []
        out = []; j = s.get("j") or {}
        for n, d in j.items():
            cut = (self.limits.get(n) or {}).get("temp_cut_c") or st.TEMP_CUT_C
            lv = st.level_temp(d.get("temp_c"), cut)
            if lv: out.append({"joint": n, "kind": "hot", "level": lv, "text": f"{d['temp_c']} C, cut at {cut} C"})
            for e in d.get("err") or []: out.append({"joint": n, "kind": "err:" + e, "level": 2, "text": f"servo error bit: {e}"})
            if st.sag_flag(list(self.sag_hist[n])): out.append({"joint": n, "kind": "sag", "level": 1, "text": "sag above its usual level"})
        v = s.get("volt_min"); lv = st.level_volt(v)
        if lv: out.append({"joint": "bus", "kind": "volt", "level": lv, "text": f"supply {v:.1f} V ({'brown-out' if lv > 1 else 'low'}; warn under {st.VOLT_WARN_V:g}, brown-out under {st.VOLT_LOW_V:g})"})
        lv = st.level_bus(s.get("total_a"))
        if lv: out.append({"joint": "bus", "kind": "bus", "level": lv, "text": f"{s['total_a']:.1f} A of the {st.BUS_A_MARK:g} A board rating"})
        return out

    def _flags(self, t, sample):
        now_up = {(f["joint"], f["kind"]): f for f in self.current_flags(sample)}; new = []
        for k, f in now_up.items():
            if self.active.get(k, 0) < f["level"]:
                ev = {"t": round(t, 2), "joint": f["joint"], "kind": f["kind"], "level": f["level"], "text": f["text"]}; self.events.append(ev); new.append(ev)
        self.active = {k: f["level"] for k, f in now_up.items()}
        return new

    # -- serving ---------------------------------------------------------------------------------------
    def api(self, since=0.0):
        with self.lock:
            rows = [r for r in self.rows if r[0] > since]; last = self.last
            return {"keys": list(KEYS), "joints": list(self.st.JOINTS), "rows": rows, "limits": self.limits, "src": self.src, "age": round(self.clock() - self.rx, 2) if self.rx else None,
                    "n": self.n, "now": round(self.clock(), 2), "t": self.rows[-1][0] if self.rows else None, "flags": self.current_flags(), "events": list(self.events)[-30:], "dt_ms": last.get("dt_ms") if last else None,
                    "marks": {"temp_cut": self.st.TEMP_CUT_C, "temp_warn": self.st.TEMP_WARN_C, "temp_hot": self.st.TEMP_HOT_C, "volt_warn": self.st.VOLT_WARN_V,
                              "volt_low": self.st.VOLT_LOW_V, "bus_a": self.st.BUS_A_MARK}}
