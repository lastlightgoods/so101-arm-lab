"""The Mission Control skin's joint strip chart: it autoscales each lane so a few degrees of sag are visible, and it says so when no joint reading arrives.
Why the old chart looked dead (docs/telemetry.md): fixed +-120 deg lanes made a 3 deg move 1 px tall, and while the arm was busy (pick_seq holds .arms_busy and
published no state file) every lane was empty with no message."""
import time

import apidata as fx
from uihelpers import FIRST_POLLS, Page

REST = {"shoulder_pan": -0.04, "shoulder_lift": -100.09, "elbow_flex": 94.99, "wrist_flex": 72.0, "wrist_roll": 0.4, "gripper": 0.54}
def arms(pos):
    return {"follower": {"t": time.time(), "role": "follower", "mode": "deg", "src": "idle-poll", "pos": pos, "rx": time.time()}}


def open_chart(chrome, stub, arms_obj):
    s = fx.state()
    s["arms"] = arms_obj
    stub.routes["/api/state"] = s
    p = Page(chrome, stub.url + "/?skin=missioncontrol", hide="#nothing")
    stub.wait_hits(FIRST_POLLS)
    return p, s


def test_three_degrees_of_sag_is_visible_not_a_flat_line(chrome, stub):
    p, s = open_chart(chrome, stub, arms(REST))
    try:
        time.sleep(1.2)
        for i in range(40):   # hold, then sag 3 deg, then hold: the pen must visibly step
            pos = dict(REST)
            pos["shoulder_lift"] += 3.0 if i >= 20 else 0.0
            s["arms"] = arms(pos)
            time.sleep(0.12)
        lo, hi = p.eval("window.so101StripRange(1)")   # the LIFT lane
        assert hi - lo <= 24, f"the LIFT lane still spans {hi - lo:.0f} deg after a 3 deg sag; a 3 deg step needs a narrow lane to be visible (old fixed scale: 240 deg, about 1 px)"
        assert lo <= REST["shoulder_lift"] and hi >= REST["shoulder_lift"] + 3.0, "the lane range must contain the pen"
        assert p.errors()["exceptions"] == []
    finally:
        p.close()


def test_no_joint_feed_says_so(chrome, stub):
    p, s = open_chart(chrome, stub, {"follower": {"t": time.time(), "role": "follower", "src": "busy", "pos": {}, "rx": time.time()}})
    try:
        time.sleep(3.2)
        amber = p.eval("""(()=>{const c=document.querySelector('#gallery canvas'),x=c.getContext('2d'),W=c.width,H=c.height,d=x.getImageData(Math.round(W*.15),Math.round(H*.12),Math.round(W*.7),Math.round(H*.8)).data;
let n=0;for(let i=0;i<d.length;i+=4){if(Math.abs(d[i]-226)<30&&Math.abs(d[i+1]-169)<30&&Math.abs(d[i+2]-61)<30)n++}return n})()""")
        assert amber > 150, "the NO JOINT FEED banner (amber text on the paper) was not drawn"
        assert p.errors()["exceptions"] == []
    finally:
        p.close()
