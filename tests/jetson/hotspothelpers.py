"""Fixtures for jetson/hotspot.sh: a fake nmcli / iw / ip / ping / ssh / systemd-run / systemctl (and optionally qrencode,
iperf3, nft) on PATH, each a tiny sh script that appends its argv to a log and answers from canned files in a temp dir.
The canned `iw list` is the real shape of the lab Jetson's RTL8822CE output (10 Nov 2026 survey): "* AP" among the
interface modes, "interface combinations are not supported", "Device supports SAE". Nothing here touches a network."""
import os, stat, subprocess, textwrap
import pytest
from jhelpers import JETSON

HOTSPOT = os.path.join(JETSON, "hotspot.sh")
DISPATCHER = os.path.join(JETSON, "90-so101-hotspot.dispatcher")
NFT_FILE = os.path.join(JETSON, "so101-hotspot.nft")
SEP = "\x1f"   # argv separator in the call log; no argument the script passes contains it

IW_LIST = textwrap.dedent("""\
    Wiphy phy0
    \twiphy index: 0
    \tmax # scan SSIDs: 9
    \tSupported Ciphers:
    \t\t* WEP40 (00-0f-ac:1)
    \t\t* CCMP-128 (00-0f-ac:4)
    \tAvailable Antennas: TX 0 RX 0
    \tSupported interface modes:
    \t\t * IBSS
    \t\t * managed
    \t\t * AP
    \t\t * P2P-client
    \t\t * P2P-GO
    \tBand 1:
    \t\tFrequencies:
    \t\t\t* 2412 MHz [1] (20.0 dBm)
    \t\t\t* 2437 MHz [6] (20.0 dBm)
    \t\t\t* 2462 MHz [11] (20.0 dBm)
    \t\t\t* 2467 MHz [12] (20.0 dBm) (no IR)
    \tBand 2:
    \t\tFrequencies:
    \t\t\t* 5180 MHz [36] (23.0 dBm)
    \t\t\t* 5200 MHz [40] (23.0 dBm)
    \t\t\t* 5220 MHz [44] (23.0 dBm)
    \t\t\t* 5240 MHz [48] (23.0 dBm)
    \t\t\t* 5260 MHz [52] (23.0 dBm) (no IR, radar detection)
    \t\t\t* 5745 MHz [149] (30.0 dBm)
    \tinterface combinations are not supported
    \tDevice supports SAE with AUTHENTICATE command
    \tDevice supports scan flush.
    """)
IW_INFO_AP = "Interface wlP1p1s0\n\tifindex 4\n\tssid so101-lab\n\ttype AP\n\twiphy 0\n\tchannel 36 (5180 MHz), width: 80 MHz, center1: 5210 MHz\n\ttxpower 23.00 dBm\n"
IW_INFO_CLIENT = "Interface wlP1p1s0\n\tifindex 4\n\tssid super-net\n\ttype managed\n\twiphy 0\n\tchannel 48 (5240 MHz), width: 80 MHz, center1: 5210 MHz\n"
IW_STATIONS = "Station aa:bb:cc:dd:ee:01 (on wlP1p1s0)\n\tinactive time:\t120 ms\nStation aa:bb:cc:dd:ee:02 (on wlP1p1s0)\n\tinactive time:\t40 ms\n"
LINK_UP = "2: enP8p1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000\\    link/ether 00:00:00:00:00:00 brd ff:ff:ff:ff:ff:ff\n"
LINK_NO_CARRIER = "2: enP8p1s0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN mode DEFAULT group default qlen 1000\\    link/ether 00:00:00:00:00:00 brd ff:ff:ff:ff:ff:ff\n"
WIFI_LINK = "4: wlP1p1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DORMANT group default qlen 1000\\    link/ether 00:00:00:00:00:01 brd ff:ff:ff:ff:ff:ff\n"
# why 192.0.2.x: RFC 5737 documentation addresses, never a real LAN address of the lab
ETH_ADDR = "2: enP8p1s0    inet 192.0.2.20/24 brd 192.0.2.255 scope global dynamic noprefixroute enP8p1s0\\       valid_lft 86000sec preferred_lft 86000sec\n"
WIFI_ADDR_AP = "4: wlP1p1s0    inet 10.101.0.1/24 brd 10.101.0.255 scope global noprefixroute wlP1p1s0\\       valid_lft forever preferred_lft forever\n"
ROUTE_ETH = "default via 192.0.2.1 dev enP8p1s0 proto dhcp src 192.0.2.20 metric 100 \ndefault via 192.0.2.1 dev wlP1p1s0 proto dhcp src 192.0.2.21 metric 600 \n"
ROUTE_WIFI_ONLY = "default via 192.0.2.1 dev wlP1p1s0 proto dhcp src 192.0.2.21 metric 600 \n"
TIMERS_PENDING = "NEXT                        LEFT      LAST PASSED UNIT                        ACTIVATES\nMon 2026-11-10 14:10:00 PST 9min left n/a  n/a    so101-hotspot-revert.timer  so101-hotspot-revert.service\n\n1 timers listed.\n"
TIMERS_NONE = "NEXT LEFT LAST PASSED UNIT ACTIVATES\n\n0 timers listed.\n"

PROLOGUE = '#!/bin/sh\n# fake {name}: log argv, answer from $FAKE_DIR\nprintf "%s{sep}" "{name}" "$@" >> "$FAKE_LOG"; printf "\\n" >> "$FAKE_LOG"\nrc() {{ [ -f "$FAKE_DIR/$1" ] && exit "$(cat "$FAKE_DIR/$1")"; exit 0; }}\n'
BODIES = {
    "nmcli": 'case "$*" in\n  "-t -f RUNNING general") echo running;;\n  "-t -f NAME con show --active") cat "$FAKE_DIR/nmcli_active";;\n  "-t -f NAME,DEVICE con show --active") cat "$FAKE_DIR/nmcli_active_dev";;\n'
             '  "-t -f NAME con show") cat "$FAKE_DIR/nmcli_cons";;\n  *"con up "*) rc nmcli_up_rc;;\n  *) rc nmcli_rc;;\nesac\n',
    "iw": 'case "$*" in\n  list) cat "$FAKE_DIR/iw_list";;\n  "dev "*" info") cat "$FAKE_DIR/iw_info";;\n  "dev "*" station dump") cat "$FAKE_DIR/iw_stations";;\nesac\n',
    "ip": 'last=""; for a in "$@"; do last="$a"; done\ncase "$*" in\n  *"link show dev"*) f="$FAKE_DIR/ip_link_$last";;\n  *"addr show dev"*) f="$FAKE_DIR/ip_addr_$last";;\n  *"route show default"*) f="$FAKE_DIR/ip_route";;\n  *) exit 1;;\nesac\n'
          '[ -f "$f" ] || exit 1\ncat "$f"\n',
    "ping": "rc ping_rc\n", "ssh": "rc ssh_rc\n", "systemd-run": "rc systemd_run_rc\n",
    "systemctl": 'case "$*" in\n  *list-timers*) cat "$FAKE_DIR/systemctl_timers";;\n  *) rc systemctl_rc;;\nesac\n',
    "qrencode": 'cat > "$FAKE_DIR/qrencode_stdin"; echo "[fake QR block]"\n',
    "iperf3": 'echo "[  5]   0.00-10.00  sec   112 MBytes  94.2 Mbits/sec                  sender"\n',
    "nft": "rc nft_rc\n", "logger": "exit 0\n",
}
REQUIRED = ("nmcli", "iw", "ip", "ping", "ssh", "systemd-run", "systemctl", "logger")


class Lab:
    """A fake $HOME/so101, a bin dir of fake tools and the canned answers; `run` executes hotspot.sh under bash."""
    def __init__(self, tmp):
        self.tmp = tmp; self.home = tmp / "home"; self.so = self.home / "so101"; self.so.mkdir(parents=True)
        self.bin = tmp / "bin"; self.bin.mkdir(); self.fake = tmp / "fake"; self.fake.mkdir()
        self.log = tmp / "calls.log"
        for name in REQUIRED: self.add_tool(name)
        self.canned("iw_list", IW_LIST); self.canned("iw_info", IW_INFO_CLIENT); self.canned("iw_stations", "")
        self.canned("ip_link_enP8p1s0", LINK_UP); self.canned("ip_link_wlP1p1s0", WIFI_LINK); self.canned("ip_addr_enP8p1s0", ETH_ADDR)
        self.canned("ip_addr_wlP1p1s0", ""); self.canned("ip_route", ROUTE_ETH)
        self.canned("nmcli_cons", "super-net\nWired connection 1\n"); self.canned("nmcli_active", "super-net\nWired connection 1\n")
        self.canned("nmcli_active_dev", "super-net:wlP1p1s0\nWired connection 1:enP8p1s0\n"); self.canned("systemctl_timers", TIMERS_NONE)
        self.psk = self.so / ".hotspot_psk"; self.ok = self.so / ".hotspot_ok"

    def add_tool(self, name):
        p = self.bin / name; p.write_text(PROLOGUE.format(name=name, sep=SEP) + BODIES[name]); p.chmod(0o755)
    def remove_tool(self, name): (self.bin / name).unlink()
    def canned(self, name, text): (self.fake / name).write_text(text)
    def rc(self, tool, code): self.canned(f"{tool.replace('-', '_')}_rc", str(code))

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SO101_")}
        env.update(HOME=str(self.home), PATH=f"{self.bin}:/usr/bin:/bin", FAKE_LOG=str(self.log), FAKE_DIR=str(self.fake), LC_ALL="C")
        env.update({k: str(v) for k, v in extra.items()}); return env

    def run(self, *argv, env=None, **extra):
        r = subprocess.run(["bash", HOTSPOT, *argv], env=env or self.env(**extra), capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout + r.stderr

    def dispatch(self, iface, action, **extra):
        """Run the committed dispatcher the way NetworkManager does: `sh <script> <iface> <action>`, as plain sh."""
        r = subprocess.run(["sh", DISPATCHER, iface, action], env=self.env(**extra), capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout + r.stderr

    def calls(self, tool=None):
        if not self.log.exists(): return []
        rows = [line.split(SEP)[:-1] for line in self.log.read_text().splitlines() if line]
        return [r for r in rows if tool is None or r[0] == tool]
    def clear(self):
        if self.log.exists(): self.log.unlink()

    def init_psk(self, key="Abcdefghij0123456789KLMN", mode=0o600):
        self.psk.write_text(key + "\n"); self.psk.chmod(mode); return key
    def mode(self, path): return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def pairs(argv):
    """nmcli `con modify NAME k v k v ...` -> {k: v}."""
    assert argv[:2] == ["nmcli", "con"] and argv[2] == "modify", argv
    rest = argv[4:]; assert len(rest) % 2 == 0, rest
    return dict(zip(rest[::2], rest[1::2]))
