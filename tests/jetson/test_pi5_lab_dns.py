"""jetson/pi5_lab_dns.sh (.lab names on the Pi's dnsmasq) with fake avahi-resolve-host-name, hostname, nmcli and systemctl
on PATH and a temp HOME / hostsdir. Runs under the Mac's bash 3.2. Addresses are from the documentation ranges (RFC 5737)."""
import os, subprocess
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "jetson" / "pi5_lab_dns.sh"


class Pi:
    def __init__(self, tmp):
        self.tmp, self.home, self.bin, self.hostsdir = tmp, tmp / "home", tmp / "bin", tmp / "hosts.d"
        for d in (self.home / "so101", self.bin):
            d.mkdir(parents=True)
        self.names = self.home / "so101" / ".lab_dns_names"
        self.names.write_text("# machine  names\nself pi dns\nmac-one.local mac review arm\n\njetson-x.local jetson\n")
        self.mdns({"mac-one.local": "192.0.2.70", "jetson-x.local": "192.0.2.60"})
        self.tool("avahi-resolve-host-name", 'h="$2"; ip=$(awk -v h="$h" \'$1 == h {print $2}\' "$FAKE/mdns"); [ -n "$ip" ] || { echo "Failed to resolve host name" >&2; exit 1; }; printf "%s\\t%s\\n" "$h" "$ip"')
        self.tool("hostname", '[ "$1" = -I ] && echo "192.0.2.61 fd00::61 " || echo tc-pi5')
        self.tool("nmcli", 'case "$*" in *DEVICE,STATE*) printf "lo:connected (externally)\\nwlan0:connected\\neth0:unavailable\\n" ;; '
                           '*"IP4.DNS dev show wlan0"*) printf "IP4.DNS[1]:203.0.113.8\\nIP4.DNS[2]:203.0.113.9\\n" ;; esac')
        self.tool("systemctl", 'echo "$*" >> "$FAKE/systemctl"; [ "$2" = is-active ] && echo active; exit 0')

    def tool(self, name, body):
        p = self.bin / name; p.write_text("#!/bin/bash\n" + body + "\n"); p.chmod(0o755)

    def mdns(self, table):
        (self.tmp / "mdns").write_text("".join(f"{h} {ip}\n" for h, ip in table.items()))

    def hosts(self):
        p = self.hostsdir / "lab.hosts"; return [l for l in p.read_text().splitlines() if not l.startswith("#")] if p.exists() else None

    def run(self, *args, **env):
        e = {**os.environ, "HOME": str(self.home), "PATH": f"{self.bin}:{os.environ['PATH']}", "TMPDIR": str(self.tmp), "USER": "tc",
             "FAKE": str(self.tmp), "SO101_DNS_HOSTSDIR": str(self.hostsdir), **env}
        if "SO101_DNS_UPSTREAM" not in env:
            e.pop("SO101_DNS_UPSTREAM", None)
        r = subprocess.run(["/bin/bash", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=30)
        return r.returncode, r.stdout + r.stderr


@pytest.fixture
def pi(tmp_path):
    return Pi(tmp_path)


def test_refresh_writes_every_mapped_name_with_the_lab_suffix(pi):
    rc, out = pi.run("refresh")
    assert rc == 0, out
    assert pi.hosts() == ["192.0.2.61 pi.lab dns.lab", "192.0.2.70 mac.lab review.lab arm.lab", "192.0.2.60 jetson.lab"]
    assert oct((pi.hostsdir / "lab.hosts").stat().st_mode & 0o777) == "0o644", "dnsmasq runs as its own user and must read it"
    assert "hosts: 192.0.2.61 pi.lab dns.lab;" in (pi.home / "so101" / "lab_dns.log").read_text()


def test_an_unchanged_map_does_not_rewrite_the_file(pi):
    pi.run("refresh"); mtime = (pi.hostsdir / "lab.hosts").stat().st_mtime_ns; log = (pi.home / "so101" / "lab_dns.log").read_text()
    rc, out = pi.run("refresh")
    assert rc == 0 and out.startswith("unchanged:") and (pi.hostsdir / "lab.hosts").stat().st_mtime_ns == mtime
    assert (pi.home / "so101" / "lab_dns.log").read_text() == log, "one log line per change, not per minute"


def test_a_dhcp_move_is_picked_up(pi):
    pi.run("refresh"); pi.mdns({"mac-one.local": "192.0.2.99", "jetson-x.local": "192.0.2.60"})
    rc, out = pi.run("refresh")
    assert rc == 0 and "192.0.2.99 mac.lab review.lab arm.lab" in pi.hosts()


def test_a_machine_that_stops_answering_keeps_its_last_address(pi):
    pi.run("refresh"); pi.mdns({"jetson-x.local": "192.0.2.60"})   # the Mac is asleep
    rc, out = pi.run("refresh")
    assert rc == 0 and "mac-one.local: no mDNS answer, keeping 192.0.2.70" in out
    assert "192.0.2.70 mac.lab review.lab arm.lab" in pi.hosts()


def test_a_machine_never_seen_is_skipped_not_invented(pi):
    pi.mdns({"jetson-x.local": "192.0.2.60"})
    rc, out = pi.run("refresh")
    assert rc == 0 and "no earlier address, skipped" in out
    assert pi.hosts() == ["192.0.2.61 pi.lab dns.lab", "192.0.2.60 jetson.lab"]


def test_dry_run_writes_nothing(pi):
    rc, out = pi.run("refresh", "--dry-run")
    assert rc == 0 and "dry-run: would write" in out and "192.0.2.70 mac.lab" in out
    assert pi.hosts() is None and not (pi.home / "so101" / "lab_dns.log").exists()


def test_a_missing_name_map_is_an_error(pi):
    pi.names.unlink(); rc, out = pi.run("refresh")
    assert rc == 1 and "no name map" in out


def test_conf_forwards_to_the_resolvers_the_router_gave_and_serves_lab_locally(pi):
    rc, out = pi.run("conf")
    lines = out.splitlines()
    assert rc == 0, out
    assert "server=203.0.113.8" in lines and "server=203.0.113.9" in lines, "same upstream as before: the household's DNS provider does not change"
    assert "local=/lab/" in lines and f"hostsdir={pi.hostsdir}" in lines and "no-resolv" in lines and "domain-needed" in lines and "bogus-priv" in lines
    assert "interface=wlan0" in lines and "interface=eth0" in lines and "bind-dynamic" in lines


def test_conf_takes_an_explicit_upstream_and_refuses_to_write_without_one(pi):
    rc, out = pi.run("conf", SO101_DNS_UPSTREAM="198.51.100.1 198.51.100.2")
    assert rc == 0 and "server=198.51.100.1" in out and "203.0.113.8" not in out
    pi.tool("nmcli", "exit 0")
    rc, out = pi.run("conf")
    assert rc == 1 and "no upstream resolvers" in out


def test_setup_only_prints_the_sudo_lines(pi):
    rc, out = pi.run("setup")
    assert rc == 0 and "sudo apt-get install -y dnsmasq avahi-utils" in out and "sudo tee /etc/dnsmasq.d/so101-lab.conf" in out
    assert not pi.hostsdir.exists() and not (pi.tmp / "systemctl").exists()


def test_install_enables_the_minute_timer(pi):
    rc, out = pi.run("install")
    d = pi.home / ".config/systemd/user"
    assert rc == 0 and "OnUnitActiveSec=60s" in (d / "so101-lab-dns.timer").read_text()
    assert f"ExecStart=/bin/bash {pi.home}/so101/pi5_lab_dns.sh refresh" in (d / "so101-lab-dns.service").read_text()
    assert "--user enable --now so101-lab-dns.timer" in (pi.tmp / "systemctl").read_text()


def test_conf_never_forwards_to_the_pi_itself_after_the_router_points_at_it(pi):
    pi.tool("nmcli", 'case "$*" in *DEVICE,STATE*) printf "wlan0:connected\\n" ;; '
                     '*"IP4.DNS dev show wlan0"*) printf "IP4.DNS[1]:192.0.2.61\\nIP4.DNS[2]:203.0.113.8\\n" ;; esac')
    rc, out = pi.run("conf")
    assert rc == 0 and "server=203.0.113.8" in out and "server=192.0.2.61" not in out, out
    pi.tool("nmcli", 'case "$*" in *DEVICE,STATE*) printf "wlan0:connected\\n" ;; *"IP4.DNS dev show wlan0"*) printf "IP4.DNS[1]:192.0.2.61\\n" ;; esac')
    rc, out = pi.run("conf")
    assert rc == 1 and "no upstream resolvers" in out, "only itself left: refuse rather than loop"
