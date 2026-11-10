"""jetson/hotspot.sh (the Jetson's radio as the lab hotspot) against fake nmcli/iw/ip/ping/ssh/systemd-run/systemctl on PATH.
The script runs under the Mac's bash 3.2, which is the compatibility floor it promises. Also: the committed dispatcher's up/down
branches under plain sh, the generated nftables file, and the so101-hotspot function in jetson/env.sh. No network is touched."""
import os, re, subprocess
import pytest
from hotspothelpers import (DISPATCHER, HOTSPOT, IW_INFO_AP, IW_LIST, IW_STATIONS, LINK_NO_CARRIER, NFT_FILE, ROUTE_WIFI_ONLY, TIMERS_PENDING,
                            WIFI_ADDR_AP, lab, pairs)   # `lab` is the fixture: importing it into this module registers it
from jhelpers import JETSON


# ---------- preflight ----------
def test_preflight_passes_on_the_surveyed_box(lab):
    lab.init_psk()
    rc, out = lab.run("preflight", "--mac", "mac.example", "--mac-ssh", "tc@mac.example")
    assert rc == 0, out
    for needle in ("enP8p1s0 has carrier", "has IPv4 192.0.2.20/24", "answers ping", "ssh tc@mac.example works", "NetworkManager is running",
                   "radio supports AP mode", "radio wlP1p1s0 exists", "passphrase file mode 600", "passphrase is 24 chars", "all good"):
        assert needle in out, (needle, out)
    assert "no client+AP at once" in out, "the 'interface combinations are not supported' line is explained"
    assert "band a channel 36, key-mgmt sae" in out
    assert "FAIL" not in out
    assert ["ping", "-c", "3", "-W", "2", "mac.example"] in lab.calls("ping")
    assert ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "tc@mac.example", "true"] in lab.calls("ssh")
    assert lab.calls("systemd-run") == [] and not any(c[1:3] == ["con", "modify"] for c in lab.calls("nmcli")), "preflight is read-only"


def test_preflight_without_mac_requires_the_default_route_via_ethernet(lab):
    lab.init_psk()
    rc, out = lab.run("preflight"); assert rc == 0 and "default route goes via enP8p1s0" in out
    lab.canned("ip_route", ROUTE_WIFI_ONLY)
    rc, out = lab.run("preflight"); assert rc == 1 and "FAIL  default route is not via enP8p1s0" in out and "--mac HOST" in out


@pytest.mark.parametrize("setup, needle", [
    (lambda lab: lab.canned("ip_link_enP8p1s0", LINK_NO_CARRIER), "no carrier"),
    (lambda lab: lab.canned("ip_addr_enP8p1s0", ""), "no IPv4 address"),
    (lambda lab: lab.rc("ping", 1), "does not answer ping"),
    (lambda lab: lab.rc("ssh", 255), "BatchMode=yes tc@mac.example failed"),
    (lambda lab: lab.canned("iw_list", IW_LIST.replace("\t\t * AP\n", "")), "cannot be a hotspot"),
    (lambda lab: lab.remove_tool("iw"), "iw not found"),
])
def test_preflight_fails_on_each_uplink_or_radio_problem(lab, setup, needle):
    lab.init_psk(); setup(lab)
    rc, out = lab.run("preflight", "--mac", "mac.example", "--mac-ssh", "tc@mac.example")
    assert rc == 1 and "FAIL" in out and needle in out, out
    assert "preflight: 1 problem(s)" in out


def test_preflight_fails_when_the_mac_answers_but_ethernet_has_no_carrier(lab):
    """Ping success does not excuse a missing cable: the hotspot's uplink is the cable, not the Wi-Fi it is about to drop."""
    lab.init_psk(); lab.canned("ip_link_enP8p1s0", LINK_NO_CARRIER)
    rc, out = lab.run("preflight", "--mac", "mac.example")
    assert rc == 1 and "FAIL  enP8p1s0 has no carrier" in out and "answers ping" in out


@pytest.mark.parametrize("setup, needle", [
    (lambda lab: None, "passphrase file .* missing"),
    (lambda lab: lab.init_psk("tooshort12"), "passphrase is 10 chars, want 20\\+"),
    (lambda lab: lab.init_psk(mode=0o644), "passphrase file mode is 644, want 600"),
])
def test_preflight_fails_on_a_bad_passphrase_file(lab, setup, needle):
    setup(lab)
    rc, out = lab.run("preflight")
    assert rc == 1 and re.search("FAIL  " + needle, out), out


def test_preflight_refuses_a_passphrase_inside_a_checkout_unless_git_ignored(lab):
    """On the Mac ~/so101 is the repo: the key file there must be in .gitignore or preflight fails."""
    lab.init_psk(); (lab.so / ".git").mkdir(); (lab.so / ".gitignore").write_text("*.log\n")
    rc, out = lab.run("preflight"); assert rc == 1 and "sits in the checkout" in out and ".gitignore does not list .hotspot_psk" in out
    (lab.so / ".gitignore").write_text("*.log\n.hotspot_psk\n.hotspot_ok\n")
    rc, out = lab.run("preflight"); assert rc == 0 and "git-ignored" in out


def test_repo_gitignore_lists_the_hotspot_files():
    ignored = open(os.path.join(os.path.dirname(JETSON), ".gitignore")).read().splitlines()
    assert ".hotspot_psk" in ignored and ".hotspot_ok" in ignored


# ---------- init / qr ----------
def test_init_writes_a_24_char_psk_with_mode_600_and_never_prints_it(lab):
    rc, out = lab.run("init"); assert rc == 0, out
    key = lab.psk.read_text().strip()
    assert re.fullmatch(r"[A-Za-z0-9]{24}", key) and lab.mode(lab.psk) == 0o600
    assert key not in out and "never printed" in out
    rc, out = lab.run("init"); assert rc == 0 and "keeping it" in out and lab.psk.read_text().strip() == key, "no silent rotation"
    rc, out = lab.run("init", "--force"); assert rc == 0 and lab.psk.read_text().strip() != key and lab.mode(lab.psk) == 0o600


def test_init_dry_run_writes_nothing(lab):
    rc, out = lab.run("init", "--dry-run"); assert rc == 0 and "dry-run" in out and not lab.psk.exists()


def test_qr_pipes_the_wifi_string_to_qrencode_without_printing_the_key(lab):
    key = lab.init_psk(); lab.add_tool("qrencode")
    rc, out = lab.run("qr"); assert rc == 0, out
    assert (lab.fake / "qrencode_stdin").read_text() == f"WIFI:T:WPA;S:so101-lab;P:{key};;"
    assert lab.calls("qrencode") == [["qrencode", "-t", "ansiutf8"]] and "[fake QR block]" in out and key not in out


def test_qr_without_qrencode_points_at_the_file(lab):
    key = lab.init_psk()
    rc, out = lab.run("qr"); assert rc == 0 and "SSID so101-lab" in out and str(lab.psk) in out and "qrencode is not installed" in out and key not in out
    rc, out = lab.run("qr", env=lab.env(SO101_HOTSPOT_SSID="bench-7")); assert "SSID bench-7" in out


# ---------- plan: channel and key management from iw list ----------
def test_plan_prefers_channel_36_and_sae(lab):
    rc, out = lab.run("plan"); assert rc == 0 and "band a channel 36 key-mgmt sae" in out and "ssid so101-lab" in out


@pytest.mark.parametrize("edit, want", [
    (lambda s: s.replace("[36] (23.0 dBm)", "[36] (23.0 dBm) (no IR)"), "band a channel 40"),
    (lambda s: s.replace("[36] (23.0 dBm)", "[36] (disabled)").replace("[40] (23.0 dBm)", "[40] (disabled)"), "band a channel 44"),
    (lambda s: re.sub(r"\((23\.0 dBm)\)", r"(\1) (no IR, radar detection)", s), "band bg channel 6"),
    (lambda s: s.replace("5180 MHz", "5180.0 MHz"), "band a channel 36"),   # newer iw prints the frequency with a decimal
])
def test_pick_channel_skips_no_ir_and_disabled_then_falls_back_to_bg_6(lab, edit, want):
    lab.canned("iw_list", edit(IW_LIST))
    rc, out = lab.run("plan"); assert rc == 0 and want in out, out


def test_key_management_falls_back_to_wpa_psk_without_sae_and_can_be_forced(lab):
    lab.canned("iw_list", IW_LIST.replace("\tDevice supports SAE with AUTHENTICATE command\n", ""))
    rc, out = lab.run("plan"); assert "key-mgmt wpa-psk" in out
    rc, out = lab.run("plan", env=lab.env(SO101_HOTSPOT_KEYMGMT="sae")); assert "key-mgmt sae" in out
    lab.canned("iw_list", IW_LIST)
    rc, out = lab.run("plan", env=lab.env(SO101_HOTSPOT_KEYMGMT="wpa-psk")); assert "key-mgmt wpa-psk" in out


# ---------- up / watchdog / confirm / revert / down ----------
def test_up_creates_the_ap_profile_activates_it_and_arms_the_watchdog(lab):
    key = lab.init_psk(); lab.ok.write_text("")
    rc, out = lab.run("up"); assert rc == 0, out
    nm = lab.calls("nmcli"); writes = [c for c in nm if c[1] == "con" and c[2] in ("add", "modify", "up", "down")]
    assert [c[2] for c in writes] == ["add", "modify", "up"], writes
    add = writes[0]; assert add[3:] == ["type", "wifi", "ifname", "wlP1p1s0", "con-name", "so101-hotspot", "autoconnect", "no", "ssid", "so101-lab"]
    p = pairs(writes[1]); assert writes[1][3] == "so101-hotspot"
    assert p["802-11-wireless.mode"] == "ap" and p["802-11-wireless.band"] == "a" and p["802-11-wireless.channel"] == "36" and p["802-11-wireless.ssid"] == "so101-lab"
    assert p["wifi-sec.key-mgmt"] == "sae" and p["wifi-sec.pmf"] == "2" and p["wifi-sec.psk"] == key and p["wifi-sec.psk-flags"] == "0"
    assert p["ipv4.method"] == "shared" and p["ipv4.addresses"] == "10.101.0.1/24" and p["ipv6.method"] == "disabled" and p["connection.autoconnect"] == "no"
    assert p["connection.interface-name"] == "wlP1p1s0"
    assert writes[2] == ["nmcli", "con", "up", "so101-hotspot"]
    (sr,) = lab.calls("systemd-run")
    assert sr[1:3] == ["--user", "--collect"] and "--on-active=10min" in sr and sr[sr.index("--unit") + 1] == "so101-hotspot-revert"
    assert sr[-3:] == ["/bin/bash", HOTSPOT, "watchdog"] and any(a.startswith("--setenv=SO101_HOME=") for a in sr)
    order = [c[0] for c in lab.calls() if c[0] in ("systemd-run",) or c[:3] in (["nmcli", "con", "up"],)]
    assert order == ["nmcli", "systemd-run"], "the timer is armed after the profile is up"
    assert not lab.ok.exists(), "a stale confirm flag from an earlier session is cleared, so this run's watchdog is live"
    assert key not in out, "the key is masked in the echoed nmcli line"
    assert "wifi-sec.psk ********" in out and "hotspot.sh confirm" in out


def test_up_updates_an_existing_profile_instead_of_adding(lab):
    lab.init_psk(); lab.canned("nmcli_cons", "super-net\nso101-hotspot\n")
    rc, out = lab.run("up", "--watchdog-min", "3"); assert rc == 0 and "exists: updating" in out
    writes = [c[2] for c in lab.calls("nmcli") if c[1] == "con" and c[2] in ("add", "modify", "up")]
    assert writes == ["modify", "up"]
    assert "--on-active=3min" in lab.calls("systemd-run")[0]


def test_up_uses_wpa_psk_and_the_ssid_override(lab):
    lab.init_psk(); lab.canned("iw_list", IW_LIST.replace("\tDevice supports SAE with AUTHENTICATE command\n", "").replace("[36] (23.0 dBm)", "[36] (no IR)"))
    rc, out = lab.run("up", env=lab.env(SO101_HOTSPOT_SSID="bench-7")); assert rc == 0, out
    mod = [c for c in lab.calls("nmcli") if c[2] == "modify"][0]; p = pairs(mod)
    assert p["wifi-sec.key-mgmt"] == "wpa-psk" and p["wifi-sec.pmf"] == "1" and p["802-11-wireless.channel"] == "40" and p["802-11-wireless.ssid"] == "bench-7"
    assert [c for c in lab.calls("nmcli") if c[2] == "add"][0][-1] == "bench-7"
    rc, out = lab.run("up", "--ssid", "bench-8"); assert pairs([c for c in lab.calls("nmcli") if c[2] == "modify"][-1])["802-11-wireless.ssid"] == "bench-8"


def test_up_refuses_without_a_sane_passphrase(lab):
    rc, out = lab.run("up"); assert rc == 1 and "missing" in out and "hotspot.sh init" in out
    assert not any(c[2] in ("add", "modify", "up") for c in lab.calls("nmcli"))
    lab.init_psk(mode=0o644)
    rc, out = lab.run("up"); assert rc == 1 and "mode is 644" in out and not any(c[2] == "modify" for c in lab.calls("nmcli"))


def test_up_dry_run_writes_nothing_and_shows_the_plan(lab):
    key = lab.init_psk(); lab.ok.write_text("")
    rc, out = lab.run("up", "--dry-run"); assert rc == 0, out
    assert [c for c in lab.calls("nmcli") if c[2] in ("add", "modify", "up")] == [] and lab.calls("systemd-run") == []
    assert lab.ok.exists(), "dry-run does not even clear the confirm flag"
    assert "dry-run: nmcli con add" in out and "dry-run: nmcli con modify so101-hotspot 802-11-wireless.mode ap" in out
    assert "dry-run: nmcli con up so101-hotspot" in out and "dry-run: systemd-run --user --collect --on-active=10min" in out
    assert key not in out and "wifi-sec.psk ********" in out


def test_up_reverts_at_once_when_the_watchdog_cannot_be_armed(lab):
    """No safety net, no hotspot: a failed systemd-run puts the home Wi-Fi back and exits 1."""
    lab.init_psk(); lab.rc("systemd-run", 1)
    rc, out = lab.run("up"); assert rc == 1 and "could not arm the watchdog" in out
    ups = [c for c in lab.calls("nmcli") if c[1:3] == ["con", "up"]]; downs = [c for c in lab.calls("nmcli") if c[1:3] == ["con", "down"]]
    assert ups == [["nmcli", "con", "up", "so101-hotspot"], ["nmcli", "con", "up", "super-net"]] and downs == [["nmcli", "con", "down", "so101-hotspot"]]


def test_up_puts_the_client_back_when_activation_fails(lab):
    lab.init_psk(); lab.rc("nmcli-up", 4)   # why: only `con up` fails; add/modify succeed, as when the driver refuses AP activation
    rc, out = lab.run("up"); assert rc == 1 and "putting super-net back" in out
    assert lab.calls("systemd-run") == [] and ["nmcli", "con", "up", "super-net"] in lab.calls("nmcli")


def test_up_no_watchdog_skips_the_timer(lab):
    lab.init_psk()
    rc, out = lab.run("up", "--no-watchdog"); assert rc == 0 and lab.calls("systemd-run") == [] and "skipped (--no-watchdog)" in out


def test_confirm_touches_the_flag_and_stops_the_timer(lab):
    rc, out = lab.run("confirm"); assert rc == 0, out
    assert lab.ok.exists() and ["systemctl", "--user", "stop", "so101-hotspot-revert.timer"] in lab.calls("systemctl")
    assert "hotspot.sh install" in out and "hotspot.sh firewall" in out
    lab.ok.unlink(); lab.clear()
    rc, out = lab.run("confirm", "--dry-run"); assert rc == 0 and not lab.ok.exists() and lab.calls("systemctl") == []


def test_revert_brings_super_net_up_hotspot_down_and_clears_the_flag(lab):
    lab.ok.write_text("")
    rc, out = lab.run("revert"); assert rc == 0, out
    nm = [c for c in lab.calls("nmcli") if c[1] == "con" and c[2] in ("up", "down")]
    assert nm == [["nmcli", "con", "down", "so101-hotspot"], ["nmcli", "con", "up", "super-net"]]
    assert not lab.ok.exists() and ["systemctl", "--user", "stop", "so101-hotspot-revert.timer"] in lab.calls("systemctl")


def test_watchdog_reverts_only_without_the_confirm_flag(lab):
    lab.ok.write_text("")
    rc, out = lab.run("watchdog"); assert rc == 0 and "confirmed" in out and lab.calls("nmcli") == [] and lab.ok.exists()
    lab.ok.unlink()
    rc, out = lab.run("watchdog"); assert rc == 0 and "no confirm within the window" in out
    assert ["nmcli", "con", "up", "super-net"] in lab.calls("nmcli") and ["nmcli", "con", "down", "so101-hotspot"] in lab.calls("nmcli")


def test_down_is_the_revert_without_timer_bookkeeping(lab):
    lab.ok.write_text("")
    rc, out = lab.run("down"); assert rc == 0
    assert [c[2:] for c in lab.calls("nmcli") if c[1] == "con"] == [["down", "so101-hotspot"], ["up", "super-net"]]
    assert lab.calls("systemctl") == [] and lab.ok.exists()
    lab.clear(); rc, out = lab.run("down", "--dry-run"); assert rc == 0 and [c for c in lab.calls("nmcli") if c[1] == "con"] == [] and "dry-run: nmcli con up super-net" in out


def test_client_profile_and_interfaces_follow_the_environment(lab):
    lab.init_psk()
    env = lab.env(SO101_HOTSPOT_CLIENT="shop-wifi", SO101_HOTSPOT_WIFI_IF="wlan0", SO101_HOTSPOT_ETH_IF="eth0", SO101_HOTSPOT_CON="lab-ap")
    lab.canned("ip_link_eth0", "2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 state UP\n"); lab.canned("ip_link_wlan0", "3: wlan0: <UP,LOWER_UP>\n")
    rc, out = lab.run("revert", env=env); assert [c[2:] for c in lab.calls("nmcli") if c[1] == "con"] == [["down", "lab-ap"], ["up", "shop-wifi"]]
    lab.clear(); rc, out = lab.run("up", "--dry-run", env=env); assert "dry-run: nmcli con add type wifi ifname wlan0 con-name lab-ap" in out


# ---------- status ----------
def test_status_is_one_screen(lab):
    lab.canned("iw_info", IW_INFO_AP); lab.canned("iw_stations", IW_STATIONS); lab.canned("ip_addr_wlP1p1s0", WIFI_ADDR_AP)
    lab.canned("nmcli_active", "so101-hotspot\nWired connection 1\n"); lab.canned("nmcli_active_dev", "so101-hotspot:wlP1p1s0\nWired connection 1:enP8p1s0\n")
    lab.canned("systemctl_timers", TIMERS_PENDING)
    rc, out = lab.run("status"); assert rc == 0, out
    for needle in ("mode AP channel 36 ssid so101-lab", "profile    so101-hotspot active on wlP1p1s0; so101-hotspot up, super-net down", "clients    2 associated",
                   "hotspot ip 10.101.0.1/24", "watchdog   pending (so101-hotspot-revert.timer)", "ethernet   enP8p1s0 carrier 192.0.2.20/24, default route via enP8p1s0"):
        assert needle in out, (needle, out)
    assert ["iw", "dev", "wlP1p1s0", "station", "dump"] in lab.calls("iw")


def test_status_before_the_hotspot_and_without_the_cable(lab):
    lab.canned("ip_link_enP8p1s0", LINK_NO_CARRIER); lab.canned("ip_addr_enP8p1s0", ""); lab.ok.write_text("")
    rc, out = lab.run("status"); assert rc == 0
    assert "mode managed channel 48 ssid super-net" in out and "so101-hotspot down, super-net up" in out and "clients    0 associated" in out
    assert "hotspot ip none" in out and "watchdog   not pending, confirmed" in out and "enP8p1s0 NO CARRIER no IPv4" in out


# ---------- install: the dispatcher ----------
def test_install_stages_the_dispatcher_and_prints_the_sudo_lines(lab):
    lab.canned("nmcli_cons", "super-net\nso101-hotspot\n")
    rc, out = lab.run("install"); assert rc == 1 and "no confirmed session" in out
    lab.ok.write_text("")
    rc, out = lab.run("install", "--dry-run"); assert rc == 0 and "dry-run: would write" in out and not (lab.so / "outputs").exists()
    rc, out = lab.run("install", env=lab.env(SO101_HOTSPOT_ETH_IF="eth0", SO101_HOTSPOT_CLIENT="shop-wifi")); assert rc == 0, out
    staged = lab.so / "outputs" / "hotspot" / "90-so101-hotspot"
    assert staged.exists() and lab.mode(staged) == 0o755
    text = staged.read_text(); assert "ETH=eth0\n" in text and "CLIENT=shop-wifi\n" in text and "CON=so101-hotspot\n" in text and "NFT=/etc/so101-hotspot.nft\n" in text
    assert f"sudo install -m 755 -o root -g root {staged} /etc/NetworkManager/dispatcher.d/90-so101-hotspot" in out
    assert "sudo" not in " ".join(" ".join(c) for c in lab.calls()), "the script never runs sudo itself"
    assert "undo: sudo rm /etc/NetworkManager/dispatcher.d/90-so101-hotspot" in out
    lab.canned("nmcli_cons", "super-net\n")
    rc, out = lab.run("install"); assert rc == 1 and "does not exist yet" in out


def test_dispatcher_up_branch_starts_the_hotspot_and_applies_the_firewall(lab):
    lab.add_tool("nft"); (lab.fake / "sys" / "enP8p1s0").mkdir(parents=True); (lab.fake / "sys" / "enP8p1s0" / "carrier").write_text("1\n")
    rc, out = lab.dispatch("enP8p1s0", "up", SO101_SYSFS_NET=str(lab.fake / "sys")); assert rc == 0, out
    assert ["nmcli", "--wait", "30", "con", "up", "so101-hotspot"] in lab.calls("nmcli") and "starting so101-hotspot" in out
    assert lab.calls("nft") == [], "no /etc/so101-hotspot.nft on this box: nothing to apply"
    assert not any(c[1:3] == ["con", "down"] for c in lab.calls("nmcli"))


def test_dispatcher_up_without_carrier_or_already_active_does_nothing(lab):
    (lab.fake / "sys" / "enP8p1s0").mkdir(parents=True); (lab.fake / "sys" / "enP8p1s0" / "carrier").write_text("0\n")
    rc, out = lab.dispatch("enP8p1s0", "up", SO101_SYSFS_NET=str(lab.fake / "sys")); assert rc == 0 and "without carrier" in out
    assert not any(c[1:] and "up" in c for c in lab.calls("nmcli"))
    lab.clear(); lab.canned("nmcli_active", "so101-hotspot\nWired connection 1\n")
    rc, out = lab.dispatch("enP8p1s0", "up"); assert rc == 0 and "already active" in out and not any("--wait" in c for c in lab.calls("nmcli"))


def test_dispatcher_down_branch_restores_the_client_and_drops_the_table(lab):
    lab.add_tool("nft")
    rc, out = lab.dispatch("enP8p1s0", "down"); assert rc == 0, out
    nm = [c for c in lab.calls("nmcli") if "con" in c]
    assert ["nmcli", "con", "down", "so101-hotspot"] in nm and ["nmcli", "--wait", "30", "con", "up", "super-net"] in nm
    assert nm.index(["nmcli", "con", "down", "so101-hotspot"]) < nm.index(["nmcli", "--wait", "30", "con", "up", "super-net"])
    assert lab.calls("nft") == [["nft", "delete", "table", "inet", "so101_hotspot"]]


@pytest.mark.parametrize("iface, action", [("wlP1p1s0", "up"), ("wlP1p1s0", "down"), ("enP8p1s0", "pre-up"), ("enP8p1s0", "connectivity-change"), ("", "")])
def test_dispatcher_ignores_other_interfaces_and_actions(lab, iface, action):
    rc, out = lab.dispatch(iface, action); assert rc == 0 and lab.calls() == [], (iface, action, out)


def test_dispatcher_file_is_executable_and_posix_sh():
    text = open(DISPATCHER).read()
    assert text.startswith("#!/bin/sh\n") and os.access(DISPATCHER, os.X_OK)
    assert "sudo install -m 755 -o root -g root" in text and "/etc/NetworkManager/dispatcher.d/90-so101-hotspot" in text
    assert "$HOME" not in text and "~/" not in text.replace("~/so101/outputs/hotspot/90-so101-hotspot", ""), "no user paths: NM runs it as root"


# ---------- firewall ----------
def test_firewall_forwards_only_the_pad_port_and_blocks_ssh(lab, tmp_path):
    out_file = tmp_path / "x.nft"
    rc, out = lab.run("firewall", "--out", str(out_file)); assert rc == 0, out
    text = out_file.read_text(); assert lab.mode(out_file) == 0o644
    forward = text[text.index("chain forward"):]
    accepts = [l.strip() for l in forward.splitlines() if " accept" in l]
    assert any('iifname "wlP1p1s0" oifname "enP8p1s0" ip saddr 10.101.0.0/24 tcp dport 8780 accept' in a for a in accepts)
    assert sum("dport" in a for a in accepts) == 1, "exactly one forwarded port, the pad gateway"
    assert "22" not in re.sub(r"#.*", "", text).replace("10.101.0", ""), "no rule mentions ssh in either chain"
    assert 'udp dport { 67, 68 } accept' in text and "udp dport 53 accept" in text and "tcp dport 53 accept" in text
    assert text.count("limit rate 6/minute log prefix") == 2 and text.count('iifname "wlP1p1s0" drop') == 2
    assert "table inet so101_hotspot {}\ndelete table inet so101_hotspot\n" in text, "idempotent reload"
    assert "sudo install -m 644" in out and "sudo nft -f /etc/so101-hotspot.nft" in out
    assert not re.search(r"\b192\.168\.|\b10\.0\.|\b172\.(1[6-9]|2\d|3[01])\.", text), "no LAN addresses in the ruleset"


def test_firewall_mac_port_and_mac_address_narrow_the_forward(lab, tmp_path):
    out_file = tmp_path / "y.nft"
    rc, out = lab.run("firewall", "--out", str(out_file), "--mac-port", "9000", "--mac", "192.0.2.9"); assert rc == 0, out
    assert "ip saddr 10.101.0.0/24 ip daddr 192.0.2.9 tcp dport 9000 accept" in out_file.read_text()
    rc, out = lab.run("firewall", "--out", str(out_file), "--mac", "mac.example"); assert rc == 1 and "must be an IPv4 address" in out


def test_firewall_dry_run_prints_the_ruleset_and_writes_nothing(lab, tmp_path):
    out_file = tmp_path / "z.nft"
    rc, out = lab.run("firewall", "--out", str(out_file), "--dry-run"); assert rc == 0 and not out_file.exists()
    assert "dry-run: would write" in out and "tcp dport 8780 accept" in out


def test_committed_nft_file_matches_the_default_generation(lab, tmp_path):
    """jetson/so101-hotspot.nft is `hotspot.sh firewall` with the defaults; regenerate it when the template changes."""
    out_file = tmp_path / "default.nft"
    rc, out = lab.run("firewall", "--out", str(out_file)); assert rc == 0
    assert open(NFT_FILE).read() == out_file.read_text(), "regenerate with: jetson/hotspot.sh firewall"


# ---------- measure ----------
def test_measure_runs_iperf3_when_present_and_skips_otherwise(lab):
    rc, out = lab.run("measure", "--iperf", "mac.example"); assert rc == 0 and "iperf3 is not installed: skipping" in out
    lab.add_tool("iperf3")
    rc, out = lab.run("measure", "--iperf", "mac.example"); assert rc == 0 and "94.2 Mbits/sec" in out and "association time" in out
    assert lab.calls("iperf3") == [["iperf3", "-c", "mac.example", "-t", "10", "-f", "m"]]
    lab.clear(); rc, out = lab.run("measure", "--iperf", "mac.example", "--dry-run"); assert lab.calls("iperf3") == [] and "dry-run: iperf3 -c mac.example" in out
    rc, out = lab.run("measure"); assert rc == 1 and "--iperf HOST" in out


# ---------- usage, bash 3.2 floor, env.sh ----------
def test_usage_and_unknown_subcommand(lab):
    rc, out = lab.run("--help"); assert rc == 0 and "hotspot.sh preflight" in out and "hotspot.sh firewall" in out and "set -u" not in out
    rc, out = lab.run("bogus"); assert rc == 1 and "unknown subcommand" in out
    rc, out = lab.run("status", "--bogus"); assert rc == 1 and "unknown option" in out
    rc, out = lab.run(); assert rc == 1


def test_script_stays_within_bash_3_2():
    code = "\n".join(l for l in open(HOTSPOT).read().splitlines() if not l.lstrip().startswith("#"))   # the header lists the bashisms it avoids
    for bashism in ("mapfile", "readarray", "declare -A", "local -n", "|&", "&>>"):
        assert bashism not in code, bashism
    assert not re.search(r"\$\{[A-Za-z_]+(,,|\^\^)", code), "no case-conversion expansions"
    v = subprocess.run(["bash", "-c", "echo $BASH_VERSION"], capture_output=True, text=True).stdout.strip()
    assert v, "bash on PATH"


def test_env_sh_defines_so101_hotspot(tmp_path):
    """`so101-hotspot ...` in jetson/env.sh hands every argument to ~/so101/hotspot.sh."""
    home = tmp_path / "home"; so = home / "so101"; (so / ".venv" / "bin").mkdir(parents=True); (so / ".venv" / "bin" / "activate").write_text("")
    argv = so / "argv"; (so / "hotspot.sh").write_text(f'printf "%s\\n" "$@" > "{argv}"\n')
    env = dict(os.environ, HOME=str(home), PATH="/usr/bin:/bin")
    r = subprocess.run(["bash", "-c", f"source {os.path.join(JETSON, 'env.sh')}; so101-hotspot preflight --mac mac.example --dry-run"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    assert argv.read_text().splitlines() == ["preflight", "--mac", "mac.example", "--dry-run"]
