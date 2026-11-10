# The lab hotspot: the Jetson's Wi-Fi radio as the iPad's network

The iPad panel (`docs/ipad-panel.md`) needs the iPad and the Mac on one network. At shows and in rooms without a usable
Wi-Fi the Jetson becomes that network: its radio runs as an access point (`so101-lab`, `10.101.0.0/24`), the iPad joins it,
and the Jetson routes the iPad to the Mac's pad gateway (port 8780) over an Ethernet cable to TC's hotspot device, which
the Mac is also on. Everything is driven through NetworkManager with `jetson/hotspot.sh` (`so101-hotspot` in the Jetson's
`env.sh`).

## What the Jetson can and cannot do (survey, 10 Nov 2026)

Orin Nano dev kit, kernel 5.15.148-tegra, Realtek RTL8822CE on the vendor driver `rtl88x2ce`, NetworkManager 1.36.6,
dnsmasq installed (NetworkManager's shared mode uses it), no hostapd, no second Wi-Fi adapter. `iw list` reports
**"interface combinations are not supported"**: the radio is a client of the home Wi-Fi *or* an access point, never both at
once. Supported modes are IBSS, managed, AP and P2P, so AP itself is fine. The home Wi-Fi profile is `super-net`
(channel 48, 80 MHz). Ethernet `enP8p1s0` exists and had no carrier at survey time.

Hence option A: the cable is the uplink, the radio is the hotspot, and while the hotspot is up the Jetson is reachable only
over the cable. Every step below assumes that.

## Status, honestly

**Nothing in this document has run on the real Jetson yet.** The script, the dispatcher, the firewall and the runbook were
written on the Mac against fake `nmcli` / `iw` / `ip` / `ping` / `ssh` / `systemd-run` / `systemctl` tools
(`tests/jetson/test_hotspot.py`, 57 tests) that answer with the surveyed `iw list` text. Unverified on hardware: that
`rtl88x2ce` actually beacons in AP mode under wpa_supplicant, that SAE (WPA3) works with it (if the iPad cannot join, force
WPA2 with `SO101_HOTSPOT_KEYMGMT=wpa-psk so101-hotspot up`), the real association time, throughput, and that the
dispatcher fires on this NetworkManager build. The first real run is the checklist below; write what happened in the
handoff doc.

## The run, in order

Do this at the bench with the Mac next to you, not remotely: the home Wi-Fi drops at step 5.

1. **Plug the cable** from the Jetson's Ethernet port into the hotspot device. The Mac joins that same device's network.
   Note the Mac's address on it (`ipconfig getifaddr en0` on the Mac, or whatever interface it uses).

2. **Preflight**, from the usual ssh session:

   ```
   so101-hotspot preflight --mac <mac-address-on-that-network> --mac-ssh tc@<that-address>
   ```

   Every line must say `ok`: carrier and an IPv4 on `enP8p1s0`, the Mac answers 3 pings (and ssh without a prompt if the
   key is installed), NetworkManager running, `* AP` in `iw list`, the radio present, the passphrase file sane. It also
   prints the plan (`band a channel 36, key-mgmt sae` on the surveyed box). Exit 1 means stop and fix; nothing has changed.

3. **Open a second ssh session over the Ethernet address** of the Jetson (the `inet` preflight printed). Do the rest from
   that session. The Wi-Fi session dies at step 5; that is expected.

4. **Make the passphrase** (once): `so101-hotspot init`. 24 random characters in `~/so101/.hotspot_psk`, mode 600, never
   printed; `.hotspot_psk` is git-ignored and preflight refuses a key that sits in a checkout without that ignore line.
   `init --force` rotates it (then every device must re-join).

5. **Bring the hotspot up**: `so101-hotspot up` (`--dry-run` first shows every nmcli line it would run, key masked).
   It creates or updates the `so101-hotspot` profile on `wlP1p1s0`: mode `ap`, band `a`, the first of channels 36/40/44/48
   that `iw list` does not mark `no IR` or disabled (else 2.4 GHz channel 6), `wifi-sec.key-mgmt sae` + `pmf 2` when the
   driver advertises SAE (else `wpa-psk`), `ipv4.method shared` with the fixed address `10.101.0.1/24`, IPv6 off,
   autoconnect off. Then `nmcli con up so101-hotspot` (NetworkManager drops `super-net` itself) and it **arms the
   watchdog**: a transient `systemd --user` timer, `so101-hotspot-revert`, runs `hotspot.sh watchdog` 10 minutes later
   (`--watchdog-min N` changes it). If `confirm` has not run by then, the hotspot goes down and `super-net` comes back.
   If the watchdog cannot be armed, `up` reverts at once rather than leave the box with no safety net.

6. **Join the iPad**: `so101-hotspot qr` draws a `WIFI:` QR code on the terminal (needs `sudo apt install qrencode`; without
   it the command prints the SSID and where the key file is). Settings > Wi-Fi > `so101-lab`, or scan with the camera.

7. **Check**: `so101-hotspot status`, one screen: radio mode (`AP`) and channel, which profile is active, number of
   associated clients (`iw station dump`), the hotspot IP, whether the watchdog is pending, Ethernet carrier and the default
   route. One client and `10.101.0.1/24` is what you want.

8. **Load the pad link** on the iPad: `so101-dash pad` on the Mac prints `http://<mac-address-on-the-cable-network>:8780/?k=...`.
   The iPad reaches it through the Jetson (NetworkManager's shared mode NATs `10.101.0.0/24` out of the cable).

9. **Confirm within 10 minutes**: `so101-hotspot confirm`. Touches `~/so101/.hotspot_ok` and stops the timer. Miss it and
   the hotspot reverts by itself; just run `up` again. `so101-hotspot revert` is the same revert by hand; `so101-hotspot
   down` is the plain hotspot-off, home-Wi-Fi-on with no timer bookkeeping.

10. **Make it permanent** (after a confirmed session): `so101-hotspot install` stages the NetworkManager dispatcher
    script (`jetson/90-so101-hotspot.dispatcher`, with the interface and profile names filled in) under
    `~/so101/outputs/hotspot/` and prints the one sudo line that installs it to `/etc/NetworkManager/dispatcher.d/90-so101-hotspot`
    (root-owned, mode 755: NetworkManager ignores anything else). From then on: cable up with carrier -> hotspot up;
    cable down -> hotspot down, `super-net` back. An unplugged cable never strands the box. The script never runs sudo
    itself; `journalctl -t so101-hotspot -f` shows it reacting.

11. **Firewall** (optional, recommended at a show): `so101-hotspot firewall` writes `so101-hotspot.nft` and prints the sudo
    lines (`sudo install -m 644 ... /etc/so101-hotspot.nft && sudo nft -f /etc/so101-hotspot.nft`). Hotspot clients may:
    get DHCP and DNS from the Jetson, ping it, and reach **tcp/8780 only** beyond it (`--mac-port N` for another port,
    `--mac <ipv4>` to pin the destination; the committed file names no address). Everything else from `10.101.0.0/24` is
    dropped and logged at 6 lines a minute (`journalctl -k | grep so101-hotspot`): the Mac's ssh, the Jetson's own ssh,
    the rest of the cable network. The dispatcher applies the file when the hotspot comes up and deletes the table when it
    goes down. Non-hotspot traffic is untouched (every rule is scoped to the radio).

12. **Measure** (optional): `so101-hotspot measure --iperf <mac-address>` runs 10 s of iperf3 to the Mac (`iperf3 -s`
    there; skipped with a note if iperf3 is not installed). Time the iPad's association by hand: tap the network, count
    until `status` shows one client; 2 to 5 s is normal. Put both numbers in the handoff doc.

## Undo everything

```
so101-hotspot down                                            # hotspot off, super-net back (works at any time)
sudo rm /etc/NetworkManager/dispatcher.d/90-so101-hotspot    # the cable no longer switches the radio
sudo nft delete table inet so101_hotspot; sudo rm /etc/so101-hotspot.nft
nmcli con delete so101-hotspot                                # optional: forget the profile (the key file stays)
```

## Knobs

`SO101_HOTSPOT_SSID` (`so101-lab`, or `--ssid`), `SO101_HOTSPOT_WIFI_IF` (`wlP1p1s0`), `SO101_HOTSPOT_ETH_IF` (`enP8p1s0`),
`SO101_HOTSPOT_CLIENT` (`super-net`), `SO101_HOTSPOT_CON` (`so101-hotspot`), `SO101_HOTSPOT_ADDR` (`10.101.0.1/24`),
`SO101_HOTSPOT_KEYMGMT` (`auto` | `sae` | `wpa-psk`), `SO101_HOME` (`~/so101`). `--dry-run` on any subcommand prints what
it would do and writes nothing. `so101-hotspot plan` shows the channel and key management `up` would pick.

## Files

| File | What |
|---|---|
| `jetson/hotspot.sh` | the script (bash 3.2 compatible, so the Mac can test it) |
| `jetson/90-so101-hotspot.dispatcher` | NetworkManager dispatcher, POSIX sh, no user paths (runs as root) |
| `jetson/so101-hotspot.nft` | the generated default ruleset (regenerate with `jetson/hotspot.sh firewall` after editing the template) |
| `jetson/env.sh` | `so101-hotspot` |
| `tests/jetson/test_hotspot.py`, `hotspothelpers.py` | fake tools on PATH, canned `iw list`, every subcommand, the dispatcher branches, the ruleset |

Copy to the Jetson with the rest of `jetson/`: `scp jetson/hotspot.sh jetson/90-so101-hotspot.dispatcher jetson/so101-hotspot.nft jetson/env.sh jetson:~/so101/`.
