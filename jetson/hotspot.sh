#!/bin/bash
# hotspot.sh · the Jetson's Wi-Fi radio as the lab hotspot (docs/hotspot.md). Everything goes through nmcli: NetworkManager
# owns the radio on this box, so the hotspot is an NM connection profile, not a hostapd fight.
#
#   hotspot.sh preflight [--mac HOST] [--mac-ssh USER@HOST]
#                        Ethernet carrier + IPv4, default route via Ethernet (or the Mac answers 3 pings when --mac is given;
#                        ssh BatchMode login when --mac-ssh is given), NetworkManager running, the radio supports AP mode,
#                        the passphrase file is sane (600, 20+ chars, git-ignored if it sits in a checkout). Exit 1 on any miss.
#   hotspot.sh init      [--force]   write a fresh 24-char passphrase to ~/so101/.hotspot_psk (mode 600). Never printed.
#   hotspot.sh plan                  what `up` would pick: band, channel, key management (from `iw list`).
#   hotspot.sh up        [--watchdog-min N] [--no-watchdog]
#                        create/update the `so101-hotspot` profile (AP, 5 GHz channel from pick_channel, SAE or WPA2-PSK,
#                        ipv4 shared 10.101.0.1/24, no autoconnect), activate it, then arm a transient systemd --user timer
#                        that runs `hotspot.sh watchdog` N minutes later (default 10): unless `confirm` ran by then the
#                        home Wi-Fi comes back and the hotspot goes down. Dropping the client profile kills an ssh session
#                        that came in over Wi-Fi: run `up` from a session over the Ethernet address.
#   hotspot.sh confirm   touch ~/so101/.hotspot_ok and stop the timer: the hotspot stays.
#   hotspot.sh watchdog  (the timer's target) revert unless ~/so101/.hotspot_ok exists.
#   hotspot.sh revert    hotspot down, home Wi-Fi up, .hotspot_ok removed, timer stopped.
#   hotspot.sh down      hotspot down, home Wi-Fi up (no timer or .hotspot_ok bookkeeping).
#   hotspot.sh status    radio mode, active profile, channel, clients, hotspot IP, watchdog pending?, Ethernet carrier?
#   hotspot.sh qr        WIFI: QR code on the terminal (qrencode -t ansiutf8) or, without qrencode, the SSID and where the key is.
#   hotspot.sh install   [--force]   stage the NetworkManager dispatcher script (Ethernet up -> hotspot, Ethernet down -> home
#                        Wi-Fi) and print the sudo lines that install it. Writes files, never runs sudo. Needs a confirmed session.
#   hotspot.sh firewall  [--mac-port N] [--mac IPV4] [--out FILE]
#                        write the nftables ruleset (hotspot clients reach only the pad gateway port; DHCP/DNS to the Jetson;
#                        everything else from the hotspot dropped and rate-logged) and print the sudo lines that apply it.
#   hotspot.sh measure   --iperf HOST   10 s iperf3 to the Mac over the current path, if iperf3 is installed.
#
# Common: --dry-run (every subcommand prints what it would do; nothing is written, no nmcli writes), --ssid NAME.
# Environment: SO101_HOTSPOT_SSID (so101-lab), SO101_HOTSPOT_WIFI_IF (wlP1p1s0), SO101_HOTSPOT_ETH_IF (enP8p1s0),
# SO101_HOTSPOT_CLIENT (super-net, the home Wi-Fi profile), SO101_HOTSPOT_CON (so101-hotspot), SO101_HOTSPOT_ADDR
# (10.101.0.1/24), SO101_HOTSPOT_KEYMGMT (auto|sae|wpa-psk; auto reads `iw list`), SO101_HOME (~/so101).
#
# Measured on the lab Jetson (Orin Nano, 5.15.148-tegra, Realtek RTL8822CE on rtl88x2ce, NetworkManager 1.36.6, 10 Nov 2026):
# `iw list` says "interface combinations are not supported", so the radio is a client OR an AP, never both. That is why
# the uplink is the Ethernet cable to the hotspot device and why the watchdog exists. Bash 3.2 compatible on purpose (the
# tests run it on the Mac): no mapfile, no ${var,,}, no associative arrays.
set -u

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"; HERE="$(dirname "$SELF")"
SSID="${SO101_HOTSPOT_SSID:-so101-lab}"
WIFI="${SO101_HOTSPOT_WIFI_IF:-wlP1p1s0}"
ETH="${SO101_HOTSPOT_ETH_IF:-enP8p1s0}"
CLIENT="${SO101_HOTSPOT_CLIENT:-super-net}"
CON="${SO101_HOTSPOT_CON:-so101-hotspot}"
ADDR="${SO101_HOTSPOT_ADDR:-10.101.0.1/24}"
SUBNET="$(printf '%s' "${ADDR%/*}" | sed 's/\.[0-9]*$/.0/')/24"   # why: the subnet is a fixed /24 so the firewall and the runbook can name it
KEYMGMT="${SO101_HOTSPOT_KEYMGMT:-auto}"
SO101_HOME="${SO101_HOME:-$HOME/so101}"
PSK_FILE="$SO101_HOME/.hotspot_psk"; OK_FILE="$SO101_HOME/.hotspot_ok"
TIMER_UNIT="so101-hotspot-revert"                     # systemd-run --on-active makes $TIMER_UNIT.timer + .service
DISPATCHER_SRC="$HERE/90-so101-hotspot.dispatcher"; DISPATCHER_DST="/etc/NetworkManager/dispatcher.d/90-so101-hotspot"
NFT_DST="/etc/so101-hotspot.nft"
STAGE="$SO101_HOME/outputs/hotspot"
DRY=0; MAC=""; MAC_SSH=""; MAC_PORT=8780; WATCHDOG_MIN=10; NO_WATCHDOG=0; FORCE=0; IPERF=""; OUT=""
FAILS=0

say() { printf '%s\n' "$*"; }
die() { say "error: $*" >&2; exit 1; }
ok() { say "ok    $*"; }
bad() { say "FAIL  $*"; FAILS=$((FAILS + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }
masked() {   # argv with the value after wifi-sec.psk hidden: the only place the key could leak into a terminal or a log
  local out="" prev="" a; for a in "$@"; do [ "$prev" = "wifi-sec.psk" ] && a="********"; out="$out $a"; prev="$a"; done; printf '%s' "${out# }"; }
do_() {      # a write: shown in full (key masked), skipped under --dry-run
  if [ "$DRY" = 1 ]; then say "dry-run: $(masked "$@")"; return 0; fi
  say "+ $(masked "$@")"; "$@"; }
write_file() {   # write_file PATH MODE < content; skipped under --dry-run
  local path="$1" mode="$2"; if [ "$DRY" = 1 ]; then say "dry-run: would write $path (mode $mode) with:"; sed 's/^/    /'; return 0; fi
  mkdir -p "$(dirname "$path")" && (umask 077; cat > "$path") && chmod "$mode" "$path" && say "wrote $path (mode $mode)"; }
file_mode() { stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1" 2>/dev/null; }

# ---- probes (read-only; these run even under --dry-run so the plan is the real one) ----
iw_list() { have iw && iw list 2>/dev/null; return 0; }
eth_carrier() { ip -o link show dev "$ETH" 2>/dev/null | grep -q 'LOWER_UP'; }
eth_addr() { ip -4 -o addr show dev "$ETH" 2>/dev/null | sed -n 's/.*inet \([0-9.]*\/[0-9]*\).*/\1/p' | head -1; }
default_via_eth() { ip -4 route show default 2>/dev/null | grep -q "dev $ETH"; }
wifi_addr() { ip -4 -o addr show dev "$WIFI" 2>/dev/null | sed -n 's/.*inet \([0-9.]*\/[0-9]*\).*/\1/p' | head -1; }
con_exists() { nmcli -t -f NAME con show 2>/dev/null | grep -qx "$CON"; }
con_active() { nmcli -t -f NAME con show --active 2>/dev/null | grep -qx "$1"; }
nm_running() { [ "$(nmcli -t -f RUNNING general 2>/dev/null)" = "running" ]; }
timer_pending() { systemctl --user list-timers --all 2>/dev/null | grep -q "$TIMER_UNIT"; }
stop_timer() { systemctl --user stop "$TIMER_UNIT.timer" >/dev/null 2>&1; systemctl --user stop "$TIMER_UNIT.service" >/dev/null 2>&1; return 0; }

pick_channel() {   # stdin: `iw list`; prints "<band> <channel>": first of 36/40/44/48 that is neither "no IR" nor disabled, else bg 6
  local list ch line; list="$(cat)"
  for ch in 36 40 44 48; do
    line="$(printf '%s\n' "$list" | grep -E "^[[:space:]]*\* [0-9]+(\.[0-9]+)? MHz \[$ch\]" | head -1)"
    [ -n "$line" ] || continue
    printf '%s' "$line" | grep -qiE 'no IR|disabled' && continue   # why: a no-IR channel may be joined but never beaconed on, which is what an AP does
    say "a $ch"; return 0
  done
  say "bg 6"; }
pick_keymgmt() {   # stdin: `iw list`; sae when the driver advertises it (WPA3, what the iPad prefers), else wpa-psk
  case "$KEYMGMT" in sae|wpa-psk) say "$KEYMGMT"; cat >/dev/null; return 0;; esac
  if grep -qE 'supports SAE|\[ SAE' ; then say sae; else say wpa-psk; fi; }

# ---- passphrase ----
repo_root() {   # the nearest ancestor of $1 that holds a .git; empty if none
  local d; d="$(cd "$1" 2>/dev/null && pwd)" || return 0
  while [ -n "$d" ] && [ "$d" != "/" ]; do [ -e "$d/.git" ] && { printf '%s' "$d"; return 0; }; d="$(dirname "$d")"; done; return 0; }
check_psk() {   # ok/FAIL lines; returns the number of misses through FAILS
  local before=$FAILS mode len root
  if [ ! -f "$PSK_FILE" ]; then bad "passphrase file $PSK_FILE missing (run: hotspot.sh init)"; return 1; fi
  mode="$(file_mode "$PSK_FILE")"; [ "$mode" = "600" ] && ok "passphrase file mode 600" || bad "passphrase file mode is $mode, want 600 (chmod 600 $PSK_FILE)"
  len="$(tr -d '\n\r ' < "$PSK_FILE" | wc -c | tr -d ' ')"; [ "$len" -ge 20 ] && ok "passphrase is $len chars" || bad "passphrase is $len chars, want 20+ (hotspot.sh init --force)"
  root="$(repo_root "$(dirname "$PSK_FILE")")"
  if [ -n "$root" ]; then   # why: on the Mac ~/so101 is the git checkout; the key must be ignored there so it can never be committed
    grep -qx '\.hotspot_psk' "$root/.gitignore" 2>/dev/null && ok "passphrase file is git-ignored in $root" || bad "passphrase file sits in the checkout $root and .gitignore does not list .hotspot_psk"
  else ok "passphrase file is outside any git checkout"; fi
  [ "$FAILS" = "$before" ]; }
read_psk() { tr -d '\n\r ' < "$PSK_FILE"; }

# ---- subcommands ----
cmd_preflight() {
  say "preflight: Ethernet uplink $ETH, radio $WIFI, hotspot profile $CON (SSID $SSID)"
  local a
  eth_carrier && ok "$ETH has carrier" || bad "$ETH has no carrier: is the cable in the hotspot device?"
  a="$(eth_addr)"; [ -n "$a" ] && ok "$ETH has IPv4 $a" || bad "$ETH has no IPv4 address (DHCP from the hotspot device?)"
  if [ -n "$MAC" ]; then
    ping -c 3 -W 2 "$MAC" >/dev/null 2>&1 && ok "Mac $MAC answers ping (3/3 sent)" || bad "Mac $MAC does not answer ping"
  else default_via_eth && ok "default route goes via $ETH" || bad "default route is not via $ETH (pass --mac HOST to check the Mac directly)"; fi
  if [ -n "$MAC_SSH" ]; then ssh -o BatchMode=yes -o ConnectTimeout=5 "$MAC_SSH" true >/dev/null 2>&1 && ok "ssh $MAC_SSH works without a prompt" || bad "ssh -o BatchMode=yes $MAC_SSH failed (key not installed?)"; fi
  if have nmcli; then nm_running && ok "NetworkManager is running" || bad "NetworkManager is not running"; else bad "nmcli not found"; fi
  if have iw; then
    local list; list="$(iw_list)"
    printf '%s\n' "$list" | grep -q '^[[:space:]]*\* AP$' && ok "radio supports AP mode" || bad "iw list does not show '* AP': this radio cannot be a hotspot"
    printf '%s\n' "$list" | grep -q 'interface combinations are not supported' && say "note  no client+AP at once on this radio: the home Wi-Fi drops while the hotspot is up (that is the plan)"
    say "plan  $(printf '%s\n' "$list" | pick_channel | awk '{print "band " $1 " channel " $2}'), key-mgmt $(printf '%s\n' "$list" | pick_keymgmt)"
  else bad "iw not found (apt install iw)"; fi
  ip -o link show dev "$WIFI" >/dev/null 2>&1 && ok "radio $WIFI exists" || bad "radio $WIFI not found (ip link)"
  check_psk || true
  [ "$FAILS" = 0 ] && { say "preflight: all good"; return 0; }
  say "preflight: $FAILS problem(s)"; return 1; }

cmd_init() {
  if [ -f "$PSK_FILE" ] && [ "$FORCE" != 1 ]; then say "passphrase file $PSK_FILE exists; keeping it (--force to replace, which also means re-running up and re-joining every device)"; return 0; fi
  say "init: writing a new 24-char passphrase to $PSK_FILE (mode 600); the key is never printed, use: hotspot.sh qr"
  [ "$DRY" = 1 ] && { say "dry-run: would write $PSK_FILE"; return 0; }
  local key; key="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24)"
  [ "${#key}" = 24 ] || die "could not draw 24 random chars"
  mkdir -p "$(dirname "$PSK_FILE")" && (umask 077; printf '%s\n' "$key" > "$PSK_FILE") && chmod 600 "$PSK_FILE" || die "cannot write $PSK_FILE"
  say "wrote $PSK_FILE"; }

cmd_plan() {
  local list; list="$(iw_list)"
  [ -n "$list" ] || say "note  iw list is empty here: falling back to band bg channel 6 and wpa-psk"
  set -- $(printf '%s\n' "$list" | pick_channel)
  say "band $1 channel $2 key-mgmt $(printf '%s\n' "$list" | pick_keymgmt) ssid $SSID address $ADDR interface $WIFI"; }

cmd_up() {
  check_psk || die "fix the passphrase file first (hotspot.sh init)"
  local list band ch km psk; list="$(iw_list)"
  set -- $(printf '%s\n' "$list" | pick_channel); band="$1"; ch="$2"
  km="$(printf '%s\n' "$list" | pick_keymgmt)"; psk="$(read_psk)"
  say "up: $CON on $WIFI, SSID $SSID, band $band channel $ch, key-mgmt $km, $ADDR shared via $ETH; $CLIENT drops while it runs"
  if [ "$DRY" = 1 ]; then say "dry-run: would remove $OK_FILE"; else rm -f "$OK_FILE"; fi   # why: a stale confirm from last time must not disarm this run's watchdog
  if con_exists; then say "profile $CON exists: updating it"
  else do_ nmcli con add type wifi ifname "$WIFI" con-name "$CON" autoconnect no ssid "$SSID" || die "nmcli con add failed"; fi
  set -- 802-11-wireless.mode ap 802-11-wireless.band "$band" 802-11-wireless.channel "$ch" 802-11-wireless.ssid "$SSID" connection.interface-name "$WIFI" \
         ipv4.method shared ipv4.addresses "$ADDR" ipv6.method disabled connection.autoconnect no wifi-sec.key-mgmt "$km"
  if [ "$km" = sae ]; then set -- "$@" wifi-sec.pmf 2; else set -- "$@" wifi-sec.pmf 1; fi   # why: WPA3 requires protected management frames; WPA2 only tolerates them
  set -- "$@" wifi-sec.psk-flags 0 wifi-sec.psk "$psk"   # why: psk-flags 0 stores the key in the system profile, so the dispatcher (root, no secret agent) can activate it
  do_ nmcli con modify "$CON" "$@" || die "nmcli con modify failed"
  do_ nmcli con up "$CON" || { say "activation failed: putting $CLIENT back"; do_ nmcli con up "$CLIENT"; exit 1; }
  if [ "$NO_WATCHDOG" = 1 ]; then say "watchdog: skipped (--no-watchdog): nothing brings $CLIENT back by itself"; return 0; fi
  say "watchdog: in $WATCHDOG_MIN min, unless 'hotspot.sh confirm' ran, $CLIENT comes back and $CON goes down"
  [ "$DRY" = 1 ] || stop_timer   # why: systemd-run refuses a unit name that still exists from an earlier run
  do_ systemd-run --user --collect --on-active="${WATCHDOG_MIN}min" --unit "$TIMER_UNIT" --setenv="SO101_HOME=$SO101_HOME" /bin/bash "$SELF" watchdog \
    || { say "could not arm the watchdog: reverting now rather than leaving the box with no safety net"; cmd_revert; exit 1; }
  say "next: join the iPad to $SSID (hotspot.sh qr), check hotspot.sh status, load the pad link, then hotspot.sh confirm"; }

cmd_confirm() {
  say "confirm: keeping $CON up; stopping the $TIMER_UNIT timer; touching $OK_FILE"
  if [ "$DRY" = 1 ]; then say "dry-run: would touch $OK_FILE and stop $TIMER_UNIT.timer"; return 0; fi
  mkdir -p "$SO101_HOME" && touch "$OK_FILE" || die "cannot touch $OK_FILE"
  do_ systemctl --user stop "$TIMER_UNIT.timer" || say "(no timer was pending)"
  say "confirmed. next: hotspot.sh install, hotspot.sh firewall"; }

cmd_down() {
  say "down: $CON off, $CLIENT back on $WIFI"
  do_ nmcli con down "$CON" || say "($CON was not active)"
  do_ nmcli con up "$CLIENT" || die "nmcli con up $CLIENT failed: the box may be off the network; use the Ethernet address"; }

cmd_revert() {
  say "revert: $CON off, $CLIENT back, $OK_FILE removed, $TIMER_UNIT stopped"
  cmd_down
  if [ "$DRY" = 1 ]; then say "dry-run: would remove $OK_FILE and stop $TIMER_UNIT.timer"; return 0; fi
  rm -f "$OK_FILE"; stop_timer; }

cmd_watchdog() {
  if [ -e "$OK_FILE" ]; then say "watchdog: $OK_FILE exists (confirmed), leaving $CON up"; return 0; fi
  say "watchdog: no confirm within the window: reverting to $CLIENT"; cmd_revert; }

cmd_status() {
  local info mode ch clients a active
  info="$(iw dev "$WIFI" info 2>/dev/null)"
  mode="$(printf '%s\n' "$info" | sed -n 's/^[[:space:]]*type \(.*\)/\1/p' | head -1)"; ch="$(printf '%s\n' "$info" | sed -n 's/^[[:space:]]*channel \([0-9]*\).*/\1/p' | head -1)"
  clients="$(iw dev "$WIFI" station dump 2>/dev/null | grep -c Station | tr -d ' ')"
  say "radio      $WIFI mode ${mode:-?} channel ${ch:-?} ssid $(printf '%s\n' "$info" | sed -n 's/^[[:space:]]*ssid \(.*\)/\1/p' | head -1)"
  active="$(nmcli -t -f NAME,DEVICE con show --active 2>/dev/null | grep ":$WIFI$" | cut -d: -f1 | head -1)"
  say "profile    ${active:-none} active on $WIFI; $CON $(con_active "$CON" && echo up || echo down), $CLIENT $(con_active "$CLIENT" && echo up || echo down)"
  say "clients    ${clients:-0} associated"
  a="$(wifi_addr)"; say "hotspot ip ${a:-none} (clients get 10.101.0.x from NetworkManager's dnsmasq)"
  say "watchdog   $(timer_pending && echo "pending ($TIMER_UNIT.timer): run hotspot.sh confirm to keep the hotspot" || echo "not pending")$( [ -e "$OK_FILE" ] && echo ", confirmed ($OK_FILE)")"
  a="$(eth_addr)"; say "ethernet   $ETH $(eth_carrier && echo carrier || echo "NO CARRIER") ${a:-no IPv4}, default route $(default_via_eth && echo "via $ETH" || echo "not via $ETH")"
  return 0; }

cmd_qr() {
  [ -f "$PSK_FILE" ] || die "no passphrase file $PSK_FILE (hotspot.sh init)"
  local psk esc_ssid; psk="$(read_psk)"; esc_ssid="$(printf '%s' "$SSID" | sed 's/[\\;,:"]/\\&/g')"
  say "join: SSID $SSID (WPA2/WPA3 passphrase in $PSK_FILE, not printed)"
  if have qrencode; then
    [ "$DRY" = 1 ] && { say "dry-run: would render the WIFI: QR code with qrencode -t ansiutf8"; return 0; }
    printf 'WIFI:T:WPA;S:%s;P:%s;;' "$esc_ssid" "$psk" | qrencode -t ansiutf8
  else say "qrencode is not installed (sudo apt install qrencode): type the passphrase from the file on the iPad (cat $PSK_FILE in a shell nobody is watching)"; fi; }

cmd_install() {
  say "install: make the hotspot follow the Ethernet cable (NetworkManager dispatcher, root-owned, no sudo run by this script)"
  [ -f "$DISPATCHER_SRC" ] || die "dispatcher source $DISPATCHER_SRC not found next to this script"
  con_exists || die "profile $CON does not exist yet: run hotspot.sh up (and confirm) first"
  if [ ! -e "$OK_FILE" ] && [ "$FORCE" != 1 ]; then die "no confirmed session ($OK_FILE missing): run up, test from the iPad, confirm; or --force"; fi
  local staged="$STAGE/90-so101-hotspot"
  sed -e "s/^ETH=.*/ETH=$ETH/" -e "s/^CON=.*/CON=$CON/" -e "s/^CLIENT=.*/CLIENT=$CLIENT/" -e "s|^NFT=.*|NFT=$NFT_DST|" "$DISPATCHER_SRC" | write_file "$staged" 755
  say "then, as TC (sudo prompts for the password):"
  say "  sudo install -m 755 -o root -g root $staged $DISPATCHER_DST"
  say "  sudo nmcli con modify $CLIENT connection.autoconnect-priority -10     # optional: the cable wins at boot even before the dispatcher runs"
  say "  journalctl -t so101-hotspot -f                                           # watch it react to the cable"
  say "undo: sudo rm $DISPATCHER_DST; hotspot.sh down"; }

cmd_firewall() {
  local out="${OUT:-$HERE/so101-hotspot.nft}" daddr=""
  if [ -n "$MAC" ]; then
    printf '%s' "$MAC" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' && daddr="ip daddr $MAC " || die "--mac for firewall must be an IPv4 address (nft resolves nothing at load time here)"
  fi
  say "firewall: hotspot clients reach only tcp/$MAC_PORT ${MAC:+on $MAC }beyond the Jetson; DHCP/DNS/ping to the Jetson; everything else from $SUBNET dropped and logged"
  write_file "$out" 644 <<EOF
#!/usr/sbin/nft -f
# so101-hotspot.nft · what hotspot clients (10.101.0.0/24 on the Jetson's radio) may do: generated by hotspot.sh firewall.
# Apply:   sudo install -m 644 ~/so101/so101-hotspot.nft ${NFT_DST} && sudo nft -f ${NFT_DST}
# Remove:  sudo nft delete table inet so101_hotspot
# The dispatcher (90-so101-hotspot) applies ${NFT_DST} when the hotspot comes up and deletes the table when it goes down.
# why its own table: NetworkManager's shared mode adds its own NAT/forward accepts for the subnet; a drop here still wins
# because every table's hook chain sees the packet and one drop is final. Non-hotspot traffic is untouched (policy accept,
# every rule is scoped to the radio interface).
table inet so101_hotspot {}
delete table inet so101_hotspot
table inet so101_hotspot {
    chain input {
        type filter hook input priority filter + 10; policy accept;
        iifname "${WIFI}" ct state established,related accept
        iifname "${WIFI}" udp dport { 67, 68 } accept                         # DHCP: a joining client has no address yet, so no saddr match here
        iifname "${WIFI}" ip saddr ${SUBNET} udp dport 53 accept        # DNS (NetworkManager's dnsmasq)
        iifname "${WIFI}" ip saddr ${SUBNET} tcp dport 53 accept
        iifname "${WIFI}" ip saddr ${SUBNET} icmp type echo-request accept   # ping the gateway while debugging
        iifname "${WIFI}" meta nfproto ipv6 drop                             # ipv6 is disabled on the profile; no need to log the chatter
        iifname "${WIFI}" limit rate 6/minute log prefix "so101-hotspot drop in: "
        iifname "${WIFI}" drop                                               # includes the Jetson's own ssh (22): clients never log in here
    }
    chain forward {
        type filter hook forward priority filter + 10; policy accept;
        iifname "${WIFI}" oifname "${ETH}" ip saddr ${SUBNET} ${daddr}tcp dport ${MAC_PORT} accept   # the pad gateway, nothing else (not the Mac's ssh)
        oifname "${WIFI}" ct state established,related accept
        iifname "${WIFI}" meta nfproto ipv6 drop
        iifname "${WIFI}" limit rate 6/minute log prefix "so101-hotspot drop fwd: "
        iifname "${WIFI}" drop
    }
}
EOF
  say "apply, as TC: sudo install -m 644 $out $NFT_DST && sudo nft -f $NFT_DST"
  say "check:        sudo nft list table inet so101_hotspot; drops: journalctl -k | grep so101-hotspot"
  say "undo:         sudo nft delete table inet so101_hotspot; sudo rm $NFT_DST"; }

cmd_measure() {
  [ -n "$IPERF" ] || die "measure needs --iperf HOST (the Mac, running: iperf3 -s)"
  say "measure: 10 s iperf3 to $IPERF over the current path (Ethernet uplink before the hotspot is up, or from a client)"
  if ! have iperf3; then say "iperf3 is not installed: skipping (sudo apt install iperf3 here, brew install iperf3 on the Mac, then iperf3 -s there)"; return 0; fi
  do_ iperf3 -c "$IPERF" -t 10 -f m || say "iperf3 failed: is 'iperf3 -s' running on $IPERF?"
  say "association time: tap $SSID on the iPad and count until 'hotspot.sh status' shows 1 client; note it in the handoff doc (2-5 s is normal)"; }

usage() { sed -n '2,/^set -u/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'; }

# ---- argv ----
[ $# -ge 1 ] || { usage; exit 1; }
CMD="$1"; shift
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1;;
    --force) FORCE=1;;
    --no-watchdog) NO_WATCHDOG=1;;
    --mac) MAC="${2:-}"; shift;;
    --mac-ssh) MAC_SSH="${2:-}"; shift;;
    --mac-port) MAC_PORT="${2:-}"; shift;;
    --watchdog-min) WATCHDOG_MIN="${2:-}"; shift;;
    --iperf) IPERF="${2:-}"; shift;;
    --out) OUT="${2:-}"; shift;;
    --ssid) SSID="${2:-}"; shift;;
    -h|--help) usage; exit 0;;
    *) die "unknown option $1 (hotspot.sh --help)";;
  esac; shift
done
[ "$DRY" = 1 ] && say "(dry-run: nothing is written and nmcli is only read)"
case "$CMD" in
  preflight) cmd_preflight;;  init) cmd_init;;  plan) cmd_plan;;  up) cmd_up;;  confirm) cmd_confirm;;  watchdog) cmd_watchdog;;
  revert) cmd_revert;;  down) cmd_down;;  status) cmd_status;;  qr) cmd_qr;;  install) cmd_install;;  firewall) cmd_firewall;;  measure) cmd_measure;;
  -h|--help|help) usage; exit 0;;
  *) die "unknown subcommand $CMD (hotspot.sh --help)";;
esac
