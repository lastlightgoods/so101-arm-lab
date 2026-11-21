#!/bin/bash
# pi5_lab_dns.sh · .lab names for the lab network, served by dnsmasq on the Raspberry Pi (tc-pi5).
#
# dnsmasq answers <name>.lab from a hosts file and forwards every other name to the upstream resolvers the router handed
# the Pi (read from NetworkManager when the config is written; nothing about the household's DNS provider changes).
# The machines get their addresses by DHCP, so the hosts file is not typed by hand: `refresh` (a systemd --user timer,
# every minute) looks each machine up by mDNS and rewrites the file when an address changed. dnsmasq rereads files in
# its hostsdir by itself (inotify), so no root and no reload are needed after setup. A machine that does not answer
# mDNS keeps its last known address (logged once), so a sleeping Mac does not make its names vanish.
#
# The name map lives on the Pi, not in this public repo (machine names are personal): $SO101_DNS_NAMES, one line each,
#   <mDNS host or "self"> <name> [name ...]      e.g.  self pi dns        /  some-mac.local mac review arm
# "self" is this Pi's own IPv4. Names get the .lab suffix.
#
#   pi5_lab_dns.sh refresh [--dry-run]  resolve the map, rewrite the hosts file only when something changed
#   pi5_lab_dns.sh conf                 print the dnsmasq config (root installs it once: see `setup`)
#   pi5_lab_dns.sh setup                print the one-time sudo lines (dnsmasq, avahi-utils, config, hostsdir)
#   pi5_lab_dns.sh install|uninstall    the refresh timer (systemd --user; linger is already on for tc)
#   pi5_lab_dns.sh status               hosts file, last refresh, dnsmasq state
# Environment: SO101_DNS_NAMES (~/so101/.lab_dns_names), SO101_DNS_HOSTSDIR (/etc/so101-dns/hosts.d), SO101_DNS_DOMAIN (lab),
#   SO101_DNS_IFACES ("wlan0 eth0"), SO101_DNS_UPSTREAM (space-separated; default: the IP4.DNS of the first active device).
# Bash 3.2 compatible (the tests run it on the Mac).
set -u

NAMES="${SO101_DNS_NAMES:-$HOME/so101/.lab_dns_names}"
HOSTSDIR="${SO101_DNS_HOSTSDIR:-/etc/so101-dns/hosts.d}"
DOMAIN="${SO101_DNS_DOMAIN:-lab}"
IFACES="${SO101_DNS_IFACES:-wlan0 eth0}"
HOSTS="$HOSTSDIR/$DOMAIN.hosts"
LOG="${SO101_DNS_LOG:-$HOME/so101/lab_dns.log}"
UNIT=so101-lab-dns
DRY=0

say() { printf '%s\n' "$*"; }
log() { [ "$DRY" = 1 ] && return 0; printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }
is_ipv4() { printf '%s' "$1" | grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$'; }

self_ip() { hostname -I 2>/dev/null </dev/null | tr ' ' '\n' | grep -E '^([0-9]{1,3}\.){3}[0-9]{1,3}$' | grep -v '^127\.' | head -1; }
mdns_ip() { avahi-resolve-host-name -4 "$1" 2>/dev/null </dev/null | awk 'NF >= 2 {print $2; exit}'; }
last_ip() {   # the address the current hosts file gives NAME (first name of a line), if any
  [ -f "$HOSTS" ] && awk -v n="$1.$DOMAIN" '$2 == n {print $1; exit}' "$HOSTS"; }

cmd_refresh() {
  [ -f "$NAMES" ] || { say "no name map at $NAMES"; return 1; }
  local tmp host first ip line names n out=""
  tmp="$(mktemp "${TMPDIR:-/tmp}/lab-hosts.XXXXXX")" || return 1
  printf '# written by pi5_lab_dns.sh refresh from %s; do not edit (the next refresh replaces it)\n' "$NAMES" > "$tmp"
  while read -r host first rest_names; do
    case "$host" in ''|'#'*) continue ;; esac
    [ -n "$first" ] || continue
    if [ "$host" = self ]; then ip="$(self_ip)"; else ip="$(mdns_ip "$host")"; fi
    if ! is_ipv4 "${ip:-}"; then
      ip="$(last_ip "$first")"
      if is_ipv4 "${ip:-}"; then say "$host: no mDNS answer, keeping $ip"; else say "$host: no mDNS answer and no earlier address, skipped"; continue; fi
    fi
    names=""; for n in $first ${rest_names:-}; do names="$names $n.$DOMAIN"; done
    line="$ip${names}"; printf '%s\n' "$line" >> "$tmp"; out="$out$line; "
  done < "$NAMES"
  if [ -f "$HOSTS" ] && cmp -s "$tmp" "$HOSTS"; then rm -f "$tmp"; say "unchanged: ${out% }"; return 0; fi
  if [ "$DRY" = 1 ]; then say "dry-run: would write $HOSTS:"; sed 's/^/    /' "$tmp"; rm -f "$tmp"; return 0; fi
  mkdir -p "$HOSTSDIR" && chmod 644 "$tmp" && mv -f "$tmp" "$HOSTS" || { rm -f "$tmp"; say "cannot write $HOSTS"; return 1; }
  log "hosts: ${out% }"; say "wrote $HOSTS: ${out% }"; }

upstream() {
  if [ -n "${SO101_DNS_UPSTREAM:-}" ]; then printf '%s\n' $SO101_DNS_UPSTREAM; return 0; fi
  local dev; dev="$(nmcli -t -f DEVICE,STATE dev 2>/dev/null | awk -F: '$2 == "connected" {print $1; exit}')"
  # never this Pi itself: once the router hands out the Pi as DNS, NetworkManager lists it too, and forwarding to it would loop
  [ -n "$dev" ] && nmcli -t -f IP4.DNS dev show "$dev" 2>/dev/null | cut -d: -f2 | grep -E '^([0-9]{1,3}\.){3}[0-9]{1,3}$' | grep -v '^127\.' \
    | grep -vxF -e "$(self_ip)" -e "${SO101_DNS_SELF:-0.0.0.0}"; }

cmd_conf() {
  local ups i; ups="$(upstream)"
  [ -n "$ups" ] || { say "no upstream resolvers found (set SO101_DNS_UPSTREAM)" >&2; return 1; }
  say "# /etc/dnsmasq.d/so101-lab.conf: written by pi5_lab_dns.sh conf. .$DOMAIN from $HOSTSDIR, the rest forwarded."
  for i in $IFACES; do say "interface=$i"; done
  say "bind-dynamic"; say "domain-needed"; say "bogus-priv"; say "no-resolv"
  for i in $ups; do say "server=$i"; done
  say "local=/$DOMAIN/"; say "hostsdir=$HOSTSDIR"; say "local-ttl=60"; say "cache-size=2000"; }

cmd_setup() {
  say "# one time, as tc on the Pi (sudo asks for tc's password):"
  say "sudo apt-get install -y dnsmasq avahi-utils"
  say "sudo install -d -o $USER -g $USER -m 755 $HOSTSDIR"
  say "bash $HOME/so101/pi5_lab_dns.sh conf | sudo tee /etc/dnsmasq.d/so101-lab.conf >/dev/null"
  say "bash $HOME/so101/pi5_lab_dns.sh refresh && sudo systemctl restart dnsmasq && bash $HOME/so101/pi5_lab_dns.sh install"; }

cmd_install() {
  local d="$HOME/.config/systemd/user"; mkdir -p "$d"
  cat > "$d/$UNIT.service" <<EOF
[Unit]
Description=SO-101 lab DNS: refresh the .$DOMAIN hosts file from mDNS (pi5_lab_dns.sh)

[Service]
Type=oneshot
ExecStart=/bin/bash $HOME/so101/pi5_lab_dns.sh refresh
EOF
  cat > "$d/$UNIT.timer" <<EOF
[Unit]
Description=SO-101 lab DNS: refresh the .$DOMAIN names every minute

[Timer]
OnBootSec=30s
OnUnitActiveSec=60s

[Install]
WantedBy=timers.target
EOF
  systemctl --user daemon-reload && systemctl --user enable --now "$UNIT.timer" && say "timer enabled: $(systemctl --user is-active "$UNIT.timer")"; }

cmd_uninstall() {
  systemctl --user disable --now "$UNIT.timer" 2>/dev/null || true
  rm -f "$HOME/.config/systemd/user/$UNIT.service" "$HOME/.config/systemd/user/$UNIT.timer"; systemctl --user daemon-reload 2>/dev/null || true; say "timer removed"; }

cmd_status() {
  if [ -f "$HOSTS" ]; then say "$HOSTS:"; grep -v '^#' "$HOSTS" | sed 's/^/  /'; else say "no hosts file yet at $HOSTS"; fi
  [ -f "$LOG" ] && say "last change: $(tail -n 1 "$LOG")"
  command -v systemctl >/dev/null 2>&1 && say "dnsmasq: $(systemctl is-active dnsmasq 2>/dev/null || true)   timer: $(systemctl --user is-active "$UNIT.timer" 2>/dev/null || true)"
  return 0; }

sub="${1:-}"; [ $# -gt 0 ] && shift
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; *) say "unknown option $a" >&2; exit 2 ;; esac; done
case "$sub" in
  refresh) cmd_refresh ;;
  conf) cmd_conf ;;
  setup) cmd_setup ;;
  install) cmd_install ;;
  uninstall) cmd_uninstall ;;
  status) cmd_status ;;
  *) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
