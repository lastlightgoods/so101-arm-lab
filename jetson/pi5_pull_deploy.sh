#!/bin/bash
# pi5_pull_deploy.sh · push-to-main deploys for the Raspberry Pi arm host (tc-pi5, scripted jobs only; jetson/pi5_setup.sh).
#
# GitHub cannot reach the Pi, so the Pi pulls: a systemd --user timer runs `pi5_pull_deploy.sh run` every 2 min. Each run
# fetches main from the public repo into ~/so101-src (anonymous https, no credentials on the Pi) and deploys the new commit
# only when all three hold:
#   1. main moved past ~/so101/.deployed_ref
#   2. GitHub CI passed on that exact commit (check run "test" = ruff + pytest; anonymous API, asked only about a new commit;
#      a failure verdict is cached 15 min so a red commit does not spend the 60/h unauthenticated limit)
#   3. no arm job is running (~/so101/.arms_busy absent, no lerobot-* or scripted-job process)
# Deploy = that commit's jetson/ files into ~/so101 and plugins/lerobot-robot-busheal into ~/so101/plugins (the Jetson's
# layout), env.sh with --policy.device=cpu. Each changed file is backed up first to ~/so101/deploy_bak/<UTC stamp>-<sha>/
# and written by rename (a running script keeps its old copy). Never touched: ~/so101/calib (written on the box, as on the
# Jetson), the Jetson-only system files, files deleted upstream, sudo, processes. Nothing is started: jobs stay manual.
# A pyproject.toml change in the plugin reinstalls it (pip install -e) in ~/so101/.venv.
#
#   pi5_pull_deploy.sh run [--dry-run]   one pass (the timer's target). --dry-run fetches, then says what it would do.
#   pi5_pull_deploy.sh status            deployed ref vs origin/main, arm busy?, timer state, last log lines
#   pi5_pull_deploy.sh install           write + enable the user timer; prints the one sudo line it needs (linger)
#   pi5_pull_deploy.sh uninstall         disable + remove the timer
# Log: ~/so101/deploy.log (one line per deploy, and per change of skip reason); journalctl --user -u so101-pull-deploy.
# Rollback: push a revert to main (it deploys like any commit), or copy files back from ~/so101/deploy_bak.
# Environment: SO101_HOME (~/so101), SO101_SRC (~/so101-src), SO101_REPO_URL, SO101_BRANCH (main), SO101_CI_API,
#   SO101_CI_CHECK (test), SO101_REQUIRE_CI (1; 0 skips the CI gate). Bash 3.2 compatible (the tests run it on the Mac).
set -u

SO101_HOME="${SO101_HOME:-$HOME/so101}"
SRC="${SO101_SRC:-$HOME/so101-src}"
URL="${SO101_REPO_URL:-https://github.com/lastlightgoods/so101-arm-lab.git}"
BRANCH="${SO101_BRANCH:-main}"
CI_API="${SO101_CI_API:-https://api.github.com/repos/lastlightgoods/so101-arm-lab}"
CI_CHECK="${SO101_CI_CHECK:-test}"
REQUIRE_CI="${SO101_REQUIRE_CI:-1}"
REF_FILE="$SO101_HOME/.deployed_ref"; LOG="$SO101_HOME/deploy.log"; STATE="$SO101_HOME/.pull_deploy_state"; CI_CACHE="$SO101_HOME/.pull_deploy_ci"
SKIP_FILES=" hotspot.sh 90-so101-hotspot.dispatcher so101-hotspot.nft 99-so101-cams.rules so101-cam-lock.service "   # Jetson-only system files
# a job = a lerobot CLI, or python RUNNING one of the scripts (not `vim goto.py`, not goto.py.bak)
ARM_PAT='lerobot-(teleoperate|record|rollout|calibrate|replay)|python[0-9.]*( -[^ ]+)* ([^ ]*/)?(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper|servo_probe|mark_pose|arm_state)\.py( |$)'
UNIT=so101-pull-deploy
DRY=0

say() { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
log() { if [ "$DRY" = 1 ]; then say "dry-run: log: $*"; return 0; fi
  mkdir -p "$SO101_HOME" && printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; say "$*"; }
note_skip() {   # the timer runs every 2 min: log a skip reason only when it differs from the last one
  local why="$1" last=""; [ -f "$STATE" ] && last="$(cat "$STATE")"
  if [ "$why" = "$last" ]; then say "skip: $why (unchanged)"; return 0; fi
  log "skip: $why"; [ "$DRY" = 1 ] || printf '%s' "$why" > "$STATE"; }
short() { printf '%s' "$1" | cut -c1-7; }

ci_state() {   # ci_state SHA -> success | pending | failure:<conclusion> | unknown:<why>
  local json
  json="$(curl -fsS -m 20 -H 'Accept: application/vnd.github+json' -A so101-pull-deploy "$CI_API/commits/$1/check-runs" 2>&1)" \
    || { printf 'unknown:api %s' "$(printf '%s' "$json" | head -c 100)"; return 0; }
  printf '%s' "$json" | python3 -I -c '
import json, sys
try: runs = [r for r in json.load(sys.stdin).get("check_runs", []) if r.get("name") == sys.argv[1]]
except Exception: print("unknown:unreadable API reply"); sys.exit()
if not runs: print("pending"); sys.exit()
r = max(runs, key=lambda r: r.get("id", 0))   # a re-run is a newer check run under the same name
if r.get("status") != "completed": print("pending")
elif r.get("conclusion") == "success": print("success")
else: print("failure:%s" % r.get("conclusion"))' "$CI_CHECK"; }

ci_cached() {   # a failure verdict for the same commit is reused for 15 min
  local v
  if [ -f "$CI_CACHE" ] && [ "$(cut -d' ' -f1 "$CI_CACHE")" = "$1" ] && [ -n "$(find "$CI_CACHE" -mmin -15 2>/dev/null)" ]; then cut -d' ' -f2- "$CI_CACHE"; return 0; fi
  v="$(ci_state "$1")"
  case "$v" in failure:*) [ "$DRY" = 1 ] || printf '%s %s\n' "$1" "$v" > "$CI_CACHE" ;; esac
  printf '%s' "$v"; }

busy_reason() {   # prints why the arm is busy and returns 0; returns 1 when idle
  [ -e "$SO101_HOME/.arms_busy" ] && { say "arm busy flag ~/so101/.arms_busy"; return 0; }
  local p; p="$(pgrep -af "$ARM_PAT" 2>/dev/null | head -1)"
  [ -n "$p" ] && { say "arm job running: $(printf '%s' "${p#* }" | cut -c1-80)"; return 0; }
  return 1; }

fetch() {   # prints origin/BRANCH after the fetch
  if [ ! -d "$SRC/.git" ]; then git clone -q --branch "$BRANCH" --single-branch "$URL" "$SRC" >&2 || return 1
  else git -C "$SRC" fetch -q origin "$BRANCH" >&2 || return 1; fi
  git -C "$SRC" rev-parse "origin/$BRANCH"; }

deploy() {   # deploy SHA
  local sha="$1" stage bak rel dst n=0 pip_again=0 fail=""
  stage="$(mktemp -d "${TMPDIR:-/tmp}/so101-deploy.XXXXXX")" || { log "deploy of $(short "$sha") FAILED: no temp dir"; return 1; }
  if ! git -C "$SRC" archive "$sha" jetson plugins/lerobot-robot-busheal | tar -x -C "$stage"; then rm -rf "$stage"; log "deploy of $(short "$sha") FAILED: git archive"; return 1; fi
  if [ -f "$stage/jetson/env.sh" ]; then sed 's/--policy\.device=cuda/--policy.device=cpu/g' "$stage/jetson/env.sh" > "$stage/env.cpu" && mv "$stage/env.cpu" "$stage/jetson/env.sh"; fi
  bak="$SO101_HOME/deploy_bak/$(date -u +%Y%m%dT%H%M%SZ)-$(short "$sha")"
  (cd "$stage" && find jetson plugins -type f | LC_ALL=C sort) > "$stage/.list"
  while IFS= read -r rel; do
    case "$rel" in jetson/calib/*) continue ;; esac
    case "$SKIP_FILES" in *" ${rel#jetson/} "*) continue ;; esac
    case "$rel" in jetson/*) dst="${rel#jetson/}" ;; *) dst="$rel" ;; esac
    if [ -f "$SO101_HOME/$dst" ] && cmp -s "$stage/$rel" "$SO101_HOME/$dst"; then continue; fi
    n=$((n + 1)); case "$rel" in plugins/*/pyproject.toml) pip_again=1 ;; esac
    if [ "$DRY" = 1 ]; then say "dry-run: would update $dst"; continue; fi
    if [ -f "$SO101_HOME/$dst" ]; then mkdir -p "$(dirname "$bak/$dst")" && cp -p "$SO101_HOME/$dst" "$bak/$dst" || { fail="backup of $dst"; break; }; fi
    mkdir -p "$(dirname "$SO101_HOME/$dst")" && cp -p "$stage/$rel" "$SO101_HOME/$dst.deploy-new" && mv -f "$SO101_HOME/$dst.deploy-new" "$SO101_HOME/$dst" \
      || { fail="write of $dst"; break; }
    say "updated $dst"
  done < "$stage/.list"
  rm -rf "$stage"
  if [ -n "$fail" ]; then log "deploy of $(short "$sha") FAILED at $fail after $((n - 1)) file(s); backup $bak; .deployed_ref unchanged"; return 1; fi
  if [ "$DRY" = 1 ]; then say "dry-run: would deploy $(short "$sha"): $n file(s)"; return 0; fi
  if [ "$pip_again" = 1 ] && [ -x "$SO101_HOME/.venv/bin/pip" ]; then
    "$SO101_HOME/.venv/bin/pip" install -q -e "$SO101_HOME/plugins/lerobot-robot-busheal" >/dev/null 2>&1 || { log "deploy of $(short "$sha"): plugin reinstall FAILED (files are in place)"; return 1; }
  fi
  printf '%s\n' "$sha" > "$REF_FILE"; rm -f "$STATE"
  log "deployed $(short "$sha") \"$(git -C "$SRC" log -1 --format=%s "$sha" | cut -c1-80)\": $n file(s)$([ "$n" -gt 0 ] && printf ', backup %s' "${bak#$SO101_HOME/}")$([ "$pip_again" = 1 ] && printf ', plugin reinstalled')"; }

cmd_run() {
  local target cur why ci
  mkdir -p "$SO101_HOME"
  if [ "$DRY" = 0 ] && have flock; then exec 9>"$SO101_HOME/.pull_deploy.lock"; flock -n 9 || { say "another run holds the lock"; return 0; }; fi
  target="$(fetch)" || { note_skip "fetch of $URL $BRANCH failed"; return 1; }
  cur="$(cat "$REF_FILE" 2>/dev/null || true)"
  if [ -n "$cur" ] && [ "$(git -C "$SRC" rev-parse -q --verify "$cur^{commit}" 2>/dev/null)" = "$target" ]; then
    say "up to date at $(short "$target")"; [ "$DRY" = 1 ] || rm -f "$STATE"; return 0; fi
  if [ "$REQUIRE_CI" = 1 ]; then
    ci="$(ci_cached "$target")"
    case "$ci" in
      success) ;;
      pending) note_skip "CI not finished on $(short "$target")"; return 0 ;;
      *) note_skip "CI $ci on $(short "$target"): not deploying"; return 0 ;;
    esac
  fi
  if why="$(busy_reason)"; then note_skip "$why ($(short "$target") waits)"; return 0; fi
  deploy "$target"; }

cmd_status() {
  local target why
  say "deployed:  $(cat "$REF_FILE" 2>/dev/null || echo none)"
  if [ -d "$SRC/.git" ]; then target="$(git -C "$SRC" rev-parse "origin/$BRANCH" 2>/dev/null)"
    say "origin/$BRANCH at last fetch: $(short "$target") $(git -C "$SRC" log -1 --format='%cd %s' --date=short "$target" 2>/dev/null | cut -c1-90)"
  else say "no clone yet at $SRC"; fi
  if why="$(busy_reason)"; then say "arm: $why"; else say "arm: idle"; fi
  have systemctl && say "timer: $(systemctl --user is-active "$UNIT.timer" 2>/dev/null || true)"
  if [ -f "$LOG" ]; then say "last log lines:"; tail -n 5 "$LOG" | sed 's/^/  /'; fi; return 0; }

cmd_install() {
  local d="$HOME/.config/systemd/user"; mkdir -p "$d"
  cat > "$d/$UNIT.service" <<EOF
[Unit]
Description=SO-101: deploy new CI-green commits on $BRANCH to ~/so101 when the arm is idle (pi5_pull_deploy.sh)
After=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash $SO101_HOME/pi5_pull_deploy.sh run
EOF
  cat > "$d/$UNIT.timer" <<EOF
[Unit]
Description=SO-101: check $BRANCH for a new commit every 2 min (pi5_pull_deploy.sh)

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
RandomizedDelaySec=15

[Install]
WantedBy=timers.target
EOF
  say "wrote $d/$UNIT.service and $UNIT.timer"
  systemctl --user daemon-reload && systemctl --user enable --now "$UNIT.timer" && say "timer enabled: $(systemctl --user is-active "$UNIT.timer")"
  say "to keep it running with nobody logged in, once (needs tc's password): sudo loginctl enable-linger $USER"; }

cmd_uninstall() {
  local d="$HOME/.config/systemd/user"
  systemctl --user disable --now "$UNIT.timer" 2>/dev/null || true
  rm -f "$d/$UNIT.service" "$d/$UNIT.timer"; systemctl --user daemon-reload 2>/dev/null || true; say "timer removed"; }

sub="${1:-}"; [ $# -gt 0 ] && shift
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; *) say "unknown option $a" >&2; exit 2 ;; esac; done
case "$sub" in
  run) cmd_run ;;
  status) cmd_status ;;
  install) cmd_install ;;
  uninstall) cmd_uninstall ;;
  *) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
