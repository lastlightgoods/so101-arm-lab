#!/usr/bin/env bash
# One-command redeploy of the so101 arm lab static site to Cloudflare Workers.
set -euo pipefail
cd "$(dirname "$0")"
CI=true /opt/homebrew/bin/wrangler deploy
echo
echo "Live at: https://so101-arm-lab.c-thomas-clark-pdx.workers.dev"
