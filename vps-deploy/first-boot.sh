#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load env for credentials and portal/vpn toggles
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

echo "Bringing up stack per .env (REQUEST_PORTAL=${REQUEST_PORTAL:-overseerr}, PROWLARR_VPN=${PROWLARR_VPN:-false})"
./up.sh

# Wait a bit for services to be reachable
sleep 5

# Bootstrap ARR integrations
if [ -x ./bootstrap-arr.sh ]; then
  ./bootstrap-arr.sh || true
fi

# Configure qBittorrent as download client in Sonarr/Radarr
if [ -x ./bootstrap-download-clients.sh ]; then
  ./bootstrap-download-clients.sh || true
fi

# Try to add Uptime Kuma monitors if credentials provided
if [ -x ./bootstrap-uptime-kuma.sh ]; then
  ./bootstrap-uptime-kuma.sh || true
fi

# Optionally provision indexers if file provided as arg
if [ $# -ge 1 ] && [ -f "$1" ]; then
  if [ -x ./provision-indexers.sh ]; then
    ./provision-indexers.sh "$1" || true
  fi
fi

echo "First boot automation complete. Review apps UIs to finish any manual steps."
