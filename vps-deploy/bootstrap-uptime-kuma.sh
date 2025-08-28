#!/usr/bin/env bash
set -euo pipefail

# Attempt to auto-create monitors in Uptime Kuma using its HTTP API.
# Requires admin credentials (UPTIME_KUMA_USERNAME / UPTIME_KUMA_PASSWORD) set in .env
# If API calls fail, script will print guidance and exit without error.

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

UK_HOST=${UPTIME_KUMA_HOST:-http://uptime-kuma:3001}
UK_USER=${UPTIME_KUMA_USERNAME:-}
UK_PASS=${UPTIME_KUMA_PASSWORD:-}
DOMAIN_URL=${DOMAIN:-localhost}
REQUEST_PORTAL=${REQUEST_PORTAL:-overseerr}
PROWLARR_VPN=${PROWLARR_VPN:-false}
MON_GRAFANA=${UPTIME_MONITOR_GRAFANA:-false}
MON_PROMETHEUS=${UPTIME_MONITOR_PROMETHEUS:-false}

if [ -z "$UK_USER" ] || [ -z "$UK_PASS" ]; then
  echo "UPTIME_KUMA_USERNAME/PASSWORD not set; skipping monitor provisioning."
  exit 0
fi

cookiejar=$(mktemp)
trap 'rm -f "$cookiejar"' EXIT

# Login
resp=$(curl -sS -X POST "$UK_HOST/api/login" \
  -c "$cookiejar" -H 'Content-Type: application/json' \
  -d "{\"username\": \"$UK_USER\", \"password\": \"$UK_PASS\"}") || true
if ! echo "$resp" | grep -q 'ok'; then
  echo "Uptime Kuma login failed (is the admin account created?). Skipping."
  exit 0
fi

add_monitor() {
  local name=$1 url=$2
  curl -sS -X POST "$UK_HOST/api/monitor/add" -b "$cookiejar" \
    -H 'Content-Type: application/json' \
    -d "{\"name\": \"$name\", \"type\": \"http\", \"url\": \"$url\", \"interval\": 60}" >/dev/null || true
}

# Core monitors
add_monitor "Jellyfin" "https://jellyfin.$DOMAIN_URL"
add_monitor "Sonarr" "https://sonarr.$DOMAIN_URL"
add_monitor "Radarr" "https://radarr.$DOMAIN_URL"
add_monitor "Prowlarr" "https://prowlarr.$DOMAIN_URL"
add_monitor "qBittorrent" "https://qbittorrent.$DOMAIN_URL"
add_monitor "Bazarr" "https://bazarr.$DOMAIN_URL"
add_monitor "Lidarr" "https://lidarr.$DOMAIN_URL"
add_monitor "Readarr" "https://readarr.$DOMAIN_URL"
if [ "$REQUEST_PORTAL" = "jellyseerr" ]; then
  add_monitor "Jellyseerr" "https://jellyseerr.$DOMAIN_URL"
else
  add_monitor "Overseerr" "https://overseerr.$DOMAIN_URL"
fi
add_monitor "Calibre-Web" "https://calibre.$DOMAIN_URL"
add_monitor "Navidrome" "https://navidrome.$DOMAIN_URL"
add_monitor "Audiobookshelf" "https://audiobooks.$DOMAIN_URL"
add_monitor "SABnzbd" "https://sabnzbd.$DOMAIN_URL"
add_monitor "Traefik" "https://traefik.$DOMAIN_URL"

# If prowlarr is routed via VPN, also monitor the VPN subdomain
if [ "$PROWLARR_VPN" = "true" ]; then
  add_monitor "Prowlarr (VPN)" "https://prowlarrvpn.$DOMAIN_URL"
fi

# Optional monitors
if [ "$MON_GRAFANA" = "true" ]; then
  add_monitor "Grafana" "https://grafana.$DOMAIN_URL"
fi
if [ "$MON_PROMETHEUS" = "true" ]; then
  add_monitor "Prometheus" "https://prometheus.$DOMAIN_URL"
fi

echo "Uptime Kuma monitors provisioned (if API available)."
