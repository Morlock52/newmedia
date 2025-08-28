#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

echo "Bringing up Traefik and base services..."
docker compose up -d traefik jellyfin sonarr radarr bazarr lidarr readarr calibre-web navidrome audiobookshelf sabnzbd uptime-kuma

# Start request portal based on REQUEST_PORTAL env
case "${REQUEST_PORTAL:-overseerr}" in
  jellyseerr)
    echo "Starting Jellyseerr (request portal)"
    docker compose rm -fsv overseerr >/dev/null 2>&1 || true
    docker compose up -d jellyseerr
    ;;
  overseerr|*)
    echo "Starting Overseerr (request portal)"
    docker compose rm -fsv jellyseerr >/dev/null 2>&1 || true
    docker compose up -d overseerr
    ;;
esac

if [ "${PROWLARR_VPN:-false}" = "true" ]; then
  echo "Enabling Prowlarr via VPN (gluetun)"
  docker compose rm -fsv prowlarr >/dev/null 2>&1 || true
  docker compose up -d gluetun prowlarr-vpn qbittorrent
  echo "Access Prowlarr via VPN at: https://prowlarrvpn.${DOMAIN}"
else
  echo "Using direct Prowlarr (not via VPN)"
  docker compose up -d prowlarr
  docker compose up -d gluetun qbittorrent
  echo "Access Prowlarr at: https://prowlarr.${DOMAIN}"
fi

# Optional: bring up monitoring UIs
if docker compose config --services | grep -q '^tautulli$'; then
  docker compose up -d tautulli
fi

