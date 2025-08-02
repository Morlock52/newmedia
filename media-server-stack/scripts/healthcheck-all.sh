#!/usr/bin/env bash
set -euo pipefail

# Healthcheck for all major services in the media server stack
SERVICES=(jellyfin sonarr radarr prowlarr qbittorrent overseerr bazarr homarr)

for svc in "${SERVICES[@]}"; do
  echo "Checking $svc..."
  docker-compose ps $svc
  docker-compose logs --tail 20 $svc | tail -10
  echo "---"
  # Optionally, add curl health endpoint checks here
  case $svc in
    jellyfin)
      curl -fsSL http://localhost:8096/System/Ping || echo "Jellyfin health endpoint failed" ;;
    sonarr)
      curl -fsSL http://localhost:8989/ping || echo "Sonarr health endpoint failed" ;;
    radarr)
      curl -fsSL http://localhost:7878/ping || echo "Radarr health endpoint failed" ;;
    prowlarr)
      curl -fsSL http://localhost:9696/ping || echo "Prowlarr health endpoint failed" ;;
    overseerr)
      curl -fsSL http://localhost:5055/api/v1/status || echo "Overseerr health endpoint failed" ;;
    bazarr)
      curl -fsSL http://localhost:6767/system/status || echo "Bazarr health endpoint failed" ;;
    homarr)
      curl -fsSL http://localhost:7575/api/configs || echo "Homarr health endpoint failed" ;;
    qbittorrent)
      curl -fsSL http://localhost:8080 || echo "qBittorrent health endpoint failed" ;;
  esac
  echo
  sleep 1
done
