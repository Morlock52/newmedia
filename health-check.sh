#!/bin/bash
# Health check script for Ultimate Media Server 2025

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🏥 Ultimate Media Server 2025 - Health Check"
echo "==========================================="

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    local name=$3
    
    if curl -f -s -o /dev/null "http://localhost:${port}"; then
        echo -e "${GREEN}✓${NC} ${name} (${service}) - Port ${port}"
        return 0
    else
        echo -e "${RED}✗${NC} ${name} (${service}) - Port ${port}"
        return 1
    fi
}

# Check Docker
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Docker daemon is running"
else
    echo -e "${RED}✗${NC} Docker daemon is not running"
    exit 1
fi

# Check containers
echo -e "\n📦 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check services
echo -e "\n🌐 Service Health:"
check_service "jellyfin" "8096" "Jellyfin Media Server"
check_service "sonarr" "8989" "Sonarr (TV Shows)"
check_service "radarr" "7878" "Radarr (Movies)"
check_service "prowlarr" "9696" "Prowlarr (Indexers)"
check_service "bazarr" "6767" "Bazarr (Subtitles)"
check_service "lidarr" "8686" "Lidarr (Music)"
check_service "readarr" "8787" "Readarr (Books)"
check_service "qbittorrent" "8080" "qBittorrent"
check_service "overseerr" "5055" "Overseerr"
check_service "jellyseerr" "5056" "Jellyseerr"
check_service "tautulli" "8181" "Tautulli"
check_service "homepage" "3001" "Homepage"
check_service "portainer" "9000" "Portainer"
check_service "grafana" "3000" "Grafana"
check_service "prometheus" "9090" "Prometheus"
check_service "authentik" "9091" "Authentik"
check_service "immich" "2283" "Immich Photos"
check_service "navidrome" "4533" "Navidrome Music"
check_service "calibre-web" "8083" "Calibre Web"
check_service "audiobookshelf" "13378" "Audiobookshelf"
check_service "duplicati" "8200" "Duplicati Backup"

# Check disk space
echo -e "\n💾 Disk Usage:"
df -h | grep -E "^/|Filesystem"

# Check memory
echo -e "\n🧠 Memory Usage:"
free -h

# Check Docker resource usage
echo -e "\n📊 Container Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo -e "\n✅ Health check complete!"
