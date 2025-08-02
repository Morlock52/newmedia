#!/bin/bash

# ============================================================================
# Quick Setup Script - Essential Media Server Configuration
# ============================================================================
# Fast configuration for the most important services
# ============================================================================

set -e

echo "⚡ Quick Media Server Setup"
echo "=========================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running! Please start Docker first.${NC}"
    exit 1
fi

# Start essential services
echo -e "${BLUE}🚀 Checking essential services...${NC}"

# Check which services are already running
RUNNING_SERVICES=$(docker ps --format "{{.Names}}")
ESSENTIAL_SERVICES=(prowlarr sonarr radarr jellyfin qbittorrent homarr grafana prometheus uptime-kuma)
TO_START=()

for service in "${ESSENTIAL_SERVICES[@]}"; do
    if echo "$RUNNING_SERVICES" | grep -q "^${service}$"; then
        echo -e "${GREEN}✅ $service is already running${NC}"
    else
        echo -e "${YELLOW}🚀 Will start $service${NC}"
        TO_START+=("$service")
    fi
done

# Only start services that aren't already running
if [ ${#TO_START[@]} -gt 0 ]; then
    echo -e "${BLUE}📦 Starting services: ${TO_START[*]}...${NC}"
    docker-compose -f docker-compose-demo.yml up -d ${TO_START[*]} 2>&1 | grep -v "is already in use" || true
else
    echo -e "${GREEN}✅ All essential services are already running${NC}"
fi

echo -e "${YELLOW}⏳ Waiting for services to start (20 seconds)...${NC}"
sleep 20

# Quick Prowlarr setup
echo -e "${BLUE}⚙️  Setting up Prowlarr with indexers...${NC}"

# Wait for Prowlarr
timeout=30
while ! curl -s http://localhost:9696 > /dev/null 2>&1; do
    sleep 1
    timeout=$((timeout - 1))
    if [ $timeout -eq 0 ]; then
        echo -e "${RED}❌ Prowlarr failed to start${NC}"
        exit 1
    fi
done

# Get Prowlarr API key
PROWLARR_KEY=$(docker exec prowlarr grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")

if [ -n "$PROWLARR_KEY" ]; then
    # Add 1337x indexer (most reliable free indexer)
    curl -X POST "http://localhost:9696/api/v1/indexer" \
        -H "X-Api-Key: $PROWLARR_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "enable": true,
            "redirect": false,
            "supportsRss": true,
            "supportsSearch": true,
            "protocol": "torrent",
            "privacy": "public",
            "name": "1337x",
            "fields": [],
            "implementationName": "Cardigann",
            "implementation": "Cardigann",
            "configContract": "CardigannSettings",
            "tags": []
        }' 2>/dev/null && echo -e "${GREEN}✅ Added 1337x indexer${NC}"
fi

# Quick qBittorrent setup
echo -e "${BLUE}⚙️  Configuring qBittorrent...${NC}"
docker exec qbittorrent mkdir -p /config/qBittorrent 2>/dev/null || true
docker exec qbittorrent bash -c 'cat > /config/qBittorrent/qBittorrent.conf << "EOF"
[Preferences]
WebUI\Username=admin
WebUI\Password_PBKDF2="@ByteArray(ARCwPWE7RbUSOoJ4n8o+jw==:KQ6n9oxPtFMJlHdqnJ9Vc6wDKdCkPvDZrhzXRvQrPrs6OedFMrKfH3G5h5sD8A9ib2LkCst7u7OpnwQJmLDK7g==)"
WebUI\Port=8080
WebUI\LocalHostAuth=false
Downloads\SavePath=/downloads/complete/
Downloads\TempPath=/downloads/incomplete/
EOF' 2>/dev/null
docker restart qbittorrent > /dev/null 2>&1

# Display results
echo ""
echo -e "${GREEN}✅ Quick setup complete!${NC}"
echo ""
echo -e "${BLUE}🌐 Access your services:${NC}"
echo -e "  ${GREEN}Main Dashboard:${NC} http://localhost:7575 (Homarr)"
echo -e "  ${GREEN}Media Server:${NC} http://localhost:8096 (Jellyfin)"
echo -e "  ${GREEN}Movies:${NC} http://localhost:7878 (Radarr)"
echo -e "  ${GREEN}TV Shows:${NC} http://localhost:8989 (Sonarr)"
echo -e "  ${GREEN}Indexers:${NC} http://localhost:9696 (Prowlarr)"
echo -e "  ${GREEN}Downloads:${NC} http://localhost:8090 (qBittorrent - admin/adminadmin)"
echo -e "  ${GREEN}Monitoring:${NC} http://localhost:3000 (Grafana - admin/admin)"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo -e "  1. Set up Jellyfin at http://localhost:8096"
echo -e "  2. Connect Sonarr/Radarr to Prowlarr (they'll auto-sync)"
echo -e "  3. Add media folders in Sonarr/Radarr"
echo ""
echo -e "${BLUE}💡 For full configuration, run:${NC} ./scripts/auto-configure-all-services.sh"