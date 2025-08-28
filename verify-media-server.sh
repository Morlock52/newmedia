#!/bin/bash

# ============================================================================
# MEDIA SERVER VERIFICATION SCRIPT
# ============================================================================
# Verifies that all services are running and accessible
# ============================================================================

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         MEDIA SERVER VERIFICATION - AUGUST 2025          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Docker services
echo -e "${YELLOW}📦 Docker Services:${NC}"
echo "-------------------"

services=(
    "jellyfin:8096:Jellyfin Media Server"
    "sonarr:8989:Sonarr TV Manager"
    "radarr:7878:Radarr Movie Manager"
    "lidarr:8686:Lidarr Music Manager"
    "bazarr:6767:Bazarr Subtitles"
    "prowlarr:9696:Prowlarr Indexer"
    "qbittorrent:8080:qBittorrent"
    "transmission:9091:Transmission"
    "sabnzbd:8082:SABnzbd"
    "overseerr:5056:Overseerr Requests"
    "jellyseerr:5055:Jellyseerr Requests"
    "homepage:3000:Homepage Dashboard"
    "portainer:9000:Portainer Management"
    "nginx-proxy-manager:81:Nginx Proxy Manager"
    "uptime-kuma:3001:Uptime Kuma Monitoring"
)

working=0
total=0

for service in "${services[@]}"; do
    IFS=':' read -r name port description <<< "$service"
    ((total++))
    
    if docker ps | grep -q "$name"; then
        echo -e "${GREEN}✅${NC} $description (http://localhost:$port)"
        ((working++))
    else
        if nc -z localhost $port 2>/dev/null; then
            echo -e "${GREEN}✅${NC} $description (http://localhost:$port) - Running"
            ((working++))
        else
            echo -e "${RED}❌${NC} $description - Not accessible on port $port"
        fi
    fi
done

echo ""
echo -e "${YELLOW}🌐 API Server:${NC}"
echo "---------------"

if curl -s http://localhost:3005/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} API Server (http://localhost:3005)"
    ((working++))
    ((total++))
else
    echo -e "${RED}❌${NC} API Server not running on port 3005"
    echo "   Run: node api/server-fixed.js"
    ((total++))
fi

echo ""
echo -e "${YELLOW}📊 Service Status via API:${NC}"
echo "--------------------------"

if curl -s http://localhost:3005/api/services/status > /dev/null 2>&1; then
    response=$(curl -s http://localhost:3005/api/services/status)
    
    online_count=$(echo "$response" | grep -o '"status":"online"' | wc -l)
    echo -e "${GREEN}✅${NC} Services Online: $online_count"
    
    # Show online services
    echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for service in data:
    if service['status'] == 'online':
        version = service.get('version', 'unknown')
        print(f'   - {service[\"name\"].capitalize()}: v{version}')
" 2>/dev/null || echo "   (Python parsing failed)"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Summary:${NC} $working/$total services accessible"

if [ $working -eq $total ]; then
    echo -e "${GREEN}✅ ALL SERVICES OPERATIONAL!${NC}"
else
    echo -e "${YELLOW}⚠️  Some services need attention${NC}"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📱 Access Points:${NC}"
echo "• Dashboard:    http://localhost:3005"
echo "• Homepage:     http://localhost:3000"
echo "• Jellyfin:     http://localhost:8096"
echo "• Sonarr:       http://localhost:8989"
echo "• Radarr:       http://localhost:7878"
echo "• qBittorrent:  http://localhost:8080"
echo "• Portainer:    http://localhost:9000"
echo ""
echo -e "${GREEN}✨ Your media server is ready to use!${NC}"