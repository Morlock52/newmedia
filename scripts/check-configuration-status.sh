#!/bin/bash

# ============================================================================
# Configuration Status Check Script
# ============================================================================
# Shows the current configuration state of all services
# ============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}📊 Media Server Configuration Status${NC}"
echo "===================================="
echo ""

# Function to check if service is running
check_service() {
    local name=$1
    local port=$2
    local container=$3
    
    printf "%-20s" "$name:"
    
    if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        if curl -s "http://localhost:$port" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Running${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  Container up, web UI not ready${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Not running${NC}"
        return 1
    fi
}

# Function to check API connectivity
check_api() {
    local service=$1
    local port=$2
    local container=$3
    
    if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        local api_key=$(docker exec "$container" grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
        if [ -n "$api_key" ]; then
            if curl -s -H "X-Api-Key: $api_key" "http://localhost:$port/api/v3/system/status" > /dev/null 2>&1; then
                echo -e "  ${GREEN}API Key: Configured ✓${NC}"
                return 0
            else
                echo -e "  ${YELLOW}API Key: Found but not working${NC}"
                return 1
            fi
        else
            echo -e "  ${RED}API Key: Not found${NC}"
            return 1
        fi
    fi
}

# Check Docker
echo -e "${BLUE}🐳 Docker Status${NC}"
echo "---------------"
if docker info > /dev/null 2>&1; then
    echo -e "Docker: ${GREEN}✅ Running${NC}"
    CONTAINERS=$(docker ps --format "{{.Names}}" | wc -l)
    echo -e "Active Containers: ${GREEN}$CONTAINERS${NC}"
else
    echo -e "Docker: ${RED}❌ Not running${NC}"
    exit 1
fi
echo ""

# Check Media Management Services
echo -e "${BLUE}📚 Media Management Services${NC}"
echo "---------------------------"
check_service "Prowlarr" 9696 "prowlarr"
check_api "prowlarr" 9696 "prowlarr"

check_service "Sonarr" 8989 "sonarr"
check_api "sonarr" 8989 "sonarr"

check_service "Radarr" 7878 "radarr"
check_api "radarr" 7878 "radarr"

check_service "Lidarr" 8686 "lidarr"
check_api "lidarr" 8686 "lidarr"

check_service "Bazarr" 6767 "bazarr"
echo ""

# Check Media Servers
echo -e "${BLUE}🎬 Media Servers${NC}"
echo "----------------"
check_service "Jellyfin" 8096 "jellyfin"
check_service "Plex" 32400 "plex"
check_service "Emby" 8096 "emby"
echo ""

# Check Download Clients
echo -e "${BLUE}📥 Download Clients${NC}"
echo "-------------------"
check_service "qBittorrent" 8090 "qbittorrent"
check_service "Transmission" 9091 "transmission"
check_service "SABnzbd" 8085 "sabnzbd"
echo ""

# Check Request Services
echo -e "${BLUE}📺 Request Services${NC}"
echo "-------------------"
check_service "Jellyseerr" 5055 "jellyseerr"
check_service "Overseerr" 5056 "overseerr"
echo ""

# Check Monitoring
echo -e "${BLUE}📊 Monitoring Services${NC}"
echo "----------------------"
check_service "Grafana" 3000 "grafana"
check_service "Prometheus" 9090 "prometheus"
check_service "Uptime Kuma" 3004 "uptime-kuma"
check_service "Loki" 3100 "loki"
echo ""

# Check Dashboards
echo -e "${BLUE}🏠 Dashboards${NC}"
echo "-------------"
check_service "Homarr" 7575 "homarr"
check_service "Homepage" 3001 "homepage"
echo ""

# Check Management Tools
echo -e "${BLUE}🛠️  Management Tools${NC}"
echo "--------------------"
check_service "Portainer" 9000 "portainer"
check_service "Nginx Proxy Mgr" 8181 "nginx-proxy-manager"
echo ""

# Check Prowlarr Integration
echo -e "${BLUE}🔗 Integration Status${NC}"
echo "---------------------"

if docker ps --format "{{.Names}}" | grep -q "^prowlarr$"; then
    PROWLARR_KEY=$(docker exec prowlarr grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
    
    if [ -n "$PROWLARR_KEY" ]; then
        # Check indexers
        INDEXER_COUNT=$(curl -s -H "X-Api-Key: $PROWLARR_KEY" "http://localhost:9696/api/v1/indexer" 2>/dev/null | grep -o '"id"' | wc -l)
        echo -e "Prowlarr Indexers: ${GREEN}$INDEXER_COUNT configured${NC}"
        
        # Check connected apps
        APP_COUNT=$(curl -s -H "X-Api-Key: $PROWLARR_KEY" "http://localhost:9696/api/v1/applications" 2>/dev/null | grep -o '"id"' | wc -l)
        echo -e "Connected Apps: ${GREEN}$APP_COUNT connected${NC}"
    else
        echo -e "Prowlarr Integration: ${RED}API key not available${NC}"
    fi
else
    echo -e "Prowlarr Integration: ${RED}Service not running${NC}"
fi
echo ""

# Check disk space
echo -e "${BLUE}💾 Storage Status${NC}"
echo "-----------------"
MEDIA_DIR="/Users/morlock/fun/newmedia/media"
if [ -d "$MEDIA_DIR" ]; then
    DISK_USAGE=$(df -h "$MEDIA_DIR" | tail -1 | awk '{print $5}')
    DISK_AVAIL=$(df -h "$MEDIA_DIR" | tail -1 | awk '{print $4}')
    echo -e "Media Directory: ${GREEN}$MEDIA_DIR${NC}"
    echo -e "Disk Usage: ${YELLOW}$DISK_USAGE${NC} used, ${GREEN}$DISK_AVAIL${NC} available"
else
    echo -e "Media Directory: ${RED}Not found${NC}"
fi
echo ""

# Summary
echo -e "${CYAN}📋 Configuration Summary${NC}"
echo "========================"

TOTAL_SERVICES=$(docker ps --format "{{.Names}}" | wc -l)
CONFIGURED_SERVICES=0

# Count configured services (simplified check)
for service in prowlarr sonarr radarr jellyfin qbittorrent; do
    if docker ps --format "{{.Names}}" | grep -q "^${service}$"; then
        CONFIGURED_SERVICES=$((CONFIGURED_SERVICES + 1))
    fi
done

echo -e "Total Running Services: ${GREEN}$TOTAL_SERVICES${NC}"
echo -e "Core Services Configured: ${GREEN}$CONFIGURED_SERVICES/5${NC}"
echo ""

if [ $CONFIGURED_SERVICES -lt 5 ]; then
    echo -e "${YELLOW}💡 Tip: Run ./scripts/auto-configure-all-services.sh for full setup${NC}"
else
    echo -e "${GREEN}✅ Core services are configured and running!${NC}"
fi

echo ""
echo -e "${BLUE}🔧 Quick Actions:${NC}"
echo "  - Full config: ./scripts/auto-configure-all-services.sh"
echo "  - Quick setup: ./scripts/quick-setup.sh"
echo "  - Health check: ./scripts/health-check.sh"
echo "  - Backup configs: ./scripts/backup-configs.sh"