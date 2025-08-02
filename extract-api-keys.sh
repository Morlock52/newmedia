#!/bin/bash

# API Key and Credential Extraction Script
# This script extracts all API keys and credentials from running services

echo "🔐 Media Server API Keys and Credentials"
echo "======================================="
echo "Generated: $(date)"
echo ""
echo "⚠️  KEEP THIS INFORMATION SECURE!"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to extract API key from container
get_api_key() {
    local service=$1
    local container=$2
    local api_key=""
    
    if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        # Try multiple methods to get API key
        api_key=$(docker exec "$container" grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
        
        if [ -z "$api_key" ]; then
            api_key=$(docker exec "$container" cat /config/app.db 2>/dev/null | strings | grep -E '^[a-f0-9]{32}$' | head -1 || echo "")
        fi
        
        if [ -n "$api_key" ]; then
            echo -e "${GREEN}$service API Key:${NC} $api_key"
        else
            echo -e "${YELLOW}$service:${NC} No API key found (may need initial setup)"
        fi
    else
        echo -e "${RED}$service:${NC} Container not running"
    fi
}

echo "📡 ARR Application API Keys:"
echo "============================"
get_api_key "Prowlarr" "prowlarr"
get_api_key "Sonarr" "sonarr"
get_api_key "Radarr" "radarr"
get_api_key "Lidarr" "lidarr"
get_api_key "Bazarr" "bazarr"
get_api_key "Readarr" "readarr"

echo ""
echo "💾 Download Client Credentials:"
echo "=============================="

# qBittorrent
if docker ps --format "{{.Names}}" | grep -q "^qbittorrent$"; then
    echo -e "${GREEN}qBittorrent:${NC}"
    echo "  URL: http://localhost:8090"
    echo "  Username: admin"
    echo "  Password: adminadmin"
    echo "  WebUI Port: 8090"
else
    echo -e "${RED}qBittorrent:${NC} Not running"
fi

# Transmission
if docker ps --format "{{.Names}}" | grep -q "^transmission$"; then
    echo -e "${GREEN}Transmission:${NC}"
    echo "  URL: http://localhost:9091"
    echo "  RPC Enabled: Yes"
    echo "  Authentication: None (open access)"
else
    echo -e "${RED}Transmission:${NC} Not running"
fi

# SABnzbd
if docker ps --format "{{.Names}}" | grep -q "^sabnzbd$"; then
    SABNZBD_KEY=$(docker exec sabnzbd grep -oP '(?<=api_key = )[^$]+' /config/sabnzbd.ini 2>/dev/null || echo "")
    echo -e "${GREEN}SABnzbd:${NC}"
    echo "  URL: http://localhost:8085"
    if [ -n "$SABNZBD_KEY" ]; then
        echo "  API Key: $SABNZBD_KEY"
    else
        echo "  API Key: Not found"
    fi
else
    echo -e "${RED}SABnzbd:${NC} Not running"
fi

echo ""
echo "🎬 Media Server Information:"
echo "==========================="

# Jellyfin
if docker ps --format "{{.Names}}" | grep -q "^jellyfin$"; then
    echo -e "${GREEN}Jellyfin:${NC}"
    echo "  URL: http://localhost:8096"
    echo "  Initial Setup: Required on first access"
    echo "  Libraries: /media/movies, /media/tv, /media/music"
else
    echo -e "${RED}Jellyfin:${NC} Not running"
fi

# Plex
if docker ps --format "{{.Names}}" | grep -q "^plex$"; then
    echo -e "${GREEN}Plex:${NC}"
    echo "  URL: http://localhost:32400/web"
    echo "  Claim Token: Required from https://www.plex.tv/claim"
    echo "  Libraries: /media/movies, /media/tv, /media/music"
else
    echo -e "${RED}Plex:${NC} Not running"
fi

# Emby
if docker ps --format "{{.Names}}" | grep -q "^emby$"; then
    echo -e "${GREEN}Emby:${NC}"
    echo "  URL: http://localhost:8096"
    echo "  Initial Setup: Required on first access"
else
    echo -e "${RED}Emby:${NC} Not running"
fi

echo ""
echo "📊 Monitoring Services:"
echo "======================"

# Grafana
if docker ps --format "{{.Names}}" | grep -q "^grafana$"; then
    echo -e "${GREEN}Grafana:${NC}"
    echo "  URL: http://localhost:3000"
    echo "  Username: admin"
    echo "  Password: admin (change on first login)"
else
    echo -e "${RED}Grafana:${NC} Not running"
fi

# Prometheus
if docker ps --format "{{.Names}}" | grep -q "^prometheus$"; then
    echo -e "${GREEN}Prometheus:${NC}"
    echo "  URL: http://localhost:9090"
    echo "  Authentication: None"
else
    echo -e "${RED}Prometheus:${NC} Not running"
fi

# Uptime Kuma
if docker ps --format "{{.Names}}" | grep -q "^uptime-kuma$"; then
    echo -e "${GREEN}Uptime Kuma:${NC}"
    echo "  URL: http://localhost:3004"
    echo "  Initial Setup: Required on first access"
else
    echo -e "${RED}Uptime Kuma:${NC} Not running"
fi

echo ""
echo "🏠 Dashboard Services:"
echo "===================="

# Homarr
if docker ps --format "{{.Names}}" | grep -q "^homarr$"; then
    echo -e "${GREEN}Homarr:${NC}"
    echo "  URL: http://localhost:7575"
    echo "  Authentication: None (configure in settings)"
else
    echo -e "${RED}Homarr:${NC} Not running"
fi

# Homepage
if docker ps --format "{{.Names}}" | grep -q "^homepage$"; then
    echo -e "${GREEN}Homepage:${NC}"
    echo "  URL: http://localhost:3001"
    echo "  Configuration: ./homepage-configs/"
else
    echo -e "${RED}Homepage:${NC} Not running"
fi

# Portainer
if docker ps --format "{{.Names}}" | grep -q "^portainer$"; then
    echo -e "${GREEN}Portainer:${NC}"
    echo "  URL: http://localhost:9000"
    echo "  Initial Setup: Create admin account on first access"
else
    echo -e "${RED}Portainer:${NC} Not running"
fi

echo ""
echo "🔌 Service Integration Status:"
echo "============================="

# Check Prowlarr apps
if docker ps --format "{{.Names}}" | grep -q "^prowlarr$"; then
    PROWLARR_KEY=$(docker exec prowlarr grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
    if [ -n "$PROWLARR_KEY" ]; then
        echo "Prowlarr Applications:"
        apps=$(curl -s "http://localhost:9696/api/v1/applications" -H "X-Api-Key: $PROWLARR_KEY" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")
        if [ -n "$apps" ]; then
            echo "$apps" | while read -r app; do
                echo "  ✅ $app connected"
            done
        else
            echo "  No applications connected"
        fi
    fi
fi

echo ""
echo "💡 Quick Setup Commands:"
echo "======================="
echo "# Make all services accessible:"
echo "docker-compose -f docker-compose-demo.yml up -d"
echo ""
echo "# Configure all services automatically:"
echo "./scripts/auto-configure-all-services.sh"
echo ""
echo "# Check service health:"
echo "./scripts/health-check.sh"
echo ""

# Save to file
OUTPUT_FILE="media-server-credentials-$(date +%Y%m%d-%H%M%S).txt"
echo ""
echo "📄 Saving credentials to: $OUTPUT_FILE"
{
    echo "Media Server API Keys and Credentials"
    echo "Generated: $(date)"
    echo "====================================="
    echo ""
    $0 | sed 's/\x1b\[[0-9;]*m//g'  # Strip color codes
} > "$OUTPUT_FILE" 2>/dev/null

echo ""
echo "⚠️  Remember to keep these credentials secure!"