#!/bin/bash

# Configuration Status Check Script
# This script checks the status of all media server services and their configurations

echo "📊 Media Server Configuration Status Report"
echo "=========================================="
echo "Generated: $(date)"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Docker status
echo "🐳 Docker Status:"
if docker info > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅ Docker is running${NC}"
    echo "  Version: $(docker version --format '{{.Server.Version}}')"
else
    echo -e "  ${RED}❌ Docker is not running${NC}"
    exit 1
fi

echo ""
echo "📦 Running Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

echo ""
echo "🔍 Service Health Check:"
echo "========================"

# Service ports to check
declare -A services=(
    ["Prowlarr"]=9696
    ["Sonarr"]=8989
    ["Radarr"]=7878
    ["Lidarr"]=8686
    ["Jellyfin"]=8096
    ["Plex"]=32400
    ["qBittorrent"]=8090
    ["Transmission"]=9091
    ["Homarr"]=7575
    ["Homepage"]=3001
    ["Grafana"]=3000
    ["Prometheus"]=9090
)

# Check each service
for service in "${!services[@]}"; do
    port=${services[$service]}
    if curl -s "http://localhost:$port" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ $service${NC} - Running on port $port"
        
        # Try to get API key for ARR services
        if [[ "$service" =~ ^(Prowlarr|Sonarr|Radarr|Lidarr)$ ]]; then
            container_name=$(echo "$service" | tr '[:upper:]' '[:lower:]')
            if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
                api_key=$(docker exec "$container_name" grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
                if [ -n "$api_key" ]; then
                    echo "     API Key: ${api_key:0:8}..."
                fi
            fi
        fi
    else
        echo -e "  ${RED}❌ $service${NC} - Not accessible on port $port"
    fi
done

echo ""
echo "📁 Directory Structure:"
echo "======================"
# Check for important directories
dirs=(
    "./media/movies"
    "./media/tv"
    "./media/music"
    "./media/downloads"
    "./config"
    "./homarr-configs"
    "./homepage-configs"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "  ${GREEN}✅${NC} $dir exists"
    else
        echo -e "  ${YELLOW}⚠️${NC}  $dir missing"
    fi
done

echo ""
echo "🔗 Prowlarr Indexers:"
echo "===================="
if docker ps --format "{{.Names}}" | grep -q "^prowlarr$"; then
    PROWLARR_KEY=$(docker exec prowlarr grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
    if [ -n "$PROWLARR_KEY" ]; then
        indexers=$(curl -s "http://localhost:9696/api/v1/indexer" -H "X-Api-Key: $PROWLARR_KEY" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")
        if [ -n "$indexers" ]; then
            echo "$indexers" | while read -r indexer; do
                echo "  - $indexer"
            done
        else
            echo "  No indexers configured"
        fi
    else
        echo "  Unable to retrieve Prowlarr API key"
    fi
else
    echo "  Prowlarr not running"
fi

echo ""
echo "🔗 ARR App Connections:"
echo "======================"
# Check if ARR apps are connected to Prowlarr
for app in sonarr radarr lidarr; do
    if docker ps --format "{{.Names}}" | grep -q "^${app}$"; then
        echo -e "  ${BLUE}$app:${NC}"
        # Check for download clients
        API_KEY=$(docker exec "$app" grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
        if [ -n "$API_KEY" ]; then
            # Check indexers
            indexer_count=$(curl -s "http://localhost:${services[${app^}]}/api/v3/indexer" -H "X-Api-Key: $API_KEY" 2>/dev/null | grep -o '"name"' | wc -l || echo "0")
            echo "    Indexers: $indexer_count configured"
            
            # Check download clients
            dl_count=$(curl -s "http://localhost:${services[${app^}]}/api/v3/downloadclient" -H "X-Api-Key: $API_KEY" 2>/dev/null | grep -o '"name"' | wc -l || echo "0")
            echo "    Download Clients: $dl_count configured"
        fi
    fi
done

echo ""
echo "💾 Volume Usage:"
echo "==============="
docker system df -v | grep -E "(VOLUME NAME|media|config|prowlarr|sonarr|radarr|jellyfin|qbittorrent)" | head -20

echo ""
echo "📝 Configuration Files:"
echo "====================="
config_files=(
    "./prometheus.yml"
    "./docker-compose-demo.yml"
    "./docker-compose.yml"
    ".env"
)

for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅${NC} $file exists ($(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null) bytes)"
    else
        echo -e "  ${YELLOW}⚠️${NC}  $file missing"
    fi
done

echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. If services are not running, execute: ./run-auto-config.sh"
echo "2. Access Homarr dashboard at: http://localhost:7575"
echo "3. Complete Jellyfin setup at: http://localhost:8096"
echo "4. Check Prowlarr indexers at: http://localhost:9696"
echo ""