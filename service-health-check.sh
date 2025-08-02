#!/bin/bash

# Ultimate Media Server 2025 - Health Check Script
# Checks the status of all deployed services

echo "🔍 ULTIMATE MEDIA SERVER 2025 - HEALTH CHECK"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Services to check
services=(
    "jellyfin:8096:🎬 Jellyfin Media Server"
    "sonarr:8989:📺 Sonarr TV Shows"
    "radarr:7878:🎞️ Radarr Movies"
    "prowlarr:9696:🔍 Prowlarr Indexers"
    "bazarr:6767:📝 Bazarr Subtitles"
    "lidarr:8686:🎵 Lidarr Music"
    "qbittorrent:8080:📥 qBittorrent Downloads"
    "sabnzbd:8081:📦 SABnzbd Usenet"
    "overseerr:5055:🎯 Overseerr Requests"
    "homepage:3001:🏠 Homepage Dashboard"
    "portainer:9000:🐳 Portainer Management"
    "grafana:3000:📊 Grafana Monitoring"
    "prometheus:9090:📈 Prometheus Metrics"
    "tautulli:8181:📊 Tautulli Analytics"
)

echo "🌐 SERVICE ACCESSIBILITY CHECK:"
echo "--------------------------------"

for service in "${services[@]}"; do
    IFS=':' read -r container port name <<< "$service"
    
    # Check if container is running
    if docker ps --format "table {{.Names}}" | grep -q "^${container}$"; then
        container_status="${GREEN}✅ Running${NC}"
        
        # Check if port is accessible
        if curl -s -f "http://localhost:${port}" > /dev/null 2>&1 || \
           curl -s -f "http://localhost:${port}/ping" > /dev/null 2>&1 || \
           nc -z localhost ${port} 2>/dev/null; then
            port_status="${GREEN}✅ Accessible${NC}"
        else
            port_status="${YELLOW}⚠️ Starting up${NC}"
        fi
    else
        container_status="${RED}❌ Not Running${NC}"
        port_status="${RED}❌ Not Accessible${NC}"
    fi
    
    printf "%-30s Container: %-20s Port: %-20s\n" "$name" "$container_status" "$port_status"
done

echo ""
echo "🗄️ INFRASTRUCTURE STATUS:"
echo "-------------------------"

infra_services=(
    "postgres:5432:🗄️ PostgreSQL Database"
    "redis:6379:🔄 Redis Cache"
    "loki:3100:📋 Loki Logs"
)

for service in "${infra_services[@]}"; do
    IFS=':' read -r container port name <<< "$service"
    
    if docker ps --format "table {{.Names}}" | grep -q "^${container}$"; then
        status="${GREEN}✅ Running${NC}"
    else
        status="${RED}❌ Not Running${NC}"
    fi
    
    printf "%-30s Status: %-20s\n" "$name" "$status"
done

echo ""
echo "📊 SYSTEM OVERVIEW:"
echo "-------------------"

total_containers=$(docker ps -q | wc -l | tr -d ' ')
total_expected=17

echo "📦 Total Containers Running: ${total_containers}/${total_expected}"

if [ "$total_containers" -eq "$total_expected" ]; then
    echo -e "${GREEN}🎉 ALL SERVICES OPERATIONAL!${NC}"
else
    echo -e "${YELLOW}⚠️ Some services may still be starting up${NC}"
fi

# Disk usage check
echo ""
echo "💾 STORAGE STATUS:"
echo "------------------"
df -h /Users/morlock/fun/newmedia/media-data 2>/dev/null || echo "Media directory not yet created"

echo ""
echo "🔗 QUICK ACCESS URLS:"
echo "--------------------"
echo "🏠 Main Dashboard:    http://localhost:3001"
echo "🎬 Jellyfin:          http://localhost:8096"
echo "🎯 Overseerr:         http://localhost:5055"
echo "🐳 Portainer:         http://localhost:9000"
echo "📊 Grafana:           http://localhost:3000"
echo ""
echo "💡 TIP: Run './open-services.sh' to open all services in your browser"
echo ""