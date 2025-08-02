#!/bin/bash

# Quick Status Check for Ultimate Media Server 2025
# Provides rapid health assessment and key metrics

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}⚡ Ultimate Media Server 2025 - Quick Status Check${NC}"
echo "=================================================="
echo ""

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

PLEX_VOLUME="${MEDIA_PATH:-/Volumes/Plex}"

# Quick checks
echo -e "${BLUE}📊 System Overview${NC}"
echo "   Media Volume: $PLEX_VOLUME"
echo "   Timestamp: $(date)"
echo ""

# Container Status
echo -e "${BLUE}🐳 Container Status${NC}"
CORE_SERVICES=("jellyfin" "sonarr" "radarr" "prowlarr" "overseerr")
RUNNING=0
TOTAL=${#CORE_SERVICES[@]}

for service in "${CORE_SERVICES[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^$service$"; then
        echo -e "   ${GREEN}✅ $service [RUNNING]${NC}"
        ((RUNNING++))
    else
        echo -e "   ${RED}❌ $service [STOPPED]${NC}"
    fi
done

echo "   Status: $RUNNING/$TOTAL services running"
echo ""

# Service Connectivity (quick test)
echo -e "${BLUE}🌐 Service Connectivity${NC}"
PORTS=("8096:Jellyfin" "8989:Sonarr" "7878:Radarr" "9696:Prowlarr" "5055:Overseerr")
ONLINE=0

for port_info in "${PORTS[@]}"; do
    IFS=':' read -r port service <<< "$port_info"
    if nc -z localhost $port 2>/dev/null; then
        echo -e "   ${GREEN}✅ $service (port $port)${NC}"
        ((ONLINE++))
    else
        echo -e "   ${RED}❌ $service (port $port)${NC}"
    fi
done

echo "   Connectivity: $ONLINE/${#PORTS[@]} services online"
echo ""

# Volume Status
echo -e "${BLUE}📂 Volume Status${NC}"
if [ -d "$PLEX_VOLUME" ]; then
    if [ -w "$PLEX_VOLUME" ]; then
        echo -e "   ${GREEN}✅ $PLEX_VOLUME [READ/WRITE]${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $PLEX_VOLUME [READ ONLY]${NC}"
    fi
    
    # Quick disk usage
    if command -v df >/dev/null; then
        USAGE=$(df -h "$PLEX_VOLUME" | tail -1 | awk '{print $5}')
        echo "   Disk Usage: $USAGE"
    fi
else
    echo -e "   ${RED}❌ $PLEX_VOLUME [NOT ACCESSIBLE]${NC}"
fi

echo ""

# Quick Resource Check
echo -e "${BLUE}💻 Resource Usage${NC}"
CONTAINERS=$(docker ps -q | wc -l | tr -d ' ')
echo "   Docker Containers: $CONTAINERS running"

if command -v ps >/dev/null; then
    CPU_USAGE=$(ps aux | awk '{sum += $3} END {print sum"%"}' 2>/dev/null || echo "N/A")
    echo "   CPU Usage: $CPU_USAGE"
fi

echo ""

# Overall Health Assessment
echo -e "${BLUE}🎯 Overall Health${NC}"
if [ $RUNNING -eq $TOTAL ] && [ $ONLINE -eq ${#PORTS[@]} ] && [ -w "$PLEX_VOLUME" ]; then
    echo -e "   ${GREEN}🎉 EXCELLENT - All systems operational${NC}"
elif [ $RUNNING -gt 3 ] && [ $ONLINE -gt 3 ]; then
    echo -e "   ${YELLOW}⚠️  GOOD - Minor issues detected${NC}"
else
    echo -e "   ${RED}❌ CRITICAL - Multiple systems down${NC}"
fi

echo ""

# Quick Access URLs
echo -e "${BLUE}🔗 Quick Access${NC}"
echo "   • Jellyfin:  http://localhost:8096"
echo "   • Sonarr:    http://localhost:8989"
echo "   • Radarr:    http://localhost:7878"
echo "   • Prowlarr:  http://localhost:9696"
echo "   • Overseerr: http://localhost:5055"

echo ""
echo -e "${PURPLE}⚡ Quick check complete!${NC}"
echo "   For detailed analysis: ./verify-deployment.sh"