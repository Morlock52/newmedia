#!/bin/bash

# Health Check Script for Ultimate Media Server
# This script checks the status of all services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check service health
check_service() {
    local service_name=$1
    local port=$2
    local endpoint=${3:-/}
    
    if curl -f -s -o /dev/null "http://localhost:${port}${endpoint}"; then
        echo -e "${GREEN}✓${NC} ${service_name} is running on port ${port}"
        return 0
    else
        echo -e "${RED}✗${NC} ${service_name} is not responding on port ${port}"
        return 1
    fi
}

# Header
echo -e "${BLUE}========================================"
echo -e "Ultimate Media Server Health Check"
echo -e "========================================${NC}"
echo

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running!${NC}"
    exit 1
fi

# Check Docker Compose services
echo -e "${BLUE}Checking Docker services...${NC}"
docker compose -f docker-compose-ultimate.yml ps --format "table {{.Service}}\t{{.Status}}"
echo

# Check individual service endpoints
echo -e "${BLUE}Checking service endpoints...${NC}"

# Media Servers
echo -e "${YELLOW}Media Servers:${NC}"
check_service "Homepage Dashboard" 3000
check_service "Jellyfin" 8096 "/health"
check_service "Navidrome" 4533 "/ping"
check_service "AudioBookshelf" 13378 "/ping"
check_service "Calibre-Web" 8083
check_service "Kavita" 5001 "/api/health"
check_service "Immich" 2283 "/server-info/ping"
echo

# Download Clients
echo -e "${YELLOW}Download Clients:${NC}"
check_service "qBittorrent" 8081
check_service "SABnzbd" 8082 "/sabnzbd/api?mode=version"
echo

# Media Management
echo -e "${YELLOW}Media Management:${NC}"
check_service "Sonarr" 8989 "/ping"
check_service "Radarr" 7878 "/ping"
check_service "Lidarr" 8686 "/ping"
check_service "Readarr" 8787 "/ping"
check_service "Prowlarr" 9696 "/ping"
check_service "Bazarr" 6767 "/api/system/status"
echo

# Request & Tools
echo -e "${YELLOW}Request & Tools:${NC}"
check_service "Overseerr" 5055 "/api/v1/status"
check_service "FileFlows" 5276 "/api/v1/status"
check_service "Podgrab" 8084
echo

# Infrastructure
echo -e "${YELLOW}Infrastructure:${NC}"
check_service "Portainer" 9000 "/api/status"
check_service "Tautulli" 8181 "/status"
check_service "FileBrowser" 8085 "/health"
echo

# Monitoring
echo -e "${YELLOW}Monitoring:${NC}"
check_service "Prometheus" 9090 "/-/healthy"
check_service "Grafana" 3001 "/api/health"
echo

# Check disk usage
echo -e "${BLUE}Disk Usage:${NC}"
df -h | grep -E "Filesystem|/Users/morlock/fun/newmedia" || df -h | head -n 1 && df -h | grep -v "^/dev/loop"
echo

# Check memory usage
echo -e "${BLUE}Memory Usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -20
echo

# Summary
echo -e "${BLUE}========================================"
echo -e "Health Check Complete"
echo -e "========================================${NC}"
echo
echo -e "${YELLOW}Tips:${NC}"
echo "- If services are not responding, check logs: docker compose -f docker-compose-ultimate.yml logs [service]"
echo "- To restart a service: docker compose -f docker-compose-ultimate.yml restart [service]"
echo "- To see real-time logs: docker compose -f docker-compose-ultimate.yml logs -f"