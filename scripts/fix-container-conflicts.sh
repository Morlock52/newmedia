#!/bin/bash

# ============================================================================
# Fix Container Conflicts Script
# ============================================================================
# This script checks for and resolves container name conflicts
# ============================================================================

set -e

echo "🔧 Container Conflict Resolution Script"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if container exists (running or stopped)
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Function to handle container conflict
handle_container_conflict() {
    local container_name=$1
    
    if container_exists "$container_name"; then
        if container_running "$container_name"; then
            echo -e "${GREEN}✅ $container_name is already running${NC}"
            # Get container info
            docker ps --filter "name=^${container_name}$" --format "table {{.ID}}\t{{.Status}}\t{{.Ports}}"
        else
            echo -e "${YELLOW}⚠️  $container_name exists but is stopped${NC}"
            echo -e "${BLUE}   Starting existing container...${NC}"
            docker start "$container_name"
            sleep 2
            if container_running "$container_name"; then
                echo -e "${GREEN}✅ $container_name started successfully${NC}"
            else
                echo -e "${RED}❌ Failed to start $container_name${NC}"
                echo -e "${YELLOW}   Container logs:${NC}"
                docker logs --tail 20 "$container_name" 2>&1
            fi
        fi
    else
        echo -e "${BLUE}ℹ️  $container_name does not exist yet${NC}"
    fi
}

# Check for common container conflicts
echo -e "${BLUE}🔍 Checking for container conflicts...${NC}"
echo ""

# List of containers that commonly have conflicts
CONTAINERS=(
    "homarr"
    "homepage"
    "prowlarr"
    "sonarr"
    "radarr"
    "jellyfin"
    "qbittorrent"
    "overseerr"
    "tautulli"
    "bazarr"
)

for container in "${CONTAINERS[@]}"; do
    handle_container_conflict "$container"
    echo ""
done

# Summary
echo -e "${CYAN}📊 Container Status Summary:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(homarr|homepage|prowlarr|sonarr|radarr|jellyfin|qbittorrent)" | head -20

echo ""
echo -e "${GREEN}✅ Container conflict check complete!${NC}"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "   - If a container keeps failing, check logs: docker logs <container_name>"
echo "   - To remove a broken container: docker rm <container_name>"
echo "   - To recreate from compose: docker-compose up -d <service_name>"
echo ""