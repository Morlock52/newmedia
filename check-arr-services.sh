#!/bin/bash

echo "Checking *ARR Services Status..."
echo "================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check service
check_service() {
    local name=$1
    local port=$2
    local api_key=$3
    
    if curl -s -f "http://localhost:$port/ping" > /dev/null; then
        echo -e "${GREEN}✓${NC} $name (port $port) - ONLINE"
        
        # Get system info
        info=$(curl -s -H "X-Api-Key: $api_key" "http://localhost:$port/api/v3/system/status" 2>/dev/null || echo "{}")
        if [ "$info" != "{}" ]; then
            version=$(echo "$info" | jq -r '.version // "unknown"' 2>/dev/null || echo "unknown")
            echo "  Version: $version"
        fi
    else
        echo -e "${RED}✗${NC} $name (port $port) - OFFLINE"
    fi
}

# Check each service
check_service "Prowlarr" 9696 "c272eec1ce1447f1b119daade4b0e268"
check_service "Sonarr" 8989 "becf19261b1a4ed0b4f92de90eaf3015"
check_service "Radarr" 7878 "f0acaf0200034ee69215482f212cda5a"
check_service "Lidarr" 8686 "47b87e7aec5646c898d60b728f6126e0"

echo ""
echo "Inter-Service Connections:"
echo "========================="

# Check Prowlarr connections
apps=$(curl -s -H "X-Api-Key: c272eec1ce1447f1b119daade4b0e268" http://localhost:9696/api/v1/applications 2>/dev/null)
if [ -n "$apps" ]; then
    count=$(echo "$apps" | jq 'length' 2>/dev/null || echo 0)
    echo -e "${GREEN}✓${NC} Prowlarr has $count connected applications"
    echo "$apps" | jq -r '.[] | "  - " + .name' 2>/dev/null || echo "  Could not parse applications"
else
    echo -e "${RED}✗${NC} Could not check Prowlarr connections"
fi

echo ""
echo "Docker Container Status:"
echo "======================="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAME|sonarr|radarr|prowlarr|lidarr)" || echo "No containers found"