#!/bin/bash

# ARR Services Configuration Verification Script
# This script checks the configuration status of all ARR services

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# API Keys
PROWLARR_API_KEY="b7ef1468932940b2a4cf27ad980f1076"
SONARR_API_KEY="6e6bfac6e15d4f9a9d0e0d35ec0b8e23"
RADARR_API_KEY="7b74da952069425f9568ea361b001a12"
LIDARR_API_KEY="e8262da767e34a6b8ca7ca1e92384d96"

# URLs
PROWLARR_URL="http://localhost:9696"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"
LIDARR_URL="http://localhost:8686"

echo -e "${BLUE}=== ARR Services Configuration Status ===${NC}"
echo ""

# Function to check service status
check_service_status() {
    local name=$1
    local url=$2
    local api_key=$3
    
    echo -e "${YELLOW}$name:${NC}"
    
    # Check if service is accessible
    status=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Api-Key: $api_key" "$url/api/v3/system/status" 2>/dev/null || echo "000")
    
    if [ "$status" = "200" ]; then
        echo -e "  Service Status: ${GREEN}Online${NC}"
        
        # Get version info
        version=$(curl -s -H "X-Api-Key: $api_key" "$url/api/v3/system/status" 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        echo -e "  Version: ${GREEN}$version${NC}"
    else
        echo -e "  Service Status: ${RED}Offline (HTTP $status)${NC}"
        return
    fi
}

# Check Prowlarr status and configuration
echo -e "${YELLOW}Prowlarr:${NC}"
status=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Api-Key: $PROWLARR_API_KEY" "$PROWLARR_URL/api/v1/system/status" 2>/dev/null || echo "000")

if [ "$status" = "200" ]; then
    echo -e "  Service Status: ${GREEN}Online${NC}"
    
    # Get version
    version=$(curl -s -H "X-Api-Key: $PROWLARR_API_KEY" "$PROWLARR_URL/api/v1/system/status" 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo -e "  Version: ${GREEN}$version${NC}"
    
    # Check applications
    apps=$(curl -s -H "X-Api-Key: $PROWLARR_API_KEY" "$PROWLARR_URL/api/v1/applications" 2>/dev/null)
    app_count=$(echo "$apps" | grep -o '"name"' | wc -l)
    echo -e "  Configured Apps: ${GREEN}$app_count${NC}"
    
    # Check indexers
    indexers=$(curl -s -H "X-Api-Key: $PROWLARR_API_KEY" "$PROWLARR_URL/api/v1/indexer" 2>/dev/null)
    indexer_count=$(echo "$indexers" | grep -o '"name"' | wc -l)
    echo -e "  Configured Indexers: ${GREEN}$indexer_count${NC}"
else
    echo -e "  Service Status: ${RED}Offline (HTTP $status)${NC}"
fi
echo ""

# Check other services
check_service_status "Sonarr" "$SONARR_URL" "$SONARR_API_KEY"

# Check Sonarr specific config
if [ "$status" = "200" ]; then
    # Check download clients
    clients=$(curl -s -H "X-Api-Key: $SONARR_API_KEY" "$SONARR_URL/api/v3/downloadclient" 2>/dev/null)
    client_count=$(echo "$clients" | grep -o '"name"' | wc -l)
    echo -e "  Download Clients: ${GREEN}$client_count${NC}"
    
    # Check indexers
    indexers=$(curl -s -H "X-Api-Key: $SONARR_API_KEY" "$SONARR_URL/api/v3/indexer" 2>/dev/null)
    indexer_count=$(echo "$indexers" | grep -o '"name"' | wc -l)
    echo -e "  Indexers: ${GREEN}$indexer_count${NC}"
    
    # Check root folders
    folders=$(curl -s -H "X-Api-Key: $SONARR_API_KEY" "$SONARR_URL/api/v3/rootfolder" 2>/dev/null)
    folder_count=$(echo "$folders" | grep -o '"path"' | wc -l)
    echo -e "  Root Folders: ${GREEN}$folder_count${NC}"
fi
echo ""

check_service_status "Radarr" "$RADARR_URL" "$RADARR_API_KEY"

# Check Radarr specific config
if [ "$status" = "200" ]; then
    clients=$(curl -s -H "X-Api-Key: $RADARR_API_KEY" "$RADARR_URL/api/v3/downloadclient" 2>/dev/null)
    client_count=$(echo "$clients" | grep -o '"name"' | wc -l)
    echo -e "  Download Clients: ${GREEN}$client_count${NC}"
    
    indexers=$(curl -s -H "X-Api-Key: $RADARR_API_KEY" "$RADARR_URL/api/v3/indexer" 2>/dev/null)
    indexer_count=$(echo "$indexers" | grep -o '"name"' | wc -l)
    echo -e "  Indexers: ${GREEN}$indexer_count${NC}"
    
    folders=$(curl -s -H "X-Api-Key: $RADARR_API_KEY" "$RADARR_URL/api/v3/rootfolder" 2>/dev/null)
    folder_count=$(echo "$folders" | grep -o '"path"' | wc -l)
    echo -e "  Root Folders: ${GREEN}$folder_count${NC}"
fi
echo ""

check_service_status "Lidarr" "$LIDARR_URL" "$LIDARR_API_KEY"

# Check Lidarr specific config
if [ "$status" = "200" ]; then
    clients=$(curl -s -H "X-Api-Key: $LIDARR_API_KEY" "$LIDARR_URL/api/v1/downloadclient" 2>/dev/null)
    client_count=$(echo "$clients" | grep -o '"name"' | wc -l)
    echo -e "  Download Clients: ${GREEN}$client_count${NC}"
    
    indexers=$(curl -s -H "X-Api-Key: $LIDARR_API_KEY" "$LIDARR_URL/api/v1/indexer" 2>/dev/null)
    indexer_count=$(echo "$indexers" | grep -o '"name"' | wc -l)
    echo -e "  Indexers: ${GREEN}$indexer_count${NC}"
    
    folders=$(curl -s -H "X-Api-Key: $LIDARR_API_KEY" "$LIDARR_URL/api/v1/rootfolder" 2>/dev/null)
    folder_count=$(echo "$folders" | grep -o '"path"' | wc -l)
    echo -e "  Root Folders: ${GREEN}$folder_count${NC}"
fi
echo ""

# Check qBittorrent
echo -e "${YELLOW}qBittorrent:${NC}"
qbt_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080" 2>/dev/null || echo "000")
if [ "$qbt_status" = "200" ] || [ "$qbt_status" = "401" ]; then
    echo -e "  Service Status: ${GREEN}Online${NC}"
    echo -e "  WebUI URL: ${GREEN}http://localhost:8080${NC}"
    echo -e "  Default Credentials: ${YELLOW}admin / adminadmin${NC}"
else
    echo -e "  Service Status: ${RED}Offline (HTTP $qbt_status)${NC}"
fi
echo ""

echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo ""

# Summary recommendations
echo -e "${YELLOW}Recommendations:${NC}"

# Check if indexers need to be added
if [ "$indexer_count" = "0" ] 2>/dev/null; then
    echo -e "  ${RED}•${NC} Add indexers to Prowlarr for content discovery"
fi

# Check if apps are connected
if [ "$app_count" = "0" ] 2>/dev/null; then
    echo -e "  ${RED}•${NC} Run the configuration script to connect ARR services to Prowlarr"
fi

# Security reminder
echo -e "  ${YELLOW}•${NC} Change qBittorrent default password for security"
echo -e "  ${YELLOW}•${NC} Consider enabling authentication on ARR services"

echo ""
echo -e "${GREEN}Run './scripts/configure-arr-services.sh' to automatically configure all services${NC}"