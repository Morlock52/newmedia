#!/bin/bash

# ARR Services Configuration Script
# This script configures Sonarr, Radarr, and Lidarr to use Prowlarr as indexer and qBittorrent as download client

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# API Keys (retrieved from config files)
PROWLARR_API_KEY="b7ef1468932940b2a4cf27ad980f1076"
SONARR_API_KEY="6e6bfac6e15d4f9a9d0e0d35ec0b8e23"
RADARR_API_KEY="7b74da952069425f9568ea361b001a12"
LIDARR_API_KEY="e8262da767e34a6b8ca7ca1e92384d96"

# Service URLs
PROWLARR_URL="http://localhost:9696"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"
LIDARR_URL="http://localhost:8686"
QBITTORRENT_URL="http://localhost:8080"

# Internal Docker network URLs
PROWLARR_INTERNAL="http://prowlarr:9696"
SONARR_INTERNAL="http://sonarr:8989"
RADARR_INTERNAL="http://radarr:7878"
LIDARR_INTERNAL="http://lidarr:8686"
QBITTORRENT_INTERNAL="http://gluetun:8080"  # qBittorrent runs through gluetun

echo -e "${GREEN}=== ARR Services Configuration Script ===${NC}"
echo ""

# Function to check if service is accessible
check_service() {
    local service_name=$1
    local service_url=$2
    local api_key=$3
    
    echo -n "Checking $service_name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Api-Key: $api_key" "$service_url/api/v3/system/status" 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED (HTTP $response)${NC}"
        return 1
    fi
}

# Check all services
echo -e "${YELLOW}Checking service availability...${NC}"
check_service "Prowlarr" "$PROWLARR_URL" "$PROWLARR_API_KEY"
check_service "Sonarr" "$SONARR_URL" "$SONARR_API_KEY"
check_service "Radarr" "$RADARR_URL" "$RADARR_API_KEY"
check_service "Lidarr" "$LIDARR_URL" "$LIDARR_API_KEY"
echo ""

# Function to add application to Prowlarr
add_app_to_prowlarr() {
    local app_name=$1
    local app_url=$2
    local app_api_key=$3
    local app_type=$4
    
    echo -n "Adding $app_name to Prowlarr... "
    
    # Check if app already exists
    existing=$(curl -s -H "X-Api-Key: $PROWLARR_API_KEY" "$PROWLARR_URL/api/v1/applications" | grep -c "\"name\":\"$app_name\"")
    
    if [ "$existing" -gt 0 ]; then
        echo -e "${YELLOW}Already exists${NC}"
        return
    fi
    
    # Add the application
    response=$(curl -s -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $PROWLARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "'"$app_name"'",
            "implementation": "'"$app_type"'",
            "configContract": "'"$app_type"'Settings",
            "fields": [
                {
                    "name": "baseUrl",
                    "value": "'"$app_url"'"
                },
                {
                    "name": "apiKey",
                    "value": "'"$app_api_key"'"
                },
                {
                    "name": "syncCategories",
                    "value": [2000, 5000, 6000, 7000, 8000]
                }
            ],
            "tags": [],
            "syncLevel": "fullSync"
        }' 2>/dev/null)
    
    if echo "$response" | grep -q '"id"'; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Response: $response"
    fi
}

# Add ARR applications to Prowlarr
echo -e "${YELLOW}Configuring Prowlarr applications...${NC}"
add_app_to_prowlarr "Sonarr" "$SONARR_INTERNAL" "$SONARR_API_KEY" "Sonarr"
add_app_to_prowlarr "Radarr" "$RADARR_INTERNAL" "$RADARR_API_KEY" "Radarr"
add_app_to_prowlarr "Lidarr" "$LIDARR_INTERNAL" "$LIDARR_API_KEY" "Lidarr"
echo ""

# Function to sync indexers from Prowlarr
sync_indexers() {
    local app_name=$1
    local app_url=$2
    
    echo -n "Syncing indexers for $app_name... "
    
    response=$(curl -s -X POST "$app_url/api/v1/indexer/testall" \
        -H "X-Api-Key: $PROWLARR_API_KEY" 2>/dev/null)
    
    echo -e "${GREEN}Triggered${NC}"
}

# Trigger indexer sync
echo -e "${YELLOW}Triggering indexer synchronization...${NC}"
sync_indexers "All apps" "$PROWLARR_URL"
echo ""

# Function to add qBittorrent as download client
add_qbittorrent_client() {
    local app_name=$1
    local app_url=$2
    local app_api_key=$3
    local category=$4
    
    echo -n "Adding qBittorrent to $app_name... "
    
    # Check if download client already exists
    existing=$(curl -s -H "X-Api-Key: $app_api_key" "$app_url/api/v3/downloadclient" | grep -c "\"name\":\"qBittorrent\"")
    
    if [ "$existing" -gt 0 ]; then
        echo -e "${YELLOW}Already exists${NC}"
        return
    fi
    
    # Add qBittorrent as download client
    response=$(curl -s -X POST "$app_url/api/v3/downloadclient" \
        -H "X-Api-Key: $app_api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "enable": true,
            "protocol": "torrent",
            "priority": 1,
            "removeCompletedDownloads": true,
            "removeFailedDownloads": true,
            "name": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "fields": [
                {
                    "name": "host",
                    "value": "gluetun"
                },
                {
                    "name": "port",
                    "value": 8080
                },
                {
                    "name": "urlBase",
                    "value": ""
                },
                {
                    "name": "username",
                    "value": "admin"
                },
                {
                    "name": "password",
                    "value": "adminadmin"
                },
                {
                    "name": "tvCategory",
                    "value": "'"$category"'"
                },
                {
                    "name": "movieCategory",
                    "value": "'"$category"'"
                },
                {
                    "name": "musicCategory",
                    "value": "'"$category"'"
                },
                {
                    "name": "recentMoviePriority",
                    "value": 0
                },
                {
                    "name": "olderMoviePriority",
                    "value": 0
                },
                {
                    "name": "recentTvPriority",
                    "value": 0
                },
                {
                    "name": "olderTvPriority",
                    "value": 0
                },
                {
                    "name": "initialState",
                    "value": 0
                },
                {
                    "name": "sequentialOrder",
                    "value": false
                },
                {
                    "name": "firstAndLast",
                    "value": false
                }
            ],
            "tags": []
        }' 2>/dev/null)
    
    if echo "$response" | grep -q '"id"'; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Response: $response"
    fi
}

# Add qBittorrent to each ARR service
echo -e "${YELLOW}Configuring download clients...${NC}"
add_qbittorrent_client "Sonarr" "$SONARR_URL" "$SONARR_API_KEY" "sonarr"
add_qbittorrent_client "Radarr" "$RADARR_URL" "$RADARR_API_KEY" "radarr"
add_qbittorrent_client "Lidarr" "$LIDARR_URL" "$LIDARR_API_KEY" "lidarr"
echo ""

# Function to configure root folders
configure_root_folder() {
    local app_name=$1
    local app_url=$2
    local app_api_key=$3
    local folder_path=$4
    
    echo -n "Configuring root folder for $app_name... "
    
    # Check if root folder already exists
    existing=$(curl -s -H "X-Api-Key: $app_api_key" "$app_url/api/v3/rootfolder" | grep -c "\"path\":\"$folder_path\"")
    
    if [ "$existing" -gt 0 ]; then
        echo -e "${YELLOW}Already exists${NC}"
        return
    fi
    
    # Add root folder
    response=$(curl -s -X POST "$app_url/api/v3/rootfolder" \
        -H "X-Api-Key: $app_api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "path": "'"$folder_path"'"
        }' 2>/dev/null)
    
    if echo "$response" | grep -q '"id"'; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Response: $response"
    fi
}

# Configure root folders
echo -e "${YELLOW}Configuring media folders...${NC}"
configure_root_folder "Sonarr" "$SONARR_URL" "$SONARR_API_KEY" "/media/tv"
configure_root_folder "Radarr" "$RADARR_URL" "$RADARR_API_KEY" "/media/movies"
configure_root_folder "Lidarr" "$LIDARR_URL" "$LIDARR_API_KEY" "/media/music"
echo ""

echo -e "${GREEN}=== Configuration Complete ===${NC}"
echo ""
echo "Summary:"
echo "- Prowlarr configured with Sonarr, Radarr, and Lidarr applications"
echo "- qBittorrent added as download client to all ARR services"
echo "- Root folders configured for media storage"
echo ""
echo "Next steps:"
echo "1. Add indexers to Prowlarr (Settings -> Indexers)"
echo "2. Configure quality profiles in each ARR service"
echo "3. Start adding media to your library!"
echo ""
echo "Service URLs:"
echo "- Prowlarr: $PROWLARR_URL"
echo "- Sonarr: $SONARR_URL"
echo "- Radarr: $RADARR_URL"
echo "- Lidarr: $LIDARR_URL"
echo "- qBittorrent: $QBITTORRENT_URL"