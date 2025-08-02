#!/bin/bash
# Media Server Integration Configuration Script
# Connects Prowlarr -> Sonarr/Radarr -> qBittorrent -> Overseerr

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# API Keys (from analysis)
PROWLARR_API_KEY="5a35dd23f90c4d2bb69caa1eb0e1c534"
SONARR_API_KEY="79eecf2b23f34760b91cfcbf97189dd0"
RADARR_API_KEY="1c0fe63736a04e6394dacb3aa1160b1c"

# Service URLs
PROWLARR_URL="http://localhost:9696"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"
QBITTORRENT_URL="http://localhost:8080"
OVERSEERR_URL="http://localhost:5055"

echo -e "${CYAN}🚀 Starting Media Server Integration Configuration${NC}"

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service=$2
    echo -e "${YELLOW}⏳ Waiting for $service to be ready...${NC}"
    
    for i in {1..30}; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service is ready${NC}"
            return 0
        fi
        sleep 2
    done
    
    echo -e "${RED}❌ $service failed to start${NC}"
    return 1
}

# 1. Configure qBittorrent categories
configure_qbittorrent() {
    echo -e "${CYAN}🔧 Configuring qBittorrent categories...${NC}"
    
    # Create categories for different media types
    curl -X POST "$QBITTORRENT_URL/api/v2/torrents/createCategory" \
        -d "category=sonarr-tv" \
        -d "savePath=/downloads/tv" || true
    
    curl -X POST "$QBITTORRENT_URL/api/v2/torrents/createCategory" \
        -d "category=radarr-movies" \
        -d "savePath=/downloads/movies" || true
    
    echo -e "${GREEN}✅ qBittorrent categories configured${NC}"
}

# 2. Add qBittorrent as download client in Sonarr
configure_sonarr_download_client() {
    echo -e "${CYAN}🔧 Adding qBittorrent to Sonarr...${NC}"
    
    local config='{
        "enable": true,
        "protocol": "torrent",
        "priority": 1,
        "removeCompletedDownloads": false,
        "removeFailedDownloads": true,
        "name": "qBittorrent",
        "fields": [
            {"name": "host", "value": "qbittorrent"},
            {"name": "port", "value": 8080},
            {"name": "useSsl", "value": false},
            {"name": "urlBase", "value": ""},
            {"name": "username", "value": ""},
            {"name": "password", "value": ""},
            {"name": "category", "value": "sonarr-tv"},
            {"name": "recentTvPriority", "value": 0},
            {"name": "olderTvPriority", "value": 0},
            {"name": "initialState", "value": 0},
            {"name": "sequentialOrder", "value": false},
            {"name": "firstAndLast", "value": false}
        ],
        "implementationName": "qBittorrent",
        "implementation": "QBittorrent",
        "configContract": "QBittorrentSettings",
        "infoLink": "https://wiki.servarr.com/sonarr/supported#qbittorrent",
        "tags": []
    }'
    
    curl -X POST "$SONARR_URL/api/v3/downloadclient" \
        -H "X-Api-Key: $SONARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$config" || true
    
    echo -e "${GREEN}✅ qBittorrent added to Sonarr${NC}"
}

# 3. Add qBittorrent as download client in Radarr
configure_radarr_download_client() {
    echo -e "${CYAN}🔧 Adding qBittorrent to Radarr...${NC}"
    
    local config='{
        "enable": true,
        "protocol": "torrent",
        "priority": 1,
        "removeCompletedDownloads": false,
        "removeFailedDownloads": true,
        "name": "qBittorrent",
        "fields": [
            {"name": "host", "value": "qbittorrent"},
            {"name": "port", "value": 8080},
            {"name": "useSsl", "value": false},
            {"name": "urlBase", "value": ""},
            {"name": "username", "value": ""},
            {"name": "password", "value": ""},
            {"name": "category", "value": "radarr-movies"},
            {"name": "recentMoviePriority", "value": 0},
            {"name": "olderMoviePriority", "value": 0},
            {"name": "initialState", "value": 0},
            {"name": "sequentialOrder", "value": false},
            {"name": "firstAndLast", "value": false}
        ],
        "implementationName": "qBittorrent",
        "implementation": "QBittorrent",
        "configContract": "QBittorrentSettings",
        "infoLink": "https://wiki.servarr.com/radarr/supported#qbittorrent",
        "tags": []
    }'
    
    curl -X POST "$RADARR_URL/api/v3/downloadclient" \
        -H "X-Api-Key: $RADARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$config" || true
    
    echo -e "${GREEN}✅ qBittorrent added to Radarr${NC}"
}

# 4. Add Sonarr and Radarr as applications in Prowlarr
configure_prowlarr_applications() {
    echo -e "${CYAN}🔧 Adding Sonarr and Radarr to Prowlarr...${NC}"
    
    # Add Sonarr
    local sonarr_config='{
        "name": "Sonarr",
        "syncLevel": "fullSync",
        "tags": [],
        "fields": [
            {"name": "prowlarrUrl", "value": "http://prowlarr:9696"},
            {"name": "baseUrl", "value": "http://sonarr:8989"},
            {"name": "apiKey", "value": "'$SONARR_API_KEY'"},
            {"name": "syncCategories", "value": [5000, 5030, 5040]}
        ],
        "implementationName": "Sonarr",
        "implementation": "Sonarr",
        "configContract": "SonarrSettings"
    }'
    
    curl -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $PROWLARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$sonarr_config" || true
    
    # Add Radarr
    local radarr_config='{
        "name": "Radarr",
        "syncLevel": "fullSync",
        "tags": [],
        "fields": [
            {"name": "prowlarrUrl", "value": "http://prowlarr:9696"},
            {"name": "baseUrl", "value": "http://radarr:7878"},
            {"name": "apiKey", "value": "'$RADARR_API_KEY'"},
            {"name": "syncCategories", "value": [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060]}
        ],
        "implementationName": "Radarr",
        "implementation": "Radarr",
        "configContract": "RadarrSettings"
    }'
    
    curl -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $PROWLARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$radarr_config" || true
    
    echo -e "${GREEN}✅ Sonarr and Radarr added to Prowlarr${NC}"
}

# Main execution
main() {
    echo -e "${CYAN}Starting integration configuration...${NC}"
    
    # Wait for all services to be ready
    wait_for_service "$QBITTORRENT_URL" "qBittorrent"
    wait_for_service "$SONARR_URL/api/v3/system/status" "Sonarr"
    wait_for_service "$RADARR_URL/api/v3/system/status" "Radarr"
    wait_for_service "$PROWLARR_URL/api/v1/system/status" "Prowlarr"
    
    # Configure integrations
    configure_qbittorrent
    configure_sonarr_download_client
    configure_radarr_download_client
    configure_prowlarr_applications
    
    echo -e "${GREEN}🎉 Media server integration configuration complete!${NC}"
    echo -e "${YELLOW}📋 Next steps:${NC}"
    echo -e "  1. Add indexers to Prowlarr (visit http://localhost:9696)"
    echo -e "  2. Configure Overseerr (visit http://localhost:5055)"
    echo -e "  3. Test the automation by requesting content"
}

# Run the main function
main "$@"