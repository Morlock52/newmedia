#!/bin/bash
# Overseerr Configuration Script
# Connects Overseerr to Jellyfin, Sonarr, and Radarr

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# API Keys
SONARR_API_KEY="79eecf2b23f34760b91cfcbf97189dd0"
RADARR_API_KEY="1c0fe63736a04e6394dacb3aa1160b1c"

# Service URLs
OVERSEERR_URL="http://localhost:5055"
JELLYFIN_URL="http://localhost:8096"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"

echo -e "${CYAN}🎬 Configuring Overseerr...${NC}"

# Wait for Overseerr to be ready
wait_for_overseerr() {
    echo -e "${YELLOW}⏳ Waiting for Overseerr to be ready...${NC}"
    
    for i in {1..30}; do
        if curl -s -f "$OVERSEERR_URL/api/v1/status" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Overseerr is ready${NC}"
            return 0
        fi
        sleep 2
    done
    
    echo -e "${RED}❌ Overseerr failed to start${NC}"
    exit 1
}

# Initialize Overseerr
initialize_overseerr() {
    echo -e "${CYAN}🔧 Initializing Overseerr...${NC}"
    
    # Check if already initialized
    local status=$(curl -s "$OVERSEERR_URL/api/v1/status" | jq -r '.initialized // false')
    
    if [ "$status" = "true" ]; then
        echo -e "${YELLOW}⚠️  Overseerr already initialized${NC}"
        return 0
    fi
    
    # Initialize with basic settings
    local init_config='{
        "applicationTitle": "Media Requests",
        "applicationUrl": "http://localhost:5055",
        "trustProxy": false,
        "csrfProtection": false,
        "cacheImages": false,
        "defaultPermissions": 2,
        "hideAvailable": false,
        "localLogin": true,
        "newPlexLogin": true,
        "region": "US",
        "originalLanguage": "en",
        "trustProxyEnabled": false
    }'
    
    curl -X POST "$OVERSEERR_URL/api/v1/settings/initialize" \
        -H "Content-Type: application/json" \
        -d "$init_config" || true
    
    echo -e "${GREEN}✅ Overseerr initialized${NC}"
}

# Configure Jellyfin connection
configure_jellyfin() {
    echo -e "${CYAN}🔧 Configuring Jellyfin connection...${NC}"
    
    # Get Jellyfin server info
    local jellyfin_info=$(curl -s "$JELLYFIN_URL/System/Info/Public" 2>/dev/null || echo '{}')
    local server_id=$(echo "$jellyfin_info" | jq -r '.Id // "jellyfin-server"')
    
    local jellyfin_config='{
        "name": "Jellyfin",
        "hostname": "jellyfin",
        "port": 8096,
        "authToken": "",
        "useSsl": false,
        "baseUrl": "",
        "externalUrl": "http://localhost:8096",
        "is4k": false,
        "isDefault": true,
        "activeProfileId": 1,
        "activeDirectory": "",
        "selectedLibraries": []
    }'
    
    curl -X POST "$OVERSEERR_URL/api/v1/settings/jellyfin" \
        -H "Content-Type: application/json" \
        -d "$jellyfin_config" || true
    
    echo -e "${GREEN}✅ Jellyfin configured${NC}"
}

# Configure Sonarr connection
configure_sonarr() {
    echo -e "${CYAN}🔧 Configuring Sonarr connection...${NC}"
    
    local sonarr_config='{
        "name": "Sonarr",
        "hostname": "sonarr",
        "port": 8989,
        "apiKey": "'$SONARR_API_KEY'",
        "useSsl": false,
        "baseUrl": "",
        "activeProfileId": 1,
        "activeRootFolder": "/tv",
        "activeLanguageProfileId": 1,
        "activeAnimeProfileId": null,
        "activeAnimeRootFolder": null,
        "activeAnimeLanguageProfileId": null,
        "externalUrl": "http://localhost:8989",
        "syncEnabled": true,
        "preventSearch": false,
        "tagRequests": true,
        "isDefault": true,
        "is4k": false,
        "enableSeasonFolders": true
    }'
    
    curl -X POST "$OVERSEERR_URL/api/v1/settings/sonarr" \
        -H "Content-Type: application/json" \
        -d "$sonarr_config" || true
    
    echo -e "${GREEN}✅ Sonarr configured${NC}"
}

# Configure Radarr connection
configure_radarr() {
    echo -e "${CYAN}🔧 Configuring Radarr connection...${NC}"
    
    local radarr_config='{
        "name": "Radarr",
        "hostname": "radarr",
        "port": 7878,
        "apiKey": "'$RADARR_API_KEY'",
        "useSsl": false,
        "baseUrl": "",
        "activeProfileId": 1,
        "activeRootFolder": "/movies",
        "externalUrl": "http://localhost:7878",
        "syncEnabled": true,
        "preventSearch": false,
        "tagRequests": true,
        "isDefault": true,
        "is4k": false,
        "minimumAvailability": "released"
    }'
    
    curl -X POST "$OVERSEERR_URL/api/v1/settings/radarr" \
        -H "Content-Type: application/json" \
        -d "$radarr_config" || true
    
    echo -e "${GREEN}✅ Radarr configured${NC}"
}

# Main execution
main() {
    wait_for_overseerr
    initialize_overseerr
    sleep 5  # Wait for initialization to complete
    configure_jellyfin
    configure_sonarr
    configure_radarr
    
    echo -e "${GREEN}🎉 Overseerr configuration complete!${NC}"
    echo -e "${YELLOW}📋 Access Overseerr at: http://localhost:5055${NC}"
    echo -e "${YELLOW}📋 Create an admin account and start requesting content!${NC}"
}

main "$@"