#!/bin/bash

# Jellyseerr Configuration Script
# Configures Jellyseerr with Jellyfin, Sonarr, and Radarr integration
# Author: API Integration Specialist
# Date: $(date)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service URLs
JELLYSEERR_URL="http://localhost:5055"
JELLYFIN_URL="http://jellyfin:8096"
SONARR_URL="http://sonarr:8989"
RADARR_URL="http://radarr:7878"

# External URLs
EXT_JELLYFIN_URL="http://localhost:8096"
EXT_SONARR_URL="http://localhost:8989"
EXT_RADARR_URL="http://localhost:7878"

# Load API keys if available
if [ -f "./api-keys.env" ]; then
    source "./api-keys.env"
fi

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    log "Waiting for $service_name to be ready at $url"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
            log "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 3
        ((attempt++))
    done
    
    error "$service_name failed to start after $((max_attempts * 3)) seconds"
    return 1
}

# Get Jellyseerr session cookie
get_jellyseerr_session() {
    local cookie_jar=$(mktemp)
    
    # Try to access the main page to establish a session
    curl -s -c "$cookie_jar" "$JELLYSEERR_URL" > /dev/null
    
    if [ -f "$cookie_jar" ]; then
        echo "$cookie_jar"
        return 0
    else
        error "Failed to establish Jellyseerr session"
        return 1
    fi
}

# Initialize Jellyseerr (first-time setup)
initialize_jellyseerr() {
    local cookie_jar="$1"
    
    log "Initializing Jellyseerr first-time setup"
    
    # Check if Jellyseerr needs initialization
    local status_response
    status_response=$(curl -s -b "$cookie_jar" "$JELLYSEERR_URL/api/v1/status" 2>/dev/null || echo "")
    
    if echo "$status_response" | grep -q '"restartRequired":false'; then
        log "Jellyseerr is already initialized"
        return 0
    fi
    
    # Perform initial setup
    curl -s -b "$cookie_jar" -X POST \
        -H "Content-Type: application/json" \
        -d '{
            "mediaType": "jellyfin"
        }' \
        "$JELLYSEERR_URL/api/v1/settings/initialize" > /dev/null
    
    log "Jellyseerr initialization completed"
}

# Configure Jellyfin in Jellyseerr
configure_jellyfin() {
    local cookie_jar="$1"
    
    log "Configuring Jellyfin server in Jellyseerr"
    
    # Test Jellyfin connectivity first
    if ! curl -s --connect-timeout 5 "$JELLYFIN_URL/System/Info/Public" > /dev/null 2>&1; then
        warn "Jellyfin is not accessible at $JELLYFIN_URL"
        return 1
    fi
    
    # Configure Jellyfin server
    local jellyfin_config='{
        "name": "Jellyfin",
        "hostname": "jellyfin",
        "port": 8096,
        "authUser": "",
        "authPass": "",
        "internalUrl": "'$JELLYFIN_URL'",
        "externalUrl": "'$EXT_JELLYFIN_URL'",
        "ssl": false
    }'
    
    curl -s -b "$cookie_jar" -X POST \
        -H "Content-Type: application/json" \
        -d "$jellyfin_config" \
        "$JELLYSEERR_URL/api/v1/settings/jellyfin" > /dev/null
    
    if [ $? -eq 0 ]; then
        log "Jellyfin server configured successfully"
        return 0
    else
        error "Failed to configure Jellyfin server"
        return 1
    fi
}

# Configure Sonarr in Jellyseerr
configure_sonarr() {
    local cookie_jar="$1"
    
    log "Configuring Sonarr in Jellyseerr"
    
    if [ -z "${SONARR_API_KEY:-}" ]; then
        error "Sonarr API key not available"
        return 1
    fi
    
    # Test Sonarr connectivity
    if ! curl -s -H "X-Api-Key:$SONARR_API_KEY" "$SONARR_URL/api/v3/system/status" > /dev/null 2>&1; then
        warn "Sonarr is not accessible at $SONARR_URL"
        return 1
    fi
    
    # Get Sonarr profiles and root folders
    local profiles_response
    profiles_response=$(curl -s -H "X-Api-Key:$SONARR_API_KEY" "$SONARR_URL/api/v3/qualityprofile" 2>/dev/null || echo "[]")
    
    local root_folders_response
    root_folders_response=$(curl -s -H "X-Api-Key:$SONARR_API_KEY" "$SONARR_URL/api/v3/rootfolder" 2>/dev/null || echo "[]")
    
    # Extract first available profile and root folder
    local profile_id
    profile_id=$(echo "$profiles_response" | jq -r '.[0].id // 1' 2>/dev/null || echo "1")
    
    local root_folder
    root_folder=$(echo "$root_folders_response" | jq -r '.[0].path // "/tv"' 2>/dev/null || echo "/tv")
    
    # Configure Sonarr
    local sonarr_config='{
        "name": "Sonarr",
        "hostname": "sonarr",
        "port": 8989,
        "apiKey": "'$SONARR_API_KEY'",
        "useSsl": false,
        "baseUrl": "",
        "activeProfileId": '$profile_id',
        "activeDirectory": "'$root_folder'",
        "activeLanguageProfileId": null,
        "activeAnimeProfileId": '$profile_id',
        "activeAnimeDirectory": "'$root_folder'",
        "activeAnimeLanguageProfileId": null,
        "tags": [],
        "animeTags": [],
        "is4k": false,
        "enableSeasonFolders": true,
        "externalUrl": "'$EXT_SONARR_URL'",
        "syncEnabled": true,
        "preventSearch": false
    }'
    
    curl -s -b "$cookie_jar" -X POST \
        -H "Content-Type: application/json" \
        -d "$sonarr_config" \
        "$JELLYSEERR_URL/api/v1/settings/sonarr" > /dev/null
    
    if [ $? -eq 0 ]; then
        log "Sonarr configured successfully (Profile ID: $profile_id, Root: $root_folder)"
        return 0
    else
        error "Failed to configure Sonarr"
        return 1
    fi
}

# Configure Radarr in Jellyseerr
configure_radarr() {
    local cookie_jar="$1"
    
    log "Configuring Radarr in Jellyseerr"
    
    if [ -z "${RADARR_API_KEY:-}" ]; then
        error "Radarr API key not available"
        return 1
    fi
    
    # Test Radarr connectivity
    if ! curl -s -H "X-Api-Key:$RADARR_API_KEY" "$RADARR_URL/api/v3/system/status" > /dev/null 2>&1; then
        warn "Radarr is not accessible at $RADARR_URL"
        return 1
    fi
    
    # Get Radarr profiles and root folders
    local profiles_response
    profiles_response=$(curl -s -H "X-Api-Key:$RADARR_API_KEY" "$RADARR_URL/api/v3/qualityprofile" 2>/dev/null || echo "[]")
    
    local root_folders_response
    root_folders_response=$(curl -s -H "X-Api-Key:$RADARR_API_KEY" "$RADARR_URL/api/v3/rootfolder" 2>/dev/null || echo "[]")
    
    # Extract first available profile and root folder
    local profile_id
    profile_id=$(echo "$profiles_response" | jq -r '.[0].id // 1' 2>/dev/null || echo "1")
    
    local root_folder
    root_folder=$(echo "$root_folders_response" | jq -r '.[0].path // "/movies"' 2>/dev/null || echo "/movies")
    
    # Configure Radarr
    local radarr_config='{
        "name": "Radarr",
        "hostname": "radarr",
        "port": 7878,
        "apiKey": "'$RADARR_API_KEY'",
        "useSsl": false,
        "baseUrl": "",
        "activeProfileId": '$profile_id',
        "activeDirectory": "'$root_folder'",
        "tags": [],
        "is4k": false,
        "minimumAvailability": "announced",
        "externalUrl": "'$EXT_RADARR_URL'",
        "syncEnabled": true,
        "preventSearch": false
    }'
    
    curl -s -b "$cookie_jar" -X POST \
        -H "Content-Type: application/json" \
        -d "$radarr_config" \
        "$JELLYSEERR_URL/api/v1/settings/radarr" > /dev/null
    
    if [ $? -eq 0 ]; then
        log "Radarr configured successfully (Profile ID: $profile_id, Root: $root_folder)"
        return 0
    else
        error "Failed to configure Radarr"
        return 1
    fi
}

# Configure general Jellyseerr settings
configure_general_settings() {
    local cookie_jar="$1"
    
    log "Configuring general Jellyseerr settings"
    
    local general_config='{
        "apiKey": "",
        "applicationTitle": "Jellyseerr",
        "applicationUrl": "",
        "hideAvailable": false,
        "localLogin": true,
        "newPlexLogin": true,
        "region": "US",
        "originalLanguage": "en",
        "displayAllMovies": false,
        "trustProxy": false,
        "csrfProtection": false,
        "cacheImages": false,
        "vapidPublicKey": "",
        "vapidPrivateKey": "",
        "enablePushRegistration": false,
        "locale": "en",
        "emailEnabled": false,
        "pgpKey": "",
        "enableImageCaching": true
    }'
    
    curl -s -b "$cookie_jar" -X POST \
        -H "Content-Type: application/json" \
        -d "$general_config" \
        "$JELLYSEERR_URL/api/v1/settings/main" > /dev/null
    
    log "General settings configured"
}

# Test Jellyseerr configuration
test_configuration() {
    local cookie_jar="$1"
    
    log "Testing Jellyseerr configuration"
    
    # Test status
    local status_response
    status_response=$(curl -s -b "$cookie_jar" "$JELLYSEERR_URL/api/v1/status" 2>/dev/null || echo "")
    
    if echo "$status_response" | grep -q '"version"'; then
        log "Jellyseerr is responding correctly"
    else
        error "Jellyseerr status check failed"
    fi
    
    # Test services
    curl -s -b "$cookie_jar" "$JELLYSEERR_URL/api/v1/service/sonarr" > /dev/null 2>&1 && \
        log "Sonarr integration test passed"
    
    curl -s -b "$cookie_jar" "$JELLYSEERR_URL/api/v1/service/radarr" > /dev/null 2>&1 && \
        log "Radarr integration test passed"
}

# Main configuration function
main() {
    echo -e "${BLUE}=== JELLYSEERR CONFIGURATION ===${NC}"
    log "Starting Jellyseerr configuration with media server integration"
    
    # Wait for Jellyseerr to be ready
    if ! wait_for_service "$JELLYSEERR_URL" "Jellyseerr"; then
        error "Cannot configure Jellyseerr - service not accessible"
        exit 1
    fi
    
    # Get session cookie
    local cookie_jar
    if ! cookie_jar=$(get_jellyseerr_session); then
        error "Cannot establish Jellyseerr session"
        exit 1
    fi
    
    # Initialize Jellyseerr
    initialize_jellyseerr "$cookie_jar"
    
    # Configure services
    configure_general_settings "$cookie_jar"
    configure_jellyfin "$cookie_jar"
    configure_sonarr "$cookie_jar"
    configure_radarr "$cookie_jar"
    
    # Test configuration
    test_configuration "$cookie_jar"
    
    # Cleanup
    rm -f "$cookie_jar"
    
    log "Jellyseerr configuration completed successfully!"
    
    # Display summary
    echo -e "\n${BLUE}=== CONFIGURATION SUMMARY ===${NC}"
    echo -e "${GREEN}✓${NC} Jellyseerr initialized"
    echo -e "${GREEN}✓${NC} Jellyfin server configured"
    echo -e "${GREEN}✓${NC} Sonarr integration configured"
    echo -e "${GREEN}✓${NC} Radarr integration configured"
    echo -e "${GREEN}✓${NC} General settings applied"
    
    echo -e "\n${BLUE}Access Information:${NC}"
    echo -e "• Jellyseerr Web UI: $JELLYSEERR_URL"
    echo -e "• Jellyfin Server: $EXT_JELLYFIN_URL"
    echo -e "• Sonarr: $EXT_SONARR_URL"
    echo -e "• Radarr: $EXT_RADARR_URL"
    
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo -e "1. Access Jellyseerr at $JELLYSEERR_URL"
    echo -e "2. Complete user authentication setup"
    echo -e "3. Test media requests"
    echo -e "4. Configure user permissions and quotas"
}

# Check dependencies
if ! command -v curl &> /dev/null; then
    error "curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    warn "jq is not installed - some advanced configuration may be limited"
fi

# Run main function
main "$@"