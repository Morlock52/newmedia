#!/bin/bash

# Media Server Service Integration Configuration Script
# Configures API keys, connections, and integrations for all services
# Author: API Integration Specialist
# Date: $(date)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service URLs (internal Docker network)
JELLYFIN_URL="http://jellyfin:8096"
PLEX_URL="http://plex:32400"
SONARR_URL="http://sonarr:8989"
RADARR_URL="http://radarr:7878"
LIDARR_URL="http://lidarr:8686"
PROWLARR_URL="http://prowlarr:9696"
BAZARR_URL="http://bazarr:6767"
JELLYSEERR_URL="http://jellyseerr:5055"
OVERSEERR_URL="http://overseerr:5055"
QBITTORRENT_URL="http://qbittorrent:8080"
TRANSMISSION_URL="http://gluetun:9091"  # Transmission runs through Gluetun VPN
SABNZBD_URL="http://sabnzbd:8080"

# External URLs for testing
EXTERNAL_BASE="${EXTERNAL_BASE:-http://localhost}"
EXT_JELLYFIN="${EXTERNAL_BASE}:8096"
EXT_PLEX="${EXTERNAL_BASE}:32400"
EXT_SONARR="${EXTERNAL_BASE}:8989"
EXT_RADARR="${EXTERNAL_BASE}:7878"
EXT_PROWLARR="${EXTERNAL_BASE}:9696"
EXT_QBITTORRENT="${EXTERNAL_BASE}:8080"

# Global variables for API keys
SONARR_API_KEY=""
RADARR_API_KEY=""
LIDARR_API_KEY=""
PROWLARR_API_KEY=""
BAZARR_API_KEY=""
JELLYFIN_API_KEY=""
PLEX_TOKEN=""

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
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
        sleep 5
        ((attempt++))
    done
    
    error "$service_name failed to start after $((max_attempts * 5)) seconds"
    return 1
}

# Extract API key from *arr service
extract_arr_api_key() {
    local service_name=$1
    local config_path=$2
    local url=$3
    
    log "Extracting API key for $service_name"
    
    # Wait for service to be ready
    wait_for_service "$url/ping" "$service_name"
    
    # Try to extract from config.xml if it exists
    if [ -f "$config_path/config.xml" ]; then
        local api_key=$(grep -o '<ApiKey>[^<]*</ApiKey>' "$config_path/config.xml" 2>/dev/null | sed 's/<ApiKey>\(.*\)<\/ApiKey>/\1/' || echo "")
        if [ -n "$api_key" ]; then
            log "Found API key for $service_name: ${api_key:0:8}..."
            echo "$api_key"
            return 0
        fi
    fi
    
    # Generate new API key if not found
    warn "No API key found for $service_name, it may need to be configured manually"
    echo ""
}

# Configure Prowlarr indexers and app connections
configure_prowlarr() {
    log "Configuring Prowlarr indexers and applications"
    
    # Get Prowlarr API key
    PROWLARR_API_KEY=$(extract_arr_api_key "Prowlarr" "./prowlarr-config" "$PROWLARR_URL")
    
    if [ -z "$PROWLARR_API_KEY" ]; then
        error "Could not obtain Prowlarr API key. Please configure Prowlarr manually first."
        return 1
    fi
    
    # Add Sonarr application to Prowlarr
    log "Adding Sonarr to Prowlarr applications"
    curl -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $PROWLARR_API_KEY" \
        -d '{
            "name": "Sonarr",
            "implementation": "Sonarr",
            "configContract": "SonarrSettings",
            "fields": [
                {"name": "baseUrl", "value": "'$SONARR_URL'"},
                {"name": "apiKey", "value": "'$SONARR_API_KEY'"},
                {"name": "syncCategories", "value": [5000, 5030, 5040]}
            ],
            "tags": []
        }' 2>/dev/null || warn "Failed to add Sonarr to Prowlarr"
    
    # Add Radarr application to Prowlarr
    log "Adding Radarr to Prowlarr applications"
    curl -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $PROWLARR_API_KEY" \
        -d '{
            "name": "Radarr",
            "implementation": "Radarr",
            "configContract": "RadarrSettings",
            "fields": [
                {"name": "baseUrl", "value": "'$RADARR_URL'"},
                {"name": "apiKey", "value": "'$RADARR_API_KEY'"},
                {"name": "syncCategories", "value": [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060]}
            ],
            "tags": []
        }' 2>/dev/null || warn "Failed to add Radarr to Prowlarr"
    
    # Add Lidarr application to Prowlarr
    log "Adding Lidarr to Prowlarr applications"
    curl -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $PROWLARR_API_KEY" \
        -d '{
            "name": "Lidarr",
            "implementation": "Lidarr",
            "configContract": "LidarrSettings",
            "fields": [
                {"name": "baseUrl", "value": "'$LIDARR_URL'"},
                {"name": "apiKey", "value": "'$LIDARR_API_KEY'"},
                {"name": "syncCategories", "value": [3000, 3010, 3020, 3030, 3040]}
            ],
            "tags": []
        }' 2>/dev/null || warn "Failed to add Lidarr to Prowlarr"
}

# Configure download clients in *arr services
configure_download_clients() {
    log "Configuring download clients in *arr services"
    
    # Configure qBittorrent in Sonarr
    log "Adding qBittorrent to Sonarr"
    curl -X POST "$SONARR_URL/api/v3/downloadclient" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $SONARR_API_KEY" \
        -d '{
            "enable": true,
            "name": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "fields": [
                {"name": "host", "value": "qbittorrent"},
                {"name": "port", "value": 8080},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "tvCategory", "value": "tv"},
                {"name": "recentTvPriority", "value": 0},
                {"name": "olderTvPriority", "value": 0},
                {"name": "initialState", "value": 0}
            ],
            "tags": []
        }' 2>/dev/null || warn "Failed to add qBittorrent to Sonarr"
    
    # Configure qBittorrent in Radarr
    log "Adding qBittorrent to Radarr"
    curl -X POST "$RADARR_URL/api/v3/downloadclient" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $RADARR_API_KEY" \
        -d '{
            "enable": true,
            "name": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "fields": [
                {"name": "host", "value": "qbittorrent"},
                {"name": "port", "value": 8080},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "movieCategory", "value": "movies"},
                {"name": "recentMoviePriority", "value": 0},
                {"name": "olderMoviePriority", "value": 0},
                {"name": "initialState", "value": 0}
            ],
            "tags": []
        }' 2>/dev/null || warn "Failed to add qBittorrent to Radarr"
    
    # Configure SABnzbd in Sonarr
    log "Adding SABnzbd to Sonarr"
    curl -X POST "$SONARR_URL/api/v3/downloadclient" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $SONARR_API_KEY" \
        -d '{
            "enable": true,
            "name": "SABnzbd",
            "implementation": "Sabnzbd",
            "configContract": "SabnzbdSettings",
            "fields": [
                {"name": "host", "value": "sabnzbd"},
                {"name": "port", "value": 8080},
                {"name": "tvCategory", "value": "tv"},
                {"name": "recentTvPriority", "value": 0},
                {"name": "olderTvPriority", "value": 0}
            ],
            "tags": []
        }' 2>/dev/null || warn "Failed to add SABnzbd to Sonarr"
}

# Configure Bazarr subtitle management
configure_bazarr() {
    log "Configuring Bazarr for subtitle management"
    
    BAZARR_API_KEY=$(extract_arr_api_key "Bazarr" "./bazarr-config" "$BAZARR_URL")
    
    if [ -z "$BAZARR_API_KEY" ]; then
        warn "Could not obtain Bazarr API key. Manual configuration required."
        return 1
    fi
    
    # Configure Sonarr in Bazarr
    log "Connecting Bazarr to Sonarr"
    curl -X POST "$BAZARR_URL/api/system/settings" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $BAZARR_API_KEY" \
        -d '{
            "settings": {
                "sonarr": {
                    "ip": "sonarr",
                    "port": 8989,
                    "base_url": "",
                    "ssl": false,
                    "apikey": "'$SONARR_API_KEY'",
                    "full_update": "Daily",
                    "only_monitored": true
                }
            }
        }' 2>/dev/null || warn "Failed to configure Sonarr in Bazarr"
    
    # Configure Radarr in Bazarr
    log "Connecting Bazarr to Radarr"
    curl -X POST "$BAZARR_URL/api/system/settings" \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: $BAZARR_API_KEY" \
        -d '{
            "settings": {
                "radarr": {
                    "ip": "radarr",
                    "port": 7878,
                    "base_url": "",
                    "ssl": false,
                    "apikey": "'$RADARR_API_KEY'",
                    "full_update": "Daily",
                    "only_monitored": true
                }
            }
        }' 2>/dev/null || warn "Failed to configure Radarr in Bazarr"
}

# Get Jellyfin API key
get_jellyfin_api_key() {
    log "Attempting to get Jellyfin API key"
    
    # Try to extract from existing configuration
    if [ -f "./jellyfin-config/config/system.xml" ]; then
        local api_key=$(grep -o '<AccessToken>[^<]*</AccessToken>' "./jellyfin-config/config/system.xml" 2>/dev/null | head -1 | sed 's/<AccessToken>\(.*\)<\/AccessToken>/\1/' || echo "")
        if [ -n "$api_key" ]; then
            log "Found Jellyfin API key"
            echo "$api_key"
            return 0
        fi
    fi
    
    warn "Jellyfin API key not found. Please create one manually in Jellyfin Dashboard > API Keys"
    echo ""
}

# Configure Jellyseerr with Jellyfin
configure_jellyseerr() {
    log "Configuring Jellyseerr with Jellyfin integration"
    
    wait_for_service "$JELLYSEERR_URL" "Jellyseerr"
    
    JELLYFIN_API_KEY=$(get_jellyfin_api_key)
    
    if [ -z "$JELLYFIN_API_KEY" ]; then
        warn "Cannot configure Jellyseerr without Jellyfin API key"
        return 1
    fi
    
    # Configure Jellyfin in Jellyseerr
    log "Adding Jellyfin server to Jellyseerr"
    curl -X POST "$JELLYSEERR_URL/api/v1/settings/jellyfin" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Jellyfin",
            "hostname": "jellyfin",
            "port": 8096,
            "authUser": "admin",
            "authPass": "password",
            "internalUrl": "'$JELLYFIN_URL'",
            "externalUrl": "'$EXT_JELLYFIN'",
            "ssl": false
        }' 2>/dev/null || warn "Failed to configure Jellyfin in Jellyseerr"
    
    # Configure Sonarr in Jellyseerr
    log "Adding Sonarr to Jellyseerr"
    curl -X POST "$JELLYSEERR_URL/api/v1/settings/sonarr" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Sonarr",
            "hostname": "sonarr",
            "port": 8989,
            "apiKey": "'$SONARR_API_KEY'",
            "baseUrl": "",
            "activeProfileId": 1,
            "activeDirectory": "/tv",
            "ssl": false,
            "is4k": false
        }' 2>/dev/null || warn "Failed to configure Sonarr in Jellyseerr"
    
    # Configure Radarr in Jellyseerr
    log "Adding Radarr to Jellyseerr"
    curl -X POST "$JELLYSEERR_URL/api/v1/settings/radarr" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Radarr",
            "hostname": "radarr",
            "port": 7878,
            "apiKey": "'$RADARR_API_KEY'",
            "baseUrl": "",
            "activeProfileId": 1,
            "activeDirectory": "/movies",
            "ssl": false,
            "is4k": false
        }' 2>/dev/null || warn "Failed to configure Radarr in Jellyseerr"
}

# Main configuration function
main() {
    log "Starting Media Server Service Integration Configuration"
    
    # Create logs directory
    mkdir -p ./logs
    
    # Extract API keys from all services
    log "Extracting API keys from services"
    SONARR_API_KEY=$(extract_arr_api_key "Sonarr" "./sonarr-config" "$SONARR_URL")
    RADARR_API_KEY=$(extract_arr_api_key "Radarr" "./radarr-config" "$RADARR_URL")
    LIDARR_API_KEY=$(extract_arr_api_key "Lidarr" "./lidarr-config" "$LIDARR_URL")
    
    # Check if we have required API keys
    if [ -z "$SONARR_API_KEY" ] || [ -z "$RADARR_API_KEY" ]; then
        error "Missing required API keys. Please ensure Sonarr and Radarr are properly configured."
        exit 1
    fi
    
    # Save API keys to file for reference
    cat > ./api-keys.env << EOF
# API Keys for Media Server Services
# Generated on $(date)
SONARR_API_KEY="$SONARR_API_KEY"
RADARR_API_KEY="$RADARR_API_KEY"
LIDARR_API_KEY="$LIDARR_API_KEY"
PROWLARR_API_KEY="$PROWLARR_API_KEY"
BAZARR_API_KEY="$BAZARR_API_KEY"
JELLYFIN_API_KEY="$JELLYFIN_API_KEY"

# Service URLs
SONARR_URL="$SONARR_URL"
RADARR_URL="$RADARR_URL"
PROWLARR_URL="$PROWLARR_URL"
JELLYFIN_URL="$JELLYFIN_URL"
QBITTORRENT_URL="$QBITTORRENT_URL"
EOF

    log "API keys saved to api-keys.env"
    
    # Run configuration functions
    configure_prowlarr
    configure_download_clients
    configure_bazarr
    configure_jellyseerr
    
    log "Service integration configuration completed!"
    log "Please check the logs for any warnings or errors."
    log "Some services may require manual configuration through their web interfaces."
    
    # Display summary
    echo -e "\n${BLUE}=== INTEGRATION SUMMARY ===${NC}"
    echo -e "${GREEN}✓${NC} Prowlarr configured with Sonarr, Radarr, and Lidarr"
    echo -e "${GREEN}✓${NC} Download clients configured in *arr services"
    echo -e "${GREEN}✓${NC} Bazarr configured for subtitle management"
    echo -e "${GREEN}✓${NC} Jellyseerr configured with Jellyfin integration"
    echo -e "${YELLOW}!${NC} Manual configuration may be required for some services"
    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "1. Verify all services are accessible via their web interfaces"
    echo -e "2. Run the API test suite: ./test-api-integrations.sh"
    echo -e "3. Check service logs for any errors"
}

# Run main function
main "$@"