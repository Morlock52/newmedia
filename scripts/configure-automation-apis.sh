#!/bin/bash

# Configure Automation APIs - Connect all *arr services together
# This script sets up the API connections between services for full automation

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Service URLs
PROWLARR_URL="http://localhost:9696"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"
LIDARR_URL="http://localhost:8686"
READARR_URL="http://localhost:8787"
BAZARR_URL="http://localhost:6767"
OVERSEERR_URL="http://localhost:5055"
QBIT_URL="http://localhost:8080"
SAB_URL="http://localhost:8081"

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get API key from config file
get_api_key() {
    local config_path=$1
    if [ -f "$config_path" ]; then
        grep -o '<ApiKey>[^<]*</ApiKey>' "$config_path" | sed 's/<ApiKey>\|<\/ApiKey>//g' | head -1
    else
        echo ""
    fi
}

# Wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=20
    
    print_step "Waiting for $name..."
    for i in $(seq 1 $max_attempts); do
        if curl -s -f "$url/ping" >/dev/null 2>&1 || curl -s -f "$url" >/dev/null 2>&1; then
            print_success "$name is ready"
            return 0
        fi
        sleep 3
    done
    print_error "$name not ready after $max_attempts attempts"
    return 1
}

# Configure qBittorrent in Prowlarr
configure_qbittorrent_in_prowlarr() {
    local prowlarr_api_key=$1
    
    print_step "Adding qBittorrent to Prowlarr..."
    
    curl -s -X POST "$PROWLARR_URL/api/v1/downloadclient" \
        -H "X-Api-Key: $prowlarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
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
                {"name": "urlBase", "value": "/"},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "tvCategory", "value": "tv"},
                {"name": "movieCategory", "value": "movies"},
                {"name": "musicCategory", "value": "music"},
                {"name": "bookCategory", "value": "books"}
            ],
            "implementationName": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "tags": []
        }' || print_error "Failed to add qBittorrent to Prowlarr"
}

# Configure SABnzbd in Prowlarr
configure_sabnzbd_in_prowlarr() {
    local prowlarr_api_key=$1
    
    print_step "Adding SABnzbd to Prowlarr..."
    
    curl -s -X POST "$PROWLARR_URL/api/v1/downloadclient" \
        -H "X-Api-Key: $prowlarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "enable": true,
            "protocol": "usenet",
            "priority": 1,
            "removeCompletedDownloads": false,
            "removeFailedDownloads": true,
            "name": "SABnzbd",
            "fields": [
                {"name": "host", "value": "sabnzbd"},
                {"name": "port", "value": 8080},
                {"name": "useSsl", "value": false},
                {"name": "urlBase", "value": "/"},
                {"name": "apiKey", "value": ""},
                {"name": "username", "value": ""},
                {"name": "password", "value": ""},
                {"name": "tvCategory", "value": "tv"},
                {"name": "movieCategory", "value": "movies"},
                {"name": "musicCategory", "value": "music"},
                {"name": "bookCategory", "value": "books"}
            ],
            "implementationName": "SABnzbd",
            "implementation": "Sabnzbd",
            "configContract": "SabnzbdSettings",
            "tags": []
        }' || print_error "Failed to add SABnzbd to Prowlarr"
}

# Add Sonarr to Prowlarr
configure_sonarr_in_prowlarr() {
    local prowlarr_api_key=$1
    local sonarr_api_key=$2
    
    print_step "Adding Sonarr to Prowlarr..."
    
    curl -s -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $prowlarr_api_key" \
        -H "Content-Type: application/json" \
        -d "{
            \"enable\": true,
            \"name\": \"Sonarr\",
            \"fields\": [
                {\"name\": \"prowlarrUrl\", \"value\": \"http://prowlarr:9696\"},
                {\"name\": \"baseUrl\", \"value\": \"http://sonarr:8989\"},
                {\"name\": \"apiKey\", \"value\": \"$sonarr_api_key\"},
                {\"name\": \"syncLevel\", \"value\": \"addOnly\"}
            ],
            \"implementationName\": \"Sonarr\",
            \"implementation\": \"Sonarr\",
            \"configContract\": \"SonarrSettings\",
            \"tags\": []
        }" || print_error "Failed to add Sonarr to Prowlarr"
}

# Add Radarr to Prowlarr
configure_radarr_in_prowlarr() {
    local prowlarr_api_key=$1
    local radarr_api_key=$2
    
    print_step "Adding Radarr to Prowlarr..."
    
    curl -s -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $prowlarr_api_key" \
        -H "Content-Type: application/json" \
        -d "{
            \"enable\": true,
            \"name\": \"Radarr\",
            \"fields\": [
                {\"name\": \"prowlarrUrl\", \"value\": \"http://prowlarr:9696\"},
                {\"name\": \"baseUrl\", \"value\": \"http://radarr:7878\"},
                {\"name\": \"apiKey\", \"value\": \"$radarr_api_key\"},
                {\"name\": \"syncLevel\", \"value\": \"addOnly\"}
            ],
            \"implementationName\": \"Radarr\",
            \"implementation\": \"Radarr\",
            \"configContract\": \"RadarrSettings\",
            \"tags\": []
        }" || print_error "Failed to add Radarr to Prowlarr"
}

# Add Lidarr to Prowlarr
configure_lidarr_in_prowlarr() {
    local prowlarr_api_key=$1
    local lidarr_api_key=$2
    
    print_step "Adding Lidarr to Prowlarr..."
    
    curl -s -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $prowlarr_api_key" \
        -H "Content-Type: application/json" \
        -d "{
            \"enable\": true,
            \"name\": \"Lidarr\",
            \"fields\": [
                {\"name\": \"prowlarrUrl\", \"value\": \"http://prowlarr:9696\"},
                {\"name\": \"baseUrl\", \"value\": \"http://lidarr:8686\"},
                {\"name\": \"apiKey\", \"value\": \"$lidarr_api_key\"},
                {\"name\": \"syncLevel\", \"value\": \"addOnly\"}
            ],
            \"implementationName\": \"Lidarr\",
            \"implementation\": \"Lidarr\",
            \"configContract\": \"LidarrSettings\",
            \"tags\": []
        }" || print_error "Failed to add Lidarr to Prowlarr"
}

# Add Readarr to Prowlarr
configure_readarr_in_prowlarr() {
    local prowlarr_api_key=$1
    local readarr_api_key=$2
    
    print_step "Adding Readarr to Prowlarr..."
    
    curl -s -X POST "$PROWLARR_URL/api/v1/applications" \
        -H "X-Api-Key: $prowlarr_api_key" \
        -H "Content-Type: application/json" \
        -d "{
            \"enable\": true,
            \"name\": \"Readarr\",
            \"fields\": [
                {\"name\": \"prowlarrUrl\", \"value\": \"http://prowlarr:9696\"},
                {\"name\": \"baseUrl\", \"value\": \"http://readarr:8787\"},
                {\"name\": \"apiKey\", \"value\": \"$readarr_api_key\"},
                {\"name\": \"syncLevel\", \"value\": \"addOnly\"}
            ],
            \"implementationName\": \"Readarr\",
            \"implementation\": \"Readarr\",
            \"configContract\": \"ReadarrSettings\",
            \"tags\": []
        }" || print_error "Failed to add Readarr to Prowlarr"
}

# Configure download clients in Sonarr
configure_sonarr_downloads() {
    local sonarr_api_key=$1
    
    print_step "Configuring download clients in Sonarr..."
    
    # Add qBittorrent
    curl -s -X POST "$SONARR_URL/api/v3/downloadclient" \
        -H "X-Api-Key: $sonarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
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
                {"name": "urlBase", "value": "/"},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "tvCategory", "value": "tv"},
                {"name": "tvImportedCategory", "value": "tv"},
                {"name": "recentTvPriority", "value": 0},
                {"name": "olderTvPriority", "value": 0},
                {"name": "initialState", "value": 0}
            ],
            "implementationName": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "tags": []
        }' || print_error "Failed to add qBittorrent to Sonarr"
    
    # Add SABnzbd
    curl -s -X POST "$SONARR_URL/api/v3/downloadclient" \
        -H "X-Api-Key: $sonarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "enable": true,
            "protocol": "usenet",
            "priority": 1,
            "removeCompletedDownloads": false,
            "removeFailedDownloads": true,
            "name": "SABnzbd",
            "fields": [
                {"name": "host", "value": "sabnzbd"},
                {"name": "port", "value": 8080},
                {"name": "useSsl", "value": false},
                {"name": "urlBase", "value": "/"},
                {"name": "apiKey", "value": ""},
                {"name": "username", "value": ""},
                {"name": "password", "value": ""},
                {"name": "tvCategory", "value": "tv"},
                {"name": "recentTvPriority", "value": 0},
                {"name": "olderTvPriority", "value": 0}
            ],
            "implementationName": "SABnzbd",
            "implementation": "Sabnzbd",
            "configContract": "SabnzbdSettings",
            "tags": []
        }' || print_error "Failed to add SABnzbd to Sonarr"
}

# Configure download clients in Radarr
configure_radarr_downloads() {
    local radarr_api_key=$1
    
    print_step "Configuring download clients in Radarr..."
    
    # Add qBittorrent
    curl -s -X POST "$RADARR_URL/api/v3/downloadclient" \
        -H "X-Api-Key: $radarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
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
                {"name": "urlBase", "value": "/"},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "movieCategory", "value": "movies"},
                {"name": "movieImportedCategory", "value": "movies"},
                {"name": "recentMoviePriority", "value": 0},
                {"name": "olderMoviePriority", "value": 0},
                {"name": "initialState", "value": 0}
            ],
            "implementationName": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "tags": []
        }' || print_error "Failed to add qBittorrent to Radarr"
    
    # Add SABnzbd
    curl -s -X POST "$RADARR_URL/api/v3/downloadclient" \
        -H "X-Api-Key: $radarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "enable": true,
            "protocol": "usenet",
            "priority": 1,
            "removeCompletedDownloads": false,
            "removeFailedDownloads": true,
            "name": "SABnzbd",
            "fields": [
                {"name": "host", "value": "sabnzbd"},
                {"name": "port", "value": 8080},
                {"name": "useSsl", "value": false},
                {"name": "urlBase", "value": "/"},
                {"name": "apiKey", "value": ""},
                {"name": "username", "value": ""},
                {"name": "password", "value": ""},
                {"name": "movieCategory", "value": "movies"},
                {"name": "recentMoviePriority", "value": 0},
                {"name": "olderMoviePriority", "value": 0}
            ],
            "implementationName": "SABnzbd",
            "implementation": "Sabnzbd",
            "configContract": "SabnzbdSettings",
            "tags": []
        }' || print_error "Failed to add SABnzbd to Radarr"
}

# Similar configurations for Lidarr and Readarr...
configure_lidarr_downloads() {
    local lidarr_api_key=$1
    
    print_step "Configuring download clients in Lidarr..."
    
    curl -s -X POST "$LIDARR_URL/api/v1/downloadclient" \
        -H "X-Api-Key: $lidarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
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
                {"name": "urlBase", "value": "/"},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "musicCategory", "value": "music"}
            ],
            "implementationName": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "tags": []
        }' || print_error "Failed to add qBittorrent to Lidarr"
}

configure_readarr_downloads() {
    local readarr_api_key=$1
    
    print_step "Configuring download clients in Readarr..."
    
    curl -s -X POST "$READARR_URL/api/v1/downloadclient" \
        -H "X-Api-Key: $readarr_api_key" \
        -H "Content-Type: application/json" \
        -d '{
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
                {"name": "urlBase", "value": "/"},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "bookCategory", "value": "books"}
            ],
            "implementationName": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "tags": []
        }' || print_error "Failed to add qBittorrent to Readarr"
}

# Main configuration function
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Automation API Configuration Script  ${NC}"
    echo -e "${BLUE}   Ultimate Media Server 2025          ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    
    # Wait for all services to be ready
    wait_for_service "$PROWLARR_URL" "Prowlarr"
    wait_for_service "$SONARR_URL" "Sonarr"
    wait_for_service "$RADARR_URL" "Radarr"
    wait_for_service "$LIDARR_URL" "Lidarr"
    wait_for_service "$READARR_URL" "Readarr"
    
    # Get API keys
    print_step "Extracting API keys..."
    PROWLARR_API_KEY=$(get_api_key "config/prowlarr/config.xml")
    SONARR_API_KEY=$(get_api_key "config/sonarr/config.xml")
    RADARR_API_KEY=$(get_api_key "config/radarr/config.xml")
    LIDARR_API_KEY=$(get_api_key "config/lidarr/config.xml")
    READARR_API_KEY=$(get_api_key "config/readarr/config.xml")
    
    if [[ -z "$PROWLARR_API_KEY" || -z "$SONARR_API_KEY" || -z "$RADARR_API_KEY" ]]; then
        print_error "Could not extract API keys. Make sure services are fully started."
        exit 1
    fi
    
    print_success "API keys extracted successfully"
    
    # Configure download clients in Prowlarr
    configure_qbittorrent_in_prowlarr "$PROWLARR_API_KEY"
    configure_sabnzbd_in_prowlarr "$PROWLARR_API_KEY"
    
    # Add applications to Prowlarr
    configure_sonarr_in_prowlarr "$PROWLARR_API_KEY" "$SONARR_API_KEY"
    configure_radarr_in_prowlarr "$PROWLARR_API_KEY" "$RADARR_API_KEY"
    configure_lidarr_in_prowlarr "$PROWLARR_API_KEY" "$LIDARR_API_KEY"
    configure_readarr_in_prowlarr "$PROWLARR_API_KEY" "$READARR_API_KEY"
    
    # Configure download clients in each *arr service
    configure_sonarr_downloads "$SONARR_API_KEY"
    configure_radarr_downloads "$RADARR_API_KEY"
    configure_lidarr_downloads "$LIDARR_API_KEY"
    configure_readarr_downloads "$READARR_API_KEY"
    
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Configuration Complete!            ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo -e "Your automation stack is now configured:"
    echo -e "  ${BLUE}1.${NC} Add indexers in Prowlarr"
    echo -e "  ${BLUE}2.${NC} Configure quality profiles in each *arr service"
    echo -e "  ${BLUE}3.${NC} Set up Overseerr for user requests"
    echo -e "  ${BLUE}4.${NC} Test the automation by requesting content"
    echo
    echo -e "API Keys saved for reference:"
    echo -e "  Prowlarr: ${PROWLARR_API_KEY:0:8}..."
    echo -e "  Sonarr:   ${SONARR_API_KEY:0:8}..."
    echo -e "  Radarr:   ${RADARR_API_KEY:0:8}..."
    echo -e "  Lidarr:   ${LIDARR_API_KEY:0:8}..."
    echo -e "  Readarr:  ${READARR_API_KEY:0:8}..."
    echo
}

# Run main function
main "$@"