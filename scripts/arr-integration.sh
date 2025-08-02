#!/bin/bash

# Arr Stack Integration Script
# Integrates Sonarr, Radarr, Lidarr, Readarr, Prowlarr, and Bazarr
# Version: 2025.1

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../config"
LOG_DIR="${SCRIPT_DIR}/../logs"
LOG_FILE="${LOG_DIR}/arr-integration-$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Service Configuration
declare -A SERVICES=(
    ["sonarr"]="http://localhost:8989"
    ["radarr"]="http://localhost:7878"
    ["lidarr"]="http://localhost:8686"
    ["readarr"]="http://localhost:8787"
    ["prowlarr"]="http://localhost:9696"
    ["bazarr"]="http://localhost:6767"
    ["jellyfin"]="http://localhost:8096"
    ["qbittorrent"]="http://localhost:8080"
)

# API Keys (update these with your actual API keys)
declare -A API_KEYS=(
    ["sonarr"]="your-sonarr-api-key"
    ["radarr"]="your-radarr-api-key"
    ["lidarr"]="your-lidarr-api-key"
    ["readarr"]="your-readarr-api-key"
    ["prowlarr"]="your-prowlarr-api-key"
    ["bazarr"]="your-bazarr-api-key"
    ["jellyfin"]="your-jellyfin-api-key"
)

# Function to check service availability
check_service() {
    local service=$1
    local url=${SERVICES[$service]}
    
    if curl -s -f -o /dev/null "${url}/api/v3/system/status" -H "X-Api-Key: ${API_KEYS[$service]}" 2>/dev/null; then
        log "INFO" "${service^} is online at ${url}"
        return 0
    else
        log "WARN" "${service^} is not accessible at ${url}"
        return 1
    fi
}

# Function to get service configuration
get_service_config() {
    local service=$1
    local endpoint=$2
    local url="${SERVICES[$service]}/api/v3/${endpoint}"
    
    curl -s "${url}" -H "X-Api-Key: ${API_KEYS[$service]}" | jq '.'
}

# Function to update service configuration
update_service_config() {
    local service=$1
    local endpoint=$2
    local data=$3
    local url="${SERVICES[$service]}/api/v3/${endpoint}"
    
    curl -s -X PUT "${url}" \
        -H "X-Api-Key: ${API_KEYS[$service]}" \
        -H "Content-Type: application/json" \
        -d "${data}"
}

# Configure Prowlarr as indexer for all *arr apps
configure_prowlarr_integration() {
    log "INFO" "Configuring Prowlarr integration with *arr apps..."
    
    # Get Prowlarr apps configuration
    local prowlarr_apps=$(get_service_config "prowlarr" "applications")
    
    # Add Sonarr to Prowlarr
    if check_service "sonarr"; then
        local sonarr_config=$(cat <<EOF
{
    "name": "Sonarr",
    "syncLevel": "fullSync",
    "implementation": "Sonarr",
    "configContract": "SonarrSettings",
    "fields": [
        {
            "name": "baseUrl",
            "value": "${SERVICES[sonarr]}"
        },
        {
            "name": "apiKey",
            "value": "${API_KEYS[sonarr]}"
        },
        {
            "name": "syncCategories",
            "value": [5000, 5010, 5020, 5030, 5040, 5045, 5050]
        }
    ]
}
EOF
)
        curl -s -X POST "${SERVICES[prowlarr]}/api/v1/applications" \
            -H "X-Api-Key: ${API_KEYS[prowlarr]}" \
            -H "Content-Type: application/json" \
            -d "${sonarr_config}" || log "WARN" "Failed to add Sonarr to Prowlarr"
    fi
    
    # Add Radarr to Prowlarr
    if check_service "radarr"; then
        local radarr_config=$(cat <<EOF
{
    "name": "Radarr",
    "syncLevel": "fullSync",
    "implementation": "Radarr",
    "configContract": "RadarrSettings",
    "fields": [
        {
            "name": "baseUrl",
            "value": "${SERVICES[radarr]}"
        },
        {
            "name": "apiKey",
            "value": "${API_KEYS[radarr]}"
        },
        {
            "name": "syncCategories",
            "value": [2000, 2010, 2020, 2030, 2040, 2045, 2050]
        }
    ]
}
EOF
)
        curl -s -X POST "${SERVICES[prowlarr]}/api/v1/applications" \
            -H "X-Api-Key: ${API_KEYS[prowlarr]}" \
            -H "Content-Type: application/json" \
            -d "${radarr_config}" || log "WARN" "Failed to add Radarr to Prowlarr"
    fi
    
    log "INFO" "Prowlarr integration configured"
}

# Configure download clients
configure_download_clients() {
    log "INFO" "Configuring download clients..."
    
    local qbit_config=$(cat <<EOF
{
    "enable": true,
    "protocol": "torrent",
    "priority": 1,
    "name": "qBittorrent",
    "implementation": "QBittorrent",
    "configContract": "QBittorrentSettings",
    "fields": [
        {
            "name": "host",
            "value": "localhost"
        },
        {
            "name": "port",
            "value": 8080
        },
        {
            "name": "username",
            "value": "admin"
        },
        {
            "name": "password",
            "value": "adminpass"
        },
        {
            "name": "movieCategory",
            "value": "movies"
        },
        {
            "name": "tvCategory",
            "value": "tv"
        },
        {
            "name": "musicCategory",
            "value": "music"
        },
        {
            "name": "bookCategory",
            "value": "books"
        },
        {
            "name": "recentMoviePriority",
            "value": 1
        },
        {
            "name": "olderMoviePriority",
            "value": 5
        },
        {
            "name": "initialState",
            "value": 0
        }
    ]
}
EOF
)
    
    # Add to each *arr app
    for service in sonarr radarr lidarr readarr; do
        if check_service "$service"; then
            curl -s -X POST "${SERVICES[$service]}/api/v3/downloadclient" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${qbit_config}" || log "WARN" "Failed to add qBittorrent to ${service}"
        fi
    done
    
    log "INFO" "Download clients configured"
}

# Configure quality profiles
sync_quality_profiles() {
    log "INFO" "Synchronizing quality profiles..."
    
    # Define standard quality profiles
    local hd_profile=$(cat <<EOF
{
    "name": "HD-1080p",
    "upgradeAllowed": true,
    "cutoff": 7,
    "items": [
        {
            "quality": {"id": 3, "name": "WEBDL-1080p"},
            "allowed": true
        },
        {
            "quality": {"id": 7, "name": "Bluray-1080p"},
            "allowed": true
        },
        {
            "quality": {"id": 20, "name": "WEBDL-720p"},
            "allowed": true
        },
        {
            "quality": {"id": 4, "name": "HDTV-720p"},
            "allowed": true
        }
    ],
    "minFormatScore": 0,
    "cutoffFormatScore": 0,
    "formatItems": []
}
EOF
)
    
    local uhd_profile=$(cat <<EOF
{
    "name": "Ultra-HD",
    "upgradeAllowed": true,
    "cutoff": 19,
    "items": [
        {
            "quality": {"id": 19, "name": "Bluray-2160p"},
            "allowed": true
        },
        {
            "quality": {"id": 18, "name": "WEBDL-2160p"},
            "allowed": true
        },
        {
            "quality": {"id": 7, "name": "Bluray-1080p"},
            "allowed": true
        },
        {
            "quality": {"id": 3, "name": "WEBDL-1080p"},
            "allowed": true
        }
    ],
    "minFormatScore": 0,
    "cutoffFormatScore": 0,
    "formatItems": []
}
EOF
)
    
    # Apply to Sonarr and Radarr
    for service in sonarr radarr; do
        if check_service "$service"; then
            # Create HD profile
            curl -s -X POST "${SERVICES[$service]}/api/v3/qualityprofile" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${hd_profile}" || log "WARN" "Failed to create HD profile in ${service}"
            
            # Create UHD profile
            curl -s -X POST "${SERVICES[$service]}/api/v3/qualityprofile" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${uhd_profile}" || log "WARN" "Failed to create UHD profile in ${service}"
        fi
    done
    
    log "INFO" "Quality profiles synchronized"
}

# Configure notifications
setup_notifications() {
    log "INFO" "Setting up notifications..."
    
    # Discord notification
    local discord_config=$(cat <<EOF
{
    "name": "Discord",
    "implementation": "Discord",
    "configContract": "DiscordSettings",
    "fields": [
        {
            "name": "webHookUrl",
            "value": "https://discord.com/api/webhooks/your-webhook-url"
        },
        {
            "name": "username",
            "value": "Media Bot"
        },
        {
            "name": "avatar",
            "value": ""
        },
        {
            "name": "grabFields",
            "value": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
    ],
    "onGrab": true,
    "onDownload": true,
    "onUpgrade": true,
    "onRename": false,
    "onSeriesAdd": true,
    "onSeriesDelete": false,
    "onEpisodeFileDelete": false,
    "onEpisodeFileDeleteForUpgrade": false,
    "onHealthIssue": true,
    "onHealthRestored": true,
    "onApplicationUpdate": true,
    "onManualInteractionRequired": true
}
EOF
)
    
    # Email notification
    local email_config=$(cat <<EOF
{
    "name": "Email",
    "implementation": "Email",
    "configContract": "EmailSettings",
    "fields": [
        {
            "name": "server",
            "value": "smtp.gmail.com"
        },
        {
            "name": "port",
            "value": 587
        },
        {
            "name": "requireEncryption",
            "value": true
        },
        {
            "name": "username",
            "value": "your-email@gmail.com"
        },
        {
            "name": "password",
            "value": "your-app-password"
        },
        {
            "name": "from",
            "value": "your-email@gmail.com"
        },
        {
            "name": "to",
            "value": ["admin@example.com"]
        }
    ],
    "onGrab": false,
    "onDownload": true,
    "onUpgrade": true,
    "onHealthIssue": true,
    "onHealthRestored": true,
    "onApplicationUpdate": true
}
EOF
)
    
    # Jellyfin notification
    local jellyfin_config=$(cat <<EOF
{
    "name": "Jellyfin",
    "implementation": "MediaBrowser",
    "configContract": "MediaBrowserSettings",
    "fields": [
        {
            "name": "host",
            "value": "localhost"
        },
        {
            "name": "port",
            "value": 8096
        },
        {
            "name": "useSsl",
            "value": false
        },
        {
            "name": "apiKey",
            "value": "${API_KEYS[jellyfin]}"
        },
        {
            "name": "notify",
            "value": true
        },
        {
            "name": "updateLibrary",
            "value": true
        }
    ],
    "onGrab": false,
    "onDownload": true,
    "onUpgrade": true,
    "onRename": true
}
EOF
)
    
    # Add notifications to each service
    for service in sonarr radarr lidarr readarr; do
        if check_service "$service"; then
            # Add Discord
            curl -s -X POST "${SERVICES[$service]}/api/v3/notification" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${discord_config}" || log "WARN" "Failed to add Discord notification to ${service}"
            
            # Add Email
            curl -s -X POST "${SERVICES[$service]}/api/v3/notification" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${email_config}" || log "WARN" "Failed to add Email notification to ${service}"
            
            # Add Jellyfin
            curl -s -X POST "${SERVICES[$service]}/api/v3/notification" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${jellyfin_config}" || log "WARN" "Failed to add Jellyfin notification to ${service}"
        fi
    done
    
    log "INFO" "Notifications configured"
}

# Configure root folders
setup_root_folders() {
    log "INFO" "Setting up root folders..."
    
    # Define root folders
    declare -A ROOT_FOLDERS=(
        ["sonarr"]="/media/tv"
        ["radarr"]="/media/movies"
        ["lidarr"]="/media/music"
        ["readarr"]="/media/books"
    )
    
    for service in "${!ROOT_FOLDERS[@]}"; do
        if check_service "$service"; then
            local folder="${ROOT_FOLDERS[$service]}"
            local folder_config=$(cat <<EOF
{
    "path": "${folder}",
    "accessible": true,
    "freeSpace": 0,
    "unmappedFolders": []
}
EOF
)
            curl -s -X POST "${SERVICES[$service]}/api/v3/rootfolder" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${folder_config}" || log "WARN" "Failed to add root folder to ${service}"
        fi
    done
    
    log "INFO" "Root folders configured"
}

# Configure naming conventions
setup_naming_conventions() {
    log "INFO" "Setting up naming conventions..."
    
    # Sonarr naming
    local sonarr_naming=$(cat <<EOF
{
    "renameEpisodes": true,
    "replaceIllegalCharacters": true,
    "standardEpisodeFormat": "{Series Title} - S{season:00}E{episode:00} - {Episode Title} [{Quality Full}]",
    "dailyEpisodeFormat": "{Series Title} - {Air-Date} - {Episode Title} [{Quality Full}]",
    "animeEpisodeFormat": "{Series Title} - S{season:00}E{episode:00} - {Episode Title} [{Quality Full}]",
    "seriesFolderFormat": "{Series Title}",
    "seasonFolderFormat": "Season {season:00}",
    "specialsFolderFormat": "Specials",
    "multiEpisodeStyle": 5
}
EOF
)
    
    # Radarr naming
    local radarr_naming=$(cat <<EOF
{
    "renameMovies": true,
    "replaceIllegalCharacters": true,
    "colonReplacementFormat": "delete",
    "standardMovieFormat": "{Movie Title} ({Release Year}) [{Quality Full}]",
    "movieFolderFormat": "{Movie Title} ({Release Year})"
}
EOF
)
    
    # Apply naming conventions
    if check_service "sonarr"; then
        curl -s -X PUT "${SERVICES[sonarr]}/api/v3/config/naming" \
            -H "X-Api-Key: ${API_KEYS[sonarr]}" \
            -H "Content-Type: application/json" \
            -d "${sonarr_naming}" || log "WARN" "Failed to set Sonarr naming conventions"
    fi
    
    if check_service "radarr"; then
        curl -s -X PUT "${SERVICES[radarr]}/api/v3/config/naming" \
            -H "X-Api-Key: ${API_KEYS[radarr]}" \
            -H "Content-Type: application/json" \
            -d "${radarr_naming}" || log "WARN" "Failed to set Radarr naming conventions"
    fi
    
    log "INFO" "Naming conventions configured"
}

# Configure recycling bin
setup_recycling_bin() {
    log "INFO" "Setting up recycling bin..."
    
    local media_management=$(cat <<EOF
{
    "autoUnmonitorPreviouslyDownloadedMovies": false,
    "recycleBin": "/media/recycled",
    "recycleBinCleanupDays": 7,
    "downloadPropersAndRepacks": "preferAndUpgrade",
    "createEmptyMovieFolders": false,
    "deleteEmptyFolders": true,
    "fileDate": "none",
    "rescanAfterRefresh": "always",
    "autoRenameFolders": true,
    "pathsDefaultStatic": false,
    "setPermissionsLinux": true,
    "chmodFolder": "755",
    "chownGroup": "media"
}
EOF
)
    
    # Apply to Sonarr and Radarr
    for service in sonarr radarr; do
        if check_service "$service"; then
            curl -s -X PUT "${SERVICES[$service]}/api/v3/config/mediamanagement" \
                -H "X-Api-Key: ${API_KEYS[$service]}" \
                -H "Content-Type: application/json" \
                -d "${media_management}" || log "WARN" "Failed to configure recycling bin for ${service}"
        fi
    done
    
    log "INFO" "Recycling bin configured"
}

# Health check function
perform_health_check() {
    log "INFO" "Performing health check..."
    
    local all_healthy=true
    
    for service in "${!SERVICES[@]}"; do
        if check_service "$service"; then
            echo -e "${GREEN}✓${NC} ${service^} is healthy"
            
            # Get health issues
            if [[ "$service" != "jellyfin" ]] && [[ "$service" != "qbittorrent" ]]; then
                local health=$(curl -s "${SERVICES[$service]}/api/v3/health" \
                    -H "X-Api-Key: ${API_KEYS[$service]}" | jq -r '.[] | .message' 2>/dev/null)
                
                if [[ -n "$health" ]]; then
                    echo -e "${YELLOW}  Warning: ${health}${NC}"
                    all_healthy=false
                fi
            fi
        else
            echo -e "${RED}✗${NC} ${service^} is not responding"
            all_healthy=false
        fi
    done
    
    if $all_healthy; then
        log "INFO" "All services are healthy"
    else
        log "WARN" "Some services have issues"
    fi
}

# Main execution
main() {
    log "INFO" "Starting Arr Stack Integration..."
    
    echo -e "${BLUE}=== Arr Stack Integration Script ===${NC}"
    echo
    
    # Check all services
    echo -e "${YELLOW}Checking service availability...${NC}"
    for service in "${!SERVICES[@]}"; do
        check_service "$service"
    done
    echo
    
    # Run integration tasks
    echo -e "${YELLOW}Configuring integrations...${NC}"
    
    configure_prowlarr_integration
    configure_download_clients
    sync_quality_profiles
    setup_notifications
    setup_root_folders
    setup_naming_conventions
    setup_recycling_bin
    
    echo
    echo -e "${YELLOW}Performing final health check...${NC}"
    perform_health_check
    
    echo
    log "INFO" "Arr Stack Integration completed"
    echo -e "${GREEN}Integration complete!${NC}"
    echo -e "Log file: ${LOG_FILE}"
}

# Help function
show_help() {
    cat << EOF
Arr Stack Integration Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -c, --check         Only perform health check
    -s, --service NAME  Configure specific service only
    -t, --test          Test mode (dry run)

EXAMPLES:
    $0                  Run full integration
    $0 --check          Check service health only
    $0 --service sonarr Configure only Sonarr

SERVICES:
    - Sonarr (TV Shows)
    - Radarr (Movies)
    - Lidarr (Music)
    - Readarr (Books)
    - Prowlarr (Indexers)
    - Bazarr (Subtitles)
    - Jellyfin (Media Server)
    - qBittorrent (Download Client)

EOF
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -c|--check)
        perform_health_check
        exit 0
        ;;
    -s|--service)
        if [[ -n "${2:-}" ]]; then
            check_service "$2"
        else
            echo "Error: Service name required"
            exit 1
        fi
        exit 0
        ;;
    -t|--test)
        log "INFO" "Running in test mode (dry run)"
        echo "Test mode not implemented yet"
        exit 0
        ;;
    *)
        main
        ;;
esac