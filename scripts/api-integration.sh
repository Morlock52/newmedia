#!/usr/bin/env bash
set -euo pipefail

# Media Stack API Integration Script - 2025 Edition
# Automates the configuration of Prowlarr → Sonarr/Radarr → qBittorrent → Overseerr
# Author: Claude Code Assistant
# Version: 2025.1.0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/logs/api-integration.log"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    log "ERROR: $1"
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}✅ $1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
    log "INFO: $1"
}

# Check if services are running
check_service() {
    local service="$1"
    local port="$2"
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -q "200\|302\|401"; then
        return 0
    else
        return 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    local service="$1"
    local port="$2"
    local max_attempts=30
    local attempt=1
    
    info "Waiting for $service to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if check_service "$service" "$port"; then
            success "$service is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    error_exit "$service failed to start within $((max_attempts * 2)) seconds"
}

# Extract API key from service
get_api_key() {
    local service="$1"
    local config_path="$2"
    
    # First try config.xml
    if [ -f "$config_path/config.xml" ]; then
        api_key=$(grep -o '<ApiKey>[^<]*</ApiKey>' "$config_path/config.xml" 2>/dev/null | sed 's/<[^>]*>//g' || echo "")
        if [ -n "$api_key" ]; then
            echo "$api_key"
            return 0
        fi
    fi
    
    # Try app.db for newer versions
    if [ -f "$config_path/app.db" ] && command -v sqlite3 >/dev/null; then
        api_key=$(sqlite3 "$config_path/app.db" "SELECT Value FROM Config WHERE Key='ApiKey';" 2>/dev/null || echo "")
        if [ -n "$api_key" ]; then
            echo "$api_key"
            return 0
        fi
    fi
    
    echo ""
}

# Configure qBittorrent via API
configure_qbittorrent() {
    local port="8080"
    local base_url="http://localhost:$port"
    
    info "Configuring qBittorrent categories and settings..."
    
    # Get CSRF token and login
    local cookie_jar=$(mktemp)
    
    # Login to qBittorrent
    curl -s -c "$cookie_jar" "$base_url/api/v2/auth/login" \
        --data "username=admin&password=adminpass" >/dev/null
    
    # Create categories for different media types
    local categories=("tv-sonarr" "movies-radarr" "music-lidarr" "books-readarr")
    
    for category in "${categories[@]}"; do
        info "Creating category: $category"
        curl -s -b "$cookie_jar" -X POST "$base_url/api/v2/torrents/createCategory" \
            --data "category=$category&savePath=/downloads/$category" >/dev/null
    done
    
    # Configure global settings
    info "Configuring qBittorrent global settings..."
    
    cat > /tmp/qbt_prefs.json << 'EOF'
{
    "auto_delete_mode": 0,
    "auto_tmm_enabled": false,
    "dht": true,
    "encryption": 0,
    "incomplete_files_ext": false,
    "listen_port": 6881,
    "max_connec": 200,
    "max_connec_per_torrent": 100,
    "max_uploads": 20,
    "max_uploads_per_torrent": 4,
    "pex": true,
    "preallocate_all": false,
    "queueing_enabled": true,
    "save_path": "/downloads",
    "temp_path": "/downloads/incomplete",
    "temp_path_enabled": true
}
EOF
    
    curl -s -b "$cookie_jar" -X POST "$base_url/api/v2/app/setPreferences" \
        --data-urlencode "json@/tmp/qbt_prefs.json" >/dev/null
    
    rm -f "$cookie_jar" /tmp/qbt_prefs.json
    success "qBittorrent configuration completed"
}

# Configure Prowlarr apps
configure_prowlarr_apps() {
    local prowlarr_api="$1"
    local sonarr_api="$2"
    local radarr_api="$3"
    
    info "Configuring Prowlarr application connections..."
    
    # Add Sonarr application
    cat > /tmp/sonarr_app.json << EOF
{
    "name": "Sonarr",
    "implementation": "Sonarr",
    "implementationName": "Sonarr",
    "fields": [
        {"name": "prowlarrUrl", "value": "http://prowlarr:9696"},
        {"name": "baseUrl", "value": "http://sonarr:8989"},
        {"name": "apiKey", "value": "$sonarr_api"},
        {"name": "syncCategories", "value": [5000, 5030, 5040, 5045, 5080]},
        {"name": "animeSyncCategories", "value": [5070]}
    ],
    "tags": [],
    "syncLevel": "fullSync"
}
EOF
    
    # Add Radarr application  
    cat > /tmp/radarr_app.json << EOF
{
    "name": "Radarr",
    "implementation": "Radarr",
    "implementationName": "Radarr", 
    "fields": [
        {"name": "prowlarrUrl", "value": "http://prowlarr:9696"},
        {"name": "baseUrl", "value": "http://radarr:7878"},
        {"name": "apiKey", "value": "$radarr_api"},
        {"name": "syncCategories", "value": [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060, 2070, 2080, 2090]}
    ],
    "tags": [],
    "syncLevel": "fullSync"
}
EOF
    
    # Send to Prowlarr
    curl -s -X POST "http://localhost:9696/api/v1/applications" \
        -H "X-Api-Key: $prowlarr_api" \
        -H "Content-Type: application/json" \
        -d @/tmp/sonarr_app.json >/dev/null
    
    curl -s -X POST "http://localhost:9696/api/v1/applications" \
        -H "X-Api-Key: $prowlarr_api" \
        -H "Content-Type: application/json" \
        -d @/tmp/radarr_app.json >/dev/null
    
    rm -f /tmp/sonarr_app.json /tmp/radarr_app.json
    success "Prowlarr applications configured"
}

# Configure Sonarr download client
configure_sonarr_download_client() {
    local sonarr_api="$1"
    
    info "Configuring Sonarr download client..."
    
    cat > /tmp/sonarr_qbt.json << 'EOF'
{
    "enable": true,
    "name": "qBittorrent",
    "implementation": "QBittorrent",
    "implementationName": "qBittorrent",
    "fields": [
        {"name": "host", "value": "qbittorrent"},
        {"name": "port", "value": 8080},
        {"name": "username", "value": "admin"},
        {"name": "password", "value": "adminpass"},
        {"name": "category", "value": "tv-sonarr"},
        {"name": "recentTvPriority", "value": 0},
        {"name": "olderTvPriority", "value": 0},
        {"name": "initialState", "value": 0},
        {"name": "sequentialOrder", "value": false},
        {"name": "firstAndLast", "value": false}
    ],
    "tags": [],
    "configContract": "QBittorrentSettings"
}
EOF
    
    curl -s -X POST "http://localhost:8989/api/v3/downloadclient" \
        -H "X-Api-Key: $sonarr_api" \
        -H "Content-Type: application/json" \
        -d @/tmp/sonarr_qbt.json >/dev/null
    
    rm -f /tmp/sonarr_qbt.json
    success "Sonarr download client configured"
}

# Configure Radarr download client
configure_radarr_download_client() {
    local radarr_api="$1"
    
    info "Configuring Radarr download client..."
    
    cat > /tmp/radarr_qbt.json << 'EOF'
{
    "enable": true,
    "name": "qBittorrent",
    "implementation": "QBittorrent",
    "implementationName": "qBittorrent",
    "fields": [
        {"name": "host", "value": "qbittorrent"},
        {"name": "port", "value": 8080},
        {"name": "username", "value": "admin"},
        {"name": "password", "value": "adminpass"},
        {"name": "category", "value": "movies-radarr"},
        {"name": "recentMoviePriority", "value": 0},
        {"name": "olderMoviePriority", "value": 0},
        {"name": "initialState", "value": 0},
        {"name": "sequentialOrder", "value": false},
        {"name": "firstAndLast", "value": false}
    ],
    "tags": [],
    "configContract": "QBittorrentSettings"
}
EOF
    
    curl -s -X POST "http://localhost:7878/api/v3/downloadclient" \
        -H "X-Api-Key: $radarr_api" \
        -H "Content-Type: application/json" \
        -d @/tmp/radarr_qbt.json >/dev/null
    
    rm -f /tmp/radarr_qbt.json
    success "Radarr download client configured"
}

# Configure Overseerr services
configure_overseerr() {
    local jellyfin_url="http://jellyfin:8096"
    local sonarr_api="$1"
    local radarr_api="$2"
    
    info "Configuring Overseerr services..."
    info "Note: Overseerr requires manual initial setup through the web interface"
    info "Please visit http://localhost:5055 to complete initial configuration"
    
    # Create configuration template
    cat > "$PROJECT_DIR/config/overseerr/service-config.json" << EOF
{
    "jellyfin": {
        "url": "$jellyfin_url",
        "note": "Add this as your media server in Overseerr setup"
    },
    "sonarr": {
        "url": "http://sonarr:8989",
        "apiKey": "$sonarr_api",
        "note": "Add this in Services → Sonarr"
    },
    "radarr": {
        "url": "http://radarr:7878", 
        "apiKey": "$radarr_api",
        "note": "Add this in Services → Radarr"
    }
}
EOF
    
    success "Overseerr configuration template created"
}

# Main execution
main() {
    echo -e "${PURPLE}🚀 Media Stack API Integration - 2025 Edition${NC}"
    echo "=================================================="
    echo ""
    
    # Check if running in correct directory
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        error_exit "docker-compose.yml not found. Please run from the project root directory."
    fi
    
    # Start services if not running
    info "Ensuring all services are running..."
    cd "$PROJECT_DIR"
    docker-compose up -d
    
    # Wait for services to be ready
    wait_for_service "qBittorrent" "8080"
    wait_for_service "Sonarr" "8989"
    wait_for_service "Radarr" "7878"
    wait_for_service "Prowlarr" "9696"
    wait_for_service "Overseerr" "5055"
    
    # Extract API keys
    info "Extracting API keys from services..."
    
    SONARR_API=$(get_api_key "Sonarr" "$PROJECT_DIR/config/sonarr")
    RADARR_API=$(get_api_key "Radarr" "$PROJECT_DIR/config/radarr")
    PROWLARR_API=$(get_api_key "Prowlarr" "$PROJECT_DIR/config/prowlarr")
    
    if [ -z "$SONARR_API" ] || [ -z "$RADARR_API" ] || [ -z "$PROWLARR_API" ]; then
        warning "Some API keys could not be extracted automatically"
        warning "Please check the services are fully initialized and try again"
        info "You can also manually configure the services using the web interfaces"
        exit 1
    fi
    
    success "API keys extracted successfully"
    
    # Save API keys to environment file
    cat > "$PROJECT_DIR/.env.apis" << EOF
# API Keys extracted on $(date)
SONARR_API_KEY="$SONARR_API"
RADARR_API_KEY="$RADARR_API"
PROWLARR_API_KEY="$PROWLARR_API"
EOF
    
    # Configure services
    configure_qbittorrent
    sleep 2
    
    configure_sonarr_download_client "$SONARR_API"
    sleep 2
    
    configure_radarr_download_client "$RADARR_API"
    sleep 2
    
    configure_prowlarr_apps "$PROWLARR_API" "$SONARR_API" "$RADARR_API"
    sleep 2
    
    configure_overseerr "$SONARR_API" "$RADARR_API"
    
    echo ""
    echo -e "${GREEN}🎉 Integration Complete!${NC}"
    echo "======================="
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Add indexers to Prowlarr: http://localhost:9696"
    echo "2. Complete Overseerr setup: http://localhost:5055"
    echo "3. Test the full workflow by requesting media in Overseerr"
    echo ""
    echo -e "${BLUE}Service URLs:${NC}"
    echo "• Jellyfin: http://localhost:8096"
    echo "• Sonarr: http://localhost:8989"
    echo "• Radarr: http://localhost:7878"
    echo "• Prowlarr: http://localhost:9696"
    echo "• qBittorrent: http://localhost:8080"
    echo "• Overseerr: http://localhost:5055"
    echo ""
    echo -e "${BLUE}API Keys saved to:${NC} $PROJECT_DIR/.env.apis"
    echo ""
    
    info "Integration script completed successfully"
}

# Run main function
main "$@"