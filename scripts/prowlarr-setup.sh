#!/usr/bin/env bash
set -euo pipefail

# Prowlarr Indexer Setup Script - 2025 Edition
# Automatically configures popular public and private indexers
# Author: Claude Code Assistant
# Version: 2025.1.0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROWLARR_URL="http://localhost:9696"
LOG_FILE="$PROJECT_DIR/logs/prowlarr-setup.log"

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

# Info message
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
    log "INFO: $1"
}

# Get Prowlarr API key
get_prowlarr_api() {
    local config_path="$PROJECT_DIR/config/prowlarr"
    
    # Wait for config to be created
    local attempts=0
    while [ ! -f "$config_path/config.xml" ] && [ $attempts -lt 30 ]; do
        echo -n "."
        sleep 2
        ((attempts++))
    done
    
    if [ -f "$config_path/config.xml" ]; then
        grep -o '<ApiKey>[^<]*</ApiKey>' "$config_path/config.xml" | sed 's/<[^>]*>//g'
    else
        echo ""
    fi
}

# Wait for Prowlarr to be ready
wait_for_prowlarr() {
    local max_attempts=30
    local attempt=1
    
    info "Waiting for Prowlarr to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$PROWLARR_URL" | grep -q "200\|302\|401"; then
            success "Prowlarr is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    error_exit "Prowlarr failed to start within $((max_attempts * 2)) seconds"
}

# Add indexer to Prowlarr
add_indexer() {
    local indexer_data="$1"
    local api_key="$2"
    local indexer_name="$3"
    
    info "Adding indexer: $indexer_name"
    
    local response
    response=$(curl -s -w "\n%{http_code}" -X POST "$PROWLARR_URL/api/v1/indexer" \
        -H "X-Api-Key: $api_key" \
        -H "Content-Type: application/json" \
        -d "$indexer_data")
    
    local http_code
    http_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "201" ] || [ "$http_code" = "200" ]; then
        success "Added $indexer_name"
    else
        echo -e "${YELLOW}⚠️  Failed to add $indexer_name (HTTP $http_code)${NC}"
        echo "Response: $body"
        log "Failed to add $indexer_name: HTTP $http_code - $body"
    fi
}

# Configure public indexers
configure_public_indexers() {
    local api_key="$1"
    
    info "Configuring public indexers..."
    
    # YTS - Movies
    local yts_config='{
        "enable": true,
        "name": "YTS",
        "implementation": "YTS",
        "implementationName": "YTS",
        "configContract": "YTSSettings",
        "fields": [
            {"name": "baseUrl", "value": "https://yts.mx"},
            {"name": "minimumSeeders", "value": 1}
        ],
        "tags": [],
        "priority": 25
    }'
    add_indexer "$yts_config" "$api_key" "YTS"
    
    # EZTV - TV Shows
    local eztv_config='{
        "enable": true,
        "name": "EZTV",
        "implementation": "EZTV",
        "implementationName": "EZTV",
        "configContract": "EZTVSettings",
        "fields": [
            {"name": "baseUrl", "value": "https://eztv.re"},
            {"name": "minimumSeeders", "value": 1}
        ],
        "tags": [],
        "priority": 25
    }'
    add_indexer "$eztv_config" "$api_key" "EZTV"
    
    # The Pirate Bay
    local tpb_config='{
        "enable": true,
        "name": "The Pirate Bay",
        "implementation": "PirateBay",
        "implementationName": "PirateBay",
        "configContract": "PirateBaySettings",
        "fields": [
            {"name": "baseUrl", "value": "https://thepiratebay.org"},
            {"name": "minimumSeeders", "value": 2}
        ],
        "tags": [],
        "priority": 20
    }'
    add_indexer "$tpb_config" "$api_key" "The Pirate Bay"
    
    # 1337x
    local leetx_config='{
        "enable": true,
        "name": "1337x",
        "implementation": "1337x",
        "implementationName": "1337x",
        "configContract": "1337xSettings",
        "fields": [
            {"name": "baseUrl", "value": "https://1337x.to"},
            {"name": "minimumSeeders", "value": 2}
        ],
        "tags": [],
        "priority": 25
    }'
    add_indexer "$leetx_config" "$api_key" "1337x"
    
    # RARBG (backup mirrors)
    local rarbg_config='{
        "enable": true,
        "name": "RARBG",
        "implementation": "RARBG",
        "implementationName": "RARBG",
        "configContract": "RARBGSettings",
        "fields": [
            {"name": "baseUrl", "value": "https://rarbgmirror.org"},
            {"name": "minimumSeeders", "value": 2}
        ],
        "tags": [],
        "priority": 30
    }'
    add_indexer "$rarbg_config" "$api_key" "RARBG"
    
    # Torlock
    local torlock_config='{
        "enable": true,
        "name": "Torlock",
        "implementation": "Torlock",
        "implementationName": "Torlock",
        "configContract": "TorlockSettings",
        "fields": [
            {"name": "baseUrl", "value": "https://www.torlock.com"},
            {"name": "minimumSeeders", "value": 1}
        ],
        "tags": [],
        "priority": 20
    }'
    add_indexer "$torlock_config" "$api_key" "Torlock"
    
    # TorrentGalaxy
    local tgx_config='{
        "enable": true,
        "name": "TorrentGalaxy",
        "implementation": "TorrentGalaxy",
        "implementationName": "TorrentGalaxy",
        "configContract": "TorrentGalaxySettings",
        "fields": [
            {"name": "baseUrl", "value": "https://torrentgalaxy.to"},
            {"name": "minimumSeeders", "value": 1}
        ],
        "tags": [],
        "priority": 25
    }'
    add_indexer "$tgx_config" "$api_key" "TorrentGalaxy"
    
    success "Public indexers configuration completed"
}

# Configure search settings
configure_search_settings() {
    local api_key="$1"
    
    info "Configuring search settings..."
    
    # Get current config
    local current_config
    current_config=$(curl -s "$PROWLARR_URL/api/v1/config/indexer" -H "X-Api-Key: $api_key")
    
    # Update with better search settings
    local updated_config
    updated_config=$(echo "$current_config" | jq '.
        | .maximumSize = 10240
        | .minimumAge = 0
        | .retention = 2555
        | .rssSyncInterval = 60
        | .preferredWords = "bluray,remux,web-dl,webrip"
        | .forbiddenWords = "cam,ts,tc,dvdscr,hdcam,hdts"')
    
    curl -s -X PUT "$PROWLARR_URL/api/v1/config/indexer" \
        -H "X-Api-Key: $api_key" \
        -H "Content-Type: application/json" \
        -d "$updated_config" >/dev/null
    
    success "Search settings configured"
}

# Test indexers
test_indexers() {
    local api_key="$1"
    
    info "Testing indexer connectivity..."
    
    # Get all indexers
    local indexers
    indexers=$(curl -s "$PROWLARR_URL/api/v1/indexer" -H "X-Api-Key: $api_key")
    
    # Test each indexer
    echo "$indexers" | jq -r '.[].id' | while read -r indexer_id; do
        local indexer_name
        indexer_name=$(echo "$indexers" | jq -r ".[] | select(.id == $indexer_id) | .name")
        
        echo -n "Testing $indexer_name... "
        
        local test_result
        test_result=$(curl -s -X POST "$PROWLARR_URL/api/v1/indexer/test/$indexer_id" \
            -H "X-Api-Key: $api_key" -H "Content-Type: application/json")
        
        if echo "$test_result" | jq -e '.isValid' >/dev/null 2>&1; then
            if [ "$(echo "$test_result" | jq -r '.isValid')" = "true" ]; then
                echo -e "${GREEN}✅ OK${NC}"
            else
                echo -e "${RED}❌ Failed${NC}"
                local error_message
                error_message=$(echo "$test_result" | jq -r '.validationFailures[0].errorMessage // "Unknown error"')
                echo "   Error: $error_message"
            fi
        else
            echo -e "${YELLOW}⚠️  Unknown${NC}"
        fi
    done
}

# Create indexer backup
backup_indexers() {
    local api_key="$1"
    
    info "Creating indexer configuration backup..."
    
    local backup_dir="$PROJECT_DIR/config/backups"
    mkdir -p "$backup_dir"
    
    local backup_file="$backup_dir/prowlarr-indexers-$(date +%Y%m%d-%H%M%S).json"
    
    curl -s "$PROWLARR_URL/api/v1/indexer" -H "X-Api-Key: $api_key" > "$backup_file"
    
    success "Indexer configuration backed up to: $backup_file"
}

# Private tracker template generator
generate_private_tracker_template() {
    local template_file="$PROJECT_DIR/config/prowlarr/private-trackers-template.json"
    
    info "Generating private tracker configuration template..."
    
    cat > "$template_file" << 'EOF'
{
    "private_trackers": [
        {
            "name": "PassThePopcorn",
            "implementation": "PassThePopcorn",
            "note": "Excellent for movies, requires invitation",
            "required_fields": ["username", "password", "passkey"],
            "categories": ["Movies"],
            "priority": 50
        },
        {
            "name": "BroadcastTheNet",
            "implementation": "BroadcastTheNet",
            "note": "Premier TV tracker, requires invitation",
            "required_fields": ["apikey"],
            "categories": ["TV"],
            "priority": 50
        },
        {
            "name": "What.CD / RED",
            "implementation": "Redacted",
            "note": "Music tracker, successor to What.CD",
            "required_fields": ["username", "password"],
            "categories": ["Music"],
            "priority": 50
        },
        {
            "name": "IPTorrents",
            "implementation": "IPTorrents",
            "note": "General tracker with good retention",
            "required_fields": ["username", "password"],
            "categories": ["Movies", "TV", "Music"],
            "priority": 40
        },
        {
            "name": "TorrentLeech",
            "implementation": "TorrentLeech",
            "note": "General tracker, good for recent content",
            "required_fields": ["username", "password"],
            "categories": ["Movies", "TV", "Music"],
            "priority": 40
        }
    ],
    "instructions": {
        "setup": [
            "1. Obtain invitations to private trackers",
            "2. Create accounts and note down credentials",
            "3. Add indexers manually in Prowlarr web interface",
            "4. Test connectivity before enabling",
            "5. Set appropriate priority levels"
        ],
        "best_practices": [
            "Keep good ratio on all trackers",
            "Enable RSS feeds for automated downloading",
            "Set reasonable retry limits",
            "Monitor tracker announcements for rules changes",
            "Backup credentials securely"
        ]
    }
}
EOF
    
    success "Private tracker template created: $template_file"
}

# Main execution
main() {
    echo -e "${PURPLE}🔍 Prowlarr Indexer Setup - 2025 Edition${NC}"
    echo "=========================================="
    echo ""
    
    # Check if Prowlarr is configured in docker-compose
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        error_exit "docker-compose.yml not found. Please run from the project root directory."
    fi
    
    if ! grep -q "prowlarr:" "$PROJECT_DIR/docker-compose.yml"; then
        error_exit "Prowlarr not found in docker-compose.yml"
    fi
    
    # Start Prowlarr if not running
    info "Ensuring Prowlarr is running..."
    cd "$PROJECT_DIR"
    docker-compose up -d prowlarr
    
    # Wait for Prowlarr to be ready
    wait_for_prowlarr
    
    # Give Prowlarr time to initialize
    info "Waiting for Prowlarr to initialize..."
    sleep 10
    
    # Get API key
    info "Retrieving Prowlarr API key..."
    local api_key
    api_key=$(get_prowlarr_api)
    
    if [ -z "$api_key" ]; then
        error_exit "Could not retrieve Prowlarr API key. Please check that Prowlarr is fully initialized."
    fi
    
    success "API key retrieved successfully"
    
    # Create backup before making changes
    backup_indexers "$api_key"
    
    # Configure indexers
    configure_public_indexers "$api_key"
    
    # Configure search settings
    configure_search_settings "$api_key"
    
    # Wait a moment for configurations to take effect
    sleep 5
    
    # Test indexers
    test_indexers "$api_key"
    
    # Generate private tracker template
    generate_private_tracker_template
    
    echo ""
    echo -e "${GREEN}🎉 Prowlarr Setup Complete!${NC}"
    echo "============================"
    echo ""
    echo -e "${BLUE}Configured Indexers:${NC}"
    echo "• YTS (Movies)"
    echo "• EZTV (TV Shows)"
    echo "• The Pirate Bay (General)"
    echo "• 1337x (General)"
    echo "• RARBG (General)"
    echo "• Torlock (General)"
    echo "• TorrentGalaxy (General)"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Visit Prowlarr: $PROWLARR_URL"
    echo "2. Review and test indexers in the web interface"
    echo "3. Add private trackers manually (see template in config/prowlarr/)"
    echo "4. Configure application sync to Sonarr/Radarr"
    echo "5. Test search functionality"
    echo ""
    echo -e "${BLUE}Private Trackers:${NC}"
    echo "• Template created: $PROJECT_DIR/config/prowlarr/private-trackers-template.json"
    echo "• Add private trackers manually for better quality and retention"
    echo ""
    echo -e "${YELLOW}Important Notes:${NC}"
    echo "• Some indexers may be blocked in your region"
    echo "• Use a VPN for better indexer connectivity"
    echo "• Keep indexer lists updated for best results"
    echo "• Respect tracker rules and maintain good ratios"
    echo ""
    
    info "Prowlarr setup completed successfully"
}

# Run main function
main "$@"