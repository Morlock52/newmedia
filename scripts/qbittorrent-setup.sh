#!/usr/bin/env bash
set -euo pipefail

# qBittorrent Setup Script - 2025 Edition
# Configures categories, settings, and optimization for media server workflows
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
QBT_URL="http://localhost:8080"
QBT_USERNAME="admin"
QBT_PASSWORD="adminpass"
LOG_FILE="$PROJECT_DIR/logs/qbittorrent-setup.log"

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

# Wait for qBittorrent to be ready
wait_for_qbittorrent() {
    local max_attempts=30
    local attempt=1
    
    info "Waiting for qBittorrent to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$QBT_URL" | grep -q "200\|302\|401"; then
            success "qBittorrent is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    error_exit "qBittorrent failed to start within $((max_attempts * 2)) seconds"
}

# Login to qBittorrent and get session cookie
qbt_login() {
    local cookie_jar=$(mktemp)
    
    info "Logging into qBittorrent..."
    
    # Perform login
    local login_response
    login_response=$(curl -s -c "$cookie_jar" -d "username=$QBT_USERNAME&password=$QBT_PASSWORD" \
        "$QBT_URL/api/v2/auth/login")
    
    if [ "$login_response" = "Ok." ]; then
        success "Login successful"
        echo "$cookie_jar"
    else
        error_exit "Login failed: $login_response"
    fi
}

# Create qBittorrent categories
create_categories() {
    local cookie_jar="$1"
    
    info "Creating qBittorrent categories..."
    
    # Define categories with save paths
    declare -A categories=(
        ["tv-sonarr"]="/downloads/tv"
        ["movies-radarr"]="/downloads/movies"
        ["music-lidarr"]="/downloads/music"
        ["audiobooks-readarr"]="/downloads/audiobooks"
        ["books-readarr"]="/downloads/books"
        ["anime-sonarr"]="/downloads/anime"
        ["documentary"]="/downloads/documentaries"
        ["software"]="/downloads/software"
        ["games"]="/downloads/games"
        ["other"]="/downloads/other"
    )
    
    for category in "${!categories[@]}"; do
        local save_path="${categories[$category]}"
        
        info "Creating category: $category -> $save_path"
        
        local response
        response=$(curl -s -b "$cookie_jar" -X POST "$QBT_URL/api/v2/torrents/createCategory" \
            --data-urlencode "category=$category" \
            --data-urlencode "savePath=$save_path")
        
        # Empty response means success for this API
        if [ -z "$response" ]; then
            success "Category created: $category"
        else
            echo -e "${YELLOW}⚠️  Category response for $category: $response${NC}"
        fi
    done
    
    success "Categories creation completed"
}

# Configure global preferences
configure_preferences() {
    local cookie_jar="$1"
    
    info "Configuring qBittorrent global preferences..."
    
    # Create optimized preferences JSON
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
    "temp_path_enabled": true,
    "torrent_changed_tmm_enabled": false,
    "upnp": true,
    "use_subcategories": true,
    "web_ui_domain_list": "*",
    "web_ui_host_header_validation_enabled": false,
    "alternative_webui_enabled": false,
    "banned_IPs": "",
    "enable_upload_suggestions": false,
    "file_log_enabled": true,
    "file_log_backup_enabled": true,
    "file_log_max_size": 67108864,
    "file_log_delete_old": true,
    "recheck_completed_torrents": false,
    "resolve_peer_countries": true,
    "save_resume_data_interval": 60,
    "torrent_content_layout": "Original",
    "start_paused_enabled": false
}
EOF
    
    local response
    response=$(curl -s -b "$cookie_jar" -X POST "$QBT_URL/api/v2/app/setPreferences" \
        --data-urlencode "json@/tmp/qbt_prefs.json")
    
    rm -f /tmp/qbt_prefs.json
    
    if [ -z "$response" ]; then
        success "Global preferences configured"
    else
        echo -e "${YELLOW}⚠️  Preferences response: $response${NC}"
    fi
}

# Setup RSS automation (if needed)
setup_rss_automation() {
    local cookie_jar="$1"
    
    info "Setting up RSS automation preferences..."
    
    # Configure RSS settings
    cat > /tmp/qbt_rss.json << 'EOF'
{
    "rss_processing_enabled": false,
    "rss_refresh_interval": 30,
    "rss_max_articles_per_feed": 50,
    "rss_download_repack_proper_episodes": true,
    "rss_smart_episode_filters": "s(\\d+)e(\\d+), (\\d+)x(\\d+), \"(\\d{4}[.\\-]\\d{1,2}[.\\-]\\d{1,2})\", \"(\\d{1,2}[.\\-]\\d{1,2}[.\\-]\\d{4})\""
}
EOF
    
    # Note: RSS automation is typically handled by Sonarr/Radarr, not qBittorrent directly
    rm -f /tmp/qbt_rss.json
    
    success "RSS automation preferences configured"
}

# Create directory structure
create_directory_structure() {
    info "Creating directory structure for downloads..."
    
    local base_dir="$PROJECT_DIR/data/torrents"
    local dirs=(
        "tv"
        "movies"
        "music"
        "audiobooks"
        "books"
        "anime"
        "documentaries"
        "software"
        "games"
        "other"
        "incomplete"
        "watch"
    )
    
    for dir in "${dirs[@]}"; do
        local full_path="$base_dir/$dir"
        if [ ! -d "$full_path" ]; then
            mkdir -p "$full_path"
            success "Created directory: $full_path"
        else
            info "Directory already exists: $full_path"
        fi
    done
    
    # Set proper permissions
    chmod -R 755 "$base_dir"
    success "Directory structure created and permissions set"
}

# Verify configuration
verify_configuration() {
    local cookie_jar="$1"
    
    info "Verifying qBittorrent configuration..."
    
    # Check categories
    local categories
    categories=$(curl -s -b "$cookie_jar" "$QBT_URL/api/v2/torrents/categories")
    
    if echo "$categories" | jq -e '.["tv-sonarr"]' >/dev/null 2>&1; then
        success "Categories verified"
    else
        echo -e "${YELLOW}⚠️  Some categories may not have been created properly${NC}"
    fi
    
    # Check preferences
    local prefs
    prefs=$(curl -s -b "$cookie_jar" "$QBT_URL/api/v2/app/preferences")
    
    local save_path
    save_path=$(echo "$prefs" | jq -r '.save_path // empty')
    
    if [ "$save_path" = "/downloads" ]; then
        success "Preferences verified"
    else
        echo -e "${YELLOW}⚠️  Some preferences may not have been set properly${NC}"
    fi
    
    # Check version and build info
    local version_info
    version_info=$(curl -s -b "$cookie_jar" "$QBT_URL/api/v2/app/version")
    
    info "qBittorrent version: $version_info"
}

# Create automation scripts
create_automation_scripts() {
    info "Creating automation helper scripts..."
    
    # Create a script to reset qBittorrent password if needed
    cat > "$PROJECT_DIR/scripts/reset-qbt-password.sh" << 'EOF'
#!/usr/bin/env bash
# Reset qBittorrent password
echo "Resetting qBittorrent password..."

docker-compose exec qbittorrent sh -c "
    if [ -f /config/qBittorrent/qBittorrent.conf ]; then
        # Backup original config
        cp /config/qBittorrent/qBittorrent.conf /config/qBittorrent/qBittorrent.conf.backup
        
        # Remove password line (forces default admin/adminpass)
        sed -i '/WebUI\\\\Password_PBKDF2/d' /config/qBittorrent/qBittorrent.conf
        
        echo 'Password reset to default (admin/adminpass)'
        echo 'Please restart qBittorrent container for changes to take effect'
    else
        echo 'qBittorrent config file not found'
    fi
"
EOF
    
    chmod +x "$PROJECT_DIR/scripts/reset-qbt-password.sh"
    
    # Create a script to check qBittorrent status
    cat > "$PROJECT_DIR/scripts/check-qbt-status.sh" << 'EOF'
#!/usr/bin/env bash
# Check qBittorrent status and configuration

echo "=== qBittorrent Status Check ==="
echo ""

# Check if container is running
if docker ps | grep -q qbittorrent; then
    echo "✅ Container: Running"
else
    echo "❌ Container: Not running"
    exit 1
fi

# Check web interface
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080" | grep -q "200\|302"; then
    echo "✅ Web Interface: Accessible"
else
    echo "❌ Web Interface: Not accessible"
fi

# Check API
if curl -s "http://localhost:8080/api/v2/app/version" >/dev/null; then
    echo "✅ API: Responsive"
    VERSION=$(curl -s "http://localhost:8080/api/v2/app/version")
    echo "   Version: $VERSION"
else
    echo "❌ API: Not responsive"
fi

echo ""
echo "Access qBittorrent at: http://localhost:8080"
echo "Default credentials: admin / adminpass"
EOF
    
    chmod +x "$PROJECT_DIR/scripts/check-qbt-status.sh"
    
    success "Automation scripts created"
}

# Cleanup temporary files
cleanup() {
    local cookie_jar="$1"
    
    if [ -f "$cookie_jar" ]; then
        rm -f "$cookie_jar"
    fi
}

# Main execution
main() {
    echo -e "${PURPLE}⬇️  qBittorrent Setup - 2025 Edition${NC}"
    echo "====================================="
    echo ""
    
    # Check if qBittorrent is configured in docker-compose
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        error_exit "docker-compose.yml not found. Please run from the project root directory."
    fi
    
    if ! grep -q "qbittorrent:" "$PROJECT_DIR/docker-compose.yml"; then
        error_exit "qBittorrent not found in docker-compose.yml"
    fi
    
    # Start qBittorrent if not running
    info "Ensuring qBittorrent is running..."
    cd "$PROJECT_DIR"
    docker-compose up -d qbittorrent
    
    # Wait for qBittorrent to be ready
    wait_for_qbittorrent
    
    # Give qBittorrent time to fully initialize
    info "Waiting for qBittorrent to fully initialize..."
    sleep 10
    
    # Create directory structure first
    create_directory_structure
    
    # Login and get session cookie
    local cookie_jar
    cookie_jar=$(qbt_login)
    
    # Ensure cleanup on exit
    trap "cleanup '$cookie_jar'" EXIT
    
    # Configure qBittorrent
    create_categories "$cookie_jar"
    configure_preferences "$cookie_jar"
    setup_rss_automation "$cookie_jar"
    
    # Verify configuration
    verify_configuration "$cookie_jar"
    
    # Create helper scripts
    create_automation_scripts
    
    echo ""
    echo -e "${GREEN}🎉 qBittorrent Setup Complete!${NC}"
    echo "================================="
    echo ""
    echo -e "${BLUE}Configuration Summary:${NC}"
    echo "• Categories: Created for TV, Movies, Music, Books, etc."
    echo "• Download Path: /downloads with subcategories"
    echo "• Incomplete Path: /downloads/incomplete"
    echo "• Max Connections: 200 total, 100 per torrent"
    echo "• Queueing: Enabled (5 active downloads, 10 total)"
    echo ""
    echo -e "${BLUE}Category Mapping:${NC}"
    echo "• tv-sonarr → /downloads/tv"
    echo "• movies-radarr → /downloads/movies" 
    echo "• music-lidarr → /downloads/music"
    echo "• audiobooks-readarr → /downloads/audiobooks"
    echo "• books-readarr → /downloads/books"
    echo ""
    echo -e "${BLUE}Access Information:${NC}"
    echo "• Web Interface: $QBT_URL"
    echo "• Username: $QBT_USERNAME"
    echo "• Password: $QBT_PASSWORD"
    echo ""
    echo -e "${BLUE}Helper Scripts Created:${NC}"
    echo "• $PROJECT_DIR/scripts/reset-qbt-password.sh"
    echo "• $PROJECT_DIR/scripts/check-qbt-status.sh"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Verify settings in the web interface"
    echo "2. Configure Sonarr/Radarr to use appropriate categories"
    echo "3. Test download functionality"
    echo "4. Monitor disk usage and performance"
    echo ""
    echo -e "${YELLOW}Important Notes:${NC}"
    echo "• Categories are automatically assigned by Sonarr/Radarr"
    echo "• Incomplete files are stored separately to avoid import issues"
    echo "• Adjust connection limits based on your internet capacity"
    echo "• Consider using a VPN for privacy and security"
    echo ""
    
    info "qBittorrent setup completed successfully"
}

# Run main function
main "$@"