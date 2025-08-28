#!/bin/bash

# qBittorrent Configuration Script
# Configures qBittorrent with optimal settings for media server integration
# Author: API Integration Specialist
# Date: $(date)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# qBittorrent settings
QB_URL="http://localhost:8080"
QB_USERNAME="admin"
QB_PASSWORD="adminadmin"
QB_CONFIG_DIR="./qbittorrent-config"

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

# Login to qBittorrent
qb_login() {
    log "Logging into qBittorrent at $QB_URL"
    
    local cookie_jar=$(mktemp)
    local login_response
    
    login_response=$(curl -s -c "$cookie_jar" -d "username=$QB_USERNAME&password=$QB_PASSWORD" "$QB_URL/api/v2/auth/login")
    
    if [ "$login_response" = "Ok." ]; then
        log "Successfully logged into qBittorrent"
        echo "$cookie_jar"
        return 0
    else
        error "Failed to login to qBittorrent. Check credentials."
        rm -f "$cookie_jar"
        return 1
    fi
}

# Set qBittorrent preferences
set_qb_preferences() {
    local cookie_jar="$1"
    
    log "Configuring qBittorrent preferences"
    
    # Configure optimal settings for media server
    curl -s -b "$cookie_jar" -d 'json={
        "save_path": "/downloads/complete",
        "temp_path": "/downloads/incomplete",
        "temp_path_enabled": true,
        "scan_dirs": {"/downloads/watch": 1},
        "export_dir": "/downloads/torrents",
        "export_dir_fin": "/downloads/completed-torrents",
        "preallocate_all": false,
        "incomplete_files_ext": true,
        "auto_delete_mode": 1,
        "torrent_changed_tmout": 60,
        "save_resume_data_interval": 60,
        "recheck_completed_torrents": false,
        "resolve_peer_countries": true,
        "reannounce_when_address_changed": true,
        "listen_port": 6881,
        "upnp": true,
        "random_port": false,
        "dl_limit": 0,
        "up_limit": 0,
        "max_connec": 200,
        "max_connec_per_torrent": 100,
        "max_uploads": 20,
        "max_uploads_per_torrent": 4,
        "stop_tracker_timeout": 5,
        "enable_piece_extent_affinity": false,
        "bittorrent_protocol": 0,
        "limit_utp_rate": true,
        "limit_tcp_overhead": true,
        "limit_lan_peers": true,
        "alt_dl_limit": 0,
        "alt_up_limit": 0,
        "scheduler_enabled": false,
        "schedule_from_hour": 8,
        "schedule_from_min": 0,
        "schedule_to_hour": 20,
        "schedule_to_min": 0,
        "scheduler_days": 0,
        "dht": true,
        "dhtSameAsBT": true,
        "dht_port": 6881,
        "pex": true,
        "lsd": true,
        "encryption": 0,
        "anonymous_mode": false,
        "proxy_type": 0,
        "proxy_ip": "",
        "proxy_port": 8080,
        "proxy_peer_connections": false,
        "proxy_auth_enabled": false,
        "proxy_username": "",
        "proxy_password": "",
        "proxy_torrents_only": false,
        "ip_filter_enabled": false,
        "ip_filter_path": "",
        "ip_filter_trackers": false,
        "web_ui_domain_list": "*",
        "web_ui_address": "*",
        "web_ui_port": 8080,
        "web_ui_upnp": false,
        "web_ui_username": "admin",
        "web_ui_password": "",
        "web_ui_csrf_protection_enabled": true,
        "web_ui_clickjacking_protection_enabled": true,
        "web_ui_secure_cookie_enabled": true,
        "web_ui_max_auth_fail_count": 5,
        "web_ui_ban_duration": 3600,
        "web_ui_session_timeout": 3600,
        "web_ui_host_header_validation_enabled": false,
        "bypass_local_auth": false,
        "bypass_auth_subnet_whitelist_enabled": false,
        "bypass_auth_subnet_whitelist": "",
        "alternative_webui_enabled": false,
        "alternative_webui_path": "",
        "use_https": false,
        "ssl_key": "",
        "ssl_cert": "",
        "web_ui_https_key_path": "",
        "web_ui_https_cert_path": "",
        "dyndns_enabled": false,
        "dyndns_service": 0,
        "dyndns_username": "",
        "dyndns_password": "",
        "dyndns_domain": "changeme.dyndns.org",
        "rss_refresh_interval": 30,
        "rss_max_articles_per_feed": 50,
        "rss_processing_enabled": false,
        "rss_auto_downloading_enabled": false,
        "rss_download_repack_proper_episodes": true,
        "rss_smart_episode_filters": "",
        "add_trackers_enabled": false,
        "add_trackers": "",
        "web_ui_use_custom_http_headers_enabled": false,
        "web_ui_custom_http_headers": "",
        "max_seeding_time_enabled": false,
        "max_seeding_time": 10080,
        "announce_ip": "",
        "announce_to_all_tiers": true,
        "announce_to_all_trackers": false,
        "async_io_threads": 10,
        "banned_ips": "",
        "checking_memory_use": 32,
        "current_interface_address": "",
        "current_network_interface": "",
        "disk_cache": 64,
        "disk_cache_ttl": 60,
        "embedded_tracker_port": 9000,
        "enable_coalesce_read_write": true,
        "enable_embedded_tracker": false,
        "enable_multi_connections_from_same_ip": false,
        "enable_os_cache": true,
        "enable_upload_suggestions": false,
        "file_pool_size": 40,
        "outgoing_ports_max": 0,
        "outgoing_ports_min": 0,
        "recheck_completed_torrents": false,
        "resolve_peer_countries": false,
        "save_resume_data_interval": 60,
        "send_buffer_low_watermark": 10,
        "send_buffer_watermark": 500,
        "send_buffer_watermark_factor": 50,
        "socket_backlog_size": 30,
        "upload_choking_algorithm": 1,
        "upload_slots_behavior": 0,
        "upnp_lease_duration": 0,
        "utp_tcp_mixed_mode": 0
    }' "$QB_URL/api/v2/app/setPreferences"
    
    if [ $? -eq 0 ]; then
        log "qBittorrent preferences configured successfully"
    else
        error "Failed to configure qBittorrent preferences"
        return 1
    fi
}

# Create categories for different media types
create_categories() {
    local cookie_jar="$1"
    
    log "Creating media categories in qBittorrent"
    
    # Create categories
    local categories=("movies" "tv" "music" "books" "software")
    local save_paths=("/movies" "/tv" "/music" "/books" "/software")
    
    for i in "${!categories[@]}"; do
        local category="${categories[$i]}"
        local save_path="/downloads/${save_paths[$i]}"
        
        curl -s -b "$cookie_jar" -d "category=$category&savePath=$save_path" "$QB_URL/api/v2/torrents/createCategory"
        log "Created category: $category (saves to $save_path)"
    done
}

# Configure RSS auto-downloading rules
configure_rss_rules() {
    local cookie_jar="$1"
    
    log "Configuring RSS auto-downloading rules"
    
    # Enable RSS processing
    curl -s -b "$cookie_jar" -d 'json={"rss_processing_enabled": true, "rss_auto_downloading_enabled": true}' "$QB_URL/api/v2/app/setPreferences"
    
    log "RSS auto-downloading enabled"
}

# Set up speed limits and scheduling
configure_speed_limits() {
    local cookie_jar="$1"
    
    log "Configuring speed limits and scheduling"
    
    # Set alternative speed limits for daytime (optional)
    curl -s -b "$cookie_jar" -d 'json={
        "scheduler_enabled": true,
        "schedule_from_hour": 8,
        "schedule_from_min": 0,
        "schedule_to_hour": 22,
        "schedule_to_min": 0,
        "scheduler_days": 31,
        "alt_dl_limit": 5000,
        "alt_up_limit": 1000
    }' "$QB_URL/api/v2/app/setPreferences"
    
    log "Speed limits configured (alternative limits during 8:00-22:00)"
}

# Configure connection settings
configure_connections() {
    local cookie_jar="$1"
    
    log "Configuring connection settings"
    
    curl -s -b "$cookie_jar" -d 'json={
        "listen_port": 6881,
        "random_port": false,
        "upnp": true,
        "max_connec": 300,
        "max_connec_per_torrent": 100,
        "max_uploads": 50,
        "max_uploads_per_torrent": 10,
        "dht": true,
        "pex": true,
        "lsd": true,
        "encryption": 0
    }' "$QB_URL/api/v2/app/setPreferences"
    
    log "Connection settings optimized for media server usage"
}

# Wait for qBittorrent to be ready
wait_for_qbittorrent() {
    local max_attempts=30
    local attempt=1
    
    log "Waiting for qBittorrent to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 3 "$QB_URL/api/v2/app/version" > /dev/null 2>&1; then
            log "qBittorrent is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    error "qBittorrent failed to start after $((max_attempts * 2)) seconds"
    return 1
}

# Main configuration function
main() {
    echo -e "${BLUE}=== QBITTORRENT CONFIGURATION ===${NC}"
    log "Starting qBittorrent configuration for media server integration"
    
    # Wait for qBittorrent to be ready
    if ! wait_for_qbittorrent; then
        error "Cannot configure qBittorrent - service not accessible"
        exit 1
    fi
    
    # Login to qBittorrent
    local cookie_jar
    if ! cookie_jar=$(qb_login); then
        error "Cannot configure qBittorrent - login failed"
        exit 1
    fi
    
    # Configure qBittorrent
    set_qb_preferences "$cookie_jar"
    create_categories "$cookie_jar"
    configure_rss_rules "$cookie_jar"
    configure_speed_limits "$cookie_jar"
    configure_connections "$cookie_jar"
    
    # Cleanup
    rm -f "$cookie_jar"
    
    log "qBittorrent configuration completed successfully!"
    
    # Display summary
    echo -e "\n${BLUE}=== CONFIGURATION SUMMARY ===${NC}"
    echo -e "${GREEN}✓${NC} Download directories configured"
    echo -e "${GREEN}✓${NC} Media categories created (movies, tv, music, books)"
    echo -e "${GREEN}✓${NC} Connection settings optimized"
    echo -e "${GREEN}✓${NC} RSS auto-downloading enabled"
    echo -e "${GREEN}✓${NC} Speed scheduling configured"
    
    echo -e "\n${BLUE}Default Categories:${NC}"
    echo -e "• movies  → /downloads/movies"
    echo -e "• tv      → /downloads/tv"
    echo -e "• music   → /downloads/music"
    echo -e "• books   → /downloads/books"
    
    echo -e "\n${BLUE}Web Interface:${NC} $QB_URL"
    echo -e "${BLUE}Username:${NC} $QB_USERNAME"
    echo -e "${BLUE}Password:${NC} $QB_PASSWORD"
    
    echo -e "\n${YELLOW}Note: Change the default password in qBittorrent settings!${NC}"
}

# Check if qBittorrent is accessible
if ! curl -s --connect-timeout 5 "$QB_URL" > /dev/null 2>&1; then
    error "qBittorrent is not accessible at $QB_URL"
    error "Please ensure the qBittorrent container is running"
    exit 1
fi

# Run main function
main "$@"