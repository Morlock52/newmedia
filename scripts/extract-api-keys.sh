#!/bin/bash

# API Key Extraction Script for Media Server Services
# Extracts and displays API keys from various services
# Author: API Integration Specialist
# Date: $(date)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Config paths
SONARR_CONFIG="./sonarr-config"
RADARR_CONFIG="./radarr-config"
LIDARR_CONFIG="./lidarr-config"
PROWLARR_CONFIG="./prowlarr-config"
BAZARR_CONFIG="./bazarr-config"
JELLYFIN_CONFIG="./jellyfin-config"

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

# Extract API key from config.xml
extract_xml_api_key() {
    local config_file="$1"
    local service_name="$2"
    
    if [ ! -f "$config_file" ]; then
        warn "$service_name config file not found: $config_file"
        return 1
    fi
    
    local api_key
    api_key=$(grep -o '<ApiKey>[^<]*</ApiKey>' "$config_file" 2>/dev/null | sed 's/<ApiKey>\(.*\)<\/ApiKey>/\1/' || echo "")
    
    if [ -n "$api_key" ]; then
        echo "$api_key"
        return 0
    else
        warn "No API key found in $service_name config"
        return 1
    fi
}

# Extract Jellyfin API key from system.xml
extract_jellyfin_key() {
    local system_file="$JELLYFIN_CONFIG/config/system.xml"
    
    if [ ! -f "$system_file" ]; then
        warn "Jellyfin system.xml not found: $system_file"
        return 1
    fi
    
    # Look for API keys in system configuration
    local api_keys
    api_keys=$(grep -o '<AccessToken>[^<]*</AccessToken>' "$system_file" 2>/dev/null | sed 's/<AccessToken>\(.*\)<\/AccessToken>/\1/' || echo "")
    
    if [ -n "$api_keys" ]; then
        echo "$api_keys" | head -1
        return 0
    else
        warn "No API keys found in Jellyfin config"
        return 1
    fi
}

# Check service database for API configuration
check_service_database() {
    local service_name="$1"
    local db_path="$2"
    
    if [ ! -f "$db_path" ]; then
        warn "$service_name database not found: $db_path"
        return 1
    fi
    
    log "$service_name database found: $(ls -lh "$db_path" | awk '{print $5}')"
    
    # Basic database info (requires sqlite3)
    if command -v sqlite3 &> /dev/null; then
        local tables
        tables=$(sqlite3 "$db_path" ".tables" 2>/dev/null || echo "Cannot read database")
        echo "  Tables: $tables"
    fi
}

# Generate API key if service is running
generate_api_key() {
    local service_name="$1"
    local url="$2"
    
    log "Checking if $service_name is accessible at $url"
    
    if curl -s --connect-timeout 5 "$url/ping" > /dev/null 2>&1; then
        log "$service_name is running and accessible"
        log "You can access the web UI to generate/view API keys"
        return 0
    else
        warn "$service_name is not accessible at $url"
        return 1
    fi
}

# Display service information
show_service_info() {
    local service_name="$1"
    local config_path="$2"
    local url="$3"
    local api_key="$4"
    
    echo -e "\n${BLUE}=== $service_name ===${NC}"
    echo -e "Config Path: $config_path"
    echo -e "Service URL: $url"
    
    if [ -n "$api_key" ]; then
        echo -e "API Key: ${GREEN}${api_key}${NC}"
        echo -e "Status: ${GREEN}✓ Configured${NC}"
    else
        echo -e "API Key: ${RED}Not found${NC}"
        echo -e "Status: ${YELLOW}⚠ Needs configuration${NC}"
    fi
    
    # Check if service is running
    if curl -s --connect-timeout 3 "$url" > /dev/null 2>&1; then
        echo -e "Service Status: ${GREEN}Running${NC}"
    else
        echo -e "Service Status: ${RED}Not accessible${NC}"
    fi
}

# Create API keys reference file
create_api_keys_file() {
    local sonarr_key="$1"
    local radarr_key="$2"
    local lidarr_key="$3"
    local prowlarr_key="$4"
    local bazarr_key="$5"
    local jellyfin_key="$6"
    
    cat > "./api-keys.env" << EOF
# Media Server API Keys
# Generated on $(date)
# Use these keys for service integrations

# *ARR Services API Keys
SONARR_API_KEY="$sonarr_key"
RADARR_API_KEY="$radarr_key"
LIDARR_API_KEY="$lidarr_key"
PROWLARR_API_KEY="$prowlarr_key"
BAZARR_API_KEY="$bazarr_key"

# Media Server API Keys
JELLYFIN_API_KEY="$jellyfin_key"
PLEX_TOKEN=""  # Manual configuration required

# Service URLs (Internal Docker Network)
SONARR_URL="http://sonarr:8989"
RADARR_URL="http://radarr:7878"
LIDARR_URL="http://lidarr:8686"
PROWLARR_URL="http://prowlarr:9696"
BAZARR_URL="http://bazarr:6767"
JELLYFIN_URL="http://jellyfin:8096"
PLEX_URL="http://plex:32400"
QBITTORRENT_URL="http://qbittorrent:8080"
JELLYSEERR_URL="http://jellyseerr:5055"
OVERSEERR_URL="http://overseerr:5055"

# External URLs (for browser access)
EXT_SONARR_URL="http://localhost:8989"
EXT_RADARR_URL="http://localhost:7878"
EXT_PROWLARR_URL="http://localhost:9696"
EXT_JELLYFIN_URL="http://localhost:8096"
EXT_QBITTORRENT_URL="http://localhost:8080"

# qBittorrent Default Credentials
QB_USERNAME="admin"
QB_PASSWORD="adminadmin"  # Change this in qBittorrent settings

# Transmission (through Gluetun VPN)
TRANSMISSION_URL="http://localhost:9091"
TRANSMISSION_USER=""  # Configure if authentication enabled
TRANSMISSION_PASS=""

# SABnzbd
SABNZBD_URL="http://localhost:8081"
SABNZBD_API_KEY=""  # Extract from SABnzbd config

# Usenet Server Configuration (Example - Configure as needed)
USENET_SERVER=""
USENET_PORT="563"
USENET_SSL="true"
USENET_USERNAME=""
USENET_PASSWORD=""
EOF
    
    log "API keys saved to ./api-keys.env"
}

# Main function
main() {
    echo -e "${BLUE}=== MEDIA SERVER API KEY EXTRACTION ===${NC}"
    echo -e "${BLUE}Extracting API keys from all configured services...${NC}\n"
    
    # Extract API keys
    local sonarr_key=""
    local radarr_key=""
    local lidarr_key=""
    local prowlarr_key=""
    local bazarr_key=""
    local jellyfin_key=""
    
    # Sonarr
    if sonarr_key=$(extract_xml_api_key "$SONARR_CONFIG/config.xml" "Sonarr"); then
        log "Sonarr API key extracted successfully"
    else
        generate_api_key "Sonarr" "http://localhost:8989"
    fi
    
    # Radarr
    if radarr_key=$(extract_xml_api_key "$RADARR_CONFIG/config.xml" "Radarr"); then
        log "Radarr API key extracted successfully"
    else
        generate_api_key "Radarr" "http://localhost:7878"
    fi
    
    # Lidarr
    if lidarr_key=$(extract_xml_api_key "$LIDARR_CONFIG/config.xml" "Lidarr"); then
        log "Lidarr API key extracted successfully"
    else
        generate_api_key "Lidarr" "http://localhost:8686"
    fi
    
    # Prowlarr
    if prowlarr_key=$(extract_xml_api_key "$PROWLARR_CONFIG/config.xml" "Prowlarr"); then
        log "Prowlarr API key extracted successfully"
    else
        generate_api_key "Prowlarr" "http://localhost:9696"
    fi
    
    # Bazarr
    if bazarr_key=$(extract_xml_api_key "$BAZARR_CONFIG/config.xml" "Bazarr"); then
        log "Bazarr API key extracted successfully"
    else
        generate_api_key "Bazarr" "http://localhost:6767"
    fi
    
    # Jellyfin
    if jellyfin_key=$(extract_jellyfin_key); then
        log "Jellyfin API key extracted successfully"
    else
        generate_api_key "Jellyfin" "http://localhost:8096"
    fi
    
    # Display service information
    show_service_info "Sonarr" "$SONARR_CONFIG" "http://localhost:8989" "$sonarr_key"
    show_service_info "Radarr" "$RADARR_CONFIG" "http://localhost:7878" "$radarr_key"
    show_service_info "Lidarr" "$LIDARR_CONFIG" "http://localhost:8686" "$lidarr_key"
    show_service_info "Prowlarr" "$PROWLARR_CONFIG" "http://localhost:9696" "$prowlarr_key"
    show_service_info "Bazarr" "$BAZARR_CONFIG" "http://localhost:6767" "$bazarr_key"
    show_service_info "Jellyfin" "$JELLYFIN_CONFIG" "http://localhost:8096" "$jellyfin_key"
    
    # Check databases
    echo -e "\n${BLUE}=== DATABASE INFORMATION ===${NC}"
    check_service_database "Sonarr" "$SONARR_CONFIG/sonarr.db"
    check_service_database "Radarr" "$RADARR_CONFIG/radarr.db"
    check_service_database "Prowlarr" "$PROWLARR_CONFIG/prowlarr.db"
    
    # Create API keys file
    create_api_keys_file "$sonarr_key" "$radarr_key" "$lidarr_key" "$prowlarr_key" "$bazarr_key" "$jellyfin_key"
    
    # Display summary
    echo -e "\n${BLUE}=== SUMMARY ===${NC}"
    local configured=0
    local total=6
    
    [ -n "$sonarr_key" ] && ((configured++))
    [ -n "$radarr_key" ] && ((configured++))
    [ -n "$lidarr_key" ] && ((configured++))
    [ -n "$prowlarr_key" ] && ((configured++))
    [ -n "$bazarr_key" ] && ((configured++))
    [ -n "$jellyfin_key" ] && ((configured++))
    
    echo -e "Services with API keys: ${GREEN}$configured${NC}/$total"
    echo -e "Configuration file: ${GREEN}./api-keys.env${NC}"
    
    if [ $configured -eq $total ]; then
        echo -e "\n${GREEN}🎉 All API keys extracted successfully!${NC}"
        echo -e "${GREEN}You can now run: ./configure-service-integrations.sh${NC}"
    else
        echo -e "\n${YELLOW}⚠️  Some API keys are missing. Please:${NC}"
        echo -e "1. Ensure all services are running"
        echo -e "2. Access the web interfaces to generate API keys"
        echo -e "3. Re-run this script to extract them"
    fi
    
    # Provide quick access links
    echo -e "\n${BLUE}=== QUICK ACCESS LINKS ===${NC}"
    echo -e "Sonarr:     http://localhost:8989 (Settings → General → API Key)"
    echo -e "Radarr:     http://localhost:7878 (Settings → General → API Key)"
    echo -e "Prowlarr:   http://localhost:9696 (Settings → General → API Key)"
    echo -e "Jellyfin:   http://localhost:8096 (Dashboard → API Keys)"
    echo -e "qBittorrent: http://localhost:8080 (admin/adminadmin)"
    echo -e "Jellyseerr: http://localhost:5055"
}

# Run main function
main "$@"