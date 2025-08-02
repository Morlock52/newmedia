#!/bin/bash

# Media Server - Initial Setup Script
# macOS compatible, modular deployment system
# Version: 1.0.0

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
readonly CONFIG_DIR="${PROJECT_ROOT}/config"
readonly DATA_DIR="${PROJECT_ROOT}/data"
readonly LOGS_DIR="${PROJECT_ROOT}/logs"
readonly BACKUP_DIR="${PROJECT_ROOT}/backups"

# Default configuration
readonly DEFAULT_TZ="America/New_York"
readonly DEFAULT_PUID="1000"
readonly DEFAULT_PGID="1000"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}==== $1 ====${NC}"
}

# Show banner
show_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    __  ___         ___         _____                           
   /  |/  /__  ____/ (_)___ _  / ___/___  ______   _____  _____
  / /|_/ / _ \/ __  / / __ `/ \__ \/ _ \/ ___/ | / / _ \/ ___/
 / /  / /  __/ /_/ / / /_/ / ___/ /  __/ /   | |/ /  __/ /    
/_/  /_/\___/\__,_/_/\__,_/ /____/\___/_/    |___/\___/_/     
                                                               
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Media Server Setup - macOS Optimized${NC}"
    echo -e "${CYAN}Version: 1.0.0${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    local missing_deps=()
    
    # Check for required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        else
            log_info "✓ $cmd found"
        fi
    done
    
    # Check Docker daemon
    if command -v docker &> /dev/null; then
        if ! docker info &> /dev/null; then
            log_error "Docker daemon is not running"
            log_warn "Please start Docker Desktop"
            return 1
        else
            log_info "✓ Docker daemon is running"
        fi
    fi
    
    # Report missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_warn "Please install missing dependencies and try again"
        return 1
    fi
    
    return 0
}

# Create directory structure
create_directories() {
    log_header "Creating Directory Structure"
    
    local dirs=(
        "$CONFIG_DIR"
        "$DATA_DIR/media/movies"
        "$DATA_DIR/media/tv"
        "$DATA_DIR/media/music"
        "$DATA_DIR/media/books"
        "$DATA_DIR/media/audiobooks"
        "$DATA_DIR/media/podcasts"
        "$DATA_DIR/downloads/complete"
        "$DATA_DIR/downloads/incomplete"
        "$DATA_DIR/torrents"
        "$DATA_DIR/usenet"
        "$LOGS_DIR"
        "$BACKUP_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created: $dir"
        else
            log_info "Exists: $dir"
        fi
    done
    
    # Create service-specific config directories
    local services=(
        "jellyfin" "radarr" "sonarr" "prowlarr" "qbittorrent"
        "bazarr" "overseerr" "tautulli" "homepage" "portainer"
    )
    
    for service in "${services[@]}"; do
        mkdir -p "$CONFIG_DIR/$service"
    done
}

# Set permissions (macOS compatible)
set_permissions() {
    log_header "Setting Permissions"
    
    # Get current user and group
    local current_user=$(id -u)
    local current_group=$(id -g)
    
    # Set ownership for data and config directories
    log_info "Setting ownership to $current_user:$current_group"
    
    # Use find to avoid issues with large directory trees
    find "$DATA_DIR" -type d -exec chmod 755 {} \; 2>/dev/null || true
    find "$CONFIG_DIR" -type d -exec chmod 755 {} \; 2>/dev/null || true
    
    # Set write permissions for service directories
    chmod -R 755 "$DATA_DIR" 2>/dev/null || true
    chmod -R 755 "$CONFIG_DIR" 2>/dev/null || true
    
    log_info "Permissions set successfully"
}

# Generate environment file
generate_env_file() {
    log_header "Generating Environment Configuration"
    
    local env_file="${PROJECT_ROOT}/.env"
    
    if [ -f "$env_file" ]; then
        log_warn "Environment file already exists"
        read -p "Overwrite existing .env file? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing environment file"
            return
        fi
        # Backup existing file
        cp "$env_file" "${env_file}.backup.$(date +%Y%m%d_%H%M%S)"
        log_info "Backed up existing .env file"
    fi
    
    cat > "$env_file" << EOF
# Media Server Environment Configuration
# Generated on $(date)

# System Configuration
TZ=${DEFAULT_TZ}
PUID=${DEFAULT_PUID}
PGID=${DEFAULT_PGID}

# Directory Paths
CONFIG_PATH=${CONFIG_DIR}
DATA_PATH=${DATA_DIR}
MEDIA_PATH=${DATA_DIR}/media
DOWNLOADS_PATH=${DATA_DIR}/downloads

# Service Ports
JELLYFIN_PORT=8096
RADARR_PORT=7878
SONARR_PORT=8989
PROWLARR_PORT=9696
QBITTORRENT_PORT=8080
BAZARR_PORT=6767
OVERSEERR_PORT=5055
TAUTULLI_PORT=8181
HOMEPAGE_PORT=3000
PORTAINER_PORT=9000

# API Keys (generated during first run)
RADARR_API_KEY=
SONARR_API_KEY=
PROWLARR_API_KEY=
JELLYFIN_API_KEY=

# Network Configuration
NETWORK_NAME=media_network
EOF
    
    chmod 600 "$env_file"
    log_info "Environment file created: $env_file"
}

# Create docker-compose override for macOS
create_macos_override() {
    log_header "Creating macOS Docker Compose Override"
    
    local override_file="${PROJECT_ROOT}/docker-compose.override.yml"
    
    cat > "$override_file" << 'EOF'
# macOS-specific overrides
version: '3.8'

services:
  # Disable hardware acceleration for Jellyfin on macOS
  jellyfin:
    environment:
      - JELLYFIN_PublishedServerUrl=http://localhost:8096
    devices: []  # Remove GPU devices on macOS
    
  # Optimize qBittorrent for macOS
  qbittorrent:
    environment:
      - UMASK=002
    volumes:
      # Use delegated for better performance on macOS
      - ./config/qbittorrent:/config:delegated
      - ./data/downloads:/downloads:delegated
EOF
    
    log_info "Created macOS override file: $override_file"
}

# Initialize configuration files
initialize_configs() {
    log_header "Initializing Service Configurations"
    
    # Homepage configuration
    create_homepage_config
    
    # Create basic service configurations
    create_service_configs
}

# Create Homepage configuration
create_homepage_config() {
    local homepage_dir="$CONFIG_DIR/homepage"
    
    # Services configuration
    cat > "$homepage_dir/services.yaml" << 'EOF'
- Media:
    - Jellyfin:
        href: http://localhost:8096
        description: Media streaming server
        icon: jellyfin.png
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: "{{HOMEPAGE_VAR_JELLYFIN_API_KEY}}"
          enableBlocks: true

- Management:
    - Radarr:
        href: http://localhost:7878
        description: Movie management
        icon: radarr.png
        widget:
          type: radarr
          url: http://radarr:7878
          key: "{{HOMEPAGE_VAR_RADARR_API_KEY}}"
    
    - Sonarr:
        href: http://localhost:8989
        description: TV show management
        icon: sonarr.png
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: "{{HOMEPAGE_VAR_SONARR_API_KEY}}"
    
    - Prowlarr:
        href: http://localhost:9696
        description: Indexer management
        icon: prowlarr.png

- Download:
    - qBittorrent:
        href: http://localhost:8080
        description: Torrent client
        icon: qbittorrent.png
        widget:
          type: qbittorrent
          url: http://qbittorrent:8080
          username: admin
          password: adminadmin

- Tools:
    - Overseerr:
        href: http://localhost:5055
        description: Request management
        icon: overseerr.png
    
    - Tautulli:
        href: http://localhost:8181
        description: Media analytics
        icon: tautulli.png
    
    - Portainer:
        href: http://localhost:9000
        description: Container management
        icon: portainer.png
EOF
    
    # Settings configuration
    cat > "$homepage_dir/settings.yaml" << 'EOF'
title: Media Server Dashboard
theme: dark
color: slate
layout:
  Media:
    style: row
    columns: 1
  Management:
    style: row
    columns: 3
  Download:
    style: row
    columns: 1
  Tools:
    style: row
    columns: 3

providers:
  openweathermap: openweathermap
  
hideVersion: true
EOF
    
    # Docker integration
    cat > "$homepage_dir/docker.yaml" << 'EOF'
my-docker:
  host: /var/run/docker.sock
EOF
    
    log_info "Homepage configuration created"
}

# Create basic service configurations
create_service_configs() {
    # qBittorrent configuration template
    cat > "$CONFIG_DIR/qbittorrent/qBittorrent.conf.template" << 'EOF'
[Preferences]
WebUI\Username=admin
WebUI\Password_PBKDF2="@ByteArray(adminadmin)"
WebUI\Port=8080
WebUI\LocalHostAuth=false
Downloads\SavePath=/downloads/complete
Downloads\TempPath=/downloads/incomplete
Downloads\TorrentExportDir=/downloads/torrents
EOF
    
    log_info "Service configuration templates created"
}

# Display next steps
show_next_steps() {
    echo -e "\n${GREEN}✅ Setup completed successfully!${NC}\n"
    
    echo -e "${CYAN}Directory Structure:${NC}"
    echo "  Config: $CONFIG_DIR"
    echo "  Data:   $DATA_DIR"
    echo "  Logs:   $LOGS_DIR"
    echo "  Backup: $BACKUP_DIR"
    
    echo -e "\n${CYAN}Next Steps:${NC}"
    echo "1. Review and edit .env file if needed"
    echo "2. Run deployment script: ./scripts/deploy/deploy.sh"
    echo "3. Access services at http://localhost:3000 (Homepage Dashboard)"
    
    echo -e "\n${CYAN}Useful Commands:${NC}"
    echo "  Start services:  ./scripts/deploy/service-control.sh start"
    echo "  Stop services:   ./scripts/deploy/service-control.sh stop"
    echo "  View logs:       ./scripts/deploy/service-control.sh logs"
    echo "  Health check:    ./scripts/deploy/health-check.sh"
    echo "  Backup:          ./scripts/deploy/backup.sh"
}

# Main setup flow
main() {
    show_banner
    
    # Run setup steps
    check_prerequisites || exit 1
    create_directories
    set_permissions
    generate_env_file
    create_macos_override
    initialize_configs
    
    show_next_steps
}

# Run main function
main "$@"