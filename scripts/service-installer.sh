#!/bin/bash

# Service Installer Script - Dynamic Docker Service Installation
# Ultimate Media Server 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"
CONFIG_DIR="$PROJECT_ROOT/config"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"
ORCHESTRATOR_API="http://localhost:3000/api/v1"

# Service definitions
declare -A SERVICES=(
    ["jellyfin"]="media-server"
    ["plex"]="media-server"
    ["emby"]="media-server"
    ["sonarr"]="arr-suite"
    ["radarr"]="arr-suite"
    ["lidarr"]="arr-suite"
    ["readarr"]="arr-suite"
    ["prowlarr"]="indexer"
    ["bazarr"]="subtitles"
    ["qbittorrent"]="download"
    ["transmission"]="download"
    ["deluge"]="download"
    ["sabnzbd"]="usenet"
    ["nzbget"]="usenet"
    ["overseerr"]="request"
    ["ombi"]="request"
    ["tautulli"]="analytics"
    ["organizr"]="dashboard"
    ["homepage"]="dashboard"
    ["heimdall"]="dashboard"
    ["portainer"]="management"
    ["prometheus"]="monitoring"
    ["grafana"]="monitoring"
    ["traefik"]="proxy"
    ["nginx-proxy-manager"]="proxy"
    ["watchtower"]="automation"
    ["flaresolverr"]="utility"
    ["vpn"]="network"
)

# Function to display header
show_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         Ultimate Media Server 2025                 ║${NC}"
    echo -e "${CYAN}║           Service Installer                        ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
    echo
}

# Function to check prerequisites
check_prerequisites() {
    local missing=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing+=("docker-compose")
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}Missing prerequisites: ${missing[*]}${NC}"
        echo -e "${YELLOW}Please install the missing tools and try again.${NC}"
        exit 1
    fi
}

# Function to check if orchestrator is running
check_orchestrator() {
    if ! curl -s -f "$ORCHESTRATOR_API/health" > /dev/null 2>&1; then
        echo -e "${YELLOW}Service Orchestrator is not running.${NC}"
        echo -e "${BLUE}Starting Orchestrator...${NC}"
        
        cd "$PROJECT_ROOT/api/service-orchestrator"
        docker-compose up -d
        
        # Wait for orchestrator to be ready
        local count=0
        while ! curl -s -f "$ORCHESTRATOR_API/health" > /dev/null 2>&1; do
            if [ $count -ge 30 ]; then
                echo -e "${RED}Failed to start Orchestrator${NC}"
                exit 1
            fi
            echo -n "."
            sleep 2
            ((count++))
        done
        echo
        echo -e "${GREEN}Orchestrator started successfully${NC}"
    fi
}

# Function to list available services
list_available_services() {
    echo -e "${BLUE}Available Services:${NC}"
    echo
    
    # Group services by category
    declare -A categories
    for service in "${!SERVICES[@]}"; do
        category="${SERVICES[$service]}"
        if [ -z "${categories[$category]}" ]; then
            categories[$category]="$service"
        else
            categories[$category]="${categories[$category]} $service"
        fi
    done
    
    # Display by category
    for category in "${!categories[@]}"; do
        echo -e "${CYAN}[$category]${NC}"
        for service in ${categories[$category]}; do
            # Check if installed
            if service_installed "$service"; then
                echo -e "  ${GREEN}✓ $service (installed)${NC}"
            else
                echo -e "  ${YELLOW}○ $service${NC}"
            fi
        done
        echo
    done
}

# Function to check if service is installed
service_installed() {
    local service=$1
    
    # Check via API
    if curl -s -f "$ORCHESTRATOR_API/services/$service" > /dev/null 2>&1; then
        return 0
    fi
    
    # Fallback to docker check
    if docker ps -a --format '{{.Names}}' | grep -q "^${service}$"; then
        return 0
    fi
    
    return 1
}

# Function to install service
install_service() {
    local service=$1
    
    echo -e "${BLUE}Installing $service...${NC}"
    
    # Get service manifest
    local manifest_file="$SERVICES_DIR/$service/manifest.yml"
    if [ ! -f "$manifest_file" ]; then
        echo -e "${YELLOW}Creating service manifest...${NC}"
        create_service_manifest "$service"
    fi
    
    # Install via API
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d @"$manifest_file" \
        "$ORCHESTRATOR_API/services")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $service installed successfully${NC}"
        
        # Show post-install information
        show_post_install_info "$service"
    else
        echo -e "${RED}✗ Failed to install $service${NC}"
        echo "$response" | jq -r '.error' 2>/dev/null || echo "$response"
        return 1
    fi
}

# Function to create service manifest
create_service_manifest() {
    local service=$1
    local category="${SERVICES[$service]}"
    local manifest_dir="$SERVICES_DIR/$service"
    
    mkdir -p "$manifest_dir"
    
    cat > "$manifest_dir/manifest.yml" << EOF
name: $service
category: $category
version: latest
description: $service service for Ultimate Media Server
image: $(get_service_image "$service")
ports: $(get_service_ports "$service")
volumes: $(get_service_volumes "$service")
environment: $(get_service_environment "$service")
networks:
  - media_network
restart: unless-stopped
dependencies: $(get_service_dependencies "$service")
healthcheck:
  test: $(get_service_healthcheck "$service")
  interval: 30s
  timeout: 10s
  retries: 3
integrations: $(get_service_integrations "$service")
EOF
}

# Function to get service image
get_service_image() {
    local service=$1
    
    case $service in
        jellyfin) echo "jellyfin/jellyfin:latest" ;;
        plex) echo "plexinc/pms-docker:latest" ;;
        emby) echo "emby/embyserver:latest" ;;
        sonarr|radarr|lidarr|readarr|prowlarr|bazarr) 
            echo "lscr.io/linuxserver/${service}:latest" ;;
        qbittorrent) echo "lscr.io/linuxserver/qbittorrent:latest" ;;
        transmission) echo "lscr.io/linuxserver/transmission:latest" ;;
        deluge) echo "lscr.io/linuxserver/deluge:latest" ;;
        sabnzbd) echo "lscr.io/linuxserver/sabnzbd:latest" ;;
        nzbget) echo "lscr.io/linuxserver/nzbget:latest" ;;
        overseerr) echo "lscr.io/linuxserver/overseerr:latest" ;;
        ombi) echo "lscr.io/linuxserver/ombi:latest" ;;
        tautulli) echo "lscr.io/linuxserver/tautulli:latest" ;;
        organizr) echo "organizr/organizr:latest" ;;
        homepage) echo "ghcr.io/gethomepage/homepage:latest" ;;
        heimdall) echo "lscr.io/linuxserver/heimdall:latest" ;;
        portainer) echo "portainer/portainer-ce:latest" ;;
        prometheus) echo "prom/prometheus:latest" ;;
        grafana) echo "grafana/grafana:latest" ;;
        traefik) echo "traefik:v3.0" ;;
        nginx-proxy-manager) echo "jc21/nginx-proxy-manager:latest" ;;
        watchtower) echo "containrrr/watchtower:latest" ;;
        flaresolverr) echo "ghcr.io/flaresolverr/flaresolverr:latest" ;;
        vpn) echo "qmcgaw/gluetun:latest" ;;
        *) echo "unknown:latest" ;;
    esac
}

# Function to get service ports
get_service_ports() {
    local service=$1
    
    case $service in
        jellyfin) echo '["8096:8096", "8920:8920", "7359:7359/udp", "1900:1900/udp"]' ;;
        plex) echo '["32400:32400"]' ;;
        emby) echo '["8096:8096"]' ;;
        sonarr) echo '["8989:8989"]' ;;
        radarr) echo '["7878:7878"]' ;;
        lidarr) echo '["8686:8686"]' ;;
        readarr) echo '["8787:8787"]' ;;
        prowlarr) echo '["9696:9696"]' ;;
        bazarr) echo '["6767:6767"]' ;;
        qbittorrent) echo '["8080:8080"]' ;;
        transmission) echo '["9091:9091"]' ;;
        deluge) echo '["8112:8112"]' ;;
        sabnzbd) echo '["8081:8080"]' ;;
        nzbget) echo '["6789:6789"]' ;;
        overseerr) echo '["5055:5055"]' ;;
        ombi) echo '["3579:3579"]' ;;
        tautulli) echo '["8181:8181"]' ;;
        organizr) echo '["9983:80"]' ;;
        homepage) echo '["3001:3000"]' ;;
        heimdall) echo '["8082:80"]' ;;
        portainer) echo '["9000:9000", "9443:9443"]' ;;
        prometheus) echo '["9090:9090"]' ;;
        grafana) echo '["3000:3000"]' ;;
        traefik) echo '["80:80", "443:443", "8083:8080"]' ;;
        nginx-proxy-manager) echo '["81:81", "80:80", "443:443"]' ;;
        *) echo '[]' ;;
    esac
}

# Function to get service volumes
get_service_volumes() {
    local service=$1
    
    case $service in
        jellyfin|plex|emby)
            echo '[
                {"source": "./config/'$service'", "target": "/config"},
                {"source": "${MEDIA_PATH:-./media-data}", "target": "/media", "mode": "ro"}
            ]' ;;
        sonarr|radarr|lidarr|readarr)
            echo '[
                {"source": "./config/'$service'", "target": "/config"},
                {"source": "${MEDIA_PATH:-./media-data}", "target": "/media"},
                {"source": "${DOWNLOADS_PATH:-./media-data/downloads}", "target": "/downloads"}
            ]' ;;
        prowlarr|bazarr|overseerr|ombi|tautulli)
            echo '[{"source": "./config/'$service'", "target": "/config"}]' ;;
        qbittorrent|transmission|deluge|sabnzbd|nzbget)
            echo '[
                {"source": "./config/'$service'", "target": "/config"},
                {"source": "${DOWNLOADS_PATH:-./media-data/downloads}", "target": "/downloads"}
            ]' ;;
        portainer)
            echo '[
                {"source": "/var/run/docker.sock", "target": "/var/run/docker.sock", "mode": "ro"},
                {"source": "./config/portainer", "target": "/data"}
            ]' ;;
        prometheus)
            echo '[
                {"source": "./config/prometheus", "target": "/etc/prometheus"},
                {"source": "prometheus_data", "target": "/prometheus"}
            ]' ;;
        grafana)
            echo '[
                {"source": "grafana_data", "target": "/var/lib/grafana"},
                {"source": "./config/grafana", "target": "/etc/grafana/provisioning"}
            ]' ;;
        traefik)
            echo '[
                {"source": "/var/run/docker.sock", "target": "/var/run/docker.sock", "mode": "ro"},
                {"source": "./config/traefik", "target": "/letsencrypt"}
            ]' ;;
        *) echo '[]' ;;
    esac
}

# Function to get service environment
get_service_environment() {
    echo '{
        "PUID": "1000",
        "PGID": "1000",
        "TZ": "${TZ:-America/New_York}"
    }'
}

# Function to get service dependencies
get_service_dependencies() {
    local service=$1
    
    case $service in
        sonarr|radarr|lidarr|readarr) echo '["prowlarr"]' ;;
        bazarr) echo '["sonarr", "radarr"]' ;;
        qbittorrent|transmission|deluge) echo '["vpn"]' ;;
        tautulli) echo '["plex"]' ;;
        overseerr|ombi) echo '["sonarr", "radarr"]' ;;
        *) echo '[]' ;;
    esac
}

# Function to get service healthcheck
get_service_healthcheck() {
    local service=$1
    
    case $service in
        jellyfin) echo '["CMD", "curl", "-f", "http://localhost:8096/health"]' ;;
        plex) echo '["CMD", "curl", "-f", "http://localhost:32400/identity"]' ;;
        sonarr) echo '["CMD", "curl", "-f", "http://localhost:8989/ping"]' ;;
        radarr) echo '["CMD", "curl", "-f", "http://localhost:7878/ping"]' ;;
        prowlarr) echo '["CMD", "curl", "-f", "http://localhost:9696/ping"]' ;;
        overseerr) echo '["CMD", "curl", "-f", "http://localhost:5055/api/v1/status"]' ;;
        portainer) echo '["CMD", "curl", "-f", "http://localhost:9000"]' ;;
        grafana) echo '["CMD", "curl", "-f", "http://localhost:3000/api/health"]' ;;
        *) echo '["CMD", "echo", "OK"]' ;;
    esac
}

# Function to get service integrations
get_service_integrations() {
    local service=$1
    
    case $service in
        sonarr|radarr|lidarr|readarr)
            echo '[
                {"type": "api", "target": "prowlarr", "bidirectional": true},
                {"type": "api", "target": "qbittorrent", "optional": true},
                {"type": "api", "target": "sabnzbd", "optional": true}
            ]' ;;
        overseerr|ombi)
            echo '[
                {"type": "api", "target": "sonarr"},
                {"type": "api", "target": "radarr"},
                {"type": "api", "target": "plex", "optional": true},
                {"type": "api", "target": "jellyfin", "optional": true}
            ]' ;;
        tautulli)
            echo '[{"type": "api", "target": "plex"}]' ;;
        *) echo '[]' ;;
    esac
}

# Function to uninstall service
uninstall_service() {
    local service=$1
    
    echo -e "${YELLOW}Uninstalling $service...${NC}"
    
    # Confirm uninstall
    read -p "Are you sure you want to uninstall $service? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Uninstall cancelled."
        return
    fi
    
    # Uninstall via API
    local response=$(curl -s -X DELETE "$ORCHESTRATOR_API/services/$service")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $service uninstalled successfully${NC}"
        
        # Ask about data removal
        read -p "Remove configuration data for $service? (y/N): " remove_data
        if [[ "$remove_data" =~ ^[Yy]$ ]]; then
            rm -rf "$CONFIG_DIR/$service"
            echo -e "${GREEN}Configuration data removed${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to uninstall $service${NC}"
        echo "$response" | jq -r '.error' 2>/dev/null || echo "$response"
        return 1
    fi
}

# Function to show post-install information
show_post_install_info() {
    local service=$1
    
    echo
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}$service Installation Complete!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    
    # Get service info
    local info=$(curl -s "$ORCHESTRATOR_API/services/$service")
    local port=$(echo "$info" | jq -r '.config.ports[0]' | cut -d: -f1)
    
    echo -e "${BLUE}Access URL:${NC} http://localhost:$port"
    
    # Service-specific instructions
    case $service in
        jellyfin)
            echo -e "${YELLOW}First-time setup:${NC}"
            echo "1. Open http://localhost:8096 in your browser"
            echo "2. Follow the setup wizard"
            echo "3. Add your media libraries from /media"
            ;;
        sonarr|radarr)
            echo -e "${YELLOW}Configuration needed:${NC}"
            echo "1. Add indexers from Prowlarr"
            echo "2. Configure download clients"
            echo "3. Set up root folders for media"
            ;;
        prowlarr)
            echo -e "${YELLOW}Setup indexers:${NC}"
            echo "1. Add your preferred indexers"
            echo "2. Configure API keys"
            echo "3. Test connections to *arr apps"
            ;;
        overseerr)
            echo -e "${YELLOW}Initial setup:${NC}"
            echo "1. Sign in with Plex account"
            echo "2. Configure Sonarr/Radarr connections"
            echo "3. Set up user permissions"
            ;;
    esac
    
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo
}

# Function to install service group
install_service_group() {
    local group=$1
    
    case $group in
        "essential")
            local services=("jellyfin" "sonarr" "radarr" "prowlarr" "qbittorrent" "overseerr")
            ;;
        "complete")
            local services=("jellyfin" "sonarr" "radarr" "lidarr" "prowlarr" "bazarr" 
                          "qbittorrent" "sabnzbd" "overseerr" "tautulli" "homepage" 
                          "portainer" "prometheus" "grafana" "traefik")
            ;;
        "monitoring")
            local services=("prometheus" "grafana" "tautulli" "portainer")
            ;;
        "download")
            local services=("qbittorrent" "sabnzbd" "vpn")
            ;;
        *)
            echo -e "${RED}Unknown service group: $group${NC}"
            return 1
            ;;
    esac
    
    echo -e "${BLUE}Installing $group service group...${NC}"
    echo -e "${YELLOW}Services to install: ${services[*]}${NC}"
    echo
    
    for service in "${services[@]}"; do
        if ! service_installed "$service"; then
            install_service "$service"
            sleep 2
        else
            echo -e "${GREEN}✓ $service already installed${NC}"
        fi
    done
    
    echo
    echo -e "${GREEN}Service group '$group' installation complete!${NC}"
}

# Main menu
main_menu() {
    while true; do
        show_header
        
        echo -e "${BLUE}Main Menu:${NC}"
        echo "1. List available services"
        echo "2. Install individual service"
        echo "3. Install service group"
        echo "4. Uninstall service"
        echo "5. Check service status"
        echo "6. Update all services"
        echo "7. Backup configuration"
        echo "8. Restore configuration"
        echo "0. Exit"
        echo
        
        read -p "Select option: " choice
        
        case $choice in
            1)
                show_header
                list_available_services
                read -p "Press Enter to continue..."
                ;;
            2)
                show_header
                list_available_services
                read -p "Enter service name to install: " service
                if [ -n "${SERVICES[$service]}" ]; then
                    install_service "$service"
                else
                    echo -e "${RED}Invalid service name${NC}"
                fi
                read -p "Press Enter to continue..."
                ;;
            3)
                show_header
                echo -e "${BLUE}Available service groups:${NC}"
                echo "1. Essential (Media server + basic arr suite)"
                echo "2. Complete (All recommended services)"
                echo "3. Monitoring (Prometheus, Grafana, etc.)"
                echo "4. Download (Download clients + VPN)"
                echo
                read -p "Select group: " group_choice
                case $group_choice in
                    1) install_service_group "essential" ;;
                    2) install_service_group "complete" ;;
                    3) install_service_group "monitoring" ;;
                    4) install_service_group "download" ;;
                    *) echo -e "${RED}Invalid selection${NC}" ;;
                esac
                read -p "Press Enter to continue..."
                ;;
            4)
                show_header
                list_available_services
                read -p "Enter service name to uninstall: " service
                if [ -n "${SERVICES[$service]}" ]; then
                    uninstall_service "$service"
                else
                    echo -e "${RED}Invalid service name${NC}"
                fi
                read -p "Press Enter to continue..."
                ;;
            5)
                "$SCRIPT_DIR/service-status.sh"
                ;;
            6)
                "$SCRIPT_DIR/update-services.sh"
                ;;
            7)
                "$SCRIPT_DIR/backup.sh"
                ;;
            8)
                "$SCRIPT_DIR/restore.sh"
                ;;
            0)
                echo -e "${GREEN}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                sleep 2
                ;;
        esac
    done
}

# Script execution
check_prerequisites
check_orchestrator
main_menu