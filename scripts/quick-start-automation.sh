#!/bin/bash

# Ultimate Media Server 2025 - Quick Start Automation
# One-click setup for complete *arr automation

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Ultimate Media Server 2025 - Quick Start    ${NC}"
    echo -e "${BLUE}  Complete Automation Setup                   ${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check disk space (minimum 50GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 52428800 ]; then  # 50GB in KB
        print_warning "Less than 50GB free space available. Media automation may need more space."
    fi
    
    print_success "Prerequisites check passed"
}

# Create directory structure
create_directories() {
    print_step "Creating directory structure..."
    
    # Main directories
    mkdir -p media-data/{movies,tv,music,books,audiobooks,podcasts,comics}
    mkdir -p media-data/downloads/{complete,incomplete,torrents,usenet}
    mkdir -p media-data/downloads/torrents/{movies,tv,music,books}
    mkdir -p media-data/downloads/usenet/{movies,tv,music,books}
    
    # Config directories
    mkdir -p config/{prowlarr,sonarr,radarr,lidarr,readarr,bazarr,overseerr}
    mkdir -p config/{qbittorrent,sabnzbd,jellyfin,tautulli,homepage,portainer}
    mkdir -p config/{grafana,prometheus,traefik}
    
    # Logs and backups
    mkdir -p logs
    mkdir -p backups
    
    # Set permissions
    chmod -R 755 media-data/ config/ logs/ backups/ 2>/dev/null || true
    
    print_success "Directory structure created"
}

# Generate environment file
generate_env_file() {
    print_step "Generating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Ultimate Media Server 2025 - Environment Configuration
# Auto-generated on $(date)

# =========================
# General Settings
# =========================
TZ=America/New_York
PUID=1000
PGID=1000

# =========================
# Paths
# =========================
MEDIA_PATH=./media-data
DOWNLOADS_PATH=./media-data/downloads
USENET_PATH=./media-data/usenet

# =========================
# VPN Configuration (Configure for secure downloads)
# =========================
VPN_PROVIDER=mullvad
VPN_PRIVATE_KEY=your_wireguard_private_key_here
VPN_ADDRESSES=10.x.x.x/32

# =========================
# Cloudflare (Optional - for SSL certificates)
# =========================
CLOUDFLARE_EMAIL=your_email@example.com
CLOUDFLARE_API_KEY=your_cloudflare_api_key_here

# =========================
# Service Passwords
# =========================
GRAFANA_USER=admin
GRAFANA_PASSWORD=secure_password_here

# =========================
# Domain (Optional)
# =========================
DOMAIN=localhost
EOF
        print_success "Environment file created (.env)"
        print_warning "Please edit .env file with your actual values before starting services"
    else
        print_info "Environment file already exists"
    fi
}

# Generate docker-compose override for customization
generate_override() {
    print_step "Creating Docker Compose override file..."
    
    if [ ! -f "docker-compose.override.yml" ]; then
        cat > docker-compose.override.yml << 'EOF'
# Docker Compose Override - Customize your deployment
# This file allows you to modify the base configuration

version: "3.9"

services:
  # Example: Add additional environment variables
  # sonarr:
  #   environment:
  #     - SONARR_BRANCH=develop
  
  # Example: Mount additional volumes
  # jellyfin:
  #   volumes:
  #     - /path/to/your/movies:/movies:ro
  
  # Example: Change port mappings
  # overseerr:
  #   ports:
  #     - "5056:5055"
EOF
        print_success "Override file created (docker-compose.override.yml)"
    else
        print_info "Override file already exists"
    fi
}

# Start services
start_services() {
    print_step "Starting automation services..."
    
    # Use the automation-focused compose file
    if [ -f "docker-compose-automation.yml" ]; then
        docker-compose -f docker-compose-automation.yml up -d
    else
        docker-compose up -d
    fi
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    services=(
        "prowlarr:9696"
        "sonarr:8989"
        "radarr:7878"
        "lidarr:8686"
        "readarr:8787"
        "bazarr:6767"
        "overseerr:5055"
        "jellyfin:8096"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        echo -n "Waiting for $name..."
        
        max_attempts=30
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if curl -s -f "http://localhost:$port/ping" >/dev/null 2>&1 || \
               curl -s -f "http://localhost:$port" >/dev/null 2>&1; then
                echo -e " ${GREEN}Ready!${NC}"
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                echo -e " ${YELLOW}Timeout (may still be starting)${NC}"
                break
            fi
            
            echo -n "."
            sleep 5
            ((attempt++))
        done
    done
    
    print_success "Service startup complete"
}

# Configure automation
configure_automation() {
    print_step "Running automation configuration..."
    
    if [ -f "scripts/configure-automation-apis.sh" ]; then
        bash scripts/configure-automation-apis.sh
    else
        print_warning "Automation configuration script not found"
        print_info "You'll need to configure API connections manually"
    fi
}

# Display access information
show_access_info() {
    print_step "Generating access information..."
    
    cat > automation-access-guide.txt << EOF
# Ultimate Media Server 2025 - Access Guide
# Generated on $(date)

## 🎯 Core Automation Services

### Prowlarr (Indexer Manager)
- URL: http://localhost:9696
- Purpose: Manages all indexers/trackers for other services
- First Steps: Add your favorite indexers (1337x, RARBG, etc.)

### Overseerr (Request Manager)  
- URL: http://localhost:5055
- Purpose: User-friendly interface for requesting movies/TV shows
- First Steps: Connect to Sonarr/Radarr, set permissions

## 📺 Media Automation

### Sonarr (TV Shows)
- URL: http://localhost:8989
- Purpose: Automatically downloads TV episodes
- Setup: Add series, configure quality profiles

### Radarr (Movies)
- URL: http://localhost:7878  
- Purpose: Automatically downloads movies
- Setup: Add movies, configure quality profiles

### Lidarr (Music)
- URL: http://localhost:8686
- Purpose: Automatically downloads music
- Setup: Add artists, configure quality profiles

### Readarr (Books)
- URL: http://localhost:8787
- Purpose: Automatically downloads ebooks/audiobooks  
- Setup: Add authors/books, configure quality profiles

### Bazarr (Subtitles)
- URL: http://localhost:6767
- Purpose: Automatically downloads subtitles
- Setup: Configure subtitle providers and languages

## 📥 Download Clients

### qBittorrent
- URL: http://localhost:8080
- Purpose: Torrent downloads (via VPN)
- Default Login: admin/adminadmin

### SABnzbd  
- URL: http://localhost:8081
- Purpose: Usenet downloads
- Setup: Configure your usenet provider

## 🎬 Media Servers

### Jellyfin
- URL: http://localhost:8096
- Purpose: Stream your media
- Setup: Add media libraries, create user accounts

## 📊 Monitoring & Management

### Tautulli
- URL: http://localhost:8181
- Purpose: Monitor Jellyfin usage

### Grafana
- URL: http://localhost:3000
- Login: admin/admin (change on first login)

### Homepage
- URL: http://localhost:3001
- Purpose: Dashboard for all services

### Portainer
- URL: http://localhost:9000
- Purpose: Docker container management

## 🚀 Quick Setup Workflow

1. **Configure Prowlarr**:
   - Add indexers (trackers/usenet)
   - Add applications (Sonarr, Radarr, etc.)

2. **Configure Download Clients**:
   - qBittorrent: Set up categories
   - SABnzbd: Add usenet provider

3. **Configure *arr Services**:
   - Add download clients
   - Set quality profiles  
   - Add root folders

4. **Configure Overseerr**:
   - Connect to Sonarr/Radarr
   - Set user permissions

5. **Test Automation**:
   - Request content via Overseerr
   - Check download progress
   - Verify media appears in Jellyfin

## 🔧 Troubleshooting

### Services not talking to each other:
- Check API keys in Prowlarr applications
- Verify container names in connections

### Downloads not starting:
- Check VPN connection for torrents
- Verify indexer access in Prowlarr

### Files not moving to media folders:
- Check path mappings in *arr services
- Verify permissions on media directories

## 📁 Directory Structure

\`\`\`
media-data/
├── downloads/
│   ├── torrents/    # Completed torrent downloads
│   ├── usenet/      # Completed usenet downloads
│   ├── complete/    # Shared completed folder
│   └── incomplete/  # Active downloads
├── movies/          # Organized movies
├── tv/              # Organized TV shows  
├── music/           # Organized music
├── books/           # Organized books
└── audiobooks/      # Organized audiobooks
\`\`\`

## 🔒 Security Notes

- Change default passwords immediately
- Configure VPN for torrent downloads
- Use strong API keys
- Consider using reverse proxy for external access

## 🎉 Automation Flow

1. User requests content in Overseerr
2. Overseerr sends request to appropriate *arr service
3. *arr service searches for content via Prowlarr
4. Download sent to qBittorrent/SABnzbd
5. Completed download imported to media library
6. Bazarr downloads subtitles
7. Content available in Jellyfin
8. Tautulli tracks viewing statistics

Enjoy your fully automated media server! 🍿
EOF

    print_success "Access guide created: automation-access-guide.txt"
    
    echo
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}         🎉 SETUP COMPLETE! 🎉                ${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo
    echo -e "${PURPLE}Quick Access URLs:${NC}"
    echo -e "  ${CYAN}Overseerr (Request Media):${NC}   http://localhost:5055"
    echo -e "  ${CYAN}Jellyfin (Watch Media):${NC}      http://localhost:8096"
    echo -e "  ${CYAN}Homepage (Dashboard):${NC}        http://localhost:3001"
    echo
    echo -e "${PURPLE}Management URLs:${NC}"
    echo -e "  ${CYAN}Prowlarr (Indexers):${NC}         http://localhost:9696"
    echo -e "  ${CYAN}Sonarr (TV):${NC}                 http://localhost:8989"
    echo -e "  ${CYAN}Radarr (Movies):${NC}             http://localhost:7878"
    echo -e "  ${CYAN}Lidarr (Music):${NC}              http://localhost:8686"
    echo -e "  ${CYAN}Readarr (Books):${NC}             http://localhost:8787"
    echo -e "  ${CYAN}Bazarr (Subtitles):${NC}          http://localhost:6767"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "  1. Configure indexers in Prowlarr"
    echo -e "  2. Set up quality profiles in *arr services"
    echo -e "  3. Configure Overseerr with your *arr services"
    echo -e "  4. Request your first movie/show!"
    echo
    echo -e "See ${CYAN}automation-access-guide.txt${NC} for detailed setup instructions"
    echo
}

# Main function
main() {
    print_header
    
    check_prerequisites
    create_directories
    generate_env_file
    generate_override
    
    echo
    print_step "Starting Ultimate Media Server 2025 automation stack..."
    echo
    
    start_services
    wait_for_services
    
    echo
    print_step "Configuring automation..."
    configure_automation
    
    show_access_info
}

# Handle script arguments
case "${1:-start}" in
    "start"|"")
        main
        ;;
    "stop")
        print_step "Stopping all services..."
        docker-compose down
        print_success "All services stopped"
        ;;
    "restart")
        print_step "Restarting all services..."
        docker-compose down
        docker-compose up -d
        print_success "All services restarted"
        ;;
    "status")
        print_step "Checking service status..."
        docker-compose ps
        ;;
    "logs")
        service="${2:-}"
        if [ -n "$service" ]; then
            docker-compose logs -f "$service"
        else
            docker-compose logs -f
        fi
        ;;
    "update")
        print_step "Updating all services..."
        docker-compose pull
        docker-compose up -d
        print_success "All services updated"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service]|update}"
        echo
        echo "Commands:"
        echo "  start    - Start all automation services (default)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status"
        echo "  logs     - Show logs (optionally for specific service)"
        echo "  update   - Update and restart all services"
        exit 1
        ;;
esac