#!/bin/bash

# ============================================================================
# Media Server Setup Script
# Version: 2025.08
# Description: Complete setup for media server stack with proper permissions
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MEDIA_BASE="${MEDIA_BASE:-/data}"
USER_ID="${PUID:-1000}"
GROUP_ID="${PGID:-1000}"
COMPOSE_FILE="${1:-docker-compose.media-complete.yml}"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}                     MEDIA SERVER SETUP SCRIPT                                ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Function to create directory with proper permissions
create_dir() {
    local dir="$1"
    local desc="$2"
    
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}📁 Creating${NC} $desc: $dir"
        mkdir -p "$dir"
        chown -R "$USER_ID:$GROUP_ID" "$dir"
        echo -e "${GREEN}✅ Created${NC} $dir"
    else
        echo -e "${BLUE}ℹ️  Exists${NC} $dir"
    fi
}

# Function to check if running as root (recommended for initial setup)
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: Not running as root. Some operations may fail.${NC}"
        echo -e "${YELLOW}   Consider running: sudo $0${NC}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to check Docker installation
check_docker() {
    echo -e "\n${BLUE}🐳 Checking Docker installation...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed!${NC}"
        echo -e "${YELLOW}   Please install Docker first: https://docs.docker.com/get-docker/${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed!${NC}"
        echo -e "${YELLOW}   Please install Docker Compose: https://docs.docker.com/compose/install/${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
}

# Function to create .env file
create_env_file() {
    echo -e "\n${BLUE}🔧 Setting up environment variables...${NC}"
    
    if [ ! -f .env ]; then
        echo -e "${YELLOW}📝 Creating .env file...${NC}"
        cat > .env << EOF
# Media Server Environment Configuration
# Generated: $(date)

# User/Group IDs
PUID=$USER_ID
PGID=$GROUP_ID

# Timezone
TZ=America/New_York

# Paths
MEDIA_PATH=$MEDIA_BASE/media
DOWNLOADS_PATH=$MEDIA_BASE/downloads

# VPN Configuration (Update with your credentials)
VPN_PROVIDER=nordvpn
VPN_TYPE=openvpn
VPN_USER=your_vpn_username
VPN_PASSWORD=your_vpn_password
VPN_COUNTRY=Netherlands

# API Keys (Will be generated after first run)
JELLYFIN_API_KEY=
SONARR_API_KEY=
RADARR_API_KEY=
LIDARR_API_KEY=
PROWLARR_API_KEY=
BAZARR_API_KEY=

# URLs
JELLYFIN_URL=http://localhost:8096

# Database Passwords
ARCHON_DB_PASSWORD=archon_secure_pass_$(date +%s | sha256sum | base64 | head -c 32)

# Optional: OpenAI for Archon
OPENAI_API_KEY=
GOOGLE_API_KEY=
EOF
        echo -e "${GREEN}✅ Created .env file${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env to add your VPN credentials and API keys${NC}"
    else
        echo -e "${BLUE}ℹ️  .env file already exists${NC}"
    fi
}

# Function to create directory structure
create_directory_structure() {
    echo -e "\n${BLUE}📂 Creating directory structure...${NC}"
    
    # Media directories
    create_dir "$MEDIA_BASE/media/movies" "Movies directory"
    create_dir "$MEDIA_BASE/media/tv" "TV Shows directory"
    create_dir "$MEDIA_BASE/media/music" "Music directory"
    create_dir "$MEDIA_BASE/media/books" "Books directory"
    create_dir "$MEDIA_BASE/media/audiobooks" "Audiobooks directory"
    
    # Download directories
    create_dir "$MEDIA_BASE/downloads/complete" "Completed downloads"
    create_dir "$MEDIA_BASE/downloads/incomplete" "Incomplete downloads"
    create_dir "$MEDIA_BASE/downloads/watch" "Watch folder"
    create_dir "$MEDIA_BASE/downloads/blackhole" "Blackhole directory"
    
    # Configuration directories
    create_dir "./configs" "Configuration root"
    create_dir "./configs/jellyfin" "Jellyfin config"
    create_dir "./configs/sonarr" "Sonarr config"
    create_dir "./configs/radarr" "Radarr config"
    create_dir "./configs/lidarr" "Lidarr config"
    create_dir "./configs/prowlarr" "Prowlarr config"
    create_dir "./configs/bazarr" "Bazarr config"
    create_dir "./configs/qbittorrent" "qBittorrent config"
    create_dir "./configs/overseerr" "Overseerr config"
    create_dir "./configs/tautulli" "Tautulli config"
    create_dir "./configs/homepage" "Homepage config"
    
    # Backup directory
    create_dir "./backups" "Backup directory"
    
    # Archon directories
    create_dir "./archon-data" "Archon data"
    create_dir "./archon-uploads" "Archon uploads"
    
    echo -e "${GREEN}✅ Directory structure created successfully${NC}"
}

# Function to create Docker networks
create_networks() {
    echo -e "\n${BLUE}🌐 Creating Docker networks...${NC}"
    
    networks=("media-net" "download-net" "archon-net")
    
    for network in "${networks[@]}"; do
        if docker network ls | grep -q "$network"; then
            echo -e "${BLUE}ℹ️  Network $network already exists${NC}"
        else
            echo -e "${YELLOW}🔗 Creating network: $network${NC}"
            docker network create "$network"
            echo -e "${GREEN}✅ Created network: $network${NC}"
        fi
    done
}

# Function to set up Homepage dashboard
setup_homepage() {
    echo -e "\n${BLUE}🎨 Setting up Homepage dashboard...${NC}"
    
    mkdir -p ./configs/homepage
    
    # Create services.yaml
    cat > ./configs/homepage/services.yaml << 'EOF'
---
# Media Services
- Media:
    - Jellyfin:
        href: http://localhost:8096
        icon: jellyfin.png
        description: Media Streaming
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: "{{HOMEPAGE_VAR_JELLYFIN_API_KEY}}"
          enableBlocks: true
          enableNowPlaying: true
    
    - Overseerr:
        href: http://localhost:5055
        icon: overseerr.png
        description: Media Requests

# Automation
- Automation:
    - Sonarr:
        href: http://localhost:8989
        icon: sonarr.png
        description: TV Management
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: "{{HOMEPAGE_VAR_SONARR_API_KEY}}"
    
    - Radarr:
        href: http://localhost:7878
        icon: radarr.png
        description: Movie Management
        widget:
          type: radarr
          url: http://radarr:7878
          key: "{{HOMEPAGE_VAR_RADARR_API_KEY}}"
    
    - Prowlarr:
        href: http://localhost:9696
        icon: prowlarr.png
        description: Indexer Management
        widget:
          type: prowlarr
          url: http://prowlarr:9696
          key: "{{HOMEPAGE_VAR_PROWLARR_API_KEY}}"

# Downloads
- Downloads:
    - qBittorrent:
        href: http://localhost:8080
        icon: qbittorrent.png
        description: Torrent Client
        widget:
          type: qbittorrent
          url: http://localhost:8080
          username: admin
          password: adminadmin

# Monitoring
- Monitoring:
    - Tautulli:
        href: http://localhost:8181
        icon: tautulli.png
        description: Media Analytics
EOF
    
    # Create settings.yaml
    cat > ./configs/homepage/settings.yaml << 'EOF'
---
title: Media Server Dashboard
background: https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85
theme: dark
color: slate
layout:
  Media:
    style: row
    columns: 3
  Automation:
    style: row
    columns: 4
  Downloads:
    style: row
    columns: 2
  Monitoring:
    style: row
    columns: 2
EOF
    
    echo -e "${GREEN}✅ Homepage dashboard configured${NC}"
}

# Function to start services
start_services() {
    echo -e "\n${BLUE}🚀 Starting services...${NC}"
    
    if [ -f "$COMPOSE_FILE" ]; then
        echo -e "${YELLOW}📦 Using compose file: $COMPOSE_FILE${NC}"
        
        # Pull images first
        echo -e "${YELLOW}⬇️  Pulling Docker images...${NC}"
        docker-compose -f "$COMPOSE_FILE" pull
        
        # Start services
        echo -e "${YELLOW}▶️  Starting services...${NC}"
        docker-compose -f "$COMPOSE_FILE" up -d
        
        echo -e "${GREEN}✅ Services started successfully${NC}"
    else
        echo -e "${RED}❌ Compose file not found: $COMPOSE_FILE${NC}"
        exit 1
    fi
}

# Function to display service URLs
display_urls() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}                           SERVICE URLS                                       ${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${GREEN}📺 Media Services:${NC}"
    echo -e "   Jellyfin:        ${BLUE}http://localhost:8096${NC}"
    echo -e "   Overseerr:       ${BLUE}http://localhost:5055${NC}"
    echo
    echo -e "${GREEN}🤖 Automation:${NC}"
    echo -e "   Sonarr:          ${BLUE}http://localhost:8989${NC}"
    echo -e "   Radarr:          ${BLUE}http://localhost:7878${NC}"
    echo -e "   Lidarr:          ${BLUE}http://localhost:8686${NC}"
    echo -e "   Prowlarr:        ${BLUE}http://localhost:9696${NC}"
    echo -e "   Bazarr:          ${BLUE}http://localhost:6767${NC}"
    echo
    echo -e "${GREEN}⬇️  Downloads:${NC}"
    echo -e "   qBittorrent:     ${BLUE}http://localhost:8080${NC}"
    echo
    echo -e "${GREEN}📊 Monitoring:${NC}"
    echo -e "   Tautulli:        ${BLUE}http://localhost:8181${NC}"
    echo -e "   Homepage:        ${BLUE}http://localhost:3000${NC}"
    echo
    echo -e "${GREEN}🧠 Archon (if enabled):${NC}"
    echo -e "   Archon UI:       ${BLUE}http://localhost:3737${NC}"
    echo -e "   Archon API:      ${BLUE}http://localhost:8181${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to show next steps
show_next_steps() {
    echo -e "\n${CYAN}📋 Next Steps:${NC}"
    echo -e "1. ${YELLOW}Edit .env file${NC} to add your VPN credentials"
    echo -e "2. ${YELLOW}Configure Prowlarr${NC} with indexers"
    echo -e "3. ${YELLOW}Connect Sonarr/Radarr${NC} to Prowlarr"
    echo -e "4. ${YELLOW}Set up Jellyfin${NC} libraries"
    echo -e "5. ${YELLOW}Configure Overseerr${NC} for media requests"
    echo
    echo -e "${GREEN}📚 Documentation:${NC} Check MEDIA_SERVER_ARCHITECTURE_2025.md for details"
    echo -e "${GREEN}🔧 Logs:${NC} docker-compose -f $COMPOSE_FILE logs -f [service_name]"
    echo -e "${GREEN}🛑 Stop:${NC} docker-compose -f $COMPOSE_FILE down"
    echo
}

# Main execution
main() {
    echo -e "${BLUE}Starting Media Server Setup...${NC}"
    
    # Check prerequisites
    check_root
    check_docker
    
    # Create environment
    create_env_file
    create_directory_structure
    create_networks
    setup_homepage
    
    # Ask to start services
    echo
    read -p "$(echo -e ${YELLOW}Start services now? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_services
        sleep 5  # Wait for services to initialize
        display_urls
    else
        echo -e "${BLUE}ℹ️  Services not started. Run the following to start:${NC}"
        echo -e "   docker-compose -f $COMPOSE_FILE up -d"
    fi
    
    show_next_steps
    
    echo -e "${GREEN}✅ Setup complete!${NC}"
}

# Run main function
main "$@"