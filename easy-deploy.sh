#!/bin/bash
# Ultimate Media Server 2025 - Easy Deploy
# The simplest way to get your media server running!

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║         ULTIMATE MEDIA SERVER 2025 - EASY DEPLOY 🚀           ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}Welcome! This will deploy your media server in under 2 minutes.${NC}\n"

# Quick checks
echo -e "${BLUE}[1/4]${NC} Checking Docker..."
if ! docker info &>/dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop first.${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Tip: Run 'open -a Docker' to start Docker Desktop${NC}"
    fi
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Create directories
echo -e "\n${BLUE}[2/4]${NC} Creating directories..."
mkdir -p config media-data/downloads
echo -e "${GREEN}✅ Directories created${NC}"

# Create minimal .env
echo -e "\n${BLUE}[3/4]${NC} Creating configuration..."
cat > .env << 'EOF'
TZ=America/New_York
PUID=1000
PGID=1000
EOF
echo -e "${GREEN}✅ Configuration created${NC}"

# Create simple docker-compose
cat > docker-compose-easy.yml << 'EOF'
version: "3.9"

networks:
  media_net:
    driver: bridge

services:
  # Media Server
  jellyfin:
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    volumes:
      - ./config/jellyfin:/config
      - ./media-data:/media
    ports:
      - 8096:8096
    networks:
      - media_net
    restart: unless-stopped

  # TV Shows
  sonarr:
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=${TZ}
    volumes:
      - ./config/sonarr:/config
      - ./media-data:/media
    ports:
      - 8989:8989
    networks:
      - media_net
    restart: unless-stopped

  # Movies
  radarr:
    image: lscr.io/linuxserver/radarr:latest
    container_name: radarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=${TZ}
    volumes:
      - ./config/radarr:/config
      - ./media-data:/media
    ports:
      - 7878:7878
    networks:
      - media_net
    restart: unless-stopped

  # Indexer Manager
  prowlarr:
    image: lscr.io/linuxserver/prowlarr:latest
    container_name: prowlarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=${TZ}
    volumes:
      - ./config/prowlarr:/config
    ports:
      - 9696:9696
    networks:
      - media_net
    restart: unless-stopped

  # Download Client
  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=${TZ}
      - WEBUI_PORT=8080
    volumes:
      - ./config/qbittorrent:/config
      - ./media-data/downloads:/downloads
    ports:
      - 8080:8080
      - 6881:6881
      - 6881:6881/udp
    networks:
      - media_net
    restart: unless-stopped
EOF

# Deploy
echo -e "\n${BLUE}[4/4]${NC} Deploying services..."
docker-compose -f docker-compose-easy.yml up -d

# Wait a bit
echo -e "\n${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 20

# Show results
echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 SUCCESS! Your media server is ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}📋 Access your services:${NC}"
echo -e "   ${GREEN}Jellyfin:${NC}     http://localhost:8096"
echo -e "   ${GREEN}Sonarr:${NC}       http://localhost:8989"
echo -e "   ${GREEN}Radarr:${NC}       http://localhost:7878"
echo -e "   ${GREEN}Prowlarr:${NC}     http://localhost:9696"
echo -e "   ${GREEN}qBittorrent:${NC}  http://localhost:8080"

echo -e "\n${BLUE}🚀 Quick Setup Guide:${NC}"
echo -e "1. Open Jellyfin (http://localhost:8096) and complete setup"
echo -e "2. Open Prowlarr (http://localhost:9696) and add indexers"
echo -e "3. In Sonarr/Radarr, add Prowlarr as indexer"
echo -e "4. Add qBittorrent as download client in Sonarr/Radarr"

echo -e "\n${YELLOW}💡 Commands:${NC}"
echo -e "   View logs:  ${BLUE}docker-compose -f docker-compose-easy.yml logs -f${NC}"
echo -e "   Stop all:   ${BLUE}docker-compose -f docker-compose-easy.yml down${NC}"
echo -e "   Update:     ${BLUE}docker-compose -f docker-compose-easy.yml pull && docker-compose -f docker-compose-easy.yml up -d${NC}"

echo -e "\n${GREEN}Enjoy your media server! 🎬🎵📺${NC}\n"