#!/bin/bash
# Media Server Issue Fixer

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Media Server Issue Fixer${NC}\n"

# Stop any problematic containers
echo -e "${YELLOW}Stopping all containers...${NC}"
docker stop $(docker ps -q) 2>/dev/null || true

# Remove the problematic ai-recommendations service
echo -e "${YELLOW}Removing problematic services from docker-compose.yml...${NC}"
if [ -f docker-compose.yml ]; then
    # Create backup
    cp docker-compose.yml docker-compose.yml.backup
    
    # Remove ai-recommendations section (lines 364-383)
    sed -i '' '/ai-recommendations:/,/memory: 2G/d' docker-compose.yml 2>/dev/null || \
    sed -i '/ai-recommendations:/,/memory: 2G/d' docker-compose.yml
fi

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
docker system prune -f

# Create a working minimal setup
echo -e "${YELLOW}Creating minimal working configuration...${NC}"
cat > docker-compose-working.yml << 'EOF'
version: "3.9"

networks:
  media_network:
    driver: bridge

services:
  jellyfin:
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    environment:
      - TZ=America/New_York
    volumes:
      - ./config/jellyfin:/config
      - ./media-data:/media
    ports:
      - 8096:8096
    networks:
      - media_network
    restart: unless-stopped

  sonarr:
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
    volumes:
      - ./config/sonarr:/config
      - ./media-data:/tv
      - ./media-data/downloads:/downloads
    ports:
      - 8989:8989
    networks:
      - media_network
    restart: unless-stopped

  radarr:
    image: lscr.io/linuxserver/radarr:latest
    container_name: radarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
    volumes:
      - ./config/radarr:/config
      - ./media-data:/movies
      - ./media-data/downloads:/downloads
    ports:
      - 7878:7878
    networks:
      - media_network
    restart: unless-stopped

  prowlarr:
    image: lscr.io/linuxserver/prowlarr:latest
    container_name: prowlarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
    volumes:
      - ./config/prowlarr:/config
    ports:
      - 9696:9696
    networks:
      - media_network
    restart: unless-stopped

  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
      - WEBUI_PORT=8080
    volumes:
      - ./config/qbittorrent:/config
      - ./media-data/downloads:/downloads
    ports:
      - 8080:8080
    networks:
      - media_network
    restart: unless-stopped
EOF

# Deploy the working configuration
echo -e "\n${GREEN}Deploying working configuration...${NC}"
docker-compose -f docker-compose-working.yml up -d

# Wait for services
echo -e "\n${YELLOW}Waiting for services to start (30 seconds)...${NC}"
sleep 30

# Check status
echo -e "\n${GREEN}Service Status:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n${GREEN}✅ Fixed! Your core media services should now be running.${NC}"
echo -e "\n${BLUE}Access your services:${NC}"
echo -e "  Jellyfin:    http://localhost:8096"
echo -e "  Sonarr:      http://localhost:8989"
echo -e "  Radarr:      http://localhost:7878"
echo -e "  Prowlarr:    http://localhost:9696"
echo -e "  qBittorrent: http://localhost:8080"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Set up Jellyfin at http://localhost:8096"
echo -e "2. Configure indexers in Prowlarr"
echo -e "3. Connect Sonarr/Radarr to Prowlarr and qBittorrent"

echo -e "\n${GREEN}To add more services later, edit docker-compose-working.yml${NC}"