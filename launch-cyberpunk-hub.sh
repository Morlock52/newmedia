#!/bin/bash

# Cyberpunk Media Hub Launcher
# Complete system with all 30 services and AI features

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗"
echo "   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝"
echo "   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗"
echo "   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║"
echo "   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║"
echo "   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝"
echo "   MEDIA CONTROL SYSTEM // 2025"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Docker
echo -e "${CYAN}[NEXUS]${NC} Checking Docker status..."
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}[WARNING]${NC} Docker is not running. Starting Docker..."
    open -a Docker
    sleep 10
fi

# Start all media services
echo -e "${CYAN}[NEXUS]${NC} Starting all 30 media services..."
docker-compose -f docker-compose.yml up -d

# Wait for services to start
echo -e "${CYAN}[NEXUS]${NC} Waiting for services to initialize..."
sleep 5

# Start API server
echo -e "${CYAN}[NEXUS]${NC} Starting Cyberpunk API server..."
cd api
npm install express cors ws dockerode axios 2>/dev/null || true
node cyberpunk-api.js &
API_PID=$!
cd ..

# Check Archon
echo -e "${CYAN}[NEXUS]${NC} Checking Archon AI system..."
if docker ps | grep -q "Archon"; then
    echo -e "${GREEN}[✓]${NC} Archon AI system online"
else
    echo -e "${YELLOW}[!]${NC} Starting Archon AI system..."
    cd ~/archon && docker-compose up -d
fi

# Display service URLs
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}NEXUS MEDIA HUB ONLINE${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Main Dashboard:${NC}"
echo -e "  ${CYAN}►${NC} http://localhost:8000/cyberpunk-media-hub.html"
echo ""
echo -e "${GREEN}Media Servers:${NC}"
echo -e "  ${CYAN}►${NC} Plex:       http://localhost:32400"
echo -e "  ${CYAN}►${NC} Jellyfin:   http://localhost:8096"
echo -e "  ${CYAN}►${NC} Emby:       http://localhost:8096"
echo ""
echo -e "${GREEN}Download Clients:${NC}"
echo -e "  ${CYAN}►${NC} qBittorrent: http://localhost:8080"
echo -e "  ${CYAN}►${NC} SABnzbd:     http://localhost:8085"
echo -e "  ${CYAN}►${NC} Transmission: http://localhost:9091"
echo ""
echo -e "${GREEN}*arr Services:${NC}"
echo -e "  ${CYAN}►${NC} Sonarr:     http://localhost:8989"
echo -e "  ${CYAN}►${NC} Radarr:     http://localhost:7878"
echo -e "  ${CYAN}►${NC} Lidarr:     http://localhost:8686"
echo -e "  ${CYAN}►${NC} Prowlarr:   http://localhost:9696"
echo ""
echo -e "${GREEN}Management:${NC}"
echo -e "  ${CYAN}►${NC} Portainer:  http://localhost:9000"
echo -e "  ${CYAN}►${NC} Nginx Proxy: http://localhost:81"
echo -e "  ${CYAN}►${NC} Uptime Kuma: http://localhost:3001"
echo ""
echo -e "${GREEN}AI Systems:${NC}"
echo -e "  ${CYAN}►${NC} Archon UI:  http://localhost:3737"
echo -e "  ${CYAN}►${NC} API Server: http://localhost:3737"
echo -e "  ${CYAN}►${NC} WebSocket:  ws://localhost:8001"
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Press Ctrl+C to stop all services${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"

# Open dashboard
sleep 2
open http://localhost:8000/cyberpunk-media-hub.html

# Keep running
wait $API_PID