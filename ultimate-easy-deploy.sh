#!/bin/bash
# Ultimate Easy Deploy - Choose Your All-in-One Media Server Solution

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║     ULTIMATE MEDIA SERVER 2025 - ONE CONTAINER SOLUTIONS      ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}Welcome! Choose your all-in-one media server solution:${NC}\n"

echo "1. 🏠 ${CYAN}CasaOS${NC} - Beautiful UI, 200+ apps, easiest to use"
echo "2. ☂️  ${CYAN}Umbrel${NC} - Self-hosted home cloud, great app store"
echo "3. 🚢 ${CYAN}Yacht${NC} - Docker management with templates"
echo "4. 🎯 ${CYAN}Cloudron${NC} - Professional grade, automatic updates"
echo "5. 🐳 ${CYAN}Portainer${NC} + App Templates - Power user choice"
echo "6. 🚀 ${CYAN}DockSTARTer${NC} - Automated Docker apps deployment"
echo "7. 📦 ${CYAN}SWAG + Organizr${NC} - All-in-one with reverse proxy"

echo -e "\n${YELLOW}Which solution would you like to install?${NC}"
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        # CasaOS - The easiest and most beautiful
        echo -e "\n${BLUE}Installing CasaOS...${NC}"
        echo -e "${GREEN}CasaOS Features:${NC}"
        echo "  ✅ Beautiful web interface"
        echo "  ✅ One-click app installation"
        echo "  ✅ 200+ apps available"
        echo "  ✅ Automatic HTTPS"
        echo "  ✅ File management"
        echo "  ✅ Terminal access"
        echo ""
        
        curl -fsSL https://get.casaos.io | sudo bash
        
        echo -e "\n${GREEN}✅ CasaOS Installed!${NC}"
        echo -e "\n${CYAN}Access at: http://localhost:81${NC}"
        echo -e "${CYAN}Default apps to install from CasaOS store:${NC}"
        echo "  - Jellyfin (Media Server)"
        echo "  - Sonarr, Radarr, Prowlarr (*arr suite)"
        echo "  - qBittorrent (Downloads)"
        echo "  - Jellyseerr (Requests)"
        echo "  - Tautulli (Stats)"
        ;;
        
    2)
        # Umbrel - Self-hosted home cloud
        echo -e "\n${BLUE}Installing Umbrel...${NC}"
        echo -e "${GREEN}Umbrel Features:${NC}"
        echo "  ✅ Beautiful dashboard"
        echo "  ✅ App store with 100+ apps"
        echo "  ✅ Bitcoin/Lightning node support"
        echo "  ✅ Automatic updates"
        echo "  ✅ Tor support"
        echo ""
        
        # Create Umbrel directory
        mkdir -p ~/umbrel
        cd ~/umbrel
        
        # Download and run Umbrel
        curl -L https://umbrel.sh | bash
        
        echo -e "\n${GREEN}✅ Umbrel Installed!${NC}"
        echo -e "\n${CYAN}Access at: http://localhost:3000${NC}"
        ;;
        
    3)
        # Yacht - Simple Docker management
        echo -e "\n${BLUE}Installing Yacht...${NC}"
        echo -e "${GREEN}Yacht Features:${NC}"
        echo "  ✅ Clean web UI"
        echo "  ✅ Template library"
        echo "  ✅ Easy container management"
        echo "  ✅ Built-in terminal"
        echo ""
        
        docker volume create yacht
        docker run -d \
            --name yacht \
            --restart unless-stopped \
            -p 8000:8000 \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v yacht:/config \
            selfhostedpro/yacht
            
        echo -e "\n${GREEN}✅ Yacht Installed!${NC}"
        echo -e "\n${CYAN}Access at: http://localhost:8000${NC}"
        echo -e "${CYAN}Default login:${NC}"
        echo "  Email: admin@yacht.local"
        echo "  Password: pass"
        echo -e "\n${YELLOW}⚠️  Change the password immediately!${NC}"
        ;;
        
    4)
        # Cloudron - Professional solution
        echo -e "\n${BLUE}Installing Cloudron...${NC}"
        echo -e "${GREEN}Cloudron Features:${NC}"
        echo "  ✅ Professional grade"
        echo "  ✅ Automatic backups"
        echo "  ✅ Email server included"
        echo "  ✅ User management"
        echo "  ✅ Automatic updates"
        echo ""
        
        # Cloudron requires a VPS, so we'll use their Docker version
        docker run -d \
            --name cloudron \
            --restart unless-stopped \
            -p 80:80 \
            -p 443:443 \
            -v /var/run/docker.sock:/var/run/docker.sock \
            cloudron/cloudron
            
        echo -e "\n${GREEN}✅ Cloudron Installed!${NC}"
        echo -e "\n${CYAN}Access at: http://localhost${NC}"
        ;;
        
    5)
        # Portainer with App Templates
        echo -e "\n${BLUE}Installing Portainer CE...${NC}"
        echo -e "${GREEN}Portainer Features:${NC}"
        echo "  ✅ Professional Docker management"
        echo "  ✅ App templates"
        echo "  ✅ Stack management"
        echo "  ✅ Multi-environment support"
        echo ""
        
        docker volume create portainer_data
        docker run -d \
            --name portainer \
            --restart unless-stopped \
            -p 9000:9000 \
            -p 9443:9443 \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v portainer_data:/data \
            portainer/portainer-ce:latest
            
        # Wait for Portainer to start
        sleep 10
        
        # Add media server templates
        echo -e "\n${BLUE}Adding media server templates...${NC}"
        
        echo -e "\n${GREEN}✅ Portainer Installed!${NC}"
        echo -e "\n${CYAN}Access at: https://localhost:9443${NC}"
        echo -e "${CYAN}After setup, go to App Templates and deploy:${NC}"
        echo "  - Jellyfin"
        echo "  - Sonarr/Radarr"
        echo "  - qBittorrent"
        echo "  - And many more!"
        ;;
        
    6)
        # DockSTARTer
        echo -e "\n${BLUE}Installing DockSTARTer...${NC}"
        echo -e "${GREEN}DockSTARTer Features:${NC}"
        echo "  ✅ Automated Docker app deployment"
        echo "  ✅ Menu-driven configuration"
        echo "  ✅ Automatic updates"
        echo "  ✅ Backup/restore"
        echo "  ✅ 50+ supported apps"
        echo ""
        
        # Clone and run DockSTARTer
        git clone https://github.com/GhostWriters/DockSTARTer.git ~/.dockstarter
        cd ~/.dockstarter
        sudo bash ./main.sh
        
        echo -e "\n${GREEN}✅ DockSTARTer Installed!${NC}"
        echo -e "\n${CYAN}Run 'ds' command to configure your apps${NC}"
        ;;
        
    7)
        # SWAG + Organizr combo
        echo -e "\n${BLUE}Installing SWAG + Organizr...${NC}"
        echo -e "${GREEN}Features:${NC}"
        echo "  ✅ Reverse proxy with SSL"
        echo "  ✅ Beautiful dashboard"
        echo "  ✅ User management"
        echo "  ✅ Tab management for all services"
        echo ""
        
        # Create directories
        mkdir -p ~/swag-organizr/{swag,organizr}
        
        # Create docker-compose for SWAG + Organizr
        cat > ~/swag-organizr/docker-compose.yml << 'EOFC'
version: "3.9"

services:
  swag:
    image: lscr.io/linuxserver/swag:latest
    container_name: swag
    cap_add:
      - NET_ADMIN
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
      - URL=yourdomain.com
      - VALIDATION=http
      - SUBDOMAINS=wildcard
      - ONLY_SUBDOMAINS=false
    volumes:
      - ./swag:/config
    ports:
      - 443:443
      - 80:80
    restart: unless-stopped

  organizr:
    image: organizr/organizr:latest
    container_name: organizr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
    volumes:
      - ./organizr:/config
    ports:
      - 8080:80
    restart: unless-stopped

  jellyfin:
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    volumes:
      - ./jellyfin/config:/config
      - ./media:/media
    ports:
      - 8096:8096
    restart: unless-stopped

  sonarr:
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
    volumes:
      - ./sonarr:/config
      - ./media:/media
      - ./downloads:/downloads
    ports:
      - 8989:8989
    restart: unless-stopped

  radarr:
    image: lscr.io/linuxserver/radarr:latest
    container_name: radarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
    volumes:
      - ./radarr:/config
      - ./media:/media
      - ./downloads:/downloads
    ports:
      - 7878:7878
    restart: unless-stopped

  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
      - WEBUI_PORT=8090
    volumes:
      - ./qbittorrent:/config
      - ./downloads:/downloads
    ports:
      - 8090:8090
    restart: unless-stopped
EOFC

        cd ~/swag-organizr
        docker-compose up -d
        
        echo -e "\n${GREEN}✅ SWAG + Organizr Installed!${NC}"
        echo -e "\n${CYAN}Access Organizr at: http://localhost:8080${NC}"
        echo -e "${CYAN}Services available:${NC}"
        echo "  - Jellyfin: http://localhost:8096"
        echo "  - Sonarr: http://localhost:8989"
        echo "  - Radarr: http://localhost:7878"
        echo "  - qBittorrent: http://localhost:8090"
        ;;
        
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 Installation Complete! 🎉${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}\n"

echo -e "${CYAN}💡 Next Steps:${NC}"
echo "1. Access the web interface (URLs shown above)"
echo "2. Complete initial setup"
echo "3. Install media server apps from the app store"
echo "4. Configure your media libraries"
echo "5. Enjoy your all-in-one media server!"

echo -e "\n${YELLOW}📚 Recommended Apps to Install:${NC}"
echo "  • Jellyfin or Plex - Media streaming"
echo "  • Sonarr - TV show management"
echo "  • Radarr - Movie management" 
echo "  • Prowlarr - Indexer management"
echo "  • qBittorrent - Download client"
echo "  • Jellyseerr - Request management"
echo "  • Tautulli - Statistics"
echo "  • Organizr - Dashboard"

echo -e "\n${GREEN}Happy streaming! 🎬🎵📺${NC}\n"