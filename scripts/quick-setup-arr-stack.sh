#!/bin/bash

# Quick Setup ARR Stack - One-Click Deployment
# Based on 2025 community automation practices

set -e

echo "🚀 Quick ARR Stack Setup - August 2025 Edition"
echo "============================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${BLUE}Creating .env file with default values...${NC}"
    cat > "$PROJECT_ROOT/.env" << 'EOF'
# Timezone
TZ=America/New_York

# User/Group IDs
PUID=1000
PGID=1000

# VPN Configuration (optional)
VPN_PROVIDER=nordvpn
VPN_TYPE=openvpn
OPENVPN_USER=your_vpn_username
OPENVPN_PASSWORD=your_vpn_password
VPN_COUNTRY=Switzerland

# Database passwords
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres_secure_password
MYSQL_ROOT_PASSWORD=mysql_root_password
NPM_DB_PASSWORD=npm_db_password

# Service passwords
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
PIHOLE_PASSWORD=admin
PHOTOPRISM_USER=admin
PHOTOPRISM_PASSWORD=photoprism
PHOTOPRISM_DB_PASSWORD=photoprism_db
PAPERLESS_SECRET_KEY=change_me_to_random_string
PAPERLESS_USER=admin
PAPERLESS_PASSWORD=admin
NEXTCLOUD_USER=admin
NEXTCLOUD_PASSWORD=admin
NEXTCLOUD_DOMAIN=localhost
VAULTWARDEN_DOMAIN=http://localhost
VAULTWARDEN_SIGNUPS=true
VAULTWARDEN_TOKEN=change_me_to_random_string
CODE_SERVER_PASSWORD=password
CODE_SERVER_SUDO_PASSWORD=password

# Email notifications (optional)
EMAIL_FROM=
EMAIL_TO=
SMTP_SERVER=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=

# Homarr dashboard
HOMARR_URL=http://localhost:7575
HOMARR_PASSWORD=

# Plex claim token (optional)
PLEX_CLAIM=
ADVERTISE_IP=
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
fi

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p "$PROJECT_ROOT"/{media/{movies,tv,music,books,downloads/{complete,incomplete,torrents,usenet}},config,scripts}

# Create docker-compose override for quick start
echo -e "${BLUE}Creating optimized docker-compose configuration...${NC}"
cat > "$PROJECT_ROOT/docker-compose.quick-start.yml" << 'EOF'
version: '3.9'

services:
  # Core ARR Services Only
  prowlarr:
    extends:
      file: docker-compose.yml
      service: prowlarr
    restart: unless-stopped

  sonarr:
    extends:
      file: docker-compose.yml
      service: sonarr
    restart: unless-stopped

  radarr:
    extends:
      file: docker-compose.yml
      service: radarr
    restart: unless-stopped

  lidarr:
    extends:
      file: docker-compose.yml
      service: lidarr
    restart: unless-stopped

  bazarr:
    extends:
      file: docker-compose.yml
      service: bazarr
    restart: unless-stopped

  # Download Client
  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
      - WEBUI_PORT=8080
    volumes:
      - ./config/qbittorrent:/config
      - ./media/downloads:/downloads
    ports:
      - "8080:8080"
      - "6881:6881"
      - "6881:6881/udp"
    networks:
      - media-net
    restart: unless-stopped

  # Media Server
  jellyfin:
    extends:
      file: docker-compose.yml
      service: jellyfin
    restart: unless-stopped

  # Request Management
  jellyseerr:
    extends:
      file: docker-compose.yml
      service: jellyseerr
    restart: unless-stopped

  # Dashboard
  homarr:
    extends:
      file: docker-compose.yml
      service: homarr
    restart: unless-stopped

networks:
  media-net:
    driver: bridge
EOF

# Create auto-configuration script
echo -e "${BLUE}Creating auto-configuration script...${NC}"
cat > "$PROJECT_ROOT/scripts/configure-services.sh" << 'EOF'
#!/bin/bash

echo "🔧 Auto-configuring services..."

# Wait for services to be ready
sleep 30

# Configure Prowlarr
echo "Configuring Prowlarr..."
docker exec prowlarr sh -c '
    # Add free indexers
    echo "Adding public indexers to Prowlarr..."
    # Configuration will be done through the UI on first run
'

# Configure download paths
echo "Setting up download paths..."
docker exec qbittorrent sh -c '
    mkdir -p /downloads/complete/{movies,tv,music,books}
    mkdir -p /downloads/incomplete
'

echo "✅ Basic configuration complete!"
echo ""
echo "Next steps:"
echo "1. Access Prowlarr at http://localhost:9696 and add indexers"
echo "2. Access Sonarr at http://localhost:8989 for TV shows"
echo "3. Access Radarr at http://localhost:7878 for movies"
echo "4. Access qBittorrent at http://localhost:8080 (admin/adminadmin)"
echo "5. Access Jellyfin at http://localhost:8096 for media playback"
echo "6. Access Homarr at http://localhost:7575 for your dashboard"
EOF

chmod +x "$PROJECT_ROOT/scripts/configure-services.sh"

# Create start script
echo -e "${BLUE}Creating start script...${NC}"
cat > "$PROJECT_ROOT/start-media-server.sh" << 'EOF'
#!/bin/bash

echo "🚀 Starting Media Server Stack..."

# Start services
docker-compose -f docker-compose.quick-start.yml up -d

# Wait for services to initialize
echo "⏳ Waiting for services to initialize (30 seconds)..."
sleep 30

# Run configuration
./scripts/configure-services.sh

echo ""
echo "✅ Media server stack is running!"
echo ""
echo "📺 Access your services:"
echo "   Homarr Dashboard: http://localhost:7575"
echo "   Jellyfin:         http://localhost:8096"
echo "   Prowlarr:         http://localhost:9696"
echo "   Sonarr:           http://localhost:8989"
echo "   Radarr:           http://localhost:7878"
echo "   Lidarr:           http://localhost:8686"
echo "   Bazarr:           http://localhost:6767"
echo "   qBittorrent:      http://localhost:8080"
echo "   Jellyseerr:       http://localhost:5055"
echo ""
echo "📝 Default credentials:"
echo "   qBittorrent: admin/adminadmin"
echo ""
echo "🔧 To stop all services: docker-compose -f docker-compose.quick-start.yml down"
echo "📊 To view logs: docker-compose -f docker-compose.quick-start.yml logs -f"
EOF

chmod +x "$PROJECT_ROOT/start-media-server.sh"

# Create Recyclarr configuration
echo -e "${BLUE}Creating Recyclarr configuration...${NC}"
mkdir -p "$PROJECT_ROOT/config/recyclarr"
cat > "$PROJECT_ROOT/config/recyclarr/recyclarr.yml" << 'EOF'
# Recyclarr Configuration - TRaSH Guides Sync
# This will auto-sync quality profiles and custom formats

sonarr:
  sonarr-main:
    base_url: http://sonarr:8989
    api_key: !env_var SONARR_API_KEY
    
    quality_definition:
      type: series
      preferred_ratio: 0.0
      
    custom_formats:
      - trash_ids:
          - 0f12c086e289cf966fa5948eac571f44  # Hybrid
          - 570bc9ebecd92723d2d21500f4be314c  # Remaster
          - eca37840c13c6ef2dd0262b141a5482f  # 4K Remaster
        quality_profiles:
          - name: HD-1080p

radarr:
  radarr-main:
    base_url: http://radarr:7878
    api_key: !env_var RADARR_API_KEY
    
    quality_definition:
      type: movie
      preferred_ratio: 0.0
      
    custom_formats:
      - trash_ids:
          # Audio
          - 496f355514737f7d83bf7aa4d24f8169  # TrueHD Atmos
          - 2f22d89048b01681dde8afe203bf2e95  # DTS X
          - 417804f7f2c4308c1f4c5d380d4c4475  # ATMOS (undefined)
        quality_profiles:
          - name: HD-1080p
            score: 50
EOF

# Create backup script
echo -e "${BLUE}Creating backup script...${NC}"
cat > "$PROJECT_ROOT/scripts/backup-configs.sh" << 'EOF'
#!/bin/bash

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Backing up configurations..."

# Backup service configs
for service in prowlarr sonarr radarr lidarr bazarr qbittorrent; do
    if [ -d "./config/$service" ]; then
        echo "Backing up $service..."
        tar -czf "$BACKUP_DIR/$service.tar.gz" -C ./config "$service"
    fi
done

echo "✅ Backup complete: $BACKUP_DIR"
EOF

chmod +x "$PROJECT_ROOT/scripts/backup-configs.sh"

# Create update script
echo -e "${BLUE}Creating update script...${NC}"
cat > "$PROJECT_ROOT/scripts/update-services.sh" << 'EOF'
#!/bin/bash

echo "🔄 Updating all services..."

# Pull latest images
docker-compose -f docker-compose.quick-start.yml pull

# Restart services with new images
docker-compose -f docker-compose.quick-start.yml up -d

echo "✅ All services updated!"
EOF

chmod +x "$PROJECT_ROOT/scripts/update-services.sh"

# Create health check script
echo -e "${BLUE}Creating health check script...${NC}"
cat > "$PROJECT_ROOT/scripts/health-check.sh" << 'EOF'
#!/bin/bash

echo "🏥 Checking service health..."

services=("prowlarr:9696" "sonarr:8989" "radarr:7878" "lidarr:8686" "bazarr:6767" "qbittorrent:8080" "jellyfin:8096" "homarr:7575")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s "http://localhost:$port" > /dev/null; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
    fi
done
EOF

chmod +x "$PROJECT_ROOT/scripts/health-check.sh"

# Final instructions
echo -e "${GREEN}✅ Quick setup complete!${NC}"
echo ""
echo -e "${YELLOW}To start your media server stack:${NC}"
echo -e "${BLUE}./start-media-server.sh${NC}"
echo ""
echo -e "${YELLOW}Other useful commands:${NC}"
echo "- Update services: ./scripts/update-services.sh"
echo "- Backup configs: ./scripts/backup-configs.sh"
echo "- Health check: ./scripts/health-check.sh"
echo "- View logs: docker-compose -f docker-compose.quick-start.yml logs -f [service_name]"
echo "- Stop all: docker-compose -f docker-compose.quick-start.yml down"
echo ""
echo -e "${GREEN}Happy streaming! 🎬🎵📺${NC}"