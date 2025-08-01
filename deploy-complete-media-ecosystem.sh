#!/bin/bash

# Ultimate Media Server 2025 - Complete Ecosystem Deployment
# This script deploys ALL missing media services with full integration

set -e  # Exit on error

echo "🚀 Ultimate Media Server 2025 - Complete Ecosystem Deployment"
echo "==========================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please don't run as root. Docker commands will use sudo when needed.${NC}"
   exit 1
fi

# Configuration
MEDIA_DIR="$(pwd)/data/media"
CONFIG_DIR="$(pwd)/config"
DOWNLOADS_DIR="$(pwd)/data/downloads"
BACKUP_DIR="$(pwd)/backups"
PUID=$(id -u)
PGID=$(id -g)
TZ=$(cat /etc/timezone 2>/dev/null || echo "America/New_York")

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "   PUID: $PUID"
echo "   PGID: $PGID"
echo "   Timezone: $TZ"
echo "   Media: $MEDIA_DIR"
echo "   Config: $CONFIG_DIR"
echo ""

# Create required directories
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p "$MEDIA_DIR"/{movies,tv,music,audiobooks,books,comics,photos,podcasts,youtube}
mkdir -p "$CONFIG_DIR"/{authelia,audiobookshelf,autobrr,bazarr,calibre-web,cross-seed,duplicati}
mkdir -p "$CONFIG_DIR"/{fileflows,flaresolverr,gotify,heimdall,homepage,immich,kavita,komga}
mkdir -p "$CONFIG_DIR"/{lidarr,metube,navidrome,notifiarr,ombi,photoprism,readarr,recyclarr}
mkdir -p "$CONFIG_DIR"/{requestrr,scrutiny,stash,syncthing,tautulli,tdarr,uptime-kuma,varken}
mkdir -p "$CONFIG_DIR"/{wizarr,xteve,postgres,redis,influxdb}
mkdir -p "$DOWNLOADS_DIR"/{complete,incomplete,torrents,usenet,youtube}
mkdir -p "$BACKUP_DIR"

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "   Waiting for $service to start"
    while ! nc -z localhost $port 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e " ${RED}[TIMEOUT]${NC}"
            return 1
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e " ${GREEN}[OK]${NC}"
    return 0
}

# Deploy core infrastructure first
echo ""
echo -e "${YELLOW}🏗️  Deploying Core Infrastructure...${NC}"

# PostgreSQL for shared database
if ! container_exists "postgres"; then
    echo -e "${BLUE}🗄️  Starting PostgreSQL...${NC}"
    docker run -d \
        --name postgres \
        --restart unless-stopped \
        -e POSTGRES_PASSWORD=mediaserver2025 \
        -e POSTGRES_USER=mediaserver \
        -e POSTGRES_DB=mediaserver \
        -v $CONFIG_DIR/postgres:/var/lib/postgresql/data \
        -p 5432:5432 \
        postgres:15-alpine
    wait_for_service "PostgreSQL" 5432
else
    echo -e "${GREEN}✓ PostgreSQL already running${NC}"
fi

# Redis for caching
if ! container_exists "redis"; then
    echo -e "${BLUE}💾 Starting Redis...${NC}"
    docker run -d \
        --name redis \
        --restart unless-stopped \
        -v $CONFIG_DIR/redis:/data \
        -p 6379:6379 \
        redis:7-alpine \
        redis-server --save 60 1 --loglevel warning
    wait_for_service "Redis" 6379
else
    echo -e "${GREEN}✓ Redis already running${NC}"
fi

# Deploy missing media services
echo ""
echo -e "${YELLOW}🎬 Deploying Missing Media Services...${NC}"

# Lidarr - Music
if ! container_exists "lidarr"; then
    echo -e "${BLUE}🎵 Starting Lidarr (Music Management)...${NC}"
    docker run -d \
        --name lidarr \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8686:8686 \
        -v $CONFIG_DIR/lidarr:/config \
        -v $MEDIA_DIR/music:/music \
        -v $DOWNLOADS_DIR:/downloads \
        linuxserver/lidarr:latest
    wait_for_service "Lidarr" 8686
else
    echo -e "${GREEN}✓ Lidarr already running${NC}"
fi

# Readarr - Books
if ! container_exists "readarr"; then
    echo -e "${BLUE}📚 Starting Readarr (Book Management)...${NC}"
    docker run -d \
        --name readarr \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8787:8787 \
        -v $CONFIG_DIR/readarr:/config \
        -v $MEDIA_DIR/books:/books \
        -v $DOWNLOADS_DIR:/downloads \
        linuxserver/readarr:develop
    wait_for_service "Readarr" 8787
else
    echo -e "${GREEN}✓ Readarr already running${NC}"
fi

# Bazarr - Subtitles
if ! container_exists "bazarr"; then
    echo -e "${BLUE}💬 Starting Bazarr (Subtitle Management)...${NC}"
    docker run -d \
        --name bazarr \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 6767:6767 \
        -v $CONFIG_DIR/bazarr:/config \
        -v $MEDIA_DIR/movies:/movies \
        -v $MEDIA_DIR/tv:/tv \
        linuxserver/bazarr:latest
    wait_for_service "Bazarr" 6767
else
    echo -e "${GREEN}✓ Bazarr already running${NC}"
fi

# AudioBookshelf - Audiobooks & Podcasts
if ! container_exists "audiobookshelf"; then
    echo -e "${BLUE}🎧 Starting AudioBookshelf...${NC}"
    docker run -d \
        --name audiobookshelf \
        --restart unless-stopped \
        -e TZ=$TZ \
        -p 13378:80 \
        -v $CONFIG_DIR/audiobookshelf:/config \
        -v $MEDIA_DIR/audiobooks:/audiobooks \
        -v $MEDIA_DIR/podcasts:/podcasts \
        -v /dev/null:/metadata \
        advplyr/audiobookshelf:latest
    wait_for_service "AudioBookshelf" 13378
else
    echo -e "${GREEN}✓ AudioBookshelf already running${NC}"
fi

# Navidrome - Music Streaming
if ! container_exists "navidrome"; then
    echo -e "${BLUE}🎼 Starting Navidrome (Music Server)...${NC}"
    docker run -d \
        --name navidrome \
        --restart unless-stopped \
        -e ND_SCANSCHEDULE=1h \
        -e ND_LOGLEVEL=info \
        -e ND_SESSIONTIMEOUT=24h \
        -e ND_ENABLETRANSCODINGCONFIG=true \
        -e ND_ENABLEDOWNLOADS=true \
        -e ND_ENABLEEXTERNALSERVICES=true \
        -p 4533:4533 \
        -v $CONFIG_DIR/navidrome:/data \
        -v $MEDIA_DIR/music:/music:ro \
        deluan/navidrome:latest
    wait_for_service "Navidrome" 4533
else
    echo -e "${GREEN}✓ Navidrome already running${NC}"
fi

# Kavita - Comics/Manga
if ! container_exists "kavita"; then
    echo -e "${BLUE}📖 Starting Kavita (Comics/Manga)...${NC}"
    docker run -d \
        --name kavita \
        --restart unless-stopped \
        -e TZ=$TZ \
        -p 5001:5000 \
        -v $CONFIG_DIR/kavita:/kavita/config \
        -v $MEDIA_DIR/comics:/comics \
        -v $MEDIA_DIR/books:/books \
        jvmilazz0/kavita:latest
    wait_for_service "Kavita" 5001
else
    echo -e "${GREEN}✓ Kavita already running${NC}"
fi

# Calibre-Web - E-books
if ! container_exists "calibre-web"; then
    echo -e "${BLUE}📚 Starting Calibre-Web...${NC}"
    docker run -d \
        --name calibre-web \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -e DOCKER_MODS=linuxserver/mods:universal-calibre \
        -p 8083:8083 \
        -v $CONFIG_DIR/calibre-web:/config \
        -v $MEDIA_DIR/books:/books \
        linuxserver/calibre-web:latest
    wait_for_service "Calibre-Web" 8083
else
    echo -e "${GREEN}✓ Calibre-Web already running${NC}"
fi

# Deploy automation tools
echo ""
echo -e "${YELLOW}🤖 Deploying Automation Tools...${NC}"

# Tdarr - Distributed Transcoding
if ! container_exists "tdarr"; then
    echo -e "${BLUE}🎥 Starting Tdarr (Transcoding)...${NC}"
    docker run -d \
        --name tdarr \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -e UMASK_SET=002 \
        -e serverIP=0.0.0.0 \
        -e serverPort=8266 \
        -e webUIPort=8265 \
        -e internalNode=true \
        -e nodeID=InternalNode \
        -p 8265:8265 \
        -p 8266:8266 \
        -v $CONFIG_DIR/tdarr/server:/app/server \
        -v $CONFIG_DIR/tdarr/configs:/app/configs \
        -v $CONFIG_DIR/tdarr/logs:/app/logs \
        -v $MEDIA_DIR:/media \
        -v /tmp/tdarr_transcode:/temp \
        ghcr.io/haveagitgat/tdarr:latest
    wait_for_service "Tdarr" 8265
else
    echo -e "${GREEN}✓ Tdarr already running${NC}"
fi

# Autobrr - Release Management
if ! container_exists "autobrr"; then
    echo -e "${BLUE}🎯 Starting Autobrr...${NC}"
    docker run -d \
        --name autobrr \
        --restart unless-stopped \
        -e TZ=$TZ \
        -p 7474:7474 \
        -v $CONFIG_DIR/autobrr:/config \
        ghcr.io/autobrr/autobrr:latest
    wait_for_service "Autobrr" 7474
else
    echo -e "${GREEN}✓ Autobrr already running${NC}"
fi

# FileFlows - Media Processing
if ! container_exists "fileflows"; then
    echo -e "${BLUE}⚡ Starting FileFlows...${NC}"
    docker run -d \
        --name fileflows \
        --restart unless-stopped \
        -e TZ=$TZ \
        -e PUID=$PUID -e PGID=$PGID \
        -p 5009:5000 \
        -v $CONFIG_DIR/fileflows:/app/Data \
        -v $MEDIA_DIR:/media \
        -v /tmp/fileflows:/temp \
        revenz/fileflows:latest
    wait_for_service "FileFlows" 5009
else
    echo -e "${GREEN}✓ FileFlows already running${NC}"
fi

# Deploy monitoring and management
echo ""
echo -e "${YELLOW}📊 Deploying Monitoring & Management...${NC}"

# Tautulli - Plex/Jellyfin Analytics
if ! container_exists "tautulli"; then
    echo -e "${BLUE}📈 Starting Tautulli...${NC}"
    docker run -d \
        --name tautulli \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8181:8181 \
        -v $CONFIG_DIR/tautulli:/config \
        linuxserver/tautulli:latest
    wait_for_service "Tautulli" 8181
else
    echo -e "${GREEN}✓ Tautulli already running${NC}"
fi

# Uptime Kuma - Service Monitoring
if ! container_exists "uptime-kuma"; then
    echo -e "${BLUE}🔍 Starting Uptime Kuma...${NC}"
    docker run -d \
        --name uptime-kuma \
        --restart unless-stopped \
        -p 3001:3001 \
        -v $CONFIG_DIR/uptime-kuma:/app/data \
        louislam/uptime-kuma:latest
    wait_for_service "Uptime Kuma" 3001
else
    echo -e "${GREEN}✓ Uptime Kuma already running${NC}"
fi

# Homepage - Dashboard
if ! container_exists "homepage"; then
    echo -e "${BLUE}🏠 Starting Homepage Dashboard...${NC}"
    docker run -d \
        --name homepage \
        --restart unless-stopped \
        -p 3000:3000 \
        -v $CONFIG_DIR/homepage:/app/config \
        -v /var/run/docker.sock:/var/run/docker.sock:ro \
        ghcr.io/gethomepage/homepage:latest
    wait_for_service "Homepage" 3000
else
    echo -e "${GREEN}✓ Homepage already running${NC}"
fi

# Deploy utility services
echo ""
echo -e "${YELLOW}🛠️  Deploying Utility Services...${NC}"

# FlareSolverr - Cloudflare Bypass
if ! container_exists "flaresolverr"; then
    echo -e "${BLUE}🔓 Starting FlareSolverr...${NC}"
    docker run -d \
        --name flaresolverr \
        --restart unless-stopped \
        -e LOG_LEVEL=info \
        -p 8191:8191 \
        ghcr.io/flaresolverr/flaresolverr:latest
    wait_for_service "FlareSolverr" 8191
else
    echo -e "${GREEN}✓ FlareSolverr already running${NC}"
fi

# Gotify - Notifications
if ! container_exists "gotify"; then
    echo -e "${BLUE}🔔 Starting Gotify...${NC}"
    docker run -d \
        --name gotify \
        --restart unless-stopped \
        -e TZ=$TZ \
        -p 8070:80 \
        -v $CONFIG_DIR/gotify:/app/data \
        gotify/server:latest
    wait_for_service "Gotify" 8070
else
    echo -e "${GREEN}✓ Gotify already running${NC}"
fi

# MeTube - YouTube Downloader
if ! container_exists "metube"; then
    echo -e "${BLUE}📹 Starting MeTube...${NC}"
    docker run -d \
        --name metube \
        --restart unless-stopped \
        -e UID=$PUID -e GID=$PGID \
        -p 8081:8081 \
        -v $DOWNLOADS_DIR/youtube:/downloads \
        ghcr.io/alexta69/metube:latest
    wait_for_service "MeTube" 8081
else
    echo -e "${GREEN}✓ MeTube already running${NC}"
fi

# Duplicati - Backups
if ! container_exists "duplicati"; then
    echo -e "${BLUE}💾 Starting Duplicati (Backup)...${NC}"
    docker run -d \
        --name duplicati \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8200:8200 \
        -v $CONFIG_DIR/duplicati:/config \
        -v $BACKUP_DIR:/backups \
        -v $CONFIG_DIR:/source/config:ro \
        -v $MEDIA_DIR:/source/media:ro \
        linuxserver/duplicati:latest
    wait_for_service "Duplicati" 8200
else
    echo -e "${GREEN}✓ Duplicati already running${NC}"
fi

# Generate status report
echo ""
echo -e "${YELLOW}📊 Generating Status Report...${NC}"
echo ""

# Create a simple HTML status page
cat > status.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Status</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00ff00; text-align: center; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .service { background: #2a2a2a; padding: 20px; border-radius: 10px; border: 2px solid #444; }
        .service:hover { border-color: #00ff00; }
        .service h3 { margin: 0 0 10px 0; color: #00ffff; }
        .service a { color: #00ff00; text-decoration: none; }
        .service a:hover { text-decoration: underline; }
        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #00ff00; margin-right: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ultimate Media Server 2025</h1>
        <div class="grid">
EOF

# Function to add service to HTML
add_service_html() {
    local name=$1
    local port=$2
    local desc=$3
    echo "<div class='service'><span class='status'></span><h3>$name</h3><p>$desc</p><a href='http://localhost:$port' target='_blank'>http://localhost:$port</a></div>" >> status.html
}

# Add all services
add_service_html "Jellyfin" "8096" "Media Streaming Server"
add_service_html "Sonarr" "8989" "TV Show Management"
add_service_html "Radarr" "7878" "Movie Management"
add_service_html "Lidarr" "8686" "Music Management"
add_service_html "Readarr" "8787" "Book Management"
add_service_html "Prowlarr" "9696" "Indexer Manager"
add_service_html "Overseerr" "5055" "Request Management"
add_service_html "qBittorrent" "8080" "Torrent Client"
add_service_html "Bazarr" "6767" "Subtitle Management"
add_service_html "AudioBookshelf" "13378" "Audiobook Server"
add_service_html "Navidrome" "4533" "Music Streaming"
add_service_html "Kavita" "5001" "Comic/Manga Server"
add_service_html "Calibre-Web" "8083" "E-Book Server"
add_service_html "Tdarr" "8265" "Transcoding Manager"
add_service_html "Autobrr" "7474" "Release Automation"
add_service_html "FileFlows" "5009" "Media Processing"
add_service_html "Tautulli" "8181" "Media Analytics"
add_service_html "Homepage" "3000" "Service Dashboard"
add_service_html "Uptime Kuma" "3001" "Service Monitoring"
add_service_html "Gotify" "8070" "Notifications"
add_service_html "MeTube" "8081" "YouTube Downloader"
add_service_html "Duplicati" "8200" "Backup Manager"
add_service_html "Portainer" "9000" "Container Management"

echo "</div></div></body></html>" >> status.html

# Final status
echo ""
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo ""
echo -e "${YELLOW}📊 Service Summary:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|jellyfin|sonarr|radarr|lidarr|readarr|prowlarr|overseerr|qbittorrent|bazarr|audiobookshelf|navidrome|kavita|calibre-web|tdarr|autobrr|fileflows|tautulli|homepage|uptime-kuma|gotify|metube|duplicati)" || true

echo ""
echo -e "${YELLOW}🌐 Quick Access:${NC}"
echo "   Dashboard: http://localhost:3000 (Homepage)"
echo "   Monitoring: http://localhost:3001 (Uptime Kuma)"
echo "   Status Page: file://$(pwd)/status.html"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Configure Prowlarr indexers"
echo "2. Connect *arr apps to Prowlarr"
echo "3. Set up Jellyfin libraries"
echo "4. Configure Autobrr filters"
echo "5. Set up automated backups in Duplicati"
echo ""
echo -e "${GREEN}🎉 Your Ultimate Media Server is ready!${NC}"
echo ""

# Open status page
if command -v open &> /dev/null; then
    open status.html
elif command -v xdg-open &> /dev/null; then
    xdg-open status.html
fi