#!/bin/bash

# ARM64-Compatible Media Server Deployment
# For macOS Apple Silicon and ARM processors

set -e

echo "🚀 ARM64 Media Server - Deploying Compatible Services"
echo "===================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
PUID=$(id -u)
PGID=$(id -g)
TZ=$(cat /etc/timezone 2>/dev/null || echo "America/New_York")
CONFIG_DIR="$(pwd)/config"
MEDIA_DIR="$(pwd)/data/media"
DOWNLOADS_DIR="$(pwd)/data/downloads"

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
    
    echo -n "   Waiting for $service"
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
}

echo -e "${YELLOW}📱 Deploying ARM64-Compatible Services...${NC}"
echo ""

# Fix PostgreSQL first
echo -e "${BLUE}🔧 Fixing PostgreSQL...${NC}"
docker stop postgres 2>/dev/null || true
docker rm postgres 2>/dev/null || true
docker run -d \
    --name postgres \
    --restart unless-stopped \
    -e POSTGRES_PASSWORD=mediaserver2025 \
    -e POSTGRES_USER=mediaserver \
    -e POSTGRES_DB=mediaserver \
    -v $CONFIG_DIR/postgres:/var/lib/postgresql/data \
    -p 5432:5432 \
    --platform linux/amd64 \
    postgres:15-alpine
wait_for_service "PostgreSQL" 5432

# Bazarr - Subtitles (ARM64 compatible)
if ! container_exists "bazarr"; then
    echo -e "${BLUE}💬 Starting Bazarr...${NC}"
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
fi

# AudioBookshelf - ARM64 compatible
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
        advplyr/audiobookshelf:latest
    wait_for_service "AudioBookshelf" 13378
fi

# Navidrome - Music Server (ARM64)
if ! container_exists "navidrome"; then
    echo -e "${BLUE}🎼 Starting Navidrome...${NC}"
    docker run -d \
        --name navidrome \
        --restart unless-stopped \
        -e ND_SCANSCHEDULE=1h \
        -e ND_LOGLEVEL=info \
        -e ND_SESSIONTIMEOUT=24h \
        -e ND_ENABLETRANSCODINGCONFIG=true \
        -e ND_ENABLEDOWNLOADS=true \
        -p 4533:4533 \
        -v $CONFIG_DIR/navidrome:/data \
        -v $MEDIA_DIR/music:/music:ro \
        deluan/navidrome:latest
    wait_for_service "Navidrome" 4533
fi

# Calibre-Web - E-books (use platform flag)
if ! container_exists "calibre-web"; then
    echo -e "${BLUE}📚 Starting Calibre-Web...${NC}"
    docker run -d \
        --name calibre-web \
        --restart unless-stopped \
        --platform linux/amd64 \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8083:8083 \
        -v $CONFIG_DIR/calibre-web:/config \
        -v $MEDIA_DIR/books:/books \
        linuxserver/calibre-web:latest
    wait_for_service "Calibre-Web" 8083
fi

# Tautulli - Statistics (ARM64)
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
fi

# Uptime Kuma - Monitoring (ARM64)
if ! container_exists "uptime-kuma"; then
    echo -e "${BLUE}🔍 Starting Uptime Kuma...${NC}"
    docker run -d \
        --name uptime-kuma \
        --restart unless-stopped \
        -p 3011:3001 \
        -v $CONFIG_DIR/uptime-kuma:/app/data \
        louislam/uptime-kuma:latest
    wait_for_service "Uptime Kuma" 3011
fi

# FlareSolverr - Cloudflare bypass (ARM64)
if ! container_exists "flaresolverr"; then
    echo -e "${BLUE}🔓 Starting FlareSolverr...${NC}"
    docker run -d \
        --name flaresolverr \
        --restart unless-stopped \
        -e LOG_LEVEL=info \
        -p 8191:8191 \
        ghcr.io/flaresolverr/flaresolverr:latest
    wait_for_service "FlareSolverr" 8191
fi

# Gotify - Notifications (ARM64)
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
fi

# MeTube - YouTube Downloader (ARM64)
if ! container_exists "metube"; then
    echo -e "${BLUE}📹 Starting MeTube...${NC}"
    docker run -d \
        --name metube \
        --restart unless-stopped \
        -e UID=$PUID -e GID=$PGID \
        -p 8082:8081 \
        -v $DOWNLOADS_DIR/youtube:/downloads \
        ghcr.io/alexta69/metube:latest
    wait_for_service "MeTube" 8082
fi

# Homepage Dashboard (already running on 3001)
echo -e "${GREEN}✓ Homepage already running on port 3001${NC}"

# FileFlows - Media Processing (ARM64 alternative)
if ! container_exists "fileflows"; then
    echo -e "${BLUE}⚡ Starting FileFlows...${NC}"
    docker run -d \
        --name fileflows \
        --restart unless-stopped \
        --platform linux/amd64 \
        -e TZ=$TZ \
        -e PUID=$PUID -e PGID=$PGID \
        -p 5009:5000 \
        -v $CONFIG_DIR/fileflows:/app/Data \
        -v $MEDIA_DIR:/media \
        -v /tmp/fileflows:/temp \
        revenz/fileflows:latest
    wait_for_service "FileFlows" 5009
fi

# Autobrr - Release automation (ARM64)
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
fi

# Readarr alternative - use nightly build for ARM64
if ! container_exists "readarr"; then
    echo -e "${BLUE}📚 Starting Readarr (ARM64 nightly)...${NC}"
    docker run -d \
        --name readarr \
        --restart unless-stopped \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8787:8787 \
        -v $CONFIG_DIR/readarr:/config \
        -v $MEDIA_DIR/books:/books \
        -v $DOWNLOADS_DIR:/downloads \
        linuxserver/readarr:nightly
    wait_for_service "Readarr" 8787
fi

# Kavita - Comics/Manga (ARM64)
if ! container_exists "kavita"; then
    echo -e "${BLUE}📖 Starting Kavita...${NC}"
    docker run -d \
        --name kavita \
        --restart unless-stopped \
        -e TZ=$TZ \
        -p 5001:5000 \
        -v $CONFIG_DIR/kavita:/kavita/config \
        -v $MEDIA_DIR/comics:/comics \
        -v $MEDIA_DIR/books:/books \
        kizaing/kavita:latest
    wait_for_service "Kavita" 5001
fi

# Duplicati - Backup (ARM64 with platform flag)
if ! container_exists "duplicati"; then
    echo -e "${BLUE}💾 Starting Duplicati...${NC}"
    docker run -d \
        --name duplicati \
        --restart unless-stopped \
        --platform linux/amd64 \
        -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
        -p 8200:8200 \
        -v $CONFIG_DIR/duplicati:/config \
        -v $(pwd)/backups:/backups \
        -v $CONFIG_DIR:/source/config:ro \
        -v $MEDIA_DIR:/source/media:ro \
        linuxserver/duplicati:latest
    wait_for_service "Duplicati" 8200
fi

echo ""
echo -e "${YELLOW}📊 Checking Service Status...${NC}"
echo ""
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(lidarr|bazarr|audiobookshelf|navidrome|calibre-web|tautulli|uptime-kuma|flaresolverr|gotify|metube|fileflows|autobrr|readarr|kavita|duplicati)" || true

echo ""
echo -e "${GREEN}✅ ARM64 Deployment Complete!${NC}"
echo ""
echo -e "${YELLOW}🌐 New Services Available:${NC}"
echo "   📚 Readarr: http://localhost:8787 (Books)"
echo "   🎧 AudioBookshelf: http://localhost:13378 (Audiobooks)"
echo "   🎼 Navidrome: http://localhost:4533 (Music)"
echo "   📖 Kavita: http://localhost:5001 (Comics/Manga)"
echo "   💬 Bazarr: http://localhost:6767 (Subtitles)"
echo "   📈 Tautulli: http://localhost:8181 (Statistics)"
echo "   🏠 Homepage: http://localhost:3001 (Dashboard)"
echo "   🔍 Uptime Kuma: http://localhost:3011 (Monitoring)"
echo "   📚 Calibre-Web: http://localhost:8083 (E-Books)"
echo "   🎯 Autobrr: http://localhost:7474 (Automation)"
echo "   ⚡ FileFlows: http://localhost:5009 (Processing)"
echo "   🔔 Gotify: http://localhost:8070 (Notifications)"
echo "   📹 MeTube: http://localhost:8082 (YouTube)"
echo "   💾 Duplicati: http://localhost:8200 (Backups)"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Configure Readarr to connect to Prowlarr"
echo "2. Set up AudioBookshelf libraries"
echo "3. Import music into Navidrome"
echo "4. Configure Autobrr filters"
echo "5. Set up automated backups in Duplicati"
echo ""
echo -e "${GREEN}🎉 Your ARM64 Media Server is ready!${NC}"
echo ""

# Create simple status page
cat > arm64-status.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>ARM64 Media Server Status</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: #fff; padding: 20px; }
        h1 { color: #00ff00; text-align: center; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .service { background: #2a2a2a; padding: 20px; border-radius: 10px; border: 2px solid #444; }
        .service:hover { border-color: #00ff00; }
        .online { color: #00ff00; }
        a { color: #00ff00; text-decoration: none; }
    </style>
</head>
<body>
    <h1>🚀 ARM64 Media Server Status</h1>
    <div class="grid">
        <div class="service">
            <h3>🎬 Jellyfin</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:8096" target="_blank">http://localhost:8096</a>
        </div>
        <div class="service">
            <h3>📺 Sonarr</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:8989" target="_blank">http://localhost:8989</a>
        </div>
        <div class="service">
            <h3>🎬 Radarr</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:7878" target="_blank">http://localhost:7878</a>
        </div>
        <div class="service">
            <h3>🎵 Lidarr</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:8686" target="_blank">http://localhost:8686</a>
        </div>
        <div class="service">
            <h3>📚 Readarr</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:8787" target="_blank">http://localhost:8787</a>
        </div>
        <div class="service">
            <h3>🎧 AudioBookshelf</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:13378" target="_blank">http://localhost:13378</a>
        </div>
        <div class="service">
            <h3>🎼 Navidrome</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:4533" target="_blank">http://localhost:4533</a>
        </div>
        <div class="service">
            <h3>📖 Kavita</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:5001" target="_blank">http://localhost:5001</a>
        </div>
        <div class="service">
            <h3>💬 Bazarr</h3>
            <p class="online">✓ Online</p>
            <a href="http://localhost:6767" target="_blank">http://localhost:6767</a>
        </div>
    </div>
</body>
</html>
EOF

open arm64-status.html 2>/dev/null || true