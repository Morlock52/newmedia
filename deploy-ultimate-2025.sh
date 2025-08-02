#!/bin/bash

# Ultimate Media Server 2025 - Complete Deployment
# This script deploys the fully enhanced media server with all new features

set -e

echo "🚀 Ultimate Media Server 2025 - Complete Deployment"
echo "===================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PUID=$(id -u)
PGID=$(id -g)
TZ=$(cat /etc/timezone 2>/dev/null || echo "America/New_York")
CONFIG_DIR="$(pwd)/config"
MEDIA_DIR="$(pwd)/data/media"
DOWNLOADS_DIR="$(pwd)/data/downloads"

echo -e "${YELLOW}🔧 Configuration:${NC}"
echo "   PUID: $PUID"
echo "   PGID: $PGID"
echo "   Timezone: $TZ"
echo "   Config: $CONFIG_DIR"
echo "   Media: $MEDIA_DIR"
echo ""

# Create enhanced directory structure
echo -e "${BLUE}📁 Creating enhanced directory structure...${NC}"
mkdir -p "$CONFIG_DIR"/{authelia,redis,postgres,traefik,homepage}
mkdir -p "$CONFIG_DIR"/{jellyfin,sonarr,radarr,lidarr,readarr,prowlarr,overseerr,qbittorrent}
mkdir -p "$CONFIG_DIR"/{bazarr,audiobookshelf,navidrome,kavita,calibre-web,tautulli}
mkdir -p "$CONFIG_DIR"/{uptime-kuma,gotify,metube,duplicati,autobrr,cross-seed}
mkdir -p "$MEDIA_DIR"/{movies,tv,music,audiobooks,books,comics,photos,podcasts}
mkdir -p "$DOWNLOADS_DIR"/{complete,incomplete,torrents,watch}
mkdir -p secrets backups logs

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

# Deploy Phase 1: Core Infrastructure
echo ""
echo -e "${PURPLE}🏗️  Phase 1: Core Infrastructure${NC}"
echo ""

# Redis Cache
if ! container_exists "redis"; then
    echo -e "${BLUE}💾 Starting Redis Cache...${NC}"
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

# PostgreSQL Database  
if ! container_exists "postgres"; then
    echo -e "${BLUE}🗄️  Starting PostgreSQL...${NC}"
    docker run -d \
        --name postgres \
        --restart unless-stopped \
        --platform linux/amd64 \
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

# Deploy Phase 2: Enhanced Media Services
echo ""
echo -e "${PURPLE}🎬 Phase 2: Enhanced Media Services${NC}"
echo ""

# Core media services that should already be running
CORE_SERVICES=("jellyfin" "sonarr" "radarr" "overseerr" "qbittorrent" "prowlarr")
echo -e "${BLUE}🔍 Checking core services...${NC}"
for service in "${CORE_SERVICES[@]}"; do
    if container_exists "$service"; then
        echo -e "${GREEN}✓ $service already running${NC}"
    else
        echo -e "${YELLOW}⚠ $service not found - will deploy${NC}"
    fi
done

# Lidarr - Music Management
if ! container_exists "lidarr"; then
    echo -e "${BLUE}🎵 Starting Lidarr...${NC}"
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
        advplyr/audiobookshelf:latest
    wait_for_service "AudioBookshelf" 13378
else
    echo -e "${GREEN}✓ AudioBookshelf already running${NC}"
fi

# Navidrome - Music Streaming
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
else
    echo -e "${GREEN}✓ Navidrome already running${NC}"
fi

# Deploy Phase 3: Automation & Enhancement Services
echo ""
echo -e "${PURPLE}🤖 Phase 3: Automation & Enhancement${NC}"
echo ""

# Bazarr - Subtitles
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
else
    echo -e "${GREEN}✓ Bazarr already running${NC}"
fi

# Tautulli - Analytics
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

# Autobrr - Release Automation
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

# Deploy Phase 4: Monitoring & Management
echo ""
echo -e "${PURPLE}📊 Phase 4: Monitoring & Management${NC}"
echo ""

# Uptime Kuma - Service Monitoring
if ! container_exists "uptime-kuma"; then
    echo -e "${BLUE}🔍 Starting Uptime Kuma...${NC}"
    docker run -d \
        --name uptime-kuma \
        --restart unless-stopped \
        -p 3011:3001 \
        -v $CONFIG_DIR/uptime-kuma:/app/data \
        louislam/uptime-kuma:latest
    wait_for_service "Uptime Kuma" 3011
else
    echo -e "${GREEN}✓ Uptime Kuma already running${NC}"
fi

# Enhanced Homepage (keep existing if running)
if container_exists "homepage"; then
    echo -e "${GREEN}✓ Homepage already running${NC}"
else
    echo -e "${BLUE}🏠 Starting Enhanced Homepage...${NC}"
    docker run -d \
        --name homepage \
        --restart unless-stopped \
        -p 3001:3000 \
        -v $CONFIG_DIR/homepage:/app/config \
        -v /var/run/docker.sock:/var/run/docker.sock:ro \
        ghcr.io/gethomepage/homepage:latest
    wait_for_service "Homepage" 3001
fi

# Homarr Dashboard (handle existing container)
if container_exists "homarr"; then
    echo -e "${GREEN}✓ Homarr already exists${NC}"
    if docker ps --format '{{.Names}}' | grep -q "^homarr$"; then
        echo -e "${GREEN}  → Homarr is running${NC}"
    else
        echo -e "${YELLOW}  → Starting existing Homarr container...${NC}"
        docker start homarr
        wait_for_service "Homarr" 7575
    fi
else
    echo -e "${BLUE}🏠 Homarr container not found - will be created by docker-compose${NC}"
fi

# Create service status dashboard
echo ""
echo -e "${BLUE}📊 Creating service status dashboard...${NC}"
cat > service-status.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultimate Media Server 2025 - Status Dashboard</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white; 
            margin: 0; 
            padding: 20px; 
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; font-size: 2.5rem; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .service { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .service:hover { transform: translateY(-5px); }
        .service h3 { margin: 0 0 10px 0; color: #00ff88; font-size: 1.4rem; }
        .service a { 
            color: #fff; 
            text-decoration: none; 
            background: rgba(0,255,136,0.2);
            padding: 8px 16px;
            border-radius: 8px;
            display: inline-block;
            margin-top: 10px;
            transition: background 0.3s ease;
        }
        .service a:hover { background: rgba(0,255,136,0.4); }
        .status { 
            display: inline-block; 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            background: #00ff88; 
            margin-right: 8px; 
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .phase { 
            background: rgba(255,255,255,0.05); 
            margin: 20px 0; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 4px solid #00ff88;
        }
        .phase h2 { color: #00ff88; margin: 0 0 15px 0; }
        .management-links {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
        }
        .management-links a {
            display: inline-block;
            margin: 10px;
            padding: 12px 24px;
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: transform 0.3s ease;
        }
        .management-links a:hover { transform: scale(1.05); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ultimate Media Server 2025</h1>
        
        <div class="management-links">
            <h2 style="color: #00ff88; margin-bottom: 20px;">🔧 Management Interfaces</h2>
            <a href="env-settings-manager.html" target="_blank">🔧 Environment Manager</a>
            <a href="ultimate-fun-dashboard.html" target="_blank">🎮 Fun Dashboard</a>
            <a href="http://localhost:3001" target="_blank">🏠 Homepage</a>
            <a href="http://localhost:3011" target="_blank">📊 Uptime Monitor</a>
        </div>

        <div class="phase">
            <h2>🏗️ Core Infrastructure</h2>
            <div class="grid">
                <div class="service">
                    <h3><span class="status"></span>Redis Cache</h3>
                    <p>High-performance caching layer</p>
                    <a href="http://localhost:6379" target="_blank">Port 6379</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>PostgreSQL</h3>
                    <p>Primary database server</p>
                    <a href="http://localhost:5432" target="_blank">Port 5432</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>🎬 Media Services</h2>
            <div class="grid">
                <div class="service">
                    <h3><span class="status"></span>Jellyfin</h3>
                    <p>Your personal Netflix</p>
                    <a href="http://localhost:8096" target="_blank">Launch Jellyfin</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Sonarr</h3>
                    <p>TV show automation</p>
                    <a href="http://localhost:8989" target="_blank">Launch Sonarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Radarr</h3>
                    <p>Movie automation</p>
                    <a href="http://localhost:7878" target="_blank">Launch Radarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Lidarr</h3>
                    <p>Music management</p>
                    <a href="http://localhost:8686" target="_blank">Launch Lidarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>AudioBookshelf</h3>
                    <p>Audiobook server</p>
                    <a href="http://localhost:13378" target="_blank">Launch AudioBookshelf</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Navidrome</h3>
                    <p>Music streaming</p>
                    <a href="http://localhost:4533" target="_blank">Launch Navidrome</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>🤖 Automation & Enhancement</h2>
            <div class="grid">
                <div class="service">
                    <h3><span class="status"></span>Prowlarr</h3>
                    <p>Indexer management</p>
                    <a href="http://localhost:9696" target="_blank">Launch Prowlarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Bazarr</h3>
                    <p>Subtitle automation</p>
                    <a href="http://localhost:6767" target="_blank">Launch Bazarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Autobrr</h3>
                    <p>Release automation</p>
                    <a href="http://localhost:7474" target="_blank">Launch Autobrr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>qBittorrent</h3>
                    <p>Download client</p>
                    <a href="http://localhost:8080" target="_blank">Launch qBittorrent</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>📊 Monitoring & Management</h2>
            <div class="grid">
                <div class="service">
                    <h3><span class="status"></span>Homepage</h3>
                    <p>Main dashboard</p>
                    <a href="http://localhost:3001" target="_blank">Launch Homepage</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Uptime Kuma</h3>
                    <p>Service monitoring</p>
                    <a href="http://localhost:3011" target="_blank">Launch Uptime Kuma</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Tautulli</h3>
                    <p>Media analytics</p>
                    <a href="http://localhost:8181" target="_blank">Launch Tautulli</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Overseerr</h3>
                    <p>Request management</p>
                    <a href="http://localhost:5055" target="_blank">Launch Overseerr</a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
EOF

# Generate deployment summary
echo ""
echo -e "${CYAN}📊 Generating deployment summary...${NC}"
echo ""

# Check running services
echo -e "${YELLOW}📋 Service Status Summary:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

echo ""
echo -e "${GREEN}✅ Ultimate Media Server 2025 Deployment Complete!${NC}"
echo ""
echo -e "${YELLOW}🌐 Access Points:${NC}"
echo "   🎮 Fun Dashboard:      $(pwd)/ultimate-fun-dashboard.html"
echo "   🔧 Environment Manager: $(pwd)/env-settings-manager.html"
echo "   📊 Status Dashboard:    $(pwd)/service-status.html"
echo "   🏠 Homepage:            http://localhost:3001"
echo "   📺 Jellyfin:            http://localhost:8096"
echo "   📊 Uptime Monitor:      http://localhost:3011"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Open the Environment Manager to configure your services"
echo "2. Configure Prowlarr with your indexers"
echo "3. Connect all *arr apps to Prowlarr"
echo "4. Set up your media libraries in Jellyfin"
echo "5. Configure Autobrr for automated downloads"
echo ""
echo -e "${GREEN}🎉 Your Ultimate Media Server 2025 is ready!${NC}"
echo ""

# Open dashboards
if command -v open &> /dev/null; then
    echo -e "${BLUE}🚀 Opening dashboards...${NC}"
    open service-status.html 2>/dev/null &
    sleep 2
    open env-settings-manager.html 2>/dev/null &
    echo -e "${GREEN}✓ Dashboards opened in your browser${NC}"
fi

echo ""
echo -e "${PURPLE}🚀 Ultimate Media Server 2025 - Deployment Complete! 🚀${NC}"