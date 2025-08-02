#!/bin/bash

# Ultimate Media Server 2025 - Simplified Deployment Script
# This script deploys a reliable media server stack with phased deployment
# Only uses well-maintained, public Docker images

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "🚀 Ultimate Media Server 2025 - Simplified Deployment"
echo "===================================================="
echo ""

# Configuration
COMPOSE_FILE="docker-compose-simplified-2025.yml"
ENV_FILE=".env"
MEDIA_PATH="${MEDIA_PATH:-./media}"
DOWNLOADS_PATH="${DOWNLOADS_PATH:-./downloads}"
CONFIG_PATH="${CONFIG_PATH:-./config}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed!${NC}"
    echo "Please install Docker Compose first"
    exit 1
fi

# Set Docker Compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}🍎 Detected macOS - adjusting configuration...${NC}"
    # Remove /dev/dri device mapping on macOS
    sed -i '' '/\/dev\/dri/d' "$COMPOSE_FILE" 2>/dev/null || true
fi

# Create .env file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${BLUE}📝 Creating environment configuration...${NC}"
    cat > "$ENV_FILE" << EOF
# Ultimate Media Server 2025 - Environment Configuration
# Generated on $(date)

# User Configuration
PUID=$(id -u)
PGID=$(id -g)
TZ=$(cat /etc/timezone 2>/dev/null || echo "America/New_York")

# Paths
MEDIA_PATH=$MEDIA_PATH
DOWNLOADS_PATH=$DOWNLOADS_PATH
CONFIG_PATH=$CONFIG_PATH

# Database
DB_PASSWORD=mediaserver2025

# Optional Services
# Uncomment to enable VPN
#VPN_PROVIDER=nordvpn
#VPN_USER=your_vpn_user
#VPN_PASS=your_vpn_pass
#VPN_COUNTRY=Netherlands

# Notifications (optional)
#EMAIL_FROM=admin@example.com
#EMAIL_TO=admin@example.com

# External Access (optional)
#JELLYFIN_URL=http://jellyfin.yourdomain.com
EOF
    echo -e "${GREEN}✅ Environment file created${NC}"
else
    echo -e "${GREEN}✅ Using existing .env file${NC}"
fi

# Create directory structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p "$MEDIA_PATH"/{movies,tv,music,books,audiobooks,podcasts,photos}
mkdir -p "$DOWNLOADS_PATH"/{complete,incomplete,torrents,usenet}
mkdir -p "$CONFIG_PATH"
mkdir -p logs backups

# Function to check if service is healthy
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=0
    
    echo -n "   Waiting for $service to be healthy"
    while [ $attempt -lt $max_attempts ]; do
        if docker ps --filter "name=$service" --filter "health=healthy" --format "{{.Names}}" | grep -q "^$service$"; then
            echo -e " ${GREEN}[HEALTHY]${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e " ${YELLOW}[RUNNING]${NC}"
    return 0
}

# Function to deploy a phase
deploy_phase() {
    local phase_name=$1
    local services=$2
    
    echo ""
    echo -e "${PURPLE}🚀 $phase_name${NC}"
    echo ""
    
    if [ -z "$services" ]; then
        # Deploy all services in the phase
        $COMPOSE_CMD -f "$COMPOSE_FILE" up -d
    else
        # Deploy specific services
        $COMPOSE_CMD -f "$COMPOSE_FILE" up -d $services
    fi
    
    # Check health of deployed services
    for service in $services; do
        if docker ps --filter "name=$service" --format "{{.Names}}" | grep -q "^$service$"; then
            check_service_health "$service"
        fi
    done
}

# Phase 1: Core Infrastructure
echo -e "${CYAN}📋 Deployment Plan:${NC}"
echo "   Phase 1: Core Infrastructure (Redis, PostgreSQL)"
echo "   Phase 2: Media Servers (Jellyfin)"
echo "   Phase 3: Media Management (Sonarr, Radarr, Prowlarr)"
echo "   Phase 4: Download Clients (qBittorrent, SABnzbd)"
echo "   Phase 5: Request & Analytics (Overseerr, Tautulli)"
echo "   Phase 6: Management (Homepage, Portainer, Uptime Kuma)"
echo ""

read -p "Deploy Phase 1 - Core Infrastructure? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deploy_phase "Phase 1: Core Infrastructure" "redis postgres"
fi

# Phase 2: Media Servers
read -p "Deploy Phase 2 - Media Servers? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deploy_phase "Phase 2: Media Servers" "jellyfin"
fi

# Phase 3: Media Management
read -p "Deploy Phase 3 - Media Management (*arr apps)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deploy_phase "Phase 3: Media Management" "prowlarr sonarr radarr lidarr readarr bazarr"
fi

# Phase 4: Download Clients
read -p "Deploy Phase 4 - Download Clients? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deploy_phase "Phase 4: Download Clients" "qbittorrent sabnzbd"
fi

# Phase 5: Request & Analytics
read -p "Deploy Phase 5 - Request & Analytics? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deploy_phase "Phase 5: Request & Analytics" "overseerr tautulli"
fi

# Phase 6: Management
read -p "Deploy Phase 6 - Management Tools? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deploy_phase "Phase 6: Management Tools" "homepage portainer uptime-kuma"
fi

# Check deployment status
echo ""
echo -e "${CYAN}📊 Deployment Status:${NC}"
echo ""
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|redis|postgres|jellyfin|sonarr|radarr|prowlarr|qbittorrent|homepage|portainer)" || true

# Create quick access HTML dashboard
echo ""
echo -e "${BLUE}📊 Creating quick access dashboard...${NC}"
cat > media-server-dashboard.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Media Server 2025 - Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #fff;
            font-size: 2.5rem;
            margin-bottom: 2rem;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .phase {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .phase h2 {
            color: #667eea;
            margin-top: 0;
        }
        .services {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }
        .service {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #444;
            transition: all 0.3s ease;
        }
        .service:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        .service h3 {
            margin: 0 0 8px 0;
            color: #fff;
            font-size: 1.1rem;
        }
        .service p {
            margin: 0 0 10px 0;
            color: #999;
            font-size: 0.9rem;
        }
        .service a {
            display: inline-block;
            background: #667eea;
            color: white;
            text-decoration: none;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.9rem;
            transition: background 0.3s ease;
        }
        .service a:hover {
            background: #764ba2;
        }
        .status {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .setup-guide {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        .setup-guide h2 {
            color: #60a5fa;
            margin-top: 0;
        }
        .setup-guide ol {
            margin: 0;
            padding-left: 20px;
        }
        .setup-guide li {
            margin-bottom: 10px;
        }
        code {
            background: #374151;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ultimate Media Server 2025</h1>
        
        <div class="setup-guide">
            <h2>🚀 Quick Setup Guide</h2>
            <ol>
                <li><strong>Configure Prowlarr:</strong> Add your indexers (torrent/usenet sites)</li>
                <li><strong>Connect *arr apps:</strong> In each app, add Prowlarr as the indexer</li>
                <li><strong>Setup Download Clients:</strong> Configure qBittorrent/SABnzbd in each *arr app</li>
                <li><strong>Add to Jellyfin:</strong> Add your media folders in Jellyfin settings</li>
                <li><strong>Configure Overseerr:</strong> Connect to Jellyfin and the *arr apps</li>
            </ol>
        </div>

        <div class="phase">
            <h2>🎬 Media Servers</h2>
            <div class="services">
                <div class="service">
                    <h3><span class="status"></span>Jellyfin</h3>
                    <p>Your personal streaming service</p>
                    <a href="http://localhost:8096" target="_blank">Open Jellyfin</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>📺 Media Management</h2>
            <div class="services">
                <div class="service">
                    <h3><span class="status"></span>Prowlarr</h3>
                    <p>Indexer management</p>
                    <a href="http://localhost:9696" target="_blank">Open Prowlarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Sonarr</h3>
                    <p>TV show automation</p>
                    <a href="http://localhost:8989" target="_blank">Open Sonarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Radarr</h3>
                    <p>Movie automation</p>
                    <a href="http://localhost:7878" target="_blank">Open Radarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Lidarr</h3>
                    <p>Music automation</p>
                    <a href="http://localhost:8686" target="_blank">Open Lidarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Readarr</h3>
                    <p>Book automation</p>
                    <a href="http://localhost:8787" target="_blank">Open Readarr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Bazarr</h3>
                    <p>Subtitle management</p>
                    <a href="http://localhost:6767" target="_blank">Open Bazarr</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>⬇️ Download Clients</h2>
            <div class="services">
                <div class="service">
                    <h3><span class="status"></span>qBittorrent</h3>
                    <p>Torrent downloads</p>
                    <a href="http://localhost:8080" target="_blank">Open qBittorrent</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>SABnzbd</h3>
                    <p>Usenet downloads</p>
                    <a href="http://localhost:8081" target="_blank">Open SABnzbd</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>🎯 Request & Analytics</h2>
            <div class="services">
                <div class="service">
                    <h3><span class="status"></span>Overseerr</h3>
                    <p>Request management</p>
                    <a href="http://localhost:5055" target="_blank">Open Overseerr</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Tautulli</h3>
                    <p>Media analytics</p>
                    <a href="http://localhost:8181" target="_blank">Open Tautulli</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>🛠️ Management Tools</h2>
            <div class="services">
                <div class="service">
                    <h3><span class="status"></span>Homepage</h3>
                    <p>Beautiful dashboard</p>
                    <a href="http://localhost:3000" target="_blank">Open Homepage</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Portainer</h3>
                    <p>Docker management</p>
                    <a href="http://localhost:9000" target="_blank">Open Portainer</a>
                </div>
                <div class="service">
                    <h3><span class="status"></span>Uptime Kuma</h3>
                    <p>Service monitoring</p>
                    <a href="http://localhost:3001" target="_blank">Open Uptime Kuma</a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
EOF

# Show completion message
echo ""
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo ""
echo -e "${YELLOW}📋 Quick Access:${NC}"
echo "   Dashboard: file://$(pwd)/media-server-dashboard.html"
echo "   Jellyfin:  http://localhost:8096"
echo "   Homepage:  http://localhost:3000"
echo "   Portainer: http://localhost:9000"
echo ""
echo -e "${YELLOW}📚 Setup Guide:${NC}"
echo "1. Open Prowlarr (http://localhost:9696) and add your indexers"
echo "2. In each *arr app, add Prowlarr as the indexer"
echo "3. Configure download clients in each *arr app"
echo "4. Add media libraries to Jellyfin"
echo "5. Connect Overseerr to Jellyfin and *arr apps"
echo ""
echo -e "${CYAN}💡 Useful Commands:${NC}"
echo "   View logs:        $COMPOSE_CMD logs -f [service-name]"
echo "   Restart service:  $COMPOSE_CMD restart [service-name]"
echo "   Stop all:         $COMPOSE_CMD down"
echo "   Update images:    $COMPOSE_CMD pull"
echo ""

# Open dashboard if on macOS
if [[ "$OSTYPE" == "darwin"* ]] && command -v open &> /dev/null; then
    echo -e "${BLUE}🌐 Opening dashboard in browser...${NC}"
    open "media-server-dashboard.html"
fi

echo -e "${GREEN}🎉 Your Ultimate Media Server 2025 is ready!${NC}"