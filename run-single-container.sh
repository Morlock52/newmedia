#!/bin/bash

# Single Container Media Server Runner
# Checks for port conflicts and runs with available ports

set -e

echo "🚀 Single Container Media Server - Smart Port Runner"
echo "==================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to find next available port
find_free_port() {
    local start_port=$1
    local port=$start_port
    while check_port $port; do
        ((port++))
    done
    echo $port
}

echo "🔍 Checking for port conflicts..."

# Default ports
PORT_CADDY=80
PORT_JELLYFIN=8096
PORT_SONARR=8989
PORT_RADARR=7878
PORT_PROWLARR=9696
PORT_QBITTORRENT=8080
PORT_HOMEPAGE=3000

# Check and find alternative ports if needed
if check_port $PORT_CADDY; then
    echo -e "${YELLOW}⚠️  Port $PORT_CADDY is in use${NC}"
    PORT_CADDY=$(find_free_port 8000)
    echo -e "${GREEN}✅ Using alternative port: $PORT_CADDY${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_CADDY is available${NC}"
fi

if check_port $PORT_JELLYFIN; then
    echo -e "${YELLOW}⚠️  Port $PORT_JELLYFIN is in use${NC}"
    PORT_JELLYFIN=$(find_free_port 8097)
    echo -e "${GREEN}✅ Using alternative port: $PORT_JELLYFIN${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_JELLYFIN is available${NC}"
fi

if check_port $PORT_SONARR; then
    echo -e "${YELLOW}⚠️  Port $PORT_SONARR is in use${NC}"
    PORT_SONARR=$(find_free_port 8990)
    echo -e "${GREEN}✅ Using alternative port: $PORT_SONARR${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_SONARR is available${NC}"
fi

if check_port $PORT_RADARR; then
    echo -e "${YELLOW}⚠️  Port $PORT_RADARR is in use${NC}"
    PORT_RADARR=$(find_free_port 7879)
    echo -e "${GREEN}✅ Using alternative port: $PORT_RADARR${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_RADARR is available${NC}"
fi

if check_port $PORT_PROWLARR; then
    echo -e "${YELLOW}⚠️  Port $PORT_PROWLARR is in use${NC}"
    PORT_PROWLARR=$(find_free_port 9697)
    echo -e "${GREEN}✅ Using alternative port: $PORT_PROWLARR${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_PROWLARR is available${NC}"
fi

if check_port $PORT_QBITTORRENT; then
    echo -e "${YELLOW}⚠️  Port $PORT_QBITTORRENT is in use${NC}"
    PORT_QBITTORRENT=$(find_free_port 8081)
    echo -e "${GREEN}✅ Using alternative port: $PORT_QBITTORRENT${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_QBITTORRENT is available${NC}"
fi

if check_port $PORT_HOMEPAGE; then
    echo -e "${YELLOW}⚠️  Port $PORT_HOMEPAGE is in use${NC}"
    PORT_HOMEPAGE=$(find_free_port 3001)
    echo -e "${GREEN}✅ Using alternative port: $PORT_HOMEPAGE${NC}"
else
    echo -e "${GREEN}✅ Port $PORT_HOMEPAGE is available${NC}"
fi

echo ""
echo "📋 Port Configuration:"
echo "   Caddy (Main): $PORT_CADDY"
echo "   Jellyfin: $PORT_JELLYFIN"
echo "   Sonarr: $PORT_SONARR"
echo "   Radarr: $PORT_RADARR"
echo "   Prowlarr: $PORT_PROWLARR"
echo "   qBittorrent: $PORT_QBITTORRENT"
echo "   Homepage: $PORT_HOMEPAGE"
echo ""

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^media-server$"; then
    echo -e "${YELLOW}⚠️  Container 'media-server' already exists${NC}"
    read -p "Remove existing container? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing container..."
        docker rm -f media-server
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p config data/media data/downloads

# Check if image exists
if ! docker images | grep -q "media-server-aio"; then
    echo ""
    echo -e "${YELLOW}⚠️  Docker image 'media-server-aio' not found${NC}"
    echo "Building image..."
    # Try multi-service first, fall back to simple if it fails
    if ! docker build -t media-server-aio -f Dockerfile.multi-service . 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Multi-service build failed, trying simple version...${NC}"
        docker build -t media-server-aio -f Dockerfile.simple .
    fi
fi

# Run the container
echo ""
echo "🚀 Starting media server container..."
docker run -d \
  --name media-server \
  -p ${PORT_CADDY}:80 \
  -p ${PORT_JELLYFIN}:8096 \
  -p ${PORT_SONARR}:8989 \
  -p ${PORT_RADARR}:7878 \
  -p ${PORT_PROWLARR}:9696 \
  -p ${PORT_QBITTORRENT}:8080 \
  -p ${PORT_HOMEPAGE}:3000 \
  -v $(pwd)/config:/config \
  -v $(pwd)/data:/data \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  -e TZ=$(cat /etc/timezone 2>/dev/null || echo "America/New_York") \
  --restart unless-stopped \
  media-server-aio

echo ""
echo -e "${GREEN}✅ Container started successfully!${NC}"
echo ""
echo "📺 Access your services at:"
echo "   Dashboard: http://localhost:${PORT_CADDY}/ or http://localhost:${PORT_HOMEPAGE}"
echo "   Jellyfin: http://localhost:${PORT_JELLYFIN}"
echo "   Sonarr: http://localhost:${PORT_SONARR}"
echo "   Radarr: http://localhost:${PORT_RADARR}"
echo "   Prowlarr: http://localhost:${PORT_PROWLARR}"
echo "   qBittorrent: http://localhost:${PORT_QBITTORRENT}"
echo ""
echo "⏳ Note: Services may take 2-3 minutes to fully start"
echo ""
echo "📝 Default credentials:"
echo "   qBittorrent: admin/adminadmin (change immediately!)"
echo "   Other services: Set up on first access"
echo ""
echo "🔍 To view logs: docker logs -f media-server"
echo "🛑 To stop: docker stop media-server"
echo ""

# Save port configuration
cat > port-config.txt << EOF
# Media Server Port Configuration
# Generated on: $(date)

CADDY_PORT=$PORT_CADDY
JELLYFIN_PORT=$PORT_JELLYFIN
SONARR_PORT=$PORT_SONARR
RADARR_PORT=$PORT_RADARR
PROWLARR_PORT=$PORT_PROWLARR
QBITTORRENT_PORT=$PORT_QBITTORRENT
HOMEPAGE_PORT=$PORT_HOMEPAGE

# Access URLs:
Dashboard: http://localhost:${PORT_CADDY}/ or http://localhost:${PORT_HOMEPAGE}
Jellyfin: http://localhost:${PORT_JELLYFIN}
Sonarr: http://localhost:${PORT_SONARR}
Radarr: http://localhost:${PORT_RADARR}
Prowlarr: http://localhost:${PORT_PROWLARR}
qBittorrent: http://localhost:${PORT_QBITTORRENT}
EOF

echo "💾 Port configuration saved to: port-config.txt"