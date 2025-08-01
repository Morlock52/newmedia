#!/bin/bash

# Simple media server deployment using full Docker path
set -e

# Docker path
DOCKER="/Applications/Docker.app/Contents/Resources/bin/docker"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}
╔══════════════════════════════════════╗
║        Simple Media Container        ║
║       Jellyfin + qBittorrent         ║
╚══════════════════════════════════════╝${NC}"
}

print_header

# Check Docker
if ! $DOCKER info &> /dev/null 2>&1; then
    echo "❌ Docker not running. Please start Docker Desktop."
    exit 1
fi

print_status "Docker is ready ✓"

# Create directories for data persistence
print_status "Creating data directories..."
mkdir -p ./media-data/{config,downloads,movies,tv,music,qbt-config}

# Stop any existing containers
print_status "Cleaning up existing containers..."
$DOCKER stop simple-media-server 2>/dev/null || true
$DOCKER rm simple-media-server 2>/dev/null || true
$DOCKER stop qbittorrent-simple 2>/dev/null || true
$DOCKER rm qbittorrent-simple 2>/dev/null || true

# Run Jellyfin container
print_status "Starting Jellyfin media server..."

$DOCKER run -d \
    --name simple-media-server \
    -p 8096:8096 \
    -v "$(pwd)/media-data/config:/config" \
    -v "$(pwd)/media-data:/media" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    --restart unless-stopped \
    jellyfin/jellyfin:latest

# Wait a moment
sleep 5

# Run qBittorrent container
print_status "Starting qBittorrent download client..."

$DOCKER run -d \
    --name qbittorrent-simple \
    -p 8080:8080 \
    -v "$(pwd)/media-data/downloads:/downloads" \
    -v "$(pwd)/media-data/qbt-config:/config" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    -e WEBUI_PORT=8080 \
    --restart unless-stopped \
    lscr.io/linuxserver/qbittorrent:latest

print_status "Containers started! Waiting for services to initialize..."
sleep 10

# Show status
echo ""
print_status "Container status:"
$DOCKER ps --filter "name=simple-media-server" --filter "name=qbittorrent-simple"

echo -e "${GREEN}
🎉 Simple Media Server Deployed Successfully!

📺 Jellyfin Media Server:  http://localhost:8096
⬇️  qBittorrent Downloads:  http://localhost:8080

📁 Data stored in: $(pwd)/media-data/

🔧 Management Commands:
├── Stop:    $DOCKER stop simple-media-server qbittorrent-simple
├── Start:   $DOCKER start simple-media-server qbittorrent-simple  
├── Logs:    $DOCKER logs simple-media-server
├── Status:  $DOCKER ps
└── Remove:  $DOCKER rm -f simple-media-server qbittorrent-simple

💡 Setup Instructions:
1. Configure Jellyfin at http://localhost:8096
2. Add media libraries pointing to /media/movies and /media/tv
3. Use qBittorrent at http://localhost:8080 for downloads
4. Downloaded files will appear in media-data/downloads/
${NC}"

print_status "Deployment complete! 🚀"

# Test connectivity
print_status "Testing service connectivity..."
sleep 5

if curl -s http://localhost:8096 > /dev/null; then
    print_status "✅ Jellyfin is responding at http://localhost:8096"
else
    echo "⚠️  Jellyfin may still be starting up..."
fi

if curl -s http://localhost:8080 > /dev/null; then
    print_status "✅ qBittorrent is responding at http://localhost:8080"  
else
    echo "⚠️  qBittorrent may still be starting up..."
fi

print_status "All services deployed! 🎬"