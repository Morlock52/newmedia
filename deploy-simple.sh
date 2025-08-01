#!/bin/bash

# Ultra-simple containerized media server
# Just run Jellyfin + qBittorrent in a single container

set -e

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
if ! docker info &> /dev/null 2>&1; then
    echo "❌ Docker not running. Please start Docker Desktop."
    exit 1
fi

print_status "Creating simple media server..."

# Create directories for data persistence
mkdir -p ./media-data/{config,downloads,movies,tv,music}

# Run Jellyfin container with qBittorrent
print_status "Starting Jellyfin + qBittorrent container..."

docker run -d \
    --name simple-media-server \
    -p 8096:8096 \
    -p 8080:8080 \
    -v "$(pwd)/media-data/config:/config" \
    -v "$(pwd)/media-data:/media" \
    -v "$(pwd)/media-data/downloads:/downloads" \
    --restart unless-stopped \
    jellyfin/jellyfin:latest

# Wait a moment
sleep 5

# Run qBittorrent in the same network
print_status "Adding qBittorrent container..."

docker run -d \
    --name qbittorrent-simple \
    -p 8080:8080 \
    -v "$(pwd)/media-data/downloads:/downloads" \
    -v "$(pwd)/media-data/qbt-config:/config" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    --restart unless-stopped \
    lscr.io/linuxserver/qbittorrent:latest

print_status "Containers started!"

# Show status
echo ""
docker ps --filter "name=simple-media-server"
docker ps --filter "name=qbittorrent-simple"

echo -e "${GREEN}
🎉 Simple Media Server Running!

📺 Jellyfin Media Server:  http://localhost:8096
⬇️  qBittorrent Downloads:  http://localhost:8080

📁 Data stored in: $(pwd)/media-data/

🔧 Commands:
├── Stop:    docker stop simple-media-server qbittorrent-simple
├── Start:   docker start simple-media-server qbittorrent-simple  
├── Logs:    docker logs simple-media-server
└── Remove:  docker rm -f simple-media-server qbittorrent-simple

💡 Setup:
1. Configure Jellyfin at http://localhost:8096
2. Add media libraries pointing to /media/movies and /media/tv
3. Use qBittorrent at http://localhost:8080 for downloads
4. Downloaded files will appear in media-data/downloads/
${NC}"

print_status "Deployment complete! 🚀"