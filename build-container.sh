#!/bin/bash

# Build and run the containerized media server
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}
╔══════════════════════════════════════╗
║    Containerized Media Server        ║
║         Docker Build & Run           ║
╚══════════════════════════════════════╝${NC}"
}

print_header

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    echo "💡 Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon not running. Please start Docker Desktop."
    exit 1
fi

print_status "Docker is ready ✓"

# Build the container
print_status "Building media server container..."
docker build -t media-server:latest .

# Create volumes for persistent data
print_status "Creating Docker volumes..."
docker volume create media-config 2>/dev/null || true
docker volume create media-data 2>/dev/null || true

# Stop existing container if running
print_status "Stopping existing container..."
docker stop media-server 2>/dev/null || true
docker rm media-server 2>/dev/null || true

# Run the container
print_status "Starting containerized media server..."
docker run -d \
    --name media-server \
    --privileged \
    -p 8096:8096 \
    -p 8989:8989 \
    -p 7878:7878 \
    -p 8080:8080 \
    -p 9696:9696 \
    -p 5055:5055 \
    -p 8181:8181 \
    -p 7575:7575 \
    -p 80:80 \
    -v media-config:/media/config \
    -v media-data:/media/data \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --restart unless-stopped \
    media-server:latest

print_status "Container started! ✓"

# Wait for services to start
print_status "Waiting for services to initialize..."
sleep 30

# Check status
print_status "Checking container status..."
docker ps --filter name=media-server

echo -e "${GREEN}
🎉 Containerized Media Server Deployed!

📊 Service URLs:
├── 🎬 Jellyfin:      http://localhost:8096
├── 📺 Sonarr:        http://localhost:8989
├── 🎬 Radarr:        http://localhost:7878
├── ⬇️  qBittorrent:   http://localhost:8080
├── 🔍 Prowlarr:      http://localhost:9696
├── 📋 Overseerr:     http://localhost:5055
├── 📈 Tautulli:      http://localhost:8181
├── 🏠 Homarr:        http://localhost:7575
└── 🌐 Web Interface: http://localhost:80

🔧 Management Commands:
├── View logs:     docker logs -f media-server
├── Shell access:  docker exec -it media-server bash
├── Stop server:   docker stop media-server
├── Start server:  docker start media-server
├── Restart:       docker restart media-server
└── Remove:        docker rm -f media-server

💾 Data Persistence:
├── Configurations: media-config volume
└── Media files:     media-data volume

⚡ Everything runs in a single container with Docker-in-Docker!
${NC}"

print_status "Deployment complete! 🚀"