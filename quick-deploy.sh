#!/bin/bash

# Quick deployment script for media server
# Usage: ./quick-deploy.sh

echo "🚀 Quick Media Server Deploy"
echo "=============================="

# Detect Docker Compose command
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose not found!"
    echo "💡 Run: ./install-docker.sh to install Docker"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "💡 Please start Docker Desktop and try again"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found!"
    exit 1
fi

# Create directories if they don't exist
echo "📁 Creating directories..."
mkdir -p config/{jellyfin,sonarr,radarr,qbittorrent,prowlarr,overseerr,tautulli,homarr}
mkdir -p data/media/{movies,tv,music}
mkdir -p data/torrents/{movies,tv,music}

# Deploy the stack
echo "🐳 Deploying Docker stack..."
$COMPOSE_CMD up -d

# Check status
echo "📊 Checking status..."
$COMPOSE_CMD ps

echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your services:"
echo "   Jellyfin:    http://localhost:8096"
echo "   Sonarr:      http://localhost:8989"
echo "   Radarr:      http://localhost:7878"
echo "   qBittorrent: http://localhost:8080"
echo "   Prowlarr:    http://localhost:9696"
echo "   Overseerr:   http://localhost:5055"
echo "   Tautulli:    http://localhost:8181"
echo "   Homarr:      http://localhost:7575"
echo ""
echo "📖 Check README.md for setup instructions"
echo "🔧 Run '$COMPOSE_CMD logs -f [service]' to view logs"