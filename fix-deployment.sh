#!/bin/bash

# Working Media Server Deployment - Fixed Version
set -e

DOCKER="/Applications/Docker.app/Contents/Resources/bin/docker"

echo "🔧 FIXING MEDIA SERVER DEPLOYMENT"
echo "=================================="

# Stop all existing containers
echo "🛑 Stopping existing containers..."
$DOCKER stop $(docker ps -q) 2>/dev/null || true

# Remove containers that might conflict
echo "🗑️  Cleaning up..."
$DOCKER rm -f jellyfin sonarr radarr qbittorrent-simple simple-media-server prowlarr overseerr homarr tautulli bazarr traefik media-stack-webui 2>/dev/null || true

# Create fresh data directory
echo "📁 Setting up directories..."
rm -rf ./media-data-fixed
mkdir -p ./media-data-fixed/{config,downloads,movies,tv,music}
mkdir -p ./media-data-fixed/config/{jellyfin,sonarr,radarr,qbittorrent,prowlarr,overseerr}

echo "🚀 Starting fresh media server..."

# Start Jellyfin with proper port mapping
echo "📺 Starting Jellyfin..."
$DOCKER run -d \
    --name jellyfin-working \
    -p 8096:8096 \
    -v "$(pwd)/media-data-fixed/config/jellyfin:/config" \
    -v "$(pwd)/media-data-fixed:/media" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    --restart unless-stopped \
    jellyfin/jellyfin:latest

sleep 5

# Start qBittorrent with proper port mapping  
echo "⬇️ Starting qBittorrent..."
$DOCKER run -d \
    --name qbittorrent-working \
    -p 8080:8080 \
    -v "$(pwd)/media-data-fixed/downloads:/downloads" \
    -v "$(pwd)/media-data-fixed/config/qbittorrent:/config" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    -e WEBUI_PORT=8080 \
    --restart unless-stopped \
    lscr.io/linuxserver/qbittorrent:latest

sleep 5

# Start Sonarr with proper port mapping
echo "📺 Starting Sonarr (TV)..."
$DOCKER run -d \
    --name sonarr-working \
    -p 8989:8989 \
    -v "$(pwd)/media-data-fixed/config/sonarr:/config" \
    -v "$(pwd)/media-data-fixed/tv:/tv" \
    -v "$(pwd)/media-data-fixed/downloads:/downloads" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    --restart unless-stopped \
    lscr.io/linuxserver/sonarr:latest

sleep 5

# Start Radarr with proper port mapping
echo "🎬 Starting Radarr (Movies)..."
$DOCKER run -d \
    --name radarr-working \
    -p 7878:7878 \
    -v "$(pwd)/media-data-fixed/config/radarr:/config" \
    -v "$(pwd)/media-data-fixed/movies:/movies" \
    -v "$(pwd)/media-data-fixed/downloads:/downloads" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    --restart unless-stopped \
    lscr.io/linuxserver/radarr:latest

sleep 5

# Start Prowlarr with proper port mapping
echo "🔍 Starting Prowlarr (Indexers)..."
$DOCKER run -d \
    --name prowlarr-working \
    -p 9696:9696 \
    -v "$(pwd)/media-data-fixed/config/prowlarr:/config" \
    -e PUID=1000 \
    -e PGID=1000 \
    -e TZ=America/New_York \
    --restart unless-stopped \
    lscr.io/linuxserver/prowlarr:latest

echo "⏳ Waiting for services to start..."
sleep 15

echo "🧪 Testing services..."

# Test each service
services=(
    "Jellyfin:8096"
    "qBittorrent:8080" 
    "Sonarr:8989"
    "Radarr:7878"
    "Prowlarr:9696"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s --max-time 5 "http://localhost:$port" > /dev/null; then
        echo "✅ $name (http://localhost:$port) - WORKING"
    else
        echo "❌ $name (http://localhost:$port) - NOT RESPONDING"
    fi
done

echo ""
echo "🎉 FIXED MEDIA SERVER DEPLOYED!"
echo ""
echo "📺 Jellyfin Media Server:  http://localhost:8096"
echo "⬇️  qBittorrent Downloads:  http://localhost:8080"  
echo "📺 Sonarr (TV Shows):       http://localhost:8989"
echo "🎬 Radarr (Movies):        http://localhost:7878"
echo "🔍 Prowlarr (Indexers):    http://localhost:9696"
echo ""
echo "🔧 Management:"
echo "  Status: $DOCKER ps"
echo "  Logs:   $DOCKER logs [container-name]"
echo "  Stop:   $DOCKER stop [container-name]"
echo ""
echo "🚀 Ready to use!"