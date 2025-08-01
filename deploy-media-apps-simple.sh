#!/bin/bash

# Simple Media Apps Deployment for macOS
# Starts core media applications without complex features

echo "🎬 Starting Media Server Apps..."
echo "=================================="

# Stop any existing containers first
docker stop jellyfin sonarr radarr prowlarr overseerr qbittorrent 2>/dev/null || true
docker rm jellyfin sonarr radarr prowlarr overseerr qbittorrent 2>/dev/null || true

# Create required directories
mkdir -p data/{downloads,media/{movies,tv,music},config}
mkdir -p config/{jellyfin,sonarr,radarr,prowlarr,overseerr,qbittorrent}

# Set up environment
export PUID=$(id -u)
export PGID=$(id -g)
export TZ="America/New_York"
export CONFIG_PATH="$(pwd)/config"
export DATA_PATH="$(pwd)/data"

echo "🔧 Configuration:"
echo "   - PUID: $PUID"
echo "   - PGID: $PGID" 
echo "   - CONFIG: $CONFIG_PATH"
echo "   - DATA: $DATA_PATH"
echo ""

echo "🚀 Starting Jellyfin (Media Server)..."
docker run -d \
  --name jellyfin \
  --restart unless-stopped \
  -p 8096:8096 \
  -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
  -v $CONFIG_PATH/jellyfin:/config \
  -v $DATA_PATH/media:/data/media \
  jellyfin/jellyfin:latest

echo "🔍 Starting Prowlarr (Indexer Manager)..."
docker run -d \
  --name prowlarr \
  --restart unless-stopped \
  -p 9696:9696 \
  -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
  -v $CONFIG_PATH/prowlarr:/config \
  linuxserver/prowlarr:latest

echo "📺 Starting Sonarr (TV Shows)..."
docker run -d \
  --name sonarr \
  --restart unless-stopped \
  -p 8989:8989 \
  -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
  -v $CONFIG_PATH/sonarr:/config \
  -v $DATA_PATH/media/tv:/tv \
  -v $DATA_PATH/downloads:/downloads \
  linuxserver/sonarr:latest

echo "🎬 Starting Radarr (Movies)..."
docker run -d \
  --name radarr \
  --restart unless-stopped \
  -p 7878:7878 \
  -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
  -v $CONFIG_PATH/radarr:/config \
  -v $DATA_PATH/media/movies:/movies \
  -v $DATA_PATH/downloads:/downloads \
  linuxserver/radarr:latest

echo "🎭 Starting Overseerr (Request Manager)..."
docker run -d \
  --name overseerr \
  --restart unless-stopped \
  -p 5055:5055 \
  -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
  -v $CONFIG_PATH/overseerr:/app/config \
  sctx/overseerr:latest

echo "⬇️ Starting qBittorrent (Download Client)..."
docker run -d \
  --name qbittorrent \
  --restart unless-stopped \
  -p 8080:8080 \
  -p 6881:6881 -p 6881:6881/udp \
  -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
  -v $CONFIG_PATH/qbittorrent:/config \
  -v $DATA_PATH/downloads:/downloads \
  linuxserver/qbittorrent:latest

# Wait for containers to start
echo ""
echo "⏳ Waiting for containers to start..."
sleep 10

# Check status
echo ""
echo "📊 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(jellyfin|sonarr|radarr|prowlarr|overseerr|qbittorrent|NAMES)"

echo ""
echo "🌐 Access URLs:"
echo "   📺 Jellyfin:    http://localhost:8096"
echo "   🔍 Prowlarr:    http://localhost:9696"
echo "   📺 Sonarr:      http://localhost:8989"
echo "   🎬 Radarr:      http://localhost:7878"
echo "   🎭 Overseerr:   http://localhost:5055"
echo "   ⬇️  qBittorrent: http://localhost:8080"
echo ""
echo "✅ Media server apps are starting up!"
echo "   Give them 1-2 minutes to fully initialize."
echo ""
echo "🔐 Default Credentials:"
echo "   qBittorrent: admin / adminadmin (change on first login)"
echo ""
echo "📊 Check status: docker ps"
echo "📜 View logs: docker logs [container-name]"