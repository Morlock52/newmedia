#!/bin/bash

# Simple all-in-one media server start script
set -e

echo "🚀 Starting All-in-One Media Server"
echo "===================================="

# Start Jellyfin in background
echo "📺 Starting Jellyfin..."
/init &

# Start qBittorrent
echo "⬇️ Starting qBittorrent..."
qbittorrent-nox --daemon --webui-port=8080

# Keep container running
echo "✅ All services started!"
echo "📊 Monitoring services..."

# Monitor and keep alive
while true; do
    sleep 60
    echo "$(date): Services running..."
done