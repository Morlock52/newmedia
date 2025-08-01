#!/bin/bash

# macOS-Optimized Media Server Deployment Script
# Addresses macOS-specific Docker limitations and performance issues

set -e

echo "🍎 Starting macOS-Optimized Media Server Deployment..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose-macos-optimized.yml down --remove-orphans 2>/dev/null || true

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{media/{movies,tv,music,books,audiobooks,podcasts},downloads/{movies,tv,music,books},usenet,torrents}
mkdir -p config/{prometheus,homepage}

# Set proper permissions (macOS specific)
echo "🔐 Setting permissions..."
sudo chown -R $(id -u):$(id -g) data/ config/ 2>/dev/null || echo "Note: Some permission changes may require sudo access"

# Create basic Prometheus config if it doesn't exist
if [ ! -f config/prometheus/prometheus.yml ]; then
    echo "📊 Creating Prometheus configuration..."
    cat > config/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik:8080']
    metrics_path: /metrics
EOF
fi

# Create basic Homepage config if it doesn't exist
if [ ! -f config/homepage/services.yaml ]; then
    echo "🏠 Creating Homepage configuration..."
    mkdir -p config/homepage
    cat > config/homepage/services.yaml << 'EOF'
- Media:
    - Jellyfin:
        href: http://localhost:8096
        description: Media streaming server
        icon: jellyfin.png
    - AudioBookshelf:
        href: http://localhost:13378
        description: Audiobooks & Podcasts
        icon: audiobookshelf.png
    - Navidrome:
        href: http://localhost:4533
        description: Music streaming
        icon: navidrome.png

- Management:
    - Radarr:
        href: http://localhost:7878
        description: Movie management
        icon: radarr.png
    - Sonarr:
        href: http://localhost:8989
        description: TV show management
        icon: sonarr.png
    - Prowlarr:
        href: http://localhost:9696
        description: Indexer management
        icon: prowlarr.png

- Download:
    - qBittorrent:
        href: http://localhost:8081
        description: Torrent client
        icon: qbittorrent.png
    - SABnzbd:
        href: http://localhost:8082
        description: Usenet downloader
        icon: sabnzbd.png

- Monitoring:
    - Grafana:
        href: http://localhost:3001
        description: Dashboards
        icon: grafana.png
    - Prometheus:
        href: http://localhost:9090
        description: Metrics
        icon: prometheus.png
EOF

    cat > config/homepage/settings.yaml << 'EOF'
title: Media Server Dashboard
theme: dark
background: https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3
color: slate
layout:
  Media:
    style: row
    columns: 3
  Management:
    style: row
    columns: 3
  Download:
    style: row
    columns: 2
  Monitoring:
    style: row
    columns: 2
EOF
fi

# Deploy the stack
echo "🚀 Deploying macOS-optimized media server stack..."
docker-compose -f docker-compose-macos-optimized.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service status..."
docker-compose -f docker-compose-macos-optimized.yml ps

echo ""
echo "✅ Deployment completed!"
echo ""
echo "🌐 Access your services:"
echo "  Dashboard:     http://localhost:3000"
echo "  Jellyfin:      http://localhost:8096"
echo "  AudioBooks:    http://localhost:13378"
echo "  Music Stream:  http://localhost:4533"
echo "  Movies:        http://localhost:7878"
echo "  TV Shows:      http://localhost:8989"
echo "  Downloads:     http://localhost:8081 (Torrents), http://localhost:8082 (Usenet)"
echo "  Monitoring:    http://localhost:3001 (Grafana), http://localhost:9090 (Prometheus)"
echo "  Docker:        http://localhost:9000"
echo ""
echo "🔧 Configuration Tips:"
echo "  1. Configure download clients to use /downloads as download directory"
echo "  2. Set media libraries to use /media subdirectories"
echo "  3. Use hardlinks in Radarr/Sonarr for efficient storage"
echo "  4. For VPN protection, consider using a separate VPN client on macOS"
echo ""
echo "⚠️  macOS Specific Notes:"
echo "  - Hardware acceleration is not available in Docker on macOS"
echo "  - VPN containers are disabled due to routing issues"
echo "  - For best performance, consider running Jellyfin natively"
echo ""