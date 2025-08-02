#!/usr/bin/env bash
set -euo pipefail

echo "📁 Creating Complete Data Directory Structure"
echo "============================================="

# Create all required data directories
echo "Creating media directories..."

# Main media directories
mkdir -p data/media/{movies,tv,music,books,comics,podcasts,online-videos,photos}

# Download directories
mkdir -p data/torrents/{movies,tv,music,books,comics,software}
mkdir -p data/usenet/{movies,tv,music,books,comics}

# Working directories
mkdir -p data/downloads/{complete,incomplete,watch}

echo "✅ Media directories created"

# Create config directories for all services
echo "Creating config directories..."

services=(
    "jellyfin" "sonarr" "radarr" "lidarr" "readarr" "bazarr" "prowlarr" 
    "overseerr" "qbittorrent" "tautulli" "mylar" "podgrab" "youtube-dl-material"
    "photoprism" "traefik" "gluetun"
)

for service in "${services[@]}"; do
    mkdir -p "config/$service"
    echo "  ✅ config/$service"
done

echo ""
echo "Setting permissions..."
# Set permissions (ignore errors on macOS)
chown -R 1000:1000 data config 2>/dev/null || true
chmod -R 755 data config 2>/dev/null || true

echo ""
echo "📊 Directory Structure Created:"
echo "=============================="

echo ""
echo "📺 Media Library:"
echo "  data/media/movies/     - Movie files"
echo "  data/media/tv/         - TV show files"
echo "  data/media/music/      - Music files"
echo "  data/media/books/      - Ebook files"
echo "  data/media/comics/     - Comic files"
echo "  data/media/podcasts/   - Podcast files"
echo "  data/media/online-videos/ - Downloaded videos"
echo "  data/media/photos/     - Photo library"

echo ""
echo "⬇️  Download Areas:"
echo "  data/torrents/         - Torrent downloads"
echo "  data/usenet/           - Usenet downloads"
echo "  data/downloads/        - Working download area"

echo ""
echo "⚙️  Service Configs:"
for service in "${services[@]}"; do
    echo "  config/$service/       - $service configuration"
done

echo ""
echo "🎉 Directory structure ready for complete media stack!"
