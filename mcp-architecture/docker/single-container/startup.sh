#!/bin/bash

# Ultimate Media Server 2025 - Single Container Startup Script
# This script initializes all 30 media services and the MCP suite

set -e

echo "🚀 Starting Ultimate Media Server 2025 with ALL 30 services..."

# Set proper permissions
chown -R media:media /opt/media-server
chmod -R 755 /opt/media-server

# Create necessary directories if they don't exist
mkdir -p /opt/media-server/{config,data,logs,downloads,media,scripts}
mkdir -p /opt/media-server/logs
mkdir -p /opt/media-server/data/{jellyfin,plex,emby,tautulli,portainer}
mkdir -p /opt/media-server/downloads/{complete,incomplete,watch}
mkdir -p /opt/media-server/media/{movies,tv,music,books,audiobooks,podcasts,documentaries,anime}

# Initialize service configurations
echo "📋 Initializing service configurations..."

# Create default configs for all services
for service in jellyfin plex emby sonarr radarr lidarr readarr bazarr prowlarr jackett flaresolverr qbittorrent transmission deluge nzbget sabnzbd overseerr requestrr ombi tautulli homepage heimdall organizr homarr nginx-proxy-manager unpackerr; do
    mkdir -p /opt/media-server/config/$service
    touch /opt/media-server/config/$service/.initialized
done

# Set proper ownership after directory creation
chown -R media:media /opt/media-server
chown -R plex:plex /opt/media-server/config/plex || true
chown -R netdata:netdata /var/lib/netdata || true

# Copy configuration templates if they exist
if [ -d "/opt/media-server/service-configs" ]; then
    echo "📁 Copying service configuration templates..."
    cp -r /opt/media-server/service-configs/* /opt/media-server/config/ 2>/dev/null || true
fi

# Initialize environment variables for all services
export JELLYFIN_DATA_DIR="/opt/media-server/config/jellyfin"
export JELLYFIN_CACHE_DIR="/opt/media-server/data/jellyfin/cache"
export JELLYFIN_LOG_DIR="/opt/media-server/logs"
export PLEX_MEDIA_SERVER_APPLICATION_SUPPORT_DIR="/opt/media-server/config/plex"
export CONFIG_DIRECTORY="/opt/media-server/config/overseerr"
export HOMEPAGE_CONFIG_DIR="/opt/media-server/config/homepage"
export UN_CONFIG_FILE="/opt/media-server/config/unpackerr/unpackerr.conf"

# Create MCP Suite environment file
cat > /opt/media-server/mcp-suite/.env << EOF
# Auto-generated environment for single container deployment
NODE_ENV=production
PORT=8090
LOG_LEVEL=info

# Service URLs (internal communication)
JELLYFIN_URL=http://localhost:8096
PLEX_URL=http://localhost:32400
EMBY_URL=http://localhost:8097
SONARR_URL=http://localhost:8989
RADARR_URL=http://localhost:7878
LIDARR_URL=http://localhost:8686
READARR_URL=http://localhost:8787
BAZARR_URL=http://localhost:6767
PROWLARR_URL=http://localhost:9696
JACKETT_URL=http://localhost:9117
FLARESOLVERR_URL=http://localhost:8191
QBITTORRENT_URL=http://localhost:8080
TRANSMISSION_URL=http://localhost:9091
DELUGE_URL=http://localhost:8112
NZBGET_URL=http://localhost:6789
SABNZBD_URL=http://localhost:8085
OVERSEERR_URL=http://localhost:5055
REQUESTRR_URL=http://localhost:4545
OMBI_URL=http://localhost:3579
TAUTULLI_URL=http://localhost:8181
HOMEPAGE_URL=http://localhost:3000
HEIMDALL_URL=http://localhost:7575
ORGANIZR_URL=http://localhost:8081
HOMARR_URL=http://localhost:7576
NGINX_PROXY_MANAGER_URL=http://localhost:81
PORTAINER_URL=http://localhost:9000
NETDATA_URL=http://localhost:19999

# Default credentials (change these!)
JELLYFIN_API_KEY=\${JELLYFIN_API_KEY:-}
PLEX_TOKEN=\${PLEX_TOKEN:-}
SONARR_API_KEY=\${SONARR_API_KEY:-}
RADARR_API_KEY=\${RADARR_API_KEY:-}
LIDARR_API_KEY=\${LIDARR_API_KEY:-}
READARR_API_KEY=\${READARR_API_KEY:-}
BAZARR_API_KEY=\${BAZARR_API_KEY:-}
PROWLARR_API_KEY=\${PROWLARR_API_KEY:-}
QBITTORRENT_USERNAME=admin
QBITTORRENT_PASSWORD=adminadmin
OPENAI_API_KEY=\${OPENAI_API_KEY:-}

# Storage paths
MEDIA_ROOT=/opt/media-server/media
DOWNLOAD_ROOT=/opt/media-server/downloads
CONFIG_ROOT=/opt/media-server/config
DATA_ROOT=/opt/media-server/data
EOF

# Create update script
cat > /opt/media-server/scripts/updater.sh << 'EOF'
#!/bin/bash
# Simple update checker - runs every 24 hours
while true; do
    echo "$(date): Checking for updates..."
    # Add update logic here for services that support it
    sleep 86400  # 24 hours
done
EOF
chmod +x /opt/media-server/scripts/updater.sh

# Create service status checker
cat > /opt/media-server/scripts/healthcheck.sh << 'EOF'
#!/bin/bash
# Health check script for all services

services=(
    "jellyfin:8096"
    "plex:32400" 
    "emby:8097"
    "sonarr:8989"
    "radarr:7878"
    "lidarr:8686"
    "readarr:8787"
    "bazarr:6767"
    "prowlarr:9696"
    "jackett:9117"
    "flaresolverr:8191"
    "qbittorrent:8080"
    "transmission:9091"
    "deluge:8112"
    "nzbget:6789"
    "sabnzbd:8085"
    "overseerr:5055"
    "requestrr:4545"
    "ombi:3579"
    "tautulli:8181"
    "homepage:3000"
    "heimdall:7575"
    "organizr:8081"
    "homarr:7576"
    "nginx-proxy-manager:81"
    "portainer:9000"
    "netdata:19999"
    "mcp-suite:8090"
)

healthy=0
total=${#services[@]}

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -sf http://localhost:$port >/dev/null 2>&1 || \
       curl -sf http://localhost:$port/health >/dev/null 2>&1 || \
       curl -sf http://localhost:$port/api/v1/status >/dev/null 2>&1; then
        echo "✅ $name (port $port) - healthy"
        ((healthy++))
    else
        echo "❌ $name (port $port) - unhealthy"
    fi
done

echo ""
echo "📊 Health Summary: $healthy/$total services healthy"
echo "📊 Health Percentage: $(( healthy * 100 / total ))%"
EOF
chmod +x /opt/media-server/scripts/healthcheck.sh

# Initialize cron for periodic tasks
echo "⏰ Setting up cron jobs..."
cat > /opt/media-server/crontab << EOF
# Health check every 5 minutes
*/5 * * * * /opt/media-server/scripts/healthcheck.sh >> /opt/media-server/logs/healthcheck.log 2>&1

# Cleanup logs weekly
0 0 * * 0 find /opt/media-server/logs -name "*.log" -mtime +7 -delete

# Restart services weekly (optional)
# 0 3 * * 0 supervisorctl restart all
EOF

crontab /opt/media-server/crontab

echo "🔧 Starting services with supervisor..."

# Start supervisor to manage all services
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf