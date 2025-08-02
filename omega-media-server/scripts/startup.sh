#!/bin/bash
# Omega Media Server - Startup Script

set -e

echo "============================================"
echo "     Omega Media Server 2025 Starting       "
echo "============================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if this is first run
if [ ! -f "${OMEGA_CONFIG}/.initialized" ]; then
    log_info "First run detected. Starting initial setup..."
    
    # Create necessary directories
    mkdir -p ${OMEGA_CONFIG}/{nginx,ssl,apps,backups,logs}
    mkdir -p ${OMEGA_MEDIA}/{movies,tv,music,photos,books,downloads}
    mkdir -p /transcode
    
    # Generate default configuration
    cp -n ${OMEGA_HOME}/config/default.json ${OMEGA_CONFIG}/config.json
    
    # Generate SSL certificates
    if [ "${SSL_ENABLE}" = "true" ] && [ ! -f "${OMEGA_CONFIG}/ssl/cert.pem" ]; then
        log_info "Generating self-signed SSL certificate..."
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ${OMEGA_CONFIG}/ssl/key.pem \
            -out ${OMEGA_CONFIG}/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Omega/CN=${DOMAIN}"
    fi
    
    # Initialize database
    log_info "Initializing database..."
    cd ${OMEGA_HOME}
    npm run db:init
    
    # Mark as initialized
    touch ${OMEGA_CONFIG}/.initialized
    
    log_info "Initial setup complete!"
fi

# Fix permissions
log_info "Setting permissions..."
chown -R ${PUID}:${PGID} ${OMEGA_CONFIG} ${OMEGA_MEDIA} /transcode

# Start Docker daemon (for Docker-in-Docker)
if [ "${ENABLE_DOCKER_IN_DOCKER}" = "true" ]; then
    log_info "Starting Docker daemon..."
    dockerd &
    sleep 5
fi

# Start K3s if enabled
if [ "${ENABLE_K3S}" = "true" ]; then
    log_info "Starting K3s..."
    k3s server --data-dir /var/lib/k3s &
    sleep 10
fi

# Configure hardware acceleration
if [ -e /dev/dri ]; then
    log_info "Hardware acceleration detected"
    export LIBVA_DRIVER_NAME=iHD
    export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
fi

# Start PostgreSQL
if [ "${USE_INTERNAL_DB}" = "true" ]; then
    log_info "Starting PostgreSQL..."
    su-exec postgres postgres -D /var/lib/postgresql/data &
    sleep 5
fi

# Start Redis
if [ "${USE_INTERNAL_CACHE}" = "true" ]; then
    log_info "Starting Redis..."
    redis-server --requirepass ${REDIS_PASSWORD} &
    sleep 2
fi

# Start Nginx
log_info "Starting Nginx..."
nginx -g "daemon off;" &

# Load AI models if enabled
if [ "${ENABLE_AI}" = "true" ]; then
    log_info "Loading AI models..."
    cd ${OMEGA_HOME}
    python3 scripts/load_models.py
fi

# Start application services
log_info "Starting application services..."

# Start core services
if [ "${ENABLE_JELLYFIN}" = "true" ]; then
    log_info "Starting Jellyfin..."
    cd ${OMEGA_APPS}/jellyfin
    ./jellyfin --datadir ${OMEGA_CONFIG}/jellyfin --configdir ${OMEGA_CONFIG}/jellyfin/config &
fi

if [ "${ENABLE_PLEX}" = "true" ]; then
    log_info "Starting Plex..."
    export LD_LIBRARY_PATH=${OMEGA_APPS}/plex/usr/lib/plexmediaserver
    ${OMEGA_APPS}/plex/usr/lib/plexmediaserver/Plex\ Media\ Server &
fi

# Start *arr services
for service in radarr sonarr lidarr readarr prowlarr bazarr; do
    if [ "${ENABLE_${service^^}}" != "false" ]; then
        log_info "Starting ${service^}..."
        cd ${OMEGA_APPS}/${service}
        ./${service^} -data=${OMEGA_CONFIG}/${service} &
    fi
done

# Start download clients
if [ "${ENABLE_QBITTORRENT}" != "false" ]; then
    log_info "Starting qBittorrent..."
    qbittorrent-nox --webui-port=8080 --profile=${OMEGA_CONFIG}/qbittorrent &
fi

# Start additional services
if [ "${ENABLE_PHOTOPRISM}" = "true" ]; then
    log_info "Starting PhotoPrism..."
    cd ${OMEGA_APPS}
    ./photoprism --config-path ${OMEGA_CONFIG}/photoprism start &
fi

if [ "${ENABLE_NAVIDROME}" = "true" ]; then
    log_info "Starting Navidrome..."
    cd ${OMEGA_APPS}/navidrome
    ND_CONFIGFILE=${OMEGA_CONFIG}/navidrome/config.toml ./navidrome &
fi

# Start VPN if configured
if [ "${ENABLE_VPN}" = "true" ] && [ -n "${VPN_ENDPOINT}" ]; then
    log_info "Starting WireGuard VPN..."
    wg-quick up ${OMEGA_CONFIG}/wireguard/wg0.conf
fi

# Start the main Omega application
log_info "Starting Omega Media Server..."
cd ${OMEGA_HOME}
exec node src/index.js