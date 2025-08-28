#!/bin/bash
# s6-overlay v3 Service Structure Generator for Single Container Media Server
# Creates the complete service hierarchy with proper dependencies

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏗️  Creating s6-overlay service structure...${NC}"

# Base directory for s6-overlay services
S6_BASE="/etc/s6-overlay/s6-rc.d"

# Create base directory structure
mkdir -p "${S6_BASE}"

# Function to create a service with dependencies
create_service() {
    local service_name="$1"
    local service_type="${2:-longrun}"
    local dependencies="${3:-}"
    
    echo -e "${GREEN}  📦 Creating service: ${service_name}${NC}"
    
    local service_dir="${S6_BASE}/${service_name}"
    mkdir -p "${service_dir}"
    
    # Service type
    echo "${service_type}" > "${service_dir}/type"
    
    # Dependencies
    if [[ -n "${dependencies}" ]]; then
        mkdir -p "${service_dir}/dependencies.d"
        IFS=',' read -ra DEPS <<< "${dependencies}"
        for dep in "${DEPS[@]}"; do
            dep=$(echo "${dep}" | xargs) # trim whitespace
            touch "${service_dir}/dependencies.d/${dep}"
            echo -e "${YELLOW}    ⚡ Added dependency: ${dep}${NC}"
        done
    fi
    
    # Add to user bundle
    mkdir -p "${S6_BASE}/user/contents.d"
    touch "${S6_BASE}/user/contents.d/${service_name}"
}

# Function to create run script
create_run_script() {
    local service_name="$1"
    local command="$2"
    local pre_exec="${3:-}"
    
    local run_dir="${S6_BASE}/${service_name}"
    mkdir -p "${run_dir}"
    
    cat > "${run_dir}/run" <<EOF
#!/command/with-contenv bash
# ${service_name} service run script
# Generated automatically by s6-overlay structure generator

set -e

# Setup logging
exec 2>&1

${pre_exec}

echo "[$(date)] Starting ${service_name}..."

# Execute the main command
exec ${command}
EOF
    
    chmod +x "${run_dir}/run"
    echo -e "${GREEN}  ✅ Created run script for ${service_name}${NC}"
}

# Function to create finish script
create_finish_script() {
    local service_name="$1"
    local cleanup_commands="${2:-}"
    
    local finish_dir="${S6_BASE}/${service_name}"
    mkdir -p "${finish_dir}"
    
    cat > "${finish_dir}/finish" <<EOF
#!/command/with-contenv bash
# ${service_name} service finish script
# Handles graceful shutdown and cleanup

set -e

echo "[$(date)] ${service_name} service stopping..."

${cleanup_commands}

echo "[$(date)] ${service_name} cleanup completed"
EOF
    
    chmod +x "${finish_dir}/finish"
    echo -e "${GREEN}  🔄 Created finish script for ${service_name}${NC}"
}

echo -e "${BLUE}🏗️  Creating Tier 1: Infrastructure Services${NC}"

# Tier 1: Infrastructure Services (no dependencies)
create_service "postgres" "longrun" ""
create_run_script "postgres" \
    "postgres -D /var/lib/postgresql/data -c config_file=/etc/postgresql/postgresql.conf" \
    "
# Wait for data directory initialization
if [[ ! -f /var/lib/postgresql/data/PG_VERSION ]]; then
    echo 'Initializing PostgreSQL database...'
    initdb -D /var/lib/postgresql/data --auth-host=md5 --auth-local=peer
    echo \"host all all 127.0.0.1/32 md5\" >> /var/lib/postgresql/data/pg_hba.conf
    echo \"listen_addresses = 'localhost'\" >> /var/lib/postgresql/data/postgresql.conf
fi

# Create databases for services
export PGPASSWORD=\${POSTGRES_PASSWORD:-postgres}
if ! pg_isready -h localhost -p 5432 -q 2>/dev/null; then
    echo 'Starting PostgreSQL for initial setup...'
    pg_ctl -D /var/lib/postgresql/data start -w
    
    # Create application databases
    createdb -h localhost -U postgres jellyfin_db || true
    createdb -h localhost -U postgres plex_db || true
    createdb -h localhost -U postgres sonarr_db || true
    createdb -h localhost -U postgres radarr_db || true
    createdb -h localhost -U postgres lidarr_db || true
    createdb -h localhost -U postgres readarr_db || true
    createdb -h localhost -U postgres prowlarr_db || true
    createdb -h localhost -U postgres overseerr_db || true
    createdb -h localhost -U postgres jellyseerr_db || true
    createdb -h localhost -U postgres ombi_db || true
    createdb -h localhost -U postgres tautulli_db || true
    createdb -h localhost -U postgres grafana_db || true
    createdb -h localhost -U postgres ai_services_db || true
    createdb -h localhost -U postgres shared_db || true
    
    pg_ctl -D /var/lib/postgresql/data stop -w
fi
"

create_service "redis" "longrun" ""
create_run_script "redis" \
    "redis-server /etc/redis/redis.conf --daemonize no" \
    "
# Ensure redis configuration exists
mkdir -p /etc/redis
if [[ ! -f /etc/redis/redis.conf ]]; then
    echo 'Creating Redis configuration...'
    cat > /etc/redis/redis.conf <<REDIS_CONF
port 6379
bind 127.0.0.1
save 60 1000
dbfilename dump.rdb
dir /var/lib/redis
loglevel notice
databases 16
REDIS_CONF
fi

# Create data directory
mkdir -p /var/lib/redis
chown redis:redis /var/lib/redis
"

create_service "rabbitmq" "longrun" ""
create_run_script "rabbitmq" \
    "rabbitmq-server" \
    "
# Setup RabbitMQ environment
export RABBITMQ_NODENAME=rabbit@localhost
export RABBITMQ_CONFIG_FILE=/etc/rabbitmq/rabbitmq
export RABBITMQ_MNESIA_BASE=/var/lib/rabbitmq/mnesia
export RABBITMQ_LOG_BASE=/var/log/rabbitmq

# Create necessary directories
mkdir -p /var/lib/rabbitmq/mnesia /var/log/rabbitmq /etc/rabbitmq
chown -R rabbitmq:rabbitmq /var/lib/rabbitmq /var/log/rabbitmq

# Create basic configuration
cat > /etc/rabbitmq/rabbitmq.conf <<RABBIT_CONF
listeners.tcp.default = 5672
management.tcp.port = 15672
loopback_users.guest = false
RABBIT_CONF
"

create_service "elasticsearch" "longrun" ""
create_run_script "elasticsearch" \
    "elasticsearch" \
    "
# Setup Elasticsearch environment
export ES_HOME=/usr/share/elasticsearch
export ES_PATH_CONF=/etc/elasticsearch
export ES_JAVA_OPTS='-Xms512m -Xmx1g'

# Create directories
mkdir -p /var/lib/elasticsearch /var/log/elasticsearch /etc/elasticsearch
chown -R elasticsearch:elasticsearch /var/lib/elasticsearch /var/log/elasticsearch

# Create basic configuration
cat > /etc/elasticsearch/elasticsearch.yml <<ES_CONF
cluster.name: media-server
node.name: media-node-1
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
ES_CONF
"

echo -e "${BLUE}🌐 Creating Tier 2: Platform Services${NC}"

# Tier 2: Platform Services (depend on infrastructure)
create_service "traefik" "longrun" "postgres,redis"
create_run_script "traefik" \
    "traefik --configfile=/etc/traefik/traefik.yml" \
    "
# Ensure traefik configuration exists
mkdir -p /etc/traefik/dynamic /var/log/traefik
if [[ ! -f /etc/traefik/traefik.yml ]]; then
    echo 'Traefik configuration not found, using minimal config'
    cat > /etc/traefik/traefik.yml <<TRAEFIK_CONF
global:
  sendAnonymousUsage: false
entryPoints:
  web:
    address: ':80'
  websecure:
    address: ':443'
api:
  dashboard: true
  insecure: true
providers:
  file:
    directory: /etc/traefik/dynamic
log:
  level: INFO
TRAEFIK_CONF
fi
"

create_service "auth_service" "longrun" "postgres,redis"
create_run_script "auth_service" \
    "node /opt/auth-service/index.js" \
    "
# Setup authentication service
export NODE_ENV=production
export PORT=8000
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shared_db
export REDIS_URL=redis://localhost:6379/0

cd /opt/auth-service || exit 1
"

echo -e "${BLUE}📺 Creating Tier 3: Media Server Core${NC}"

# Tier 3: Media Servers (depend on infrastructure + platform)
create_service "jellyfin" "longrun" "postgres,redis,traefik,auth_service"
create_run_script "jellyfin" \
    "jellyfin --datadir /config/jellyfin --configdir /config/jellyfin --logdir /var/log/jellyfin --cachedir /var/cache/jellyfin" \
    "
# Setup Jellyfin environment
export JELLYFIN_DATA_DIR=/config/jellyfin
export JELLYFIN_CONFIG_DIR=/config/jellyfin
export JELLYFIN_LOG_DIR=/var/log/jellyfin
export JELLYFIN_CACHE_DIR=/var/cache/jellyfin
export JELLYFIN_WEB_DIR=/usr/share/jellyfin/web

# Create directories
mkdir -p \${JELLYFIN_DATA_DIR} \${JELLYFIN_CONFIG_DIR} \${JELLYFIN_LOG_DIR} \${JELLYFIN_CACHE_DIR}
"

create_service "plex" "longrun" "postgres,redis,traefik,auth_service"
create_run_script "plex" \
    "'/usr/lib/plexmediaserver/Plex Media Server'" \
    "
# Setup Plex environment
export PLEX_MEDIA_SERVER_APPLICATION_SUPPORT_DIR=/config/plex
export PLEX_MEDIA_SERVER_HOME=/usr/lib/plexmediaserver
export PLEX_MEDIA_SERVER_MAX_PLUGIN_PROCS=6
export PLEX_MEDIA_SERVER_TMPDIR=/tmp

# Create directories
mkdir -p /config/plex
"

create_service "emby" "longrun" "postgres,redis,traefik,auth_service"
create_run_script "emby" \
    "embyserver -programdata /config/emby" \
    "
# Setup Emby environment
export EMBY_DATA=/config/emby
export EMBY_MEDIA=/data/media

# Create directories
mkdir -p /config/emby
"

echo -e "${BLUE}🔍 Creating Tier 4A: *arr Stack Management${NC}"

# Tier 4A: *arr Stack (depend on media core + download clients)
create_service "prowlarr" "longrun" "postgres,traefik,auth_service"
create_run_script "prowlarr" \
    "Prowlarr -nobrowser -data=/config/prowlarr" \
    "
export HOME=/config/prowlarr
mkdir -p /config/prowlarr
"

create_service "sonarr" "longrun" "postgres,prowlarr,traefik,auth_service"
create_run_script "sonarr" \
    "Sonarr -nobrowser -data=/config/sonarr" \
    "
export HOME=/config/sonarr
mkdir -p /config/sonarr
"

create_service "radarr" "longrun" "postgres,prowlarr,traefik,auth_service"
create_run_script "radarr" \
    "Radarr -nobrowser -data=/config/radarr" \
    "
export HOME=/config/radarr  
mkdir -p /config/radarr
"

create_service "lidarr" "longrun" "postgres,prowlarr,traefik,auth_service"
create_run_script "lidarr" \
    "Lidarr -nobrowser -data=/config/lidarr" \
    "
export HOME=/config/lidarr
mkdir -p /config/lidarr
"

create_service "readarr" "longrun" "postgres,prowlarr,traefik,auth_service"
create_run_script "readarr" \
    "Readarr -nobrowser -data=/config/readarr" \
    "
export HOME=/config/readarr
mkdir -p /config/readarr
"

create_service "bazarr" "longrun" "postgres,sonarr,radarr,traefik,auth_service"
create_run_script "bazarr" \
    "python3 /opt/bazarr/bazarr.py --no-update --config /config/bazarr" \
    "
export HOME=/config/bazarr
mkdir -p /config/bazarr
"

echo -e "${BLUE}⬇️  Creating Tier 4B: Download Clients${NC}"

# Tier 4B: Download Clients (depend on redis for caching)
create_service "qbittorrent" "longrun" "redis,traefik,auth_service"
create_run_script "qbittorrent" \
    "qbittorrent-nox --webui-port=8080 --profile=/config/qbittorrent" \
    "
mkdir -p /config/qbittorrent /data/downloads/torrents
"

create_service "transmission" "longrun" "redis,traefik,auth_service"
create_run_script "transmission" \
    "transmission-daemon --foreground --config-dir /config/transmission" \
    "
mkdir -p /config/transmission /data/downloads/torrents
"

create_service "sabnzbd" "longrun" "redis,traefik,auth_service"
create_run_script "sabnzbd" \
    "python3 -OO /opt/sabnzbd/SABnzbd.py --config-file /config/sabnzbd/sabnzbd.ini --server 0.0.0.0:8081" \
    "
mkdir -p /config/sabnzbd /data/downloads/usenet
"

create_service "nzbget" "longrun" "redis,traefik,auth_service"
create_run_script "nzbget" \
    "nzbget --daemon --configfile /config/nzbget/nzbget.conf" \
    "
mkdir -p /config/nzbget /data/downloads/usenet
"

echo -e "${BLUE}📋 Creating Tier 5A: Request Services${NC}"

# Tier 5A: Request Services (depend on media servers + *arr stack)
create_service "overseerr" "longrun" "postgres,plex,sonarr,radarr,traefik,auth_service"
create_run_script "overseerr" \
    "node /opt/overseerr/dist/index.js" \
    "
export NODE_ENV=production
export PORT=5055
export CONFIG_DIRECTORY=/config/overseerr
mkdir -p /config/overseerr
cd /opt/overseerr
"

create_service "jellyseerr" "longrun" "postgres,jellyfin,sonarr,radarr,traefik,auth_service"
create_run_script "jellyseerr" \
    "node /opt/jellyseerr/dist/index.js" \
    "
export NODE_ENV=production
export PORT=5056
export CONFIG_DIRECTORY=/config/jellyseerr
mkdir -p /config/jellyseerr
cd /opt/jellyseerr
"

create_service "ombi" "longrun" "postgres,plex,emby,traefik,auth_service"
create_run_script "ombi" \
    "Ombi --host http://0.0.0.0:3579 --storage /config/ombi" \
    "
mkdir -p /config/ombi
"

echo -e "${BLUE}🤖 Creating Tier 5B: AI Services${NC}"

# Tier 5B: AI Services (depend on infrastructure + media servers)
create_service "ai_safety_system" "longrun" "postgres,redis,elasticsearch,jellyfin,plex,traefik,auth_service"
create_run_script "ai_safety_system" \
    "python3 /opt/ai-services/safety_system.py" \
    "
export PYTHONPATH=/opt/ai-services
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_services_db
export REDIS_URL=redis://localhost:6379/4
export ELASTICSEARCH_URL=http://localhost:9200
cd /opt/ai-services
"

create_service "content_moderation" "longrun" "postgres,redis,ai_safety_system,traefik,auth_service"
create_run_script "content_moderation" \
    "python3 /opt/ai-services/content_moderation.py" \
    "
export PYTHONPATH=/opt/ai-services
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_services_db
export REDIS_URL=redis://localhost:6379/4
cd /opt/ai-services
"

create_service "recommendation_engine" "longrun" "postgres,redis,jellyfin,plex,traefik,auth_service"
create_run_script "recommendation_engine" \
    "python3 /opt/ai-services/recommendation_engine.py" \
    "
export PYTHONPATH=/opt/ai-services
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_services_db
export REDIS_URL=redis://localhost:6379/4
cd /opt/ai-services
"

create_service "social_media_integration" "longrun" "postgres,redis,rabbitmq,traefik,auth_service"
create_run_script "social_media_integration" \
    "python3 /opt/ai-services/social_media.py" \
    "
export PYTHONPATH=/opt/ai-services
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_services_db
export REDIS_URL=redis://localhost:6379/5
export RABBITMQ_URL=amqp://localhost:5672
cd /opt/ai-services
"

echo -e "${BLUE}🔧 Creating Tier 5C: Management Tools${NC}"

# Tier 5C: Management Tools
create_service "tautulli" "longrun" "postgres,plex,traefik,auth_service"
create_run_script "tautulli" \
    "python3 /opt/tautulli/Tautulli.py --config /config/tautulli --nolaunch" \
    "
mkdir -p /config/tautulli
"

create_service "organizr" "longrun" "postgres,traefik,auth_service"
create_run_script "organizr" \
    "nginx -g 'daemon off;' -c /etc/organizr/nginx.conf" \
    "
mkdir -p /config/organizr
"

create_service "heimdall" "longrun" "traefik,auth_service"
create_run_script "heimdall" \
    "apache2-foreground" \
    "
mkdir -p /config/heimdall
export APACHE_DOCUMENT_ROOT=/opt/heimdall/public
"

echo -e "${BLUE}📊 Creating Tier 6: Monitoring & Observability${NC}"

# Tier 6: Monitoring (can depend on all other services)
create_service "prometheus" "longrun" "redis"
create_run_script "prometheus" \
    "prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus --web.console.libraries=/etc/prometheus/console_libraries --web.console.templates=/etc/prometheus/consoles --web.enable-lifecycle" \
    "
mkdir -p /var/lib/prometheus /etc/prometheus
if [[ ! -f /etc/prometheus/prometheus.yml ]]; then
    echo 'Creating basic Prometheus config...'
    cat > /etc/prometheus/prometheus.yml <<PROM_CONF
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
PROM_CONF
fi
"

create_service "grafana" "longrun" "postgres,prometheus,traefik,auth_service"
create_run_script "grafana" \
    "grafana-server --config=/etc/grafana/grafana.ini --homepath=/usr/share/grafana" \
    "
mkdir -p /var/lib/grafana /etc/grafana /var/log/grafana
export GF_PATHS_CONFIG=/etc/grafana/grafana.ini
export GF_PATHS_DATA=/var/lib/grafana
export GF_PATHS_LOGS=/var/log/grafana
"

create_service "uptime_kuma" "longrun" "postgres,traefik,auth_service"
create_run_script "uptime_kuma" \
    "node /opt/uptime-kuma/server/server.js" \
    "
export NODE_ENV=production
export DATA_DIR=/config/uptime-kuma
mkdir -p /config/uptime-kuma
cd /opt/uptime-kuma
"

echo -e "${BLUE}🎯 Creating Service Bundles${NC}"

# Create service bundles for different tiers
create_service "infrastructure" "bundle" ""
mkdir -p "${S6_BASE}/infrastructure/contents.d"
touch "${S6_BASE}/infrastructure/contents.d/postgres"
touch "${S6_BASE}/infrastructure/contents.d/redis" 
touch "${S6_BASE}/infrastructure/contents.d/rabbitmq"
touch "${S6_BASE}/infrastructure/contents.d/elasticsearch"

create_service "platform" "bundle" "infrastructure"
mkdir -p "${S6_BASE}/platform/contents.d"
touch "${S6_BASE}/platform/contents.d/traefik"
touch "${S6_BASE}/platform/contents.d/auth_service"

create_service "media_core" "bundle" "platform"
mkdir -p "${S6_BASE}/media_core/contents.d"
touch "${S6_BASE}/media_core/contents.d/jellyfin"
touch "${S6_BASE}/media_core/contents.d/plex"
touch "${S6_BASE}/media_core/contents.d/emby"

create_service "media_management" "bundle" "media_core"
mkdir -p "${S6_BASE}/media_management/contents.d"
touch "${S6_BASE}/media_management/contents.d/prowlarr"
touch "${S6_BASE}/media_management/contents.d/sonarr"
touch "${S6_BASE}/media_management/contents.d/radarr"
touch "${S6_BASE}/media_management/contents.d/lidarr"
touch "${S6_BASE}/media_management/contents.d/readarr"
touch "${S6_BASE}/media_management/contents.d/bazarr"

create_service "download_clients" "bundle" "platform"
mkdir -p "${S6_BASE}/download_clients/contents.d"
touch "${S6_BASE}/download_clients/contents.d/qbittorrent"
touch "${S6_BASE}/download_clients/contents.d/transmission"
touch "${S6_BASE}/download_clients/contents.d/sabnzbd"
touch "${S6_BASE}/download_clients/contents.d/nzbget"

create_service "request_services" "bundle" "media_management"
mkdir -p "${S6_BASE}/request_services/contents.d"
touch "${S6_BASE}/request_services/contents.d/overseerr"
touch "${S6_BASE}/request_services/contents.d/jellyseerr"
touch "${S6_BASE}/request_services/contents.d/ombi"

create_service "ai_services" "bundle" "media_core"
mkdir -p "${S6_BASE}/ai_services/contents.d"
touch "${S6_BASE}/ai_services/contents.d/ai_safety_system"
touch "${S6_BASE}/ai_services/contents.d/content_moderation"
touch "${S6_BASE}/ai_services/contents.d/recommendation_engine"
touch "${S6_BASE}/ai_services/contents.d/social_media_integration"

create_service "management_tools" "bundle" "media_core"
mkdir -p "${S6_BASE}/management_tools/contents.d"
touch "${S6_BASE}/management_tools/contents.d/tautulli"
touch "${S6_BASE}/management_tools/contents.d/organizr"
touch "${S6_BASE}/management_tools/contents.d/heimdall"

create_service "monitoring" "bundle" "platform"
mkdir -p "${S6_BASE}/monitoring/contents.d"
touch "${S6_BASE}/monitoring/contents.d/prometheus"
touch "${S6_BASE}/monitoring/contents.d/grafana"
touch "${S6_BASE}/monitoring/contents.d/uptime_kuma"

# Create the main user bundle that includes all service bundles
mkdir -p "${S6_BASE}/user/contents.d"
touch "${S6_BASE}/user/contents.d/infrastructure"
touch "${S6_BASE}/user/contents.d/platform"
touch "${S6_BASE}/user/contents.d/media_core"
touch "${S6_BASE}/user/contents.d/media_management"
touch "${S6_BASE}/user/contents.d/download_clients"
touch "${S6_BASE}/user/contents.d/request_services"
touch "${S6_BASE}/user/contents.d/ai_services"
touch "${S6_BASE}/user/contents.d/management_tools"
touch "${S6_BASE}/user/contents.d/monitoring"

echo -e "${GREEN}✅ s6-overlay service structure created successfully!${NC}"
echo -e "${BLUE}📝 Service startup order:${NC}"
echo -e "  1. Infrastructure (postgres, redis, rabbitmq, elasticsearch)"
echo -e "  2. Platform (traefik, auth_service)"
echo -e "  3. Media Core (jellyfin, plex, emby)"
echo -e "  4A. Media Management (*arr stack)"
echo -e "  4B. Download Clients (qbittorrent, transmission, sabnzbd, nzbget)"
echo -e "  5A. Request Services (overseerr, jellyseerr, ombi)"
echo -e "  5B. AI Services (safety, moderation, recommendations, social)"
echo -e "  5C. Management Tools (tautulli, organizr, heimdall)"
echo -e "  6. Monitoring (prometheus, grafana, uptime-kuma)"

echo -e "${YELLOW}⚠️  Remember to:${NC}"
echo -e "  • Set appropriate environment variables"
echo -e "  • Configure volume mounts for persistent data"
echo -e "  • Review and customize service configurations"
echo -e "  • Test service dependencies and startup order"

exit 0