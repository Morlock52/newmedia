#!/command/with-contenv bash
# ========================================
# ULTIMATE MEDIA SERVER 2025 - FIXED ENTRYPOINT
# Comprehensive initialization for 30+ services
# ========================================

set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INIT:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $1" >&2
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

log "🚀 Ultimate Media Server 2025 - Starting initialization..."

# ========================================
# ENVIRONMENT SETUP AND VALIDATION
# ========================================

log "⚙️ Setting up environment..."

# Set default values for required variables
export PUID=${PUID:-1000}
export PGID=${PGID:-1000}
export TZ=${TZ:-UTC}
export UMASK=${UMASK:-002}

# Service discovery variables
export POSTGRES_HOST="127.0.0.1"
export POSTGRES_PORT="5432" 
export REDIS_HOST="127.0.0.1"
export REDIS_PORT="6379"
export TRAEFIK_HOST="127.0.0.1"
export TRAEFIK_PORT="8080"

# Media server variables
export JELLYFIN_DATA_DIR="/config/jellyfin/data"
export JELLYFIN_CONFIG_DIR="/config/jellyfin/config"
export JELLYFIN_LOG_DIR="/config/jellyfin/log"
export JELLYFIN_CACHE_DIR="/config/jellyfin/cache"
export JELLYFIN_WEB_DIR="/usr/share/jellyfin/web"

# *ARR configuration
export SONARR_DATA=/config/sonarr
export RADARR_DATA=/config/radarr
export LIDARR_DATA=/config/lidarr
export READARR_DATA=/config/readarr
export PROWLARR_DATA=/config/prowlarr
export BAZARR_DATA=/config/bazarr

# Performance and resource settings
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=8192}"
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
export DOTNET_CLI_TELEMETRY_OPTOUT=1

log "✅ Environment variables configured"

# ========================================
# USER AND GROUP SETUP
# ========================================

log "👥 Setting up users and groups..."

# Ensure media group exists
if ! getent group media > /dev/null 2>&1; then
    groupadd -g ${PGID} media
    log "Created media group with GID ${PGID}"
fi

# Ensure media user exists
if ! getent passwd media > /dev/null 2>&1; then
    useradd -u ${PUID} -g media -d /config -s /bin/bash media
    log "Created media user with UID ${PUID}"
fi

# Create service users if they don't exist
declare -A service_users=(
    ["postgres"]="999"
    ["redis"]="998" 
    ["grafana"]="472"
    ["prometheus"]="65534"
)

for username in "${!service_users[@]}"; do
    uid=${service_users[$username]}
    if ! getent passwd $username > /dev/null 2>&1; then
        useradd -u $uid -r -g media -d /var/lib/$username -s /bin/bash $username 2>/dev/null || true
        log "Created service user: $username (UID: $uid)"
    fi
done

log "✅ User and group setup completed"

# ========================================
# DIRECTORY STRUCTURE CREATION
# ========================================

log "📁 Creating directory structure..."

# Create all necessary directories
directories=(
    # Configuration directories
    "/config/jellyfin/"{data,config,log,cache,plugins,metadata}
    "/config/plex/Library/Application Support/Plex Media Server"
    "/config/"{sonarr,radarr,lidarr,readarr,bazarr,prowlarr}
    "/config/"{qbittorrent,transmission,sabnzbd,nzbget}
    "/config/"{overseerr,jellyseerr,ombi,tautulli}
    "/config/"{organizr,homepage,homarr}
    "/config/"{audiobookshelf,navidrome,vaultwarden}
    "/config/"{prometheus,grafana,uptime-kuma}
    "/config/"{traefik,nginx,redis,postgres}
    "/config/"{ollama,ai-assistant}
    
    # Data directories
    "/data/media/"{movies,tv,music,books,audiobooks,photos,documents,comics}
    "/data/downloads/"{complete,incomplete,watch,torrents,usenet}
    "/data/"{backups,logs,cache,tmp}
    
    # Database directories
    "/data/databases/"{postgres,redis,grafana,prometheus}
    
    # Runtime directories
    "/var/run/"{postgresql,redis,grafana,prometheus}
    "/var/log/media-server"
    "/tmp/media-server"
    
    # AI and model storage
    "/models/"{ollama,whisper,stable-diffusion}
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

log "✅ Directory structure created"

# ========================================
# PERMISSION SETUP
# ========================================

log "🔒 Setting up permissions..."

# Set ownership for main directories
chown -R media:media /config /data /models /var/log/media-server /tmp/media-server

# Set specific ownership for service directories
chown -R postgres:media /config/postgres /data/databases/postgres /var/run/postgresql 2>/dev/null || true
chown -R redis:media /config/redis /data/databases/redis /var/run/redis 2>/dev/null || true
chown -R grafana:media /config/grafana /data/databases/grafana /var/run/grafana 2>/dev/null || true
chown -R prometheus:media /config/prometheus /data/databases/prometheus /var/run/prometheus 2>/dev/null || true

# Set permissions
chmod -R 775 /config /data /models
chmod -R 755 /var/log/media-server /tmp/media-server

# Set special permissions for database directories
chmod 700 /data/databases/postgres 2>/dev/null || true
chmod 755 /data/databases/redis 2>/dev/null || true

log "✅ Permissions configured"

# ========================================
# DATABASE INITIALIZATION
# ========================================

log "🗄️ Initializing databases..."

# PostgreSQL initialization
if [ ! -f "/data/databases/postgres/PG_VERSION" ]; then
    log "Initializing PostgreSQL database..."
    
    # Ensure postgres user owns the directory
    chown -R postgres:postgres /data/databases/postgres
    chmod 700 /data/databases/postgres
    
    # Initialize database as postgres user
    runuser -u postgres -- /usr/lib/postgresql/15/bin/initdb \
        -D /data/databases/postgres \
        --locale=C.UTF-8 \
        --encoding=UTF8 \
        --auth-local=peer \
        --auth-host=md5
    
    # Configure PostgreSQL
    cat > /data/databases/postgres/postgresql.conf << 'EOF'
listen_addresses = 'localhost'
port = 5432
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.7
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
log_destination = 'stderr'
logging_collector = on
log_directory = '/var/log/media-server'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 10MB
log_min_messages = warning
log_min_error_statement = error
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
EOF

    cat > /data/databases/postgres/pg_hba.conf << 'EOF'
local   all             postgres                                peer
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
EOF
    
    chown postgres:postgres /data/databases/postgres/postgresql.conf /data/databases/postgres/pg_hba.conf
    log "✅ PostgreSQL initialized"
else
    log "PostgreSQL database already exists"
fi

# Redis initialization
if [ ! -f "/data/databases/redis/dump.rdb" ]; then
    log "Initializing Redis..."
    
    mkdir -p /data/databases/redis
    cat > /config/redis/redis.conf << 'EOF'
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300
daemonize no
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/media-server/redis.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data/databases/redis
maxmemory 1024mb
maxmemory-policy allkeys-lru
EOF
    
    chown -R redis:redis /config/redis /data/databases/redis
    log "✅ Redis configured"
else
    log "Redis configuration already exists"
fi

log "✅ Database initialization completed"

# ========================================
# SERVICE CONFIGURATION GENERATION
# ========================================

log "⚙️ Generating service configurations..."

# Generate Traefik configuration
mkdir -p /config/traefik/{config,acme,logs}
if [ ! -f "/config/traefik/traefik.yml" ]; then
    cat > /config/traefik/traefik.yml << 'EOF'
global:
  checkNewVersion: false
  sendAnonymousUsage: false

api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  file:
    directory: /config/traefik/config
    watch: true

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@localhost
      storage: /config/traefik/acme/acme.json
      httpChallenge:
        entryPoint: web

log:
  level: INFO
  filePath: /config/traefik/logs/traefik.log

accessLog:
  filePath: /config/traefik/logs/access.log
EOF
    
    # Create dynamic configuration
    mkdir -p /config/traefik/config
    cat > /config/traefik/config/dynamic.yml << 'EOF'
http:
  routers:
    jellyfin:
      rule: "PathPrefix(`/jellyfin`) || PathPrefix(`/`)"
      service: jellyfin
      priority: 1
    sonarr:
      rule: "PathPrefix(`/sonarr`)"
      service: sonarr
    radarr:
      rule: "PathPrefix(`/radarr`)"
      service: radarr
    prowlarr:
      rule: "PathPrefix(`/prowlarr`)"
      service: prowlarr
    qbittorrent:
      rule: "PathPrefix(`/qbittorrent`)"
      service: qbittorrent

  services:
    jellyfin:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8096"
    sonarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8989"
    radarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:7878"
    prowlarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:9696"
    qbittorrent:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8080"
EOF
    
    # Create empty acme.json with proper permissions
    touch /config/traefik/acme/acme.json
    chmod 600 /config/traefik/acme/acme.json
    
    log "✅ Traefik configuration generated"
fi

# Generate Prometheus configuration
if [ ! -f "/config/prometheus/prometheus.yml" ]; then
    cat > /config/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'jellyfin'
    static_configs:
      - targets: ['localhost:8096']
    scrape_interval: 30s

  - job_name: 'sonarr'
    static_configs:
      - targets: ['localhost:8989']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'radarr'
    static_configs:
      - targets: ['localhost:7878']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'qbittorrent'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s

  - job_name: 'traefik'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
EOF
    log "✅ Prometheus configuration generated"
fi

# Generate Grafana configuration
if [ ! -f "/config/grafana/grafana.ini" ]; then
    cat > /config/grafana/grafana.ini << 'EOF'
[server]
http_addr = 0.0.0.0
http_port = 3000
root_url = %(protocol)s://%(domain)s:%(http_port)s/

[database]
type = sqlite3
path = /data/databases/grafana/grafana.db

[security]
admin_user = admin
admin_password = admin123
secret_key = SW2YcwTIb9zpOOhoPsMm
disable_gravatar = true

[users]
allow_sign_up = false
allow_org_create = false
auto_assign_org = true
auto_assign_org_role = Viewer

[auth.anonymous]
enabled = false

[log]
mode = file
level = info
filters = rendering:debug

[paths]
data = /data/databases/grafana
logs = /var/log/media-server
plugins = /config/grafana/plugins
provisioning = /config/grafana/provisioning

[alerting]
enabled = true

[unified_alerting]
enabled = true
EOF
    
    mkdir -p /config/grafana/{plugins,provisioning/{datasources,dashboards}}
    
    # Create datasource configuration
    cat > /config/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
EOF
    
    log "✅ Grafana configuration generated"
fi

# Generate Nginx configuration for services that need it
mkdir -p /config/nginx
if [ ! -f "/config/nginx/nginx.conf" ]; then
    cat > /config/nginx/nginx.conf << 'EOF'
user media;
worker_processes auto;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';
                   
    access_log /var/log/media-server/nginx_access.log main;
    error_log /var/log/media-server/nginx_error.log warn;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Organizr proxy configuration
    server {
        listen 8181;
        server_name localhost;
        
        root /opt/organizr;
        index index.php index.html;
        
        location ~ \.php$ {
            fastcgi_pass unix:/var/run/php/php8.2-fpm.sock;
            fastcgi_index index.php;
            include fastcgi_params;
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        }
        
        location / {
            try_files $uri $uri/ /index.php?$args;
        }
    }
}
EOF
    log "✅ Nginx configuration generated"
fi

log "✅ Service configurations generated"

# ========================================
# CREATE MISSING S6 SERVICE SCRIPTS
# ========================================

log "📝 Creating s6-overlay service scripts..."

# Create services that might be missing
services_to_create=(
    "postgres"
    "redis" 
    "nginx"
    "ai-assistant"
    "emby"
    "calibre-web"
    "immich"
    "paperless"
    "nextcloud"
    "pihole"
    "adguard"
    "syncthing"
    "code-server"
    "gitea"
)

for service in "${services_to_create[@]}"; do
    service_dir="/etc/s6-overlay/s6-rc.d/$service"
    if [ ! -d "$service_dir" ]; then
        mkdir -p "$service_dir"
        echo "longrun" > "$service_dir/type"
        
        # Create basic run script based on service type
        case $service in
            "postgres")
                cat > "$service_dir/run" << 'EOF'
#!/command/with-contenv bash
set -e
echo "Starting PostgreSQL..."
exec s6-setuidgid postgres /usr/lib/postgresql/15/bin/postgres -D /data/databases/postgres
EOF
                ;;
            "redis")
                cat > "$service_dir/run" << 'EOF'
#!/command/with-contenv bash
set -e
echo "Starting Redis..."
exec s6-setuidgid redis redis-server /config/redis/redis.conf
EOF
                ;;
            "nginx")
                cat > "$service_dir/run" << 'EOF'
#!/command/with-contenv bash
set -e
echo "Starting Nginx..."
exec nginx -g "daemon off;" -c /config/nginx/nginx.conf
EOF
                ;;
            "ai-assistant")
                cat > "$service_dir/run" << 'EOF'
#!/command/with-contenv bash
set -e
echo "Starting AI Assistant..."
cd /opt/ai-assistant
exec s6-setuidgid media npm start
EOF
                ;;
            *)
                cat > "$service_dir/run" << EOF
#!/command/with-contenv bash
set -e
echo "Starting $service..."
# Placeholder - service not fully configured
sleep infinity
EOF
                ;;
        esac
        
        chmod +x "$service_dir/run"
        echo "$service" >> /etc/s6-overlay/s6-rc.d/user/contents.d/contents
        
        log "Created s6 service: $service"
    fi
done

log "✅ s6-overlay service scripts created"

# ========================================
# API KEYS AND SECURITY SETUP
# ========================================

log "🔐 Setting up API keys and security..."

mkdir -p /config/api-keys

# Generate API keys if they don't exist
api_keys=(
    "sonarr"
    "radarr"
    "lidarr"
    "readarr"
    "prowlarr"
    "bazarr"
    "jellyfin"
    "overseerr"
    "jellyseerr"
    "tautulli"
)

for service in "${api_keys[@]}"; do
    key_file="/config/api-keys/$service"
    if [ ! -f "$key_file" ]; then
        # Generate a secure API key
        api_key=$(openssl rand -hex 32)
        echo "$api_key" > "$key_file"
        chown media:media "$key_file"
        chmod 600 "$key_file"
        log "Generated API key for $service"
    fi
done

log "✅ API keys and security configured"

# ========================================
# STARTUP OPTIMIZATION
# ========================================

log "⚡ Applying startup optimizations..."

# Set system limits for better performance
if [ -w /proc/sys/fs/inotify/max_user_watches ]; then
    echo 1048576 > /proc/sys/fs/inotify/max_user_watches
fi

if [ -w /proc/sys/fs/inotify/max_user_instances ]; then
    echo 1024 > /proc/sys/fs/inotify/max_user_instances  
fi

# Set network optimizations
if [ -w /proc/sys/net/core/rmem_max ]; then
    echo 16777216 > /proc/sys/net/core/rmem_max
fi

if [ -w /proc/sys/net/core/wmem_max ]; then
    echo 16777216 > /proc/sys/net/core/wmem_max
fi

log "✅ System optimizations applied"

# ========================================
# HEALTH MONITORING SETUP
# ========================================

log "🏥 Setting up health monitoring..."

# Create health check script for services
cat > /app/scripts/service-health-monitor.sh << 'EOF'
#!/bin/bash
# Service health monitoring script
# This runs continuously to monitor service health

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check critical services
    critical_services=("postgres" "redis" "jellyfin" "sonarr" "radarr" "prowlarr")
    
    for service in "${critical_services[@]}"; do
        case $service in
            "postgres")
                if ! pg_isready -h 127.0.0.1 -p 5432 -q 2>/dev/null; then
                    echo "$timestamp - WARNING: PostgreSQL is not responding" >> /var/log/media-server/health.log
                fi
                ;;
            "redis")
                if ! redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -q PONG; then
                    echo "$timestamp - WARNING: Redis is not responding" >> /var/log/media-server/health.log
                fi
                ;;
            *)
                # For other services, check if the process is running
                if ! pgrep -f "$service" > /dev/null; then
                    echo "$timestamp - WARNING: $service process not found" >> /var/log/media-server/health.log
                fi
                ;;
        esac
    done
    
    sleep 60
done
EOF

chmod +x /app/scripts/service-health-monitor.sh

log "✅ Health monitoring configured"

# ========================================
# FINAL VALIDATION
# ========================================

log "🔍 Performing final validation..."

# Verify critical directories exist and have correct permissions
critical_dirs=(
    "/config"
    "/data"
    "/models" 
    "/var/log/media-server"
    "/data/databases/postgres"
    "/data/databases/redis"
)

for dir in "${critical_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        error "Critical directory missing: $dir"
        exit 1
    fi
done

# Verify key configuration files exist
critical_files=(
    "/config/traefik/traefik.yml"
    "/config/prometheus/prometheus.yml"
    "/config/grafana/grafana.ini"
)

for file in "${critical_files[@]}"; do
    if [ ! -f "$file" ]; then
        warn "Configuration file missing: $file"
    fi
done

log "✅ Validation completed"

# ========================================
# STARTUP COMPLETION
# ========================================

log "🎉 Initialization completed successfully!"
log "🚀 Starting service orchestration with s6-overlay..."
log "📊 Services will be available at:"
log "   - Jellyfin: http://localhost:8096"
log "   - Sonarr: http://localhost:8989"  
log "   - Radarr: http://localhost:7878"
log "   - Prowlarr: http://localhost:9696"
log "   - qBittorrent: http://localhost:8080"
log "   - Grafana: http://localhost:3000"
log "   - Traefik Dashboard: http://localhost:8080"

# Start health monitor in background
/app/scripts/service-health-monitor.sh &

log "🌟 Ultimate Media Server 2025 initialization complete!"

# Hand over to s6-overlay
exec /init