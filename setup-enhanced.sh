#!/bin/bash

# Enhanced Media Server Setup Wizard
# Production-ready setup with all services and security

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    __  __          _ _         ____                              ║
║   |  \/  | ___  __| (_) __ _  / ___|  ___ _ ____   _____ _ __   ║
║   | |\/| |/ _ \/ _` | |/ _` | \___ \ / _ \ '__\ \ / / _ \ '__|  ║
║   | |  | |  __/ (_| | | (_| |  ___) |  __/ |   \ V /  __/ |     ║
║   |_|  |_|\___|\__,_|_|\__,_| |____/ \___|_|    \_/ \___|_|     ║
║                                                                  ║
║                 Enhanced Production Setup v2.0                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF

echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_section() {
    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    local prereqs_met=true
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker is installed ($(docker --version))"
    else
        print_error "Docker is not installed"
        prereqs_met=false
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose is installed"
    else
        print_error "Docker Compose is not installed"
        prereqs_met=false
    fi
    
    # Check available disk space
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -gt 50 ]; then
        print_success "Sufficient disk space available (${available_space}GB)"
    else
        print_warning "Low disk space (${available_space}GB). Recommend at least 50GB"
    fi
    
    # Check available memory
    local total_memory=$(free -g | awk 'NR==2 {print $2}')
    if [ "$total_memory" -ge 4 ]; then
        print_success "Sufficient memory available (${total_memory}GB)"
    else
        print_warning "Low memory (${total_memory}GB). Recommend at least 4GB"
    fi
    
    if [ "$prereqs_met" = false ]; then
        print_error "Please install missing prerequisites before continuing"
        exit 1
    fi
}

# Function to create directory structure
create_directories() {
    print_section "Creating Directory Structure"
    
    local dirs=(
        "config/traefik/dynamic"
        "config/authelia"
        "config/homepage"
        "config/jellyfin"
        "config/plex"
        "config/emby"
        "config/prowlarr"
        "config/sonarr"
        "config/radarr"
        "config/lidarr"
        "config/readarr"
        "config/bazarr"
        "config/overseerr"
        "config/ombi"
        "config/tautulli"
        "config/qbittorrent"
        "config/sabnzbd"
        "config/gluetun"
        "config/navidrome"
        "config/airsonic"
        "config/audiobookshelf"
        "config/calibre-web"
        "config/kavita"
        "config/mylar3"
        "config/immich"
        "config/photoprism"
        "config/fileflows"
        "config/tdarr"
        "config/portainer"
        "config/duplicati"
        "config/prometheus"
        "config/grafana/provisioning/dashboards"
        "config/grafana/provisioning/datasources"
        "config/postgres"
        "config/homepage"
        "config/homarr"
        "data/media/movies"
        "data/media/tv"
        "data/media/music"
        "data/media/books"
        "data/media/audiobooks"
        "data/media/podcasts"
        "data/media/photos"
        "data/media/comics"
        "data/media/manga"
        "data/downloads/complete"
        "data/downloads/incomplete"
        "data/downloads/usenet"
        "cache/jellyfin"
        "logs/traefik"
        "logs/authelia"
        "logs/fileflows"
        "logs/tdarr"
        "certs"
        "backups"
        "metadata/audiobookshelf"
        "secrets"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        print_success "Created $dir"
    done
    
    # Set permissions
    chmod 600 certs 2>/dev/null || true
}

# Function to generate random passwords
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Function to create .env file
create_env_file() {
    print_section "Environment Configuration"
    
    if [ -f .env ]; then
        print_warning ".env file already exists"
        read -p "Do you want to backup and recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
            print_success "Backed up existing .env file"
        else
            return
        fi
    fi
    
    # Collect user input
    echo ""
    print_info "Please provide the following configuration values:"
    echo ""
    
    read -p "Enter your domain (e.g., media.example.com): " DOMAIN
    read -p "Enter your email address: " ADMIN_EMAIL
    read -p "Enter your timezone (e.g., America/New_York): " TZ
    
    # Cloudflare configuration
    echo ""
    print_info "Cloudflare Configuration (for SSL certificates):"
    read -p "Enter your Cloudflare email: " CLOUDFLARE_EMAIL
    read -sp "Enter your Cloudflare API token: " CLOUDFLARE_API_TOKEN
    echo ""
    
    # VPN configuration
    echo ""
    print_info "VPN Configuration (optional - press Enter to skip):"
    read -p "Enter VPN provider (e.g., nordvpn, pia, surfshark): " VPN_PROVIDER
    if [ -n "$VPN_PROVIDER" ]; then
        read -p "Enter VPN username: " VPN_USERNAME
        read -sp "Enter VPN password: " VPN_PASSWORD
        echo ""
        read -p "Enter VPN country (e.g., Switzerland): " VPN_COUNTRY
    fi
    
    # SMTP configuration
    echo ""
    print_info "SMTP Configuration (for notifications - optional):"
    read -p "Enter SMTP host: " SMTP_HOST
    if [ -n "$SMTP_HOST" ]; then
        read -p "Enter SMTP port (e.g., 587): " SMTP_PORT
        read -p "Enter SMTP username: " SMTP_USER
        read -sp "Enter SMTP password: " SMTP_PASSWORD
        echo ""
        read -p "Enter FROM email address: " SMTP_FROM
    fi
    
    # Generate passwords
    print_info "Generating secure passwords..."
    
    # Create .env file
    cat > .env << EOF
# Basic Configuration
DOMAIN=${DOMAIN}
ADMIN_EMAIL=${ADMIN_EMAIL}
TZ=${TZ:-America/New_York}
PUID=1000
PGID=1000

# Network Configuration
MEDIA_NETWORK_SUBNET=172.20.0.0/16

# Cloudflare Configuration
CLOUDFLARE_EMAIL=${CLOUDFLARE_EMAIL}
CLOUDFLARE_API_TOKEN=${CLOUDFLARE_API_TOKEN}

# VPN Configuration
VPN_PROVIDER=${VPN_PROVIDER:-nordvpn}
VPN_USERNAME=${VPN_USERNAME}
VPN_PASSWORD=${VPN_PASSWORD}
VPN_COUNTRIES=${VPN_COUNTRY:-Switzerland}

# Database Passwords
POSTGRES_USER=mediaserver
POSTGRES_PASSWORD=$(generate_password)
POSTGRES_DB=mediaserver
REDIS_PASSWORD=$(generate_password)
IMMICH_DB_PASSWORD=$(generate_password)
PHOTOPRISM_DB_PASSWORD=$(generate_password)
PHOTOPRISM_DB_ROOT_PASSWORD=$(generate_password)

# Service Passwords
PHOTOPRISM_ADMIN_USER=admin
PHOTOPRISM_ADMIN_PASSWORD=$(generate_password)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(generate_password)

# Plex Configuration
PLEX_CLAIM=

# SMTP Configuration
SMTP_HOST=${SMTP_HOST}
SMTP_PORT=${SMTP_PORT:-587}
SMTP_USER=${SMTP_USER}
SMTP_PASSWORD=${SMTP_PASSWORD}
SMTP_FROM=${SMTP_FROM}

# Hardware Acceleration
RENDER_GROUP_ID=989

# API Keys (will be generated on first run)
JELLYFIN_API_KEY=
SONARR_API_KEY=
RADARR_API_KEY=
LIDARR_API_KEY=
READARR_API_KEY=
PROWLARR_API_KEY=
BAZARR_API_KEY=
OVERSEERR_API_KEY=
TAUTULLI_API_KEY=

# URLs
JELLYFIN_PUBLISHED_SERVER_URL=https://jellyfin.${DOMAIN}
EOF
    
    chmod 600 .env
    print_success "Created .env file with secure passwords"
}

# Function to create Authelia configuration
create_authelia_config() {
    print_section "Creating Authelia Configuration"
    
    # Generate secrets
    local jwt_secret=$(generate_password)
    local session_secret=$(generate_password)
    local storage_encryption_key=$(generate_password)
    
    cat > config/authelia/configuration.yml << EOF
---
# Authelia Configuration

server:
  host: 0.0.0.0
  port: 9091

log:
  level: info
  format: json
  file_path: /logs/authelia.log

theme: dark

totp:
  issuer: authelia.com
  period: 30
  skew: 1

authentication_backend:
  file:
    path: /config/users_database.yml
    password:
      algorithm: argon2id
      iterations: 1
      salt_length: 16
      parallelism: 8
      memory: 64

access_control:
  default_policy: deny
  rules:
    # Public access
    - domain: "jellyfin.${DOMAIN}"
      policy: bypass
    - domain: "requests.${DOMAIN}"
      policy: bypass
    - domain: "overseerr.${DOMAIN}"
      policy: bypass
    
    # Admin only
    - domain:
        - "traefik.${DOMAIN}"
        - "portainer.${DOMAIN}"
        - "prometheus.${DOMAIN}"
        - "grafana.${DOMAIN}"
      policy: two_factor
      subject: "group:admins"
    
    # Authenticated users
    - domain:
        - "*.${DOMAIN}"
      policy: one_factor

session:
  name: authelia_session
  secret: ${session_secret}
  expiration: 1h
  inactivity: 15m
  remember_me_duration: 1M
  domain: ${DOMAIN}
  redis:
    host: redis
    port: 6379
    password: \${REDIS_PASSWORD}

regulation:
  max_retries: 3
  find_time: 2m
  ban_time: 5m

storage:
  encryption_key: ${storage_encryption_key}
  postgres:
    host: postgres
    port: 5432
    database: authelia
    username: authelia
    password: \${POSTGRES_PASSWORD}

notifier:
  smtp:
    host: \${SMTP_HOST}
    port: \${SMTP_PORT}
    username: \${SMTP_USER}
    password: \${SMTP_PASSWORD}
    sender: \${SMTP_FROM}
    subject: "[Authelia] {title}"
    startup_check_address: \${ADMIN_EMAIL}
    disable_require_tls: false
    disable_html_emails: false

identity_providers:
  oidc:
    hmac_secret: ${jwt_secret}
    issuer_private_key: |
      -----BEGIN RSA PRIVATE KEY-----
      # Generate with: openssl genrsa -out private.pem 4096
      -----END RSA PRIVATE KEY-----
EOF
    
    # Create users database
    cat > config/authelia/users_database.yml << EOF
---
users:
  admin:
    displayname: "Administrator"
    password: "\$argon2id\$v=19\$m=65536,t=1,p=8\$YWJjZGVmZ2hpams\$Zc0soLQ2IBOdxCBr2K0BssY9w6yPMGkH"  # Change this!
    email: ${ADMIN_EMAIL}
    groups:
      - admins
      - users
EOF
    
    print_success "Created Authelia configuration"
    print_warning "Remember to update the admin password in users_database.yml"
}

# Function to create Prometheus configuration
create_prometheus_config() {
    print_section "Creating Prometheus Configuration"
    
    cat > config/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'media-server'

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - "alert_rules.yml"

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # Docker
  - job_name: 'docker'
    static_configs:
      - targets: ['172.17.0.1:9323']

  # Traefik
  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik:8080']

  # Media Services
  - job_name: 'media-services'
    static_configs:
      - targets:
          - 'jellyfin:8096'
          - 'sonarr:8989'
          - 'radarr:7878'
          - 'lidarr:8686'
          - 'prowlarr:9696'
          - 'bazarr:6767'
    metrics_path: '/metrics'
    
  # Containers via cAdvisor
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
EOF
    
    # Create alert rules
    cat > config/prometheus/alert_rules.yml << EOF
groups:
  - name: media_server_alerts
    interval: 30s
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ \$labels.job }} is down"
          description: "{{ \$labels.instance }} of job {{ \$labels.job }} has been down for more than 2 minutes."
      
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "{{ \$labels.instance }} has high CPU usage ({{ \$value }})"
      
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "{{ \$labels.instance }} has high memory usage ({{ \$value }})"
      
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "{{ \$labels.instance }} has less than 10% disk space available"
EOF
    
    print_success "Created Prometheus configuration"
}

# Function to create Grafana provisioning
create_grafana_config() {
    print_section "Creating Grafana Configuration"
    
    # Create datasources
    cat > config/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    # Create dashboard provisioning
    cat > config/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'Media Server Dashboards'
    orgId: 1
    folder: ''
    folderUid: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    print_success "Created Grafana configuration"
}

# Function to create Homepage configuration
create_homepage_config() {
    print_section "Creating Homepage Configuration"
    
    # Create services configuration
    cat > config/homepage/services.yaml << EOF
---
# Homepage Services Configuration

- Media Servers:
    - Jellyfin:
        href: https://jellyfin.${DOMAIN}
        icon: jellyfin.png
        description: Primary media server
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: \${JELLYFIN_API_KEY}

    - Plex:
        href: https://plex.${DOMAIN}
        icon: plex.png
        description: Alternative media server
        widget:
          type: plex
          url: http://plex:32400
          key: \${PLEX_TOKEN}

- Media Management:
    - Sonarr:
        href: https://sonarr.${DOMAIN}
        icon: sonarr.png
        description: TV show management
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: \${SONARR_API_KEY}

    - Radarr:
        href: https://radarr.${DOMAIN}
        icon: radarr.png
        description: Movie management
        widget:
          type: radarr
          url: http://radarr:7878
          key: \${RADARR_API_KEY}

    - Lidarr:
        href: https://lidarr.${DOMAIN}
        icon: lidarr.png
        description: Music management
        widget:
          type: lidarr
          url: http://lidarr:8686
          key: \${LIDARR_API_KEY}

    - Prowlarr:
        href: https://prowlarr.${DOMAIN}
        icon: prowlarr.png
        description: Indexer management
        widget:
          type: prowlarr
          url: http://prowlarr:9696
          key: \${PROWLARR_API_KEY}

- Downloads:
    - qBittorrent:
        href: https://qbittorrent.${DOMAIN}
        icon: qbittorrent.png
        description: Torrent client
        widget:
          type: qbittorrent
          url: http://gluetun:8080
          username: admin
          password: adminadmin

    - SABnzbd:
        href: https://sabnzbd.${DOMAIN}
        icon: sabnzbd.png
        description: Usenet client
        widget:
          type: sabnzbd
          url: http://sabnzbd:8080
          key: \${SABNZBD_API_KEY}

- Request Management:
    - Overseerr:
        href: https://requests.${DOMAIN}
        icon: overseerr.png
        description: Media requests
        widget:
          type: overseerr
          url: http://overseerr:5055
          key: \${OVERSEERR_API_KEY}

- Specialized Servers:
    - Navidrome:
        href: https://music.${DOMAIN}
        icon: navidrome.png
        description: Music streaming

    - Audiobookshelf:
        href: https://audiobooks.${DOMAIN}
        icon: audiobookshelf.png
        description: Audiobook server

    - Calibre-Web:
        href: https://books.${DOMAIN}
        icon: calibre-web.png
        description: E-book server

    - Immich:
        href: https://photos.${DOMAIN}
        icon: immich.png
        description: Photo management

- System:
    - Portainer:
        href: https://portainer.${DOMAIN}
        icon: portainer.png
        description: Container management

    - Traefik:
        href: https://traefik.${DOMAIN}
        icon: traefik.png
        description: Reverse proxy

    - Grafana:
        href: https://grafana.${DOMAIN}
        icon: grafana.png
        description: Monitoring dashboards

    - Uptime Kuma:
        href: https://uptime.${DOMAIN}
        icon: uptime-kuma.png
        description: Service monitoring
EOF
    
    # Create settings
    cat > config/homepage/settings.yaml << EOF
---
title: Media Server Dashboard
startUrl: https://home.${DOMAIN}
background: https://images.unsplash.com/photo-1502920917128-1aa500764cbd
cardBlur: sm
theme: dark
color: slate
layout:
  Media Servers:
    style: row
    columns: 4
  Media Management:
    style: row
    columns: 4
  Downloads:
    style: row
    columns: 3
  Request Management:
    style: row
    columns: 2
  Specialized Servers:
    style: row
    columns: 4
  System:
    style: row
    columns: 4
EOF
    
    # Create widgets
    cat > config/homepage/widgets.yaml << EOF
---
- greeting:
    text_size: xl
    text: Welcome to your Media Server

- datetime:
    text_size: xl
    format:
      dateStyle: long
      timeStyle: short
      hourCycle: h23

- openweathermap:
    latitude: 40.7128
    longitude: -74.0060
    units: imperial
    apiKey: \${OPENWEATHER_API_KEY}

- search:
    provider: google
    target: _blank
EOF
    
    # Create docker integration
    cat > config/homepage/docker.yaml << EOF
---
# Docker socket integration
docker:
  socket: /var/run/docker.sock
EOF
    
    print_success "Created Homepage configuration"
}

# Function to create initial PostgreSQL setup
create_postgres_init() {
    print_section "Creating PostgreSQL Initialization"
    
    cat > config/postgres/init.sql << 'EOF'
-- Create databases for services
CREATE DATABASE authelia;
CREATE DATABASE homarr;
CREATE DATABASE immich;
CREATE DATABASE photoprism;

-- Create users
CREATE USER authelia WITH PASSWORD :'POSTGRES_PASSWORD';
CREATE USER homarr WITH PASSWORD :'POSTGRES_PASSWORD';
CREATE USER immich WITH PASSWORD :'IMMICH_DB_PASSWORD';
CREATE USER photoprism WITH PASSWORD :'PHOTOPRISM_DB_PASSWORD';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE authelia TO authelia;
GRANT ALL PRIVILEGES ON DATABASE homarr TO homarr;
GRANT ALL PRIVILEGES ON DATABASE immich TO immich;
GRANT ALL PRIVILEGES ON DATABASE photoprism TO photoprism;

-- Enable extensions for Immich
\c immich;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vectors";
CREATE EXTENSION IF NOT EXISTS "earthdistance";
EOF
    
    print_success "Created PostgreSQL initialization script"
}

# Function to generate basic auth password
generate_htpasswd() {
    local username=$1
    local password=$2
    echo $(htpasswd -nbB "$username" "$password")
}

# Function to create security files
create_security_files() {
    print_section "Creating Security Files"
    
    # Create Traefik users file
    if command -v htpasswd &> /dev/null; then
        local admin_password=$(generate_password)
        htpasswd -cbB config/traefik/users.htpasswd admin "$admin_password"
        echo "admin:$admin_password" > secrets/traefik_admin_credentials.txt
        chmod 600 secrets/traefik_admin_credentials.txt
        print_success "Created Traefik admin user"
        print_info "Admin credentials saved to secrets/traefik_admin_credentials.txt"
    else
        print_warning "htpasswd not found. Skipping Traefik users file creation"
    fi
    
    # Create acme.json for Let's Encrypt
    touch certs/acme.json
    chmod 600 certs/acme.json
    print_success "Created acme.json for SSL certificates"
}

# Function to create health check script
create_health_check_script() {
    print_section "Creating Health Check Script"
    
    cat > scripts/health-check-all.sh << 'EOF'
#!/bin/bash

# Health check script for all services

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================"
echo "Media Server Health Check"
echo "======================================"
echo ""

# Function to check service
check_service() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    if curl -fsS -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_code"; then
        echo -e "${GREEN}✅ $name is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ $name is unhealthy${NC}"
        return 1
    fi
}

# Check core services
echo "Checking Core Services:"
check_service "Traefik" "http://localhost:8080/ping"
check_service "Jellyfin" "http://localhost:8096/health"
check_service "Homepage" "http://localhost:3000"

echo ""
echo "Checking Media Management:"
check_service "Sonarr" "http://localhost:8989/ping"
check_service "Radarr" "http://localhost:7878/ping"
check_service "Lidarr" "http://localhost:8686/ping"
check_service "Prowlarr" "http://localhost:9696/ping"
check_service "Bazarr" "http://localhost:6767/api/system/status" 401

echo ""
echo "Checking Download Clients:"
check_service "qBittorrent" "http://localhost:8081"
check_service "SABnzbd" "http://localhost:8082/api?mode=version"

echo ""
echo "Checking Additional Services:"
check_service "Overseerr" "http://localhost:5055/api/v1/status"
check_service "Tautulli" "http://localhost:8181/status"
check_service "Portainer" "http://localhost:9000/api/status"

echo ""
echo "======================================"
echo "Health check complete"
echo "======================================"
EOF
    
    chmod +x scripts/health-check-all.sh
    print_success "Created health check script"
}

# Function to create backup script
create_backup_script() {
    print_section "Creating Backup Script"
    
    cat > scripts/backup-configs.sh << 'EOF'
#!/bin/bash

# Backup script for media server configurations

set -euo pipefail

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Starting backup to $BACKUP_DIR..."

# Backup configurations
echo "Backing up configurations..."
tar -czf "$BACKUP_DIR/configs.tar.gz" config/

# Backup environment file
cp .env "$BACKUP_DIR/.env"

# Backup compose files
cp docker-compose*.yml "$BACKUP_DIR/"

# Create backup info
cat > "$BACKUP_DIR/backup_info.txt" << EOL
Backup created: $(date)
Docker version: $(docker --version)
Compose version: $(docker-compose --version || docker compose version)
Services running: $(docker ps --format "table {{.Names}}" | tail -n +2 | wc -l)
EOL

echo "Backup completed successfully!"
echo "Location: $BACKUP_DIR"

# Clean old backups (keep last 7)
echo "Cleaning old backups..."
ls -t backups/ | tail -n +8 | xargs -I {} rm -rf backups/{}

echo "Backup process finished!"
EOF
    
    chmod +x scripts/backup-configs.sh
    print_success "Created backup script"
}

# Function to create update script
create_update_script() {
    print_section "Creating Update Script"
    
    cat > scripts/update-containers.sh << 'EOF'
#!/bin/bash

# Update script for media server containers

set -euo pipefail

echo "======================================"
echo "Media Server Update Process"
echo "======================================"
echo ""

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose-production.yml pull

# Recreate containers with new images
echo ""
echo "Recreating containers..."
docker-compose -f docker-compose-production.yml up -d

# Clean up old images
echo ""
echo "Cleaning up old images..."
docker image prune -f

echo ""
echo "Update completed!"
echo ""

# Show container status
docker-compose -f docker-compose-production.yml ps
EOF
    
    chmod +x scripts/update-containers.sh
    print_success "Created update script"
}

# Function to display next steps
display_next_steps() {
    print_section "Setup Complete!"
    
    cat << EOF

${GREEN}✨ Your enhanced media server is ready to deploy!${NC}

${CYAN}Next Steps:${NC}

1. ${YELLOW}Review and edit the .env file${NC} to ensure all settings are correct

2. ${YELLOW}Update Authelia users${NC} in config/authelia/users_database.yml

3. ${YELLOW}Start the services:${NC}
   ${BLUE}docker-compose -f docker-compose-production.yml up -d${NC}

4. ${YELLOW}Access your services:${NC}
   - Homepage: ${BLUE}https://home.${DOMAIN}${NC}
   - Jellyfin: ${BLUE}https://jellyfin.${DOMAIN}${NC}
   - Requests: ${BLUE}https://requests.${DOMAIN}${NC}
   - Traefik: ${BLUE}https://traefik.${DOMAIN}${NC}

5. ${YELLOW}Configure services:${NC}
   - Set up Prowlarr indexers
   - Connect Sonarr/Radarr to Prowlarr
   - Configure Jellyfin libraries
   - Set up Overseerr with Jellyfin

${CYAN}Useful Commands:${NC}
- Check logs: ${BLUE}docker-compose -f docker-compose-production.yml logs -f [service]${NC}
- Health check: ${BLUE}./scripts/health-check-all.sh${NC}
- Backup: ${BLUE}./scripts/backup-configs.sh${NC}
- Update: ${BLUE}./scripts/update-containers.sh${NC}

${CYAN}Security Notes:${NC}
- Admin credentials are in ${BLUE}secrets/traefik_admin_credentials.txt${NC}
- All passwords are in ${BLUE}.env${NC} file
- SSL certificates will be auto-generated via Cloudflare

${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}🎉 Happy streaming!${NC}

EOF
}

# Main execution
main() {
    echo ""
    print_info "Starting enhanced media server setup..."
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Create directory structure
    create_directories
    
    # Create configuration files
    create_env_file
    create_authelia_config
    create_prometheus_config
    create_grafana_config
    create_homepage_config
    create_postgres_init
    create_security_files
    
    # Create utility scripts
    mkdir -p scripts
    create_health_check_script
    create_backup_script
    create_update_script
    
    # Display completion message
    display_next_steps
}

# Run main function
main "$@"