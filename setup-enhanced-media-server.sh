#!/bin/bash

# Enhanced Media Server Setup Script 2025
# Implements security hardening, hardware transcoding, and performance optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root!"
   exit 1
fi

print_header "Enhanced Media Server Setup 2025"

# Check Docker installation
print_header "Checking Prerequisites"

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker is installed"

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_success "Docker Compose is installed"

# Create directory structure
print_header "Creating Directory Structure"

directories=(
    "config/traefik/dynamic"
    "config/authelia"
    "config/nginx"
    "config/prometheus"
    "config/grafana/provisioning/dashboards"
    "config/grafana/provisioning/datasources"
    "homepage-config"
    "data/media/movies"
    "data/media/tv"
    "data/media/music"
    "data/downloads/complete"
    "data/downloads/incomplete"
    "backups"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_success "Created $dir"
done

# Generate secrets
print_header "Generating Secure Secrets"

generate_secret() {
    openssl rand -hex 32
}

# Check if .env exists
if [ -f ".env" ]; then
    print_warning ".env file already exists. Backing up to .env.backup"
    cp .env .env.backup
fi

# Create .env file from template
if [ -f "env.template" ]; then
    cp env.template .env
    print_success "Created .env file from template"
else
    print_error "env.template not found!"
    exit 1
fi

# Generate secrets and update .env
print_header "Configuring Environment Variables"

AUTHELIA_JWT_SECRET=$(generate_secret)
AUTHELIA_SESSION_SECRET=$(generate_secret)
AUTHELIA_STORAGE_ENCRYPTION_KEY=$(generate_secret)

# Update .env with generated secrets
sed -i.bak "s/AUTHELIA_JWT_SECRET=/AUTHELIA_JWT_SECRET=$AUTHELIA_JWT_SECRET/" .env
sed -i.bak "s/AUTHELIA_SESSION_SECRET=/AUTHELIA_SESSION_SECRET=$AUTHELIA_SESSION_SECRET/" .env
sed -i.bak "s/AUTHELIA_STORAGE_ENCRYPTION_KEY=/AUTHELIA_STORAGE_ENCRYPTION_KEY=$AUTHELIA_STORAGE_ENCRYPTION_KEY/" .env

print_success "Generated secure secrets"

# Create Traefik configuration
print_header "Creating Traefik Configuration"

cat > config/traefik/traefik.yml << 'EOF'
api:
  dashboard: true
  debug: true

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
          permanent: true
  websecure:
    address: ":443"
    http:
      tls:
        certResolver: cloudflare
        domains:
          - main: "${DOMAIN}"
            sans:
              - "*.${DOMAIN}"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
  file:
    directory: "/dynamic"
    watch: true

certificatesResolvers:
  cloudflare:
    acme:
      email: "${CLOUDFLARE_EMAIL}"
      storage: "/letsencrypt/acme.json"
      dnsChallenge:
        provider: cloudflare
        resolvers:
          - "1.1.1.1:53"
          - "1.0.0.1:53"
EOF

print_success "Created Traefik configuration"

# Create Authelia configuration
print_header "Creating Authelia Configuration"

cat > config/authelia/configuration.yml << 'EOF'
server:
  host: 0.0.0.0
  port: 9091

log:
  level: info

theme: dark

jwt_secret: ${AUTHELIA_JWT_SECRET}

default_redirection_url: https://${DOMAIN}

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
    - domain: ${DOMAIN}
      policy: bypass
    - domain: "*.${DOMAIN}"
      policy: two_factor

session:
  name: authelia_session
  secret: ${AUTHELIA_SESSION_SECRET}
  expiration: 3600
  inactivity: 300
  domain: ${DOMAIN}

regulation:
  max_retries: 3
  find_time: 120
  ban_time: 300

storage:
  encryption_key: ${AUTHELIA_STORAGE_ENCRYPTION_KEY}
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt
EOF

print_success "Created Authelia configuration"

# Create default user for Authelia
print_header "Creating Default Admin User"

read -p "Enter admin username (default: admin): " ADMIN_USER
ADMIN_USER=${ADMIN_USER:-admin}

read -s -p "Enter admin password: " ADMIN_PASSWORD
echo

# Generate password hash
ADMIN_PASSWORD_HASH=$(docker run --rm authelia/authelia:latest authelia hash-password "$ADMIN_PASSWORD" 2>/dev/null | grep "Password hash:" | cut -d' ' -f3-)

cat > config/authelia/users_database.yml << EOF
users:
  $ADMIN_USER:
    displayname: "Administrator"
    password: "$ADMIN_PASSWORD_HASH"
    email: admin@${DOMAIN}
    groups:
      - admins
      - dev
EOF

print_success "Created admin user"

# Create Nginx cache configuration
print_header "Creating Nginx Cache Configuration"

cat > config/nginx/nginx-cache.conf << 'EOF'
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Caching
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=media_cache:100m max_size=10g inactive=30d use_temp_path=off;

    # Logging
    access_log /var/log/nginx/access.log;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    gzip on;
    gzip_disable "msie6";
    gzip_vary on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    server {
        listen 80;
        server_name cache.${DOMAIN};

        location / {
            proxy_pass http://jellyfin:8096;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Caching for static content
            location ~* \.(jpg|jpeg|png|gif|ico|css|js|mp4|mkv|avi|webm)$ {
                proxy_cache media_cache;
                proxy_cache_valid 200 30d;
                proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
                proxy_cache_background_update on;
                proxy_cache_lock on;
                add_header X-Cache-Status $upstream_cache_status;
            }
        }
    }
}
EOF

print_success "Created Nginx cache configuration"

# Create Prometheus configuration
print_header "Creating Prometheus Configuration"

cat > config/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node_exporter:9100']

  - job_name: 'docker'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik:8080']
EOF

cat > config/prometheus/alert_rules.yml << 'EOF'
groups:
  - name: media_server_alerts
    rules:
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High CPU usage detected
          description: "CPU usage is above 80% (current value: {{ $value }}%)"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High memory usage detected
          description: "Memory usage is above 85% (current value: {{ $value }}%)"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: Low disk space
          description: "Disk space is below 10% (current value: {{ $value }}%)"
EOF

print_success "Created Prometheus configuration"

# Create Homepage configuration
print_header "Creating Homepage Configuration"

cat > homepage-config/settings.yaml << 'EOF'
title: Media Server 2025
theme: dark
color: blue
target: _blank
layout:
  Media:
    style: row
    columns: 4
  Services:
    style: row
    columns: 3
  Monitoring:
    style: row
    columns: 3
EOF

cat > homepage-config/services.yaml << 'EOF'
- Media:
    - Jellyfin:
        href: https://jellyfin.${DOMAIN}
        description: Media streaming server
        icon: jellyfin.png
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: ${JELLYFIN_API_KEY}

    - Overseerr:
        href: https://requests.${DOMAIN}
        description: Media request management
        icon: overseerr.png

- Services:
    - Sonarr:
        href: https://sonarr.${DOMAIN}
        description: TV show management
        icon: sonarr.png
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: ${SONARR_API_KEY}

    - Radarr:
        href: https://radarr.${DOMAIN}
        description: Movie management
        icon: radarr.png
        widget:
          type: radarr
          url: http://radarr:7878
          key: ${RADARR_API_KEY}

    - Prowlarr:
        href: https://prowlarr.${DOMAIN}
        description: Indexer management
        icon: prowlarr.png

- Monitoring:
    - Grafana:
        href: https://grafana.${DOMAIN}
        description: Metrics dashboard
        icon: grafana.png

    - Prometheus:
        href: https://prometheus.${DOMAIN}
        description: Metrics collection
        icon: prometheus.png
EOF

print_success "Created Homepage configuration"

# Check for GPU support
print_header "Checking Hardware Acceleration Support"

if [ -e /dev/dri ]; then
    print_success "GPU device found at /dev/dri"
    print_warning "Make sure your user is in the 'video' and 'render' groups:"
    echo "  sudo usermod -aG video,render $USER"
else
    print_warning "No GPU device found. Hardware transcoding will not be available."
fi

# Final instructions
print_header "Setup Complete!"

echo -e "${GREEN}Your enhanced media server is ready to deploy!${NC}"
echo
echo "Next steps:"
echo "1. Edit the .env file and update:"
echo "   - DOMAIN (your domain name)"
echo "   - CLOUDFLARE_EMAIL and CLOUDFLARE_API_TOKEN"
echo "   - MEDIA_PATH and DOWNLOADS_PATH"
echo
echo "2. Start the services:"
echo "   docker-compose -f docker-compose-enhanced-2025.yml up -d"
echo
echo "3. Access the services:"
echo "   - Homepage: https://your-domain.com"
echo "   - Jellyfin: https://jellyfin.your-domain.com"
echo "   - Overseerr: https://requests.your-domain.com"
echo
echo "4. Get API keys from each service and update .env"
echo
echo "5. View the holographic dashboard:"
echo "   Open holographic-media-dashboard.html in your browser"
echo
print_warning "Remember to configure your DNS records to point to this server!"

# Create start script
cat > start-enhanced-server.sh << 'EOF'
#!/bin/bash
docker-compose -f docker-compose-enhanced-2025.yml up -d
echo "Media server starting... Check status with: docker-compose -f docker-compose-enhanced-2025.yml ps"
EOF
chmod +x start-enhanced-server.sh

# Create stop script
cat > stop-enhanced-server.sh << 'EOF'
#!/bin/bash
docker-compose -f docker-compose-enhanced-2025.yml down
echo "Media server stopped."
EOF
chmod +x stop-enhanced-server.sh

print_success "Created helper scripts: start-enhanced-server.sh and stop-enhanced-server.sh"