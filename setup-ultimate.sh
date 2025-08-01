#!/bin/bash

# Ultimate Media Server Setup Script
# This script sets up the complete media server stack with all media types

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Header
print_color "$BLUE" "========================================"
print_color "$BLUE" "Ultimate Media Server Setup Script"
print_color "$BLUE" "Supports ALL Media Types"
print_color "$BLUE" "========================================"
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_color "$RED" "This script should not be run as root!"
   exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    print_color "$RED" "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if ! docker compose version &> /dev/null; then
    print_color "$RED" "Docker Compose is not installed. Please install Docker Compose v2."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_color "$YELLOW" "Creating .env file from template..."
    cp .env.example .env
    print_color "$GREEN" "✓ .env file created"
    print_color "$YELLOW" "⚠️  Please edit .env file with your configuration before continuing"
    echo
    read -p "Press Enter after editing .env file..."
fi

# Source environment variables
source .env

# Create directory structure
print_color "$BLUE" "Creating directory structure..."

# Main directories
directories=(
    # Config directories
    "config/authelia"
    "config/traefik/dynamic"
    "config/jellyfin"
    "config/navidrome"
    "config/audiobookshelf"
    "config/calibre-web"
    "config/kavita"
    "config/immich"
    "config/qbittorrent"
    "config/sabnzbd"
    "config/prowlarr"
    "config/sonarr"
    "config/radarr"
    "config/lidarr"
    "config/readarr"
    "config/bazarr"
    "config/overseerr"
    "config/fileflows"
    "config/podgrab"
    "config/homepage"
    "config/portainer"
    "config/tautulli"
    "config/prometheus"
    "config/grafana/provisioning/dashboards"
    "config/grafana/provisioning/datasources"
    "config/duplicati"
    "config/filebrowser"
    "config/gluetun"
    
    # Data directories
    "data/media/movies"
    "data/media/tv"
    "data/media/music"
    "data/media/audiobooks"
    "data/media/books"
    "data/media/comics"
    "data/media/manga"
    "data/media/photos"
    "data/media/podcasts"
    "data/downloads/complete"
    "data/downloads/incomplete"
    "data/downloads/usenet"
    
    # Cache and temp directories
    "cache/jellyfin"
    "transcodes"
    "temp/fileflows"
    "metadata/audiobookshelf"
    
    # Backup directory
    "backups"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_color "$GREEN" "✓ Created $dir"
done

# Set permissions
print_color "$BLUE" "Setting permissions..."
sudo chown -R $PUID:$PGID config data cache transcodes temp metadata backups
sudo chmod -R 755 config data cache transcodes temp metadata backups

# Create Traefik configuration
print_color "$BLUE" "Creating Traefik configuration..."
cat > config/traefik/traefik.yml << EOF
api:
  dashboard: true
  debug: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
  file:
    directory: /dynamic
    watch: true

certificatesResolvers:
  cloudflare:
    acme:
      email: ${CLOUDFLARE_EMAIL}
      storage: acme.json
      dnsChallenge:
        provider: cloudflare
        resolvers:
          - "1.1.1.1:53"
          - "1.0.0.1:53"
EOF

# Create dynamic configuration
cat > config/traefik/dynamic/middlewares.yml << EOF
http:
  middlewares:
    auth:
      forwardAuth:
        address: "http://authelia:9091/api/verify?rd=https://auth.${DOMAIN}/"
        trustForwardHeader: true
        authResponseHeaders:
          - "Remote-User"
          - "Remote-Groups"
          - "Remote-Name"
          - "Remote-Email"
    
    security-headers:
      headers:
        frameDeny: true
        browserXssFilter: true
        contentTypeNosniff: true
        forceSTSHeader: true
        stsIncludeSubdomains: true
        stsPreload: true
        stsSeconds: 315360000
        customFrameOptionsValue: "SAMEORIGIN"
        customResponseHeaders:
          X-Robots-Tag: "noindex,nofollow,nosnippet,noarchive,notranslate,noimageindex"
    
    rate-limit:
      rateLimit:
        average: 100
        burst: 50
    
    compression:
      compress: true
EOF

# Create Authelia configuration
print_color "$BLUE" "Creating Authelia configuration..."
cat > config/authelia/configuration.yml << EOF
theme: dark
default_redirection_url: https://home.${DOMAIN}/

server:
  host: 0.0.0.0
  port: 9091

log:
  level: info

totp:
  issuer: Media Server
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
    - domain: "requests.${DOMAIN}"
      policy: bypass
    
    # Admin access
    - domain: "*.${DOMAIN}"
      policy: two_factor

session:
  name: authelia_session
  domain: ${DOMAIN}
  same_site: lax
  secret: unsecure_session_secret
  expiration: 1h
  inactivity: 5m
  remember_me_duration: 1M

regulation:
  max_retries: 3
  find_time: 2m
  ban_time: 5m

storage:
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt
EOF

# Create users database
print_color "$BLUE" "Creating default users..."
cat > config/authelia/users_database.yml << EOF
users:
  admin:
    displayname: "Admin User"
    password: "\$argon2id\$v=19\$m=65536,t=1,p=8\$c29tZXNhbHQ\$YTrwgpDZLbHqkNcZKFE4JiSPKQdGXPyJ7Wv2/Fv8VqY"
    email: admin@example.com
    groups:
      - admins
      - users
EOF

# Create Homepage configuration
print_color "$BLUE" "Creating Homepage dashboard configuration..."
cat > config/homepage/settings.yaml << EOF
title: Media Server
background: 
  image: https://images.unsplash.com/photo-1518676590629-3dcbd9c5a5c9
  blur: sm
  brightness: 50
  opacity: 100
theme: dark
color: slate
language: en
layout:
  Media Management:
    style: row
    columns: 4
  Download Clients:
    style: row
    columns: 3
  Media Servers:
    style: row
    columns: 4
  Tools:
    style: row
    columns: 4
EOF

cat > config/homepage/services.yaml << EOF
- Media Management:
    - Sonarr:
        href: http://sonarr:8989
        icon: sonarr.png
        description: TV Shows
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: 
    - Radarr:
        href: http://radarr:7878
        icon: radarr.png
        description: Movies
        widget:
          type: radarr
          url: http://radarr:7878
          key: 
    - Lidarr:
        href: http://lidarr:8686
        icon: lidarr.png
        description: Music
        widget:
          type: lidarr
          url: http://lidarr:8686
          key: 
    - Readarr:
        href: http://readarr:8787
        icon: readarr.png
        description: Books

- Download Clients:
    - qBittorrent:
        href: http://qbittorrent:8080
        icon: qbittorrent.png
        description: Torrent Client
        widget:
          type: qbittorrent
          url: http://qbittorrent:8080
          username: admin
          password: adminadmin
    - SABnzbd:
        href: http://sabnzbd:8080
        icon: sabnzbd.png
        description: Usenet Client
    - Prowlarr:
        href: http://prowlarr:9696
        icon: prowlarr.png
        description: Indexer Manager

- Media Servers:
    - Jellyfin:
        href: http://jellyfin:8096
        icon: jellyfin.png
        description: Media Server
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: 
    - Navidrome:
        href: http://navidrome:4533
        icon: navidrome.png
        description: Music Server
    - AudioBookshelf:
        href: http://audiobookshelf
        icon: audiobookshelf.png
        description: Audiobook Server
    - Kavita:
        href: http://kavita:5000
        icon: kavita.png
        description: Comics & Manga

- Tools:
    - Overseerr:
        href: http://overseerr:5055
        icon: overseerr.png
        description: Request Management
    - Bazarr:
        href: http://bazarr:6767
        icon: bazarr.png
        description: Subtitles
    - FileFlows:
        href: http://fileflows:5000
        icon: fileflows.png
        description: Media Processing
    - Tautulli:
        href: http://tautulli:8181
        icon: tautulli.png
        description: Analytics
EOF

# Create Prometheus configuration
print_color "$BLUE" "Creating Prometheus configuration..."
cat > config/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'docker'
    static_configs:
      - targets: ['docker-exporter:9417']
EOF

# Create Grafana datasource
print_color "$BLUE" "Creating Grafana configuration..."
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

# Create acme.json for Let's Encrypt
touch config/traefik/acme.json
chmod 600 config/traefik/acme.json

# Create filebrowser config
cat > config/filebrowser/settings.json << EOF
{
  "port": 80,
  "baseURL": "",
  "address": "",
  "log": "stdout",
  "database": "/database.db",
  "root": "/srv"
}
EOF

# Pull all images
print_color "$BLUE" "Pulling Docker images (this may take a while)..."
docker compose -f docker-compose-ultimate.yml pull

# Start services in order
print_color "$BLUE" "Starting services..."

# Start infrastructure first
print_color "$YELLOW" "Starting infrastructure services..."
docker compose -f docker-compose-ultimate.yml up -d traefik authelia gluetun

# Wait for infrastructure
sleep 10

# Start databases
print_color "$YELLOW" "Starting database services..."
docker compose -f docker-compose-ultimate.yml up -d immich-postgres immich-redis

# Wait for databases
sleep 10

# Start main services
print_color "$YELLOW" "Starting main services..."
docker compose -f docker-compose-ultimate.yml up -d

# Wait for services to be ready
print_color "$YELLOW" "Waiting for services to be ready..."
sleep 30

# Show status
print_color "$BLUE" "Checking service status..."
docker compose -f docker-compose-ultimate.yml ps

# Display access information
print_color "$GREEN" "========================================"
print_color "$GREEN" "Setup Complete!"
print_color "$GREEN" "========================================"
echo
print_color "$BLUE" "Access your services at:"
echo
print_color "$YELLOW" "Dashboard: http://localhost:3000"
print_color "$YELLOW" "Jellyfin: http://localhost:8096"
print_color "$YELLOW" "Navidrome: http://localhost:4533"
print_color "$YELLOW" "AudioBookshelf: http://localhost:13378"
print_color "$YELLOW" "Immich: http://localhost:2283"
print_color "$YELLOW" "Overseerr: http://localhost:5055"
echo
print_color "$BLUE" "Management Tools:"
print_color "$YELLOW" "Portainer: http://localhost:9000"
print_color "$YELLOW" "Traefik: http://localhost:8080"
echo
print_color "$RED" "⚠️  Important Notes:"
print_color "$YELLOW" "1. Default Authelia password is 'authelia' - CHANGE THIS!"
print_color "$YELLOW" "2. Configure your VPN settings in .env file"
print_color "$YELLOW" "3. Set up your domain and SSL certificates"
print_color "$YELLOW" "4. Configure API keys for each service"
echo
print_color "$GREEN" "Run 'docker compose -f docker-compose-ultimate.yml logs -f' to view logs"