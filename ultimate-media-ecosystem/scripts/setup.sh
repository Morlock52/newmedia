#!/bin/bash
# Ultimate Media Server Ecosystem Setup Script
# Version: 2025.1.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root!"
   exit 1
fi

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables if .env exists
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        log_info "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed!"
        log_info "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check if user is in docker group
    if ! groups | grep -q docker; then
        log_warning "Current user is not in the docker group!"
        log_info "Run: sudo usermod -aG docker $USER"
        log_info "Then log out and back in."
    fi
    
    log_success "Prerequisites check passed!"
}

# Function to create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    # Media directories
    mkdir -p "$PROJECT_ROOT/media"/{movies,tvshows,music,audiobooks,books,comics,manga,photos,podcasts}
    
    # Download directories
    mkdir -p "$PROJECT_ROOT/downloads"/{torrents,usenet,incomplete}
    
    # Config directories
    mkdir -p "$PROJECT_ROOT/configs"/{authelia,traefik,homepage,prometheus,grafana}
    
    # Backup directory
    mkdir -p "$PROJECT_ROOT/backups"
    
    # Secrets directory
    mkdir -p "$PROJECT_ROOT/secrets"
    chmod 700 "$PROJECT_ROOT/secrets"
    
    # Transcode cache
    mkdir -p "$PROJECT_ROOT/transcode"
    
    log_success "Directory structure created!"
}

# Function to generate secrets
generate_secrets() {
    log_info "Generating secrets..."
    
    SECRETS_DIR="$PROJECT_ROOT/secrets"
    
    # Generate Authelia secrets
    openssl rand -base64 32 > "$SECRETS_DIR/authelia_jwt_secret.txt"
    openssl rand -base64 32 > "$SECRETS_DIR/authelia_session_secret.txt"
    openssl rand -base64 32 > "$SECRETS_DIR/authelia_encryption_key.txt"
    
    # Generate database passwords
    openssl rand -base64 24 > "$SECRETS_DIR/mysql_root_password.txt"
    openssl rand -base64 24 > "$SECRETS_DIR/mysql_password.txt"
    openssl rand -base64 24 > "$SECRETS_DIR/postgres_password.txt"
    openssl rand -base64 24 > "$SECRETS_DIR/redis_password.txt"
    
    # Generate API keys
    openssl rand -hex 32 > "$SECRETS_DIR/jellyfin_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/sonarr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/radarr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/lidarr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/readarr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/prowlarr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/bazarr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/overseerr_api_key.txt"
    openssl rand -hex 32 > "$SECRETS_DIR/tautulli_api_key.txt"
    
    # Set proper permissions
    chmod 600 "$SECRETS_DIR"/*.txt
    
    log_success "Secrets generated!"
}

# Function to setup environment file
setup_environment() {
    log_info "Setting up environment file..."
    
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        cp "$PROJECT_ROOT/.env.template" "$PROJECT_ROOT/.env"
        
        # Get user input for critical settings
        read -p "Enter your domain (e.g., media.example.com): " DOMAIN
        read -p "Enter your email address: " EMAIL
        read -p "Enter your timezone (e.g., America/New_York): " TZ
        
        # Update .env file
        sed -i "s/DOMAIN=.*/DOMAIN=$DOMAIN/g" "$PROJECT_ROOT/.env"
        sed -i "s/EMAIL=.*/EMAIL=$EMAIL/g" "$PROJECT_ROOT/.env"
        sed -i "s/TZ=.*/TZ=$TZ/g" "$PROJECT_ROOT/.env"
        
        # Set PUID and PGID
        sed -i "s/PUID=.*/PUID=$(id -u)/g" "$PROJECT_ROOT/.env"
        sed -i "s/PGID=.*/PGID=$(id -g)/g" "$PROJECT_ROOT/.env"
        
        log_success "Environment file created!"
    else
        log_warning "Environment file already exists, skipping..."
    fi
}

# Function to configure Authelia
configure_authelia() {
    log_info "Configuring Authelia..."
    
    # Create users database
    cat > "$PROJECT_ROOT/configs/authelia/users_database.yml" <<EOF
---
users:
  admin:
    displayname: "Admin User"
    password: "\$argon2id\$v=19\$m=65536,t=3,p=4\$YnBwTXNuS2QxaFpWR1VYVA\$bQ2I5nfmV3nqq7n0BsVQe3BR0dJqJLDHfjNkS6jLYCA"  # password: admin (CHANGE THIS!)
    email: admin@example.com
    groups:
      - admins
      - users
      - media
  
  user:
    displayname: "Regular User"
    password: "\$argon2id\$v=19\$m=65536,t=3,p=4\$YnBwTXNuS2QxaFpWR1VYVA\$bQ2I5nfmV3nqq7n0BsVQe3BR0dJqJLDHfjNkS6jLYCA"  # password: user (CHANGE THIS!)
    email: user@example.com
    groups:
      - users
  
  media:
    displayname: "Media Manager"
    password: "\$argon2id\$v=19\$m=65536,t=3,p=4\$YnBwTXNuS2QxaFpWR1VYVA\$bQ2I5nfmV3nqq7n0BsVQe3BR0dJqJLDHfjNkS6jLYCA"  # password: media (CHANGE THIS!)
    email: media@example.com
    groups:
      - users
      - media
...
EOF
    
    log_warning "Default passwords set! Please change them immediately!"
    log_info "Use 'docker exec -it authelia authelia hash-password' to generate new passwords"
}

# Function to configure Traefik
configure_traefik() {
    log_info "Configuring Traefik..."
    
    # Create dynamic configuration
    cat > "$PROJECT_ROOT/configs/traefik/dynamic.yml" <<EOF
http:
  middlewares:
    security-headers:
      headers:
        frameDeny: true
        sslRedirect: true
        browserXssFilter: true
        contentTypeNosniff: true
        forceSTSHeader: true
        stsIncludeSubdomains: true
        stsPreload: true
        stsSeconds: 31536000
        customFrameOptionsValue: "SAMEORIGIN"
        customRequestHeaders:
          X-Forwarded-Proto: https
    
    rate-limit:
      rateLimit:
        average: 100
        burst: 50
    
    compression:
      compress: {}

  routers:
    api:
      rule: Host(\`traefik.${DOMAIN}\`)
      service: api@internal
      middlewares:
        - authelia@docker
        - security-headers
      tls:
        certResolver: letsencrypt

tls:
  options:
    default:
      minVersion: VersionTLS12
      cipherSuites:
        - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
        - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
        - TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305
        - TLS_AES_128_GCM_SHA256
        - TLS_AES_256_GCM_SHA384
        - TLS_CHACHA20_POLY1305_SHA256
      curvePreferences:
        - CurveP521
        - CurveP384
      sniStrict: true
EOF
}

# Function to configure Homepage dashboard
configure_homepage() {
    log_info "Configuring Homepage dashboard..."
    
    # Create services configuration
    cat > "$PROJECT_ROOT/configs/homepage/services.yaml" <<EOF
---
- Media Servers:
    - Jellyfin:
        href: https://jellyfin.${DOMAIN}
        description: Primary media server
        icon: jellyfin.svg
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: {{HOMEPAGE_VAR_JELLYFIN_API_KEY}}
    
    - Emby:
        href: https://emby.${DOMAIN}
        description: Alternative media server
        icon: emby.svg
        widget:
          type: emby
          url: http://emby:8096
          key: {{HOMEPAGE_VAR_EMBY_API_KEY}}

- Media Management:
    - Sonarr:
        href: https://sonarr.${DOMAIN}
        description: TV series management
        icon: sonarr.svg
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: {{HOMEPAGE_VAR_SONARR_API_KEY}}
    
    - Radarr:
        href: https://radarr.${DOMAIN}
        description: Movie management
        icon: radarr.svg
        widget:
          type: radarr
          url: http://radarr:7878
          key: {{HOMEPAGE_VAR_RADARR_API_KEY}}
    
    - Lidarr:
        href: https://lidarr.${DOMAIN}
        description: Music management
        icon: lidarr.svg
        widget:
          type: lidarr
          url: http://lidarr:8686
          key: {{HOMEPAGE_VAR_LIDARR_API_KEY}}
    
    - Readarr:
        href: https://readarr.${DOMAIN}
        description: Book/audiobook management
        icon: readarr.svg
        widget:
          type: readarr
          url: http://readarr:8787
          key: {{HOMEPAGE_VAR_READARR_API_KEY}}

- Downloads:
    - qBittorrent:
        href: https://qbittorrent.${DOMAIN}
        description: Torrent client
        icon: qbittorrent.svg
        widget:
          type: qbittorrent
          url: http://gluetun:8080
          username: admin
          password: adminadmin
    
    - SABnzbd:
        href: https://sabnzbd.${DOMAIN}
        description: Usenet client
        icon: sabnzbd.svg
        widget:
          type: sabnzbd
          url: http://sabnzbd:8080
          key: {{HOMEPAGE_VAR_SABNZBD_API_KEY}}

- Requests:
    - Overseerr:
        href: https://requests.${DOMAIN}
        description: Media requests
        icon: overseerr.svg
        widget:
          type: overseerr
          url: http://overseerr:5055
          key: {{HOMEPAGE_VAR_OVERSEERR_API_KEY}}

- Monitoring:
    - Tautulli:
        href: https://analytics.${DOMAIN}
        description: Media analytics
        icon: tautulli.svg
        widget:
          type: tautulli
          url: http://tautulli:8181
          key: {{HOMEPAGE_VAR_TAUTULLI_API_KEY}}
    
    - Grafana:
        href: https://grafana.${DOMAIN}
        description: System monitoring
        icon: grafana.svg
    
    - Uptime Kuma:
        href: https://uptime.${DOMAIN}
        description: Uptime monitoring
        icon: uptime-kuma.svg
        widget:
          type: uptimekuma
          url: http://uptime-kuma:3001
          slug: main

- Management:
    - Portainer:
        href: https://portainer.${DOMAIN}
        description: Container management
        icon: portainer.svg
    
    - Traefik:
        href: https://traefik.${DOMAIN}
        description: Reverse proxy
        icon: traefik.svg
    
    - Authelia:
        href: https://auth.${DOMAIN}
        description: Authentication
        icon: authelia.svg
EOF
}

# Function to configure monitoring
configure_monitoring() {
    log_info "Configuring monitoring stack..."
    
    # Prometheus configuration
    cat > "$PROJECT_ROOT/configs/prometheus/prometheus.yml" <<EOF
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
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik:8082']

  - job_name: 'authelia'
    static_configs:
      - targets: ['authelia:9959']

  - job_name: 'jellyfin'
    static_configs:
      - targets: ['jellyfin:8096']
    metrics_path: '/metrics'

  - job_name: 'sonarr'
    static_configs:
      - targets: ['sonarr:8989']
    metrics_path: '/metrics'

  - job_name: 'radarr'
    static_configs:
      - targets: ['radarr:7878']
    metrics_path: '/metrics'
EOF
}

# Function to start services
start_services() {
    log_info "Starting services..."
    
    cd "$PROJECT_ROOT"
    
    # Start infrastructure first
    docker-compose up -d traefik authelia redis_authelia
    
    log_info "Waiting for infrastructure to be ready..."
    sleep 30
    
    # Start databases
    docker-compose up -d mariadb postgres_immich redis_immich
    
    log_info "Waiting for databases to be ready..."
    sleep 20
    
    # Start media servers
    docker-compose up -d jellyfin emby
    
    # Start arr suite
    docker-compose up -d prowlarr sonarr radarr lidarr readarr bazarr
    
    # Start download clients
    docker-compose up -d gluetun qbittorrent sabnzbd
    
    # Start remaining services
    docker-compose up -d
    
    log_success "All services started!"
}

# Function to display post-setup instructions
post_setup_instructions() {
    log_success "Setup complete!"
    echo
    echo "===== NEXT STEPS ====="
    echo
    echo "1. Access your dashboard at: https://${DOMAIN}"
    echo "2. Default Authelia credentials:"
    echo "   - Username: admin"
    echo "   - Password: admin"
    echo "   ⚠️  CHANGE THESE IMMEDIATELY!"
    echo
    echo "3. Configure your services:"
    echo "   - Jellyfin: https://jellyfin.${DOMAIN}"
    echo "   - Sonarr: https://sonarr.${DOMAIN}"
    echo "   - Radarr: https://radarr.${DOMAIN}"
    echo "   - Prowlarr: https://prowlarr.${DOMAIN}"
    echo
    echo "4. Setup VPN:"
    echo "   - Add your VPN credentials to .env file"
    echo "   - Restart gluetun container"
    echo
    echo "5. Configure indexers in Prowlarr"
    echo "6. Connect Prowlarr to Sonarr/Radarr/Lidarr"
    echo "7. Configure download clients"
    echo
    echo "===== SECURITY REMINDERS ====="
    echo "- Change all default passwords"
    echo "- Enable 2FA in Authelia"
    echo "- Review firewall rules"
    echo "- Setup backup strategy"
    echo
    echo "Documentation: $PROJECT_ROOT/docs/"
    echo "Logs: docker-compose logs -f [service-name]"
}

# Main setup flow
main() {
    echo "===== Ultimate Media Server Ecosystem Setup ====="
    echo
    
    check_prerequisites
    create_directories
    
    if [[ ! -d "$PROJECT_ROOT/secrets" ]] || [[ -z "$(ls -A $PROJECT_ROOT/secrets)" ]]; then
        generate_secrets
    fi
    
    setup_environment
    configure_authelia
    configure_traefik
    configure_homepage
    configure_monitoring
    
    read -p "Start all services now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_services
        post_setup_instructions
    else
        log_info "Setup complete. Run 'docker-compose up -d' when ready."
    fi
}

# Run main function
main "$@"