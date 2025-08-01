#!/bin/bash
# Media Server Stack Setup Script
# ================================
# This script sets up the complete media server stack

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Functions
print_header() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
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

# Check requirements
check_requirements() {
    print_header "Checking Requirements"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker is installed"
    else
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose is installed"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then 
        print_warning "Running as root. This is not recommended."
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    # Base directories
    dirs=(
        "config"
        "data/media/movies"
        "data/media/tv"
        "data/media/music"
        "data/media/audiobooks"
        "data/media/books"
        "data/media/podcasts"
        "data/media/comics"
        "data/media/manga"
        "data/media/photos"
        "data/downloads/torrents"
        "data/downloads/usenet"
        "data/downloads/incomplete"
        "backups"
        "secrets"
        "logs"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created $dir"
        else
            print_warning "$dir already exists"
        fi
    done
    
    # Set permissions
    chmod 755 data config
    chmod 700 secrets
}

# Generate secrets
generate_secrets() {
    print_header "Generating Secrets"
    
    # Create secrets directory
    mkdir -p secrets
    
    # Generate secrets if they don't exist
    secrets=(
        "jwt_secret"
        "session_secret"
        "storage_encryption_key"
    )
    
    for secret in "${secrets[@]}"; do
        if [ ! -f "secrets/${secret}.txt" ]; then
            openssl rand -base64 32 > "secrets/${secret}.txt"
            chmod 600 "secrets/${secret}.txt"
            print_success "Generated ${secret}"
        else
            print_warning "${secret} already exists"
        fi
    done
}

# Setup environment file
setup_env() {
    print_header "Setting Up Environment"
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.production" ]; then
            cp .env.production .env
            print_success "Created .env from template"
            print_warning "Please edit .env with your actual values"
        else
            print_error ".env.production template not found"
            exit 1
        fi
    else
        print_warning ".env already exists"
    fi
}

# Create Traefik dynamic configuration
create_traefik_config() {
    print_header "Creating Traefik Configuration"
    
    # Create middlewares configuration
    cat > config/traefik/dynamic/middlewares.yml << 'EOF'
http:
  middlewares:
    # Security headers
    security-headers:
      headers:
        customResponseHeaders:
          X-Frame-Options: "SAMEORIGIN"
          X-Content-Type-Options: "nosniff"
          X-XSS-Protection: "1; mode=block"
          Referrer-Policy: "strict-origin-when-cross-origin"
          Permissions-Policy: "camera=(), microphone=(), geolocation=()"
        sslRedirect: true
        stsSeconds: 31536000
        stsIncludeSubdomains: true
        stsPreload: true
        forceSTSHeader: true

    # Rate limiting
    rate-limit:
      rateLimit:
        average: 100
        period: 1m
        burst: 50

    # Basic auth (backup)
    basic-auth:
      basicAuth:
        usersFile: /dynamic/users.txt
        realm: "Media Server"
        removeHeader: true

    # Chain for secured services
    secured:
      chain:
        middlewares:
          - security-headers
          - rate-limit
          - authelia@docker
EOF
    
    print_success "Created Traefik middlewares configuration"
}

# Create Homepage configuration
create_homepage_config() {
    print_header "Creating Homepage Configuration"
    
    mkdir -p config/homepage
    
    # Create settings.yaml
    cat > config/homepage/settings.yaml << 'EOF'
title: Media Server Dashboard

theme: dark
color: slate

layout:
  Media:
    style: row
    columns: 4
  Downloads:
    style: row
    columns: 3
  Management:
    style: row
    columns: 3
  Monitoring:
    style: row
    columns: 4

providers:
  openweathermap: openweathermapapikey
EOF
    
    # Create services.yaml
    cat > config/homepage/services.yaml << 'EOF'
- Media:
    - Jellyfin:
        icon: jellyfin.svg
        href: https://jellyfin.${DOMAIN}
        description: Media streaming
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: ${JELLYFIN_API_KEY}

    - Navidrome:
        icon: navidrome.svg
        href: https://music.${DOMAIN}
        description: Music streaming

    - AudioBookshelf:
        icon: audiobookshelf.svg
        href: https://audiobooks.${DOMAIN}
        description: Audiobook server

    - Immich:
        icon: immich.svg
        href: https://photos.${DOMAIN}
        description: Photo management

- Downloads:
    - qBittorrent:
        icon: qbittorrent.svg
        href: https://qbittorrent.${DOMAIN}
        description: Torrent client
        widget:
          type: qbittorrent
          url: http://gluetun:8080
          username: admin
          password: ${QBT_PASSWORD}

    - SABnzbd:
        icon: sabnzbd.svg
        href: https://sabnzbd.${DOMAIN}
        description: Usenet client

- Management:
    - Sonarr:
        icon: sonarr.svg
        href: https://sonarr.${DOMAIN}
        description: TV management
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: ${SONARR_API_KEY}

    - Radarr:
        icon: radarr.svg
        href: https://radarr.${DOMAIN}
        description: Movie management
        widget:
          type: radarr
          url: http://radarr:7878
          key: ${RADARR_API_KEY}

    - Overseerr:
        icon: overseerr.svg
        href: https://requests.${DOMAIN}
        description: Request management

- Monitoring:
    - Grafana:
        icon: grafana.svg
        href: https://grafana.${DOMAIN}
        description: Metrics dashboard

    - Prometheus:
        icon: prometheus.svg
        href: https://prometheus.${DOMAIN}
        description: Metrics collection

    - Portainer:
        icon: portainer.svg
        href: https://portainer.${DOMAIN}
        description: Container management
EOF
    
    print_success "Created Homepage configuration"
}

# Create docker-compose alias
create_compose_alias() {
    print_header "Creating Docker Compose Alias"
    
    cat > dc << 'EOF'
#!/bin/bash
# Docker Compose helper script

if docker compose version &> /dev/null; then
    docker compose -f docker-compose-optimized-2025.yml "$@"
else
    docker-compose -f docker-compose-optimized-2025.yml "$@"
fi
EOF
    
    chmod +x dc
    print_success "Created 'dc' helper script"
}

# Initialize PostgreSQL
init_postgres() {
    print_header "Creating PostgreSQL Init Script"
    
    mkdir -p config/postgres
    cat > config/postgres/init.sql << 'EOF'
-- Create databases for services
CREATE DATABASE IF NOT EXISTS authelia;
CREATE DATABASE IF NOT EXISTS immich;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE authelia TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE immich TO mediauser;
EOF
    
    print_success "Created PostgreSQL initialization script"
}

# Main setup function
main() {
    clear
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║   Media Server Stack Setup Script     ║"
    echo "║          Version 2025.1               ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    
    # Run setup steps
    check_requirements
    create_directories
    generate_secrets
    setup_env
    create_traefik_config
    create_homepage_config
    init_postgres
    create_compose_alias
    
    print_header "Setup Complete!"
    echo
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Review docker-compose-optimized-2025.yml"
    echo "3. Start the stack: ./dc up -d"
    echo "4. Check logs: ./dc logs -f"
    echo "5. Access services at https://home.yourdomain.com"
    echo
    print_warning "Remember to:"
    echo "- Set up your domain DNS records"
    echo "- Configure Cloudflare API token"
    echo "- Set strong passwords in .env"
    echo "- Configure VPN credentials if using"
    echo
}

# Run main function
main