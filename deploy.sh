#!/bin/bash

# Media Server Deployment Script
# Optimized for macOS and Linux

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if running on macOS
is_macos() {
    [[ "$OSTYPE" == "darwin"* ]]
}

# Create required directories
create_directories() {
    print_header "Creating Directory Structure"
    
    # Main directories
    directories=(
        "config"
        "media-data/movies"
        "media-data/tv"
        "media-data/music"
        "media-data/downloads/complete"
        "media-data/downloads/incomplete"
        "media-data/usenet/complete"
        "media-data/usenet/incomplete"
    )
    
    # Service config directories
    services=(
        "jellyfin" "prowlarr" "sonarr" "radarr" "lidarr" "bazarr"
        "qbittorrent" "sabnzbd" "overseerr" "tautulli" "homepage"
        "portainer" "prometheus" "grafana" "traefik"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created $dir"
    done
    
    for service in "${services[@]}"; do
        mkdir -p "config/$service"
        print_success "Created config/$service"
    done
}

# Check Docker installation
check_docker() {
    print_header "Checking Docker Installation"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running!"
        echo "Please start Docker Desktop"
        exit 1
    fi
    
    print_success "Docker is installed and running"
    docker --version
}

# Setup environment file
setup_env() {
    print_header "Setting Up Environment"
    
    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            print_warning "Created .env from template"
            print_warning "Please edit .env and update with your values"
            echo -e "\nRequired configuration:"
            echo "  - VPN credentials (if using torrents)"
            echo "  - Cloudflare API credentials (for SSL)"
            echo "  - Service passwords"
            read -p "Press Enter to continue after editing .env..."
        else
            print_error ".env.example not found!"
            exit 1
        fi
    else
        print_success ".env file exists"
    fi
}

# Create Prometheus configuration
create_prometheus_config() {
    print_header "Creating Prometheus Configuration"
    
    cat > config/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9090']
        
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
EOF
    
    print_success "Created Prometheus configuration"
}

# Create Homepage configuration
create_homepage_config() {
    print_header "Creating Homepage Configuration"
    
    # Create services.yaml
    cat > config/homepage/services.yaml << 'EOF'
---
- Media:
    - Jellyfin:
        href: http://localhost:8096
        icon: jellyfin.png
        description: Media Server
        
    - Overseerr:
        href: http://localhost:5055
        icon: overseerr.png
        description: Request Management

- Automation:
    - Sonarr:
        href: http://localhost:8989
        icon: sonarr.png
        description: TV Shows
        
    - Radarr:
        href: http://localhost:7878
        icon: radarr.png
        description: Movies
        
    - Lidarr:
        href: http://localhost:8686
        icon: lidarr.png
        description: Music
        
    - Bazarr:
        href: http://localhost:6767
        icon: bazarr.png
        description: Subtitles
        
    - Prowlarr:
        href: http://localhost:9696
        icon: prowlarr.png
        description: Indexers

- Downloads:
    - qBittorrent:
        href: http://localhost:8080
        icon: qbittorrent.png
        description: Torrent Client
        
    - SABnzbd:
        href: http://localhost:8081
        icon: sabnzbd.png
        description: Usenet Client

- Monitoring:
    - Grafana:
        href: http://localhost:3000
        icon: grafana.png
        description: Analytics
        
    - Tautulli:
        href: http://localhost:8181
        icon: tautulli.png
        description: Media Stats
        
    - Portainer:
        href: http://localhost:9000
        icon: portainer.png
        description: Container Management
EOF

    # Create settings.yaml
    cat > config/homepage/settings.yaml << 'EOF'
---
title: Media Server Dashboard
theme: dark
background: 
  image: https://images.unsplash.com/photo-1489599849927-2ee91cede3ba
  blur: sm
  brightness: 50
color: slate
layout:
  Media:
    style: row
    columns: 3
  Automation:
    style: row
    columns: 3
  Downloads:
    style: row
    columns: 2
  Monitoring:
    style: row
    columns: 3
EOF

    print_success "Created Homepage configuration"
}

# Deploy services
deploy_services() {
    print_header "Deploying Media Server Stack"
    
    # Pull latest images
    print_warning "Pulling latest Docker images..."
    docker compose pull
    
    # Start services
    print_warning "Starting services..."
    docker compose up -d
    
    print_success "Services deployed successfully!"
}

# Health check
health_check() {
    print_header "Performing Health Check"
    
    sleep 10  # Wait for services to start
    
    services=(
        "jellyfin:8096"
        "prowlarr:9696"
        "sonarr:8989"
        "radarr:7878"
        "overseerr:5055"
        "homepage:3001"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -q "200\|302"; then
            print_success "$name is running on port $port"
        else
            print_warning "$name may still be starting on port $port"
        fi
    done
}

# Main execution
main() {
    echo -e "${BLUE}Media Server Deployment Script${NC}"
    echo -e "${BLUE}==============================${NC}"
    
    # Check Docker
    check_docker
    
    # Create directories
    create_directories
    
    # Setup environment
    setup_env
    
    # Create configurations
    create_prometheus_config
    create_homepage_config
    
    # Deploy services
    deploy_services
    
    # Health check
    health_check
    
    print_header "Deployment Complete!"
    echo -e "\n${GREEN}Access your services:${NC}"
    echo "  Dashboard:   http://localhost:3001"
    echo "  Jellyfin:    http://localhost:8096"
    echo "  Overseerr:   http://localhost:5055"
    echo "  Portainer:   http://localhost:9000"
    echo ""
    echo "Next steps:"
    echo "1. Configure Prowlarr with indexers"
    echo "2. Connect Sonarr/Radarr to Prowlarr"
    echo "3. Configure download clients"
    echo "4. Setup Jellyfin libraries"
    echo ""
    print_warning "Remember to configure your VPN if using torrents!"
}

# Run main function
main "$@"