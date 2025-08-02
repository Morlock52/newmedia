#!/bin/bash
# Single Container Media Server 2025 - Setup Script
# This script sets up the all-in-one media server with optimal configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
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
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root!"
        print_warning "Please run as a regular user with sudo privileges"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker installed: $DOCKER_VERSION"
    else
        print_error "Docker not found! Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker Compose installed: $COMPOSE_VERSION"
    elif docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version | cut -d' ' -f4)
        print_success "Docker Compose (plugin) installed: $COMPOSE_VERSION"
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose not found! Please install Docker Compose."
        exit 1
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 50 ]]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (50GB+ recommended)"
    else
        print_success "Disk space: ${AVAILABLE_SPACE}GB available"
    fi
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk 'NR==2 {print $2}')
    if [[ $TOTAL_RAM -lt 4 ]]; then
        print_warning "Low RAM: ${TOTAL_RAM}GB detected (4GB+ recommended)"
    else
        print_success "RAM: ${TOTAL_RAM}GB detected"
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    # Base directories
    mkdir -p config/{caddy,jellyfin,radarr,sonarr,prowlarr,qbittorrent}
    mkdir -p media/{movies,tv,music,photos}
    mkdir -p downloads/{complete,incomplete}
    mkdir -p transcodes
    mkdir -p backups
    
    print_success "Directory structure created"
}

# Set up environment file
setup_environment() {
    print_header "Setting Up Environment"
    
    if [[ -f .env ]]; then
        print_warning ".env file already exists. Backing up to .env.backup"
        cp .env .env.backup
    fi
    
    # Get user input
    read -p "Enter your domain (e.g., media.example.com) [localhost]: " DOMAIN
    DOMAIN=${DOMAIN:-localhost}
    
    read -p "Enter your timezone (e.g., America/New_York) [UTC]: " TZ
    TZ=${TZ:-UTC}
    
    read -p "Enter media storage path [./media]: " MEDIA_PATH
    MEDIA_PATH=${MEDIA_PATH:-./media}
    
    read -p "Enter downloads path [./downloads]: " DOWNLOADS_PATH
    DOWNLOADS_PATH=${DOWNLOADS_PATH:-./downloads}
    
    read -p "Enter transcoding path [./transcodes]: " TRANSCODES_PATH
    TRANSCODES_PATH=${TRANSCODES_PATH:-./transcodes}
    
    # Create .env file
    cat > .env << EOF
# Single Container Media Server 2025 Configuration
# Generated on $(date)

# Domain Configuration
DOMAIN=https://${DOMAIN}

# Timezone
TZ=${TZ}

# User/Group IDs (current user)
PUID=$(id -u)
PGID=$(id -g)

# Storage Paths
MEDIA_PATH=${MEDIA_PATH}
MOVIES_PATH=${MEDIA_PATH}/movies
TV_PATH=${MEDIA_PATH}/tv
MUSIC_PATH=${MEDIA_PATH}/music
PHOTOS_PATH=${MEDIA_PATH}/photos
DOWNLOADS_PATH=${DOWNLOADS_PATH}
DOWNLOADS_COMPLETE=${DOWNLOADS_PATH}/complete
DOWNLOADS_INCOMPLETE=${DOWNLOADS_PATH}/incomplete
TRANSCODES_PATH=${TRANSCODES_PATH}

# Port Configuration
HTTP_PORT=80
HTTPS_PORT=443

# Resource Limits
MEMORY_LIMIT=8G
CPU_LIMIT=4.0
MEMORY_RESERVATION=2G
CPU_RESERVATION=1.0
EOF

    print_success "Environment file created"
}

# Set up Caddy configuration
setup_caddy() {
    print_header "Setting Up Caddy Configuration"
    
    if [[ ! -f config/caddy/Caddyfile ]]; then
        mkdir -p config/caddy
        
        cat > config/caddy/Caddyfile << 'EOF'
{
    email admin@example.com
    # Uncomment for production
    # acme_ca https://acme-v02.api.letsencrypt.org/directory
}

# Development configuration with self-signed certificates
:443 {
    tls internal

    # Homepage
    handle / {
        redir /web/ permanent
    }

    # Jellyfin
    handle /web/* {
        reverse_proxy localhost:8096
    }
    handle /socket {
        reverse_proxy localhost:8096
    }

    # Radarr
    handle /radarr/* {
        reverse_proxy localhost:7878
    }

    # Sonarr
    handle /sonarr/* {
        reverse_proxy localhost:8989
    }

    # Prowlarr
    handle /prowlarr/* {
        reverse_proxy localhost:9696
    }

    # qBittorrent
    handle /qbittorrent/* {
        reverse_proxy localhost:8080
    }

    # Security headers
    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "SAMEORIGIN"
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    # Enable compression
    encode gzip

    # Logging
    log {
        output file /config/caddy/access.log
        format console
    }
}

# HTTP to HTTPS redirect
:80 {
    redir https://{host}{uri} permanent
}
EOF
        print_success "Caddy configuration created"
    else
        print_warning "Caddy configuration already exists"
    fi
}

# Set permissions
set_permissions() {
    print_header "Setting Permissions"
    
    # Set ownership to current user
    sudo chown -R $(id -u):$(id -g) config media downloads transcodes
    
    # Set directory permissions
    find config media downloads transcodes -type d -exec chmod 755 {} \;
    
    # Set file permissions
    find config media downloads transcodes -type f -exec chmod 644 {} \;
    
    print_success "Permissions set"
}

# Build and start services
start_services() {
    print_header "Building and Starting Services"
    
    # Build image
    print_warning "Building Docker image (this may take several minutes)..."
    ${COMPOSE_CMD:-docker-compose} -f docker-compose.all-in-one.yml build
    
    # Start services
    print_warning "Starting services..."
    ${COMPOSE_CMD:-docker-compose} -f docker-compose.all-in-one.yml up -d
    
    print_success "Services started"
}

# Check service health
check_health() {
    print_header "Checking Service Health"
    
    echo "Waiting for services to start..."
    sleep 30
    
    # Check if container is running
    if docker ps | grep -q mediaserver-aio; then
        print_success "Container is running"
        
        # Check individual services
        docker exec mediaserver-aio s6-svstat /var/run/s6/services/*
        
        # Show logs
        print_header "Recent Logs"
        docker logs --tail 20 mediaserver-aio
    else
        print_error "Container is not running!"
        docker logs mediaserver-aio
    fi
}

# Display access information
show_info() {
    print_header "Access Information"
    
    DOMAIN=$(grep DOMAIN .env | cut -d'=' -f2 | sed 's/https:\/\///')
    
    echo -e "${GREEN}Your media server is ready!${NC}"
    echo ""
    echo "Access URLs:"
    echo "  Main Interface:  https://${DOMAIN}"
    echo "  Jellyfin:       https://${DOMAIN}/web/"
    echo "  Radarr:         https://${DOMAIN}/radarr/"
    echo "  Sonarr:         https://${DOMAIN}/sonarr/"
    echo "  Prowlarr:       https://${DOMAIN}/prowlarr/"
    echo "  qBittorrent:    https://${DOMAIN}/qbittorrent/"
    echo ""
    echo "Default Credentials:"
    echo "  Jellyfin:       Setup on first access"
    echo "  Radarr/Sonarr:  No auth by default (configure in settings)"
    echo "  qBittorrent:    admin/adminadmin (change immediately!)"
    echo ""
    echo "Commands:"
    echo "  View logs:      docker logs -f mediaserver-aio"
    echo "  Stop services:  docker-compose -f docker-compose.all-in-one.yml down"
    echo "  Start services: docker-compose -f docker-compose.all-in-one.yml up -d"
    echo "  Restart:        docker-compose -f docker-compose.all-in-one.yml restart"
    echo ""
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════╗"
    echo "║   Single Container Media Server 2025         ║"
    echo "║   All-in-One Setup Script                    ║"
    echo "╚══════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_root
    check_requirements
    create_directories
    setup_environment
    setup_caddy
    set_permissions
    start_services
    check_health
    show_info
}

# Run main function
main "$@"