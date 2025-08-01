#!/bin/bash

# Media Server 2025 Deployment Script
# Enhanced setup with security and performance optimizations

set -e

echo "🎬 Media Server 2025 - Professional Deployment"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose v2 is not available. Please install Docker Compose."
    exit 1
fi

print_status "Checking system requirements..."

# Get user IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)
print_status "Detected PUID: $USER_ID, PGID: $GROUP_ID"

# Check for Intel GPU
if [ -d "/dev/dri" ]; then
    RENDER_GROUP=$(getent group render | cut -d: -f3 2>/dev/null || echo "989")
    print_success "Intel GPU detected. Render group ID: $RENDER_GROUP"
else
    print_warning "No Intel GPU detected. Hardware transcoding may not be available."
    RENDER_GROUP="989"
fi

# Create directory structure
print_status "Creating directory structure..."

directories=(
    "config/traefik"
    "config/jellyfin"
    "config/prowlarr" 
    "config/sonarr"
    "config/radarr"
    "config/lidarr"
    "config/bazarr"
    "config/qbittorrent"
    "config/overseerr"
    "config/homepage"
    "config/tautulli"
    "config/portainer"
    "config/gluetun"
    "data/downloads/incomplete"
    "data/downloads/movies"
    "data/downloads/tv"
    "data/downloads/music"
    "data/media/movies"
    "data/media/tv"
    "data/media/music"
    "data/torrents"
    "cache/jellyfin"
    "transcodes"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

# Set proper permissions
chmod -R 755 data/
chown -R $USER:$USER data/

print_success "Directory structure created"

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating environment file..."
    
    # Get domain from user
    read -p "Enter your domain name (e.g., yourdomain.com): " DOMAIN
    read -p "Enter your Cloudflare email: " CF_EMAIL
    read -s -p "Enter your Cloudflare API token: " CF_TOKEN
    echo
    
    # Optional VPN setup
    print_status "VPN Configuration (optional but recommended)"
    read -p "VPN Provider (nordvpn/expressvpn/surfshark/skip): " VPN_PROVIDER
    
    if [ "$VPN_PROVIDER" != "skip" ]; then
        read -p "VPN Username: " VPN_USER
        read -s -p "VPN Password: " VPN_PASS
        echo
        read -p "VPN Country (e.g., Switzerland): " VPN_COUNTRY
    else
        VPN_PROVIDER="nordvpn"
        VPN_USER="username"
        VPN_PASS="password"
        VPN_COUNTRY="Switzerland"
    fi
    
    # Create .env file
    cat > .env << EOF
# Media Server 2025 Configuration
TZ=America/New_York
PUID=$USER_ID
PGID=$GROUP_ID
RENDER_GROUP_ID=$RENDER_GROUP

# Domain Configuration
DOMAIN=$DOMAIN

# Cloudflare Configuration
CLOUDFLARE_EMAIL=$CF_EMAIL
CLOUDFLARE_API_TOKEN=$CF_TOKEN

# VPN Configuration
VPN_PROVIDER=$VPN_PROVIDER
VPN_USERNAME=$VPN_USER
VPN_PASSWORD=$VPN_PASS
VPN_COUNTRY=$VPN_COUNTRY

# Network Configuration
MEDIA_NETWORK_SUBNET=172.20.0.0/16

# URLs
JELLYFIN_PUBLISHED_SERVER_URL=https://jellyfin.$DOMAIN
EOF

    print_success "Environment file created"
else
    print_status "Using existing .env file"
fi

# Create ACME file for Traefik
touch config/traefik/acme.json
chmod 600 config/traefik/acme.json

print_success "Traefik ACME file created with correct permissions"

# Check if compose file exists
COMPOSE_FILE="docker-compose-2025-enhanced.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Docker Compose file $COMPOSE_FILE not found!"
    exit 1
fi

# Pull latest images
print_status "Pulling latest Docker images..."
docker compose -f $COMPOSE_FILE pull

# Start the stack
print_status "Starting media server stack..."
docker compose -f $COMPOSE_FILE up -d

# Wait for services to start
print_status "Waiting for services to initialize..."
sleep 30

# Check service health
print_status "Checking service health..."

services=("traefik" "jellyfin" "prowlarr" "sonarr" "radarr" "overseerr")
failed_services=()

for service in "${services[@]}"; do
    if docker compose -f $COMPOSE_FILE ps $service | grep -q "Up"; then
        print_success "$service is running"
    else
        print_error "$service failed to start"
        failed_services+=($service)
    fi
done

# Display access URLs
if [ ${#failed_services[@]} -eq 0 ]; then
    print_success "All services started successfully!"
    echo
    echo "🌐 Access URLs:"
    echo "=============="
    source .env
    echo "📊 Dashboard: https://home.$DOMAIN"
    echo "🎬 Jellyfin: https://jellyfin.$DOMAIN"  
    echo "📺 Sonarr: https://sonarr.$DOMAIN"
    echo "🎭 Radarr: https://radarr.$DOMAIN"
    echo "🔍 Prowlarr: https://prowlarr.$DOMAIN"
    echo "📋 Overseerr: https://requests.$DOMAIN"
    echo "⚙️ Traefik: https://traefik.$DOMAIN"
    echo "🐳 Portainer: https://portainer.$DOMAIN"
    echo
    echo "📝 Next Steps:"
    echo "============="
    echo "1. Configure Cloudflare DNS to point to your server"
    echo "2. Wait for SSL certificates to generate (5-10 minutes)"
    echo "3. Access Jellyfin to complete initial setup"
    echo "4. Configure Prowlarr with indexers"
    echo "5. Set up Sonarr/Radarr with download client"
    echo
    echo "📖 Full setup guide: MEDIA_SERVER_SETUP_GUIDE_2025.md"
else
    print_error "Some services failed to start: ${failed_services[*]}"
    echo
    echo "🔍 Troubleshooting:"
    echo "==================="
    echo "1. Check logs: docker compose -f $COMPOSE_FILE logs [service_name]"
    echo "2. Check .env file configuration"
    echo "3. Ensure ports are not in use by other services"
    echo "4. Verify Docker and Docker Compose installation"
fi

# Create helpful aliases
echo
print_status "Creating helpful commands..."

cat > media-server-commands.sh << 'EOF'
#!/bin/bash
# Media Server Management Commands

alias media-status='docker compose -f docker-compose-2025-enhanced.yml ps'
alias media-logs='docker compose -f docker-compose-2025-enhanced.yml logs -f'
alias media-stop='docker compose -f docker-compose-2025-enhanced.yml down'
alias media-start='docker compose -f docker-compose-2025-enhanced.yml up -d'
alias media-update='docker compose -f docker-compose-2025-enhanced.yml pull && docker compose -f docker-compose-2025-enhanced.yml up -d'
alias media-backup='tar -czf backup-$(date +%Y%m%d).tar.gz config .env'

echo "Media Server Commands Loaded:"
echo "  media-status  - Show container status"
echo "  media-logs    - Show container logs"  
echo "  media-stop    - Stop all services"
echo "  media-start   - Start all services"
echo "  media-update  - Update all containers"
echo "  media-backup  - Create configuration backup"
EOF

chmod +x media-server-commands.sh

print_success "Management commands created in media-server-commands.sh"
echo "Run: source media-server-commands.sh"

echo
print_success "🎉 Media Server 2025 deployment completed!"
print_status "Monitor startup: docker compose -f $COMPOSE_FILE logs -f"