#!/bin/bash

# Ultimate Media Server 2025 - Installation Script
# This script automates the deployment of the media server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Ultimate Media Server 2025 - Installation              ║"
echo "║                                                                  ║"
echo "║  Single-container architecture • Hardware acceleration • Auto-config ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored output
print_step() {
    echo -e "${BLUE}▶ $1${NC}"
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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root!"
   echo "Please run as a regular user with sudo privileges."
   exit 1
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_step "Detected OS: $OS $([ "$OS" = "linux" ] && echo "($DISTRO)")"

# Check prerequisites
print_step "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo "Please install Docker first:"
    if [[ "$OS" == "linux" ]]; then
        echo "  Ubuntu/Debian: sudo apt install docker.io"
        echo "  Fedora: sudo dnf install docker"
        echo "  Arch: sudo pacman -S docker"
    elif [[ "$OS" == "macos" ]]; then
        echo "  brew install --cask docker"
    fi
    exit 1
else
    print_success "Docker is installed"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    # Check for docker compose (v2)
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed!"
        exit 1
    else
        DOCKER_COMPOSE="docker compose"
        print_success "Docker Compose v2 is installed"
    fi
else
    DOCKER_COMPOSE="docker-compose"
    print_success "Docker Compose is installed"
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

# Get installation directory
print_step "Setting up installation directory..."
DEFAULT_DIR="$HOME/mediaserver"
read -p "Installation directory [$DEFAULT_DIR]: " INSTALL_DIR
INSTALL_DIR=${INSTALL_DIR:-$DEFAULT_DIR}

# Create directory structure
print_step "Creating directory structure..."
mkdir -p "$INSTALL_DIR"/{config,media,downloads}
mkdir -p "$INSTALL_DIR"/config/{caddy,jellyfin,sonarr,radarr,lidarr,readarr,prowlarr,bazarr,qbittorrent,homepage,uptime-kuma}
mkdir -p "$INSTALL_DIR"/media/{movies,tv,music,books,audiobooks,comics,photos,podcasts}
mkdir -p "$INSTALL_DIR"/downloads/{complete,incomplete,torrents,watch}
mkdir -p "$INSTALL_DIR"/{logs,backups,scripts}

print_success "Directory structure created"

# Change to installation directory
cd "$INSTALL_DIR"

# Download configuration files
print_step "Downloading configuration files..."

# Download docker-compose.yml
if [[ -f "docker-compose.yml" ]]; then
    print_warning "docker-compose.yml already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_step "Keeping existing docker-compose.yml"
    else
        mv docker-compose.yml docker-compose.yml.backup
        curl -fsSL https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/docker-compose.yml -o docker-compose.yml 2>/dev/null || \
        cp /Users/morlock/fun/newmedia/docker-compose.yml . 2>/dev/null || \
        print_warning "Could not download docker-compose.yml, using local template"
    fi
else
    curl -fsSL https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/docker-compose.yml -o docker-compose.yml 2>/dev/null || \
    cp /Users/morlock/fun/newmedia/docker-compose.yml . 2>/dev/null || \
    print_warning "Could not download docker-compose.yml, using local template"
fi

# Create .env file
print_step "Creating environment configuration..."

# Get user/group IDs
PUID=$(id -u)
PGID=$(id -g)

# Get timezone
if [[ -f /etc/timezone ]]; then
    TZ=$(cat /etc/timezone)
elif command -v timedatectl &> /dev/null; then
    TZ=$(timedatectl | grep "Time zone" | awk '{print $3}')
else
    TZ="America/New_York"
fi

# Get domain
print_step "Domain configuration..."
echo "Enter your domain (e.g., media.example.com)"
echo "For local access only, press Enter to use 'localhost'"
read -p "Domain [localhost]: " DOMAIN
DOMAIN=${DOMAIN:-localhost}

# Get email for SSL
if [[ "$DOMAIN" != "localhost" ]]; then
    read -p "Email for SSL certificates: " EMAIL
else
    EMAIL="admin@localhost"
fi

# Media path
print_step "Media storage configuration..."
echo "Where do you want to store your media files?"
echo "This should be a path with plenty of storage space."
read -p "Media path [$INSTALL_DIR/media]: " MEDIA_PATH
MEDIA_PATH=${MEDIA_PATH:-$INSTALL_DIR/media}

# Create .env file
cat > .env << EOF
# User and Group IDs
PUID=$PUID
PGID=$PGID

# Timezone
TZ=$TZ

# Paths
CONFIG_PATH=./config
MEDIA_PATH=$MEDIA_PATH
DOWNLOADS_PATH=./downloads

# Domain Configuration
DOMAIN=$DOMAIN
EMAIL=$EMAIL

# Service Ports
JELLYFIN_PORT=8096
SONARR_PORT=8989
RADARR_PORT=7878
LIDARR_PORT=8686
READARR_PORT=8787
PROWLARR_PORT=9696
BAZARR_PORT=6767
QBITTORRENT_PORT=8080
HOMEPAGE_PORT=3000
UPTIME_KUMA_PORT=3011

# Resource Limits
MEMORY_LIMIT=8G
CPU_LIMIT=4.0

# Features
ENABLE_HARDWARE_ACCELERATION=true
ENABLE_HTTPS=true
ENABLE_AUTO_UPDATES=false

# Passwords (auto-generated)
JELLYFIN_API_KEY=$(openssl rand -hex 32)
SONARR_API_KEY=$(openssl rand -hex 32)
RADARR_API_KEY=$(openssl rand -hex 32)
PROWLARR_API_KEY=$(openssl rand -hex 32)
QBITTORRENT_PASSWORD=$(openssl rand -base64 12)
EOF

print_success "Environment configuration created"

# Create Caddyfile
print_step "Creating Caddy configuration..."
mkdir -p config/caddy
cat > config/caddy/Caddyfile << 'EOF'
{
    email {$EMAIL}
    # Uncomment for local/development certificates
    # local_certs
}

# Main site
{$DOMAIN} {
    # Homepage dashboard
    handle / {
        reverse_proxy homepage:3000
    }
    
    # Jellyfin
    handle /jellyfin* {
        reverse_proxy jellyfin:8096
    }
    
    # Sonarr
    handle /sonarr* {
        reverse_proxy sonarr:8989
    }
    
    # Radarr  
    handle /radarr* {
        reverse_proxy radarr:7878
    }
    
    # Lidarr
    handle /lidarr* {
        reverse_proxy lidarr:8686
    }
    
    # Prowlarr
    handle /prowlarr* {
        reverse_proxy prowlarr:9696
    }
    
    # qBittorrent
    handle /qbittorrent* {
        reverse_proxy qbittorrent:8080
    }
    
    # Uptime Kuma
    handle /status* {
        reverse_proxy uptime-kuma:3001
    }
}
EOF

# Create backup script
print_step "Creating backup script..."
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
# Backup script for Ultimate Media Server

BACKUP_DIR="../backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Backing up configuration..."
tar -czf "$BACKUP_DIR/config.tar.gz" config/

echo "Backup complete: $BACKUP_DIR"
EOF
chmod +x scripts/backup.sh

# Create health check script
cat > scripts/health-check.sh << 'EOF'
#!/bin/bash
# Health check script

echo "🔍 Checking service health..."

services=("jellyfin:8096" "sonarr:8989" "radarr:7878" "prowlarr:9696" "homepage:3000")

for service in "${services[@]}"; do
    name="${service%%:*}"
    port="${service##*:}"
    
    if curl -s -o /dev/null "http://localhost:$port"; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
    fi
done
EOF
chmod +x scripts/health-check.sh

# Ask about deployment method
print_step "Choose deployment method:"
echo "1) Multi-container setup (recommended)"
echo "2) Single container with all services"
echo "3) Skip deployment (manual setup)"
read -p "Choice [1]: " DEPLOY_METHOD
DEPLOY_METHOD=${DEPLOY_METHOD:-1}

case $DEPLOY_METHOD in
    1)
        print_step "Starting multi-container deployment..."
        $DOCKER_COMPOSE up -d
        ;;
    2)
        print_step "Building single container..."
        if [[ -f "../Dockerfile.multi-service" ]]; then
            docker build -t mediaserver-aio -f ../Dockerfile.multi-service ..
            docker run -d \
                --name mediaserver \
                -p 80:80 -p 443:443 \
                -v "$INSTALL_DIR/config:/config" \
                -v "$MEDIA_PATH:/data/media" \
                -v "$INSTALL_DIR/downloads:/data/downloads" \
                -e PUID=$PUID -e PGID=$PGID -e TZ=$TZ \
                --restart unless-stopped \
                mediaserver-aio
        else
            print_error "Dockerfile.multi-service not found"
        fi
        ;;
    3)
        print_warning "Skipping deployment. Run 'docker-compose up -d' when ready."
        ;;
esac

# Wait for services to start
if [[ $DEPLOY_METHOD == "1" ]] || [[ $DEPLOY_METHOD == "2" ]]; then
    print_step "Waiting for services to start..."
    sleep 10
    
    # Run health check
    ./scripts/health-check.sh
fi

# Final summary
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo
echo "📁 Installation directory: $INSTALL_DIR"
echo "🌐 Domain: $DOMAIN"
echo
echo "🎯 Access your services:"
if [[ "$DOMAIN" == "localhost" ]]; then
    echo "   Homepage: http://localhost:3000"
    echo "   Jellyfin: http://localhost:8096"
    echo "   Sonarr: http://localhost:8989"
    echo "   Radarr: http://localhost:7878"
else
    echo "   Homepage: https://$DOMAIN"
    echo "   Jellyfin: https://$DOMAIN/jellyfin"
    echo "   Sonarr: https://$DOMAIN/sonarr"
    echo "   Radarr: https://$DOMAIN/radarr"
fi
echo
echo "📚 Next steps:"
echo "   1. Access the Homepage dashboard"
echo "   2. Configure Prowlarr indexers"
echo "   3. Connect *arr apps to Prowlarr"
echo "   4. Set up your media libraries"
echo
echo "📖 Documentation: https://github.com/yourusername/ultimate-media-server"
echo
echo -e "${PURPLE}Happy streaming! 🎬${NC}"