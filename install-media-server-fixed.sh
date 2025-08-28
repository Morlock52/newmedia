#!/bin/bash

# Ultimate Media Server 2025 - Installation Script (Fixed Version)
# This script automates the deployment of the media server
# Version: 1.1.0 - Fixed hardcoded paths and added error handling

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

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

# Rollback function for failed installations
rollback_installation() {
    print_error "Installation failed, rolling back..."
    
    if [ -n "${INSTALL_DIR:-}" ] && [ -d "$INSTALL_DIR" ]; then
        # Stop any running containers
        cd "$INSTALL_DIR" 2>/dev/null && {
            if [ -n "${DOCKER_COMPOSE:-}" ]; then
                $DOCKER_COMPOSE down --remove-orphans 2>/dev/null || true
            fi
        }
        
        # Backup failed installation
        local backup_dir="${INSTALL_DIR}.failed.$(date +%s)"
        mv "$INSTALL_DIR" "$backup_dir"
        print_warning "Failed installation backed up to: $backup_dir"
    fi
    
    exit 1
}

# Set up error handling
trap rollback_installation ERR

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

# Pre-flight checks
preflight_checks() {
    print_step "Running pre-flight checks..."
    
    # Check available disk space
    local available_space
    if [[ "$OS" == "macos" ]]; then
        available_space=$(df -g "$HOME" | awk 'NR==2 {print $4}')
    else
        available_space=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    
    if [ "${available_space:-0}" -lt 50 ]; then
        print_warning "Low disk space: ${available_space}GB available (50GB recommended)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    fi
    
    # Check for port conflicts
    local ports=(8096 8989 7878 9696 8080 3000)
    local conflicts=0
    for port in "${ports[@]}"; do
        if lsof -i :$port &> /dev/null; then
            print_warning "Port $port is already in use"
            ((conflicts++))
        fi
    done
    
    if [ $conflicts -gt 0 ]; then
        print_warning "$conflicts port conflicts detected"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    fi
}

# Run pre-flight checks
preflight_checks

# Check prerequisites
print_step "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo "Please install Docker first:"
    if [[ "$OS" == "linux" ]]; then
        echo "  Ubuntu/Debian: sudo apt install docker.io docker-compose"
        echo "  Fedora: sudo dnf install docker docker-compose"
        echo "  Arch: sudo pacman -S docker docker-compose"
    elif [[ "$OS" == "macos" ]]; then
        echo "  brew install --cask docker"
        echo "  Or download from: https://docs.docker.com/desktop/mac/install/"
    fi
    exit 1
else
    print_success "Docker is installed"
fi

# Better Docker Compose detection
detect_docker_compose() {
    if docker compose version &> /dev/null; then
        echo "docker compose"
    elif command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        return 1
    fi
}

DOCKER_COMPOSE=$(detect_docker_compose)
if [ -z "$DOCKER_COMPOSE" ]; then
    print_error "Docker Compose is not installed!"
    echo "Docker Compose is required for this installation."
    if [[ "$OS" == "linux" ]]; then
        echo "Install with: sudo apt install docker-compose (or equivalent)"
    elif [[ "$OS" == "macos" ]]; then
        echo "Docker Desktop includes Docker Compose"
    fi
    exit 1
else
    print_success "Docker Compose detected: $DOCKER_COMPOSE"
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running!"
    echo "Please start Docker and try again."
    if [[ "$OS" == "macos" ]]; then
        echo "Start Docker Desktop from Applications"
    else
        echo "Start with: sudo systemctl start docker"
    fi
    exit 1
fi

# Check Docker resources (if possible)
if docker system info &> /dev/null; then
    local docker_memory=$(docker system info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
    if [ "$docker_memory" != "0" ] && [ "$docker_memory" -lt 8589934592 ]; then
        print_warning "Docker has less than 8GB memory allocated"
        echo "Consider increasing Docker Desktop memory allocation for better performance"
    fi
fi

# Get installation directory
print_step "Setting up installation directory..."
DEFAULT_DIR="$HOME/mediaserver"
read -p "Installation directory [$DEFAULT_DIR]: " INSTALL_DIR
INSTALL_DIR=${INSTALL_DIR:-$DEFAULT_DIR}

# Validate installation directory
if [[ ! "$INSTALL_DIR" =~ ^/ ]] && [[ ! "$INSTALL_DIR" =~ ^~ ]]; then
    INSTALL_DIR="$PWD/$INSTALL_DIR"
fi

# Expand tilde if present
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

# Create directory structure
print_step "Creating directory structure..."
mkdir -p "$INSTALL_DIR"/{config,media,downloads,logs,backups,scripts}
mkdir -p "$INSTALL_DIR"/config/{caddy,jellyfin,sonarr,radarr,lidarr,readarr,prowlarr,bazarr,qbittorrent,homepage,uptime-kuma}
mkdir -p "$INSTALL_DIR"/media/{movies,tv,music,books,audiobooks,comics,photos,podcasts}
mkdir -p "$INSTALL_DIR"/downloads/{complete,incomplete,torrents,watch}

print_success "Directory structure created"

# Change to installation directory
cd "$INSTALL_DIR"

# Download or copy configuration files
print_step "Setting up configuration files..."

# Copy docker-compose.yml
if [[ -f "$SCRIPT_DIR/docker-compose.yml" ]]; then
    cp "$SCRIPT_DIR/docker-compose.yml" .
    print_success "Copied docker-compose.yml from script directory"
elif [[ -f "$SCRIPT_DIR/docker/docker-compose.yml" ]]; then
    cp "$SCRIPT_DIR/docker/docker-compose.yml" .
    print_success "Copied docker-compose.yml from docker directory"
else
    print_warning "docker-compose.yml not found locally"
    # Try to download from repository
    if command -v curl &> /dev/null; then
        print_step "Attempting to download docker-compose.yml..."
        # Replace with your actual repository URL
        REPO_URL="https://raw.githubusercontent.com/yourusername/ultimate-media-server/main"
        if curl -fsSL "$REPO_URL/docker-compose.yml" -o docker-compose.yml 2>/dev/null; then
            print_success "Downloaded docker-compose.yml"
        else
            print_error "Failed to download docker-compose.yml"
            echo "Please ensure docker-compose.yml is in the same directory as this script"
            exit 1
        fi
    else
        print_error "docker-compose.yml not found and curl not available"
        exit 1
    fi
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
elif [[ "$OS" == "macos" ]]; then
    TZ=$(sudo systemsetup -gettimezone 2>/dev/null | awk '{print $3}' || echo "America/New_York")
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
    while [[ ! "$EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; do
        print_error "Invalid email format"
        read -p "Email for SSL certificates: " EMAIL
    done
else
    EMAIL="admin@localhost"
fi

# Media path
print_step "Media storage configuration..."
echo "Where do you want to store your media files?"
echo "This should be a path with plenty of storage space."
read -p "Media path [$INSTALL_DIR/media]: " MEDIA_PATH
MEDIA_PATH=${MEDIA_PATH:-$INSTALL_DIR/media}

# Expand and validate media path
MEDIA_PATH="${MEDIA_PATH/#\~/$HOME}"
mkdir -p "$MEDIA_PATH"

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
ENABLE_HARDWARE_ACCELERATION=$([ "$OS" == "linux" ] && echo "true" || echo "false")
ENABLE_HTTPS=$([ "$DOMAIN" != "localhost" ] && echo "true" || echo "false")
ENABLE_AUTO_UPDATES=false

# Passwords (auto-generated)
JELLYFIN_API_KEY=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | cut -d' ' -f1)
SONARR_API_KEY=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | cut -d' ' -f1)
RADARR_API_KEY=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | cut -d' ' -f1)
PROWLARR_API_KEY=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | cut -d' ' -f1)
QBITTORRENT_PASSWORD=$(openssl rand -base64 12 2>/dev/null || date +%s | sha256sum | cut -c1-12)
EOF

# Set secure permissions
chmod 600 .env

print_success "Environment configuration created"

# Create Caddyfile
print_step "Creating Caddy configuration..."
mkdir -p config/caddy

if [[ "$DOMAIN" == "localhost" ]]; then
    # Local development configuration
    cat > config/caddy/Caddyfile << 'EOF'
{
    admin off
    local_certs
}

:80 {
    # Homepage dashboard
    handle / {
        reverse_proxy homepage:3000
    }
    
    # Service proxies with proper headers
    handle_path /jellyfin* {
        reverse_proxy jellyfin:8096 {
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto {scheme}
        }
    }
    
    handle_path /sonarr* {
        reverse_proxy sonarr:8989 {
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
        }
    }
    
    handle_path /radarr* {
        reverse_proxy radarr:7878 {
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
        }
    }
}
EOF
else
    # Production configuration with HTTPS
    cat > config/caddy/Caddyfile << EOF
{
    email $EMAIL
}

$DOMAIN {
    # Homepage dashboard
    handle / {
        reverse_proxy homepage:3000
    }
    
    # Jellyfin
    handle_path /jellyfin* {
        reverse_proxy jellyfin:8096
    }
    
    # Sonarr
    handle_path /sonarr* {
        reverse_proxy sonarr:8989
    }
    
    # Radarr  
    handle_path /radarr* {
        reverse_proxy radarr:7878
    }
    
    # Prowlarr
    handle_path /prowlarr* {
        reverse_proxy prowlarr:9696
    }
    
    # qBittorrent
    handle_path /qbittorrent* {
        reverse_proxy qbittorrent:8080
    }
    
    # Uptime Kuma
    handle_path /status* {
        reverse_proxy uptime-kuma:3001
    }
}
EOF
fi

# Create backup script
print_step "Creating utility scripts..."
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
# Backup script for Ultimate Media Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"

echo "Creating backup in: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Stop services before backup (optional)
read -p "Stop services before backup? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$PROJECT_ROOT"
    docker-compose stop
fi

echo "Backing up configuration..."
tar -czf "$BACKUP_DIR/config.tar.gz" -C "$PROJECT_ROOT" config/

echo "Backing up environment file..."
cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/.env.backup"

# Restart services if they were stopped
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose start
fi

echo "Backup complete: $BACKUP_DIR"
echo "Total size: $(du -sh "$BACKUP_DIR" | cut -f1)"
EOF
chmod +x scripts/backup.sh

# Create health check script
cat > scripts/health-check.sh << 'EOF'
#!/bin/bash
# Health check script

echo "🔍 Checking service health..."

services=(
    "jellyfin:8096:Jellyfin"
    "sonarr:8989:Sonarr"
    "radarr:7878:Radarr"
    "prowlarr:9696:Prowlarr"
    "homepage:3000:Homepage"
    "qbittorrent:8080:qBittorrent"
)

failed=0
for service_info in "${services[@]}"; do
    IFS=':' read -r name port display <<< "$service_info"
    
    if curl -sf -m 5 "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ $display is healthy"
    else
        echo "❌ $display is not responding on port $port"
        ((failed++))
    fi
done

if [ $failed -eq 0 ]; then
    echo "✅ All services are healthy!"
else
    echo "⚠️  $failed service(s) need attention"
fi
EOF
chmod +x scripts/health-check.sh

# Create update script
cat > scripts/update.sh << 'EOF'
#!/bin/bash
# Update script for Ultimate Media Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "📦 Updating Ultimate Media Server..."

# Create backup first
echo "Creating backup before update..."
./scripts/backup.sh

# Pull latest images
echo "Pulling latest Docker images..."
docker-compose pull

# Recreate containers
echo "Recreating containers..."
docker-compose up -d

echo "✅ Update complete!"
echo "Run ./scripts/health-check.sh to verify services"
EOF
chmod +x scripts/update.sh

# Ask about deployment method
print_step "Choose deployment method:"
echo "1) Deploy core services only (Jellyfin, Homepage, qBittorrent)"
echo "2) Deploy all services"
echo "3) Skip deployment (manual setup)"
read -p "Choice [1]: " DEPLOY_METHOD
DEPLOY_METHOD=${DEPLOY_METHOD:-1}

case $DEPLOY_METHOD in
    1)
        print_step "Starting core services deployment..."
        $DOCKER_COMPOSE up -d jellyfin homepage qbittorrent
        ;;
    2)
        print_step "Starting full deployment..."
        $DOCKER_COMPOSE up -d
        ;;
    3)
        print_warning "Skipping deployment. Run 'docker-compose up -d' when ready."
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Wait for services to start
if [[ $DEPLOY_METHOD == "1" ]] || [[ $DEPLOY_METHOD == "2" ]]; then
    print_step "Waiting for services to start..."
    
    # Show progress
    for i in {1..15}; do
        echo -ne "\rStarting services... $i/15"
        sleep 1
    done
    echo
    
    # Run health check
    ./scripts/health-check.sh
fi

# Save installation info
cat > installation-info.txt << EOF
Ultimate Media Server Installation
==================================
Date: $(date)
Directory: $INSTALL_DIR
Domain: $DOMAIN
Media Path: $MEDIA_PATH
Docker Compose: $DOCKER_COMPOSE
Services Deployed: $([ $DEPLOY_METHOD == "1" ] && echo "Core" || echo "All")
EOF

# Final summary
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo
echo "📁 Installation directory: $INSTALL_DIR"
echo "🌐 Domain: $DOMAIN"
echo "📺 Media location: $MEDIA_PATH"
echo
echo "🎯 Access your services:"
if [[ "$DOMAIN" == "localhost" ]]; then
    echo "   Homepage: http://localhost:3000"
    echo "   Jellyfin: http://localhost:8096"
    echo "   qBittorrent: http://localhost:8080"
    if [[ $DEPLOY_METHOD == "2" ]]; then
        echo "   Sonarr: http://localhost:8989"
        echo "   Radarr: http://localhost:7878"
        echo "   Prowlarr: http://localhost:9696"
    fi
else
    echo "   Homepage: https://$DOMAIN"
    echo "   Services: https://$DOMAIN/[service-name]"
fi
echo
echo "📚 Default Credentials:"
echo "   qBittorrent: admin / adminadmin (change immediately!)"
echo "   Other services: Set up on first access"
echo
echo "🛠️  Utility Scripts:"
echo "   Backup: ./scripts/backup.sh"
echo "   Health Check: ./scripts/health-check.sh"
echo "   Update: ./scripts/update.sh"
echo
echo "📖 Next steps:"
echo "   1. Change default passwords"
echo "   2. Configure Jellyfin libraries"
echo "   3. Set up Prowlarr indexers"
echo "   4. Connect *arr apps to Prowlarr"
echo
echo -e "${PURPLE}Happy streaming! 🎬${NC}"
echo
echo "Installation details saved to: installation-info.txt"