#!/bin/bash

# Media Server Status Check Script
# Diagnose and troubleshoot the media stack

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

echo -e "${BLUE}
╔══════════════════════════════════════╗
║       Media Server Status Check     ║
╚══════════════════════════════════════╝${NC}"

# Check Docker installation
echo ""
echo "🐳 Docker Status:"
echo "=================="

if command -v docker &> /dev/null; then
    print_status "Docker is installed"
    docker --version
    
    if docker info &> /dev/null; then
        print_status "Docker daemon is running"
    else
        print_error "Docker daemon is not running"
        print_warning "Please start Docker Desktop"
        exit 1
    fi
else
    print_error "Docker is not installed"
    print_warning "Run: ./install-docker.sh"
    exit 1
fi

# Check Docker Compose
echo ""
echo "🔧 Docker Compose Status:"
echo "=========================="

if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    print_status "Docker Compose V2 available"
    docker compose version
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    print_status "Docker Compose V1 available"
    docker-compose --version
else
    print_error "Docker Compose not found"
    print_warning "Install Docker Desktop or docker-compose"
    exit 1
fi

# Check if compose file exists
echo ""
echo "📄 Configuration Files:"
echo "======================="

if [ -f "docker-compose.yml" ]; then
    print_status "docker-compose.yml found"
else
    print_error "docker-compose.yml not found"
    exit 1
fi

# Check directory structure
echo ""
echo "📁 Directory Structure:"
echo "======================"

required_dirs=(
    "config"
    "config/jellyfin"
    "config/sonarr"
    "config/radarr"
    "config/qbittorrent"
    "config/prowlarr"
    "config/overseerr"
    "config/tautulli"
    "config/homarr"
    "data"
    "data/media"
    "data/media/movies"
    "data/media/tv"
    "data/media/music"
    "data/torrents"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_status "$dir exists"
    else
        print_warning "$dir missing (will be created)"
        mkdir -p "$dir"
        print_status "Created $dir"
    fi
done

# Check container status
echo ""
echo "🚀 Container Status:"
echo "==================="

if $COMPOSE_CMD ps | grep -q "Up"; then
    print_status "Some containers are running"
    $COMPOSE_CMD ps
else
    print_warning "No containers are running"
    echo ""
    echo "To start the stack:"
    echo "  ./quick-deploy.sh"
    echo "  or"
    echo "  ./deploy-media.sh"
fi

# Check port availability
echo ""
echo "🌐 Port Status:"
echo "==============="

ports=(8096 8989 7878 8080 9696 5055 8181 7575)
port_names=("Jellyfin" "Sonarr" "Radarr" "qBittorrent" "Prowlarr" "Overseerr" "Tautulli" "Homarr")

for i in "${!ports[@]}"; do
    port=${ports[$i]}
    name=${port_names[$i]}
    
    if lsof -i :$port &> /dev/null; then
        print_status "Port $port ($name) is in use"
    else
        print_warning "Port $port ($name) is free"
    fi
done

# Check if services are responding
echo ""
echo "🔗 Service Health:"
echo "=================="

services=(
    "Jellyfin:8096"
    "Sonarr:8989"
    "Radarr:7878"
    "qBittorrent:8080"
    "Prowlarr:9696"
    "Overseerr:5055"
    "Tautulli:8181"
    "Homarr:7575"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s --max-time 3 "http://localhost:$port" > /dev/null 2>&1; then
        print_status "$name (http://localhost:$port) is responding"
    else
        print_warning "$name (http://localhost:$port) is not responding"
    fi
done

echo ""
echo "🎯 Quick Actions:"
echo "=================="
echo "Start stack:     ./quick-deploy.sh"
echo "Full deploy:     ./deploy-media.sh"
echo "View logs:       $COMPOSE_CMD logs -f [service]"
echo "Stop stack:      $COMPOSE_CMD down"
echo "Restart stack:   $COMPOSE_CMD restart"