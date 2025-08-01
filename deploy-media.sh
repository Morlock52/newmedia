#!/bin/bash

# Media Server Docker Deployment Script
# Deploy complete media server stack with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}
╔══════════════════════════════════════╗
║        Media Server Deployment      ║
║             Docker Stack             ║
╚══════════════════════════════════════╝${NC}"
}

# Detect Docker Compose command
detect_compose_command() {
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
        print_status "Using Docker Compose V2"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        print_status "Using Docker Compose V1"
    else
        print_error "Docker Compose not found. Please install Docker first."
        print_warning "Run: ./install-docker.sh"
        exit 1
    fi
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed."
        print_warning "Run: ./install-docker.sh to install Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running."
        print_warning "Please start Docker Desktop and try again"
        exit 1
    fi
    
    detect_compose_command
    print_status "Docker is ready ✓"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Config directories
    mkdir -p config/{jellyfin,sonarr,radarr,qbittorrent,prowlarr,overseerr,tautulli,homarr}
    
    # Data directories
    mkdir -p data/media/{movies,tv,music}
    mkdir -p data/torrents/{movies,tv,music}
    
    # Set permissions (if needed)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo chown -R $(id -u):$(id -g) config/ data/ 2>/dev/null || true
    fi
    
    print_status "Directory structure created ✓"
}

# Deploy the stack
deploy_stack() {
    print_status "Deploying media server stack..."
    
    # Pull latest images
    print_status "Pulling Docker images..."
    $COMPOSE_CMD pull
    
    # Start services
    print_status "Starting services..."
    $COMPOSE_CMD up -d
    
    print_status "Media server stack deployed ✓"
}

# Check service status
check_services() {
    print_status "Checking service status..."
    $COMPOSE_CMD ps
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to start..."
    sleep 15
    
    services=(
        "jellyfin:8096"
        "sonarr:8989"
        "radarr:7878"
        "qbittorrent:8080"
        "prowlarr:9696"
        "overseerr:5055"
        "tautulli:8181"
        "homarr:7575"
    )
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        print_status "Checking $name on port $port..."
        timeout=30
        while [ $timeout -gt 0 ]; do
            if curl -s "http://localhost:$port" > /dev/null 2>&1; then
                print_status "$name is ready ✓"
                break
            fi
            sleep 2
            ((timeout-=2))
        done
        
        if [ $timeout -le 0 ]; then
            print_warning "$name may not be ready yet (still starting)"
        fi
    done
}

# Display service URLs
show_services() {
    echo -e "${GREEN}
🎉 Media Server Stack Deployed Successfully!

📊 Service Dashboard:
├── 🎬 Jellyfin (Media Server):     http://localhost:8096
├── 📺 Sonarr (TV Shows):           http://localhost:8989
├── 🎬 Radarr (Movies):             http://localhost:7878
├── ⬇️  qBittorrent (Downloads):     http://localhost:8080
├── 🔍 Prowlarr (Indexers):         http://localhost:9696
├── 📋 Overseerr (Requests):        http://localhost:5055
├── 📈 Tautulli (Analytics):        http://localhost:8181
└── 🏠 Homarr (Dashboard):          http://localhost:7575

🔧 Quick Commands:
├── View logs:        $COMPOSE_CMD logs -f [service]
├── Restart service:  $COMPOSE_CMD restart [service]
├── Stop all:         $COMPOSE_CMD down
├── Update stack:     $COMPOSE_CMD pull && $COMPOSE_CMD up -d
└── Check status:     $COMPOSE_CMD ps

💡 First-time setup:
1. Configure Jellyfin at http://localhost:8096
2. Set up indexers in Prowlarr
3. Configure download client in Sonarr/Radarr
4. Point services to qBittorrent
5. Connect Overseerr to Jellyfin

📁 Data Structure:
├── ./config/     - Service configurations
└── ./data/       - Media and download storage
${NC}"
}

# Main execution
main() {
    print_header
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    print_status "Starting media server deployment..."
    
    check_docker
    create_directories
    deploy_stack
    check_services
    wait_for_services
    show_services
    
    print_status "Deployment complete! 🚀"
}

# Run main function
main "$@"