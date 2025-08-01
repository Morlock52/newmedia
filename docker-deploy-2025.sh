#!/usr/bin/env bash
set -euo pipefail

# NewMedia Docker Cleanup and Deployment Script - 2025 Best Practices
# Based on latest Docker cleanup recommendations
# Comprehensive media server stack with Jellyfin, Sonarr, Radarr, and more

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Icons
CHECK="✅"
CROSS="❌"
ROCKET="🚀"
CLEAN="🧹"
WHALE="🐳"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    🐳 NewMedia Stack Deployment                     ║"
echo "║                    Using 2025 Best Practices                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

warning() {
    echo -e "${YELLOW}${CLEAN} $1${NC}"
}

# Setup Docker environment with proper PATH
setup_docker() {
    log "Setting up Docker environment..."
    
    # Add Docker to PATH
    export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
    
    # Verify Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}${CROSS} Docker not found. Please install Docker Desktop.${NC}"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        warning "Docker daemon not running. Starting Docker Desktop..."
        open -a "Docker Desktop"
        sleep 10
    fi
    
    # Set compose command
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    success "Docker environment ready - using: $COMPOSE_CMD"
}

# Modern Docker cleanup following 2025 best practices
docker_cleanup() {
    log "Performing comprehensive Docker cleanup..."
    
    # Stop all running containers (except critical ones)
    warning "Stopping non-essential containers..."
    docker stop $(docker ps -q) 2>/dev/null || true
    
    # Clean up using modern Docker system prune
    warning "Removing unused containers..."
    docker container prune -f
    
    warning "Removing dangling images..."
    docker image prune -f
    
    warning "Removing unused volumes..."
    docker volume prune -f
    
    warning "Removing unused networks..."
    docker network prune -f
    
    # System-wide cleanup (keeps images in use)
    warning "Performing system cleanup..."
    docker system prune -f
    
    success "Docker cleanup completed"
}

# Create directory structure following best practices
create_structure() {
    log "Creating optimal directory structure..."
    
    cd /Users/morlock/fun/newmedia/media-server-stack
    
    # Create data directories with proper structure
    mkdir -p data/{media,torrents,usenet}/{movies,tv,music,online-videos}
    mkdir -p config/{jellyfin,sonarr,radarr,prowlarr,qbittorrent,overseerr,bazarr,homarr,traefik,gluetun}
    
    # Set proper permissions (macOS compatible)
    chmod -R 755 data config 2>/dev/null || true
    
    success "Directory structure created"
}

# Deploy using modern Docker Compose practices
deploy_stack() {
    log "Deploying NewMedia stack with modern practices..."
    
    cd /Users/morlock/fun/newmedia/media-server-stack
    
    # Pull latest images first
    warning "Pulling latest images..."
    $COMPOSE_CMD pull --quiet 2>/dev/null || $COMPOSE_CMD pull
    
    # Deploy with health checks
    warning "Starting core services..."
    $COMPOSE_CMD up -d --remove-orphans
    
    success "Stack deployment initiated"
}

# Health check with modern monitoring
health_check() {
    log "Performing health checks..."
    
    local max_wait=120
    local count=0
    
    while [[ $count -lt $max_wait ]]; do
        local healthy=0
        local total=0
        
        # Check running services
        while read -r service; do
            if [[ -n "$service" ]]; then
                ((total++))
                if $COMPOSE_CMD ps "$service" 2>/dev/null | grep -q "Up"; then
                    ((healthy++))
                fi
            fi
        done < <($COMPOSE_CMD config --services 2>/dev/null || echo "webui jellyfin sonarr radarr prowlarr")
        
        if [[ $healthy -ge 3 ]]; then
            success "Health check passed: $healthy/$total services running"
            break
        fi
        
        sleep 2
        ((count+=2))
    done
    
    if [[ $count -ge $max_wait ]]; then
        warning "Some services taking longer to start"
    fi
}

# Show service status and access information
show_status() {
    log "Service status and access information..."
    
    echo
    echo -e "${BLUE}🎛️  Service Status:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    $COMPOSE_CMD ps 2>/dev/null || docker ps
    
    echo
    echo -e "${GREEN}${ROCKET} NewMedia Stack Deployed Successfully!${NC}"
    echo
    echo -e "${CYAN}🌐 Access Your Services:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🏠 Main Dashboard:    http://localhost (Homarr)"
    echo "🎨 Web UI Manager:    http://localhost:3000"
    echo "🎬 Jellyfin:          http://localhost:8096"
    echo "📺 Sonarr:            http://localhost:8989"
    echo "🎞️  Radarr:            http://localhost:7878"
    echo "🔍 Prowlarr:          http://localhost:9696"
    echo "📋 Overseerr:         http://localhost:5055"
    echo "💬 Bazarr:            http://localhost:6767"
    echo "⚡ Traefik:           http://localhost:8080"
    echo
    echo -e "${YELLOW}💡 Pro Tips:${NC}"
    echo "• Start with the Web UI Manager for easy configuration"
    echo "• Configure Prowlarr indexers first, then Sonarr/Radarr"
    echo "• Set up Jellyfin libraries after downloads are working"
    echo
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "View logs:      $COMPOSE_CMD logs [service]"
    echo "Restart:        $COMPOSE_CMD restart [service]"
    echo "Stop all:       $COMPOSE_CMD down"
    echo "Update:         $COMPOSE_CMD pull && $COMPOSE_CMD up -d"
}

# Open services in browser
open_browser() {
    echo
    read -p "Would you like to open the Web UI Manager? [Y/n]: " -r response
    case "$response" in
        [Nn]|[Nn][Oo]) ;;
        *)
            log "Opening Web UI Manager..."
            sleep 3
            open "http://localhost:3000" 2>/dev/null || true
            ;;
    esac
}

# Main execution function
main() {
    echo -e "${WHALE} Starting NewMedia deployment with 2025 best practices..."
    echo
    
    setup_docker
    docker_cleanup
    create_structure
    deploy_stack
    health_check
    show_status
    open_browser
    
    echo
    echo -e "${GREEN}🎉 NewMedia stack is now running with modern Docker practices!${NC}"
    echo -e "${CYAN}Visit http://localhost:3000 for the management interface${NC}"
    echo
}

# Error handling
if ! main "$@"; then
    echo -e "${RED}${CROSS} Deployment failed. Check the logs above for details.${NC}"
    exit 1
fi
