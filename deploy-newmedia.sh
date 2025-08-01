#!/usr/bin/env bash
set -euo pipefail

# NewMedia Stack Deployment Script
# Handles Docker path configuration and full stack deployment

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
INFO="ℹ️"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    🎬 NewMedia Stack Deployment                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

error() {
    echo -e "${RED}${CROSS} $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}${INFO} $1${NC}"
}

# Set up Docker environment
setup_docker_env() {
    log "Setting up Docker environment..."
    
    # Add Docker to PATH if it exists
    if [[ -d "/Applications/Docker.app/Contents/Resources/bin" ]]; then
        export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
        success "Docker path configured"
    fi
    
    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker not found. Please install Docker Desktop first."
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        warning "Docker daemon not running. Starting Docker Desktop..."
        open -a "Docker Desktop"
        log "Waiting for Docker to start..."
        
        # Wait up to 60 seconds for Docker to start
        for i in {1..60}; do
            if docker info >/dev/null 2>&1; then
                success "Docker is now running"
                break
            fi
            if [[ $i -eq 60 ]]; then
                error "Docker failed to start within 60 seconds"
            fi
            sleep 1
        done
    else
        success "Docker daemon is running"
    fi
    
    # Check Docker Compose (try both docker-compose and docker compose)
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        error "Docker Compose not found"
    fi
    
    # Set compose command preference
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    success "Docker Compose available: $COMPOSE_CMD"
    
    success "Docker environment ready"
}

# Navigate to project directory
cd_to_project() {
    local project_dir="/Users/morlock/fun/newmedia/media-server-stack"
    
    if [[ ! -d "$project_dir" ]]; then
        error "Project directory not found: $project_dir"
    fi
    
    cd "$project_dir"
    success "Changed to project directory: $(pwd)"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check environment file
    if [[ ! -f ".env" ]]; then
        error "Environment file (.env) not found"
    fi
    
    # Check Docker Compose file
    if [[ ! -f "docker-compose.yml" ]]; then
        error "Docker Compose file not found"
    fi
    
    # Check secrets directory
    if [[ ! -d "secrets" ]]; then
        error "Secrets directory not found"
    fi
    
    # Check for key secrets
    local required_secrets=(
        "traefik_dashboard_auth.txt"
        "wg_private_key.txt"
        "postgres_password.txt"
    )
    
    for secret in "${required_secrets[@]}"; do
        if [[ ! -f "secrets/$secret" ]]; then
            warning "Secret file missing: $secret"
        fi
    done
    
    success "Prerequisites checked"
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    
    # Source environment variables
    set -o allexport
    source .env
    set +o allexport
    
    # Create data directories
    mkdir -p data/media/{movies,tv,music,online-videos}
    mkdir -p data/torrents/{movies,tv,music,online-videos}
    mkdir -p data/usenet/{movies,tv,music,online-videos}
    
    # Create config directories
    local services=(
        "jellyfin" "sonarr" "radarr" "lidarr" "readarr" 
        "prowlarr" "qbittorrent" "overseerr" "bazarr" 
        "homarr" "mylar" "podgrab" "photoprism" 
        "tautulli" "youtube-dl-material" "traefik" "gluetun"
    )
    
    for service in "${services[@]}"; do
        mkdir -p "config/$service"
    done
    
    # Set permissions (ignore failures on macOS)
    chmod -R 755 data config 2>/dev/null || true
    
    success "Directories created"
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    $COMPOSE_CMD pull
    success "Images updated"
}

# Deploy the stack
deploy_stack() {
    log "Deploying NewMedia stack..."
    
    # Start core services first
    $COMPOSE_CMD up -d
    
    success "Stack deployed"
}

# Wait for services to be healthy
wait_for_services() {
    log "Waiting for services to start..."
    
    local max_wait=300  # 5 minutes
    local waited=0
    
    while [[ $waited -lt $max_wait ]]; do
        local healthy_count=0
        local total_services=0
        
        # Check running containers
        while IFS= read -r line; do
            if [[ $line == *"Up"* ]]; then
                ((healthy_count++))
            fi
            ((total_services++))
        done < <($COMPOSE_CMD ps --services | xargs -I {} $COMPOSE_CMD ps {})
        
        if [[ $healthy_count -gt 5 ]]; then  # At least 5 services running
            success "Services are starting up"
            break
        fi
        
        sleep 5
        ((waited+=5))
    done
    
    if [[ $waited -ge $max_wait ]]; then
        warning "Services taking longer than expected to start"
    fi
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check service status
    echo
    echo -e "${BLUE}Service Status:${NC}"
    $COMPOSE_CMD ps
    
    echo
    echo -e "${GREEN}${ROCKET} Deployment completed!${NC}"
    echo
    echo -e "${BLUE}Access your services:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🏠 Main Dashboard:  http://localhost (Homarr)"
    echo "🎬 Jellyfin:        http://localhost:8096"
    echo "🎨 Web UI Manager:  http://localhost:3000"
    echo "📺 Sonarr:          http://sonarr.localhost"
    echo "🎞️  Radarr:          http://radarr.localhost"
    echo "🔍 Prowlarr:        http://prowlarr.localhost"
    echo "📋 Overseerr:       http://overseerr.localhost"
    echo "💬 Bazarr:          http://bazarr.localhost"
    echo "⚡ Traefik:         http://traefik.localhost:8080"
    echo
    echo -e "${YELLOW}Note: Some services may take a few minutes to fully initialize.${NC}"
    echo
    echo -e "${BLUE}Management Commands:${NC}"
    echo "View logs:          $COMPOSE_CMD logs [service-name]"
    echo "Restart service:    $COMPOSE_CMD restart [service-name]"
    echo "Stop all:           $COMPOSE_CMD down"
    echo "Health check:       ./scripts/health-check.sh"
}

# Open services in browser
open_services() {
    echo
    read -p "Would you like to open the main dashboard in your browser? [y/N]: " -r response
    case "$response" in
        [Yy]|[Yy][Ee][Ss])
            log "Opening dashboard..."
            sleep 3  # Give services time to start
            open "http://localhost:3000" 2>/dev/null || true
            open "http://localhost:8096" 2>/dev/null || true
            ;;
    esac
}

# Main execution
main() {
    setup_docker_env
    cd_to_project
    check_prerequisites
    create_directories
    pull_images
    deploy_stack
    wait_for_services
    verify_deployment
    open_services
    
    echo
    echo -e "${GREEN}🎉 NewMedia stack is now running!${NC}"
    echo -e "${BLUE}Check the Web UI at http://localhost:3000 for management${NC}"
}

# Run with error handling
if main "$@"; then
    exit 0
else
    error "Deployment failed"
fi
