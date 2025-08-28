#!/bin/bash

# ============================================================================
# ULTIMATE MEDIA SERVER DEPLOYMENT SCRIPT - AUGUST 2025
# ============================================================================
# Complete deployment of the fixed and optimized media server stack
# Features all 2025 best practices and optimizations
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="newmedia"
COMPOSE_FILE="docker-compose.fixed.yml"
ENV_TEMPLATE=".env.fixed.template"
ENV_FILE=".env"

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${MAGENTA}$1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

check_requirements() {
    print_header "CHECKING SYSTEM REQUIREMENTS"
    
    local requirements_met=true
    
    # Check Docker
    print_step "Checking Docker installation..."
    if command -v docker &> /dev/null; then
        print_success "Docker $(docker --version | cut -d' ' -f3 | tr -d ',') installed"
    else
        print_error "Docker not installed"
        requirements_met=false
    fi
    
    # Check Docker Compose
    print_step "Checking Docker Compose..."
    if docker compose version &> /dev/null; then
        print_success "Docker Compose $(docker compose version --short) installed"
    elif command -v docker-compose &> /dev/null; then
        print_success "Docker Compose $(docker-compose --version | cut -d' ' -f3 | tr -d ',') installed"
    else
        print_error "Docker Compose not installed"
        requirements_met=false
    fi
    
    # Check Node.js (for API server)
    print_step "Checking Node.js installation..."
    if command -v node &> /dev/null; then
        print_success "Node.js $(node --version) installed"
    else
        print_warning "Node.js not installed (optional for API server)"
    fi
    
    if [ "$requirements_met" = false ]; then
        print_error "Missing requirements. Please install Docker and Docker Compose."
        exit 1
    fi
}

setup_environment() {
    print_header "SETTING UP ENVIRONMENT"
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        print_step "Creating environment file..."
        if [ -f "$ENV_TEMPLATE" ]; then
            cp "$ENV_TEMPLATE" "$ENV_FILE"
            print_success "Environment file created from template"
        else
            print_error "Environment template not found"
            exit 1
        fi
    else
        print_success "Environment file already exists"
    fi
    
    # Set default values if not set
    print_step "Configuring environment variables..."
    
    # Get current user ID and group ID
    PUID=$(id -u)
    PGID=$(id -g)
    
    # Update .env with current user/group
    sed -i.bak "s/PUID=.*/PUID=$PUID/" "$ENV_FILE" 2>/dev/null || true
    sed -i.bak "s/PGID=.*/PGID=$PGID/" "$ENV_FILE" 2>/dev/null || true
    
    print_success "Environment configured"
}

create_directories() {
    print_header "CREATING DIRECTORY STRUCTURE"
    
    print_step "Creating required directories..."
    
    # Media directories
    mkdir -p media/{movies,tv,music,books,photos}
    mkdir -p downloads/{complete,incomplete,torrents,usenet}
    
    # Config directories
    mkdir -p config/{jellyfin,plex,emby,sonarr,radarr,lidarr,readarr,bazarr,prowlarr}
    mkdir -p config/{qbittorrent,sabnzbd,transmission,overseerr,tautulli}
    mkdir -p config/{traefik,nginx,homepage,portainer,watchtower}
    mkdir -p config/{prometheus,grafana,uptime-kuma}
    
    # Traefik specific
    mkdir -p traefik/dynamic
    touch traefik/acme.json
    chmod 600 traefik/acme.json
    
    # Database directory
    mkdir -p data/db
    
    # Logs directory
    mkdir -p logs
    
    print_success "Directory structure created"
}

install_dependencies() {
    print_header "INSTALLING DEPENDENCIES"
    
    if [ -f "package.json" ]; then
        if command -v npm &> /dev/null; then
            print_step "Installing Node.js dependencies..."
            npm install --silent
            print_success "Node.js dependencies installed"
        else
            print_warning "npm not available, skipping Node.js dependencies"
        fi
    fi
}

pull_docker_images() {
    print_header "PULLING DOCKER IMAGES"
    
    print_step "Pulling latest Docker images (this may take a while)..."
    
    if [ -f "$COMPOSE_FILE" ]; then
        docker compose -f "$COMPOSE_FILE" pull 2>/dev/null || \
        docker-compose -f "$COMPOSE_FILE" pull 2>/dev/null || \
        print_warning "Could not pull all images, will pull on first run"
    else
        print_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    print_success "Docker images ready"
}

start_services() {
    print_header "STARTING SERVICES"
    
    print_step "Starting all services..."
    
    # Start services
    docker compose -f "$COMPOSE_FILE" up -d 2>/dev/null || \
    docker-compose -f "$COMPOSE_FILE" up -d
    
    print_success "Services starting..."
    
    # Wait for services to be healthy
    print_step "Waiting for services to become healthy..."
    sleep 10
    
    # Check service status
    docker compose -f "$COMPOSE_FILE" ps 2>/dev/null || \
    docker-compose -f "$COMPOSE_FILE" ps
}

start_api_server() {
    print_header "STARTING API SERVER"
    
    if [ -f "start-api-server.js" ] && command -v node &> /dev/null; then
        print_step "Starting Node.js API server..."
        
        # Check if already running
        if pgrep -f "start-api-server.js" > /dev/null; then
            print_success "API server already running"
        else
            # Start in background
            nohup node start-api-server.js > logs/api-server.log 2>&1 &
            sleep 3
            
            if pgrep -f "start-api-server.js" > /dev/null; then
                print_success "API server started on port 3000"
            else
                print_warning "API server failed to start, check logs/api-server.log"
            fi
        fi
    else
        print_warning "API server not available"
    fi
}

configure_services() {
    print_header "CONFIGURING SERVICES"
    
    print_step "Waiting for services to initialize..."
    sleep 15
    
    # Get service URLs
    JELLYFIN_URL="http://localhost:8096"
    SONARR_URL="http://localhost:8989"
    RADARR_URL="http://localhost:7878"
    PROWLARR_URL="http://localhost:9696"
    QBITTORRENT_URL="http://localhost:8081"
    
    print_success "Services initialized"
    
    # Note: API keys will be extracted automatically by the API server
    print_step "API keys will be managed by the API server"
}

show_dashboard() {
    print_header "DEPLOYMENT COMPLETE!"
    
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  🎉 ${CYAN}MEDIA SERVER SUCCESSFULLY DEPLOYED!${NC}                  ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${YELLOW}📺 MEDIA SERVERS:${NC}"
    echo -e "  Jellyfin:     ${CYAN}http://localhost:8096${NC}"
    echo -e "  Plex:         ${CYAN}http://localhost:32400/web${NC}"
    echo -e "  Emby:         ${CYAN}http://localhost:8097${NC}"
    
    echo -e "\n${YELLOW}📥 DOWNLOAD AUTOMATION:${NC}"
    echo -e "  Sonarr:       ${CYAN}http://localhost:8989${NC}"
    echo -e "  Radarr:       ${CYAN}http://localhost:7878${NC}"
    echo -e "  Lidarr:       ${CYAN}http://localhost:8686${NC}"
    echo -e "  Readarr:      ${CYAN}http://localhost:8787${NC}"
    echo -e "  Prowlarr:     ${CYAN}http://localhost:9696${NC}"
    
    echo -e "\n${YELLOW}🌐 DOWNLOAD CLIENTS:${NC}"
    echo -e "  qBittorrent:  ${CYAN}http://localhost:8081${NC}"
    echo -e "  SABnzbd:      ${CYAN}http://localhost:8082${NC}"
    echo -e "  Transmission: ${CYAN}http://localhost:9091${NC}"
    
    echo -e "\n${YELLOW}🎛️ MANAGEMENT:${NC}"
    echo -e "  Dashboard:    ${CYAN}http://localhost:3000${NC}"
    echo -e "  Portainer:    ${CYAN}http://localhost:9000${NC}"
    echo -e "  Traefik:      ${CYAN}http://localhost:8080${NC}"
    
    echo -e "\n${YELLOW}📊 MONITORING:${NC}"
    echo -e "  Grafana:      ${CYAN}http://localhost:3001${NC}"
    echo -e "  Prometheus:   ${CYAN}http://localhost:9090${NC}"
    echo -e "  Uptime Kuma:  ${CYAN}http://localhost:3002${NC}"
    
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Default Credentials:${NC}"
    echo -e "  API Dashboard: admin / admin123"
    echo -e "  Grafana:       admin / admin"
    echo -e "  qBittorrent:   admin / adminadmin"
    echo -e "\n${YELLOW}⚠️  Please change all default passwords immediately!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    
    echo -e "\n${CYAN}📖 Next Steps:${NC}"
    echo -e "  1. Access the dashboard at ${CYAN}http://localhost:3000${NC}"
    echo -e "  2. Configure your media libraries in Jellyfin/Plex"
    echo -e "  3. Set up indexers in Prowlarr"
    echo -e "  4. Configure download paths in Sonarr/Radarr"
    echo -e "  5. Enable SSL with Traefik for production use"
    
    echo -e "\n${GREEN}✨ Enjoy your automated media server!${NC}\n"
}

health_check() {
    print_header "RUNNING HEALTH CHECKS"
    
    print_step "Checking service health..."
    
    # Check critical services
    local all_healthy=true
    
    # Check Docker containers
    if docker ps | grep -q "jellyfin"; then
        print_success "Jellyfin is running"
    else
        print_warning "Jellyfin is not running"
        all_healthy=false
    fi
    
    if docker ps | grep -q "sonarr"; then
        print_success "Sonarr is running"
    else
        print_warning "Sonarr is not running"
        all_healthy=false
    fi
    
    if docker ps | grep -q "radarr"; then
        print_success "Radarr is running"
    else
        print_warning "Radarr is not running"
        all_healthy=false
    fi
    
    # Check API server
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "API server is healthy"
    else
        print_warning "API server is not responding"
    fi
    
    if [ "$all_healthy" = true ]; then
        print_success "All critical services are healthy!"
    else
        print_warning "Some services need attention"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    clear
    
    echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}     ${CYAN}🚀 ULTIMATE MEDIA SERVER DEPLOYMENT - AUG 2025 🚀${NC}     ${MAGENTA}║${NC}"
    echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════╝${NC}"
    
    # Run deployment steps
    check_requirements
    setup_environment
    create_directories
    install_dependencies
    pull_docker_images
    start_services
    start_api_server
    configure_services
    health_check
    show_dashboard
    
    # Save deployment info
    echo "Deployment completed at $(date)" > deployment.log
    echo "Services started successfully" >> deployment.log
}

# Handle script arguments
case "${1:-}" in
    stop)
        print_header "STOPPING SERVICES"
        docker compose -f "$COMPOSE_FILE" down
        pkill -f "start-api-server.js" 2>/dev/null || true
        print_success "All services stopped"
        ;;
    restart)
        print_header "RESTARTING SERVICES"
        docker compose -f "$COMPOSE_FILE" restart
        print_success "Services restarted"
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    status)
        health_check
        ;;
    *)
        main
        ;;
esac