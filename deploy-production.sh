#!/bin/bash
# Ultimate Media Server Stack - Production Deployment Script
# ==========================================================
# Version: 3.0.0 - Production-Ready Deployment
# Features: Security validation, health checks, automated setup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.production"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.production.yml"
LOG_FILE="$SCRIPT_DIR/deployment.log"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

section() {
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════${NC}\n"
}

# Banner
show_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
 _____ _____ ____  ___    _       __  __ _____ ____  ___    _    
|     | __  |    \|_ _|  / \     |  \/  | ____|  _ \|_ _|  / \   
| | | |  __/|  |  || |  / _ \    | |\/| |  _| | | | || |  / _ \  
| | | | |___|  |  || | / ___ \   | |  | | |___| |_| || | / ___ \ 
|_|_|_|_____|____/|___/_/   \_\  |_|  |_|_____|____/|___/_/   \_\

 _____ _____ ____  __     _____ ____     ____ _____ _   _ _   _ 
|     | ____|  _ \ \ \   / /____|  _ \   / ___|_   _| | | | | | |
| | | |  _| | | | | \ \ / / _  | |_) |  \___ \ | | | | | | | | |
| | | | |___| |_| |  \ V / |_| |  _ <    ___) || | | |_| | |_| |
|_|_|_|_____|____/    \_/ \____|_| \_\  |____/ |_|  \___/ \___/ 

Ultimate Media Server Stack - Production Deployment v3.0.0
EOF
    echo -e "${NC}\n"
}

# Prerequisites check
check_prerequisites() {
    section "CHECKING PREREQUISITES"
    
    local errors=0
    
    # Check Docker
    if command -v docker >/dev/null 2>&1; then
        local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log "Docker version: $docker_version"
        
        # Check if Docker is running
        if docker info >/dev/null 2>&1; then
            success "Docker is running"
        else
            error "Docker is not running"
            ((errors++))
        fi
    else
        error "Docker is not installed"
        ((errors++))
    fi
    
    # Check Docker Compose
    if command -v docker-compose >/dev/null 2>&1; then
        local compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log "Docker Compose version: $compose_version"
    else
        error "Docker Compose is not installed"
        ((errors++))
    fi
    
    # Check available disk space
    local available_space=$(df "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -gt 50 ]]; then
        success "Available disk space: ${available_gb}GB"
    else
        warn "Low disk space: ${available_gb}GB (recommended: 50GB+)"
    fi
    
    # Check memory
    local total_memory=$(free -g | grep '^Mem:' | awk '{print $2}')
    if [[ $total_memory -gt 8 ]]; then
        success "Total memory: ${total_memory}GB"
    else
        warn "Low memory: ${total_memory}GB (recommended: 8GB+)"
    fi
    
    # Check required files
    local required_files=(
        "$COMPOSE_FILE"
        "Dockerfile.production"
        "docker/traefik/dynamic/dynamic.yml"
        "docker/authelia/configuration.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/$file" ]]; then
            success "Found: $file"
        else
            error "Missing: $file"
            ((errors++))
        fi
    done
    
    if [[ $errors -gt 0 ]]; then
        error "Prerequisites check failed with $errors errors"
        exit 1
    else
        success "All prerequisites satisfied"
    fi
}

# Environment configuration
setup_environment() {
    section "ENVIRONMENT SETUP"
    
    # Check if production environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        warn "Production environment file not found"
        
        if [[ -f "$SCRIPT_DIR/.env.production.template" ]]; then
            log "Copying template to production environment file"
            cp "$SCRIPT_DIR/.env.production.template" "$ENV_FILE"
            warn "Please edit $ENV_FILE and replace all CHANGEME values"
            info "Critical items to configure:"
            info "  - DOMAIN (your domain name)"
            info "  - CF_API_EMAIL and CF_API_KEY (Cloudflare credentials)"
            info "  - Database passwords"
            info "  - VPN credentials"
            info "  - Admin passwords"
            echo -e "\nPress Enter after configuring the environment file..."
            read -r
        else
            error "No environment template found"
            exit 1
        fi
    fi
    
    # Validate environment file
    log "Validating environment configuration..."
    
    # Source the environment file
    if ! source "$ENV_FILE"; then
        error "Failed to source environment file"
        exit 1
    fi
    
    # Check for CHANGEME values
    local changeme_count=$(grep -c "CHANGEME" "$ENV_FILE" || true)
    if [[ $changeme_count -gt 0 ]]; then
        error "Found $changeme_count CHANGEME values in environment file"
        error "Please replace all CHANGEME values with actual configuration"
        exit 1
    fi
    
    # Validate critical variables
    local required_vars=(
        "DOMAIN"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "AUTHELIA_JWT_SECRET"
        "CF_API_EMAIL"
        "CF_API_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Missing required environment variable: $var"
            exit 1
        else
            success "✓ $var is set"
        fi
    done
    
    success "Environment configuration validated"
}

# Security setup
setup_security() {
    section "SECURITY SETUP"
    
    # Generate Authelia password hashes
    log "Setting up Authelia authentication..."
    
    # Check if users database needs password hashes
    local users_file="$SCRIPT_DIR/docker/authelia/users_database.yml"
    if grep -q "CHANGEME_GENERATE_HASH" "$users_file" 2>/dev/null; then
        warn "Authelia users database contains placeholder hashes"
        info "To generate password hashes, run:"
        info "  docker run --rm authelia/authelia:latest authelia hash-password 'your-password'"
        info "Then update $users_file with the generated hashes"
    fi
    
    # Create SSL certificate directories
    log "Setting up SSL certificate storage..."
    mkdir -p "$SCRIPT_DIR/docker/traefik/letsencrypt"
    chmod 700 "$SCRIPT_DIR/docker/traefik/letsencrypt"
    
    # Set secure permissions on configuration directories
    log "Setting secure permissions..."
    find "$SCRIPT_DIR/docker" -name "*.yml" -exec chmod 600 {} \;
    find "$SCRIPT_DIR/docker" -type d -exec chmod 700 {} \;
    
    success "Security setup completed"
}

# Network setup
setup_networks() {
    section "NETWORK SETUP"
    
    log "Creating Docker networks..."
    
    # Remove existing networks if they exist
    local networks=("public" "frontend" "backend" "storage" "downloads" "monitoring")
    
    for network in "${networks[@]}"; do
        if docker network ls | grep -q "media_${network}"; then
            log "Removing existing network: media_${network}"
            docker network rm "media_${network}" || true
        fi
    done
    
    success "Network setup completed"
}

# Build custom images
build_images() {
    section "BUILDING CUSTOM IMAGES"
    
    log "Building production images..."
    
    # Build multi-stage images
    local images=(
        "api-service"
        "web-runtime"
        "monitoring-agent"
        "backup-agent"
        "init-container"
    )
    
    for image in "${images[@]}"; do
        log "Building $image..."
        if docker build --target "$image" -t "media-$image:latest" -f Dockerfile.production .; then
            success "Built: media-$image:latest"
        else
            error "Failed to build: media-$image:latest"
            exit 1
        fi
    done
    
    success "All custom images built successfully"
}

# Deploy services
deploy_services() {
    section "DEPLOYING SERVICES"
    
    log "Starting production deployment..."
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    
    # Start infrastructure services first
    log "Starting infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        postgres redis traefik
    
    # Wait for infrastructure to be ready
    log "Waiting for infrastructure services..."
    sleep 30
    
    # Start authentication service
    log "Starting authentication service..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d authelia
    sleep 15
    
    # Start media services
    log "Starting media services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        jellyfin overseerr tautulli
    
    # Start management services
    log "Starting management services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        prowlarr sonarr radarr lidarr readarr bazarr
    
    # Start download services
    log "Starting download services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        vpn qbittorrent sabnzbd
    
    # Start monitoring services
    log "Starting monitoring services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        prometheus grafana loki promtail
    
    # Start management interfaces
    log "Starting management interfaces..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        homepage portainer duplicati watchtower
    
    success "All services deployed"
}

# Health checks
run_health_checks() {
    section "HEALTH CHECKS"
    
    log "Running comprehensive health checks..."
    
    # Wait for services to fully start
    log "Waiting for services to initialize..."
    sleep 60
    
    # Check service health
    local services=(
        "postgres:5432"
        "redis:6379"
        "traefik:80"
        "authelia:9091"
        "jellyfin:8096"
        "prometheus:9090"
        "grafana:3000"
    )
    
    local healthy_services=0
    
    for service in "${services[@]}"; do
        local name=${service%:*}
        local port=${service#*:}
        
        log "Checking $name..."
        
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
           exec -T "$name" nc -z localhost "$port" 2>/dev/null; then
            success "$name is healthy"
            ((healthy_services++))
        else
            error "$name health check failed"
        fi
    done
    
    log "Health check summary: $healthy_services/${#services[@]} services healthy"
    
    if [[ $healthy_services -eq ${#services[@]} ]]; then
        success "All services are healthy!"
    else
        warn "Some services failed health checks - check logs for details"
    fi
}

# Show service URLs
show_urls() {
    section "SERVICE ACCESS INFORMATION"
    
    # Source environment to get domain
    source "$ENV_FILE"
    
    echo -e "${CYAN}Your media stack is accessible at:${NC}\n"
    
    # Media services
    echo -e "${GREEN}Media Services:${NC}"
    echo -e "  🎬 Jellyfin:    https://jellyfin.${DOMAIN}"
    echo -e "  📺 Overseerr:   https://overseerr.${DOMAIN}"
    echo -e "  📊 Tautulli:    https://tautulli.${DOMAIN}"
    echo ""
    
    # Management services
    echo -e "${GREEN}Management:${NC}"
    echo -e "  📋 Dashboard:   https://${DOMAIN}"
    echo -e "  🔧 Portainer:   https://portainer.${DOMAIN}"
    echo -e "  📈 Grafana:     https://grafana.${DOMAIN}"
    echo -e "  🔒 Authelia:    https://auth.${DOMAIN}"
    echo ""
    
    # Download services (admin only)
    echo -e "${YELLOW}Download Services (Admin Only):${NC}"
    echo -e "  🌊 qBittorrent: https://qbittorrent.${DOMAIN}"
    echo -e "  📡 Prowlarr:    https://prowlarr.${DOMAIN}"
    echo -e "  📺 Sonarr:      https://sonarr.${DOMAIN}"
    echo -e "  🎬 Radarr:      https://radarr.${DOMAIN}"
    echo ""
    
    echo -e "${CYAN}Default credentials:${NC}"
    echo -e "  Username: admin"
    echo -e "  Password: Check your .env.production file"
    echo ""
    
    warn "IMPORTANT: Change all default passwords immediately!"
    warn "Configure your services through their web interfaces"
    warn "Set up indexers in Prowlarr first, then configure other *arr services"
}

# Cleanup function
cleanup() {
    if [[ $? -ne 0 ]]; then
        error "Deployment failed. Check $LOG_FILE for details."
        echo -e "\n${CYAN}To troubleshoot:${NC}"
        echo -e "  1. Check logs: docker-compose -f $COMPOSE_FILE logs [service]"
        echo -e "  2. Restart services: docker-compose -f $COMPOSE_FILE restart"
        echo -e "  3. Reset stack: docker-compose -f $COMPOSE_FILE down -v"
    fi
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Initialize log file
    echo "Deployment started at $(date)" > "$LOG_FILE"
    
    show_banner
    check_prerequisites
    setup_environment
    setup_security
    setup_networks
    build_images
    deploy_services
    run_health_checks
    show_urls
    
    success "🎉 Deployment completed successfully!"
    info "Logs saved to: $LOG_FILE"
    info "To manage your stack: docker-compose -f $COMPOSE_FILE [command]"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        section "STOPPING SERVICES"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
        success "All services stopped"
        ;;
    "restart")
        section "RESTARTING SERVICES"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
        success "All services restarted"
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f "${2:-}"
        ;;
    "update")
        section "UPDATING SERVICES"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
        success "Services updated"
        ;;
    "backup")
        section "BACKUP CONFIGURATION"
        tar -czf "media-stack-backup-$(date +%Y%m%d-%H%M%S).tar.gz" \
            config/ docker/ .env.production docker-compose.production.yml
        success "Backup created"
        ;;
    "reset")
        warn "This will destroy all data and containers!"
        read -p "Are you sure? (type 'yes' to confirm): " confirm
        if [[ "$confirm" == "yes" ]]; then
            docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v
            docker system prune -f
            success "Stack reset completed"
        else
            info "Reset cancelled"
        fi
        ;;
    "help"|"--help"|"-h")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the full stack (default)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show logs (optionally for specific service)"
        echo "  update   - Update and restart services"
        echo "  backup   - Create configuration backup"
        echo "  reset    - Reset entire stack (destructive)"
        echo "  help     - Show this help"
        ;;
    *)
        error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac