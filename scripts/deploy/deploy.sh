#!/bin/bash

# Media Server - Deployment Script
# Handles Docker container deployment and orchestration
# Version: 1.0.0

set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
readonly COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
readonly OVERRIDE_FILE="${PROJECT_ROOT}/docker-compose.override.yml"
readonly ENV_FILE="${PROJECT_ROOT}/.env"

# Deployment options
DEPLOYMENT_MODE="${1:-full}"
FORCE_RECREATE=false
PULL_IMAGES=true
REMOVE_ORPHANS=true

# Service groups
readonly CORE_SERVICES=(
    "jellyfin"
    "qbittorrent"
    "homepage"
)

readonly MEDIA_MANAGEMENT=(
    "radarr"
    "sonarr"
    "prowlarr"
    "bazarr"
)

readonly UTILITY_SERVICES=(
    "overseerr"
    "tautulli"
    "portainer"
)

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}==== $1 ====${NC}"
}

# Show deployment banner
show_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    ____             __                                __ 
   / __ \___  ____  / /___  __  ______ ___  ___  ____  / /_
  / / / / _ \/ __ \/ / __ \/ / / / __ `__ \/ _ \/ __ \/ __/
 / /_/ /  __/ /_/ / / /_/ / /_/ / / / / / /  __/ / / / /_  
/_____/\___/ .___/_/\____/\__, /_/ /_/ /_/\___/_/ /_/\__/  
          /_/            /____/                             
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Media Server Deployment System${NC}"
    echo -e "${CYAN}Mode: ${DEPLOYMENT_MODE}${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_header "Pre-deployment Checks"
    
    # Check Docker
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        log_warn "Please start Docker Desktop and try again"
        return 1
    fi
    log_info "✓ Docker daemon is running"
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose not found"
        return 1
    fi
    log_info "✓ docker-compose is available"
    
    # Check compose files
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found at $COMPOSE_FILE"
        return 1
    fi
    log_info "✓ Docker Compose file found"
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_warn "Environment file not found"
        log_info "Running setup script..."
        "${SCRIPT_DIR}/setup.sh"
    else
        log_info "✓ Environment file found"
    fi
    
    # Check required directories
    source "$ENV_FILE"
    local required_dirs=("$CONFIG_PATH" "$DATA_PATH")
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_warn "Creating missing directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    return 0
}

# Clean up existing containers
cleanup_existing() {
    log_header "Cleaning Up Existing Containers"
    
    local compose_project="${COMPOSE_PROJECT_NAME:-newmedia}"
    
    # Check for existing containers
    local existing=$(docker ps -aq --filter "label=com.docker.compose.project=$compose_project" | wc -l)
    
    if [ "$existing" -gt 0 ]; then
        log_info "Found $existing existing containers"
        
        if [ "$FORCE_RECREATE" = true ]; then
            log_info "Force recreate enabled, removing existing containers"
            docker-compose down --remove-orphans
        else
            read -p "Remove existing containers? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose down --remove-orphans
                log_info "Existing containers removed"
            fi
        fi
    else
        log_info "No existing containers found"
    fi
}

# Pull latest images
pull_images() {
    if [ "$PULL_IMAGES" = true ]; then
        log_header "Pulling Latest Images"
        
        # Get list of services to deploy
        local services=()
        case "$DEPLOYMENT_MODE" in
            "core")
                services=("${CORE_SERVICES[@]}")
                ;;
            "media")
                services=("${CORE_SERVICES[@]}" "${MEDIA_MANAGEMENT[@]}")
                ;;
            "full")
                services=("${CORE_SERVICES[@]}" "${MEDIA_MANAGEMENT[@]}" "${UTILITY_SERVICES[@]}")
                ;;
        esac
        
        # Pull images for selected services
        for service in "${services[@]}"; do
            log_info "Pulling image for $service..."
            docker-compose pull "$service" 2>/dev/null || log_warn "Failed to pull $service image"
        done
    fi
}

# Deploy services
deploy_services() {
    log_header "Deploying Services"
    
    local compose_cmd="docker-compose"
    
    # Add override file if it exists
    if [ -f "$OVERRIDE_FILE" ]; then
        compose_cmd="$compose_cmd -f $COMPOSE_FILE -f $OVERRIDE_FILE"
    fi
    
    # Build service list based on deployment mode
    local services=""
    case "$DEPLOYMENT_MODE" in
        "core")
            services="${CORE_SERVICES[*]}"
            log_info "Deploying core services only"
            ;;
        "media")
            services="${CORE_SERVICES[*]} ${MEDIA_MANAGEMENT[*]}"
            log_info "Deploying core + media management services"
            ;;
        "full")
            log_info "Deploying all services"
            ;;
        *)
            log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
            return 1
            ;;
    esac
    
    # Deploy services
    if [ -z "$services" ]; then
        # Full deployment
        $compose_cmd up -d
    else
        # Selective deployment
        $compose_cmd up -d $services
    fi
    
    # Check deployment status
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_info "Services deployed successfully"
    else
        log_error "Deployment failed with exit code: $exit_code"
        return $exit_code
    fi
}

# Wait for services to be healthy
wait_for_services() {
    log_header "Waiting for Services to Initialize"
    
    local max_attempts=30
    local attempt=0
    
    # Get list of deployed services
    local services=$(docker-compose ps --services)
    
    for service in $services; do
        log_info "Checking $service..."
        attempt=0
        
        while [ $attempt -lt $max_attempts ]; do
            if docker-compose ps "$service" | grep -q "Up"; then
                log_info "✓ $service is running"
                break
            fi
            
            ((attempt++))
            sleep 2
        done
        
        if [ $attempt -eq $max_attempts ]; then
            log_warn "⚠ $service failed to start within timeout"
        fi
    done
}

# Post-deployment configuration
post_deployment_config() {
    log_header "Post-Deployment Configuration"
    
    # Wait for services to be ready
    sleep 10
    
    # Get API keys from services
    log_info "Retrieving API keys from services..."
    
    # Update .env with API keys if services are running
    if docker ps --format "table {{.Names}}" | grep -q "radarr"; then
        local radarr_key=$(docker exec radarr cat /config/config.xml 2>/dev/null | grep -oP '(?<=<ApiKey>)[^<]+' || echo "")
        if [ -n "$radarr_key" ]; then
            sed -i.bak "s/RADARR_API_KEY=.*/RADARR_API_KEY=$radarr_key/" "$ENV_FILE"
            log_info "✓ Retrieved Radarr API key"
        fi
    fi
    
    if docker ps --format "table {{.Names}}" | grep -q "sonarr"; then
        local sonarr_key=$(docker exec sonarr cat /config/config.xml 2>/dev/null | grep -oP '(?<=<ApiKey>)[^<]+' || echo "")
        if [ -n "$sonarr_key" ]; then
            sed -i.bak "s/SONARR_API_KEY=.*/SONARR_API_KEY=$sonarr_key/" "$ENV_FILE"
            log_info "✓ Retrieved Sonarr API key"
        fi
    fi
    
    # Clean up backup files
    rm -f "${ENV_FILE}.bak"
}

# Show deployment summary
show_summary() {
    log_header "Deployment Summary"
    
    # Get running containers
    local running_count=$(docker-compose ps --services | wc -l)
    
    echo -e "${GREEN}✅ Deployment completed!${NC}\n"
    echo -e "${CYAN}Services Status:${NC}"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\n${CYAN}Access Points:${NC}"
    echo "  📊 Dashboard:    http://localhost:3000"
    echo "  📺 Jellyfin:     http://localhost:8096"
    echo "  ⬇️  qBittorrent:  http://localhost:8080"
    
    if [ "$DEPLOYMENT_MODE" != "core" ]; then
        echo "  🎬 Radarr:       http://localhost:7878"
        echo "  📺 Sonarr:       http://localhost:8989"
        echo "  🔍 Prowlarr:     http://localhost:9696"
    fi
    
    if [ "$DEPLOYMENT_MODE" = "full" ]; then
        echo "  📋 Overseerr:    http://localhost:5055"
        echo "  📊 Tautulli:     http://localhost:8181"
        echo "  🐳 Portainer:    http://localhost:9000"
    fi
    
    echo -e "\n${CYAN}Quick Commands:${NC}"
    echo "  View logs:    docker-compose logs -f [service]"
    echo "  Stop all:     docker-compose stop"
    echo "  Start all:    docker-compose start"
    echo "  Remove all:   docker-compose down"
    
    echo -e "\n${YELLOW}First-time Setup:${NC}"
    echo "  1. Configure Jellyfin at http://localhost:8096"
    echo "  2. Set up qBittorrent at http://localhost:8080 (default: admin/adminadmin)"
    echo "  3. Connect services through Prowlarr"
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            core|media|full)
                DEPLOYMENT_MODE="$1"
                ;;
            --force|-f)
                FORCE_RECREATE=true
                ;;
            --no-pull)
                PULL_IMAGES=false
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

# Show help
show_help() {
    cat << EOF
Usage: $0 [mode] [options]

Deployment Modes:
  core    Deploy only core services (Jellyfin, qBittorrent, Homepage)
  media   Deploy core + media management services
  full    Deploy all services (default)

Options:
  -f, --force      Force recreate containers
  --no-pull        Skip pulling latest images
  -h, --help       Show this help message

Examples:
  $0              # Full deployment
  $0 core         # Core services only
  $0 full --force # Force recreate all containers
EOF
}

# Main deployment flow
main() {
    parse_arguments "$@"
    show_banner
    
    # Run deployment steps
    check_prerequisites || exit 1
    cleanup_existing
    pull_images
    deploy_services || exit 1
    wait_for_services
    post_deployment_config
    
    show_summary
}

# Run main function
main "$@"