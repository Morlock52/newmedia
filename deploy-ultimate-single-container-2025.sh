#!/bin/bash
# Ultimate Media Server 2025 Single Container Deployment Script
# Automated deployment with comprehensive checks and optimization

set -euo pipefail

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
LOG_FILE="${SCRIPT_DIR}/deployment-$(date +%Y%m%d-%H%M%S).log"
CONTAINER_NAME="ultimate-media-server-2025"
IMAGE_NAME="ultimate-media-server:2025-single"
COMPOSE_FILE="docker-compose.ultimate-single-container-2025.yml"

# Default configuration
DEFAULT_CONFIG_PATH="./config"
DEFAULT_DATA_PATH="./data"
DEFAULT_MODELS_PATH="./models"
DEFAULT_DOMAIN="media.local"

# Deployment options
SKIP_BUILD=${SKIP_BUILD:-false}
SKIP_PULL=${SKIP_PULL:-false}
PRODUCTION_MODE=${PRODUCTION_MODE:-true}
ENABLE_MONITORING=${ENABLE_MONITORING:-true}
ENABLE_AI=${ENABLE_AI:-true}
QUICK_START=${QUICK_START:-false}

# Function definitions
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

info() { log "INFO" "${GREEN}$1${NC}"; }
warn() { log "WARN" "${YELLOW}$1${NC}"; }
error() { log "ERROR" "${RED}$1${NC}"; }
debug() { log "DEBUG" "${BLUE}$1${NC}"; }

banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🚀 ULTIMATE MEDIA SERVER 2025 - SINGLE CONTAINER DEPLOYMENT 🚀           ║
║                                                                              ║
║    Complete media server stack with 30+ services in one container           ║
║    • Jellyfin, Plex, Emby (Media Servers)                                   ║
║    • Sonarr, Radarr, Lidarr, Readarr, Bazarr, Prowlarr (*ARR Suite)        ║
║    • qBittorrent, Transmission, SABnzbd, NZBGet (Download Clients)          ║
║    • AI Assistant with Ollama integration                                   ║
║    • Monitoring with Prometheus, Grafana, Uptime Kuma                       ║
║    • And much more...                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_requirements() {
    info "🔍 Checking system requirements..."
    
    local required_commands=("docker" "docker-compose" "curl" "jq")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        error "Missing required commands: ${missing_commands[*]}"
        error "Please install the missing commands and try again."
        exit 1
    fi
    
    # Check Docker version
    local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    info "Docker version: $docker_version"
    
    # Check Docker Compose version
    local compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    info "Docker Compose version: $compose_version"
    
    # Check available disk space (macOS compatible)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        local available_space=$(df -g . | tail -1 | awk '{print $4}')
    else
        local available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    
    if [ -n "$available_space" ] && [ "$available_space" -lt 50 ]; then
        warn "Low disk space: ${available_space}GB available. Recommended: 100GB+"
    else
        info "Disk space: ${available_space}GB available"
    fi
    
    # Check available memory (macOS compatible)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # On macOS, use vm_stat
        local available_memory=$(( $(vm_stat | grep "Pages free:" | awk '{print $3}' | sed 's/\.//') * 4096 / 1073741824 ))
    else
        local available_memory=$(free -g | awk 'NR==2{printf "%.0f", $7}')
    fi
    
    if [ -n "$available_memory" ] && [ "$available_memory" -lt 8 ]; then
        warn "Low available memory: ${available_memory}GB. Recommended: 16GB+ total system memory"
    else
        info "Available memory: ${available_memory}GB"
    fi
    
    info "✅ System requirements check completed"
}

setup_environment() {
    info "⚙️ Setting up deployment environment..."
    
    # Create necessary directories
    local directories=(
        "$DEFAULT_CONFIG_PATH"
        "$DEFAULT_DATA_PATH"
        "$DEFAULT_MODELS_PATH"
        "$DEFAULT_DATA_PATH/media/movies"
        "$DEFAULT_DATA_PATH/media/tv"
        "$DEFAULT_DATA_PATH/media/music"
        "$DEFAULT_DATA_PATH/media/books"
        "$DEFAULT_DATA_PATH/media/audiobooks"
        "$DEFAULT_DATA_PATH/media/photos"
        "$DEFAULT_DATA_PATH/downloads"
        "$DEFAULT_DATA_PATH/backups"
        "./logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
    done
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        info "Creating default .env file..."
        cat > .env << EOF
# Ultimate Media Server 2025 Configuration
# Generated on $(date)

# System Settings
PUID=1000
PGID=1000
TZ=${TZ:-UTC}
DOMAIN=${DEFAULT_DOMAIN}

# Paths
CONFIG_PATH=${DEFAULT_CONFIG_PATH}
DATA_PATH=${DEFAULT_DATA_PATH}
AI_MODELS_PATH=${DEFAULT_MODELS_PATH}

# Security
API_KEY=$(openssl rand -hex 32)
REDIS_PASSWORD=$(openssl rand -hex 16)
POSTGRES_PASSWORD=$(openssl rand -hex 16)

# Features
AI_ENABLED=${ENABLE_AI}
ENABLE_MONITORING=${ENABLE_MONITORING}
ENABLE_HARDWARE_TRANSCODING=true
ENABLE_4K_TRANSCODING=true

# Performance
PYTHON_WORKERS=4

# External API Keys (fill in your own)
TMDB_API_KEY=
TVDB_API_KEY=
FANART_API_KEY=
OMDB_API_KEY=

# Notifications (optional)
DISCORD_WEBHOOK_URL=
SLACK_WEBHOOK_URL=
EMAIL_FROM=
EMAIL_TO=
EOF
        info "✅ Created .env file with secure defaults"
        warn "⚠️  Please review and customize the .env file before proceeding"
        
        if [ "$QUICK_START" != "true" ]; then
            read -p "Press Enter to continue after reviewing .env file..." -r
        fi
    else
        info "Using existing .env file"
    fi
    
    # Set proper permissions
    chmod 644 .env
    chmod -R 755 "$DEFAULT_CONFIG_PATH" "$DEFAULT_DATA_PATH" 2>/dev/null || true
    
    info "✅ Environment setup completed"
}

cleanup_previous() {
    info "🧹 Cleaning up previous deployment..."
    
    # Stop and remove existing container
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "Stopping existing container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" || true
        docker rm "$CONTAINER_NAME" || true
    fi
    
    # Clean up unused images (optional)
    if [ "$PRODUCTION_MODE" = "true" ]; then
        info "Cleaning up unused Docker images..."
        docker image prune -f || true
    fi
    
    info "✅ Cleanup completed"
}

build_image() {
    if [ "$SKIP_BUILD" = "true" ]; then
        info "⏭️ Skipping image build (SKIP_BUILD=true)"
        return 0
    fi
    
    info "🔨 Building Ultimate Media Server 2025 image..."
    
    local build_args=(
        "--file" "Dockerfile.ultimate-single-container-2025"
        "--tag" "$IMAGE_NAME"
        "--build-arg" "BUILDKIT_PROGRESS=plain"
    )
    
    if [ "$PRODUCTION_MODE" = "true" ]; then
        build_args+=("--build-arg" "NODE_ENV=production")
        build_args+=("--build-arg" "PYTHON_ENV=production")
    fi
    
    # Add caching for faster subsequent builds
    build_args+=("--build-arg" "BUILDKIT_INLINE_CACHE=1")
    
    info "Build command: docker build ${build_args[*]} ."
    
    if docker build "${build_args[@]}" .; then
        info "✅ Image build completed successfully"
    else
        error "❌ Image build failed"
        exit 1
    fi
}

deploy_container() {
    info "🚀 Deploying Ultimate Media Server 2025..."
    
    # Source environment variables
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi
    
    # Deploy using docker-compose
    if docker-compose -f "$COMPOSE_FILE" up -d; then
        info "✅ Container deployment initiated"
    else
        error "❌ Container deployment failed"
        exit 1
    fi
    
    info "⏳ Waiting for services to initialize..."
    sleep 30
    
    # Check container status
    if docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -q "$CONTAINER_NAME.*Up"; then
        info "✅ Container is running"
    else
        error "❌ Container failed to start"
        docker logs "$CONTAINER_NAME" --tail 50
        exit 1
    fi
}

perform_health_checks() {
    info "🏥 Performing comprehensive health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        info "Health check attempt $attempt/$max_attempts"
        
        if docker exec "$CONTAINER_NAME" /app/healthcheck.sh; then
            info "✅ Health check passed"
            break
        else
            warn "⚠️  Health check failed, retrying in 30 seconds..."
            sleep 30
            ((attempt++))
        fi
        
        if [ $attempt -gt $max_attempts ]; then
            error "❌ Health checks failed after $max_attempts attempts"
            error "Container logs:"
            docker logs "$CONTAINER_NAME" --tail 100
            exit 1
        fi
    done
}

display_access_info() {
    info "📋 Deployment completed successfully!"
    
    # Get container IP
    local container_ip=$(docker inspect "$CONTAINER_NAME" | jq -r '.[0].NetworkSettings.Networks[].IPAddress' | head -1)
    
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                          🎉 ACCESS INFORMATION 🎉                           ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📊 MAIN DASHBOARD:                                                          ║${NC}"
    echo -e "${GREEN}║     http://localhost                                                         ║${NC}"
    echo -e "${GREEN}║     http://${container_ip}                                                   ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  🎬 MEDIA SERVERS:                                                           ║${NC}"
    echo -e "${GREEN}║     Jellyfin:    http://localhost:8096                                      ║${NC}"
    echo -e "${GREEN}║     Plex:        http://localhost:32400/web                                 ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📺 MEDIA MANAGEMENT:                                                        ║${NC}"
    echo -e "${GREEN}║     Sonarr:      http://localhost:8989                                      ║${NC}"
    echo -e "${GREEN}║     Radarr:      http://localhost:7878                                      ║${NC}"
    echo -e "${GREEN}║     Prowlarr:    http://localhost:9696                                      ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  ⬇️  DOWNLOADS:                                                               ║${NC}"
    echo -e "${GREEN}║     qBittorrent: http://localhost:8080 (admin/adminadmin)                   ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  🤖 AI ASSISTANT:                                                            ║${NC}"
    echo -e "${GREEN}║     AI API:      http://localhost:8090                                      ║${NC}"
    echo -e "${GREEN}║     Ollama:      http://localhost:11434                                     ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📊 MONITORING:                                                              ║${NC}"
    echo -e "${GREEN}║     Traefik:     http://localhost:8088                                      ║${NC}"
    echo -e "${GREEN}║     Grafana:     http://localhost:3000 (admin/admin)                        ║${NC}"
    echo -e "${GREEN}║     Prometheus:  http://localhost:9090                                      ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📈 MANAGEMENT:                                                              ║${NC}"
    echo -e "${GREEN}║     Portainer:   http://localhost:9000                                      ║${NC}"
    echo -e "${GREEN}║     Uptime Kuma: http://localhost:3001                                      ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${CYAN}🔧 MANAGEMENT COMMANDS:${NC}"
    echo -e "  View logs:           ${YELLOW}docker logs $CONTAINER_NAME -f${NC}"
    echo -e "  Restart container:   ${YELLOW}docker restart $CONTAINER_NAME${NC}"
    echo -e "  Stop container:      ${YELLOW}docker stop $CONTAINER_NAME${NC}"
    echo -e "  Health check:        ${YELLOW}docker exec $CONTAINER_NAME /app/healthcheck.sh${NC}"
    echo -e "  Shell access:        ${YELLOW}docker exec -it $CONTAINER_NAME /bin/bash${NC}"
    
    echo -e "\n${PURPLE}📖 DOCUMENTATION:${NC}"
    echo -e "  Configuration:       ${YELLOW}./config/${NC}"
    echo -e "  Data directory:      ${YELLOW}./data/${NC}"
    echo -e "  AI models:           ${YELLOW}./models/${NC}"
    echo -e "  Deployment log:      ${YELLOW}$LOG_FILE${NC}"
    
    echo -e "\n${GREEN}✅ Ultimate Media Server 2025 is ready to use!${NC}"
}

optimize_system() {
    if [ "$PRODUCTION_MODE" = "true" ]; then
        info "⚡ Applying production optimizations..."
        
        # Set Docker logging limits
        info "Configuring Docker logging limits..."
        
        # Set system limits (if running as root or with sudo)
        if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
            info "Applying system optimizations..."
            
            # Increase inotify limits for file watching
            echo 'fs.inotify.max_user_watches=1048576' | sudo tee -a /etc/sysctl.conf > /dev/null
            echo 'fs.inotify.max_user_instances=1024' | sudo tee -a /etc/sysctl.conf > /dev/null
            
            # Apply immediately
            sudo sysctl -p || true
            
            info "✅ System optimizations applied"
        else
            warn "⚠️  Run as root or with sudo for additional system optimizations"
        fi
    fi
}

setup_monitoring() {
    if [ "$ENABLE_MONITORING" = "true" ]; then
        info "📊 Setting up enhanced monitoring..."
        
        # Create monitoring configuration
        mkdir -p ./config/monitoring
        
        # Basic monitoring script
        cat > ./monitoring-check.sh << 'EOF'
#!/bin/bash
# Simple monitoring check for Ultimate Media Server 2025

CONTAINER_NAME="ultimate-media-server-2025"

echo "=== Container Status ==="
docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n=== Container Resources ==="
docker stats "$CONTAINER_NAME" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo -e "\n=== Health Check ==="
docker exec "$CONTAINER_NAME" /app/healthcheck.sh 2>/dev/null && echo "✅ Healthy" || echo "❌ Unhealthy"

echo -e "\n=== Recent Logs ==="
docker logs "$CONTAINER_NAME" --since 1h --tail 20
EOF
        
        chmod +x ./monitoring-check.sh
        info "✅ Monitoring setup completed"
        info "📊 Run './monitoring-check.sh' to check system status"
    fi
}

# Main execution flow
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --quick-start)
                QUICK_START=true
                shift
                ;;
            --no-ai)
                ENABLE_AI=false
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --dev)
                PRODUCTION_MODE=false
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-build        Skip Docker image build"
                echo "  --quick-start       Skip interactive prompts"
                echo "  --no-ai            Disable AI features"
                echo "  --no-monitoring    Disable monitoring setup"
                echo "  --dev              Development mode"
                echo "  -h, --help         Show this help"
                exit 0
                ;;
            *)
                warn "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Main deployment sequence
    banner
    
    info "🚀 Starting Ultimate Media Server 2025 deployment..."
    info "📁 Working directory: $(pwd)"
    info "📝 Log file: $LOG_FILE"
    
    check_requirements
    setup_environment
    cleanup_previous
    build_image
    deploy_container
    optimize_system
    setup_monitoring
    perform_health_checks
    display_access_info
    
    info "🎉 Deployment completed successfully!"
    info "📝 Full deployment log available at: $LOG_FILE"
}

# Error handling
trap 'error "❌ Deployment failed at line $LINENO. Check $LOG_FILE for details."' ERR

# Execute main function
main "$@"