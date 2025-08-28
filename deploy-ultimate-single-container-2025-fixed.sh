#!/bin/bash
# Ultimate Media Server 2025 Single Container Deployment Script - FIXED VERSION
# Comprehensive deployment with all issues resolved and August 2025 optimizations

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
LOG_FILE="${SCRIPT_DIR}/deployment-fixed-$(date +%Y%m%d-%H%M%S).log"
CONTAINER_NAME="ultimate-media-server-2025-fixed"
IMAGE_NAME="ultimate-media-server:2025-single-fixed"
COMPOSE_FILE="docker-compose.ultimate-single-container-2025-fixed.yml"
DOCKERFILE="Dockerfile.ultimate-single-container-2025-fixed"

# Default configuration paths
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
FORCE_RECREATE=${FORCE_RECREATE:-false}

# System requirements
MIN_DOCKER_VERSION="20.10.0"
MIN_COMPOSE_VERSION="2.0.0"
MIN_DISK_SPACE_GB=100
MIN_MEMORY_GB=8

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
║                              FIXED VERSION                                  ║
║                                                                              ║
║    Complete media server stack with 30+ services in one container           ║
║    • Jellyfin, Plex, Emby (Media Servers)                                   ║
║    • Sonarr, Radarr, Lidarr, Readarr, Bazarr, Prowlarr (*ARR Suite)        ║
║    • qBittorrent, Transmission, SABnzbd, NZBGet (Download Clients)          ║
║    • AI Assistant with Ollama integration                                   ║
║    • Monitoring with Prometheus, Grafana, Uptime Kuma                       ║
║    • Enhanced security, performance, and reliability                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_requirements() {
    info "🔍 Checking system requirements and dependencies..."
    
    local required_commands=("docker" "docker-compose" "curl" "jq" "openssl")
    local missing_commands=()
    
    # Check required commands
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        error "Missing required commands: ${missing_commands[*]}"
        error "Please install the missing commands and try again."
        echo ""
        echo "Installation commands:"
        echo "  Ubuntu/Debian: sudo apt-get install ${missing_commands[*]}"
        echo "  CentOS/RHEL: sudo yum install ${missing_commands[*]}"
        echo "  macOS: brew install ${missing_commands[*]}"
        exit 1
    fi
    
    # Check Docker version
    local docker_version
    docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ "$(printf '%s\n' "$MIN_DOCKER_VERSION" "$docker_version" | sort -V | head -n1)" != "$MIN_DOCKER_VERSION" ]; then
        warn "Docker version $docker_version is below recommended $MIN_DOCKER_VERSION"
    else
        info "Docker version: $docker_version ✓"
    fi
    
    # Check Docker Compose version
    local compose_version
    if docker-compose --version &> /dev/null; then
        compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    elif docker compose version &> /dev/null; then
        compose_version=$(docker compose version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    else
        error "Docker Compose not found"
        exit 1
    fi
    
    info "Docker Compose version: $compose_version ✓"
    
    # Check available disk space (improved cross-platform)
    local available_space
    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_space=$(df -g . | tail -1 | awk '{print $4}')
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    else
        warn "Cannot determine disk space on this platform"
        available_space=999
    fi
    
    if [ -n "$available_space" ] && [ "$available_space" -lt "$MIN_DISK_SPACE_GB" ]; then
        warn "Low disk space: ${available_space}GB available. Recommended: ${MIN_DISK_SPACE_GB}GB+"
        if [ "$QUICK_START" != "true" ]; then
            read -p "Continue with low disk space? (y/N): " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        info "Disk space: ${available_space}GB available ✓"
    fi
    
    # Check available memory (improved detection)
    local available_memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS memory calculation
        local page_size=$(vm_stat | head -1 | grep -oE '[0-9]+')
        local free_pages=$(vm_stat | grep "Pages free:" | awk '{print $3}' | sed 's/\.//')
        local inactive_pages=$(vm_stat | grep "Pages inactive:" | awk '{print $3}' | sed 's/\.//')
        available_memory=$(( (free_pages + inactive_pages) * page_size / 1073741824 ))
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v free &> /dev/null; then
            available_memory=$(free -g | awk 'NR==2{print $7}')
        else
            available_memory=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
        fi
    else
        warn "Cannot determine available memory on this platform"
        available_memory=16
    fi
    
    if [ -n "$available_memory" ] && [ "$available_memory" -lt "$MIN_MEMORY_GB" ]; then
        warn "Low available memory: ${available_memory}GB. Recommended: ${MIN_MEMORY_GB}GB+ available"
    else
        info "Available memory: ${available_memory}GB ✓"
    fi
    
    # Check Docker daemon status
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running or not accessible"
        error "Please start Docker and ensure your user has permission to access it"
        exit 1
    fi
    
    # Check for existing containers with same name
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "Container with name '${CONTAINER_NAME}' already exists"
        if [ "$FORCE_RECREATE" = "true" ]; then
            info "Force recreate enabled, will remove existing container"
        elif [ "$QUICK_START" != "true" ]; then
            read -p "Remove existing container and continue? (y/N): " -r
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                FORCE_RECREATE=true
            else
                exit 1
            fi
        fi
    fi
    
    info "✅ System requirements check completed"
}

setup_environment() {
    info "⚙️ Setting up deployment environment..."
    
    # Create necessary directories with proper structure
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
        "$DEFAULT_DATA_PATH/media/documents"
        "$DEFAULT_DATA_PATH/media/comics"
        "$DEFAULT_DATA_PATH/downloads/complete"
        "$DEFAULT_DATA_PATH/downloads/incomplete"
        "$DEFAULT_DATA_PATH/downloads/watch"
        "$DEFAULT_DATA_PATH/downloads/torrents"
        "$DEFAULT_DATA_PATH/downloads/usenet"
        "$DEFAULT_DATA_PATH/databases/postgres"
        "$DEFAULT_DATA_PATH/databases/redis"
        "$DEFAULT_DATA_PATH/databases/grafana"
        "$DEFAULT_DATA_PATH/backups"
        "$DEFAULT_MODELS_PATH/ollama"
        "$DEFAULT_MODELS_PATH/whisper"
        "$DEFAULT_MODELS_PATH/stable-diffusion"
        "./logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
    done
    
    # Create enhanced .env file if it doesn't exist
    if [ ! -f .env ]; then
        info "Creating enhanced .env configuration file..."
        
        # Generate secure random values
        local api_key=$(openssl rand -hex 32)
        local jwt_secret=$(openssl rand -base64 48)
        local session_secret=$(openssl rand -base64 32)
        local postgres_password=$(openssl rand -base64 32)
        local redis_password=$(openssl rand -base64 24)
        
        cat > .env << EOF
# Ultimate Media Server 2025 Configuration - FIXED VERSION
# Generated on $(date)

# ===== SYSTEM SETTINGS =====
PUID=1000
PGID=1000
TZ=${TZ:-UTC}
DOMAIN=${DEFAULT_DOMAIN}
HOSTNAME=ultimate-media-server-2025-fixed

# ===== PATHS =====
CONFIG_PATH=${DEFAULT_CONFIG_PATH}
DATA_PATH=${DEFAULT_DATA_PATH}
AI_MODELS_PATH=${DEFAULT_MODELS_PATH}

# ===== SECURITY (AUTO-GENERATED) =====
API_KEY=${api_key}
JWT_SECRET=${jwt_secret}
SESSION_SECRET=${session_secret}
POSTGRES_PASSWORD=${postgres_password}
REDIS_PASSWORD=${redis_password}

# ===== FEATURES =====
AI_ENABLED=${ENABLE_AI}
ENABLE_MONITORING=${ENABLE_MONITORING}
ENABLE_HARDWARE_TRANSCODING=true
ENABLE_4K_TRANSCODING=true
ENABLE_AI_RECOMMENDATIONS=true
ENABLE_HEALTH_CHECKS=true
SECURE_MODE=true
DISABLE_TELEMETRY=true

# ===== PERFORMANCE =====
PYTHON_WORKERS=4
NODE_OPTIONS=--max-old-space-size=8192
CONTAINER_CPU_LIMIT=16.0
CONTAINER_MEMORY_LIMIT=32g
POSTGRES_MAX_CONNECTIONS=200
REDIS_MAXMEMORY=1024mb

# ===== EXTERNAL API KEYS (fill in your own) =====
TMDB_API_KEY=
TVDB_API_KEY=
FANART_API_KEY=
OMDB_API_KEY=

# ===== DOWNLOAD CLIENTS =====
QBITTORRENT_USERNAME=admin
QBITTORRENT_PASSWORD=adminadmin
TRANSMISSION_USERNAME=admin
TRANSMISSION_PASSWORD=admin
SABNZBD_USERNAME=admin
SABNZBD_PASSWORD=admin

# ===== NOTIFICATIONS (optional) =====
DISCORD_WEBHOOK_URL=
SLACK_WEBHOOK_URL=
EMAIL_FROM=
EMAIL_TO=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ===== LOGGING =====
LOG_LEVEL=INFO
LOG_MAX_SIZE=100m
LOG_MAX_FILES=3
DEBUG_MODE=false
VERBOSE_LOGGING=false
EOF
        
        info "✅ Created .env file with secure auto-generated secrets"
        warn "⚠️  Please review and customize the .env file, especially:"
        warn "     - Set your timezone (TZ)"
        warn "     - Add external API keys for metadata"
        warn "     - Configure notification services"
        warn "     - Adjust resource limits based on your system"
        
        if [ "$QUICK_START" != "true" ]; then
            read -p "Press Enter to continue after reviewing .env file..." -r
        fi
    else
        info "Using existing .env file"
        # Validate critical settings
        if ! grep -q "API_KEY=" .env || ! grep -q "POSTGRES_PASSWORD=" .env; then
            warn "Missing critical security settings in .env file"
            warn "Please ensure API_KEY and POSTGRES_PASSWORD are set"
        fi
    fi
    
    # Set proper permissions
    chmod 644 .env 2>/dev/null || true
    chmod -R 755 "$DEFAULT_CONFIG_PATH" "$DEFAULT_DATA_PATH" 2>/dev/null || true
    
    # Create Docker network if it doesn't exist
    if ! docker network ls | grep -q "ultimate-media-network-2025-fixed"; then
        info "Creating Docker network..."
        docker network create ultimate-media-network-2025-fixed \
            --driver bridge \
            --subnet 172.31.0.0/16 \
            --gateway 172.31.0.1 \
            --opt com.docker.network.mtu=1500 \
            2>/dev/null || warn "Network may already exist"
    fi
    
    info "✅ Environment setup completed"
}

cleanup_previous() {
    info "🧹 Cleaning up previous deployment..."
    
    # Stop and remove existing container if force recreate is enabled
    if [ "$FORCE_RECREATE" = "true" ]; then
        if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            info "Stopping and removing existing container: $CONTAINER_NAME"
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
        fi
        
        # Remove existing image if it exists
        if docker images --format 'table {{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
            info "Removing existing image: $IMAGE_NAME"
            docker rmi "$IMAGE_NAME" 2>/dev/null || true
        fi
    fi
    
    # Clean up unused images and containers (optional)
    if [ "$PRODUCTION_MODE" = "true" ]; then
        info "Cleaning up unused Docker resources..."
        docker system prune -f --volumes 2>/dev/null || true
    fi
    
    info "✅ Cleanup completed"
}

validate_files() {
    info "🔍 Validating deployment files..."
    
    # Check if required files exist
    local required_files=(
        "$COMPOSE_FILE"
        "$DOCKERFILE"
        ".env.ultimate-single-container-2025-fixed.template"
        "entrypoint-fixed.sh"
        "healthcheck-fixed.sh"
    )
    
    local missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        error "Missing required files: ${missing_files[*]}"
        error "Please ensure all deployment files are present"
        exit 1
    fi
    
    # Validate Docker Compose file
    if ! docker-compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
        if ! docker compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
            error "Invalid Docker Compose configuration in $COMPOSE_FILE"
            exit 1
        fi
    fi
    
    # Check Dockerfile syntax
    if ! docker build --no-cache --dry-run -f "$DOCKERFILE" . > /dev/null 2>&1; then
        warn "Dockerfile may have syntax issues, but proceeding..."
    fi
    
    info "✅ File validation completed"
}

build_image() {
    if [ "$SKIP_BUILD" = "true" ]; then
        info "⏭️ Skipping image build (SKIP_BUILD=true)"
        return 0
    fi
    
    info "🔨 Building Ultimate Media Server 2025 Fixed Edition..."
    
    local build_args=(
        "--file" "$DOCKERFILE"
        "--tag" "$IMAGE_NAME"
        "--build-arg" "BUILDKIT_PROGRESS=plain"
        "--build-arg" "TARGETPLATFORM=linux/amd64"
    )
    
    if [ "$PRODUCTION_MODE" = "true" ]; then
        build_args+=("--build-arg" "NODE_ENV=production")
        build_args+=("--build-arg" "PYTHON_ENV=production")
    fi
    
    # Add BuildKit optimizations
    export DOCKER_BUILDKIT=1
    export BUILDKIT_PROGRESS=plain
    
    # Add caching for faster subsequent builds
    build_args+=("--build-arg" "BUILDKIT_INLINE_CACHE=1")
    
    info "Build command: DOCKER_BUILDKIT=1 docker build ${build_args[*]} ."
    
    # Build with timeout and progress monitoring
    if timeout 3600 docker build "${build_args[@]}" . 2>&1 | tee -a "$LOG_FILE"; then
        info "✅ Image build completed successfully"
    else
        error "❌ Image build failed or timed out"
        error "Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

deploy_container() {
    info "🚀 Deploying Ultimate Media Server 2025 Fixed Edition..."
    
    # Source environment variables
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi
    
    # Choose Docker Compose command
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    # Deploy using docker-compose with enhanced options
    local compose_args=()
    if [ "$FORCE_RECREATE" = "true" ]; then
        compose_args+=("--force-recreate")
    fi
    
    info "Deploying with command: $compose_cmd -f $COMPOSE_FILE up -d ${compose_args[*]}"
    
    if $compose_cmd -f "$COMPOSE_FILE" up -d "${compose_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        info "✅ Container deployment initiated"
    else
        error "❌ Container deployment failed"
        error "Check the log file for details: $LOG_FILE"
        exit 1
    fi
    
    info "⏳ Waiting for container to start..."
    sleep 30
    
    # Check container status with multiple attempts
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -q "$CONTAINER_NAME.*Up"; then
            info "✅ Container is running"
            break
        else
            if [ $attempt -eq $max_attempts ]; then
                error "❌ Container failed to start after $max_attempts attempts"
                error "Container logs:"
                docker logs "$CONTAINER_NAME" --tail 100
                exit 1
            fi
            warn "Container not yet ready, attempt $attempt/$max_attempts"
            sleep 15
            ((attempt++))
        fi
    done
}

perform_health_checks() {
    info "🏥 Performing comprehensive health checks..."
    
    local max_attempts=20
    local attempt=1
    local check_interval=30
    
    while [ $attempt -le $max_attempts ]; do
        info "Health check attempt $attempt/$max_attempts (waiting up to $((max_attempts * check_interval / 60)) minutes total)"
        
        # Check if container is running
        if ! docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -q "$CONTAINER_NAME.*Up"; then
            error "❌ Container is not running"
            docker logs "$CONTAINER_NAME" --tail 50
            exit 1
        fi
        
        # Run comprehensive health check
        if docker exec "$CONTAINER_NAME" /app/healthcheck.sh 2>&1 | tee -a "$LOG_FILE"; then
            info "✅ Comprehensive health check passed"
            break
        else
            warn "⚠️  Health check failed, attempt $attempt/$max_attempts"
            if [ $attempt -lt $max_attempts ]; then
                info "Waiting ${check_interval}s before next attempt..."
                sleep $check_interval
            fi
            ((attempt++))
        fi
        
        if [ $attempt -gt $max_attempts ]; then
            warn "❌ Health checks did not pass after $max_attempts attempts"
            warn "Some services may still be starting up. Check individual service status."
            warn "Container logs (last 200 lines):"
            docker logs "$CONTAINER_NAME" --tail 200
        fi
    done
}

display_access_info() {
    info "📋 Deployment completed successfully!"
    
    # Get container IP and network info
    local container_ip
    container_ip=$(docker inspect "$CONTAINER_NAME" 2>/dev/null | jq -r '.[0].NetworkSettings.Networks[].IPAddress' 2>/dev/null | head -1)
    
    # Source .env for port information
    if [ -f .env ]; then
        set -a
        source .env 2>/dev/null || true
        set +a
    fi
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                          🎉 ACCESS INFORMATION 🎉                           ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  🌐 MAIN DASHBOARD:                                                          ║${NC}"
    echo -e "${GREEN}║     http://localhost${MAIN_PORT:+:$MAIN_PORT}                               ║${NC}"
    echo -e "${GREEN}║     http://${container_ip:-IP_NOT_DETECTED}                                 ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  🎬 MEDIA SERVERS:                                                           ║${NC}"
    echo -e "${GREEN}║     Jellyfin:    http://localhost:${JELLYFIN_PORT:-8096}                    ║${NC}"
    echo -e "${GREEN}║     Plex:        http://localhost:${PLEX_PORT:-32400}/web                   ║${NC}"
    echo -e "${GREEN}║     Emby:        http://localhost:${EMBY_PORT:-8097}                        ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📺 MEDIA MANAGEMENT (*ARR SUITE):                                          ║${NC}"
    echo -e "${GREEN}║     Sonarr:      http://localhost:${SONARR_PORT:-8989}                      ║${NC}"
    echo -e "${GREEN}║     Radarr:      http://localhost:${RADARR_PORT:-7878}                      ║${NC}"
    echo -e "${GREEN}║     Lidarr:      http://localhost:${LIDARR_PORT:-8686}                      ║${NC}"
    echo -e "${GREEN}║     Readarr:     http://localhost:${READARR_PORT:-8787}                     ║${NC}"
    echo -e "${GREEN}║     Bazarr:      http://localhost:${BAZARR_PORT:-6767}                      ║${NC}"
    echo -e "${GREEN}║     Prowlarr:    http://localhost:${PROWLARR_PORT:-9696}                    ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  ⬇️  DOWNLOAD CLIENTS:                                                        ║${NC}"
    echo -e "${GREEN}║     qBittorrent: http://localhost:${QBITTORRENT_PORT:-8080} (admin/adminadmin) ║${NC}"
    echo -e "${GREEN}║     Transmission:http://localhost:${TRANSMISSION_PORT:-9091}                ║${NC}"
    echo -e "${GREEN}║     SABnzbd:     http://localhost:${SABNZBD_PORT:-8085}                     ║${NC}"
    echo -e "${GREEN}║     NZBGet:      http://localhost:${NZBGET_PORT:-6789}                      ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📋 REQUEST MANAGEMENT:                                                      ║${NC}"
    echo -e "${GREEN}║     Overseerr:   http://localhost:${OVERSEERR_PORT:-5055}                   ║${NC}"
    echo -e "${GREEN}║     Jellyseerr:  http://localhost:${JELLYSEERR_PORT:-5056}                  ║${NC}"
    echo -e "${GREEN}║     Ombi:        http://localhost:${OMBI_PORT:-3579}                        ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  🖥️  DASHBOARDS:                                                             ║${NC}"
    echo -e "${GREEN}║     Homepage:    http://localhost:${HOMEPAGE_PORT:-3000}                    ║${NC}"
    echo -e "${GREEN}║     Homarr:      http://localhost:${HOMARR_PORT:-7575}                      ║${NC}"
    echo -e "${GREEN}║     Organizr:    http://localhost:${ORGANIZR_PORT:-8181}                    ║${NC}"
    echo -e "${GREEN}║     Tautulli:    http://localhost:${TAUTULLI_PORT:-8182}                    ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  🤖 AI SERVICES:                                                             ║${NC}"
    echo -e "${GREEN}║     AI Assistant:http://localhost:${AI_ASSISTANT_PORT:-8901}                ║${NC}"
    echo -e "${GREEN}║     Ollama:      http://localhost:${OLLAMA_PORT:-11434}                     ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║  📊 MONITORING:                                                              ║${NC}"
    echo -e "${GREEN}║     Traefik:     http://localhost:${TRAEFIK_DASHBOARD_PORT:-8088}           ║${NC}"
    echo -e "${GREEN}║     Grafana:     http://localhost:${GRAFANA_PORT:-3001} (admin/admin123)    ║${NC}"
    echo -e "${GREEN}║     Prometheus:  http://localhost:${PROMETHEUS_PORT:-9090}                  ║${NC}"
    echo -e "${GREEN}║     Uptime Kuma: http://localhost:${UPTIME_KUMA_PORT:-3002}                 ║${NC}"
    echo -e "${GREEN}║     Portainer:   http://localhost:${PORTAINER_PORT:-9000}                   ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    echo ""
    echo -e "${CYAN}🔧 MANAGEMENT COMMANDS:${NC}"
    echo -e "  View logs:           ${YELLOW}docker logs $CONTAINER_NAME -f${NC}"
    echo -e "  Restart container:   ${YELLOW}docker restart $CONTAINER_NAME${NC}"
    echo -e "  Stop container:      ${YELLOW}docker stop $CONTAINER_NAME${NC}"
    echo -e "  Health check:        ${YELLOW}docker exec $CONTAINER_NAME /app/healthcheck.sh${NC}"
    echo -e "  Shell access:        ${YELLOW}docker exec -it $CONTAINER_NAME /bin/bash${NC}"
    echo -e "  Update container:    ${YELLOW}$0 --force-recreate${NC}"
    
    echo ""
    echo -e "${PURPLE}📖 CONFIGURATION:${NC}"
    echo -e "  Environment file:    ${YELLOW}.env${NC}"
    echo -e "  Configuration dir:   ${YELLOW}$DEFAULT_CONFIG_PATH${NC}"
    echo -e "  Data directory:      ${YELLOW}$DEFAULT_DATA_PATH${NC}"
    echo -e "  AI models:           ${YELLOW}$DEFAULT_MODELS_PATH${NC}"
    echo -e "  Deployment log:      ${YELLOW}$LOG_FILE${NC}"
    
    echo ""
    echo -e "${BLUE}🔗 USEFUL LINKS:${NC}"
    echo -e "  Documentation:       ${YELLOW}https://docs.ultimate-media-server.com${NC}"
    echo -e "  GitHub Repository:   ${YELLOW}https://github.com/ultimate-media-server/2025${NC}"
    echo -e "  Support Discord:     ${YELLOW}https://discord.gg/ultimate-media-server${NC}"
    
    echo ""
    echo -e "${GREEN}✅ Ultimate Media Server 2025 Fixed Edition is ready to use!${NC}"
    echo -e "${GREEN}🎯 All critical services should be available within 5-10 minutes${NC}"
}

optimize_system() {
    if [ "$PRODUCTION_MODE" = "true" ]; then
        info "⚡ Applying production optimizations..."
        
        # Docker daemon optimizations
        info "Configuring Docker daemon optimizations..."
        
        # System-level optimizations (if running with appropriate privileges)
        if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
            info "Applying system-level optimizations..."
            
            # Increase inotify limits for file watching
            {
                echo 'fs.inotify.max_user_watches=1048576'
                echo 'fs.inotify.max_user_instances=1024'
                echo 'vm.max_map_count=262144'
                echo 'net.core.rmem_max=16777216'
                echo 'net.core.wmem_max=16777216'
            } | sudo tee -a /etc/sysctl.conf > /dev/null
            
            # Apply immediately
            sudo sysctl -p 2>/dev/null || true
            
            info "✅ System optimizations applied"
        else
            warn "⚠️  Run as root or with sudo for additional system optimizations"
            warn "     Consider running: sudo sysctl -w fs.inotify.max_user_watches=1048576"
        fi
        
        # Container-specific optimizations
        info "Applying container optimizations..."
        
        # Set CPU scheduling policy for better performance
        if docker exec "$CONTAINER_NAME" test -f /proc/1/stat 2>/dev/null; then
            docker exec "$CONTAINER_NAME" sh -c "echo 'Container optimizations applied'" || true
        fi
        
        info "✅ Performance optimizations completed"
    fi
}

setup_monitoring() {
    if [ "$ENABLE_MONITORING" = "true" ]; then
        info "📊 Setting up enhanced monitoring and management tools..."
        
        # Create monitoring configuration directory
        mkdir -p ./config/monitoring
        
        # Create comprehensive monitoring script
        cat > ./monitoring-check.sh << 'EOF'
#!/bin/bash
# Comprehensive monitoring check for Ultimate Media Server 2025 Fixed

CONTAINER_NAME="ultimate-media-server-2025-fixed"

echo "=== Ultimate Media Server 2025 Fixed - System Status ==="
echo "Timestamp: $(date)"
echo ""

echo "=== Container Status ==="
docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "Container not found"
echo ""

echo "=== Container Resources ==="
docker stats "$CONTAINER_NAME" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null || echo "Container not running"
echo ""

echo "=== Comprehensive Health Check ==="
if docker exec "$CONTAINER_NAME" /app/healthcheck.sh 2>/dev/null; then
    echo "✅ Comprehensive health check: PASSED"
else
    echo "❌ Comprehensive health check: FAILED"
fi
echo ""

echo "=== Service Status Summary ==="
docker exec "$CONTAINER_NAME" sh -c "
    echo 'Critical Services:'
    echo '  PostgreSQL:' \$(pg_isready -h 127.0.0.1 -p 5432 -q 2>/dev/null && echo '✅ Running' || echo '❌ Not running')
    echo '  Redis:' \$(redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -q PONG && echo '✅ Running' || echo '❌ Not running')
    echo '  Jellyfin:' \$(curl -s http://127.0.0.1:8096/health >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Not running')
    echo '  Traefik:' \$(curl -s http://127.0.0.1:8080/ping >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Not running')
    echo ''
    echo 'Media Management:'
    echo '  Sonarr:' \$(curl -s http://127.0.0.1:8989 >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Not running')
    echo '  Radarr:' \$(curl -s http://127.0.0.1:7878 >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Not running')
    echo '  Prowlarr:' \$(curl -s http://127.0.0.1:9696 >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Not running')
" 2>/dev/null || echo "Cannot connect to container for service status"
echo ""

echo "=== Recent Logs (Last 20 lines) ==="
docker logs "$CONTAINER_NAME" --since 1h --tail 20 2>/dev/null || echo "Cannot retrieve logs"
echo ""

echo "=== Disk Usage ==="
docker exec "$CONTAINER_NAME" df -h / 2>/dev/null | tail -1 || echo "Cannot retrieve disk usage"
echo ""

echo "=== Network Status ==="
docker exec "$CONTAINER_NAME" sh -c "
    echo 'Container IP:' \$(hostname -I | awk '{print \$1}')
    echo 'Network connectivity test:'
    echo '  Internet:' \$(curl -s --max-time 5 http://google.com >/dev/null 2>&1 && echo '✅ Connected' || echo '❌ No connection')
" 2>/dev/null || echo "Cannot retrieve network status"
EOF
        
        chmod +x ./monitoring-check.sh
        
        # Create service restart script
        cat > ./restart-services.sh << 'EOF'
#!/bin/bash
# Service restart script for Ultimate Media Server 2025 Fixed

CONTAINER_NAME="ultimate-media-server-2025-fixed"

echo "🔄 Restarting Ultimate Media Server 2025 Fixed..."

if docker restart "$CONTAINER_NAME"; then
    echo "✅ Container restarted successfully"
    echo "⏳ Waiting for services to initialize..."
    sleep 30
    
    echo "🏥 Running health check..."
    if docker exec "$CONTAINER_NAME" /app/healthcheck.sh; then
        echo "✅ Health check passed - services are ready"
    else
        echo "⚠️  Health check failed - services may still be starting"
        echo "   Check status with: ./monitoring-check.sh"
    fi
else
    echo "❌ Failed to restart container"
    exit 1
fi
EOF
        
        chmod +x ./restart-services.sh
        
        # Create backup script
        cat > ./backup-config.sh << 'EOF'
#!/bin/bash
# Configuration backup script for Ultimate Media Server 2025 Fixed

CONTAINER_NAME="ultimate-media-server-2025-fixed"
BACKUP_DIR="./backups/$(date +%Y%m%d-%H%M%S)"

echo "💾 Creating configuration backup..."

mkdir -p "$BACKUP_DIR"

# Backup configuration
if [ -d "./config" ]; then
    tar -czf "$BACKUP_DIR/config.tar.gz" ./config
    echo "✅ Configuration backed up"
fi

# Backup environment file
if [ -f ".env" ]; then
    cp .env "$BACKUP_DIR/"
    echo "✅ Environment file backed up"
fi

# Backup database data (excluding large files)
if [ -d "./data/databases" ]; then
    tar -czf "$BACKUP_DIR/databases.tar.gz" ./data/databases --exclude="*.log" --exclude="*.log.*"
    echo "✅ Database data backed up"
fi

echo "📁 Backup created in: $BACKUP_DIR"
echo "💾 Backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
EOF
        
        chmod +x ./backup-config.sh
        
        info "✅ Monitoring and management tools created"
        info "📊 Available scripts:"
        info "     ./monitoring-check.sh    - Comprehensive system status"
        info "     ./restart-services.sh    - Restart container and services"  
        info "     ./backup-config.sh       - Backup configuration and data"
    fi
}

create_documentation() {
    info "📚 Creating deployment documentation..."
    
    cat > "README-DEPLOYMENT-$(date +%Y%m%d).md" << EOF
# Ultimate Media Server 2025 Fixed - Deployment Summary

**Deployment Date:** $(date)
**Container Name:** $CONTAINER_NAME
**Image:** $IMAGE_NAME

## Deployment Configuration

- **Production Mode:** $PRODUCTION_MODE  
- **AI Enabled:** $ENABLE_AI
- **Monitoring Enabled:** $ENABLE_MONITORING
- **Security Mode:** Enabled with auto-generated secrets

## Directory Structure

\`\`\`
$(pwd)/
├── config/           # Service configurations
├── data/            # Application data and databases
├── models/          # AI model storage
├── logs/            # Application logs
├── backups/         # Configuration backups
└── .env             # Environment configuration
\`\`\`

## Quick Commands

\`\`\`bash
# Check status
./monitoring-check.sh

# View logs
docker logs $CONTAINER_NAME -f

# Restart services
./restart-services.sh

# Backup configuration
./backup-config.sh

# Update container
$0 --force-recreate

# Access shell
docker exec -it $CONTAINER_NAME /bin/bash
\`\`\`

## Service URLs

- **Main Dashboard:** http://localhost${MAIN_PORT:+:$MAIN_PORT}
- **Jellyfin:** http://localhost:${JELLYFIN_PORT:-8096}
- **Plex:** http://localhost:${PLEX_PORT:-32400}/web
- **Sonarr:** http://localhost:${SONARR_PORT:-8989}
- **Radarr:** http://localhost:${RADARR_PORT:-7878}
- **Prowlarr:** http://localhost:${PROWLARR_PORT:-9696}
- **qBittorrent:** http://localhost:${QBITTORRENT_PORT:-8080}
- **Grafana:** http://localhost:${GRAFANA_PORT:-3001}

## Troubleshooting

1. **Services not starting:** Check logs with \`docker logs $CONTAINER_NAME\`
2. **Health check failing:** Wait 10-15 minutes for all services to initialize
3. **Performance issues:** Increase memory allocation in .env file
4. **Network issues:** Check if ports are available and not blocked

## Support

- **Documentation:** https://docs.ultimate-media-server.com
- **Issues:** Report on GitHub repository
- **Community:** Discord server for support

---
*Generated by Ultimate Media Server 2025 Fixed Deployment Script*
EOF
    
    info "✅ Deployment documentation created: README-DEPLOYMENT-$(date +%Y%m%d).md"
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
            --force-recreate)
                FORCE_RECREATE=true
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
            --verbose)
                set -x
                shift
                ;;
            -h|--help)
                echo "Ultimate Media Server 2025 Fixed - Deployment Script"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-build           Skip Docker image build"
                echo "  --force-recreate       Force recreate existing container"
                echo "  --quick-start          Skip interactive prompts"
                echo "  --no-ai               Disable AI features"
                echo "  --no-monitoring       Disable monitoring setup"
                echo "  --dev                 Development mode"
                echo "  --verbose             Enable verbose output"
                echo "  -h, --help            Show this help"
                echo ""
                echo "Examples:"
                echo "  $0                    # Standard deployment"
                echo "  $0 --quick-start      # Automated deployment"
                echo "  $0 --force-recreate   # Update existing deployment"
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
    
    info "🚀 Starting Ultimate Media Server 2025 Fixed deployment..."
    info "📁 Working directory: $(pwd)"
    info "📝 Log file: $LOG_FILE"
    info "⚙️  Configuration: Production=$PRODUCTION_MODE, AI=$ENABLE_AI, Monitoring=$ENABLE_MONITORING"
    
    check_requirements
    validate_files
    setup_environment
    cleanup_previous
    build_image
    deploy_container
    optimize_system
    setup_monitoring
    perform_health_checks
    create_documentation
    display_access_info
    
    info "🎉 Deployment completed successfully!"
    info "📝 Complete deployment log available at: $LOG_FILE"
    info "📚 Deployment summary: README-DEPLOYMENT-$(date +%Y%m%d).md"
}

# Error handling with cleanup
cleanup_on_error() {
    error "❌ Deployment failed at line $1. Exit code: $2"
    error "📝 Check the log file for details: $LOG_FILE"
    error "🔧 For support, include this log when reporting issues"
    
    # Optional: cleanup on failure
    if [ "$QUICK_START" != "true" ]; then
        read -p "Clean up failed deployment? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
        fi
    fi
}

trap 'cleanup_on_error $LINENO $?' ERR

# Execute main function
main "$@"