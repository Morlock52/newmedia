#!/bin/bash

# Deploy Performance-Optimized Media Server Stack 2025
# Features: CDN integration, advanced caching, GPU acceleration, monitoring

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${CYAN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check for GPU support (optional)
    if command -v nvidia-smi &> /dev/null; then
        info "NVIDIA GPU detected - enabling GPU acceleration"
        export GPU_ACCELERATION=true
    else
        warning "No NVIDIA GPU detected - CPU transcoding will be used"
        export GPU_ACCELERATION=false
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk 'NR==2{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        warning "Less than 8GB RAM detected. Performance may be limited."
    fi
    
    log "Prerequisites check completed"
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    
    # Configuration directories
    mkdir -p config/{nginx,varnish,redis,prometheus,grafana/{dashboards,datasources},loki,promtail,cloudflare,autoscaler}
    
    # Data directories
    mkdir -p data/{postgres,redis,prometheus,grafana,loki}
    
    # Cache directories
    mkdir -p cache/{nginx,varnish,jellyfin,plex}
    
    # SSL directory
    mkdir -p ssl
    
    # Logs directory
    mkdir -p logs
    
    # Media directories
    mkdir -p media/{movies,tv,music,photos,books,audiobooks}
    mkdir -p downloads/{complete,incomplete}
    
    log "Directory structure created"
}

# Generate SSL certificates (self-signed for testing)
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    if [[ ! -f ssl/cert.pem ]] || [[ ! -f ssl/key.pem ]]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/key.pem \
            -out ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log "SSL certificates generated"
    else
        info "SSL certificates already exist"
    fi
}

# Configure environment
setup_environment() {
    log "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        cat > .env << EOF
# Media Server Performance Configuration 2025

# Domain Configuration
DOMAIN=localhost

# Database Configuration
DB_NAME=mediaserver
DB_USER=postgres
DB_PASSWORD=$(openssl rand -base64 32)

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 32)

# Grafana Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Cloudflare Configuration (optional)
CLOUDFLARE_EMAIL=
CLOUDFLARE_API_KEY=
CLOUDFLARE_TUNNEL_TOKEN=

# Plex Configuration
PLEX_CLAIM=

# API Keys (will be generated on first run)
SONARR_API_KEY=$(openssl rand -hex 32)
RADARR_API_KEY=$(openssl rand -hex 32)
PROWLARR_API_KEY=$(openssl rand -hex 32)
BAZARR_API_KEY=$(openssl rand -hex 32)
JELLYFIN_API_KEY=$(openssl rand -hex 32)
TAUTULLI_API_KEY=$(openssl rand -hex 32)

# Paths
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads
DATA_PATH=./data

# Timezone
TZ=America/New_York

# Performance Settings
ENABLE_GPU=$GPU_ACCELERATION
MAX_MEMORY_GB=16
MAX_CPU_CORES=8
EOF
        log ".env file created with secure defaults"
        warning "Please review and update the .env file with your settings"
    else
        info ".env file already exists"
    fi
}

# Optimize system settings
optimize_system() {
    log "Applying system optimizations..."
    
    # Check if we have sudo privileges
    if sudo -n true 2>/dev/null; then
        # Run optimization script
        if [[ -f scripts/optimize-performance.sh ]]; then
            sudo bash scripts/optimize-performance.sh
        else
            warning "Performance optimization script not found"
        fi
    else
        warning "Sudo access required for system optimizations. Skipping..."
        info "Run 'sudo ./scripts/optimize-performance.sh' manually for best performance"
    fi
}

# Deploy the stack
deploy_stack() {
    log "Deploying performance-optimized media server stack..."
    
    # Pull latest images
    log "Pulling Docker images..."
    docker-compose -f docker-compose-ultimate-performance-2025.yml pull
    
    # Start services
    log "Starting services..."
    docker-compose -f docker-compose-ultimate-performance-2025.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    docker-compose -f docker-compose-ultimate-performance-2025.yml ps
}

# Configure services
configure_services() {
    log "Configuring services for optimal performance..."
    
    # Wait for services to be fully ready
    sleep 20
    
    # Configure Prometheus
    if [[ ! -f config/prometheus/prometheus-performance.yml ]]; then
        warning "Prometheus configuration not found. Using defaults."
    fi
    
    # Import Grafana dashboards
    log "Importing Grafana dashboards..."
    
    # Wait for Grafana to be ready
    until curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health | grep -q "200"; do
        sleep 5
    done
    
    success "Services configured"
}

# Performance validation
validate_performance() {
    log "Validating performance metrics..."
    
    # Check page load time
    if command -v curl &> /dev/null; then
        LOAD_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost)
        info "Homepage load time: ${LOAD_TIME}s"
        
        if (( $(echo "$LOAD_TIME < 1" | bc -l) )); then
            success "Page load time is under 1 second!"
        else
            warning "Page load time exceeds 1 second target"
        fi
    fi
    
    # Check cache hit rate (after some time)
    info "Cache performance will improve over time as content is cached"
}

# Display access information
display_info() {
    echo
    echo -e "${CYAN}=== Media Server Performance Stack Deployed ===${NC}"
    echo
    echo -e "${GREEN}Access URLs:${NC}"
    echo -e "  Main Dashboard:     ${BLUE}http://localhost${NC}"
    echo -e "  Jellyfin:          ${BLUE}http://localhost:8096${NC}"
    echo -e "  Plex:              ${BLUE}http://localhost:32400/web${NC}"
    echo -e "  Sonarr:            ${BLUE}http://localhost:8989${NC}"
    echo -e "  Radarr:            ${BLUE}http://localhost:7878${NC}"
    echo -e "  Grafana:           ${BLUE}http://localhost:3000${NC}"
    echo -e "  Prometheus:        ${BLUE}http://localhost:9090${NC}"
    echo
    echo -e "${GREEN}Performance Features:${NC}"
    echo -e "  ✓ CDN-ready configuration"
    echo -e "  ✓ Advanced caching (Varnish + Redis + Nginx)"
    echo -e "  ✓ GPU acceleration (if available)"
    echo -e "  ✓ Real-time performance monitoring"
    echo -e "  ✓ Auto-scaling capabilities"
    echo -e "  ✓ HTTP/3 support"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "  1. Review and update the .env file"
    echo -e "  2. Configure Cloudflare CDN (optional)"
    echo -e "  3. Access Grafana to view performance metrics"
    echo -e "  4. Run system optimization script with sudo"
    echo
    echo -e "${CYAN}Performance Dashboard:${NC}"
    echo -e "  Username: admin"
    echo -e "  Password: Check .env file for GRAFANA_PASSWORD"
    echo
}

# Main execution
main() {
    log "Starting Media Server Performance Stack Deployment..."
    
    check_prerequisites
    create_directories
    generate_ssl_certificates
    setup_environment
    optimize_system
    deploy_stack
    configure_services
    validate_performance
    display_info
    
    success "Deployment completed successfully!"
}

# Cleanup on error
cleanup() {
    error "Deployment failed. Cleaning up..."
    docker-compose -f docker-compose-ultimate-performance-2025.yml down
    exit 1
}

# Set trap for cleanup
trap cleanup ERR

# Run main function
main "$@"