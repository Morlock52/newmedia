#!/bin/bash

# ==================================================================
# ULTIMATE AUTOMATED DEPLOYMENT SCRIPT
# Single-command deployment for complete media server stack
# ==================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="newmedia"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/deployment-$(date +%Y%m%d_%H%M%S).log"
COMPOSE_FILE="docker-compose.yml"
PRODUCTION_FILE="docker-compose.production.yml"
HEALTH_CHECK_TIMEOUT=300
RETRY_COUNT=3

# Ensure directories exist
mkdir -p logs backups scripts

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Header
echo -e "${PURPLE}"
cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║                  ULTIMATE MEDIA SERVER DEPLOYMENT                ║
║                     Automated Installation                       ║
╚══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Pre-deployment checks
pre_deployment_checks() {
    log "Starting pre-deployment checks..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        warn "Low disk space detected. Recommended minimum: 10GB"
    fi
    
    # Check available memory (minimum 4GB recommended)
    available_memory=$(free -m | awk 'NR==2{print $7}')
    if [ "$available_memory" -lt 4096 ]; then
        warn "Low memory detected. Recommended minimum: 4GB"
    fi
    
    success "Pre-deployment checks completed"
}

# Create necessary directories and files
setup_environment() {
    log "Setting up environment..."
    
    # Create directory structure
    mkdir -p {data,config,logs,backups,media,downloads}
    mkdir -p {media/{movies,tv,music,audiobooks,books,photos,comics},downloads/{complete,incomplete,torrents,usenet}}
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log "Creating default .env file..."
        cp .env.template .env 2>/dev/null || cat > .env << 'EOF'
# Media Server Configuration
TZ=America/New_York
PUID=1000
PGID=1000

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_postgres_password
MYSQL_ROOT_PASSWORD=secure_mysql_password

# Service Passwords
GRAFANA_PASSWORD=admin123
PLEX_CLAIM=
VPN_PROVIDER=nordvpn
VPN_COUNTRY=Switzerland

# Notification Settings
EMAIL_FROM=
EMAIL_TO=
SMTP_SERVER=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=

# Domain Settings
DOMAIN=localhost
NEXTCLOUD_DOMAIN=localhost
VAULTWARDEN_DOMAIN=http://localhost

# API Keys (will be auto-generated)
SONARR_API_KEY=
RADARR_API_KEY=
PROWLARR_API_KEY=
LIDARR_API_KEY=
EOF
        warn "Please review and update the .env file with your settings"
    fi
    
    success "Environment setup completed"
}

# Backup existing configuration
backup_existing() {
    log "Creating backup of existing configuration..."
    
    if [ -d "config" ] || [ -d "data" ] || docker ps -q &> /dev/null; then
        mkdir -p "$BACKUP_DIR"
        
        # Backup configuration files
        [ -d "config" ] && cp -r config "$BACKUP_DIR/" 2>/dev/null || true
        [ -f ".env" ] && cp .env "$BACKUP_DIR/" 2>/dev/null || true
        [ -f "docker-compose.yml" ] && cp docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || true
        
        # Export running containers
        if docker ps -q &> /dev/null; then
            docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$BACKUP_DIR/running_containers.txt"
        fi
        
        success "Backup created at $BACKUP_DIR"
    else
        log "No existing configuration to backup"
    fi
}

# Deploy the stack
deploy_stack() {
    log "Deploying media server stack..."
    
    # Choose compose file
    COMPOSE_CMD="docker-compose"
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    fi
    
    # Use production compose if available
    if [ -f "$PRODUCTION_FILE" ] && [ "$1" = "production" ]; then
        COMPOSE_FILE="$PRODUCTION_FILE"
        log "Using production configuration"
    fi
    
    # Pull latest images
    log "Pulling latest images..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" pull
    
    # Start core services first
    log "Starting core services (databases, networking)..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d postgres mariadb redis
    
    # Wait for databases
    sleep 10
    
    # Start infrastructure services
    log "Starting infrastructure services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d nginx-proxy-manager traefik portainer
    
    # Wait for infrastructure
    sleep 15
    
    # Start media management services
    log "Starting media management services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d prowlarr sonarr radarr lidarr bazarr
    
    # Wait for arr services
    sleep 20
    
    # Start download clients
    log "Starting download clients..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d gluetun qbittorrent transmission sabnzbd
    
    # Wait for download clients
    sleep 15
    
    # Start media servers
    log "Starting media servers..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d jellyfin plex emby
    
    # Start remaining services
    log "Starting remaining services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d
    
    success "Stack deployment completed"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local services=(
        "jellyfin:8096:/health"
        "sonarr:8989:/ping"
        "radarr:7878:/ping"
        "prowlarr:9696:/ping"
        "qbittorrent:8080/api/v2/app/version"
        "grafana:3000/api/health"
        "uptime-kuma:3001/"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port path <<< "$service"
        
        log "Checking $name service..."
        
        local count=0
        while [ $count -lt $HEALTH_CHECK_TIMEOUT ]; do
            if curl -sf "http://localhost:$port$path" > /dev/null 2>&1; then
                success "$name is ready"
                break
            fi
            
            if [ $count -eq $((HEALTH_CHECK_TIMEOUT - 1)) ]; then
                warn "$name failed to start within timeout"
                break
            fi
            
            sleep 2
            count=$((count + 1))
        done
    done
}

# Configure services
configure_services() {
    log "Configuring services..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    # Run service configuration scripts
    if [ -f "scripts/configure-arr-services.sh" ]; then
        log "Running ARR services configuration..."
        bash scripts/configure-arr-services.sh || warn "ARR configuration failed"
    fi
    
    if [ -f "scripts/configure-download-clients.sh" ]; then
        log "Running download client configuration..."
        bash scripts/configure-download-clients.sh || warn "Download client configuration failed"
    fi
    
    success "Service configuration completed"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    local failed_services=()
    local running_containers
    running_containers=$(docker ps --format "{{.Names}}" | wc -l)
    
    # Check if core services are running
    local core_services=("postgres" "redis" "jellyfin" "sonarr" "radarr" "prowlarr" "qbittorrent")
    
    for service in "${core_services[@]}"; do
        if ! docker ps --format "{{.Names}}" | grep -q "$service"; then
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        success "All core services are running ($running_containers containers active)"
    else
        error "Failed services: ${failed_services[*]}"
        return 1
    fi
    
    # Test web interfaces
    log "Testing web interfaces..."
    local web_tests=(
        "http://localhost:8096:Jellyfin"
        "http://localhost:8989:Sonarr"
        "http://localhost:7878:Radarr"
        "http://localhost:9696:Prowlarr"
        "http://localhost:8080:qBittorrent"
    )
    
    for test in "${web_tests[@]}"; do
        IFS=':' read -r url name <<< "$test"
        if curl -sf "$url" > /dev/null 2>&1; then
            success "$name web interface accessible"
        else
            warn "$name web interface not accessible at $url"
        fi
    done
    
    return 0
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local report_file="./logs/deployment-report-$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Deployment Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; background: #2d2d2d; border-radius: 8px; }
        .service { display: inline-block; margin: 10px; padding: 10px; background: #3d3d3d; border-radius: 5px; min-width: 200px; }
        .status-running { border-left: 4px solid #4CAF50; }
        .status-error { border-left: 4px solid #f44336; }
        .url { color: #64B5F6; text-decoration: none; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #555; }
        th { background: #4a4a4a; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Media Server Deployment Report</h1>
        <p>Deployment completed on $(date)</p>
    </div>
    
    <div class="section">
        <h2>📊 System Overview</h2>
        <p><strong>Total Containers:</strong> $(docker ps --format "{{.Names}}" | wc -l)</p>
        <p><strong>Docker Version:</strong> $(docker --version)</p>
        <p><strong>Compose Version:</strong> $(docker-compose --version 2>/dev/null || docker compose version)</p>
        <p><strong>System Load:</strong> $(uptime | awk -F'load average:' '{print $2}')</p>
        <p><strong>Available Memory:</strong> $(free -h | awk 'NR==2{printf "%.1f GB", $7/1024}')</p>
    </div>
    
    <div class="section">
        <h2>🎯 Quick Access</h2>
        <div class="service status-running">
            <h3>Jellyfin</h3>
            <a href="http://localhost:8096" class="url">http://localhost:8096</a>
            <p>Main media server interface</p>
        </div>
        <div class="service status-running">
            <h3>Sonarr</h3>
            <a href="http://localhost:8989" class="url">http://localhost:8989</a>
            <p>TV show management</p>
        </div>
        <div class="service status-running">
            <h3>Radarr</h3>
            <a href="http://localhost:7878" class="url">http://localhost:7878</a>
            <p>Movie management</p>
        </div>
        <div class="service status-running">
            <h3>Prowlarr</h3>
            <a href="http://localhost:9696" class="url">http://localhost:9696</a>
            <p>Indexer management</p>
        </div>
        <div class="service status-running">
            <h3>qBittorrent</h3>
            <a href="http://localhost:8080" class="url">http://localhost:8080</a>
            <p>Torrent client</p>
        </div>
        <div class="service status-running">
            <h3>Grafana</h3>
            <a href="http://localhost:3000" class="url">http://localhost:3000</a>
            <p>Monitoring dashboard</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Running Services</h2>
        <table>
            <tr><th>Container</th><th>Image</th><th>Status</th><th>Ports</th></tr>
EOF

    # Add running containers to report
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" | tail -n +2 | while read -r line; do
        echo "            <tr><td>$(echo "$line" | cut -f1)</td><td>$(echo "$line" | cut -f2)</td><td>$(echo "$line" | cut -f3)</td><td>$(echo "$line" | cut -f4)</td></tr>" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
        </table>
    </div>
    
    <div class="section">
        <h2>🔧 Next Steps</h2>
        <ul>
            <li>Configure Prowlarr indexers</li>
            <li>Add media libraries to Sonarr/Radarr</li>
            <li>Set up Jellyfin libraries</li>
            <li>Configure download client settings</li>
            <li>Set up monitoring alerts</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📝 Support</h2>
        <p>Logs available at: <code>./logs/</code></p>
        <p>Configuration backup: <code>$BACKUP_DIR</code></p>
        <p>Health checks: <code>./scripts/health-check.sh</code></p>
    </div>
</body>
</html>
EOF

    success "Deployment report generated: $report_file"
    
    # Open report in browser if possible
    if command -v xdg-open &> /dev/null; then
        xdg-open "$report_file"
    elif command -v open &> /dev/null; then
        open "$report_file"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add any cleanup tasks here
}

# Error handling
handle_error() {
    error "Deployment failed at step: $1"
    log "Rolling back changes..."
    
    # Stop any running containers
    docker-compose down 2>/dev/null || docker compose down 2>/dev/null || true
    
    # Restore backup if available
    if [ -d "$BACKUP_DIR" ]; then
        log "Restoring from backup..."
        [ -f "$BACKUP_DIR/.env" ] && cp "$BACKUP_DIR/.env" . 2>/dev/null || true
        [ -d "$BACKUP_DIR/config" ] && cp -r "$BACKUP_DIR/config" . 2>/dev/null || true
    fi
    
    error "Deployment failed. Check logs at: $LOG_FILE"
    exit 1
}

# Trap errors
trap 'handle_error "unknown"' ERR

# Main deployment function
main() {
    local environment=${1:-development}
    
    log "Starting automated deployment (environment: $environment)..."
    
    # Run deployment steps
    pre_deployment_checks || handle_error "pre_deployment_checks"
    setup_environment || handle_error "setup_environment"
    backup_existing || handle_error "backup_existing"
    deploy_stack "$environment" || handle_error "deploy_stack"
    wait_for_services || handle_error "wait_for_services"
    configure_services || handle_error "configure_services"
    
    # Validate deployment
    if validate_deployment; then
        generate_report
        success "🎉 Deployment completed successfully!"
        
        echo -e "${GREEN}"
        cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║                      DEPLOYMENT SUCCESSFUL!                      ║
║                                                                  ║
║  Your media server is now running and ready to use.             ║
║  Check the deployment report for service URLs and next steps.   ║
╚══════════════════════════════════════════════════════════════════╝
EOF
        echo -e "${NC}"
        
        log "Access your services:"
        log "  - Jellyfin: http://localhost:8096"
        log "  - Sonarr: http://localhost:8989"
        log "  - Radarr: http://localhost:7878"
        log "  - Prowlarr: http://localhost:9696"
        log "  - qBittorrent: http://localhost:8080"
        log "  - Grafana: http://localhost:3000"
        
    else
        handle_error "validation"
    fi
    
    cleanup
}

# Parse command line arguments
case "${1:-}" in
    "production")
        main production
        ;;
    "development"|"")
        main development
        ;;
    "--help"|"-h")
        echo "Usage: $0 [environment]"
        echo "Environments: development (default), production"
        echo ""
        echo "This script will:"
        echo "  1. Check system requirements"
        echo "  2. Set up the environment"
        echo "  3. Back up existing configuration"
        echo "  4. Deploy the complete media server stack"
        echo "  5. Configure services automatically"
        echo "  6. Validate the deployment"
        echo "  7. Generate a deployment report"
        exit 0
        ;;
    *)
        error "Unknown environment: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac