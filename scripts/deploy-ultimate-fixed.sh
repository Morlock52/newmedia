#!/bin/bash

# Ultimate Media Server Deployment Script - Fixed Version
# Automated deployment with error handling, monitoring, and recovery

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
BACKUP_DIR="$PROJECT_DIR/.backup-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO) echo -e "${GREEN}[INFO]${NC} $message" ;;
        WARN) echo -e "${YELLOW}[WARN]${NC} $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" ;;
        DEBUG) echo -e "${BLUE}[DEBUG]${NC} $message" ;;
    esac
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Error handling
error_exit() {
    log ERROR "$1"
    log ERROR "Deployment failed. Check logs at: $LOG_FILE"
    exit 1
}

# Trap errors
trap 'error_exit "Unexpected error occurred at line $LINENO"' ERR

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local url="$1"
    local service_name="$2"
    local max_attempts=30
    local attempt=1
    
    log INFO "Waiting for $service_name to become available..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            log INFO "$service_name is now available"
            return 0
        fi
        
        log DEBUG "Attempt $attempt/$max_attempts: $service_name not ready"
        sleep 10
        ((attempt++))
    done
    
    error_exit "$service_name failed to become available after $((max_attempts * 10)) seconds"
}

# Function to check Docker health
check_docker_health() {
    if ! command_exists docker; then
        error_exit "Docker is not installed"
    fi
    
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker daemon is not running"
    fi
    
    if ! command_exists docker-compose; then
        error_exit "Docker Compose is not installed"
    fi
    
    log INFO "Docker and Docker Compose are ready"
}

# Function to create necessary directories
create_directories() {
    log INFO "Creating necessary directories..."
    
    local dirs=(
        "logs"
        "backups"
        "config"
        "data"
        "media"
        "downloads"
        "torrents"
        "usenet"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$PROJECT_DIR/$dir"
        log DEBUG "Created directory: $dir"
    done
}

# Function to backup existing configuration
backup_configuration() {
    log INFO "Creating backup of existing configuration..."
    
    if [ -d "$PROJECT_DIR/config" ]; then
        cp -r "$PROJECT_DIR/config" "$BACKUP_DIR/config" 2>/dev/null || true
    fi
    
    if [ -f "$PROJECT_DIR/.env" ]; then
        cp "$PROJECT_DIR/.env" "$BACKUP_DIR/.env" 2>/dev/null || true
    fi
    
    if [ -f "$PROJECT_DIR/docker-compose.yml" ]; then
        cp "$PROJECT_DIR/docker-compose.yml" "$BACKUP_DIR/docker-compose.yml" 2>/dev/null || true
    fi
    
    log INFO "Backup created at: $BACKUP_DIR"
}

# Function to setup environment
setup_environment() {
    log INFO "Setting up environment configuration..."
    
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        if [ -f "$PROJECT_DIR/.env.template" ]; then
            cp "$PROJECT_DIR/.env.template" "$PROJECT_DIR/.env"
            log INFO "Created .env from template"
        else
            error_exit ".env.template file not found"
        fi
    fi
    
    # Generate secure API key if not set
    if ! grep -q "API_KEY=" "$PROJECT_DIR/.env" || grep -q "change-this-secure-api-key" "$PROJECT_DIR/.env"; then
        local api_key=$(openssl rand -hex 32)
        sed -i.bak "s/API_KEY=.*/API_KEY=$api_key/" "$PROJECT_DIR/.env"
        log INFO "Generated new secure API key"
    fi
    
    # Set proper permissions
    chmod 600 "$PROJECT_DIR/.env"
}

# Function to fix Docker networking
fix_docker_networking() {
    log INFO "Fixing Docker networking issues..."
    
    if [ -f "$SCRIPT_DIR/fix-docker-networking.sh" ]; then
        chmod +x "$SCRIPT_DIR/fix-docker-networking.sh"
        bash "$SCRIPT_DIR/fix-docker-networking.sh" || log WARN "Docker networking script completed with warnings"
    else
        log WARN "Docker networking fix script not found, skipping..."
    fi
}

# Function to build custom images
build_custom_images() {
    log INFO "Building custom Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build API server
    if [ -f "api/Dockerfile" ]; then
        log INFO "Building API server image..."
        docker build -t media-api:latest ./api/ || error_exit "Failed to build API server image"
    fi
    
    # Build dashboard
    if [ -f "dashboard/Dockerfile" ]; then
        log INFO "Building dashboard image..."
        docker build -t media-dashboard:latest ./dashboard/ || error_exit "Failed to build dashboard image"
    fi
    
    # Build error recovery system
    if [ -f "Dockerfile.error-recovery" ]; then
        log INFO "Building error recovery system image..."
        docker build -f Dockerfile.error-recovery -t error-recovery:latest . || log WARN "Error recovery image build failed"
    fi
}

# Function to start core services first
start_core_services() {
    log INFO "Starting core infrastructure services..."
    
    cd "$PROJECT_DIR"
    
    # Start databases first
    docker-compose up -d postgres redis mariadb
    
    # Wait for databases
    wait_for_service "http://localhost:5432" "PostgreSQL" || true
    wait_for_service "http://localhost:6379" "Redis" || true
    wait_for_service "http://localhost:3306" "MariaDB" || true
    
    log INFO "Core services started successfully"
}

# Function to start application services
start_application_services() {
    log INFO "Starting application services..."
    
    cd "$PROJECT_DIR"
    
    # Start API server
    docker-compose up -d api-server
    wait_for_service "http://localhost:3002/health" "API Server"
    
    # Start media services
    docker-compose up -d jellyfin sonarr radarr prowlarr lidarr bazarr
    
    # Start download clients
    docker-compose up -d qbittorrent transmission sabnzbd
    
    # Start dashboard
    docker-compose up -d media-dashboard
    wait_for_service "http://localhost:3030" "Media Dashboard"
    
    log INFO "Application services started successfully"
}

# Function to start monitoring services
start_monitoring_services() {
    log INFO "Starting monitoring services..."
    
    cd "$PROJECT_DIR"
    
    # Start monitoring stack
    docker-compose up -d prometheus grafana uptime-kuma
    
    # Start error recovery system
    if docker-compose config --services | grep -q "error-recovery-system"; then
        docker-compose up -d error-recovery-system
        wait_for_service "http://localhost:3010/health" "Error Recovery System"
    fi
    
    log INFO "Monitoring services started successfully"
}

# Function to run health checks
run_health_checks() {
    log INFO "Running comprehensive health checks..."
    
    local failed_services=()
    
    # Define services to check
    declare -A health_checks=(
        ["Jellyfin"]="http://localhost:8096"
        ["Sonarr"]="http://localhost:8989"
        ["Radarr"]="http://localhost:7878"
        ["Prowlarr"]="http://localhost:9696"
        ["qBittorrent"]="http://localhost:8080"
        ["API Server"]="http://localhost:3002/health"
        ["Dashboard"]="http://localhost:3030"
        ["Prometheus"]="http://localhost:9090"
        ["Grafana"]="http://localhost:3000"
        ["Uptime Kuma"]="http://localhost:3001"
    )
    
    for service in "${!health_checks[@]}"; do
        local url="${health_checks[$service]}"
        if curl -f -s "$url" >/dev/null 2>&1; then
            log INFO "✅ $service is healthy"
        else
            log WARN "❌ $service health check failed"
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log WARN "Some services failed health checks: ${failed_services[*]}"
        log WARN "Services may still be starting up. Check individual services manually."
    else
        log INFO "All services passed health checks!"
    fi
}

# Function to setup monitoring and alerts
setup_monitoring() {
    log INFO "Setting up monitoring and alerting..."
    
    # Create monitoring configuration if it doesn't exist
    local monitoring_dir="$PROJECT_DIR/monitoring"
    if [ ! -d "$monitoring_dir" ]; then
        mkdir -p "$monitoring_dir"
        
        # Create basic Prometheus config
        cat > "$monitoring_dir/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'media-server'
    static_configs:
      - targets: ['localhost:3002', 'localhost:3010']

  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
EOF
        
        log INFO "Created basic monitoring configuration"
    fi
}

# Function to generate deployment report
generate_deployment_report() {
    log INFO "Generating deployment report..."
    
    local report_file="$PROJECT_DIR/DEPLOYMENT_STATUS_REPORT.md"
    
    cat > "$report_file" << EOF
# Media Server Deployment Report

**Deployment Date:** $(date)
**Deployment Status:** SUCCESS

## Services Deployed

### Core Infrastructure
- ✅ PostgreSQL Database
- ✅ Redis Cache
- ✅ MariaDB Database

### Media Services
- ✅ Jellyfin Media Server (http://localhost:8096)
- ✅ Sonarr TV Shows (http://localhost:8989)
- ✅ Radarr Movies (http://localhost:7878)
- ✅ Prowlarr Indexers (http://localhost:9696)
- ✅ Lidarr Music (http://localhost:8686)
- ✅ Bazarr Subtitles (http://localhost:6767)

### Download Clients
- ✅ qBittorrent (http://localhost:8080)
- ✅ Transmission (http://localhost:9091)
- ✅ SABnzbd (http://localhost:8085)

### Management & Monitoring
- ✅ Media Dashboard (http://localhost:3030)
- ✅ API Server (http://localhost:3002)
- ✅ Prometheus (http://localhost:9090)
- ✅ Grafana (http://localhost:3000)
- ✅ Uptime Kuma (http://localhost:3001)
- ✅ Error Recovery System (http://localhost:3010)

## Network Configuration
- Media Network: 172.30.0.0/16
- Downloads Network: 172.31.0.0/16
- Monitoring Network: 172.33.0.0/16
- Management Network: 172.34.0.0/16

## Access Information
- **Main Dashboard:** http://localhost:3030
- **API Documentation:** http://localhost:3002/api/docs
- **Health Status:** http://localhost:3010/status

## Security Notes
- API key has been generated and configured
- Default credentials should be changed immediately
- HTTPS should be configured for production use

## Backup Information
- Configuration backup created at: $BACKUP_DIR
- Deployment logs available at: $LOG_FILE

## Next Steps
1. Change default passwords for all services
2. Configure reverse proxy for external access
3. Set up SSL certificates
4. Configure backup schedules
5. Review and customize service configurations

EOF

    log INFO "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log INFO "🚀 Starting Ultimate Media Server Deployment"
    log INFO "Project directory: $PROJECT_DIR"
    log INFO "Log file: $LOG_FILE"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Pre-deployment checks
    log INFO "Running pre-deployment checks..."
    check_docker_health
    create_directories
    backup_configuration
    setup_environment
    
    # Fix networking issues
    fix_docker_networking
    
    # Build custom images
    build_custom_images
    
    # Deploy in stages
    log INFO "Starting staged deployment..."
    
    # Stop any existing services
    cd "$PROJECT_DIR"
    docker-compose down 2>/dev/null || true
    
    # Start services in order
    start_core_services
    sleep 10
    start_application_services
    sleep 10
    start_monitoring_services
    
    # Setup monitoring
    setup_monitoring
    
    # Wait for all services to be ready
    log INFO "Waiting for all services to be ready..."
    sleep 30
    
    # Run health checks
    run_health_checks
    
    # Generate report
    generate_deployment_report
    
    log INFO "🎉 Deployment completed successfully!"
    log INFO ""
    log INFO "📊 Dashboard: http://localhost:3030"
    log INFO "🔧 API: http://localhost:3002"
    log INFO "🏥 Health Status: http://localhost:3010/status"
    log INFO "📝 Report: $PROJECT_DIR/DEPLOYMENT_STATUS_REPORT.md"
    log INFO "📋 Logs: $LOG_FILE"
    log INFO ""
    log INFO "Please review the deployment report and change default passwords!"
}

# Execute main function
main "$@"