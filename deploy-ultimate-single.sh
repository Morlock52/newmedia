#!/bin/bash

# Ultimate Media Server - Single Container Deployment Script
# Comprehensive DevOps automation for rapid deployment and management
# Features: Build, deploy, monitor, backup, security, and update automation

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script metadata
SCRIPT_VERSION="2.0.0"
SCRIPT_NAME="Ultimate Media Server Deployment"
SCRIPT_DATE="2025-08-03"

# Color codes for enhanced output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Configuration defaults
CONTAINER_NAME="ultimate-media-server"
IMAGE_NAME="ultimate-media-server:latest"
COMPOSE_FILE="docker-compose.ultimate-single.yml"
ENV_FILE=".env.ultimate"
LOG_FILE="ultimate-deploy.log"
BACKUP_DIR="./backups"
CONFIG_DIR="./ultimate-config"
MEDIA_DIR="./media"
DOWNLOADS_DIR="./downloads"

# Default ports
DEFAULT_WEB_PORT=80
DEFAULT_JELLYFIN_PORT=8096
DEFAULT_DASHBOARD_PORT=3000
DEFAULT_GRAFANA_PORT=3001

# Performance and resource settings
DEFAULT_MEMORY_LIMIT="4G"
DEFAULT_CPU_LIMIT="2.0"
DEFAULT_MEMORY_RESERVATION="2G"
DEFAULT_CPU_RESERVATION="1.0"

# Functions for enhanced output
print_header() {
    echo -e "\n${PURPLE}======================================${NC}"
    echo -e "${WHITE}$1${NC}"
    echo -e "${PURPLE}======================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${CYAN}🔄 $1${NC}"
}

# Logging function
log() {
    local level="$1"
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $*" >> "$LOG_FILE"
    
    case "$level" in
        "ERROR") print_error "$*" ;;
        "WARN")  print_warning "$*" ;;
        "INFO")  print_info "$*" ;;
        "SUCCESS") print_success "$*" ;;
        *) echo "$*" ;;
    esac
}

# Error handling
handle_error() {
    local line_number="$1"
    local error_code="$2"
    log "ERROR" "Script failed at line $line_number with exit code $error_code"
    print_error "Deployment failed at line $line_number (exit code: $error_code)"
    print_info "Check $LOG_FILE for detailed error information"
    cleanup_on_error
    exit "$error_code"
}

trap 'handle_error ${LINENO} $?' ERR

# Cleanup function for error scenarios
cleanup_on_error() {
    log "INFO" "Performing error cleanup..."
    # Stop any partially started containers
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
}

# System requirements check
check_system_requirements() {
    print_step "Checking system requirements..."
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root is not recommended for security reasons"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Aborted by user"
            exit 1
        fi
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space
    available_space=$(df . | tail -1 | awk '{print $4}')
    local required_space=$((10 * 1024 * 1024))  # 10GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        print_warning "Low disk space detected. At least 10GB recommended."
        print_info "Available: $(( available_space / 1024 / 1024 ))GB"
    fi
    
    # Check available memory (minimum 4GB recommended)
    local available_memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    
    if [[ $available_memory -lt 4096 ]]; then
        print_warning "Low memory detected. At least 4GB recommended."
        print_info "Available: ${available_memory}MB"
    fi
    
    # Check if ports are available
    local ports_to_check=(80 8096 8989 7878 8686 9696 8080 3000 3001 9090)
    local ports_in_use=()
    
    for port in "${ports_to_check[@]}"; do
        if netstat -tulpn 2>/dev/null | grep -q ":$port "; then
            ports_in_use+=("$port")
        fi
    done
    
    if [[ ${#ports_in_use[@]} -gt 0 ]]; then
        print_warning "The following ports are already in use: ${ports_in_use[*]}"
        print_info "You may need to modify port mappings in the configuration"
    fi
    
    log "SUCCESS" "System requirements check completed"
    print_success "System requirements check passed"
}

# Create directory structure
create_directories() {
    print_step "Creating directory structure..."
    
    local directories=(
        "$CONFIG_DIR"
        "$CONFIG_DIR/caddy"
        "$CONFIG_DIR/jellyfin"
        "$CONFIG_DIR/sonarr"
        "$CONFIG_DIR/radarr"
        "$CONFIG_DIR/lidarr"
        "$CONFIG_DIR/prowlarr"
        "$CONFIG_DIR/qbittorrent"
        "$CONFIG_DIR/homepage"
        "$CONFIG_DIR/prometheus"
        "$CONFIG_DIR/grafana"
        "$CONFIG_DIR/backup"
        "$CONFIG_DIR/security"
        "$MEDIA_DIR"
        "$MEDIA_DIR/movies"
        "$MEDIA_DIR/tv"
        "$MEDIA_DIR/music"
        "$MEDIA_DIR/books"
        "$DOWNLOADS_DIR"
        "$DOWNLOADS_DIR/complete"
        "$DOWNLOADS_DIR/incomplete"
        "$BACKUP_DIR"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "INFO" "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    local current_user
    current_user=$(id -u)
    local current_group
    current_group=$(id -g)
    
    chown -R "$current_user:$current_group" "$CONFIG_DIR" "$MEDIA_DIR" "$DOWNLOADS_DIR" "$BACKUP_DIR" 2>/dev/null || true
    
    print_success "Directory structure created"
}

# Generate environment file
generate_env_file() {
    print_step "Generating environment configuration..."
    
    if [[ -f "$ENV_FILE" ]]; then
        print_warning "Environment file already exists: $ENV_FILE"
        read -p "Overwrite existing configuration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing environment file"
            return 0
        fi
    fi
    
    # Get user input for configuration
    echo -e "\n${CYAN}Configuration Setup:${NC}"
    
    read -p "Enter your timezone (default: UTC): " tz
    tz=${tz:-UTC}
    
    read -p "Enter web interface port (default: 80): " web_port
    web_port=${web_port:-$DEFAULT_WEB_PORT}
    
    read -p "Enter domain name (default: localhost): " domain
    domain=${domain:-localhost}
    
    read -p "Enable hardware acceleration? (y/N): " hw_accel
    hw_accel=${hw_accel:-N}
    
    read -p "Memory limit (default: 4G): " memory_limit
    memory_limit=${memory_limit:-$DEFAULT_MEMORY_LIMIT}
    
    read -p "CPU limit (default: 2.0): " cpu_limit
    cpu_limit=${cpu_limit:-$DEFAULT_CPU_LIMIT}
    
    # Advanced options
    echo -e "\n${CYAN}Advanced Options (press Enter for defaults):${NC}"
    
    read -p "Enable VPN integration? (y/N): " vpn_enabled
    vpn_enabled=${vpn_enabled:-N}
    
    read -p "Enable automatic backups? (Y/n): " backup_enabled
    backup_enabled=${backup_enabled:-Y}
    
    read -p "Backup retention days (default: 7): " backup_retention
    backup_retention=${backup_retention:-7}
    
    read -p "Enable email notifications? (y/N): " email_enabled
    email_enabled=${email_enabled:-N}
    
    # Generate .env file
    cat > "$ENV_FILE" <<EOF
# Ultimate Media Server Configuration
# Generated on $(date)

# System Configuration
PUID=$(id -u)
PGID=$(id -g)
TZ=$tz

# Network Configuration
DOMAIN=$domain
WEB_PORT=$web_port
JELLYFIN_PORT=$DEFAULT_JELLYFIN_PORT
SONARR_PORT=8989
RADARR_PORT=7878
LIDARR_PORT=8686
PROWLARR_PORT=9696
QBITTORRENT_PORT=8080
DASHBOARD_PORT=$DEFAULT_DASHBOARD_PORT
GRAFANA_PORT=$DEFAULT_GRAFANA_PORT
PROMETHEUS_PORT=9090
MCP_SUITE_PORT=8090

# Storage Paths
CONFIG_PATH=$CONFIG_DIR
MEDIA_PATH=$MEDIA_DIR
DOWNLOADS_PATH=$DOWNLOADS_DIR
BACKUP_PATH=$BACKUP_DIR

# Resource Limits
MEMORY_LIMIT=$memory_limit
CPU_LIMIT=$cpu_limit
MEMORY_RESERVATION=$DEFAULT_MEMORY_RESERVATION
CPU_RESERVATION=$DEFAULT_CPU_RESERVATION

# Feature Toggles
HARDWARE_ACCELERATION=$([ "$hw_accel" = "y" ] || [ "$hw_accel" = "Y" ] && echo "true" || echo "false")
VPN_ENABLED=$([ "$vpn_enabled" = "y" ] || [ "$vpn_enabled" = "Y" ] && echo "true" || echo "false")
SECURITY_SCAN_ENABLED=true
FAIL2BAN_ENABLED=true
AUTO_UPDATE=true

# Backup Configuration
BACKUP_RETENTION_DAYS=$backup_retention
BACKUP_SCHEDULE="0 */6 * * *"

# Monitoring Configuration
PROMETHEUS_RETENTION_TIME=15d
GRAFANA_USER=admin
GRAFANA_PASSWORD=$(openssl rand -base64 12)

# Performance Tuning
JELLYFIN_CACHE_SIZE=256
SONARR_MEMORY=512
RADARR_MEMORY=512

# AI/MCP Suite
MCP_SUITE_ENABLED=true
AI_ASSISTANT_ENABLED=true

# Optional: VPN Configuration (fill if VPN_ENABLED=true)
VPN_PROVIDER=
OPENVPN_USER=
OPENVPN_PASSWORD=
VPN_COUNTRIES=Switzerland

# Optional: Email Notifications (fill if email_enabled=true)
SMTP_SERVER=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
EMAIL_FROM=
EMAIL_TO=

# Optional: Database Configuration
POSTGRES_DB=mediaserver
POSTGRES_USER=mediaserver
POSTGRES_PASSWORD=$(openssl rand -base64 12)

# Optional: External Services
TRAEFIK_ENABLED=false
TRAEFIK_TLS=false

# Build Configuration
BUILDKIT_PROGRESS=plain
EOF
    
    log "SUCCESS" "Environment file generated: $ENV_FILE"
    print_success "Configuration saved to $ENV_FILE"
    
    # Show generated passwords
    echo -e "\n${YELLOW}Generated Passwords (save these securely):${NC}"
    echo "Grafana Admin Password: $(grep GRAFANA_PASSWORD "$ENV_FILE" | cut -d'=' -f2)"
    echo "Database Password: $(grep POSTGRES_PASSWORD "$ENV_FILE" | cut -d'=' -f2)"
}

# Build container image
build_image() {
    print_step "Building Ultimate Media Server container..."
    
    log "INFO" "Starting container build process"
    
    # Enable BuildKit for improved build performance
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    # Build with progress output
    if ! docker build \
        --file Dockerfile.ultimate-single \
        --tag "$IMAGE_NAME" \
        --progress=plain \
        --build-arg BUILDKIT_PROGRESS=plain \
        --build-arg TARGETPLATFORM=linux/amd64 \
        .; then
        log "ERROR" "Container build failed"
        print_error "Failed to build container image"
        return 1
    fi
    
    log "SUCCESS" "Container build completed successfully"
    print_success "Container image built: $IMAGE_NAME"
}

# Deploy services
deploy_services() {
    print_step "Deploying Ultimate Media Server..."
    
    # Source environment file
    if [[ -f "$ENV_FILE" ]]; then
        set -a  # Automatically export all variables
        source "$ENV_FILE"
        set +a
    fi
    
    log "INFO" "Starting service deployment"
    
    # Pull any base images that might be needed
    docker-compose -f "$COMPOSE_FILE" pull --ignore-pull-failures || print_warning "Some images couldn't be pulled (using local builds)"
    
    # Deploy with docker-compose
    if ! docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans; then
        log "ERROR" "Service deployment failed"
        print_error "Failed to deploy services"
        return 1
    fi
    
    log "SUCCESS" "Services deployed successfully"
    print_success "Ultimate Media Server deployed successfully"
    
    # Wait for services to be ready
    wait_for_services
}

# Wait for services to be ready
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    local max_attempts=60  # 5 minutes max
    local attempt=0
    local services_ready=false
    
    while [[ $attempt -lt $max_attempts ]] && [[ $services_ready == false ]]; do
        attempt=$((attempt + 1))
        
        if docker exec "$CONTAINER_NAME" /usr/local/bin/healthcheck &>/dev/null; then
            services_ready=true
            break
        fi
        
        echo -n "."
        sleep 5
    done
    
    echo  # New line after dots
    
    if [[ $services_ready == true ]]; then
        log "SUCCESS" "All services are ready"
        print_success "All services are ready and healthy"
    else
        log "WARN" "Services may not be fully ready yet"
        print_warning "Services are taking longer than expected to start"
        print_info "This is normal for the first startup. Check status with: $0 status"
    fi
}

# Show service status
show_status() {
    print_header "Ultimate Media Server Status"
    
    # Container status
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "$CONTAINER_NAME"; then
        print_success "Container is running"
        echo
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep "$CONTAINER_NAME"
        echo
    else
        print_error "Container is not running"
        return 1
    fi
    
    # Service health check
    print_step "Checking service health..."
    if docker exec "$CONTAINER_NAME" /usr/local/bin/healthcheck; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
    fi
    
    # Show service URLs
    local web_port
    web_port=$(grep WEB_PORT "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 || echo "$DEFAULT_WEB_PORT")
    local domain
    domain=$(grep DOMAIN "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 || echo "localhost")
    
    echo -e "\n${CYAN}Service URLs:${NC}"
    echo "🏠 Main Dashboard:    http://$domain:$web_port"
    echo "🎬 Jellyfin:          http://$domain:$web_port/jellyfin"
    echo "📺 Sonarr:            http://$domain:$web_port/sonarr"
    echo "🎥 Radarr:            http://$domain:$web_port/radarr"
    echo "🎵 Lidarr:            http://$domain:$web_port/lidarr"
    echo "🔍 Prowlarr:          http://$domain:$web_port/prowlarr"
    echo "⬇️  qBittorrent:       http://$domain:$web_port/qbittorrent"
    echo "📊 Grafana:           http://$domain:$web_port/grafana"
    echo "🤖 AI Assistant:      http://$domain:$web_port/mcp"
    
    # Resource usage
    echo -e "\n${CYAN}Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" "$CONTAINER_NAME" 2>/dev/null || print_warning "Could not retrieve resource stats"
}

# View logs
view_logs() {
    local service="${1:-}"
    
    if [[ -n "$service" ]]; then
        print_step "Viewing logs for service: $service"
        docker-compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        print_step "Viewing all service logs"
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Update services
update_services() {
    print_step "Updating Ultimate Media Server..."
    
    log "INFO" "Starting update process"
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Rebuild if necessary
    read -p "Rebuild container from source? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_image
    fi
    
    # Recreate containers with new images
    docker-compose -f "$COMPOSE_FILE" up -d --force-recreate --remove-orphans
    
    log "SUCCESS" "Update completed"
    print_success "Services updated successfully"
    
    wait_for_services
}

# Backup data
backup_data() {
    print_step "Creating backup..."
    
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/ultimate_media_backup_$timestamp.tar.gz"
    
    log "INFO" "Creating backup: $backup_file"
    
    # Create compressed backup
    if tar -czf "$backup_file" -C . \
        --exclude="./backups" \
        --exclude="./media" \
        --exclude="./downloads" \
        --exclude=".git" \
        --exclude="*.log" \
        "$CONFIG_DIR" "$ENV_FILE" "$COMPOSE_FILE" 2>/dev/null; then
        
        local backup_size
        backup_size=$(du -h "$backup_file" | cut -f1)
        
        log "SUCCESS" "Backup created successfully: $backup_file ($backup_size)"
        print_success "Backup created: $backup_file ($backup_size)"
        
        # Cleanup old backups (keep last 7 by default)
        local retention_days
        retention_days=$(grep BACKUP_RETENTION_DAYS "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 || echo "7")
        
        find "$BACKUP_DIR" -name "ultimate_media_backup_*.tar.gz" -mtime +"$retention_days" -delete 2>/dev/null || true
        
        print_info "Old backups cleaned up (retention: $retention_days days)"
    else
        log "ERROR" "Backup creation failed"
        print_error "Failed to create backup"
        return 1
    fi
}

# Restore from backup
restore_backup() {
    local backup_file="$1"
    
    if [[ ! -f "$backup_file" ]]; then
        print_error "Backup file not found: $backup_file"
        return 1
    fi
    
    print_warning "This will restore configuration from backup and restart services"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Restore cancelled"
        return 0
    fi
    
    print_step "Restoring from backup: $backup_file"
    
    # Stop services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Extract backup
    if tar -xzf "$backup_file"; then
        log "SUCCESS" "Backup restored successfully"
        print_success "Configuration restored from backup"
        
        # Restart services
        deploy_services
    else
        log "ERROR" "Backup restore failed"
        print_error "Failed to restore from backup"
        return 1
    fi
}

# Stop services
stop_services() {
    print_step "Stopping Ultimate Media Server..."
    
    log "INFO" "Stopping all services"
    
    if docker-compose -f "$COMPOSE_FILE" down --remove-orphans; then
        log "SUCCESS" "Services stopped successfully"
        print_success "All services stopped"
    else
        log "ERROR" "Failed to stop some services"
        print_error "Failed to stop services cleanly"
        return 1
    fi
}

# Remove everything
remove_all() {
    print_warning "This will remove all containers, images, and optionally data"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removal cancelled"
        return 0
    fi
    
    read -p "Also remove configuration and data directories? (y/N): " -n 1 -r
    echo
    local remove_data=$REPLY
    
    print_step "Removing Ultimate Media Server..."
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans --volumes
    
    # Remove images
    docker rmi "$IMAGE_NAME" 2>/dev/null || print_warning "Could not remove image (may not exist)"
    
    # Remove data directories if requested
    if [[ $remove_data =~ ^[Yy]$ ]]; then
        rm -rf "$CONFIG_DIR" "$BACKUP_DIR" 2>/dev/null || print_warning "Could not remove some directories"
        print_info "Configuration and backup directories removed"
    fi
    
    print_success "Ultimate Media Server removed"
}

# Show help
show_help() {
    cat << EOF
${WHITE}$SCRIPT_NAME v$SCRIPT_VERSION${NC}
${CYAN}Complete DevOps automation for Ultimate Media Server${NC}

${WHITE}USAGE:${NC}
    $0 [COMMAND] [OPTIONS]

${WHITE}COMMANDS:${NC}
    ${GREEN}install${NC}         Complete installation (check requirements, build, deploy)
    ${GREEN}build${NC}           Build container image from source
    ${GREEN}deploy${NC}          Deploy/start services
    ${GREEN}status${NC}          Show service status and health
    ${GREEN}logs${NC} [service]  View logs (all services or specific service)
    ${GREEN}update${NC}          Update services to latest versions
    ${GREEN}backup${NC}          Create configuration backup
    ${GREEN}restore${NC} <file>  Restore from backup file
    ${GREEN}stop${NC}            Stop all services
    ${GREEN}restart${NC}         Restart all services
    ${GREEN}remove${NC}          Remove all containers and optionally data
    ${GREEN}help${NC}            Show this help message

${WHITE}EXAMPLES:${NC}
    $0 install                    # Complete fresh installation
    $0 deploy                     # Deploy with existing configuration
    $0 status                     # Check service status
    $0 logs jellyfin             # View Jellyfin logs
    $0 backup                     # Create backup
    $0 restore backup_file.tar.gz # Restore from backup
    $0 update                     # Update to latest versions

${WHITE}FILES:${NC}
    $ENV_FILE              # Environment configuration
    $COMPOSE_FILE      # Docker Compose configuration
    $LOG_FILE                 # Deployment logs
    Dockerfile.ultimate-single   # Container build file

${WHITE}DIRECTORIES:${NC}
    $CONFIG_DIR/              # Service configurations
    $MEDIA_DIR/                    # Media files
    $DOWNLOADS_DIR/            # Downloads
    $BACKUP_DIR/                # Backups

${WHITE}MONITORING:${NC}
    Main Dashboard:  http://localhost/
    Grafana:         http://localhost/grafana
    Prometheus:      http://localhost/prometheus
    Health Check:    http://localhost/health

${WHITE}SUPPORT:${NC}
    Documentation: README.md
    Logs: $LOG_FILE
    Health Check: $0 status

${CYAN}For detailed configuration options, edit $ENV_FILE after running 'install'${NC}
EOF
}

# Main command processing
main() {
    local command="${1:-help}"
    
    # Create log file
    touch "$LOG_FILE"
    
    log "INFO" "Ultimate Media Server Deployment Script v$SCRIPT_VERSION started"
    log "INFO" "Command: $command"
    
    case "$command" in
        "install")
            print_header "$SCRIPT_NAME v$SCRIPT_VERSION"
            print_info "Starting complete installation process..."
            check_system_requirements
            create_directories
            generate_env_file
            build_image
            deploy_services
            show_status
            print_success "Installation completed successfully!"
            echo -e "\n${CYAN}Next steps:${NC}"
            echo "1. Visit http://localhost to access the dashboard"
            echo "2. Configure your media libraries in Jellyfin"
            echo "3. Set up download clients and indexers"
            echo "4. Check $0 status for service health"
            ;;
        "build")
            check_system_requirements
            build_image
            ;;
        "deploy")
            create_directories
            deploy_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            view_logs "$2"
            ;;
        "update")
            update_services
            ;;
        "backup")
            backup_data
            ;;
        "restore")
            if [[ -z "${2:-}" ]]; then
                print_error "Please specify backup file to restore"
                print_info "Usage: $0 restore <backup_file.tar.gz>"
                exit 1
            fi
            restore_backup "$2"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            deploy_services
            ;;
        "remove")
            remove_all
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
    
    log "SUCCESS" "Command '$command' completed successfully"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi