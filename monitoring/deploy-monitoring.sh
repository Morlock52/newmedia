#!/bin/bash

# Ultimate Media Server Monitoring Deployment Script
# Deploys comprehensive monitoring stack with Prometheus, Grafana, and advanced alerting

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
MONITORING_STACK_NAME="media-monitoring"
MEDIA_STACK_NAME="media-server"

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${PURPLE}================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}================================${NC}\n"
}

# Check if Docker and Docker Compose are available
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Create necessary directories
create_directories() {
    log_header "Creating Directory Structure"
    
    local dirs=(
        "$SCRIPT_DIR/config/prometheus"
        "$SCRIPT_DIR/config/grafana/provisioning/datasources"
        "$SCRIPT_DIR/config/grafana/provisioning/dashboards"
        "$SCRIPT_DIR/config/grafana/dashboards"
        "$SCRIPT_DIR/config/alertmanager"
        "$SCRIPT_DIR/config/loki"
        "$SCRIPT_DIR/config/vector"
        "$SCRIPT_DIR/logs"
        "$SCRIPT_DIR/data/prometheus"
        "$SCRIPT_DIR/data/grafana"
        "$SCRIPT_DIR/data/loki"
        "$SCRIPT_DIR/data/alertmanager"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directory structure created"
}

# Set proper permissions
set_permissions() {
    log_header "Setting Permissions"
    
    # Set ownership for Grafana
    sudo chown -R 472:472 "$SCRIPT_DIR/data/grafana" 2>/dev/null || {
        log_warning "Could not set Grafana ownership (may need sudo)"
    }
    
    # Set ownership for Prometheus
    sudo chown -R 65534:65534 "$SCRIPT_DIR/data/prometheus" 2>/dev/null || {
        log_warning "Could not set Prometheus ownership (may need sudo)"
    }
    
    # Set ownership for Loki
    sudo chown -R 10001:10001 "$SCRIPT_DIR/data/loki" 2>/dev/null || {
        log_warning "Could not set Loki ownership (may need sudo)"
    }
    
    # Make scripts executable
    chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
    
    log_success "Permissions configured"
}

# Copy configuration files to proper locations
copy_configurations() {
    log_header "Copying Configuration Files"
    
    # Copy Prometheus config
    if [[ -f "$SCRIPT_DIR/prometheus/prometheus.yml" ]]; then
        cp "$SCRIPT_DIR/prometheus/prometheus.yml" "$SCRIPT_DIR/config/prometheus/"
        log_info "Copied Prometheus configuration"
    fi
    
    # Copy Grafana provisioning
    if [[ -d "$SCRIPT_DIR/grafana/provisioning" ]]; then
        cp -r "$SCRIPT_DIR/grafana/provisioning/"* "$SCRIPT_DIR/config/grafana/provisioning/"
        log_info "Copied Grafana provisioning configuration"
    fi
    
    # Copy dashboards
    if [[ -d "$SCRIPT_DIR/grafana/dashboards" ]]; then
        cp "$SCRIPT_DIR/grafana/dashboards/"*.json "$SCRIPT_DIR/config/grafana/dashboards/"
        log_info "Copied Grafana dashboards"
    fi
    
    # Copy Alertmanager config
    if [[ -f "$SCRIPT_DIR/alertmanager/config.yml" ]]; then
        cp "$SCRIPT_DIR/alertmanager/config.yml" "$SCRIPT_DIR/config/alertmanager/"
        log_info "Copied Alertmanager configuration"
    fi
    
    # Copy Vector config
    if [[ -f "$SCRIPT_DIR/vector/vector.toml" ]]; then
        cp "$SCRIPT_DIR/vector/vector.toml" "$SCRIPT_DIR/config/vector/"
        log_info "Copied Vector configuration"
    fi
    
    log_success "Configuration files copied"
}

# Generate environment file if it doesn't exist
generate_env_file() {
    log_header "Generating Environment Configuration"
    
    local env_file="$SCRIPT_DIR/.env"
    
    if [[ ! -f "$env_file" ]]; then
        cat > "$env_file" << EOF
# Media Server Monitoring Configuration
# Generated on $(date)

# User/Group IDs
PUID=1000
PGID=1000

# Timezone
TZ=America/New_York

# Grafana Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin123
GRAFANA_PORT=3000

# Prometheus Configuration
PROMETHEUS_PORT=9090

# Alertmanager Configuration
ALERTMANAGER_PORT=9093

# Email Configuration for Alerts
ADMIN_EMAIL=admin@mediaserver.local
SMTP_HOST=localhost:587
SMTP_USERNAME=
SMTP_PASSWORD=
ALERT_FROM_EMAIL=alerts@mediaserver.local

# Webhook URLs (optional)
DISCORD_WEBHOOK_URL=
SLACK_WEBHOOK_URL=

# API Keys (configure these after setup)
JELLYFIN_API_KEY=
SONARR_API_KEY=
RADARR_API_KEY=
TAUTULLI_API_KEY=

# Service URLs
JELLYFIN_URL=http://jellyfin:8096
SONARR_URL=http://sonarr:8989
RADARR_URL=http://radarr:7878
QBITTORRENT_URL=http://qbittorrent:8080

# Storage Paths
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads
EOF
        log_success "Generated environment file: $env_file"
        log_warning "Please review and update the environment variables in $env_file"
    else
        log_info "Environment file already exists: $env_file"
    fi
}

# Pull required Docker images
pull_images() {
    log_header "Pulling Docker Images"
    
    local images=(
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "prom/alertmanager:latest"
        "grafana/loki:latest"
        "grafana/promtail:latest"
        "prom/node-exporter:latest"
        "gcr.io/cadvisor/cadvisor:latest"
        "prom/blackbox-exporter:latest"
        "timberio/vector:latest-alpine"
        "miguelndecarvalho/speedtest-exporter:latest"
        "prometheuscommunity/smartctl-exporter:latest"
    )
    
    for image in "${images[@]}"; do
        log_info "Pulling $image..."
        docker pull "$image" || log_warning "Failed to pull $image"
    done
    
    log_success "Docker images pulled"
}

# Create external networks if they don't exist
create_networks() {
    log_header "Creating Docker Networks"
    
    # Check if media network exists
    if ! docker network ls | grep -q media_network; then
        docker network create media_network
        log_info "Created media_network"
    else
        log_info "media_network already exists"
    fi
    
    log_success "Docker networks ready"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_header "Deploying Monitoring Stack"
    
    cd "$SCRIPT_DIR"
    
    # Stop existing monitoring stack
    log_info "Stopping any existing monitoring containers..."
    docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true
    
    # Deploy the monitoring stack
    log_info "Starting monitoring stack..."
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Monitoring stack deployed successfully!"
}

# Check health of deployed services
check_service_health() {
    log_header "Checking Service Health"
    
    local services=(
        "prometheus:9090"
        "grafana:3000"
        "alertmanager:9093"
        "loki:3100"
        "node-exporter:9100"
        "cadvisor:8083"
    )
    
    for service in "${services[@]}"; do
        local name="${service%%:*}"
        local port="${service##*:}"
        
        if curl -sf "http://localhost:$port" >/dev/null 2>&1; then
            log_success "$name is healthy (port $port)"
        else
            log_warning "$name is not responding (port $port)"
        fi
    done
}

# Display access information
show_access_info() {
    log_header "Access Information"
    
    echo -e "${CYAN}Monitoring Services:${NC}"
    echo -e "  🎨 Grafana Dashboard:  ${GREEN}http://localhost:3000${NC} (admin/admin123)"
    echo -e "  📊 Prometheus:         ${GREEN}http://localhost:9090${NC}"
    echo -e "  🚨 Alertmanager:       ${GREEN}http://localhost:9093${NC}"
    echo -e "  📝 Loki:               ${GREEN}http://localhost:3100${NC}"
    echo -e "  🖥️  Node Exporter:      ${GREEN}http://localhost:9100${NC}"
    echo -e "  📦 cAdvisor:           ${GREEN}http://localhost:8083${NC}"
    
    echo -e "\n${CYAN}Pre-configured Dashboards:${NC}"
    echo -e "  • Media Server Overview"
    echo -e "  • Performance Deep Dive"
    echo -e "  • User Activity Analytics"
    echo -e "  • Container Overview"
    echo -e "  • Resource Usage"
    echo -e "  • Security Alerts"
    
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo -e "  1. Update API keys in $SCRIPT_DIR/.env"
    echo -e "  2. Configure email settings for alerts"
    echo -e "  3. Customize dashboard variables"
    echo -e "  4. Set up notification channels (Discord, Slack)"
    echo -e "  5. Review and adjust alert thresholds"
    
    echo -e "\n${CYAN}Documentation:${NC}"
    echo -e "  📖 Prometheus: https://prometheus.io/docs/"
    echo -e "  📖 Grafana: https://grafana.com/docs/"
    echo -e "  📖 Alertmanager: https://prometheus.io/docs/alerting/latest/alertmanager/"
}

# Main deployment function
main() {
    log_header "🚀 Ultimate Media Server Monitoring Deployment"
    
    echo -e "${CYAN}This script will deploy a comprehensive monitoring stack including:${NC}"
    echo -e "  • Prometheus (metrics collection)"
    echo -e "  • Grafana (visualization & dashboards)"
    echo -e "  • Alertmanager (alerting)"
    echo -e "  • Loki (log aggregation)"
    echo -e "  • Vector (log processing)"
    echo -e "  • Various exporters (system, container, custom metrics)"
    echo -e "  • Pre-built beautiful dashboards"
    echo -e "  • Intelligent alerting rules"
    
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
    
    # Run deployment steps
    check_prerequisites
    create_directories
    generate_env_file
    copy_configurations
    set_permissions
    create_networks
    pull_images
    deploy_monitoring
    
    # Show final information
    show_access_info
    
    log_success "🎉 Monitoring deployment completed successfully!"
    echo -e "\n${GREEN}Your media server monitoring is now active and collecting metrics!${NC}"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping monitoring stack..."
        cd "$SCRIPT_DIR"
        docker-compose -f docker-compose.monitoring.yml down
        log_success "Monitoring stack stopped"
        ;;
    "restart")
        log_info "Restarting monitoring stack..."
        cd "$SCRIPT_DIR"
        docker-compose -f docker-compose.monitoring.yml restart
        log_success "Monitoring stack restarted"
        ;;
    "logs")
        cd "$SCRIPT_DIR"
        docker-compose -f docker-compose.monitoring.yml logs -f
        ;;
    "status")
        check_service_health
        ;;
    "update")
        log_info "Updating monitoring stack..."
        pull_images
        cd "$SCRIPT_DIR"
        docker-compose -f docker-compose.monitoring.yml up -d
        log_success "Monitoring stack updated"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|update}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete monitoring stack"
        echo "  stop     - Stop all monitoring services"
        echo "  restart  - Restart all monitoring services"
        echo "  logs     - View logs from all services"
        echo "  status   - Check health of all services"
        echo "  update   - Update and restart monitoring stack"
        exit 1
        ;;
esac