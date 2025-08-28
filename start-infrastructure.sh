#!/bin/bash
# Infrastructure Startup Script
# Deploys the complete monitoring, security, and infrastructure stack

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.infrastructure.yml"
ENV_FILE=".env"
PROJECT_NAME="media-server-infrastructure"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%H:%M:%S')] ${message}${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "$BLUE" "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_status "$RED" "ERROR: Docker is not running"
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_status "$RED" "ERROR: docker-compose is not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        print_status "$YELLOW" "WARNING: .env file not found, creating from template..."
        if [[ -f ".env.template" ]]; then
            cp ".env.template" "$ENV_FILE"
            print_status "$YELLOW" "Please edit .env file with your configuration before continuing"
            exit 1
        else
            print_status "$RED" "ERROR: Neither .env nor .env.template found"
            exit 1
        fi
    fi
    
    # Check if infrastructure compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_status "$RED" "ERROR: $COMPOSE_FILE not found"
        exit 1
    fi
    
    print_status "$GREEN" "Prerequisites check passed"
}

# Function to create required directories
create_directories() {
    print_status "$BLUE" "Creating required directories..."
    
    local directories=(
        "prometheus"
        "grafana/provisioning/datasources"
        "grafana/provisioning/dashboards"
        "grafana/dashboards"
        "authelia"
        "traefik/dynamic"
        "traefik/certs"
        "webhooks"
        "scripts"
        "loki"
        "promtail"
        "duplicati-config"
        "duplicati-backups"
        "fail2ban"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "$GREEN" "Created directory: $dir"
        fi
    done
}

# Function to set correct permissions
set_permissions() {
    print_status "$BLUE" "Setting correct permissions..."
    
    # Make scripts executable
    if [[ -d "scripts" ]]; then
        chmod +x scripts/*.sh 2>/dev/null || true
    fi
    
    # Set Traefik certificate directory permissions
    if [[ -d "traefik/certs" ]]; then
        chmod 600 traefik/certs/* 2>/dev/null || true
    fi
    
    # Set Authelia config permissions
    if [[ -f "authelia/configuration.yml" ]]; then
        chmod 600 authelia/configuration.yml
    fi
    
    if [[ -f "authelia/users_database.yml" ]]; then
        chmod 600 authelia/users_database.yml
    fi
}

# Function to validate configuration
validate_configuration() {
    print_status "$BLUE" "Validating configuration..."
    
    # Load environment variables
    set -a
    source "$ENV_FILE"
    set +a
    
    # Check critical variables
    local required_vars=(
        "DOMAIN"
        "ACME_EMAIL"
        "AUTHELIA_JWT_SECRET"
        "AUTHELIA_SESSION_SECRET"
        "AUTHELIA_STORAGE_ENCRYPTION_KEY"
        "POSTGRES_PASSWORD"
        "GRAFANA_PASSWORD"
        "ADMIN_EMAIL"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_status "$RED" "ERROR: Missing required environment variables:"
        printf "%s\n" "${missing_vars[@]}"
        print_status "$YELLOW" "Please configure these variables in $ENV_FILE"
        exit 1
    fi
    
    # Validate secret lengths
    if [[ ${#AUTHELIA_JWT_SECRET} -lt 32 ]]; then
        print_status "$RED" "ERROR: AUTHELIA_JWT_SECRET must be at least 32 characters"
        exit 1
    fi
    
    if [[ ${#AUTHELIA_SESSION_SECRET} -lt 32 ]]; then
        print_status "$RED" "ERROR: AUTHELIA_SESSION_SECRET must be at least 32 characters"
        exit 1
    fi
    
    if [[ ${#AUTHELIA_STORAGE_ENCRYPTION_KEY} -lt 32 ]]; then
        print_status "$RED" "ERROR: AUTHELIA_STORAGE_ENCRYPTION_KEY must be at least 32 characters"
        exit 1
    fi
    
    print_status "$GREEN" "Configuration validation passed"
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_status "$BLUE" "Deploying infrastructure services..."
    
    # Pull latest images
    print_status "$YELLOW" "Pulling latest Docker images..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" pull
    
    # Start infrastructure services in order
    print_status "$BLUE" "Starting core infrastructure..."
    
    # Start databases first
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d redis postgres
    
    # Wait for databases
    print_status "$YELLOW" "Waiting for databases to be ready..."
    sleep 10
    
    # Start VPN
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d gluetun
    
    # Start authentication
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d authelia
    
    # Start reverse proxy
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d traefik
    
    # Start monitoring stack
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d \
        prometheus alertmanager node-exporter cadvisor
    
    # Start log aggregation
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d loki promtail
    
    # Start visualization
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d grafana
    
    # Start remaining services
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
}

# Function to wait for services
wait_for_services() {
    print_status "$BLUE" "Waiting for services to become healthy..."
    
    local services=(
        "traefik:8080/ping"
        "prometheus:9090/-/healthy"
        "grafana:3000/api/health"
        "loki:3100/ready"
        "authelia:9091/api/health"
    )
    
    for service in "${services[@]}"; do
        local name="${service%%:*}"
        local endpoint="http://${service}"
        
        print_status "$YELLOW" "Waiting for $name..."
        
        local attempts=0
        local max_attempts=30
        
        while (( attempts < max_attempts )); do
            if curl -sf "$endpoint" >/dev/null 2>&1; then
                print_status "$GREEN" "$name is ready"
                break
            fi
            
            ((attempts++))
            sleep 10
        done
        
        if (( attempts >= max_attempts )); then
            print_status "$RED" "WARNING: $name did not become ready within expected time"
        fi
    done
}

# Function to display status
display_status() {
    print_status "$BLUE" "Infrastructure deployment summary:"
    echo ""
    
    # Load environment variables for URLs
    set -a
    source "$ENV_FILE"
    set +a
    
    echo "🌐 Web Interfaces:"
    echo "   Traefik Dashboard: http://traefik.${DOMAIN}:8080"
    echo "   Grafana:          http://grafana.${DOMAIN}"
    echo "   Prometheus:       http://prometheus.${DOMAIN}"
    echo "   Alertmanager:     http://alertmanager.${DOMAIN}"
    echo "   Uptime Kuma:      http://uptime.${DOMAIN}"
    echo "   Authelia:         http://auth.${DOMAIN}"
    echo ""
    
    echo "🔐 Default Credentials:"
    echo "   Grafana:    admin / ${GRAFANA_PASSWORD}"
    echo "   Authelia:   Configure in authelia/users_database.yml"
    echo ""
    
    echo "📊 Monitoring:"
    echo "   Prometheus metrics collection active"
    echo "   Grafana dashboards provisioned"
    echo "   Log aggregation via Loki/Promtail"
    echo "   Alert routing via Alertmanager"
    echo ""
    
    echo "🛡️ Security:"
    echo "   2FA authentication via Authelia"
    echo "   SSL termination via Traefik"
    echo "   VPN protection for downloads"
    echo "   Rate limiting and security headers"
    echo ""
    
    print_status "$GREEN" "Infrastructure deployment completed successfully!"
    print_status "$YELLOW" "Remember to:"
    echo "   1. Configure users in authelia/users_database.yml"
    echo "   2. Set up DNS records for your domain"
    echo "   3. Configure webhook endpoints in your *arr services"
    echo "   4. Review and customize Grafana dashboards"
}

# Function to show help
show_help() {
    echo "Infrastructure Deployment Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start     Deploy infrastructure services (default)"
    echo "  stop      Stop all infrastructure services"
    echo "  restart   Restart all infrastructure services"
    echo "  status    Show service status"
    echo "  logs      Show service logs"
    echo "  update    Pull latest images and restart"
    echo "  help      Show this help message"
    echo ""
}

# Function to stop services
stop_services() {
    print_status "$BLUE" "Stopping infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
    print_status "$GREEN" "Infrastructure services stopped"
}

# Function to show status
show_status() {
    print_status "$BLUE" "Infrastructure service status:"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
}

# Function to show logs
show_logs() {
    local service="${1:-}"
    if [[ -n "$service" ]]; then
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f "$service"
    else
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
    fi
}

# Function to update services
update_services() {
    print_status "$BLUE" "Updating infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" pull
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
    print_status "$GREEN" "Infrastructure services updated"
}

# Main execution
main() {
    local action="${1:-start}"
    
    case "$action" in
        "start")
            check_prerequisites
            create_directories
            set_permissions
            validate_configuration
            deploy_infrastructure
            wait_for_services
            display_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 5
            deploy_infrastructure
            wait_for_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "update")
            update_services
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_status "$RED" "Unknown action: $action"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"