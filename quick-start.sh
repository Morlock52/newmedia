#!/bin/bash

# 🚀 Ultimate Media Server 2025 - Quick Start Script
# Built with enterprise-grade 2025 technologies and best practices

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
SHIELD="🛡️"
CHART="📊"
GEAR="⚙️"
CHECK="✅"
WARNING="⚠️"
ERROR="❌"
INFO="ℹ️"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
COMPOSE_PROJECT_NAME="ultimate-media-server"
ENV_FILE=".env.production"
ENV_TEMPLATE=".env.production.template"

# Print colored output
print_color() {
    printf "${2}${1}${NC}\n"
}

# Print header
print_header() {
    echo
    print_color "================================================================" "$CYAN"
    print_color "$1" "$CYAN"
    print_color "================================================================" "$CYAN"
    echo
}

# Print step
print_step() {
    print_color "${GEAR} $1" "$BLUE"
}

# Print success
print_success() {
    print_color "${CHECK} $1" "$GREEN"
}

# Print warning
print_warning() {
    print_color "${WARNING} $1" "$YELLOW"
}

# Print error
print_error() {
    print_color "${ERROR} $1" "$RED"
}

# Print info
print_info() {
    print_color "${INFO} $1" "$PURPLE"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if ! command_exists curl; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo
        print_info "Please install the missing dependencies:"
        echo
        for dep in "${missing_deps[@]}"; do
            case $dep in
                docker)
                    echo "  🐳 Docker: https://docs.docker.com/get-docker/"
                    ;;
                docker-compose)
                    echo "  🐙 Docker Compose: https://docs.docker.com/compose/install/"
                    ;;
                curl)
                    echo "  🌐 curl: Use your package manager (apt, yum, brew, etc.)"
                    ;;
            esac
        done
        echo
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "All prerequisites met!"
}

# Check system requirements
check_system_requirements() {
    print_step "Checking system requirements..."
    
    # Check available memory (minimum 4GB recommended)
    if command_exists free; then
        local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$mem_gb" -lt 4 ]; then
            print_warning "System has ${mem_gb}GB RAM. 4GB+ recommended for optimal performance."
        else
            print_success "Memory check passed: ${mem_gb}GB RAM available"
        fi
    fi
    
    # Check available disk space (minimum 20GB recommended)
    local disk_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$disk_gb" -lt 20 ]; then
        print_warning "Available disk space: ${disk_gb}GB. 20GB+ recommended."
    else
        print_success "Disk space check passed: ${disk_gb}GB available"
    fi
}

# Setup environment
setup_environment() {
    print_step "Setting up environment configuration..."
    
    if [ ! -f "$ENV_TEMPLATE" ]; then
        print_error "Environment template not found: $ENV_TEMPLATE"
        exit 1
    fi
    
    if [ ! -f "$ENV_FILE" ]; then
        print_info "Creating environment file from template..."
        cp "$ENV_TEMPLATE" "$ENV_FILE"
        print_success "Environment file created: $ENV_FILE"
        
        print_warning "IMPORTANT: You must edit $ENV_FILE to configure your settings!"
        echo
        print_info "Required changes:"
        echo "  🌐 DOMAIN_NAME=yourdomain.com"
        echo "  🔑 Cloudflare API credentials"
        echo "  🛡️ Strong passwords for all services"
        echo "  📧 Email for SSL certificates"
        echo
        read -p "Press Enter to open the configuration file in nano, or Ctrl+C to exit..."
        nano "$ENV_FILE"
    else
        print_success "Environment file exists: $ENV_FILE"
    fi
}

# Create required directories
create_directories() {
    print_step "Creating required directories..."
    
    local directories=(
        "data"
        "logs"
        "backups"
        "docker/init"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        fi
    done
}

# Validate configuration
validate_configuration() {
    print_step "Validating configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    # Source environment file
    source "$ENV_FILE"
    
    # Check critical variables
    local required_vars=(
        "DOMAIN_NAME"
        "CLOUDFLARE_EMAIL"
        "CLOUDFLARE_API_KEY"
        "PUID"
        "PGID"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ] || [ "${!var}" = "CHANGEME" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing or unchanged configuration variables:"
        for var in "${missing_vars[@]}"; do
            echo "  ❌ $var"
        done
        echo
        print_info "Please edit $ENV_FILE and set these variables."
        exit 1
    fi
    
    print_success "Configuration validation passed!"
}

# Deploy services
deploy_services() {
    local deployment_type="$1"
    
    case $deployment_type in
        "minimal")
            print_step "Deploying minimal media server stack..."
            docker-compose -f docker-compose.production.yml --profile minimal up -d
            ;;
        "full")
            print_step "Deploying full production stack..."
            docker-compose -f docker-compose.production.yml up -d
            ;;
        "monitoring")
            print_step "Deploying monitoring stack..."
            cd monitoring && ./deploy-monitoring.sh deploy
            cd ..
            ;;
        *)
            print_error "Unknown deployment type: $deployment_type"
            exit 1
            ;;
    esac
}

# Wait for services
wait_for_services() {
    print_step "Waiting for services to start..."
    
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose -f docker-compose.production.yml ps | grep -q "Up"; then
            print_success "Services are starting up!"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Services are taking longer than expected to start."
        print_info "You can check status with: docker-compose -f docker-compose.production.yml ps"
    fi
}

# Run health checks
run_health_checks() {
    print_step "Running health checks..."
    
    if [ -f "tests/run-tests.sh" ]; then
        chmod +x tests/run-tests.sh
        ./tests/run-tests.sh smoke
    else
        print_info "Test suite not found. Running basic health checks..."
        
        # Basic container health check
        local unhealthy_containers=$(docker-compose -f docker-compose.production.yml ps --filter "health=unhealthy" -q)
        
        if [ -n "$unhealthy_containers" ]; then
            print_warning "Some containers are unhealthy. Check logs for details."
        else
            print_success "Basic health checks passed!"
        fi
    fi
}

# Show access information
show_access_info() {
    source "$ENV_FILE"
    
    print_header "${ROCKET} Your Media Server is Ready!"
    
    echo
    print_info "Access your services at:"
    echo
    echo "  🏠 Dashboard:     https://${DOMAIN_NAME}"
    echo "  🎬 Jellyfin:      https://jellyfin.${DOMAIN_NAME}"
    echo "  📺 Sonarr:        https://sonarr.${DOMAIN_NAME}"
    echo "  🎬 Radarr:        https://radarr.${DOMAIN_NAME}"
    echo "  📊 Grafana:       https://grafana.${DOMAIN_NAME}"
    echo "  🔐 Authelia:      https://auth.${DOMAIN_NAME}"
    echo "  ⚙️ Portainer:     https://portainer.${DOMAIN_NAME}"
    echo
    
    print_info "Default credentials (change these!):"
    echo "  📊 Grafana:  admin / changeme"
    echo "  ⚙️ Portainer: admin / (set on first login)"
    echo
    
    print_info "Useful commands:"
    echo "  🔍 Check status:  docker-compose -f docker-compose.production.yml ps"
    echo "  📋 View logs:     docker-compose -f docker-compose.production.yml logs -f"
    echo "  🔄 Restart:       docker-compose -f docker-compose.production.yml restart"
    echo "  🛑 Stop:          docker-compose -f docker-compose.production.yml down"
    echo "  📊 Monitor:       ./monitoring/deploy-monitoring.sh deploy"
    echo "  🧪 Test:          ./tests/run-tests.sh all"
    echo
}

# Show help
show_help() {
    cat << EOF
${ROCKET} Ultimate Media Server 2025 - Quick Start

Usage: $0 [OPTIONS] [COMMAND]

Commands:
  minimal     Deploy minimal media server (Jellyfin, Sonarr, Radarr)
  full        Deploy full production stack (default)
  monitoring  Deploy monitoring stack only
  check       Check prerequisites and configuration
  help        Show this help message

Options:
  --no-health-check    Skip health checks after deployment
  --skip-validation    Skip configuration validation (not recommended)

Examples:
  $0                   # Deploy full production stack
  $0 minimal           # Deploy minimal media server
  $0 monitoring        # Deploy monitoring only
  $0 check             # Check system and configuration

EOF
}

# Main function
main() {
    local deployment_type="full"
    local skip_health_check=false
    local skip_validation=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            minimal|full|monitoring|check)
                deployment_type="$1"
                shift
                ;;
            help|--help|-h)
                show_help
                exit 0
                ;;
            --no-health-check)
                skip_health_check=true
                shift
                ;;
            --skip-validation)
                skip_validation=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Print banner
    print_header "${ROCKET} Ultimate Media Server 2025 - Quick Start"
    print_info "Enterprise-grade containerized media server with 2025 best practices"
    echo
    
    # Run checks
    check_prerequisites
    check_system_requirements
    
    if [ "$deployment_type" = "check" ]; then
        setup_environment
        if [ "$skip_validation" = false ]; then
            validate_configuration
        fi
        print_success "System check completed successfully!"
        exit 0
    fi
    
    # Setup and deploy
    setup_environment
    create_directories
    
    if [ "$skip_validation" = false ]; then
        validate_configuration
    fi
    
    # Deploy based on type
    deploy_services "$deployment_type"
    wait_for_services
    
    # Health checks
    if [ "$skip_health_check" = false ]; then
        run_health_checks
    fi
    
    # Show access information
    show_access_info
    
    print_success "Deployment completed successfully!"
    
    # Final tips
    echo
    print_info "Next steps:"
    echo "  1. Configure your services through their web interfaces"
    echo "  2. Set up your media libraries in Jellyfin"
    echo "  3. Configure download clients in Sonarr/Radarr"
    echo "  4. Check monitoring dashboards in Grafana"
    echo "  5. Read the complete guide: DOCKER_DEPLOYMENT_GUIDE_2025.md"
    echo
    
    print_color "${ROCKET} Enjoy your enterprise-grade media server!" "$GREEN"
}

# Run main function with all arguments
main "$@"