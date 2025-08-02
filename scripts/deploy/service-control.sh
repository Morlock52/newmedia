#!/bin/bash

# Media Server - Service Control Script
# Start, stop, restart, and manage services
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

# Command from arguments
COMMAND="${1:-help}"
SERVICE="${2:-}"

# Service groups
declare -A SERVICE_GROUPS=(
    ["core"]="jellyfin qbittorrent homepage"
    ["media"]="radarr sonarr prowlarr bazarr"
    ["utility"]="overseerr tautulli portainer"
    ["all"]=""
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

# Get compose command
get_compose_cmd() {
    local cmd="docker-compose"
    
    if [ -f "$OVERRIDE_FILE" ]; then
        cmd="$cmd -f $COMPOSE_FILE -f $OVERRIDE_FILE"
    fi
    
    echo "$cmd"
}

# Start services
start_services() {
    log_header "Starting Services"
    
    local compose_cmd=$(get_compose_cmd)
    local services="$1"
    
    if [ -z "$services" ]; then
        log_info "Starting all services..."
        $compose_cmd up -d
    else
        log_info "Starting: $services"
        $compose_cmd up -d $services
    fi
    
    # Wait for services to be ready
    sleep 5
    
    # Show status
    status_services "$services"
}

# Stop services
stop_services() {
    log_header "Stopping Services"
    
    local compose_cmd=$(get_compose_cmd)
    local services="$1"
    
    if [ -z "$services" ]; then
        log_info "Stopping all services..."
        $compose_cmd stop
    else
        log_info "Stopping: $services"
        $compose_cmd stop $services
    fi
}

# Restart services
restart_services() {
    log_header "Restarting Services"
    
    local compose_cmd=$(get_compose_cmd)
    local services="$1"
    
    if [ -z "$services" ]; then
        log_info "Restarting all services..."
        $compose_cmd restart
    else
        log_info "Restarting: $services"
        $compose_cmd restart $services
    fi
    
    # Wait for services to be ready
    sleep 5
    
    # Show status
    status_services "$services"
}

# Show service status
status_services() {
    log_header "Service Status"
    
    local compose_cmd=$(get_compose_cmd)
    local services="$1"
    
    if [ -z "$services" ]; then
        $compose_cmd ps
    else
        $compose_cmd ps $services
    fi
    
    # Show container health
    echo -e "\n${CYAN}Container Health:${NC}"
    
    local running_services=$($compose_cmd ps --services)
    for service in $running_services; do
        local health=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "none")
        local state=$(docker inspect --format='{{.State.Status}}' "$service" 2>/dev/null || echo "unknown")
        
        case "$health" in
            "healthy")
                echo -e "  ${GREEN}✓${NC} $service: $state (healthy)"
                ;;
            "unhealthy")
                echo -e "  ${RED}✗${NC} $service: $state (unhealthy)"
                ;;
            "starting")
                echo -e "  ${YELLOW}⟳${NC} $service: $state (starting)"
                ;;
            *)
                if [ "$state" = "running" ]; then
                    echo -e "  ${GREEN}✓${NC} $service: $state"
                else
                    echo -e "  ${RED}✗${NC} $service: $state"
                fi
                ;;
        esac
    done
}

# Show service logs
logs_services() {
    log_header "Service Logs"
    
    local compose_cmd=$(get_compose_cmd)
    local services="$1"
    local follow="${FOLLOW:-true}"
    
    if [ "$follow" = "true" ]; then
        if [ -z "$services" ]; then
            $compose_cmd logs -f --tail 100
        else
            $compose_cmd logs -f --tail 100 $services
        fi
    else
        if [ -z "$services" ]; then
            $compose_cmd logs --tail 100
        else
            $compose_cmd logs --tail 100 $services
        fi
    fi
}

# Pull latest images
pull_images() {
    log_header "Pulling Latest Images"
    
    local compose_cmd=$(get_compose_cmd)
    local services="$1"
    
    if [ -z "$services" ]; then
        log_info "Pulling all images..."
        $compose_cmd pull
    else
        log_info "Pulling images for: $services"
        $compose_cmd pull $services
    fi
}

# Update services
update_services() {
    log_header "Updating Services"
    
    local services="$1"
    
    # Pull latest images
    pull_images "$services"
    
    # Recreate containers with new images
    local compose_cmd=$(get_compose_cmd)
    
    if [ -z "$services" ]; then
        log_info "Recreating all containers..."
        $compose_cmd up -d --force-recreate
    else
        log_info "Recreating: $services"
        $compose_cmd up -d --force-recreate $services
    fi
    
    # Clean up old images
    log_info "Cleaning up old images..."
    docker image prune -f
}

# Execute command in service
exec_service() {
    log_header "Executing Command"
    
    local service="$1"
    shift
    local command="$*"
    
    if [ -z "$service" ]; then
        log_error "Service name required"
        exit 1
    fi
    
    if [ -z "$command" ]; then
        log_info "Opening shell in $service..."
        docker-compose exec "$service" sh
    else
        log_info "Executing in $service: $command"
        docker-compose exec "$service" $command
    fi
}

# Show service ports
ports_services() {
    log_header "Service Ports"
    
    echo -e "${CYAN}Service${NC}          ${CYAN}Port${NC}      ${CYAN}URL${NC}"
    echo "────────────────────────────────────────────────"
    
    # Define service ports
    declare -A SERVICE_PORTS=(
        ["Homepage"]="3000"
        ["Jellyfin"]="8096"
        ["qBittorrent"]="8080"
        ["Radarr"]="7878"
        ["Sonarr"]="8989"
        ["Prowlarr"]="9696"
        ["Bazarr"]="6767"
        ["Overseerr"]="5055"
        ["Tautulli"]="8181"
        ["Portainer"]="9000"
    )
    
    for service in "${!SERVICE_PORTS[@]}"; do
        local port="${SERVICE_PORTS[$service]}"
        local container_name=$(echo "$service" | tr '[:upper:]' '[:lower:]')
        
        # Check if service is running
        if docker ps --format "{{.Names}}" | grep -q "$container_name"; then
            printf "%-15s  ${GREEN}%-8s${NC}  http://localhost:%s\n" "$service" "$port" "$port"
        else
            printf "%-15s  ${RED}%-8s${NC}  (not running)\n" "$service" "$port"
        fi
    done
}

# Clean up containers and volumes
cleanup_services() {
    log_header "Cleanup"
    
    echo -e "${YELLOW}⚠️  WARNING: This will remove containers and optionally volumes${NC}"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        return
    fi
    
    local compose_cmd=$(get_compose_cmd)
    
    # Stop and remove containers
    log_info "Removing containers..."
    $compose_cmd down
    
    # Ask about volumes
    read -p "Remove volumes? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing volumes..."
        $compose_cmd down -v
    fi
    
    # Clean up orphaned resources
    log_info "Cleaning up orphaned resources..."
    docker system prune -f
}

# Reset service
reset_service() {
    log_header "Reset Service"
    
    local service="$1"
    
    if [ -z "$service" ]; then
        log_error "Service name required"
        exit 1
    fi
    
    echo -e "${YELLOW}⚠️  WARNING: This will reset $service configuration${NC}"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Reset cancelled"
        return
    fi
    
    # Stop service
    log_info "Stopping $service..."
    docker-compose stop "$service"
    
    # Remove container
    log_info "Removing $service container..."
    docker-compose rm -f "$service"
    
    # Ask about config
    read -p "Remove $service configuration? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local config_path="${PROJECT_ROOT}/config/$service"
        if [ -d "$config_path" ]; then
            log_info "Backing up configuration..."
            mv "$config_path" "${config_path}.backup.$(date +%Y%m%d_%H%M%S)"
        fi
    fi
    
    # Recreate service
    log_info "Recreating $service..."
    docker-compose up -d "$service"
}

# Show help
show_help() {
    cat << EOF
${CYAN}Media Server Service Control${NC}

Usage: $0 <command> [service|group]

${CYAN}Commands:${NC}
  start [service]     Start service(s)
  stop [service]      Stop service(s)
  restart [service]   Restart service(s)
  status [service]    Show service status
  logs [service]      Show service logs (use FOLLOW=false for static)
  pull [service]      Pull latest images
  update [service]    Update and restart service(s)
  exec <service> [cmd] Execute command in service
  ports               Show service ports
  cleanup             Remove containers and volumes
  reset <service>     Reset service configuration
  help                Show this help

${CYAN}Service Groups:${NC}
  core     Core services (Jellyfin, qBittorrent, Homepage)
  media    Media management (Radarr, Sonarr, Prowlarr, Bazarr)
  utility  Utility services (Overseerr, Tautulli, Portainer)
  all      All services (default)

${CYAN}Examples:${NC}
  $0 start              # Start all services
  $0 stop core          # Stop core services
  $0 restart jellyfin   # Restart Jellyfin
  $0 logs sonarr        # Follow Sonarr logs
  $0 update             # Update all services
  $0 exec qbittorrent sh # Open shell in qBittorrent

EOF
}

# Resolve service group
resolve_services() {
    local input="$1"
    
    # Check if it's a group
    if [ -n "${SERVICE_GROUPS[$input]:-}" ]; then
        echo "${SERVICE_GROUPS[$input]}"
    else
        # Return as-is (individual service)
        echo "$input"
    fi
}

# Main control flow
main() {
    # Check Docker
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check docker-compose.yml
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found"
        exit 1
    fi
    
    # Resolve services
    local services=$(resolve_services "$SERVICE")
    
    # Execute command
    case "$COMMAND" in
        start)
            start_services "$services"
            ;;
        stop)
            stop_services "$services"
            ;;
        restart)
            restart_services "$services"
            ;;
        status)
            status_services "$services"
            ;;
        logs)
            logs_services "$services"
            ;;
        pull)
            pull_images "$services"
            ;;
        update)
            update_services "$services"
            ;;
        exec)
            shift 2
            exec_service "$SERVICE" "$@"
            ;;
        ports)
            ports_services
            ;;
        cleanup)
            cleanup_services
            ;;
        reset)
            reset_service "$SERVICE"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"