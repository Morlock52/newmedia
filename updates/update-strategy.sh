#!/bin/bash

# Docker Container Update Strategy Script
# Safe, reliable updates with backup and rollback capability
# Based on 2025 best practices - manual updates with automated safeguards

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
LOG_FILE="${LOG_FILE:-./updates/update.log}"
MAX_BACKUPS="${MAX_BACKUPS:-10}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Ensure required directories exist
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    local level=$1
    shift
    echo -e "${level}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "$BLUE" "Checking prerequisites..."
    
    local missing=()
    
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing+=("docker-compose")
    command -v jq >/dev/null 2>&1 || missing+=("jq")
    
    if [ ${#missing[@]} -ne 0 ]; then
        log "$RED" "Missing required tools: ${missing[*]}"
        log "$RED" "Please install missing tools and try again."
        exit 1
    fi
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        log "$RED" "Docker compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log "$GREEN" "Prerequisites check passed"
}

# Get list of running services
get_running_services() {
    docker-compose -f "$COMPOSE_FILE" ps --services --filter "status=running" 2>/dev/null || true
}

# Create backup of current state
create_backup() {
    local service=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="${service}_${timestamp}"
    
    log "$BLUE" "Creating backup for $service..."
    
    # Backup container volumes
    local volumes=$(docker inspect "$service" 2>/dev/null | jq -r '.[0].Mounts[] | select(.Type == "volume") | .Name' || true)
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            log "$YELLOW" "  Backing up volume: $volume"
            if [ "$DRY_RUN" = "false" ]; then
                docker run --rm \
                    -v "$volume":/source:ro \
                    -v "$BACKUP_DIR":/backup \
                    alpine tar czf "/backup/${backup_name}_${volume}.tar.gz" -C /source .
            fi
        done
    fi
    
    # Backup container configuration
    if [ "$DRY_RUN" = "false" ]; then
        docker inspect "$service" > "$BACKUP_DIR/${backup_name}_config.json"
    fi
    
    # Clean up old backups
    cleanup_old_backups "$service"
    
    log "$GREEN" "Backup completed for $service"
}

# Clean up old backups
cleanup_old_backups() {
    local service=$1
    local backup_count=$(find "$BACKUP_DIR" -name "${service}_*" -type f | wc -l)
    
    if [ "$backup_count" -gt "$MAX_BACKUPS" ]; then
        log "$YELLOW" "Cleaning up old backups for $service (keeping last $MAX_BACKUPS)..."
        if [ "$DRY_RUN" = "false" ]; then
            find "$BACKUP_DIR" -name "${service}_*" -type f -printf '%T@ %p\n' | \
                sort -n | head -n -"$MAX_BACKUPS" | cut -d' ' -f2- | xargs rm -f
        fi
    fi
}

# Health check for service
health_check() {
    local service=$1
    local timeout=$2
    local start_time=$(date +%s)
    
    log "$BLUE" "Performing health check for $service..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ "$elapsed" -gt "$timeout" ]; then
            log "$RED" "Health check timeout for $service"
            return 1
        fi
        
        # Check if container is running
        if ! docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log "$YELLOW" "  Container not running, waiting..."
            sleep 5
            continue
        fi
        
        # Check container health status if available
        local health_status=$(docker inspect "$service" 2>/dev/null | jq -r '.[0].State.Health.Status // "none"')
        
        case "$health_status" in
            "healthy")
                log "$GREEN" "  Health check passed for $service"
                return 0
                ;;
            "unhealthy")
                log "$RED" "  Container is unhealthy"
                return 1
                ;;
            "starting")
                log "$YELLOW" "  Health check starting, waiting..."
                ;;
            "none")
                # No health check defined, check if container is running
                if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
                    log "$GREEN" "  Container is running (no health check defined)"
                    return 0
                fi
                ;;
        esac
        
        sleep 5
    done
}

# Update single service
update_service() {
    local service=$1
    local rollback_info=""
    
    log "$BLUE" "========================================="
    log "$BLUE" "Updating service: $service"
    log "$BLUE" "========================================="
    
    # Get current image info
    local current_image=$(docker inspect "$service" 2>/dev/null | jq -r '.[0].Config.Image' || echo "unknown")
    log "$YELLOW" "Current image: $current_image"
    
    # Create backup
    create_backup "$service"
    
    # Pull new image
    log "$BLUE" "Pulling latest image..."
    if [ "$DRY_RUN" = "false" ]; then
        if ! docker-compose -f "$COMPOSE_FILE" pull "$service" 2>&1 | tee -a "$LOG_FILE"; then
            log "$RED" "Failed to pull image for $service"
            return 1
        fi
    else
        log "$YELLOW" "DRY RUN: Would pull latest image"
    fi
    
    # Get new image info
    local new_image=$(docker-compose -f "$COMPOSE_FILE" config | grep -A5 "^  $service:" | grep "image:" | awk '{print $2}')
    log "$YELLOW" "New image: $new_image"
    
    # Stop and remove old container
    log "$BLUE" "Stopping old container..."
    if [ "$DRY_RUN" = "false" ]; then
        docker-compose -f "$COMPOSE_FILE" stop "$service"
        docker-compose -f "$COMPOSE_FILE" rm -f "$service"
    else
        log "$YELLOW" "DRY RUN: Would stop and remove container"
    fi
    
    # Start new container
    log "$BLUE" "Starting new container..."
    if [ "$DRY_RUN" = "false" ]; then
        if ! docker-compose -f "$COMPOSE_FILE" up -d "$service" 2>&1 | tee -a "$LOG_FILE"; then
            log "$RED" "Failed to start new container for $service"
            if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
                rollback_service "$service" "$current_image"
            fi
            return 1
        fi
    else
        log "$YELLOW" "DRY RUN: Would start new container"
    fi
    
    # Perform health check
    if [ "$DRY_RUN" = "false" ]; then
        if ! health_check "$service" "$HEALTH_CHECK_TIMEOUT"; then
            log "$RED" "Health check failed for $service"
            if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
                rollback_service "$service" "$current_image"
            fi
            return 1
        fi
    else
        log "$YELLOW" "DRY RUN: Would perform health check"
    fi
    
    # Clean up old images
    log "$BLUE" "Cleaning up old images..."
    if [ "$DRY_RUN" = "false" ]; then
        docker image prune -f >/dev/null 2>&1 || true
    fi
    
    log "$GREEN" "Successfully updated $service"
    return 0
}

# Rollback service to previous version
rollback_service() {
    local service=$1
    local previous_image=$2
    
    log "$YELLOW" "Rolling back $service to $previous_image..."
    
    # Stop current container
    docker-compose -f "$COMPOSE_FILE" stop "$service"
    docker-compose -f "$COMPOSE_FILE" rm -f "$service"
    
    # Update compose file with previous image (temporarily)
    # Note: In production, you might want to pin versions in compose file
    
    # Start previous version
    docker run -d --name "$service" "$previous_image"
    
    if health_check "$service" 60; then
        log "$GREEN" "Rollback successful for $service"
    else
        log "$RED" "Rollback failed for $service - manual intervention required!"
        exit 1
    fi
}

# Update all services
update_all_services() {
    local services=$(get_running_services)
    local failed_services=()
    local updated_services=()
    
    log "$BLUE" "Starting update process for all services..."
    log "$YELLOW" "Services to update: $(echo $services | tr '\n' ' ')"
    
    for service in $services; do
        if update_service "$service"; then
            updated_services+=("$service")
        else
            failed_services+=("$service")
        fi
        echo
    done
    
    # Summary
    log "$BLUE" "========================================="
    log "$BLUE" "Update Summary"
    log "$BLUE" "========================================="
    
    if [ ${#updated_services[@]} -gt 0 ]; then
        log "$GREEN" "Successfully updated: ${updated_services[*]}"
    fi
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log "$RED" "Failed to update: ${failed_services[*]}"
        return 1
    fi
    
    return 0
}

# Check for updates without applying
check_updates() {
    log "$BLUE" "Checking for available updates..."
    
    local updates_available=false
    local services=$(get_running_services)
    
    for service in $services; do
        local current_image=$(docker inspect "$service" 2>/dev/null | jq -r '.[0].Config.Image' || echo "unknown")
        local current_id=$(docker inspect "$service" 2>/dev/null | jq -r '.[0].Image' || echo "unknown")
        
        # Pull latest image info
        docker-compose -f "$COMPOSE_FILE" pull "$service" >/dev/null 2>&1
        
        local new_id=$(docker images --format "{{.ID}}" "$(echo "$current_image" | cut -d: -f1):latest" | head -1)
        
        if [ "$current_id" != "$new_id" ] && [ -n "$new_id" ]; then
            log "$YELLOW" "Update available for $service"
            updates_available=true
        else
            log "$GREEN" "No update for $service"
        fi
    done
    
    if [ "$updates_available" = "true" ]; then
        log "$YELLOW" "Updates are available. Run with --update to apply."
        return 0
    else
        log "$GREEN" "All services are up to date."
        return 1
    fi
}

# Main function
main() {
    case "${1:-}" in
        --check)
            check_prerequisites
            check_updates
            ;;
        --update)
            check_prerequisites
            if [ "${2:-}" = "--dry-run" ]; then
                DRY_RUN=true
                log "$YELLOW" "DRY RUN MODE - No changes will be made"
            fi
            update_all_services
            ;;
        --update-service)
            check_prerequisites
            if [ -z "${2:-}" ]; then
                log "$RED" "Service name required"
                exit 1
            fi
            if [ "${3:-}" = "--dry-run" ]; then
                DRY_RUN=true
                log "$YELLOW" "DRY RUN MODE - No changes will be made"
            fi
            update_service "$2"
            ;;
        --backup)
            check_prerequisites
            if [ -z "${2:-}" ]; then
                log "$RED" "Service name required"
                exit 1
            fi
            create_backup "$2"
            ;;
        --help|*)
            cat << EOF
Docker Container Update Strategy Script

Usage: $0 [command] [options]

Commands:
  --check              Check for available updates
  --update             Update all services
  --update-service     Update specific service
  --backup             Create backup of specific service
  --help               Show this help message

Options:
  --dry-run            Simulate update without making changes

Environment Variables:
  COMPOSE_FILE         Path to docker-compose.yml (default: ./docker-compose.yml)
  BACKUP_DIR           Backup directory (default: ./backups)
  LOG_FILE             Log file path (default: ./updates/update.log)
  MAX_BACKUPS          Maximum backups to keep per service (default: 10)
  HEALTH_CHECK_TIMEOUT Health check timeout in seconds (default: 300)
  ROLLBACK_ON_FAILURE  Auto-rollback on failure (default: true)

Examples:
  $0 --check                           # Check for updates
  $0 --update                          # Update all services
  $0 --update --dry-run                # Simulate update
  $0 --update-service plex             # Update specific service
  $0 --backup sonarr                   # Backup specific service

EOF
            ;;
    esac
}

# Run main function
main "$@"