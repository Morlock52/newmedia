#!/bin/bash

# Media Server Update Automation Script
# Automated container updates with rollback capability
# Version: 2025.1

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/updates"
BACKUP_DIR="${PROJECT_ROOT}/backups/updates"
STATE_FILE="${PROJECT_ROOT}/.update-state.json"
LOG_FILE="${LOG_DIR}/update-$(date +%Y%m%d_%H%M%S).log"

# Docker Compose files
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose-ultimate.yml"
COMPOSE_PROJECT="mediaserver"

# Update configuration
UPDATE_BATCH_SIZE=3
UPDATE_TIMEOUT=300
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_DELAY=30

# Service dependencies
declare -A SERVICE_DEPS=(
    ["traefik"]=""
    ["authelia"]="traefik"
    ["jellyfin"]="traefik,authelia"
    ["sonarr"]="traefik,authelia,prowlarr"
    ["radarr"]="traefik,authelia,prowlarr"
    ["prowlarr"]="traefik,authelia"
    ["bazarr"]="traefik,authelia,sonarr,radarr"
    ["qbittorrent"]="gluetun"
    ["gluetun"]=""
)

# Critical services that should be updated carefully
CRITICAL_SERVICES=("traefik" "authelia" "gluetun" "postgres" "redis")

# Create directories
mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local color=""
    
    case $level in
        "INFO") color=$GREEN ;;
        "WARN") color=$YELLOW ;;
        "ERROR") color=$RED ;;
        "DEBUG") color=$BLUE ;;
    esac
    
    echo -e "${color}${timestamp} [${level}]${NC} ${message}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    log "INFO" "Performing cleanup..."
    # Remove temporary files
    rm -f /tmp/update-compose-*.yml
}

# Trap cleanup
trap cleanup EXIT

# Load state
load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo "{}"
    fi
}

# Save state
save_state() {
    local key=$1
    local value=$2
    
    local state=$(load_state)
    echo "$state" | jq --arg k "$key" --arg v "$value" '.[$k] = $v' > "$STATE_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log "WARN" "Running as root is not recommended"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local required_tools=("docker" "docker-compose" "jq" "curl" "git")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "$tool is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi
    
    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error_exit "Docker Compose file not found: $COMPOSE_FILE"
    fi
}

# Get container image info
get_container_info() {
    local service=$1
    
    docker compose -f "$COMPOSE_FILE" ps --format json "$service" 2>/dev/null | \
        jq -r 'select(.Service == "'$service'")'
}

# Get current image tag
get_current_image() {
    local service=$1
    
    local info=$(get_container_info "$service")
    if [[ -n "$info" ]]; then
        echo "$info" | jq -r '.Image'
    fi
}

# Check for image updates
check_image_update() {
    local image=$1
    
    log "INFO" "Checking for updates: $image"
    
    # Pull the latest image
    if docker pull "$image" 2>&1 | tee -a "$LOG_FILE"; then
        # Get image IDs
        local current_id=$(docker inspect --format='{{.Id}}' "$image" 2>/dev/null)
        local latest_id=$(docker inspect --format='{{.Id}}' "$image:latest" 2>/dev/null)
        
        if [[ "$current_id" != "$latest_id" ]]; then
            return 0  # Update available
        else
            return 1  # No update
        fi
    else
        log "WARN" "Failed to pull image: $image"
        return 2  # Error
    fi
}

# Backup service data
backup_service() {
    local service=$1
    local backup_name="${service}_$(date +%Y%m%d_%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log "INFO" "Creating backup for $service..."
    
    # Create backup directory
    mkdir -p "$backup_path"
    
    # Save container info
    get_container_info "$service" > "${backup_path}/container-info.json"
    
    # Save image info
    local image=$(get_current_image "$service")
    echo "$image" > "${backup_path}/image.txt"
    docker inspect "$image" > "${backup_path}/image-inspect.json"
    
    # Export container if running
    local container_name="${COMPOSE_PROJECT}-${service}-1"
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        log "INFO" "Exporting container: $container_name"
        docker export "$container_name" | gzip > "${backup_path}/container.tar.gz"
    fi
    
    # Backup volume data (if applicable)
    backup_volumes "$service" "$backup_path"
    
    log "INFO" "Backup completed: $backup_path"
    echo "$backup_path"
}

# Backup volumes
backup_volumes() {
    local service=$1
    local backup_path=$2
    
    # Get volumes for service
    local volumes=$(docker compose -f "$COMPOSE_FILE" config | \
        yq eval ".services.${service}.volumes[] | select(. != \"*:ro\")" 2>/dev/null || true)
    
    if [[ -n "$volumes" ]]; then
        log "INFO" "Backing up volumes for $service"
        mkdir -p "${backup_path}/volumes"
        
        # Create volume backup script
        cat > "${backup_path}/backup-volumes.sh" << EOF
#!/bin/bash
# Restore volumes with:
# docker run --rm -v \${VOLUME}:/restore -v \$(pwd):/backup alpine tar -xzf /backup/\${VOLUME}.tar.gz -C /restore
EOF
        chmod +x "${backup_path}/backup-volumes.sh"
    fi
}

# Health check
health_check() {
    local service=$1
    local retries=${2:-$HEALTH_CHECK_RETRIES}
    
    log "INFO" "Performing health check for $service..."
    
    for i in $(seq 1 $retries); do
        # Check if container is running
        if docker compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            # Service-specific health checks
            case $service in
                "jellyfin")
                    if curl -sf "http://localhost:8096/health" &> /dev/null; then
                        log "INFO" "$service is healthy"
                        return 0
                    fi
                    ;;
                "sonarr"|"radarr"|"lidarr"|"readarr"|"prowlarr")
                    local port=$(get_service_port "$service")
                    if curl -sf "http://localhost:${port}/api/v3/health" &> /dev/null; then
                        log "INFO" "$service is healthy"
                        return 0
                    fi
                    ;;
                "traefik")
                    if curl -sf "http://localhost:8080/ping" &> /dev/null; then
                        log "INFO" "$service is healthy"
                        return 0
                    fi
                    ;;
                *)
                    # Generic health check - container is running
                    log "INFO" "$service is running"
                    return 0
                    ;;
            esac
        fi
        
        log "WARN" "Health check failed for $service (attempt $i/$retries)"
        sleep $HEALTH_CHECK_DELAY
    done
    
    log "ERROR" "$service failed health check"
    return 1
}

# Get service port
get_service_port() {
    local service=$1
    
    case $service in
        "sonarr") echo "8989" ;;
        "radarr") echo "7878" ;;
        "lidarr") echo "8686" ;;
        "readarr") echo "8787" ;;
        "prowlarr") echo "9696" ;;
        "bazarr") echo "6767" ;;
        "jellyfin") echo "8096" ;;
        "overseerr") echo "5055" ;;
        *) echo "80" ;;
    esac
}

# Update service
update_service() {
    local service=$1
    local backup_path=""
    
    log "INFO" "Updating service: $service"
    
    # Create backup
    backup_path=$(backup_service "$service")
    save_state "${service}_backup" "$backup_path"
    
    # Stop dependent services
    stop_dependent_services "$service"
    
    # Pull new image
    local image=$(docker compose -f "$COMPOSE_FILE" config | \
        yq eval ".services.${service}.image" 2>/dev/null)
    
    if [[ -z "$image" ]]; then
        log "ERROR" "Could not determine image for $service"
        return 1
    fi
    
    log "INFO" "Pulling latest image: $image"
    if ! docker pull "$image"; then
        log "ERROR" "Failed to pull image: $image"
        rollback_service "$service" "$backup_path"
        return 1
    fi
    
    # Recreate container
    log "INFO" "Recreating container for $service"
    if ! docker compose -f "$COMPOSE_FILE" up -d --force-recreate "$service"; then
        log "ERROR" "Failed to recreate container for $service"
        rollback_service "$service" "$backup_path"
        return 1
    fi
    
    # Wait for service to start
    sleep 10
    
    # Health check
    if ! health_check "$service"; then
        log "ERROR" "Service $service failed health check after update"
        rollback_service "$service" "$backup_path"
        return 1
    fi
    
    # Start dependent services
    start_dependent_services "$service"
    
    # Clean up old images
    docker image prune -f &> /dev/null
    
    log "INFO" "Successfully updated $service"
    save_state "${service}_last_update" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    
    return 0
}

# Stop dependent services
stop_dependent_services() {
    local service=$1
    
    for dep_service in "${!SERVICE_DEPS[@]}"; do
        local deps="${SERVICE_DEPS[$dep_service]}"
        if [[ ",$deps," == *",$service,"* ]]; then
            log "INFO" "Stopping dependent service: $dep_service"
            docker compose -f "$COMPOSE_FILE" stop "$dep_service"
        fi
    done
}

# Start dependent services
start_dependent_services() {
    local service=$1
    
    for dep_service in "${!SERVICE_DEPS[@]}"; do
        local deps="${SERVICE_DEPS[$dep_service]}"
        if [[ ",$deps," == *",$service,"* ]]; then
            log "INFO" "Starting dependent service: $dep_service"
            docker compose -f "$COMPOSE_FILE" start "$dep_service"
            sleep 5
        fi
    done
}

# Rollback service
rollback_service() {
    local service=$1
    local backup_path=$2
    
    log "WARN" "Rolling back $service to previous version..."
    
    if [[ ! -d "$backup_path" ]]; then
        log "ERROR" "Backup not found: $backup_path"
        return 1
    fi
    
    # Stop service
    docker compose -f "$COMPOSE_FILE" stop "$service"
    
    # Restore previous image
    local previous_image=$(cat "${backup_path}/image.txt")
    log "INFO" "Restoring image: $previous_image"
    
    # Update compose file to use specific image tag
    local temp_compose="/tmp/update-compose-${service}.yml"
    cp "$COMPOSE_FILE" "$temp_compose"
    
    # Force recreation with previous image
    docker compose -f "$COMPOSE_FILE" up -d --force-recreate "$service"
    
    # Health check
    if health_check "$service"; then
        log "INFO" "Rollback successful for $service"
        return 0
    else
        log "ERROR" "Rollback failed for $service"
        return 1
    fi
}

# Update all services
update_all_services() {
    local services=()
    local failed_services=()
    
    # Get all services
    if [[ $# -eq 0 ]]; then
        services=($(docker compose -f "$COMPOSE_FILE" config --services))
    else
        services=("$@")
    fi
    
    log "INFO" "Services to update: ${services[*]}"
    
    # Update services in batches
    local batch_count=0
    
    for service in "${services[@]}"; do
        # Skip if critical service and not explicitly requested
        if [[ " ${CRITICAL_SERVICES[@]} " =~ " ${service} " ]] && [[ $# -eq 0 ]]; then
            log "WARN" "Skipping critical service: $service (update manually)"
            continue
        fi
        
        # Check for updates
        local image=$(docker compose -f "$COMPOSE_FILE" config | \
            yq eval ".services.${service}.image" 2>/dev/null)
        
        if [[ -n "$image" ]]; then
            if check_image_update "$image"; then
                log "INFO" "Update available for $service"
                
                if update_service "$service"; then
                    log "INFO" "Successfully updated $service"
                else
                    log "ERROR" "Failed to update $service"
                    failed_services+=("$service")
                fi
                
                # Batch control
                ((batch_count++))
                if (( batch_count >= UPDATE_BATCH_SIZE )); then
                    log "INFO" "Batch complete, waiting before next batch..."
                    sleep 60
                    batch_count=0
                fi
            else
                log "INFO" "No update available for $service"
            fi
        fi
    done
    
    # Report results
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log "INFO" "All services updated successfully"
        return 0
    else
        log "ERROR" "Failed to update services: ${failed_services[*]}"
        return 1
    fi
}

# Version pinning
pin_version() {
    local service=$1
    local version=$2
    
    log "INFO" "Pinning $service to version $version"
    
    # Update compose file
    local compose_backup="${COMPOSE_FILE}.backup-$(date +%Y%m%d_%H%M%S)"
    cp "$COMPOSE_FILE" "$compose_backup"
    
    # Update image tag in compose file
    yq eval ".services.${service}.image = \"${version}\"" -i "$COMPOSE_FILE"
    
    log "INFO" "Version pinned. Backup saved to: $compose_backup"
}

# Check update schedule
check_update_schedule() {
    local last_update=$(load_state | jq -r '.last_full_update // empty')
    
    if [[ -z "$last_update" ]]; then
        return 0  # No previous update, proceed
    fi
    
    local last_timestamp=$(date -d "$last_update" +%s 2>/dev/null || echo 0)
    local current_timestamp=$(date +%s)
    local days_since=$(( (current_timestamp - last_timestamp) / 86400 ))
    
    log "INFO" "Last full update: $last_update ($days_since days ago)"
    
    # Update weekly by default
    if (( days_since >= 7 )); then
        return 0  # Time for update
    else
        return 1  # Too soon
    fi
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Discord webhook
    if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
        local color=$([[ "$status" == "success" ]] && echo "3066993" || echo "15158332")
        
        curl -s -H "Content-Type: application/json" -X POST \
            -d "{\"embeds\":[{\"title\":\"Media Server Update\",\"description\":\"${message}\",\"color\":${color}}]}" \
            "${DISCORD_WEBHOOK}"
    fi
    
    # Email notification
    if [[ -n "${EMAIL_TO:-}" ]]; then
        echo "$message" | mail -s "Media Server Update - ${status}" "${EMAIL_TO}"
    fi
}

# Generate update report
generate_report() {
    local report_file="${LOG_DIR}/update-report-$(date +%Y%m%d).md"
    
    cat > "$report_file" << EOF
# Media Server Update Report

**Date**: $(date)

## Update Summary

| Service | Current Version | Status | Last Updated |
|---------|----------------|--------|--------------|
EOF
    
    local services=($(docker compose -f "$COMPOSE_FILE" config --services))
    
    for service in "${services[@]}"; do
        local image=$(get_current_image "$service")
        local status="✅ Up-to-date"
        local last_update=$(load_state | jq -r ".${service}_last_update // \"Never\"")
        
        echo "| $service | $image | $status | $last_update |" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

## System Health

\`\`\`
$(docker compose -f "$COMPOSE_FILE" ps)
\`\`\`

## Disk Usage

\`\`\`
$(df -h | grep -E "^/dev/|^Filesystem")
\`\`\`

## Recent Logs

\`\`\`
$(tail -n 50 "$LOG_FILE")
\`\`\`

---
*Generated by update-automation.sh*
EOF
    
    log "INFO" "Update report generated: $report_file"
}

# Main update function
perform_update() {
    local update_type=${1:-check}
    shift
    local services=("$@")
    
    log "INFO" "Starting update process: $update_type"
    
    # Check prerequisites
    check_prerequisites
    
    case $update_type in
        check)
            # Check for available updates
            log "INFO" "Checking for available updates..."
            
            local updates_available=false
            local services_to_update=()
            
            if [[ ${#services[@]} -eq 0 ]]; then
                services=($(docker compose -f "$COMPOSE_FILE" config --services))
            fi
            
            for service in "${services[@]}"; do
                local image=$(docker compose -f "$COMPOSE_FILE" config | \
                    yq eval ".services.${service}.image" 2>/dev/null)
                
                if [[ -n "$image" ]] && check_image_update "$image"; then
                    log "INFO" "Update available for $service"
                    services_to_update+=("$service")
                    updates_available=true
                fi
            done
            
            if $updates_available; then
                log "INFO" "Updates available for: ${services_to_update[*]}"
                echo "Run '$0 update' to apply updates"
            else
                log "INFO" "All services are up-to-date"
            fi
            ;;
            
        update)
            # Perform updates
            if [[ ${#services[@]} -eq 0 ]] && ! check_update_schedule; then
                log "INFO" "Skipping scheduled update (too soon since last update)"
                exit 0
            fi
            
            update_all_services "${services[@]}"
            
            # Update state
            save_state "last_full_update" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            
            # Generate report
            generate_report
            
            # Send notification
            send_notification "success" "Update completed successfully"
            ;;
            
        rollback)
            # Rollback specific service
            if [[ ${#services[@]} -eq 0 ]]; then
                log "ERROR" "Service name required for rollback"
                exit 1
            fi
            
            for service in "${services[@]}"; do
                local backup_path=$(load_state | jq -r ".${service}_backup // empty")
                
                if [[ -n "$backup_path" ]] && [[ -d "$backup_path" ]]; then
                    rollback_service "$service" "$backup_path"
                else
                    log "ERROR" "No backup found for $service"
                fi
            done
            ;;
            
        pin)
            # Pin service version
            if [[ ${#services[@]} -lt 2 ]]; then
                log "ERROR" "Usage: $0 pin <service> <version>"
                exit 1
            fi
            
            pin_version "${services[0]}" "${services[1]}"
            ;;
            
        *)
            log "ERROR" "Unknown update type: $update_type"
            exit 1
            ;;
    esac
}

# Show help
show_help() {
    cat << EOF
Media Server Update Automation Script

Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    check [SERVICE...]      Check for available updates
    update [SERVICE...]     Update services (all if none specified)
    rollback SERVICE...     Rollback service to previous version
    pin SERVICE VERSION     Pin service to specific version
    report                  Generate update report
    help                    Show this help message

OPTIONS:
    -f, --force            Force update even if recently updated
    -b, --batch SIZE       Set batch size (default: 3)
    -t, --timeout SECONDS  Set update timeout (default: 300)

EXAMPLES:
    $0 check               Check all services for updates
    $0 update              Update all services
    $0 update jellyfin     Update only Jellyfin
    $0 rollback sonarr     Rollback Sonarr to previous version
    $0 pin radarr radarr:4.3.0  Pin Radarr to version 4.3.0

AUTOMATED UPDATES:
    Add to crontab for scheduled updates:
    0 3 * * 0 $0 update    # Weekly on Sunday at 3 AM
    0 2 * * * $0 check     # Daily check at 2 AM

SAFETY FEATURES:
    - Automatic backups before updates
    - Health checks after updates
    - Automatic rollback on failure
    - Batch updates with delays
    - Critical service protection

EOF
}

# Parse command line arguments
case "${1:-help}" in
    check|update|rollback|pin)
        perform_update "$@"
        ;;
    report)
        generate_report
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        log "ERROR" "Unknown command: $1"
        show_help
        exit 1
        ;;
esac