#!/bin/bash

# ==================================================================
# AUTOMATIC RECOVERY SYSTEM
# Monitors services and automatically recovers failed containers
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
LOG_FILE="./logs/recovery-$(date +%Y%m%d_%H%M%S).log"
RECOVERY_LOG="./logs/recovery-actions.log"
COMPOSE_FILE="docker-compose.yml"
HEALTH_CHECK_INTERVAL=60  # seconds between health checks
MAX_RESTART_ATTEMPTS=3
RESTART_DELAY=30  # seconds to wait between restart attempts
ALERT_WEBHOOK=""

# Recovery strategies for each service type
declare -A RECOVERY_STRATEGIES=(
    ["database"]="restart,recreate,restore_backup"
    ["media_server"]="restart,clear_cache,recreate"
    ["arr_service"]="restart,reset_config,recreate"
    ["download_client"]="restart,clear_temp,recreate"
    ["monitoring"]="restart,recreate"
    ["proxy"]="restart,recreate"
    ["default"]="restart,recreate"
)

# Service categorization
declare -A SERVICE_TYPES=(
    ["postgres"]="database"
    ["mariadb"]="database"
    ["redis"]="database"
    ["jellyfin"]="media_server"
    ["plex"]="media_server"
    ["emby"]="media_server"
    ["sonarr"]="arr_service"
    ["radarr"]="arr_service"
    ["lidarr"]="arr_service"
    ["readarr"]="arr_service"
    ["prowlarr"]="arr_service"
    ["bazarr"]="arr_service"
    ["qbittorrent"]="download_client"
    ["transmission"]="download_client"
    ["sabnzbd"]="download_client"
    ["nzbget"]="download_client"
    ["grafana"]="monitoring"
    ["prometheus"]="monitoring"
    ["uptime-kuma"]="monitoring"
    ["nginx-proxy-manager"]="proxy"
    ["traefik"]="proxy"
)

# Critical services that require immediate attention
CRITICAL_SERVICES=("postgres" "redis" "jellyfin" "sonarr" "radarr" "prowlarr")

# Services that depend on others
declare -A SERVICE_DEPENDENCIES=(
    ["sonarr"]="postgres prowlarr"
    ["radarr"]="postgres prowlarr"
    ["lidarr"]="postgres prowlarr"
    ["jellyfin"]="postgres"
    ["grafana"]="postgres"
    ["overseerr"]="postgres"
    ["jellyseerr"]="postgres"
)

# Ensure directories exist
mkdir -p logs

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

recovery_log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$RECOVERY_LOG"
    log "RECOVERY: $1"
}

# Send alert notification
send_alert() {
    local message="$1"
    local severity="${2:-warning}"
    
    if [ -n "$ALERT_WEBHOOK" ]; then
        local color="16776960"  # Yellow
        case "$severity" in
            "critical") color="16711680" ;;  # Red
            "warning") color="16776960" ;;   # Yellow
            "success") color="65280" ;;      # Green
        esac
        
        curl -s -H "Content-Type: application/json" \
             -d "{\"embeds\": [{\"title\": \"Recovery System Alert\", \"description\": \"$message\", \"color\": $color, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\"}]}" \
             "$ALERT_WEBHOOK" || true
    fi
}

# Check if service is healthy
check_service_health() {
    local service="$1"
    
    # Check if container is running
    if ! docker ps --format "{{.Names}}" | grep -q "^$service$"; then
        return 1
    fi
    
    # Service-specific health checks
    case "$service" in
        "jellyfin")
            curl -sf http://localhost:8096/health > /dev/null 2>&1
            ;;
        "sonarr")
            curl -sf http://localhost:8989/ping > /dev/null 2>&1
            ;;
        "radarr")
            curl -sf http://localhost:7878/ping > /dev/null 2>&1
            ;;
        "prowlarr")
            curl -sf http://localhost:9696/ping > /dev/null 2>&1
            ;;
        "lidarr")
            curl -sf http://localhost:8686/ping > /dev/null 2>&1
            ;;
        "qbittorrent")
            curl -sf http://localhost:8080/api/v2/app/version > /dev/null 2>&1
            ;;
        "plex")
            curl -sf http://localhost:32400/identity > /dev/null 2>&1
            ;;
        "grafana")
            curl -sf http://localhost:3000/api/health > /dev/null 2>&1
            ;;
        "prometheus")
            curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1
            ;;
        "postgres")
            PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h localhost -U "${POSTGRES_USER:-postgres}" -d postgres -c 'SELECT 1;' > /dev/null 2>&1
            ;;
        "mariadb")
            mysql -h localhost -u root -p"${MYSQL_ROOT_PASSWORD:-root}" -e 'SELECT 1;' > /dev/null 2>&1
            ;;
        "redis")
            redis-cli -h localhost ping > /dev/null 2>&1
            ;;
        *)
            # Generic container health check
            docker inspect "$service" --format '{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy" || \
            docker inspect "$service" --format '{{.State.Status}}' 2>/dev/null | grep -q "running"
            ;;
    esac
}

# Get service type
get_service_type() {
    local service="$1"
    echo "${SERVICE_TYPES[$service]:-default}"
}

# Wait for dependencies
wait_for_dependencies() {
    local service="$1"
    local deps="${SERVICE_DEPENDENCIES[$service]:-}"
    
    if [ -n "$deps" ]; then
        log "Waiting for dependencies of $service: $deps"
        
        for dep in $deps; do
            local count=0
            while [ $count -lt 30 ]; do  # Wait up to 5 minutes
                if check_service_health "$dep"; then
                    log "Dependency $dep is healthy"
                    break
                fi
                
                if [ $count -eq 29 ]; then
                    warn "Dependency $dep is still not healthy after 5 minutes"
                    return 1
                fi
                
                sleep 10
                count=$((count + 1))
            done
        done
    fi
    
    return 0
}

# Simple restart recovery
restart_service() {
    local service="$1"
    
    recovery_log "Attempting restart recovery for $service"
    
    # Get compose command
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    # Restart the service
    $compose_cmd -f "$COMPOSE_FILE" restart "$service"
    
    # Wait for it to start
    sleep "$RESTART_DELAY"
    
    # Wait for dependencies if any
    wait_for_dependencies "$service"
    
    # Check if restart was successful
    if check_service_health "$service"; then
        success "Service $service restarted successfully"
        recovery_log "SUCCESS: Restart recovery worked for $service"
        return 0
    else
        warn "Service $service restart failed"
        recovery_log "FAILED: Restart recovery failed for $service"
        return 1
    fi
}

# Clear cache recovery (for media servers)
clear_cache_recovery() {
    local service="$1"
    
    recovery_log "Attempting cache clear recovery for $service"
    
    case "$service" in
        "jellyfin")
            # Clear jellyfin cache
            docker exec -it "$service" rm -rf /config/cache/* 2>/dev/null || true
            docker exec -it "$service" rm -rf /config/transcodes/* 2>/dev/null || true
            ;;
        "plex")
            # Clear plex cache
            docker exec -it "$service" rm -rf /config/Library/Application\ Support/Plex\ Media\ Server/Cache/* 2>/dev/null || true
            ;;
        "emby")
            # Clear emby cache
            docker exec -it "$service" rm -rf /config/cache/* 2>/dev/null || true
            ;;
    esac
    
    # Restart after cache clear
    return $(restart_service "$service")
}

# Reset configuration recovery (for arr services)
reset_config_recovery() {
    local service="$1"
    
    recovery_log "Attempting config reset recovery for $service"
    
    # Backup current config before reset
    local backup_dir="./backups/recovery-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    case "$service" in
        "sonarr"|"radarr"|"lidarr"|"readarr"|"prowlarr")
            # Backup config
            docker cp "$service:/config" "$backup_dir/$service-config" 2>/dev/null || true
            
            # Reset to minimal config (keep database)
            docker exec -it "$service" rm -f /config/config.xml 2>/dev/null || true
            ;;
    esac
    
    # Restart after config reset
    return $(restart_service "$service")
}

# Clear temporary files recovery (for download clients)
clear_temp_recovery() {
    local service="$1"
    
    recovery_log "Attempting temp clear recovery for $service"
    
    case "$service" in
        "qbittorrent")
            # Clear qBittorrent temp files
            docker exec -it "$service" rm -rf /config/qBittorrent/logs/* 2>/dev/null || true
            docker exec -it "$service" rm -rf /tmp/* 2>/dev/null || true
            ;;
        "transmission")
            # Clear transmission temp files
            docker exec -it "$service" rm -rf /tmp/* 2>/dev/null || true
            ;;
    esac
    
    # Restart after temp clear
    return $(restart_service "$service")
}

# Recreate container recovery
recreate_service() {
    local service="$1"
    
    recovery_log "Attempting recreate recovery for $service"
    
    # Get compose command
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    # Stop and remove container
    $compose_cmd -f "$COMPOSE_FILE" stop "$service" || true
    docker rm "$service" 2>/dev/null || true
    
    # Recreate the service
    $compose_cmd -f "$COMPOSE_FILE" up -d "$service"
    
    # Wait for it to start
    sleep $((RESTART_DELAY * 2))
    
    # Wait for dependencies if any
    wait_for_dependencies "$service"
    
    # Check if recreation was successful
    if check_service_health "$service"; then
        success "Service $service recreated successfully"
        recovery_log "SUCCESS: Recreate recovery worked for $service"
        return 0
    else
        warn "Service $service recreation failed"
        recovery_log "FAILED: Recreate recovery failed for $service"
        return 1
    fi
}

# Restore from backup recovery
restore_backup_recovery() {
    local service="$1"
    
    recovery_log "Attempting backup restore recovery for $service"
    
    # Find latest backup
    local backup_dir
    backup_dir=$(find ./backups -type d -name "*" | sort -r | head -1)
    
    if [ -n "$backup_dir" ] && [ -d "$backup_dir" ]; then
        log "Restoring $service from backup: $backup_dir"
        
        # Service-specific restore logic
        case "$service" in
            "postgres")
                # Restore PostgreSQL database
                if [ -f "$backup_dir/postgres-dump.sql" ]; then
                    docker exec -i postgres psql -U "${POSTGRES_USER:-postgres}" < "$backup_dir/postgres-dump.sql" || true
                fi
                ;;
            *)
                # Restore config files
                if [ -d "$backup_dir/$service-config" ]; then
                    docker cp "$backup_dir/$service-config" "$service:/config" || true
                fi
                ;;
        esac
        
        # Restart after restore
        return $(restart_service "$service")
    else
        warn "No backup found for $service"
        recovery_log "FAILED: No backup found for $service"
        return 1
    fi
}

# Execute recovery strategy
execute_recovery() {
    local service="$1"
    local strategy="$2"
    
    case "$strategy" in
        "restart")
            restart_service "$service"
            ;;
        "clear_cache")
            clear_cache_recovery "$service"
            ;;
        "reset_config")
            reset_config_recovery "$service"
            ;;
        "clear_temp")
            clear_temp_recovery "$service"
            ;;
        "recreate")
            recreate_service "$service"
            ;;
        "restore_backup")
            restore_backup_recovery "$service"
            ;;
        *)
            warn "Unknown recovery strategy: $strategy"
            return 1
            ;;
    esac
}

# Attempt service recovery
attempt_recovery() {
    local service="$1"
    local attempt="${2:-1}"
    
    recovery_log "Starting recovery attempt $attempt for $service"
    
    # Get service type and recovery strategies
    local service_type
    service_type=$(get_service_type "$service")
    local strategies="${RECOVERY_STRATEGIES[$service_type]}"
    
    # Try each strategy
    IFS=',' read -ra strategy_list <<< "$strategies"
    for strategy in "${strategy_list[@]}"; do
        log "Trying $strategy recovery for $service"
        
        if execute_recovery "$service" "$strategy"; then
            success "Recovery successful for $service using $strategy strategy"
            send_alert "Successfully recovered $service using $strategy strategy" "success"
            return 0
        else
            warn "$strategy recovery failed for $service"
        fi
        
        # Wait between strategies
        sleep 10
    done
    
    error "All recovery strategies failed for $service (attempt $attempt)"
    return 1
}

# Monitor and recover services
monitor_and_recover() {
    log "Starting continuous monitoring and recovery..."
    
    declare -A failure_counts
    
    while true; do
        # Check each service
        for service in "${!SERVICE_TYPES[@]}"; do
            if ! check_service_health "$service"; then
                # Increment failure count
                failure_counts[$service]=$((${failure_counts[$service]:-0} + 1))
                
                warn "Service $service is unhealthy (failure count: ${failure_counts[$service]})"
                
                # Check if it's a critical service
                local is_critical=false
                for critical in "${CRITICAL_SERVICES[@]}"; do
                    if [ "$critical" = "$service" ]; then
                        is_critical=true
                        break
                    fi
                done
                
                # Attempt recovery if failures exceed threshold or if critical
                if [ "${failure_counts[$service]}" -ge 2 ] || [ "$is_critical" = true ]; then
                    if [ "${failure_counts[$service]}" -le $MAX_RESTART_ATTEMPTS ]; then
                        warn "Attempting recovery for $service"
                        send_alert "Service $service is down, attempting recovery..." "warning"
                        
                        if attempt_recovery "$service" "${failure_counts[$service]}"; then
                            # Reset failure count on successful recovery
                            failure_counts[$service]=0
                        else
                            # Send critical alert if recovery fails
                            if [ "${failure_counts[$service]}" -ge $MAX_RESTART_ATTEMPTS ]; then
                                send_alert "CRITICAL: Failed to recover $service after $MAX_RESTART_ATTEMPTS attempts" "critical"
                                recovery_log "CRITICAL: Maximum recovery attempts reached for $service"
                            fi
                        fi
                    else
                        recovery_log "CRITICAL: Service $service has exceeded maximum recovery attempts"
                    fi
                fi
            else
                # Service is healthy, reset failure count
                if [ "${failure_counts[$service]:-0}" -gt 0 ]; then
                    log "Service $service is healthy again"
                    failure_counts[$service]=0
                fi
            fi
        done
        
        # Wait before next check
        sleep "$HEALTH_CHECK_INTERVAL"
    done
}

# Recovery status report
recovery_status() {
    echo -e "${PURPLE}Recovery System Status${NC}"
    echo "======================"
    echo "Log file: $LOG_FILE"
    echo "Recovery log: $RECOVERY_LOG"
    echo "Health check interval: ${HEALTH_CHECK_INTERVAL}s"
    echo "Max restart attempts: $MAX_RESTART_ATTEMPTS"
    echo ""
    
    echo "Service Health:"
    for service in "${!SERVICE_TYPES[@]}"; do
        if check_service_health "$service"; then
            echo -e "${GREEN}✓${NC} $service (${SERVICE_TYPES[$service]})"
        else
            echo -e "${RED}✗${NC} $service (${SERVICE_TYPES[$service]})"
        fi
    done
    
    echo ""
    echo "Recent Recovery Actions:"
    if [ -f "$RECOVERY_LOG" ]; then
        tail -10 "$RECOVERY_LOG"
    else
        echo "No recovery actions logged yet"
    fi
}

# Manual recovery trigger
manual_recovery() {
    local service="$1"
    
    if [ -z "$service" ]; then
        error "Please specify a service to recover"
        return 1
    fi
    
    log "Manual recovery triggered for $service"
    
    if attempt_recovery "$service" 1; then
        success "Manual recovery completed successfully for $service"
    else
        error "Manual recovery failed for $service"
        return 1
    fi
}

# Main function
main() {
    case "${1:-monitor}" in
        "monitor"|"start")
            monitor_and_recover
            ;;
        "status")
            recovery_status
            ;;
        "recover")
            manual_recovery "${2:-}"
            ;;
        "test")
            local service="${2:-jellyfin}"
            log "Testing recovery for $service"
            attempt_recovery "$service" 1
            ;;
        "--help"|"-h")
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  monitor, start      - Start continuous monitoring and recovery"
            echo "  status             - Show current recovery system status"
            echo "  recover [service]  - Manually trigger recovery for a service"
            echo "  test [service]     - Test recovery procedures for a service"
            echo ""
            echo "Environment Variables:"
            echo "  ALERT_WEBHOOK      - Discord/Slack webhook for notifications"
            echo "  HEALTH_CHECK_INTERVAL - Seconds between health checks (default: 60)"
            echo "  MAX_RESTART_ATTEMPTS  - Maximum recovery attempts (default: 3)"
            echo ""
            exit 0
            ;;
        *)
            error "Unknown command: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'log "Recovery system shutting down..."; exit 0' SIGINT SIGTERM

# Run main function
main "$@"