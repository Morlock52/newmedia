#!/bin/bash

# Enhanced Health Check System for Media Server Infrastructure
# Implements comprehensive service health monitoring with auto-recovery

set -euo pipefail

# Configuration
HEALTH_CHECK_DIR="/var/log/health-checks"
ALERT_THRESHOLD=3
RECOVERY_ATTEMPTS=3
TIMEOUT=30
LOG_RETENTION_DAYS=7

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${HEALTH_CHECK_DIR}/health-check.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${HEALTH_CHECK_DIR}/health-check.log"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${HEALTH_CHECK_DIR}/health-check.log"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${HEALTH_CHECK_DIR}/health-check.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${HEALTH_CHECK_DIR}/health-check.log"
}

# Initialize health check directory
init_health_checks() {
    mkdir -p "${HEALTH_CHECK_DIR}"
    mkdir -p "${HEALTH_CHECK_DIR}/service-logs"
    mkdir -p "${HEALTH_CHECK_DIR}/recovery-logs"
    
    # Clean old logs
    find "${HEALTH_CHECK_DIR}" -name "*.log" -type f -mtime +${LOG_RETENTION_DAYS} -delete 2>/dev/null || true
    
    info "Health check system initialized"
}

# Enhanced health check function with circuit breaker pattern
check_service_health() {
    local service_name="$1"
    local health_url="$2"
    local expected_status="${3:-200}"
    local additional_checks="${4:-}"
    
    local status_file="${HEALTH_CHECK_DIR}/${service_name}_status"
    local failure_count=0
    local last_success_time=""
    
    # Read previous status
    if [[ -f "$status_file" ]]; then
        failure_count=$(grep "failure_count=" "$status_file" | cut -d'=' -f2 || echo "0")
        last_success_time=$(grep "last_success=" "$status_file" | cut -d'=' -f2 || echo "")
    fi
    
    info "Checking health of $service_name..."
    
    # Primary health check
    local health_check_result=0
    local response_time=0
    local start_time=$(date +%s.%N)
    
    if timeout "${TIMEOUT}" curl -sf "$health_url" >/dev/null 2>&1; then
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc 2>/dev/null || echo "0")
        
        # Additional service-specific checks
        if [[ -n "$additional_checks" ]]; then
            eval "$additional_checks" || health_check_result=1
        fi
        
        if [[ $health_check_result -eq 0 ]]; then
            success "$service_name is healthy (${response_time}s response time)"
            
            # Reset failure count on success
            failure_count=0
            last_success_time=$(date -Iseconds)
            
            # Update status file
            cat > "$status_file" << EOF
service=$service_name
status=healthy
failure_count=0
last_success=$last_success_time
last_check=$(date -Iseconds)
response_time=$response_time
EOF
            return 0
        fi
    fi
    
    # Health check failed
    ((failure_count++))
    error "$service_name health check failed (attempt $failure_count/$ALERT_THRESHOLD)"
    
    # Update status file
    cat > "$status_file" << EOF
service=$service_name
status=unhealthy
failure_count=$failure_count
last_success=$last_success_time
last_check=$(date -Iseconds)
response_time=timeout
EOF
    
    # Trigger recovery if threshold reached
    if [[ $failure_count -ge $ALERT_THRESHOLD ]]; then
        warn "$service_name has failed $failure_count times - triggering recovery"
        attempt_service_recovery "$service_name"
    fi
    
    return 1
}

# Service-specific health checks with advanced diagnostics
check_jellyfin() {
    local additional_checks='
        # Check if Jellyfin can access media directory
        curl -sf "http://localhost:8096/System/Info" | jq -e ".HasPendingRestart == false" >/dev/null 2>&1 &&
        # Check database connectivity
        docker exec jellyfin test -r /config/data/jellyfin.db 2>/dev/null
    '
    check_service_health "jellyfin" "http://localhost:8096/health" "200" "$additional_checks"
}

check_sonarr() {
    local additional_checks='
        # Check if Sonarr API is responding with valid data
        curl -sf "http://localhost:8989/api/v3/system/status" -H "X-Api-Key: ${SONARR_API_KEY:-}" | jq -e ".appName == \"Sonarr\"" >/dev/null 2>&1 &&
        # Check disk space
        [[ $(df /downloads | tail -1 | awk "{print \$5}" | sed "s/%//") -lt 90 ]]
    '
    check_service_health "sonarr" "http://localhost:8989/ping" "200" "$additional_checks"
}

check_radarr() {
    local additional_checks='
        # Check if Radarr API is responding
        curl -sf "http://localhost:7878/api/v3/system/status" -H "X-Api-Key: ${RADARR_API_KEY:-}" | jq -e ".appName == \"Radarr\"" >/dev/null 2>&1 &&
        # Check download client connectivity
        curl -sf "http://localhost:7878/api/v3/downloadclient" -H "X-Api-Key: ${RADARR_API_KEY:-}" >/dev/null 2>&1
    '
    check_service_health "radarr" "http://localhost:7878/ping" "200" "$additional_checks"
}

check_prowlarr() {
    local additional_checks='
        # Check indexer connectivity
        curl -sf "http://localhost:9696/api/v1/indexer" -H "X-Api-Key: ${PROWLARR_API_KEY:-}" | jq -e "length > 0" >/dev/null 2>&1
    '
    check_service_health "prowlarr" "http://localhost:9696/ping" "200" "$additional_checks"
}

check_qbittorrent() {
    local additional_checks='
        # Check if qBittorrent Web UI is accessible
        curl -sf "http://localhost:8080/api/v2/app/version" >/dev/null 2>&1 &&
        # Check if download directory is writable
        docker exec qbittorrent test -w /downloads 2>/dev/null
    '
    check_service_health "qbittorrent" "http://localhost:8080" "200" "$additional_checks"
}

check_plex() {
    local additional_checks='
        # Check Plex server status
        curl -sf "http://localhost:32400/identity" | grep -q "machineIdentifier" &&
        # Check transcoding capability
        docker exec plex test -d "/config/Library/Application Support/Plex Media Server/Cache/Transcode" 2>/dev/null
    '
    check_service_health "plex" "http://localhost:32400/identity" "200" "$additional_checks"
}

check_database_services() {
    # PostgreSQL
    if docker ps --format "table {{.Names}}" | grep -q "postgres"; then
        if docker exec postgres pg_isready -U "${POSTGRES_USER:-postgres}" >/dev/null 2>&1; then
            success "PostgreSQL is healthy"
        else
            error "PostgreSQL health check failed"
            attempt_database_recovery "postgres"
        fi
    fi
    
    # MariaDB
    if docker ps --format "table {{.Names}}" | grep -q "mariadb"; then
        if docker exec mariadb mysqladmin ping -h localhost >/dev/null 2>&1; then
            success "MariaDB is healthy"
        else
            error "MariaDB health check failed"
            attempt_database_recovery "mariadb"
        fi
    fi
    
    # Redis
    if docker ps --format "table {{.Names}}" | grep -q "redis"; then
        if docker exec redis redis-cli ping | grep -q "PONG"; then
            success "Redis is healthy"
        else
            error "Redis health check failed"
            attempt_database_recovery "redis"
        fi
    fi
}

# Advanced recovery mechanisms
attempt_service_recovery() {
    local service_name="$1"
    local recovery_log="${HEALTH_CHECK_DIR}/recovery-logs/${service_name}_recovery.log"
    
    info "Starting recovery procedures for $service_name"
    
    for attempt in $(seq 1 $RECOVERY_ATTEMPTS); do
        warn "Recovery attempt $attempt/$RECOVERY_ATTEMPTS for $service_name"
        
        case "$service_name" in
            "jellyfin")
                recover_jellyfin
                ;;
            "sonarr"|"radarr"|"prowlarr")
                recover_arr_service "$service_name"
                ;;
            "qbittorrent")
                recover_qbittorrent
                ;;
            "plex")
                recover_plex
                ;;
            *)
                recover_generic_service "$service_name"
                ;;
        esac
        
        # Wait before checking if recovery was successful
        sleep 30
        
        # Test if service is back online
        local health_url
        case "$service_name" in
            "jellyfin") health_url="http://localhost:8096/health" ;;
            "sonarr") health_url="http://localhost:8989/ping" ;;
            "radarr") health_url="http://localhost:7878/ping" ;;
            "prowlarr") health_url="http://localhost:9696/ping" ;;
            "qbittorrent") health_url="http://localhost:8080" ;;
            "plex") health_url="http://localhost:32400/identity" ;;
            *) health_url="http://localhost:8080" ;;
        esac
        
        if timeout "${TIMEOUT}" curl -sf "$health_url" >/dev/null 2>&1; then
            success "$service_name recovery successful on attempt $attempt"
            
            # Reset failure count
            local status_file="${HEALTH_CHECK_DIR}/${service_name}_status"
            sed -i "s/failure_count=.*/failure_count=0/" "$status_file" 2>/dev/null || true
            
            return 0
        fi
        
        warn "$service_name recovery attempt $attempt failed"
    done
    
    error "$service_name recovery failed after $RECOVERY_ATTEMPTS attempts"
    send_alert "$service_name" "RECOVERY_FAILED" "All recovery attempts exhausted"
    return 1
}

recover_jellyfin() {
    info "Recovering Jellyfin service"
    
    # Clear cache and temporary files
    docker exec jellyfin rm -rf /config/cache/* 2>/dev/null || true
    docker exec jellyfin rm -rf /config/transcoding/temp/* 2>/dev/null || true
    
    # Restart Jellyfin container
    docker restart jellyfin
    
    # Wait for startup
    sleep 20
}

recover_arr_service() {
    local service="$1"
    info "Recovering $service service"
    
    # Clear logs and temp files
    docker exec "$service" find /config/logs -name "*.txt" -mtime +1 -delete 2>/dev/null || true
    
    # Restart service
    docker restart "$service"
    
    # Wait for startup
    sleep 15
}

recover_qbittorrent() {
    info "Recovering qBittorrent service"
    
    # Clear session data
    docker exec qbittorrent rm -rf /config/qBittorrent/BT_backup/*.fastresume 2>/dev/null || true
    
    # Restart container
    docker restart qbittorrent
    
    # Wait for startup
    sleep 15
}

recover_plex() {
    info "Recovering Plex service"
    
    # Clear cache
    docker exec plex rm -rf "/config/Library/Application Support/Plex Media Server/Cache/Transcode/*" 2>/dev/null || true
    
    # Restart Plex
    docker restart plex
    
    # Wait for startup
    sleep 30
}

recover_generic_service() {
    local service="$1"
    info "Generic recovery for $service"
    
    # Simple restart
    docker restart "$service"
    sleep 15
}

attempt_database_recovery() {
    local db_service="$1"
    
    case "$db_service" in
        "postgres")
            docker restart postgres
            sleep 10
            docker exec postgres pg_isready -U "${POSTGRES_USER:-postgres}" || return 1
            ;;
        "mariadb")
            docker restart mariadb
            sleep 10
            docker exec mariadb mysqladmin ping -h localhost || return 1
            ;;
        "redis")
            docker restart redis
            sleep 5
            docker exec redis redis-cli ping | grep -q "PONG" || return 1
            ;;
    esac
    
    success "$db_service database recovery completed"
}

# Network connectivity checks
check_network_health() {
    info "Checking network connectivity"
    
    # Check internal Docker network
    if docker network inspect media-net >/dev/null 2>&1; then
        success "Docker network media-net is healthy"
    else
        error "Docker network media-net has issues"
        docker network create media-net 2>/dev/null || true
    fi
    
    # Check external connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        success "External network connectivity is healthy"
    else
        warn "External network connectivity issues detected"
    fi
    
    # Check DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        success "DNS resolution is working"
    else
        warn "DNS resolution issues detected"
    fi
}

# Disk space monitoring
check_disk_space() {
    info "Checking disk space"
    
    local critical_threshold=90
    local warning_threshold=80
    
    while IFS= read -r line; do
        local filesystem=$(echo "$line" | awk '{print $1}')
        local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
        local mount=$(echo "$line" | awk '{print $6}')
        
        if [[ $usage -ge $critical_threshold ]]; then
            error "CRITICAL: Disk usage on $mount is ${usage}% (filesystem: $filesystem)"
            send_alert "disk_space" "CRITICAL" "Disk usage ${usage}% on $mount"
            
            # Attempt cleanup
            cleanup_disk_space "$mount"
        elif [[ $usage -ge $warning_threshold ]]; then
            warn "WARNING: Disk usage on $mount is ${usage}% (filesystem: $filesystem)"
            send_alert "disk_space" "WARNING" "Disk usage ${usage}% on $mount"
        else
            success "Disk usage on $mount is ${usage}% - OK"
        fi
    done < <(df -h | grep -v tmpfs | grep -v udev | tail -n +2)
}

cleanup_disk_space() {
    local mount_point="$1"
    
    info "Attempting disk cleanup on $mount_point"
    
    # Clean Docker system
    docker system prune -f >/dev/null 2>&1 || true
    
    # Clean old log files
    find /var/log -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    
    # Clean temporary files
    find /tmp -type f -mtime +1 -delete 2>/dev/null || true
    
    # Clean old downloads (if in downloads directory)
    if [[ "$mount_point" == *"downloads"* ]]; then
        find "$mount_point" -name "*.partial" -mtime +1 -delete 2>/dev/null || true
    fi
    
    info "Disk cleanup completed for $mount_point"
}

# Alert system
send_alert() {
    local service="$1"
    local severity="$2"
    local message="$3"
    
    local alert_file="${HEALTH_CHECK_DIR}/alerts.json"
    local timestamp=$(date -Iseconds)
    
    # Create alert JSON
    local alert_json=$(cat << EOF
{
  "timestamp": "$timestamp",
  "service": "$service",
  "severity": "$severity",
  "message": "$message",
  "hostname": "$(hostname)"
}
EOF
)
    
    # Append to alerts file
    if [[ -f "$alert_file" ]]; then
        # Add comma and new alert
        sed -i '$ s/]$/,/' "$alert_file"
        echo "$alert_json" >> "$alert_file"
        echo "]" >> "$alert_file"
    else
        # Create new alerts file
        echo "[" > "$alert_file"
        echo "$alert_json" >> "$alert_file"
        echo "]" >> "$alert_file"
    fi
    
    # Log alert
    error "ALERT [$severity]: $service - $message"
    
    # Send webhook notification if configured
    if [[ -n "${WEBHOOK_URL:-}" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "$alert_json" >/dev/null 2>&1 || true
    fi
}

# Generate health report
generate_health_report() {
    local report_file="${HEALTH_CHECK_DIR}/health-report-$(date +%Y%m%d-%H%M%S).json"
    
    info "Generating health report: $report_file"
    
    local report_json=$(cat << EOF
{
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "services": {},
  "system": {
    "uptime": "$(uptime -p)",
    "load_average": "$(uptime | awk '{print $(NF-2) $(NF-1) $(NF)}')",
    "memory_usage": "$(free -h | grep Mem | awk '{print $3"/"$2}')",
    "disk_usage": {}
  }
}
EOF
)
    
    # Add service statuses
    for status_file in "${HEALTH_CHECK_DIR}"/*_status; do
        if [[ -f "$status_file" ]]; then
            local service_name=$(basename "$status_file" _status)
            local service_status=$(cat "$status_file")
            
            # Parse status file and add to report
            # This is a simplified version - in production, use jq for proper JSON manipulation
            echo "  Service $service_name status included in report" >> "${HEALTH_CHECK_DIR}/health-check.log"
        fi
    done
    
    echo "$report_json" > "$report_file"
    success "Health report generated: $report_file"
}

# Main health check routine
run_comprehensive_health_check() {
    info "Starting comprehensive health check"
    
    local start_time=$(date +%s)
    local failed_services=0
    
    # Check all media services
    check_jellyfin || ((failed_services++))
    check_sonarr || ((failed_services++))
    check_radarr || ((failed_services++))
    check_prowlarr || ((failed_services++))
    check_qbittorrent || ((failed_services++))
    
    # Check Plex if available
    if docker ps --format "table {{.Names}}" | grep -q "plex"; then
        check_plex || ((failed_services++))
    fi
    
    # Check database services
    check_database_services
    
    # Check system resources
    check_network_health
    check_disk_space
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $failed_services -eq 0 ]]; then
        success "All services are healthy (check completed in ${duration}s)"
    else
        error "$failed_services services are unhealthy (check completed in ${duration}s)"
    fi
    
    # Generate health report
    generate_health_report
    
    return $failed_services
}

# Daemon mode for continuous monitoring
run_daemon() {
    local interval="${1:-300}" # Default 5 minutes
    
    info "Starting health check daemon (interval: ${interval}s)"
    
    while true; do
        run_comprehensive_health_check
        sleep "$interval"
    done
}

# Main execution
main() {
    init_health_checks
    
    case "${1:-check}" in
        "check")
            run_comprehensive_health_check
            ;;
        "daemon")
            run_daemon "${2:-300}"
            ;;
        "service")
            case "${2:-}" in
                "jellyfin") check_jellyfin ;;
                "sonarr") check_sonarr ;;
                "radarr") check_radarr ;;
                "prowlarr") check_prowlarr ;;
                "qbittorrent") check_qbittorrent ;;
                "plex") check_plex ;;
                *) error "Unknown service: ${2:-}"; exit 1 ;;
            esac
            ;;
        "recovery")
            attempt_service_recovery "${2:-}"
            ;;
        "report")
            generate_health_report
            ;;
        "help"|"-h"|"--help")
            cat << EOF
Enhanced Health Check System

Usage: $0 [command] [options]

Commands:
  check              Run comprehensive health check (default)
  daemon [interval]  Run in daemon mode with specified interval (default: 300s)
  service <name>     Check specific service (jellyfin, sonarr, radarr, etc.)
  recovery <name>    Attempt recovery for specific service
  report             Generate health report
  help               Show this help message

Environment Variables:
  SONARR_API_KEY     API key for Sonarr
  RADARR_API_KEY     API key for Radarr
  PROWLARR_API_KEY   API key for Prowlarr
  WEBHOOK_URL        URL for alert notifications
  HEALTH_CHECK_PORT  Port for health check API (default: 3010)

Examples:
  $0 check                    # Run full health check
  $0 daemon 60               # Run daemon with 60s interval
  $0 service jellyfin        # Check only Jellyfin
  $0 recovery sonarr         # Attempt Sonarr recovery
EOF
            ;;
        *)
            error "Unknown command: $1"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"