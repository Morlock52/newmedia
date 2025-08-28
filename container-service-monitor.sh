#!/bin/bash
# Container Service Monitor
# Monitors all services running inside the single container
# Validates s6-overlay service management and process health

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CONTAINER_NAME=""
LOG_FILE="/tmp/container-health-$(date +%Y%m%d-%H%M%S).log"
TIMEOUT=30
VERBOSE=false

# Service definitions with s6 service names
declare -A SERVICES=(
    # Media Servers
    ["jellyfin"]="8096:/health"
    ["plex"]="32400:/identity"
    ["emby"]="8097:/System/Info/Public"
    
    # *ARR Stack
    ["sonarr"]="8989:/ping"
    ["radarr"]="7878:/ping"
    ["lidarr"]="8686:/ping"
    ["readarr"]="8787:/ping"
    ["prowlarr"]="9696:/ping"
    ["bazarr"]="6767:/system/status"
    
    # Download Clients
    ["qbittorrent"]="8080:/api/v2/app/version"
    ["transmission"]="9091:/transmission/rpc"
    ["sabnzbd"]="8081:/api"
    ["nzbget"]="6789:"
    
    # Request Management
    ["overseerr"]="5055:/api/v1/status"
    ["jellyseerr"]="5056:/api/v1/status"
    ["ombi"]="3579:/api/v1/Status"
    
    # Dashboards
    ["homarr"]="7575:"
    ["homepage"]="3003:"
    ["tautulli"]="8181:/api/v2"
    ["organizr"]="8083:"
    
    # Content Libraries
    ["calibre-web"]="8083:"
    ["audiobookshelf"]="13378:/healthcheck"
    ["navidrome"]="4533:"
    ["photoprism"]="2342:/api/v1/status"
    
    # Utilities
    ["vaultwarden"]="8085:/alive"
    ["pihole"]="8053:/admin/api.php"
    ["syncthing"]="8384:/rest/system/status"
    ["nextcloud"]="8084:"
    
    # Monitoring
    ["prometheus"]="9090:/-/healthy"
    ["grafana"]="3000:/api/health"
    ["uptime-kuma"]="3001:"
    
    # Databases
    ["postgres"]="5432:"
    ["redis"]="6379:"
    ["mariadb"]="3306:"
    
    # Management
    ["portainer"]="9000:/api/status"
)

# Process name mappings for s6 services that differ
declare -A PROCESS_NAMES=(
    ["jellyfin"]="jellyfin"
    ["plex"]="plex"
    ["emby"]="emby"
    ["qbittorrent"]="qbittorrent-nox"
    ["transmission"]="transmission-daemon"
    ["sabnzbd"]="sabnzbd"
    ["nzbget"]="nzbget"
    ["postgres"]="postgres"
    ["redis"]="redis-server"
    ["mariadb"]="mysqld"
    ["prometheus"]="prometheus"
    ["grafana"]="grafana-server"
)

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    
    case "$level" in
        "ERROR") echo -e "${RED}❌ $message${NC}" >&2 ;;
        "WARN") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "INFO") echo -e "${GREEN}ℹ️  $message${NC}" ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}🔍 $message${NC}" ;;
    esac
}

# Usage information
usage() {
    cat << EOF
Container Service Monitor - Health check for single-container media server

Usage: $0 [OPTIONS] [CONTAINER_NAME]

OPTIONS:
    -c, --container NAME    Container name to monitor (auto-detect if not provided)
    -v, --verbose          Enable verbose output
    -t, --timeout SEC      Timeout for health checks (default: 30)
    -l, --log FILE         Log file path (default: /tmp/container-health-TIMESTAMP.log)
    -h, --help             Show this help

EXAMPLES:
    $0                                    # Auto-detect container
    $0 -c ultimate-media-server-2025      # Monitor specific container
    $0 -v -t 60                          # Verbose mode with 60s timeout

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--container)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                CONTAINER_NAME="$1"
                shift
                ;;
        esac
    done
}

# Detect container automatically
detect_container() {
    if [[ -z "$CONTAINER_NAME" ]]; then
        log "INFO" "Auto-detecting media server container..."
        
        # Look for common container names
        local candidates=(
            "ultimate-media-server"
            "media-server"
            "newmedia"
            "ultimate-media-server-2025"
        )
        
        for candidate in "${candidates[@]}"; do
            if docker ps --format "{{.Names}}" | grep -q "^${candidate}"; then
                CONTAINER_NAME="$candidate"
                log "INFO" "Detected container: $CONTAINER_NAME"
                return 0
            fi
        done
        
        # If no specific name found, try to find any container with media server services
        local running_containers=$(docker ps --format "{{.Names}}")
        for container in $running_containers; do
            if docker exec "$container" netstat -tulpn 2>/dev/null | grep -q ":8096\|:32400\|:8989"; then
                CONTAINER_NAME="$container"
                log "INFO" "Detected media server container: $CONTAINER_NAME"
                return 0
            fi
        done
        
        log "ERROR" "No media server container detected. Please specify with -c option."
        return 1
    fi
    
    # Verify container exists and is running
    if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log "ERROR" "Container '$CONTAINER_NAME' not found or not running"
        return 1
    fi
    
    log "INFO" "Monitoring container: $CONTAINER_NAME"
    return 0
}

# Check if container is using s6-overlay
check_s6_overlay() {
    log "INFO" "Checking s6-overlay process supervision..."
    
    if docker exec "$CONTAINER_NAME" test -d /run/s6 2>/dev/null; then
        log "INFO" "s6-overlay detected"
        
        # List s6 services
        local s6_services
        s6_services=$(docker exec "$CONTAINER_NAME" find /run/s6/services -name "run" -executable 2>/dev/null | wc -l)
        log "INFO" "Found $s6_services s6-supervised services"
        
        # Check s6 service status
        if docker exec "$CONTAINER_NAME" command -v s6-svstat >/dev/null 2>&1; then
            local service_status
            service_status=$(docker exec "$CONTAINER_NAME" s6-svstat /run/s6/services/* 2>/dev/null | head -10)
            log "DEBUG" "s6 service status sample:\n$service_status"
        fi
        
        return 0
    else
        log "WARN" "s6-overlay not detected - using alternative process monitoring"
        return 1
    fi
}

# Check individual service with s6
check_s6_service() {
    local service_name="$1"
    
    # Check if s6 service exists
    if docker exec "$CONTAINER_NAME" test -d "/run/s6/services/$service_name" 2>/dev/null; then
        local status
        status=$(docker exec "$CONTAINER_NAME" s6-svstat "/run/s6/services/$service_name" 2>/dev/null || echo "unknown")
        
        if echo "$status" | grep -q "up"; then
            log "DEBUG" "$service_name s6 service: UP"
            return 0
        else
            log "WARN" "$service_name s6 service: $status"
            return 1
        fi
    else
        log "DEBUG" "$service_name: No s6 service found"
        return 1
    fi
}

# Check if process is running
check_process() {
    local service_name="$1"
    local process_name="${PROCESS_NAMES[$service_name]:-$service_name}"
    
    local pids
    pids=$(docker exec "$CONTAINER_NAME" pgrep -f "$process_name" 2>/dev/null || true)
    
    if [[ -n "$pids" ]]; then
        local pid_count
        pid_count=$(echo "$pids" | wc -l)
        log "DEBUG" "$service_name process ($process_name): $pid_count instance(s) running"
        return 0
    else
        log "WARN" "$service_name process ($process_name): Not running"
        return 1
    fi
}

# Check if port is listening
check_port() {
    local service_name="$1"
    local port="$2"
    
    if docker exec "$CONTAINER_NAME" netstat -tulpn 2>/dev/null | grep -q ":$port "; then
        log "DEBUG" "$service_name: Port $port is listening"
        return 0
    else
        log "WARN" "$service_name: Port $port is not listening"
        return 1
    fi
}

# Check HTTP health endpoint
check_health_endpoint() {
    local service_name="$1"
    local port="$2"
    local endpoint="$3"
    
    if [[ -z "$endpoint" ]]; then
        log "DEBUG" "$service_name: No health endpoint configured"
        return 0
    fi
    
    local url="http://localhost:$port$endpoint"
    local response
    
    response=$(docker exec "$CONTAINER_NAME" curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" "$url" 2>/dev/null || echo "000")
    
    case "$response" in
        200|201|202)
            log "DEBUG" "$service_name: Health endpoint OK ($response)"
            return 0
            ;;
        409)
            # Special case for Transmission RPC
            if [[ "$service_name" == "transmission" ]]; then
                log "DEBUG" "$service_name: Health endpoint OK (RPC 409 expected)"
                return 0
            fi
            ;;&
        *)
            log "WARN" "$service_name: Health endpoint failed (HTTP $response)"
            return 1
            ;;
    esac
}

# Get process resource usage
get_process_resources() {
    local service_name="$1"
    local process_name="${PROCESS_NAMES[$service_name]:-$service_name}"
    
    local stats
    stats=$(docker exec "$CONTAINER_NAME" ps -C "$process_name" -o %cpu,%mem,pid,command --no-headers 2>/dev/null | head -1)
    
    if [[ -n "$stats" ]]; then
        local cpu mem pid
        read -r cpu mem pid _ <<< "$stats"
        log "DEBUG" "$service_name resources: CPU=${cpu}% MEM=${mem}% PID=$pid"
        echo "$cpu,$mem,$pid"
    else
        echo "0,0,0"
    fi
}

# Check container-level resources
check_container_resources() {
    log "INFO" "Checking container resource usage..."
    
    # Docker stats
    local container_stats
    container_stats=$(docker stats "$CONTAINER_NAME" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | tail -n 1)
    
    if [[ -n "$container_stats" ]]; then
        log "INFO" "Container stats: $container_stats"
    fi
    
    # Internal process count
    local process_count
    process_count=$(docker exec "$CONTAINER_NAME" ps aux --no-headers | wc -l)
    log "INFO" "Total processes in container: $process_count"
    
    # Memory breakdown
    local memory_info
    memory_info=$(docker exec "$CONTAINER_NAME" cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable" | tr '\n' ' ')
    log "DEBUG" "Memory info: $memory_info"
}

# Main service health check
check_service_health() {
    local service_name="$1"
    local port_endpoint="$2"
    
    local port endpoint
    IFS=':' read -r port endpoint <<< "$port_endpoint"
    
    log "INFO" "Checking $service_name..."
    
    local checks_passed=0
    local total_checks=0
    
    # Check 1: s6 service (if available)
    ((total_checks++))
    if check_s6_service "$service_name"; then
        ((checks_passed++))
    fi
    
    # Check 2: Process running
    ((total_checks++))
    if check_process "$service_name"; then
        ((checks_passed++))
    fi
    
    # Check 3: Port listening
    ((total_checks++))
    if check_port "$service_name" "$port"; then
        ((checks_passed++))
    fi
    
    # Check 4: Health endpoint (if available)
    if [[ -n "$endpoint" ]]; then
        ((total_checks++))
        if check_health_endpoint "$service_name" "$port" "$endpoint"; then
            ((checks_passed++))
        fi
    fi
    
    # Get resource usage
    local resources
    resources=$(get_process_resources "$service_name")
    
    # Determine service health
    local health_percentage=$((checks_passed * 100 / total_checks))
    local status
    
    if [[ $health_percentage -eq 100 ]]; then
        status="✅ HEALTHY"
        log "INFO" "$service_name: $status ($checks_passed/$total_checks checks passed)"
    elif [[ $health_percentage -ge 75 ]]; then
        status="⚠️ DEGRADED"
        log "WARN" "$service_name: $status ($checks_passed/$total_checks checks passed)"
    else
        status="❌ UNHEALTHY"
        log "ERROR" "$service_name: $status ($checks_passed/$total_checks checks passed)"
    fi
    
    # Store results
    echo "$service_name,$health_percentage,$checks_passed,$total_checks,$resources,$status"
}

# Generate comprehensive report
generate_report() {
    local results_file="/tmp/service-results-$(date +%Y%m%d-%H%M%S).csv"
    
    log "INFO" "Generating comprehensive health report..."
    
    # CSV Header
    echo "Service,HealthPercentage,ChecksPassed,TotalChecks,CPU,Memory,PID,Status" > "$results_file"
    
    local total_services=0
    local healthy_services=0
    local degraded_services=0
    local unhealthy_services=0
    
    # Check all services
    for service_name in "${!SERVICES[@]}"; do
        ((total_services++))
        
        local result
        result=$(check_service_health "$service_name" "${SERVICES[$service_name]}")
        
        echo "$result" >> "$results_file"
        
        # Count by status
        if echo "$result" | grep -q "HEALTHY"; then
            ((healthy_services++))
        elif echo "$result" | grep -q "DEGRADED"; then
            ((degraded_services++))
        else
            ((unhealthy_services++))
        fi
    done
    
    # Summary
    log "INFO" "Health Check Summary:"
    log "INFO" "  Total Services: $total_services"
    log "INFO" "  Healthy: $healthy_services ($(( healthy_services * 100 / total_services ))%)"
    log "INFO" "  Degraded: $degraded_services ($(( degraded_services * 100 / total_services ))%)"
    log "INFO" "  Unhealthy: $unhealthy_services ($(( unhealthy_services * 100 / total_services ))%)"
    
    log "INFO" "Detailed results saved to: $results_file"
    
    # Overall health assessment
    local overall_health_percentage=$(( healthy_services * 100 / total_services ))
    
    if [[ $overall_health_percentage -ge 90 ]]; then
        log "INFO" "🎉 Overall Container Health: EXCELLENT ($overall_health_percentage%)"
        return 0
    elif [[ $overall_health_percentage -ge 75 ]]; then
        log "WARN" "⚠️ Overall Container Health: GOOD ($overall_health_percentage%)"
        return 0
    elif [[ $overall_health_percentage -ge 50 ]]; then
        log "WARN" "⚠️ Overall Container Health: DEGRADED ($overall_health_percentage%)"
        return 1
    else
        log "ERROR" "❌ Overall Container Health: CRITICAL ($overall_health_percentage%)"
        return 2
    fi
}

# Main execution
main() {
    echo "🐳 Container Service Monitor - Media Server Health Check"
    echo "================================================================"
    
    # Parse arguments
    parse_args "$@"
    
    # Detect container
    if ! detect_container; then
        exit 1
    fi
    
    # Initialize log
    log "INFO" "Starting health check for container: $CONTAINER_NAME"
    log "INFO" "Log file: $LOG_FILE"
    
    # Check s6-overlay
    check_s6_overlay
    
    # Check container resources
    check_container_resources
    
    # Run comprehensive health checks
    generate_report
    local exit_code=$?
    
    log "INFO" "Health check completed. Log file: $LOG_FILE"
    
    return $exit_code
}

# Trap for cleanup
trap 'log "INFO" "Health check interrupted"' INT TERM

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit $?
fi