#!/bin/bash

# Media Server Health Monitor
# Monitors all service integrations and provides health reports
# Author: API Integration Specialist
# Date: $(date)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
HEALTH_CHECK_INTERVAL=300  # 5 minutes
LOG_RETENTION_DAYS=7
ALERT_THRESHOLD_FAILURES=3
STATUS_LOG_DIR="./logs/health-monitor"
STATUS_DASHBOARD="./logs/health-dashboard.html"

# Load API keys if available
if [ -f "./api-keys.env" ]; then
    source "./api-keys.env"
fi

# Service definitions
declare -A SERVICES=(
    ["jellyfin"]="http://localhost:8096"
    ["plex"]="http://localhost:32400"
    ["sonarr"]="http://localhost:8989"
    ["radarr"]="http://localhost:7878"
    ["lidarr"]="http://localhost:8686"
    ["prowlarr"]="http://localhost:9696"
    ["bazarr"]="http://localhost:6767"
    ["jellyseerr"]="http://localhost:5055"
    ["overseerr"]="http://localhost:5056"
    ["qbittorrent"]="http://localhost:8080"
    ["transmission"]="http://localhost:9091"
    ["sabnzbd"]="http://localhost:8081"
    ["grafana"]="http://localhost:3000"
    ["prometheus"]="http://localhost:9090"
    ["uptime-kuma"]="http://localhost:3001"
)

declare -A SERVICE_API_KEYS=(
    ["sonarr"]="$SONARR_API_KEY"
    ["radarr"]="$RADARR_API_KEY"
    ["lidarr"]="$LIDARR_API_KEY"
    ["prowlarr"]="$PROWLARR_API_KEY"
    ["bazarr"]="$BAZARR_API_KEY"
)

declare -A SERVICE_HEALTH_ENDPOINTS=(
    ["jellyfin"]="/System/Info/Public"
    ["plex"]="/identity"
    ["sonarr"]="/api/v3/system/status"
    ["radarr"]="/api/v3/system/status"
    ["lidarr"]="/api/v1/system/status"
    ["prowlarr"]="/api/v1/system/status"
    ["bazarr"]="/api/system/status"
    ["jellyseerr"]="/api/v1/status"
    ["overseerr"]="/api/v1/status"
    ["qbittorrent"]="/api/v2/app/version"
    ["transmission"]="/transmission/rpc"
    ["sabnzbd"]="/api?mode=version&output=json"
    ["grafana"]="/api/health"
    ["prometheus"]="/api/v1/query?query=up"
    ["uptime-kuma"]="/api/status-page/heartbeat"
)

# Global variables
TOTAL_SERVICES=0
HEALTHY_SERVICES=0
UNHEALTHY_SERVICES=0
declare -A SERVICE_STATUS
declare -A SERVICE_RESPONSE_TIME
declare -A SERVICE_LAST_ERROR
declare -A FAILURE_COUNT

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

debug() {
    if [ "${DEBUG:-}" = "1" ]; then
        echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}"
    fi
}

# Initialize monitoring
initialize_monitoring() {
    mkdir -p "$STATUS_LOG_DIR"
    log "Health monitoring initialized"
    log "Monitoring ${#SERVICES[@]} services"
    log "Check interval: ${HEALTH_CHECK_INTERVAL}s"
    log "Log retention: ${LOG_RETENTION_DAYS} days"
}

# Check service health
check_service_health() {
    local service="$1"
    local base_url="${SERVICES[$service]}"
    local endpoint="${SERVICE_HEALTH_ENDPOINTS[$service]:-}"
    local api_key="${SERVICE_API_KEYS[$service]:-}"
    
    local full_url="${base_url}${endpoint}"
    local start_time=$(date +%s%3N)
    local headers=""
    
    # Add API key header if available
    if [ -n "$api_key" ]; then
        headers="-H X-Api-Key:$api_key"
    fi
    
    # Special handling for transmission
    if [ "$service" = "transmission" ]; then
        headers="-H X-Transmission-Session-Id:$(curl -s -D - "$base_url" 2>/dev/null | grep -i 'X-Transmission-Session-Id' | cut -d' ' -f2 | tr -d '\r' || echo '')"
    fi
    
    debug "Checking $service at $full_url"
    
    local response
    local http_code
    
    if response=$(curl -s -w "%{http_code}" --connect-timeout 10 --max-time 30 $headers "$full_url" 2>/dev/null); then
        http_code="${response: -3}"
        response="${response%???}"
        
        local end_time=$(date +%s%3N)
        local response_time=$((end_time - start_time))
        
        if [ "$http_code" = "200" ] || [ "$http_code" = "409" ]; then  # 409 is OK for transmission
            SERVICE_STATUS["$service"]="healthy"
            SERVICE_RESPONSE_TIME["$service"]="$response_time"
            FAILURE_COUNT["$service"]=0
            ((HEALTHY_SERVICES++))
            debug "$service is healthy (${response_time}ms, HTTP $http_code)"
        else
            SERVICE_STATUS["$service"]="unhealthy"
            SERVICE_LAST_ERROR["$service"]="HTTP $http_code"
            ((FAILURE_COUNT["$service"]++))
            ((UNHEALTHY_SERVICES++))
            debug "$service is unhealthy (HTTP $http_code)"
        fi
    else
        SERVICE_STATUS["$service"]="unreachable"
        SERVICE_LAST_ERROR["$service"]="Connection failed"
        ((FAILURE_COUNT["$service"]++))
        ((UNHEALTHY_SERVICES++))
        debug "$service is unreachable"
    fi
    
    ((TOTAL_SERVICES++))
}

# Check Docker container status
check_docker_containers() {
    log "Checking Docker container status"
    
    for service in "${!SERVICES[@]}"; do
        if docker ps --filter "name=$service" --format "{{.Names}}" | grep -q "^$service$"; then
            debug "Container $service is running"
        else
            warn "Container $service is not running"
            SERVICE_STATUS["$service"]="container_down"
            SERVICE_LAST_ERROR["$service"]="Container not running"
        fi
    done
}

# Check service integrations
check_integrations() {
    log "Checking service integrations"
    
    # Check Prowlarr applications
    if [ "${SERVICE_STATUS["prowlarr"]:-}" = "healthy" ] && [ -n "${PROWLARR_API_KEY:-}" ]; then
        local apps_response
        if apps_response=$(curl -s -H "X-Api-Key:$PROWLARR_API_KEY" "http://localhost:9696/api/v1/applications" 2>/dev/null); then
            local app_count=$(echo "$apps_response" | jq '. | length' 2>/dev/null || echo "0")
            debug "Prowlarr has $app_count applications configured"
            if [ "$app_count" -gt 0 ]; then
                log "✓ Prowlarr integrations: $app_count applications"
            else
                warn "⚠ Prowlarr has no applications configured"
            fi
        fi
    fi
    
    # Check download clients in Sonarr
    if [ "${SERVICE_STATUS["sonarr"]:-}" = "healthy" ] && [ -n "${SONARR_API_KEY:-}" ]; then
        local dl_response
        if dl_response=$(curl -s -H "X-Api-Key:$SONARR_API_KEY" "http://localhost:8989/api/v3/downloadclient" 2>/dev/null); then
            local dl_count=$(echo "$dl_response" | jq '. | length' 2>/dev/null || echo "0")
            debug "Sonarr has $dl_count download clients configured"
            if [ "$dl_count" -gt 0 ]; then
                log "✓ Sonarr download clients: $dl_count configured"
            else
                warn "⚠ Sonarr has no download clients configured"
            fi
        fi
    fi
}

# Generate status report
generate_status_report() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local health_percentage=$((HEALTHY_SERVICES * 100 / TOTAL_SERVICES))
    
    # Console report
    echo -e "\n${CYAN}=== HEALTH MONITOR REPORT ===${NC}"
    echo -e "${CYAN}Timestamp: $timestamp${NC}"
    echo -e "${CYAN}Overall Health: $health_percentage% ($HEALTHY_SERVICES/$TOTAL_SERVICES services)${NC}"
    
    # Service status
    echo -e "\n${BLUE}Service Status:${NC}"
    for service in "${!SERVICES[@]}"; do
        local status="${SERVICE_STATUS[$service]:-unknown}"
        local response_time="${SERVICE_RESPONSE_TIME[$service]:-0}"
        local failures="${FAILURE_COUNT[$service]:-0}"
        
        case "$status" in
            "healthy")
                echo -e "${GREEN}✓${NC} $service (${response_time}ms)"
                ;;
            "unhealthy")
                local error="${SERVICE_LAST_ERROR[$service]:-Unknown error}"
                echo -e "${RED}✗${NC} $service - $error (failures: $failures)"
                ;;
            "unreachable")
                local error="${SERVICE_LAST_ERROR[$service]:-Connection failed}"
                echo -e "${RED}✗${NC} $service - $error (failures: $failures)"
                ;;
            "container_down")
                echo -e "${RED}✗${NC} $service - Container not running"
                ;;
            *)
                echo -e "${YELLOW}?${NC} $service - Status unknown"
                ;;
        esac
    done
    
    # Critical alerts
    if [ $UNHEALTHY_SERVICES -gt 0 ]; then
        echo -e "\n${RED}=== CRITICAL ALERTS ===${NC}"
        for service in "${!SERVICE_STATUS[@]}"; do
            local failures="${FAILURE_COUNT[$service]:-0}"
            if [ "$failures" -ge "$ALERT_THRESHOLD_FAILURES" ]; then
                error "$service has failed $failures times consecutively"
            fi
        done
    fi
}

# Generate HTML dashboard
generate_html_dashboard() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local health_percentage=$((HEALTHY_SERVICES * 100 / TOTAL_SERVICES))
    
    cat > "$STATUS_DASHBOARD" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Media Server Health Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .services { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .service-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status-healthy { color: #28a745; font-weight: bold; }
        .status-unhealthy { color: #dc3545; font-weight: bold; }
        .status-unknown { color: #ffc107; font-weight: bold; }
        .health-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
        .health-fill { height: 100%; background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%); transition: width 0.3s ease; }
        .timestamp { text-align: center; margin: 20px 0; color: #666; }
        .alert { background: #f8d7da; color: #721c24; padding: 15px; border: 1px solid #f5c6cb; border-radius: 5px; margin: 10px 0; }
        .success { background: #d4edda; color: #155724; padding: 15px; border: 1px solid #c3e6cb; border-radius: 5px; margin: 10px 0; }
        .auto-refresh { position: fixed; top: 20px; right: 20px; background: #007bff; color: white; padding: 10px 15px; border-radius: 5px; font-size: 12px; }
    </style>
    <script>
        // Auto-refresh every 5 minutes
        setTimeout(() => location.reload(), 300000);
    </script>
</head>
<body>
    <div class="auto-refresh">Auto-refresh: 5min</div>
    <div class="container">
        <div class="header">
            <h1>📊 Media Server Health Dashboard</h1>
            <p>Real-time monitoring of all service integrations</p>
        </div>
        
        <div class="timestamp">Last Updated: $timestamp</div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Overall Health</h3>
                <div class="health-bar">
                    <div class="health-fill" style="width: ${health_percentage}%"></div>
                </div>
                <h2>$health_percentage%</h2>
                <p>$HEALTHY_SERVICES of $TOTAL_SERVICES services healthy</p>
            </div>
            <div class="stat-card">
                <h3>Healthy Services</h3>
                <h2 class="status-healthy">$HEALTHY_SERVICES</h2>
                <p>Services operating normally</p>
            </div>
            <div class="stat-card">
                <h3>Unhealthy Services</h3>
                <h2 class="status-unhealthy">$UNHEALTHY_SERVICES</h2>
                <p>Services requiring attention</p>
            </div>
        </div>
        
        $(if [ $UNHEALTHY_SERVICES -eq 0 ]; then
            echo '<div class="success">✅ All services are healthy! Your media server is operating optimally.</div>'
        else
            echo '<div class="alert">⚠️ '$UNHEALTHY_SERVICES' service(s) need attention. Check the status below for details.</div>'
        fi)
        
        <h2>Service Status</h2>
        <div class="services">
EOF
    
    # Add service cards
    for service in $(echo "${!SERVICES[@]}" | tr ' ' '\n' | sort); do
        local status="${SERVICE_STATUS[$service]:-unknown}"
        local url="${SERVICES[$service]}"
        local response_time="${SERVICE_RESPONSE_TIME[$service]:-0}"
        local failures="${FAILURE_COUNT[$service]:-0}"
        local last_error="${SERVICE_LAST_ERROR[$service]:-}"
        
        local status_class="status-unknown"
        local status_icon="❓"
        local status_text="Unknown"
        
        case "$status" in
            "healthy")
                status_class="status-healthy"
                status_icon="✅"
                status_text="Healthy"
                ;;
            "unhealthy"|"unreachable"|"container_down")
                status_class="status-unhealthy"
                status_icon="❌"
                status_text="Unhealthy"
                ;;
        esac
        
        cat >> "$STATUS_DASHBOARD" << EOF
            <div class="service-card">
                <h3>$status_icon $service</h3>
                <p><strong>Status:</strong> <span class="$status_class">$status_text</span></p>
                <p><strong>URL:</strong> <a href="$url" target="_blank">$url</a></p>
                $(if [ "$status" = "healthy" ]; then
                    echo "<p><strong>Response Time:</strong> ${response_time}ms</p>"
                fi)
                $(if [ "$failures" -gt 0 ]; then
                    echo "<p><strong>Consecutive Failures:</strong> $failures</p>"
                fi)
                $(if [ -n "$last_error" ] && [ "$status" != "healthy" ]; then
                    echo "<p><strong>Last Error:</strong> $last_error</p>"
                fi)
            </div>
EOF
    done
    
    cat >> "$STATUS_DASHBOARD" << EOF
        </div>
        
        <div style="margin-top: 40px; text-align: center; color: #666; font-size: 12px;">
            <p>Generated by Media Server Health Monitor | $(date)</p>
        </div>
    </div>
</body>
</html>
EOF

    log "HTML dashboard generated: $STATUS_DASHBOARD"
}

# Log status to file
log_status() {
    local log_file="$STATUS_LOG_DIR/health-$(date +%Y%m%d).log"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    
    {
        echo "[$timestamp] HEALTH_CHECK_START"
        echo "[$timestamp] TOTAL_SERVICES=$TOTAL_SERVICES"
        echo "[$timestamp] HEALTHY_SERVICES=$HEALTHY_SERVICES"
        echo "[$timestamp] UNHEALTHY_SERVICES=$UNHEALTHY_SERVICES"
        
        for service in "${!SERVICE_STATUS[@]}"; do
            local status="${SERVICE_STATUS[$service]}"
            local response_time="${SERVICE_RESPONSE_TIME[$service]:-0}"
            local failures="${FAILURE_COUNT[$service]:-0}"
            echo "[$timestamp] SERVICE=$service STATUS=$status RESPONSE_TIME=${response_time}ms FAILURES=$failures"
        done
        
        echo "[$timestamp] HEALTH_CHECK_END"
    } >> "$log_file"
}

# Clean old logs
cleanup_logs() {
    find "$STATUS_LOG_DIR" -name "health-*.log" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    debug "Cleaned up logs older than $LOG_RETENTION_DAYS days"
}

# Single health check
run_health_check() {
    # Reset counters
    TOTAL_SERVICES=0
    HEALTHY_SERVICES=0
    UNHEALTHY_SERVICES=0
    
    log "Running health check for all services..."
    
    # Check Docker containers first
    check_docker_containers
    
    # Check each service health
    for service in "${!SERVICES[@]}"; do
        check_service_health "$service"
    done
    
    # Check integrations
    check_integrations
    
    # Generate reports
    generate_status_report
    generate_html_dashboard
    log_status
    cleanup_logs
    
    log "Health check completed"
}

# Continuous monitoring mode
run_continuous_monitoring() {
    log "Starting continuous health monitoring (interval: ${HEALTH_CHECK_INTERVAL}s)"
    
    while true; do
        run_health_check
        echo -e "\n${BLUE}Next check in $HEALTH_CHECK_INTERVAL seconds...${NC}"
        sleep "$HEALTH_CHECK_INTERVAL"
    done
}

# Show usage
usage() {
    echo "Media Server Health Monitor"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --continuous    Run continuous monitoring"
    echo "  -o, --once         Run single health check"
    echo "  -i, --interval N   Set check interval in seconds (default: 300)"
    echo "  -d, --debug        Enable debug output"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --once              # Single health check"
    echo "  $0 --continuous        # Continuous monitoring"
    echo "  $0 -c -i 60           # Monitor every 60 seconds"
    echo ""
    echo "Dashboard: $STATUS_DASHBOARD"
}

# Parse command line arguments
CONTINUOUS=0
SINGLE_CHECK=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--continuous)
            CONTINUOUS=1
            shift
            ;;
        -o|--once)
            SINGLE_CHECK=1
            shift
            ;;
        -i|--interval)
            HEALTH_CHECK_INTERVAL="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo -e "${BLUE}=== MEDIA SERVER HEALTH MONITOR ===${NC}"
    
    initialize_monitoring
    
    if [ "$CONTINUOUS" -eq 1 ]; then
        run_continuous_monitoring
    elif [ "$SINGLE_CHECK" -eq 1 ]; then
        run_health_check
    else
        # Default to single check
        run_health_check
        echo -e "\n${YELLOW}Use --continuous for continuous monitoring${NC}"
        echo -e "${YELLOW}Dashboard available at: file://$PWD/$STATUS_DASHBOARD${NC}"
    fi
}

# Check dependencies
if ! command -v curl &> /dev/null; then
    error "curl is required but not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    error "docker is required but not installed"
    exit 1
fi

# Run main function
main "$@"