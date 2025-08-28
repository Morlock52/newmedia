#!/bin/bash

# ==================================================================
# COMPREHENSIVE HEALTH CHECK SYSTEM
# Monitors all services and provides detailed health information
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
LOG_FILE="./logs/health-check-$(date +%Y%m%d_%H%M%S).log"
ALERT_WEBHOOK=""  # Slack/Discord webhook for alerts
HEALTH_CHECK_INTERVAL=30  # seconds between checks
MAX_FAILURES=3  # consecutive failures before alert

# Ensure log directory exists
mkdir -p logs

# Service definitions with health check endpoints
declare -A SERVICES=(
    ["jellyfin"]="8096:/health"
    ["sonarr"]="8989:/ping"
    ["radarr"]="7878:/ping"
    ["prowlarr"]="9696:/ping"
    ["lidarr"]="8686:/ping"
    ["bazarr"]="6767:"
    ["qbittorrent"]="8080:/api/v2/app/version"
    ["plex"]="32400:/identity"
    ["grafana"]="3000:/api/health"
    ["prometheus"]="9090:/-/healthy"
    ["uptime-kuma"]="3001:"
    ["portainer"]="9000:/api/status"
    ["nginx-proxy-manager"]="81:/api/health"
    ["postgres"]="5432:"  # TCP check only
    ["redis"]="6379:"     # TCP check only
    ["mariadb"]="3306:"   # TCP check only
)

# Database connection tests
declare -A DB_TESTS=(
    ["postgres"]="PGPASSWORD=\${POSTGRES_PASSWORD:-postgres} psql -h localhost -U \${POSTGRES_USER:-postgres} -d postgres -c 'SELECT 1;'"
    ["mariadb"]="mysql -h localhost -u root -p\${MYSQL_ROOT_PASSWORD:-root} -e 'SELECT 1;'"
    ["redis"]="redis-cli -h localhost ping"
)

# Critical services that must be running
CRITICAL_SERVICES=("postgres" "redis" "jellyfin" "sonarr" "radarr" "prowlarr")

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

# Send alert to webhook
send_alert() {
    local message="$1"
    local severity="${2:-warning}"
    
    if [ -n "$ALERT_WEBHOOK" ]; then
        local color="16776960"  # Yellow
        case "$severity" in
            "critical") color="16711680" ;;  # Red
            "warning") color="16776960" ;;   # Yellow
            "info") color="65280" ;;         # Green
        esac
        
        curl -s -H "Content-Type: application/json" \
             -d "{\"embeds\": [{\"title\": \"Media Server Alert\", \"description\": \"$message\", \"color\": $color, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\"}]}" \
             "$ALERT_WEBHOOK" || true
    fi
}

# Check if Docker container is running
check_container() {
    local service="$1"
    
    if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
        return 0
    else
        return 1
    fi
}

# Check HTTP endpoint
check_http_endpoint() {
    local host="$1"
    local port="$2"
    local path="${3:-/}"
    local timeout="${4:-5}"
    
    curl -sf --connect-timeout "$timeout" "http://$host:$port$path" > /dev/null 2>&1
}

# Check TCP connection
check_tcp_connection() {
    local host="$1"
    local port="$2"
    local timeout="${3:-5}"
    
    timeout "$timeout" bash -c "</dev/tcp/$host/$port" > /dev/null 2>&1
}

# Check database connectivity
check_database() {
    local db_type="$1"
    
    if [ -n "${DB_TESTS[$db_type]:-}" ]; then
        eval "${DB_TESTS[$db_type]}" > /dev/null 2>&1
        return $?
    fi
    
    return 1
}

# Get container stats
get_container_stats() {
    local service="$1"
    
    if check_container "$service"; then
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" "$service" 2>/dev/null | tail -1
    else
        echo "Container not running"
    fi
}

# Check disk usage
check_disk_usage() {
    local path="${1:-.}"
    local threshold="${2:-85}"
    
    local usage
    usage=$(df "$path" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$usage" -gt "$threshold" ]; then
        warn "Disk usage is ${usage}% (threshold: ${threshold}%)"
        return 1
    fi
    
    return 0
}

# Check memory usage
check_memory_usage() {
    local threshold="${1:-85}"
    
    local usage
    usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    
    if [ "$usage" -gt "$threshold" ]; then
        warn "Memory usage is ${usage}% (threshold: ${threshold}%)"
        return 1
    fi
    
    return 0
}

# Comprehensive service health check
check_service_health() {
    local service="$1"
    local endpoint="${SERVICES[$service]:-}"
    local status="UNKNOWN"
    local details=""
    
    # Check if container is running
    if ! check_container "$service"; then
        status="DOWN"
        details="Container not running"
        return 1
    fi
    
    # Extract port and path from endpoint
    if [ -n "$endpoint" ]; then
        IFS=':' read -r port path <<< "$endpoint"
        
        if [ -n "$path" ] && [ "$path" != "" ]; then
            # HTTP health check
            if check_http_endpoint "localhost" "$port" "$path"; then
                status="HEALTHY"
                details="HTTP check passed"
            else
                status="UNHEALTHY"
                details="HTTP check failed"
                return 1
            fi
        else
            # TCP connection check
            if check_tcp_connection "localhost" "$port"; then
                status="HEALTHY"
                details="TCP connection successful"
            else
                status="UNHEALTHY"
                details="TCP connection failed"
                return 1
            fi
        fi
    fi
    
    # Additional database checks
    case "$service" in
        "postgres"|"mariadb"|"redis")
            if check_database "$service"; then
                details="$details, DB query successful"
            else
                status="UNHEALTHY"
                details="$details, DB query failed"
                return 1
            fi
            ;;
    esac
    
    echo "$status:$details"
    return 0
}

# Generate health report
generate_health_report() {
    local report_file="./logs/health-report-$(date +%Y%m%d_%H%M%S).html"
    local total_services=0
    local healthy_services=0
    local unhealthy_services=0
    
    # HTML header
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Health Report</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }
        .header { background: linear-gradient(135deg, #238636 0%, #1f6feb 100%); padding: 20px; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card h3 { margin-top: 0; color: #58a6ff; }
        .status { padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: bold; display: inline-block; }
        .status-healthy { background: #238636; color: white; }
        .status-unhealthy { background: #da3633; color: white; }
        .status-down { background: #6e7681; color: white; }
        .stats { font-family: monospace; font-size: 12px; background: #0d1117; padding: 8px; border-radius: 4px; margin: 8px 0; }
        .metrics { display: flex; justify-content: space-between; margin: 20px 0; }
        .metric { text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #58a6ff; }
        .critical-alert { background: #da3633; color: white; padding: 16px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Media Server Health Dashboard</h1>
        <p>Last updated: <span id="timestamp"></span></p>
    </div>
    
    <div class="container">
EOF

    # System overview
    local system_load
    system_load=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^[ \t]*//')
    
    local memory_info
    memory_info=$(free -h | awk 'NR==2{printf "%s used / %s total (%.1f%%)", $3, $2, $3*100/$2}')
    
    local disk_info
    disk_info=$(df -h . | awk 'NR==2{printf "%s used / %s total (%s)", $3, $2, $5}')
    
    cat >> "$report_file" << EOF
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="healthy-count">0</div>
                <div>Healthy Services</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="unhealthy-count">0</div>
                <div>Unhealthy Services</div>
            </div>
            <div class="metric">
                <div class="metric-value">$(docker ps --format "{{.Names}}" | wc -l)</div>
                <div>Running Containers</div>
            </div>
        </div>
        
        <div class="card">
            <h3>System Resources</h3>
            <div class="stats">
                <div><strong>Load Average:</strong> $system_load</div>
                <div><strong>Memory:</strong> $memory_info</div>
                <div><strong>Disk:</strong> $disk_info</div>
                <div><strong>Docker Version:</strong> $(docker --version)</div>
            </div>
        </div>
        
        <div class="grid">
EOF

    # Check each service
    for service in "${!SERVICES[@]}"; do
        total_services=$((total_services + 1))
        
        local health_result
        local status="UNKNOWN"
        local details="No details available"
        
        if health_result=$(check_service_health "$service" 2>&1); then
            IFS=':' read -r status details <<< "$health_result"
            healthy_services=$((healthy_services + 1))
        else
            if [ -n "$health_result" ]; then
                IFS=':' read -r status details <<< "$health_result"
            fi
            unhealthy_services=$((unhealthy_services + 1))
        fi
        
        # Get container stats
        local stats
        stats=$(get_container_stats "$service")
        
        # Determine CSS class
        local css_class="status-down"
        case "$status" in
            "HEALTHY") css_class="status-healthy" ;;
            "UNHEALTHY") css_class="status-unhealthy" ;;
        esac
        
        # Service card
        cat >> "$report_file" << EOF
            <div class="card">
                <h3>$service</h3>
                <div class="status $css_class">$status</div>
                <p><strong>Details:</strong> $details</p>
                <div class="stats">
                    <strong>Container Stats:</strong><br>
                    $stats
                </div>
                <p><strong>Endpoint:</strong> ${SERVICES[$service]}</p>
            </div>
EOF
    done
    
    # HTML footer
    cat >> "$report_file" << 'EOF'
        </div>
        
        <div class="card">
            <h3>Docker Containers Status</h3>
            <div class="stats">
EOF

    # Add running containers info
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | while IFS=$'\t' read -r name status ports; do
        [ "$name" = "NAMES" ] && continue
        echo "                <div><strong>$name:</strong> $status</div>" >> "$report_file"
    done
    
    cat >> "$report_file" << 'EOF'
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
EOF

    echo "        document.getElementById('healthy-count').textContent = '$healthy_services';" >> "$report_file"
    echo "        document.getElementById('unhealthy-count').textContent = '$unhealthy_services';" >> "$report_file"
    
    cat >> "$report_file" << 'EOF'
    </script>
</body>
</html>
EOF

    success "Health report generated: $report_file"
    echo "$report_file"
}

# Quick status check
quick_status() {
    echo -e "${PURPLE}Quick Status Check${NC}"
    echo "===================="
    
    local healthy=0
    local total=0
    local critical_down=()
    
    for service in "${!SERVICES[@]}"; do
        total=$((total + 1))
        
        if check_service_health "$service" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $service"
            healthy=$((healthy + 1))
        else
            echo -e "${RED}✗${NC} $service"
            
            # Check if it's a critical service
            for critical in "${CRITICAL_SERVICES[@]}"; do
                if [ "$critical" = "$service" ]; then
                    critical_down+=("$service")
                    break
                fi
            done
        fi
    done
    
    echo "===================="
    echo -e "Status: $healthy/$total services healthy"
    
    if [ ${#critical_down[@]} -gt 0 ]; then
        echo -e "${RED}CRITICAL: ${critical_down[*]} are down!${NC}"
        send_alert "Critical services down: ${critical_down[*]}" "critical"
        return 1
    fi
    
    return 0
}

# Continuous monitoring
monitor() {
    local duration="${1:-3600}"  # Default 1 hour
    local end_time=$(($(date +%s) + duration))
    
    log "Starting continuous monitoring for $duration seconds..."
    
    while [ $(date +%s) -lt $end_time ]; do
        if ! quick_status; then
            send_alert "Health check failed - critical services are down" "critical"
        fi
        
        sleep "$HEALTH_CHECK_INTERVAL"
    done
    
    log "Monitoring period completed"
}

# Main function
main() {
    case "${1:-status}" in
        "status"|"quick")
            quick_status
            ;;
        "report")
            report_file=$(generate_health_report)
            
            # Open report if possible
            if command -v xdg-open &> /dev/null; then
                xdg-open "$report_file"
            elif command -v open &> /dev/null; then
                open "$report_file"
            fi
            ;;
        "monitor")
            monitor "${2:-3600}"
            ;;
        "continuous")
            while true; do
                quick_status
                sleep "$HEALTH_CHECK_INTERVAL"
            done
            ;;
        "--help"|"-h")
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  status, quick    - Quick status check of all services"
            echo "  report          - Generate comprehensive HTML health report"
            echo "  monitor [time]  - Monitor for specified time (default: 3600s)"
            echo "  continuous      - Continuous monitoring (runs forever)"
            echo ""
            echo "Environment Variables:"
            echo "  ALERT_WEBHOOK   - Slack/Discord webhook URL for alerts"
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

# Run main function
main "$@"