#!/bin/bash

# Media Server - Health Check Script
# Monitors service health and system resources
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
readonly HEALTH_LOG="${PROJECT_ROOT}/logs/health-check.log"
readonly TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Health check configuration
declare -A SERVICE_PORTS=(
    ["jellyfin"]="8096"
    ["radarr"]="7878"
    ["sonarr"]="8989"
    ["prowlarr"]="9696"
    ["qbittorrent"]="8080"
    ["bazarr"]="6767"
    ["overseerr"]="5055"
    ["tautulli"]="8181"
    ["homepage"]="3000"
    ["portainer"]="9000"
)

# Health status tracking
declare -A SERVICE_STATUS
TOTAL_CHECKS=0
PASSED_CHECKS=0
WARNINGS=0
FAILURES=0

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[$TIMESTAMP] INFO: $1" >> "$HEALTH_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[$TIMESTAMP] WARN: $1" >> "$HEALTH_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$TIMESTAMP] ERROR: $1" >> "$HEALTH_LOG"
}

# Status icon
status_icon() {
    case $1 in
        "healthy") echo -e "${GREEN}✓${NC}" ;;
        "warning") echo -e "${YELLOW}⚠${NC}" ;;
        "error") echo -e "${RED}✗${NC}" ;;
        *) echo -e "${BLUE}?${NC}" ;;
    esac
}

# Initialize health check
init_health_check() {
    # Create log directory
    mkdir -p "$(dirname "$HEALTH_LOG")"
    
    # Write header to log
    echo "=== Media Server Health Check - $TIMESTAMP ===" >> "$HEALTH_LOG"
}

# Check Docker status
check_docker() {
    echo -e "\n${CYAN}━━━ Docker Status ━━━${NC}"
    
    ((TOTAL_CHECKS++))
    
    if ! command -v docker &> /dev/null; then
        SERVICE_STATUS["docker"]="error"
        ((FAILURES++))
        echo -e "$(status_icon error) Docker: Not installed"
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        SERVICE_STATUS["docker"]="error"
        ((FAILURES++))
        echo -e "$(status_icon error) Docker: Daemon not running"
        return 1
    fi
    
    SERVICE_STATUS["docker"]="healthy"
    ((PASSED_CHECKS++))
    
    # Get Docker info
    local version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    echo -e "$(status_icon healthy) Docker: Running (v$version)"
    
    # Check Docker resources
    local containers=$(docker ps -q | wc -l)
    local images=$(docker images -q | wc -l)
    echo -e "  Containers: $containers running"
    echo -e "  Images: $images available"
}

# Check system resources
check_system_resources() {
    echo -e "\n${CYAN}━━━ System Resources ━━━${NC}"
    
    # CPU usage (macOS compatible)
    if command -v top &> /dev/null; then
        local cpu_usage=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
        ((TOTAL_CHECKS++))
        
        if (( $(echo "$cpu_usage < 80" | bc -l) )); then
            SERVICE_STATUS["cpu"]="healthy"
            ((PASSED_CHECKS++))
            echo -e "$(status_icon healthy) CPU Usage: ${cpu_usage}%"
        elif (( $(echo "$cpu_usage < 90" | bc -l) )); then
            SERVICE_STATUS["cpu"]="warning"
            ((WARNINGS++))
            echo -e "$(status_icon warning) CPU Usage: ${cpu_usage}% (High)"
        else
            SERVICE_STATUS["cpu"]="error"
            ((FAILURES++))
            echo -e "$(status_icon error) CPU Usage: ${cpu_usage}% (Critical)"
        fi
    fi
    
    # Memory usage (macOS compatible)
    if command -v vm_stat &> /dev/null; then
        # macOS memory calculation
        local page_size=$(vm_stat | grep "page size" | awk '{print $8}')
        local pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        local pages_active=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
        local pages_inactive=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        local pages_wired=$(vm_stat | grep "Pages wired" | awk '{print $4}' | sed 's/\.//')
        
        local mem_free=$((pages_free * page_size / 1024 / 1024))
        local mem_used=$(((pages_active + pages_inactive + pages_wired) * page_size / 1024 / 1024))
        local mem_total=$((mem_free + mem_used))
        local mem_percent=$((mem_used * 100 / mem_total))
        
        ((TOTAL_CHECKS++))
        
        if [ $mem_percent -lt 80 ]; then
            SERVICE_STATUS["memory"]="healthy"
            ((PASSED_CHECKS++))
            echo -e "$(status_icon healthy) Memory: ${mem_percent}% used (${mem_used}MB/${mem_total}MB)"
        elif [ $mem_percent -lt 90 ]; then
            SERVICE_STATUS["memory"]="warning"
            ((WARNINGS++))
            echo -e "$(status_icon warning) Memory: ${mem_percent}% used (${mem_used}MB/${mem_total}MB)"
        else
            SERVICE_STATUS["memory"]="error"
            ((FAILURES++))
            echo -e "$(status_icon error) Memory: ${mem_percent}% used (${mem_used}MB/${mem_total}MB)"
        fi
    fi
    
    # Disk usage
    local disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2{print $5}' | sed 's/%//')
    local disk_avail=$(df -h "$PROJECT_ROOT" | awk 'NR==2{print $4}')
    
    ((TOTAL_CHECKS++))
    
    if [ $disk_usage -lt 80 ]; then
        SERVICE_STATUS["disk"]="healthy"
        ((PASSED_CHECKS++))
        echo -e "$(status_icon healthy) Disk: ${disk_usage}% used, ${disk_avail} free"
    elif [ $disk_usage -lt 90 ]; then
        SERVICE_STATUS["disk"]="warning"
        ((WARNINGS++))
        echo -e "$(status_icon warning) Disk: ${disk_usage}% used, ${disk_avail} free"
    else
        SERVICE_STATUS["disk"]="error"
        ((FAILURES++))
        echo -e "$(status_icon error) Disk: ${disk_usage}% used, ${disk_avail} free"
    fi
}

# Check container health
check_containers() {
    echo -e "\n${CYAN}━━━ Container Health ━━━${NC}"
    
    # Get list of running containers
    local containers=$(docker-compose ps --services 2>/dev/null || echo "")
    
    if [ -z "$containers" ]; then
        echo -e "$(status_icon warning) No containers found"
        return
    fi
    
    for container in $containers; do
        ((TOTAL_CHECKS++))
        
        # Check if container is running
        if docker-compose ps "$container" 2>/dev/null | grep -q "Up"; then
            # Check container health status
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
            
            case $health in
                "healthy")
                    SERVICE_STATUS["container_$container"]="healthy"
                    ((PASSED_CHECKS++))
                    echo -e "$(status_icon healthy) $container: Running (Healthy)"
                    ;;
                "unhealthy")
                    SERVICE_STATUS["container_$container"]="error"
                    ((FAILURES++))
                    echo -e "$(status_icon error) $container: Running (Unhealthy)"
                    ;;
                "starting")
                    SERVICE_STATUS["container_$container"]="warning"
                    ((WARNINGS++))
                    echo -e "$(status_icon warning) $container: Starting"
                    ;;
                *)
                    SERVICE_STATUS["container_$container"]="healthy"
                    ((PASSED_CHECKS++))
                    echo -e "$(status_icon healthy) $container: Running"
                    ;;
            esac
            
            # Show resource usage
            local stats=$(docker stats --no-stream --format "CPU: {{.CPUPerc}} | MEM: {{.MemUsage}}" "$container" 2>/dev/null || echo "")
            if [ -n "$stats" ]; then
                echo -e "    └─ $stats"
            fi
        else
            SERVICE_STATUS["container_$container"]="error"
            ((FAILURES++))
            echo -e "$(status_icon error) $container: Not running"
        fi
    done
}

# Check service endpoints
check_service_endpoints() {
    echo -e "\n${CYAN}━━━ Service Endpoints ━━━${NC}"
    
    for service in "${!SERVICE_PORTS[@]}"; do
        local port="${SERVICE_PORTS[$service]}"
        ((TOTAL_CHECKS++))
        
        # Check if service is accessible
        if curl -sf -m 5 "http://localhost:$port" > /dev/null 2>&1; then
            SERVICE_STATUS["endpoint_$service"]="healthy"
            ((PASSED_CHECKS++))
            echo -e "$(status_icon healthy) $service: http://localhost:$port ✓"
        else
            # Check if container exists
            if docker ps --format "{{.Names}}" | grep -q "$service"; then
                SERVICE_STATUS["endpoint_$service"]="warning"
                ((WARNINGS++))
                echo -e "$(status_icon warning) $service: http://localhost:$port (Not responding)"
            else
                SERVICE_STATUS["endpoint_$service"]="error"
                ((FAILURES++))
                echo -e "$(status_icon error) $service: http://localhost:$port (Container not running)"
            fi
        fi
    done
}

# Check logs for errors
check_logs() {
    echo -e "\n${CYAN}━━━ Recent Log Analysis ━━━${NC}"
    
    local log_dir="${PROJECT_ROOT}/logs"
    local error_count=0
    local warning_count=0
    
    # Check Docker logs for errors
    for container in $(docker-compose ps --services 2>/dev/null || echo ""); do
        if docker ps --format "{{.Names}}" | grep -q "$container"; then
            # Count errors in last 100 lines
            local container_errors=$(docker logs --tail 100 "$container" 2>&1 | grep -ci "error" || true)
            local container_warnings=$(docker logs --tail 100 "$container" 2>&1 | grep -ci "warning" || true)
            
            error_count=$((error_count + container_errors))
            warning_count=$((warning_count + container_warnings))
        fi
    done
    
    ((TOTAL_CHECKS++))
    
    if [ $error_count -eq 0 ]; then
        SERVICE_STATUS["logs"]="healthy"
        ((PASSED_CHECKS++))
        echo -e "$(status_icon healthy) No recent errors in logs"
    elif [ $error_count -lt 5 ]; then
        SERVICE_STATUS["logs"]="warning"
        ((WARNINGS++))
        echo -e "$(status_icon warning) $error_count errors found in recent logs"
    else
        SERVICE_STATUS["logs"]="error"
        ((FAILURES++))
        echo -e "$(status_icon error) $error_count errors found in recent logs"
    fi
    
    if [ $warning_count -gt 0 ]; then
        echo -e "  └─ $warning_count warnings found"
    fi
}

# Generate summary
generate_summary() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Health Check Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    echo -e "\nTotal Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo -e "Failed: ${RED}$FAILURES${NC}"
    
    # Calculate health score
    if [ $TOTAL_CHECKS -gt 0 ]; then
        local health_score=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
        
        echo -ne "\nOverall Health: "
        if [ $health_score -ge 90 ]; then
            echo -e "${GREEN}${health_score}% - Excellent${NC}"
        elif [ $health_score -ge 70 ]; then
            echo -e "${YELLOW}${health_score}% - Good${NC}"
        elif [ $health_score -ge 50 ]; then
            echo -e "${YELLOW}${health_score}% - Fair${NC}"
        else
            echo -e "${RED}${health_score}% - Poor${NC}"
        fi
    fi
    
    # Recommendations
    if [ $FAILURES -gt 0 ] || [ $WARNINGS -gt 0 ]; then
        echo -e "\n${CYAN}Recommendations:${NC}"
        
        # Check for specific issues
        if [ "${SERVICE_STATUS[docker]:-}" = "error" ]; then
            echo "• Start Docker Desktop"
        fi
        
        if [ "${SERVICE_STATUS[cpu]:-}" = "error" ] || [ "${SERVICE_STATUS[cpu]:-}" = "warning" ]; then
            echo "• High CPU usage detected - check resource-intensive services"
        fi
        
        if [ "${SERVICE_STATUS[memory]:-}" = "error" ] || [ "${SERVICE_STATUS[memory]:-}" = "warning" ]; then
            echo "• High memory usage - consider increasing Docker memory limit"
        fi
        
        if [ "${SERVICE_STATUS[disk]:-}" = "error" ]; then
            echo "• Low disk space - clean up old downloads and logs"
        fi
        
        # Check for failed containers
        for key in "${!SERVICE_STATUS[@]}"; do
            if [[ $key == container_* ]] && [ "${SERVICE_STATUS[$key]}" = "error" ]; then
                local container="${key#container_}"
                echo "• Start $container: docker-compose up -d $container"
            fi
        done
    fi
    
    echo -e "\nDetailed log: $HEALTH_LOG"
}

# Export health data as JSON
export_json() {
    local json_file="${PROJECT_ROOT}/logs/health-check-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$json_file" << EOF
{
  "timestamp": "$TIMESTAMP",
  "summary": {
    "total_checks": $TOTAL_CHECKS,
    "passed": $PASSED_CHECKS,
    "warnings": $WARNINGS,
    "failures": $FAILURES,
    "health_score": $((TOTAL_CHECKS > 0 ? PASSED_CHECKS * 100 / TOTAL_CHECKS : 0))
  },
  "services": {
$(for service in "${!SERVICE_STATUS[@]}"; do
    echo "    \"$service\": \"${SERVICE_STATUS[$service]}\","
done | sed '$ s/,$//')
  }
}
EOF
    
    echo -e "\nJSON report: $json_file"
}

# Main health check flow
main() {
    echo -e "${BLUE}Media Server Health Check${NC}"
    echo -e "${BLUE}$TIMESTAMP${NC}"
    
    init_health_check
    
    # Run all checks
    check_docker
    check_system_resources
    check_containers
    check_service_endpoints
    check_logs
    
    # Generate summary
    generate_summary
    
    # Export JSON if requested
    if [[ "${1:-}" == "--json" ]]; then
        export_json
    fi
    
    # Exit with appropriate code
    if [ $FAILURES -gt 0 ]; then
        exit 2
    elif [ $WARNINGS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"