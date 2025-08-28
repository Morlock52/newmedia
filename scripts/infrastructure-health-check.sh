#!/bin/bash
# Infrastructure Health Check Script
# Monitors critical infrastructure services and reports status

set -euo pipefail

# Configuration
LOG_FILE="/var/log/health-checks/infrastructure.log"
METRICS_FILE="/tmp/infrastructure-metrics.prom"
ALERT_THRESHOLD=3  # Number of consecutive failures before alerting

# Create directories
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$METRICS_FILE")"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to write Prometheus metrics
write_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local labels="${3:-}"
    
    if [[ -n "$labels" ]]; then
        echo "${metric_name}{${labels}} ${metric_value}" >> "$METRICS_FILE"
    else
        echo "${metric_name} ${metric_value}" >> "$METRICS_FILE"
    fi
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    local timeout="${4:-10}"
    
    log "Checking $name at $url"
    
    local status_code
    local response_time
    local is_healthy=0
    
    if response_time=$(curl -s -o /dev/null -w "%{time_total}" -m "$timeout" \
        --connect-timeout 5 -w "%{http_code}" "$url" 2>/dev/null); then
        status_code="${response_time: -3}"
        response_time="${response_time%???}"
        
        if [[ "$status_code" == "$expected_status" ]]; then
            is_healthy=1
            log "$name is healthy (${status_code}, ${response_time}s)"
        else
            log "ERROR: $name returned status $status_code (expected $expected_status)"
        fi
    else
        response_time="0"
        status_code="0"
        log "ERROR: $name is unreachable"
    fi
    
    # Write metrics
    write_metric "infrastructure_service_up" "$is_healthy" "service=\"$name\""
    write_metric "infrastructure_service_response_time" "$response_time" "service=\"$name\""
    write_metric "infrastructure_service_status_code" "$status_code" "service=\"$name\""
}

# Function to check TCP port
check_tcp_port() {
    local name="$1"
    local host="$2"
    local port="$3"
    local timeout="${4:-5}"
    
    log "Checking $name TCP connection at $host:$port"
    
    local is_healthy=0
    local response_time
    
    if response_time=$(timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null && echo "connected"); then
        if [[ "$response_time" == "connected" ]]; then
            is_healthy=1
            log "$name TCP connection is healthy"
            response_time="0.1"  # Estimate for successful TCP connection
        fi
    else
        response_time="0"
        log "ERROR: $name TCP connection failed"
    fi
    
    # Write metrics
    write_metric "infrastructure_tcp_up" "$is_healthy" "service=\"$name\""
    write_metric "infrastructure_tcp_response_time" "$response_time" "service=\"$name\""
}

# Function to check Docker container
check_docker_container() {
    local container_name="$1"
    
    log "Checking Docker container: $container_name"
    
    local is_running=0
    local is_healthy=0
    local restart_count=0
    
    if docker inspect "$container_name" >/dev/null 2>&1; then
        local status
        status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "unknown")
        
        if [[ "$status" == "running" ]]; then
            is_running=1
            
            # Check health status if available
            local health_status
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
            
            if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "none" ]]; then
                is_healthy=1
                log "$container_name is running and healthy"
            else
                log "WARNING: $container_name is running but unhealthy ($health_status)"
            fi
        else
            log "ERROR: $container_name is not running (status: $status)"
        fi
        
        # Get restart count
        restart_count=$(docker inspect --format='{{.RestartCount}}' "$container_name" 2>/dev/null || echo "0")
    else
        log "ERROR: $container_name container not found"
    fi
    
    # Write metrics
    write_metric "infrastructure_container_up" "$is_running" "container=\"$container_name\""
    write_metric "infrastructure_container_healthy" "$is_healthy" "container=\"$container_name\""
    write_metric "infrastructure_container_restart_count" "$restart_count" "container=\"$container_name\""
}

# Function to check disk space
check_disk_space() {
    local mount_point="$1"
    local warning_threshold="${2:-80}"
    local critical_threshold="${3:-90}"
    
    log "Checking disk space for $mount_point"
    
    if [[ ! -d "$mount_point" ]]; then
        log "WARNING: Mount point $mount_point does not exist"
        write_metric "infrastructure_disk_usage_percent" "0" "mount=\"$mount_point\""
        return
    fi
    
    local usage_percent
    usage_percent=$(df "$mount_point" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    local status="ok"
    if [[ "$usage_percent" -ge "$critical_threshold" ]]; then
        status="critical"
        log "CRITICAL: Disk usage for $mount_point is ${usage_percent}% (critical threshold: ${critical_threshold}%)"
    elif [[ "$usage_percent" -ge "$warning_threshold" ]]; then
        status="warning"
        log "WARNING: Disk usage for $mount_point is ${usage_percent}% (warning threshold: ${warning_threshold}%)"
    else
        log "Disk usage for $mount_point is healthy: ${usage_percent}%"
    fi
    
    write_metric "infrastructure_disk_usage_percent" "$usage_percent" "mount=\"$mount_point\",status=\"$status\""
}

# Function to check system resources
check_system_resources() {
    log "Checking system resources"
    
    # CPU usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "0")
    write_metric "infrastructure_cpu_usage_percent" "$cpu_usage"
    
    # Memory usage
    local mem_total mem_available mem_usage_percent
    mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_usage_percent=$(( (mem_total - mem_available) * 100 / mem_total ))
    write_metric "infrastructure_memory_usage_percent" "$mem_usage_percent"
    
    # Load average
    local load_1m load_5m load_15m
    read -r load_1m load_5m load_15m _ < /proc/loadavg
    write_metric "infrastructure_load_1m" "$load_1m"
    write_metric "infrastructure_load_5m" "$load_5m"
    write_metric "infrastructure_load_15m" "$load_15m"
    
    log "System resources: CPU ${cpu_usage}%, Memory ${mem_usage_percent}%, Load ${load_1m}"
}

# Main health check function
main() {
    log "=== Infrastructure Health Check Started ==="
    
    # Clear previous metrics
    > "$METRICS_FILE"
    
    # Add timestamp metric
    write_metric "infrastructure_health_check_timestamp" "$(date +%s)"
    
    # Check system resources
    check_system_resources
    
    # Check disk space
    check_disk_space "/" 80 90
    check_disk_space "/media" 85 95
    check_disk_space "/downloads" 80 90
    
    # Check core infrastructure containers
    local containers=(
        "traefik"
        "authelia"
        "prometheus"
        "grafana"
        "loki"
        "promtail"
        "uptime-kuma"
        "gluetun"
    )
    
    for container in "${containers[@]}"; do
        check_docker_container "$container"
    done
    
    # Check HTTP endpoints
    check_http_endpoint "Traefik Dashboard" "http://traefik:8080/ping" "200" 10
    check_http_endpoint "Prometheus" "http://prometheus:9090/-/healthy" "200" 10
    check_http_endpoint "Grafana" "http://grafana:3000/api/health" "200" 10
    check_http_endpoint "Loki" "http://loki:3100/ready" "200" 10
    check_http_endpoint "Uptime Kuma" "http://uptime-kuma:3001" "200" 15
    check_http_endpoint "Authelia" "http://authelia:9091/api/health" "200" 10
    
    # Check TCP services
    check_tcp_port "Redis" "redis" "6379" 5
    check_tcp_port "PostgreSQL" "postgres" "5432" 5
    
    # Check VPN connectivity
    if docker ps --format "table {{.Names}}" | grep -q "gluetun"; then
        check_http_endpoint "VPN Status" "http://gluetun:8000" "200" 30
    fi
    
    # Add health check completion metric
    write_metric "infrastructure_health_check_completed" "1"
    
    log "=== Infrastructure Health Check Completed ==="
    
    # Return status
    if [[ -f "$METRICS_FILE" ]]; then
        echo '{"status":"success","message":"Health check completed","metrics_file":"'$METRICS_FILE'"}'
    else
        echo '{"status":"error","message":"Health check failed"}'
        exit 1
    fi
}

# Error handling
trap 'log "ERROR: Health check script failed at line $LINENO"' ERR

# Execute main function
main "$@"