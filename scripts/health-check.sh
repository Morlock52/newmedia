#!/bin/bash

# HoloMedia Hub Health Check Script
# Version: 1.0.0
# Description: Comprehensive system health monitoring

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Health check configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HEALTH_LOG="$PROJECT_ROOT/logs/health-check.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Service endpoints
BACKEND_URL="${BACKEND_URL:-http://localhost:3000}"
POSTGRES_HOST="${DB_HOST:-localhost}"
POSTGRES_PORT="${DB_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"
MINIO_URL="${MINIO_URL:-http://localhost:9000}"
RABBITMQ_MGMT_URL="${RABBITMQ_MGMT_URL:-http://localhost:15672}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3001}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Health status tracking
declare -A SERVICE_STATUS
TOTAL_CHECKS=0
PASSED_CHECKS=0
WARNINGS=0
FAILURES=0

# ASCII Art Banner
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    __  __           ____  __       ________              __  
   / / / /__  ____ _/ / /_/ /_     / ____/ /_  ___  _____/ /__
  / /_/ / _ \/ __ `/ / __/ __ \   / /   / __ \/ _ \/ ___/ //_/
 / __  /  __/ /_/ / / /_/ / / /  / /___/ / / /  __/ /__/ ,<   
/_/ /_/\___/\__,_/_/\__/_/ /_/   \____/_/ /_/\___/\___/_/|_|  
                                                               
EOF
    echo -e "${NC}"
}

# Logging function
log() {
    echo "[$TIMESTAMP] $1" >> "$HEALTH_LOG"
}

# Status indicators
status_icon() {
    case $1 in
        "ok") echo -e "${GREEN}✓${NC}" ;;
        "warning") echo -e "${YELLOW}⚠${NC}" ;;
        "error") echo -e "${RED}✗${NC}" ;;
        "unknown") echo -e "${BLUE}?${NC}" ;;
    esac
}

# Check result handler
check_result() {
    local service=$1
    local status=$2
    local message=$3
    
    ((TOTAL_CHECKS++))
    SERVICE_STATUS[$service]=$status
    
    case $status in
        "ok")
            ((PASSED_CHECKS++))
            echo -e "$(status_icon ok) ${GREEN}$service${NC}: $message"
            log "OK: $service - $message"
            ;;
        "warning")
            ((WARNINGS++))
            echo -e "$(status_icon warning) ${YELLOW}$service${NC}: $message"
            log "WARNING: $service - $message"
            ;;
        "error")
            ((FAILURES++))
            echo -e "$(status_icon error) ${RED}$service${NC}: $message"
            log "ERROR: $service - $message"
            ;;
    esac
}

# System resource check
check_system_resources() {
    echo -e "\n${CYAN}━━━ System Resources ━━━${NC}"
    
    # CPU usage
    if command -v top &> /dev/null; then
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        if (( $(echo "$CPU_USAGE < 80" | bc -l) )); then
            check_result "CPU Usage" "ok" "${CPU_USAGE}% (Normal)"
        elif (( $(echo "$CPU_USAGE < 90" | bc -l) )); then
            check_result "CPU Usage" "warning" "${CPU_USAGE}% (High)"
        else
            check_result "CPU Usage" "error" "${CPU_USAGE}% (Critical)"
        fi
    fi
    
    # Memory usage
    if command -v free &> /dev/null; then
        MEM_TOTAL=$(free -m | awk 'NR==2{print $2}')
        MEM_USED=$(free -m | awk 'NR==2{print $3}')
        MEM_PERCENT=$((MEM_USED * 100 / MEM_TOTAL))
        
        if [ $MEM_PERCENT -lt 80 ]; then
            check_result "Memory Usage" "ok" "${MEM_PERCENT}% (${MEM_USED}MB/${MEM_TOTAL}MB)"
        elif [ $MEM_PERCENT -lt 90 ]; then
            check_result "Memory Usage" "warning" "${MEM_PERCENT}% (${MEM_USED}MB/${MEM_TOTAL}MB)"
        else
            check_result "Memory Usage" "error" "${MEM_PERCENT}% (${MEM_USED}MB/${MEM_TOTAL}MB)"
        fi
    fi
    
    # Disk usage
    DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2{print $5}' | sed 's/%//')
    DISK_AVAIL=$(df -h "$PROJECT_ROOT" | awk 'NR==2{print $4}')
    
    if [ $DISK_USAGE -lt 80 ]; then
        check_result "Disk Usage" "ok" "${DISK_USAGE}% used, ${DISK_AVAIL} available"
    elif [ $DISK_USAGE -lt 90 ]; then
        check_result "Disk Usage" "warning" "${DISK_USAGE}% used, ${DISK_AVAIL} available"
    else
        check_result "Disk Usage" "error" "${DISK_USAGE}% used, ${DISK_AVAIL} available"
    fi
}

# Docker containers check
check_docker_containers() {
    echo -e "\n${CYAN}━━━ Docker Containers ━━━${NC}"
    
    if ! command -v docker &> /dev/null; then
        check_result "Docker" "error" "Docker not installed"
        return
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        check_result "Docker Daemon" "error" "Not running"
        return
    fi
    
    check_result "Docker Daemon" "ok" "Running"
    
    # Check individual containers
    local containers=("holomedia-postgres" "holomedia-redis" "holomedia-backend" "holomedia-nginx" "holomedia-elasticsearch" "holomedia-minio" "holomedia-rabbitmq")
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
            # Get container health status
            HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
            
            case $HEALTH in
                "healthy")
                    check_result "Container: $container" "ok" "Running (Healthy)"
                    ;;
                "unhealthy")
                    check_result "Container: $container" "error" "Running (Unhealthy)"
                    ;;
                "starting")
                    check_result "Container: $container" "warning" "Starting"
                    ;;
                "none")
                    check_result "Container: $container" "ok" "Running (No health check)"
                    ;;
                *)
                    check_result "Container: $container" "warning" "Running (Status: $HEALTH)"
                    ;;
            esac
        else
            check_result "Container: $container" "error" "Not running"
        fi
    done
}

# Backend API check
check_backend_api() {
    echo -e "\n${CYAN}━━━ Backend API ━━━${NC}"
    
    # Check health endpoint
    if curl -sf "${BACKEND_URL}/health" -o /dev/null; then
        RESPONSE=$(curl -s "${BACKEND_URL}/health")
        check_result "Backend API" "ok" "Responding at ${BACKEND_URL}"
        
        # Check API version
        VERSION=$(echo "$RESPONSE" | jq -r '.version // "unknown"' 2>/dev/null || echo "unknown")
        if [ "$VERSION" != "unknown" ]; then
            check_result "API Version" "ok" "$VERSION"
        fi
    else
        check_result "Backend API" "error" "Not responding at ${BACKEND_URL}"
    fi
}

# Database check
check_database() {
    echo -e "\n${CYAN}━━━ Database ━━━${NC}"
    
    # PostgreSQL check
    if command -v pg_isready &> /dev/null; then
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" &> /dev/null; then
            check_result "PostgreSQL" "ok" "Accepting connections on ${POSTGRES_HOST}:${POSTGRES_PORT}"
            
            # Check database size if credentials are available
            if [ -n "$DB_PASSWORD" ]; then
                DB_SIZE=$(PGPASSWORD="$DB_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "${DB_USER:-holomedia}" -d "${DB_NAME:-holomedia}" -t -c "SELECT pg_size_pretty(pg_database_size('${DB_NAME:-holomedia}'));" 2>/dev/null | tr -d ' ')
                if [ -n "$DB_SIZE" ]; then
                    check_result "Database Size" "ok" "$DB_SIZE"
                fi
            fi
        else
            check_result "PostgreSQL" "error" "Not accepting connections on ${POSTGRES_HOST}:${POSTGRES_PORT}"
        fi
    else
        # Fallback to netcat
        if nc -z "$POSTGRES_HOST" "$POSTGRES_PORT" 2>/dev/null; then
            check_result "PostgreSQL" "warning" "Port ${POSTGRES_PORT} is open (pg_isready not available)"
        else
            check_result "PostgreSQL" "error" "Port ${POSTGRES_PORT} is not accessible"
        fi
    fi
}

# Redis check
check_redis() {
    echo -e "\n${CYAN}━━━ Redis ━━━${NC}"
    
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
            check_result "Redis" "ok" "Responding on ${REDIS_HOST}:${REDIS_PORT}"
            
            # Get Redis info
            if [ -n "$REDIS_PASSWORD" ]; then
                REDIS_INFO=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO server 2>/dev/null | grep redis_version | cut -d: -f2 | tr -d '\r')
                if [ -n "$REDIS_INFO" ]; then
                    check_result "Redis Version" "ok" "$REDIS_INFO"
                fi
            fi
        else
            check_result "Redis" "error" "Not responding on ${REDIS_HOST}:${REDIS_PORT}"
        fi
    else
        # Fallback to netcat
        if nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; then
            check_result "Redis" "warning" "Port ${REDIS_PORT} is open (redis-cli not available)"
        else
            check_result "Redis" "error" "Port ${REDIS_PORT} is not accessible"
        fi
    fi
}

# Elasticsearch check
check_elasticsearch() {
    echo -e "\n${CYAN}━━━ Elasticsearch ━━━${NC}"
    
    if curl -sf "${ELASTICSEARCH_URL}/_cluster/health" -o /dev/null; then
        HEALTH=$(curl -s "${ELASTICSEARCH_URL}/_cluster/health" | jq -r '.status' 2>/dev/null || echo "unknown")
        
        case $HEALTH in
            "green")
                check_result "Elasticsearch" "ok" "Cluster is healthy (Green)"
                ;;
            "yellow")
                check_result "Elasticsearch" "warning" "Cluster is operational (Yellow)"
                ;;
            "red")
                check_result "Elasticsearch" "error" "Cluster is unhealthy (Red)"
                ;;
            *)
                check_result "Elasticsearch" "warning" "Status unknown"
                ;;
        esac
    else
        check_result "Elasticsearch" "error" "Not responding at ${ELASTICSEARCH_URL}"
    fi
}

# MinIO check
check_minio() {
    echo -e "\n${CYAN}━━━ MinIO (Object Storage) ━━━${NC}"
    
    if curl -sf "${MINIO_URL}/minio/health/live" -o /dev/null; then
        check_result "MinIO" "ok" "Service is healthy"
    else
        check_result "MinIO" "error" "Not responding at ${MINIO_URL}"
    fi
}

# RabbitMQ check
check_rabbitmq() {
    echo -e "\n${CYAN}━━━ RabbitMQ ━━━${NC}"
    
    if curl -sf "${RABBITMQ_MGMT_URL}/api/overview" -u "${RABBITMQ_USER:-holomedia}:${RABBITMQ_PASSWORD:-changeme}" -o /dev/null; then
        check_result "RabbitMQ" "ok" "Management API responding"
    else
        check_result "RabbitMQ" "error" "Management API not responding at ${RABBITMQ_MGMT_URL}"
    fi
}

# SSL Certificate check
check_ssl_certificates() {
    echo -e "\n${CYAN}━━━ SSL Certificates ━━━${NC}"
    
    SSL_DIR="$PROJECT_ROOT/ssl"
    
    if [ -d "$SSL_DIR" ]; then
        # Check for certificate files
        if [ -f "$SSL_DIR/cert.pem" ] && [ -f "$SSL_DIR/key.pem" ]; then
            # Check certificate expiry
            if command -v openssl &> /dev/null; then
                CERT_EXPIRY=$(openssl x509 -enddate -noout -in "$SSL_DIR/cert.pem" 2>/dev/null | cut -d= -f2)
                if [ -n "$CERT_EXPIRY" ]; then
                    EXPIRY_EPOCH=$(date -d "$CERT_EXPIRY" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$CERT_EXPIRY" +%s 2>/dev/null)
                    CURRENT_EPOCH=$(date +%s)
                    DAYS_LEFT=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))
                    
                    if [ $DAYS_LEFT -gt 30 ]; then
                        check_result "SSL Certificate" "ok" "Valid for $DAYS_LEFT days"
                    elif [ $DAYS_LEFT -gt 7 ]; then
                        check_result "SSL Certificate" "warning" "Expires in $DAYS_LEFT days"
                    else
                        check_result "SSL Certificate" "error" "Expires in $DAYS_LEFT days!"
                    fi
                else
                    check_result "SSL Certificate" "warning" "Found but unable to check expiry"
                fi
            else
                check_result "SSL Certificate" "warning" "Found but openssl not available for validation"
            fi
        else
            check_result "SSL Certificate" "warning" "Not configured"
        fi
    else
        check_result "SSL Certificate" "warning" "SSL directory not found"
    fi
}

# Monitoring services check
check_monitoring() {
    echo -e "\n${CYAN}━━━ Monitoring Services ━━━${NC}"
    
    # Grafana
    if curl -sf "${GRAFANA_URL}/api/health" -o /dev/null; then
        check_result "Grafana" "ok" "Dashboard available at ${GRAFANA_URL}"
    else
        check_result "Grafana" "warning" "Not responding at ${GRAFANA_URL}"
    fi
    
    # Prometheus
    if curl -sf "${PROMETHEUS_URL}/-/healthy" -o /dev/null; then
        check_result "Prometheus" "ok" "Metrics collection active"
    else
        check_result "Prometheus" "warning" "Not responding at ${PROMETHEUS_URL}"
    fi
}

# Application logs check
check_logs() {
    echo -e "\n${CYAN}━━━ Application Logs ━━━${NC}"
    
    LOG_DIR="$PROJECT_ROOT/logs"
    
    if [ -d "$LOG_DIR" ]; then
        # Check for recent errors
        ERROR_COUNT=$(find "$LOG_DIR" -name "*.log" -mtime -1 -exec grep -i "error" {} \; 2>/dev/null | wc -l)
        WARNING_COUNT=$(find "$LOG_DIR" -name "*.log" -mtime -1 -exec grep -i "warning" {} \; 2>/dev/null | wc -l)
        
        if [ $ERROR_COUNT -eq 0 ]; then
            check_result "Recent Errors" "ok" "No errors in last 24 hours"
        elif [ $ERROR_COUNT -lt 10 ]; then
            check_result "Recent Errors" "warning" "$ERROR_COUNT errors in last 24 hours"
        else
            check_result "Recent Errors" "error" "$ERROR_COUNT errors in last 24 hours"
        fi
        
        if [ $WARNING_COUNT -lt 50 ]; then
            check_result "Recent Warnings" "ok" "$WARNING_COUNT warnings in last 24 hours"
        else
            check_result "Recent Warnings" "warning" "$WARNING_COUNT warnings in last 24 hours"
        fi
        
        # Check log disk usage
        LOG_SIZE=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
        check_result "Log Directory Size" "ok" "$LOG_SIZE"
    else
        check_result "Application Logs" "warning" "Log directory not found"
    fi
}

# Backup status check
check_backups() {
    echo -e "\n${CYAN}━━━ Backup Status ━━━${NC}"
    
    BACKUP_DIR="$PROJECT_ROOT/backups"
    
    if [ -d "$BACKUP_DIR" ]; then
        # Find most recent backup
        LATEST_BACKUP=$(find "$BACKUP_DIR" -name "*.tar.gz" -o -name "*.sql" | sort -r | head -n1)
        
        if [ -n "$LATEST_BACKUP" ]; then
            BACKUP_AGE=$(( ($(date +%s) - $(stat -f %m "$LATEST_BACKUP" 2>/dev/null || stat -c %Y "$LATEST_BACKUP")) / 3600 ))
            
            if [ $BACKUP_AGE -lt 24 ]; then
                check_result "Latest Backup" "ok" "$(basename "$LATEST_BACKUP") ($BACKUP_AGE hours ago)"
            elif [ $BACKUP_AGE -lt 48 ]; then
                check_result "Latest Backup" "warning" "$(basename "$LATEST_BACKUP") ($BACKUP_AGE hours ago)"
            else
                check_result "Latest Backup" "error" "$(basename "$LATEST_BACKUP") ($BACKUP_AGE hours ago)"
            fi
            
            # Check backup count
            BACKUP_COUNT=$(find "$BACKUP_DIR" -name "*.tar.gz" -o -name "*.sql" | wc -l)
            check_result "Total Backups" "ok" "$BACKUP_COUNT backups found"
        else
            check_result "Latest Backup" "error" "No backups found"
        fi
    else
        check_result "Backup Directory" "error" "Not found"
    fi
}

# Security checks
check_security() {
    echo -e "\n${CYAN}━━━ Security Checks ━━━${NC}"
    
    # Check file permissions
    if [ -f "$PROJECT_ROOT/.env" ]; then
        ENV_PERMS=$(stat -c %a "$PROJECT_ROOT/.env" 2>/dev/null || stat -f %p "$PROJECT_ROOT/.env" | cut -c4-6)
        if [ "$ENV_PERMS" = "600" ] || [ "$ENV_PERMS" = "400" ]; then
            check_result "Environment File Permissions" "ok" "Secure ($ENV_PERMS)"
        else
            check_result "Environment File Permissions" "warning" "Should be 600 (current: $ENV_PERMS)"
        fi
    fi
    
    # Check for default passwords
    if [ -f "$PROJECT_ROOT/.env" ]; then
        if grep -q "changeme\|password123\|admin" "$PROJECT_ROOT/.env"; then
            check_result "Default Passwords" "error" "Found in configuration!"
        else
            check_result "Default Passwords" "ok" "None detected"
        fi
    fi
    
    # Check firewall status (Linux only)
    if command -v ufw &> /dev/null; then
        UFW_STATUS=$(ufw status | grep -i "status:" | awk '{print $2}')
        if [ "$UFW_STATUS" = "active" ]; then
            check_result "Firewall (UFW)" "ok" "Active"
        else
            check_result "Firewall (UFW)" "warning" "Inactive"
        fi
    fi
}

# Performance metrics
check_performance() {
    echo -e "\n${CYAN}━━━ Performance Metrics ━━━${NC}"
    
    # Check API response time
    if command -v curl &> /dev/null; then
        RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "${BACKEND_URL}/health" 2>/dev/null)
        if [ -n "$RESPONSE_TIME" ]; then
            # Convert to milliseconds
            RESPONSE_MS=$(echo "$RESPONSE_TIME * 1000" | bc 2>/dev/null || echo "0")
            RESPONSE_MS=${RESPONSE_MS%.*}
            
            if [ "$RESPONSE_MS" -lt 200 ]; then
                check_result "API Response Time" "ok" "${RESPONSE_MS}ms"
            elif [ "$RESPONSE_MS" -lt 1000 ]; then
                check_result "API Response Time" "warning" "${RESPONSE_MS}ms"
            else
                check_result "API Response Time" "error" "${RESPONSE_MS}ms"
            fi
        fi
    fi
    
    # Check database connection pool
    # Add your specific performance checks here
}

# Generate summary report
generate_summary() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}Health Check Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    echo -e "\nTotal Checks: ${WHITE}$TOTAL_CHECKS${NC}"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo -e "Failed: ${RED}$FAILURES${NC}"
    
    # Calculate health score
    if [ $TOTAL_CHECKS -gt 0 ]; then
        HEALTH_SCORE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
        
        echo -ne "\nOverall Health Score: "
        if [ $HEALTH_SCORE -ge 90 ]; then
            echo -e "${GREEN}${HEALTH_SCORE}% - Excellent${NC}"
        elif [ $HEALTH_SCORE -ge 70 ]; then
            echo -e "${YELLOW}${HEALTH_SCORE}% - Good (with warnings)${NC}"
        elif [ $HEALTH_SCORE -ge 50 ]; then
            echo -e "${YELLOW}${HEALTH_SCORE}% - Fair (needs attention)${NC}"
        else
            echo -e "${RED}${HEALTH_SCORE}% - Poor (immediate action required)${NC}"
        fi
    fi
    
    # Recommendations
    if [ $FAILURES -gt 0 ] || [ $WARNINGS -gt 0 ]; then
        echo -e "\n${WHITE}Recommendations:${NC}"
        
        # Check for critical failures
        for service in "${!SERVICE_STATUS[@]}"; do
            if [ "${SERVICE_STATUS[$service]}" = "error" ]; then
                case $service in
                    "Docker Daemon")
                        echo -e "• ${RED}Start Docker service${NC}"
                        ;;
                    "PostgreSQL")
                        echo -e "• ${RED}Check PostgreSQL connection and credentials${NC}"
                        ;;
                    "Backend API")
                        echo -e "• ${RED}Check if backend service is running${NC}"
                        ;;
                    "Latest Backup")
                        echo -e "• ${RED}Run backup immediately${NC}"
                        ;;
                esac
            elif [ "${SERVICE_STATUS[$service]}" = "warning" ]; then
                case $service in
                    "CPU Usage")
                        echo -e "• ${YELLOW}Monitor CPU usage - consider scaling${NC}"
                        ;;
                    "Memory Usage")
                        echo -e "• ${YELLOW}Memory usage is high - check for memory leaks${NC}"
                        ;;
                    "Disk Usage")
                        echo -e "• ${YELLOW}Disk space is running low - clean up old files${NC}"
                        ;;
                    "SSL Certificate")
                        echo -e "• ${YELLOW}SSL certificate needs renewal soon${NC}"
                        ;;
                esac
            fi
        done
    fi
    
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Full report saved to: ${WHITE}$HEALTH_LOG${NC}"
    echo -e "Run with ${WHITE}--verbose${NC} for detailed diagnostics"
}

# Export health data as JSON
export_json() {
    local json_file="$PROJECT_ROOT/logs/health-check-$(date +%Y%m%d-%H%M%S).json"
    
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
    
    echo -e "\nJSON report exported to: ${WHITE}$json_file${NC}"
}

# Main health check flow
main() {
    show_banner
    
    echo -e "${CYAN}Running comprehensive health check...${NC}"
    echo -e "${CYAN}Time: $TIMESTAMP${NC}\n"
    
    # Initialize log
    mkdir -p "$(dirname "$HEALTH_LOG")"
    echo "=== HoloMedia Hub Health Check - $TIMESTAMP ===" > "$HEALTH_LOG"
    
    # Run all checks
    check_system_resources
    check_docker_containers
    check_backend_api
    check_database
    check_redis
    check_elasticsearch
    check_minio
    check_rabbitmq
    check_ssl_certificates
    check_monitoring
    check_logs
    check_backups
    check_security
    check_performance
    
    # Generate summary
    generate_summary
    
    # Export JSON if requested
    if [[ "$1" == "--json" ]]; then
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