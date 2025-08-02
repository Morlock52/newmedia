#!/bin/bash

# Health Check Validator Script - 2025
# Comprehensive health checking for media server Docker stack
# Based on MEDIA_SERVER_INTEGRATION_RESEARCH_2025.md findings

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_DIR}/logs/health-check-$(date +%Y%m%d-%H%M%S).log"
REPORT_FILE="${PROJECT_DIR}/reports/health-report-$(date +%Y%m%d-%H%M%S).json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Arrays to store results
declare -A SERVICE_STATUS
declare -A SERVICE_RESPONSE_TIME
declare -A SERVICE_ERRORS

# Service definitions with health check endpoints
declare -A SERVICES=(
    ["jellyfin"]="http://localhost:8096/health"
    ["plex"]="http://localhost:32400/identity"
    ["emby"]="http://localhost:8097/health"
    ["sonarr"]="http://localhost:8989/ping"
    ["radarr"]="http://localhost:7878/ping"
    ["lidarr"]="http://localhost:8686/ping"
    ["bazarr"]="http://localhost:6767/ping"
    ["prowlarr"]="http://localhost:9696/ping"
    ["jellyseerr"]="http://localhost:5055/api/v1/status"
    ["overseerr"]="http://localhost:5056/api/v1/status"
    ["ombi"]="http://localhost:3579/api/v1/status"
    ["qbittorrent"]="http://localhost:8080"
    ["sabnzbd"]="http://localhost:8081/api?mode=version"
    ["nzbget"]="http://localhost:6789/jsonrpc"
    ["prometheus"]="http://localhost:9090/-/healthy"
    ["grafana"]="http://localhost:3000/api/health"
    ["loki"]="http://localhost:3100/ready"
    ["uptime-kuma"]="http://localhost:3001"
    ["portainer"]="http://localhost:9000/api/status"
    ["nginx-proxy-manager"]="http://localhost:81/api/nginx/proxy-hosts"
    ["postgres"]="localhost:5432"
    ["mariadb"]="localhost:3306"
    ["redis"]="localhost:6379"
)

# Service categories for organized reporting
declare -A SERVICE_CATEGORIES=(
    ["jellyfin"]="Media Servers"
    ["plex"]="Media Servers"
    ["emby"]="Media Servers"
    ["sonarr"]="*ARR Services"
    ["radarr"]="*ARR Services"
    ["lidarr"]="*ARR Services"
    ["bazarr"]="*ARR Services"
    ["prowlarr"]="*ARR Services"
    ["jellyseerr"]="Request Services"
    ["overseerr"]="Request Services"
    ["ombi"]="Request Services"
    ["qbittorrent"]="Download Clients"
    ["sabnzbd"]="Download Clients"
    ["nzbget"]="Download Clients"
    ["prometheus"]="Monitoring"
    ["grafana"]="Monitoring"
    ["loki"]="Monitoring"
    ["uptime-kuma"]="Monitoring"
    ["portainer"]="Management"
    ["nginx-proxy-manager"]="Management"
    ["postgres"]="Databases"
    ["mariadb"]="Databases"
    ["redis"]="Databases"
)

# Initialize logging
init_logging() {
    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$REPORT_FILE")"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Health Check Validator Started" > "$LOG_FILE"
}

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo -e "$1"
}

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                     Media Server Health Check Validator 2025                ║"
    echo "║                          Docker Stack Validation                            ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check if Docker Compose is running
check_docker_compose() {
    log "${BLUE}🐳 Checking Docker Compose status...${NC}"
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log "${RED}❌ Docker not found. Please install Docker.${NC}"
        exit 1
    fi
    
    # Check if docker-compose.yml exists
    if [[ ! -f "${PROJECT_DIR}/docker-compose.yml" ]]; then
        log "${RED}❌ docker-compose.yml not found in ${PROJECT_DIR}${NC}"
        exit 1
    fi
    
    # Check if any containers are running
    local running_containers=$(docker ps --format "table {{.Names}}" | grep -v NAMES | wc -l)
    if [[ $running_containers -eq 0 ]]; then
        log "${YELLOW}⚠️  No Docker containers are currently running${NC}"
        return 1
    fi
    
    log "${GREEN}✅ Docker Compose environment detected with $running_containers running containers${NC}"
    return 0
}

# Test HTTP service health
test_http_service() {
    local service_name=$1
    local url=$2
    local timeout=${3:-10}
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    # Measure response time
    local start_time=$(date +%s%N)
    
    if curl -f -s -m "$timeout" "$url" >/dev/null 2>&1; then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        SERVICE_STATUS[$service_name]="HEALTHY"
        SERVICE_RESPONSE_TIME[$service_name]=$response_time
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        
        local status_icon="✅"
        if [[ $response_time -gt 5000 ]]; then
            status_icon="🐌"
        elif [[ $response_time -gt 2000 ]]; then
            status_icon="⚡"
        fi
        
        log "  $status_icon $service_name: HEALTHY (${response_time}ms)"
    else
        SERVICE_STATUS[$service_name]="UNHEALTHY"
        SERVICE_RESPONSE_TIME[$service_name]=0
        SERVICE_ERRORS[$service_name]="HTTP request failed"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        log "  ❌ $service_name: UNHEALTHY (HTTP request failed)"
    fi
}

# Test database service health
test_database_service() {
    local service_name=$1
    local connection_string=$2
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    case $service_name in
        "postgres")
            if docker exec postgres pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
                SERVICE_STATUS[$service_name]="HEALTHY"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                log "  ✅ $service_name: HEALTHY"
            else
                SERVICE_STATUS[$service_name]="UNHEALTHY"
                SERVICE_ERRORS[$service_name]="PostgreSQL not ready"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                log "  ❌ $service_name: UNHEALTHY (PostgreSQL not ready)"
            fi
            ;;
        "mariadb")
            if docker exec mariadb mysqladmin ping -h localhost >/dev/null 2>&1; then
                SERVICE_STATUS[$service_name]="HEALTHY"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                log "  ✅ $service_name: HEALTHY"
            else
                SERVICE_STATUS[$service_name]="UNHEALTHY"
                SERVICE_ERRORS[$service_name]="MariaDB not responding"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                log "  ❌ $service_name: UNHEALTHY (MariaDB not responding)"
            fi
            ;;
        "redis")
            if docker exec redis redis-cli ping >/dev/null 2>&1; then
                SERVICE_STATUS[$service_name]="HEALTHY"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                log "  ✅ $service_name: HEALTHY"
            else
                SERVICE_STATUS[$service_name]="UNHEALTHY"
                SERVICE_ERRORS[$service_name]="Redis not responding to ping"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                log "  ❌ $service_name: UNHEALTHY (Redis not responding)"
            fi
            ;;
    esac
}

# Test container health using Docker health checks
test_container_health() {
    local service_name=$1
    
    # Check if container exists and is running
    if ! docker ps --format "{{.Names}}" | grep -q "^${service_name}$"; then
        SERVICE_STATUS[$service_name]="NOT_RUNNING"
        SERVICE_ERRORS[$service_name]="Container not running"
        return
    fi
    
    # Check Docker health status if available
    local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$service_name" 2>/dev/null || echo "unknown")
    
    if [[ "$health_status" == "healthy" ]]; then
        log "  🔍 $service_name: Docker health check PASSED"
    elif [[ "$health_status" == "unhealthy" ]]; then
        log "  ⚠️  $service_name: Docker health check FAILED"
        SERVICE_ERRORS[$service_name]="${SERVICE_ERRORS[$service_name]} | Docker health check failed"
    fi
}

# Run comprehensive health checks
run_health_checks() {
    log "${PURPLE}🔍 Running comprehensive health checks...${NC}"
    
    # Test HTTP services
    for service in "${!SERVICES[@]}"; do
        local url="${SERVICES[$service]}"
        local category="${SERVICE_CATEGORIES[$service]}"
        
        case $category in
            "Databases")
                test_database_service "$service" "$url"
                ;;
            *)
                test_http_service "$service" "$url"
                ;;
        esac
        
        # Also test container health
        test_container_health "$service"
    done
}

# Check API integrations
test_api_integrations() {
    log "${PURPLE}🔗 Testing API integrations...${NC}"
    
    # Test if services can communicate with each other
    local integration_tests=(
        "sonarr_prowlarr:Test Sonarr can reach Prowlarr"
        "radarr_prowlarr:Test Radarr can reach Prowlarr"
        "jellyseerr_jellyfin:Test Jellyseerr can reach Jellyfin"
    )
    
    for integration in "${integration_tests[@]}"; do
        local test_name=$(echo "$integration" | cut -d: -f1)
        local description=$(echo "$integration" | cut -d: -f2)
        
        case $test_name in
            "sonarr_prowlarr")
                if [[ "${SERVICE_STATUS[sonarr]}" == "HEALTHY" ]] && [[ "${SERVICE_STATUS[prowlarr]}" == "HEALTHY" ]]; then
                    log "  ✅ $description: Both services healthy"
                else
                    log "  ❌ $description: One or both services unhealthy"
                fi
                ;;
            "radarr_prowlarr")
                if [[ "${SERVICE_STATUS[radarr]}" == "HEALTHY" ]] && [[ "${SERVICE_STATUS[prowlarr]}" == "HEALTHY" ]]; then
                    log "  ✅ $description: Both services healthy"
                else
                    log "  ❌ $description: One or both services unhealthy"
                fi
                ;;
            "jellyseerr_jellyfin")
                if [[ "${SERVICE_STATUS[jellyseerr]}" == "HEALTHY" ]] && [[ "${SERVICE_STATUS[jellyfin]}" == "HEALTHY" ]]; then
                    log "  ✅ $description: Both services healthy"
                else
                    log "  ❌ $description: One or both services unhealthy"
                fi
                ;;
        esac
    done
}

# Check Docker network connectivity
test_network_connectivity() {
    log "${PURPLE}🌐 Testing Docker network connectivity...${NC}"
    
    # Test if containers can resolve each other by name
    local network_tests=(
        "jellyfin:sonarr"
        "sonarr:prowlarr"
        "radarr:prowlarr"
        "jellyseerr:jellyfin"
    )
    
    for test in "${network_tests[@]}"; do
        local from_container=$(echo "$test" | cut -d: -f1)
        local to_container=$(echo "$test" | cut -d: -f2)
        
        if docker exec "$from_container" nslookup "$to_container" >/dev/null 2>&1; then
            log "  ✅ Network: $from_container can resolve $to_container"
        else
            log "  ❌ Network: $from_container cannot resolve $to_container"
        fi
    done
}

# Generate performance recommendations
generate_recommendations() {
    local recommendations=()
    
    # Check response times
    for service in "${!SERVICE_RESPONSE_TIME[@]}"; do
        local response_time=${SERVICE_RESPONSE_TIME[$service]}
        if [[ $response_time -gt 5000 ]]; then
            recommendations+=("Optimize $service performance (${response_time}ms response time)")
        fi
    done
    
    # Check failed services
    for service in "${!SERVICE_STATUS[@]}"; do
        if [[ "${SERVICE_STATUS[$service]}" != "HEALTHY" ]]; then
            recommendations+=("Fix $service service (currently ${SERVICE_STATUS[$service]})")
        fi
    done
    
    # Check missing health checks
    local missing_health_checks=$(docker ps --format "{{.Names}}" | while read container; do
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
        if [[ "$health_status" == "none" ]]; then
            echo "$container"
        fi
    done)
    
    if [[ -n "$missing_health_checks" ]]; then
        recommendations+=("Add health checks to containers: $(echo "$missing_health_checks" | tr '\n' ' ')")
    fi
    
    printf '%s\n' "${recommendations[@]}"
}

# Generate detailed JSON report
generate_json_report() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local success_rate=$(echo "scale=1; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc -l 2>/dev/null || echo "0")
    
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$timestamp",
  "summary": {
    "total_checks": $TOTAL_CHECKS,
    "passed": $PASSED_CHECKS,
    "failed": $FAILED_CHECKS,
    "success_rate": "${success_rate}%"
  },
  "services": {
EOF

    local first=true
    for service in "${!SERVICE_STATUS[@]}"; do
        if [[ $first == true ]]; then
            first=false
        else
            echo "," >> "$REPORT_FILE"
        fi
        
        cat >> "$REPORT_FILE" << EOF
    "$service": {
      "status": "${SERVICE_STATUS[$service]}",
      "category": "${SERVICE_CATEGORIES[$service]}",
      "response_time_ms": ${SERVICE_RESPONSE_TIME[$service]:-0},
      "error": "${SERVICE_ERRORS[$service]:-null}"
    }
EOF
    done

    cat >> "$REPORT_FILE" << EOF
  },
  "recommendations": [
EOF

    local recommendations=($(generate_recommendations))
    for i in "${!recommendations[@]}"; do
        if [[ $i -gt 0 ]]; then
            echo "," >> "$REPORT_FILE"
        fi
        echo "    \"${recommendations[$i]}\"" >> "$REPORT_FILE"
    done

    cat >> "$REPORT_FILE" << EOF
  ]
}
EOF

    log "${GREEN}📊 JSON report generated: $REPORT_FILE${NC}"
}

# Print summary report
print_summary() {
    local success_rate=$(echo "scale=1; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc -l 2>/dev/null || echo "0")
    
    echo
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                              HEALTH CHECK SUMMARY                           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BLUE}📊 Overall Statistics:${NC}"
    echo -e "   Total Checks: $TOTAL_CHECKS"
    echo -e "   ${GREEN}Passed: $PASSED_CHECKS${NC}"
    echo -e "   ${RED}Failed: $FAILED_CHECKS${NC}"
    echo -e "   Success Rate: ${success_rate}%"
    echo
    
    # Print by category
    local categories=($(printf '%s\n' "${SERVICE_CATEGORIES[@]}" | sort -u))
    
    for category in "${categories[@]}"; do
        echo -e "${PURPLE}$category:${NC}"
        for service in "${!SERVICE_CATEGORIES[@]}"; do
            if [[ "${SERVICE_CATEGORIES[$service]}" == "$category" ]]; then
                local status="${SERVICE_STATUS[$service]}"
                local response_time="${SERVICE_RESPONSE_TIME[$service]:-0}"
                
                case $status in
                    "HEALTHY")
                        echo -e "  ✅ $service (${response_time}ms)"
                        ;;
                    "UNHEALTHY")
                        echo -e "  ❌ $service - ${SERVICE_ERRORS[$service]}"
                        ;;
                    "NOT_RUNNING")
                        echo -e "  🔴 $service - Container not running"
                        ;;
                    *)
                        echo -e "  ❓ $service - Unknown status"
                        ;;
                esac
            fi
        done
        echo
    done
    
    # Print recommendations
    local recommendations=($(generate_recommendations))
    if [[ ${#recommendations[@]} -gt 0 ]]; then
        echo -e "${YELLOW}💡 Recommendations:${NC}"
        for rec in "${recommendations[@]}"; do
            echo -e "   • $rec"
        done
        echo
    fi
    
    echo -e "${BLUE}📋 Reports generated:${NC}"
    echo -e "   Log: $LOG_FILE"
    echo -e "   JSON: $REPORT_FILE"
}

# Main execution function
main() {
    print_banner
    init_logging
    
    # Check prerequisites
    if ! check_docker_compose; then
        log "${RED}❌ Docker Compose environment check failed${NC}"
        exit 1
    fi
    
    # Run all health checks
    run_health_checks
    test_api_integrations
    test_network_connectivity
    
    # Generate reports
    generate_json_report
    print_summary
    
    # Exit with appropriate code
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        log "${RED}❌ Health check completed with failures${NC}"
        exit 1
    else
        log "${GREEN}✅ All health checks passed successfully${NC}"
        exit 0
    fi
}

# Run main function
main "$@"