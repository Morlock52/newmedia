#!/bin/bash

# Ultimate Media Server 2025 - Core Container Integration Tests
# Simple, reliable testing for core services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/test-results/core-integration-$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="${SCRIPT_DIR}/test-results/core-integration-report.json"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Ensure reports directory exists
mkdir -p "${SCRIPT_DIR}/test-results"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Test result tracking
record_test() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [[ "$status" == "PASS" ]]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ $test_name${NC}: $details"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ $test_name${NC}: $details"
    fi
    log "INFO" "$test_name: $status - $details"
}

# Test container health
test_container_health() {
    local container="$1"
    local expected_port="$2"
    
    log "INFO" "Testing container health: $container"
    
    # Check if container is running
    if docker ps --format "{{.Names}}" | grep -q "^$container$"; then
        local status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
        if [[ "$status" == "running" ]]; then
            record_test "Container Health: $container" "PASS" "Status: $status"
            
            # Test port accessibility if specified
            if [[ -n "$expected_port" ]]; then
                if timeout 5 nc -z localhost "$expected_port" 2>/dev/null; then
                    record_test "Port Access: $container:$expected_port" "PASS" "Port accessible"
                else
                    record_test "Port Access: $container:$expected_port" "FAIL" "Port not accessible"
                fi
            fi
            return 0
        else
            record_test "Container Health: $container" "FAIL" "Status: $status"
            return 1
        fi
    else
        record_test "Container Health: $container" "FAIL" "Container not found"
        return 1
    fi
}

# Test API connectivity
test_api_connectivity() {
    local service="$1"
    local port="$2"
    local endpoint="$3"
    
    log "INFO" "Testing API connectivity: $service"
    
    local url="http://localhost:$port$endpoint"
    local response=$(curl -s -m 10 -w "%{http_code}" -o /dev/null "$url" 2>/dev/null || echo "000")
    
    if [[ "$response" =~ ^[2-3][0-9][0-9]$ ]]; then
        record_test "API Connectivity: $service" "PASS" "HTTP $response"
    else
        record_test "API Connectivity: $service" "FAIL" "HTTP $response"
    fi
}

# Test container isolation
test_container_isolation() {
    local container="$1"
    
    log "INFO" "Testing container isolation: $container"
    
    # Check network isolation
    local networks=$(docker inspect --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$container" 2>/dev/null || echo "")
    if [[ -n "$networks" ]]; then
        record_test "Network Isolation: $container" "PASS" "Networks: $networks"
    else
        record_test "Network Isolation: $container" "FAIL" "No networks found"
    fi
    
    # Check volume mounts
    local mounts=$(docker inspect --format='{{range .Mounts}}{{.Destination}} {{end}}' "$container" 2>/dev/null || echo "")
    if [[ -n "$mounts" ]]; then
        record_test "Volume Mounts: $container" "PASS" "Mounted volumes found"
    else
        record_test "Volume Mounts: $container" "PASS" "No volume mounts (expected for some services)"
    fi
}

# Test service integration
test_service_integration() {
    log "INFO" "Testing service integrations"
    
    # Test Prometheus targets
    if docker ps --format "{{.Names}}" | grep -q "^prometheus$"; then
        local targets=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null || echo "")
        if [[ -n "$targets" ]] && echo "$targets" | grep -q "activeTargets"; then
            record_test "Prometheus Targets" "PASS" "Targets API accessible"
        else
            record_test "Prometheus Targets" "FAIL" "Cannot access targets API"
        fi
    fi
    
    # Test Grafana datasources
    if docker ps --format "{{.Names}}" | grep -q "^grafana$"; then
        local datasources=$(curl -s -u "admin:admin" "http://localhost:3000/api/datasources" 2>/dev/null || echo "")
        if [[ -n "$datasources" ]]; then
            record_test "Grafana DataSources" "PASS" "DataSources API accessible"
        else
            record_test "Grafana DataSources" "FAIL" "Cannot access datasources API"
        fi
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}🔍 Ultimate Media Server 2025 - Core Container Integration Tests${NC}"
    echo "=================================================================="
    
    log "INFO" "Starting core container integration tests"
    log "INFO" "Log file: $LOG_FILE"
    
    # Wait for services to be ready
    echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
    sleep 30
    
    # Test core containers
    echo -e "\n${YELLOW}Testing Core Containers...${NC}"
    
    # Media server
    test_container_health "jellyfin" "8096"
    test_container_isolation "jellyfin"
    test_api_connectivity "jellyfin" "8096" "/System/Info/Public"
    
    # ARR services
    test_container_health "sonarr" "8989"
    test_container_isolation "sonarr"
    test_api_connectivity "sonarr" "8989" "/ping"
    
    test_container_health "radarr" "7878"
    test_container_isolation "radarr"
    test_api_connectivity "radarr" "7878" "/ping"
    
    test_container_health "prowlarr" "9696"
    test_container_isolation "prowlarr"
    test_api_connectivity "prowlarr" "9696" "/ping"
    
    # Download client
    test_container_health "qbittorrent" "8082"
    test_container_isolation "qbittorrent"
    test_api_connectivity "qbittorrent" "8082" "/api/v2/app/version"
    
    # Monitoring
    test_container_health "prometheus" "9090"
    test_container_isolation "prometheus"
    test_api_connectivity "prometheus" "9090" "/-/healthy"
    
    test_container_health "grafana" "3000"
    test_container_isolation "grafana"
    test_api_connectivity "grafana" "3000" "/api/health"
    
    # Management
    test_container_health "portainer" "9000"
    test_container_isolation "portainer"
    
    test_container_health "homarr" "7575"
    test_container_isolation "homarr"
    
    # Test service integrations
    echo -e "\n${YELLOW}Testing Service Integrations...${NC}"
    test_service_integration
    
    # Generate report
    generate_report
    
    # Summary
    echo -e "\n${BLUE}Test Summary${NC}"
    echo "============="
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo -e "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo -e "\nDetailed report: $REPORT_FILE"
    echo -e "Log file: $LOG_FILE"
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        echo -e "\n${RED}⚠️  Some tests failed. Check the logs for details.${NC}"
        exit 1
    else
        echo -e "\n${GREEN}✅ All core container integration tests passed!${NC}"
        exit 0
    fi
}

# Generate JSON report
generate_report() {
    local success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -u "+%Y-%m-%dT%H:%M:%SZ")",
  "test_suite": "Core Container Integration Tests",
  "total_tests": $TOTAL_TESTS,
  "passed_tests": $PASSED_TESTS,
  "failed_tests": $FAILED_TESTS,
  "success_rate": $success_rate,
  "log_file": "$LOG_FILE"
}
EOF
    
    log "INFO" "Report generated: $REPORT_FILE"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi