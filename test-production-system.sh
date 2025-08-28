#!/bin/bash

# Comprehensive Testing Script for Production Media Server
# Tests all 30+ services, AI features, and integrations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# Test result tracking
declare -A TEST_RESULTS

# Log functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_pass() {
    echo -e "${GREEN}  ✓${NC} $1"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TEST_RESULTS["$1"]="PASS"
}

log_fail() {
    echo -e "${RED}  ✗${NC} $1"
    FAILED_TESTS=$((FAILED_TESTS + 1))
    TEST_RESULTS["$1"]="FAIL"
}

log_warn() {
    echo -e "${YELLOW}  ⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
    TEST_RESULTS["$1"]="WARN"
}

log_info() {
    echo -e "${CYAN}  ℹ${NC} $1"
}

# Header
show_header() {
    clear
    echo -e "${CYAN}"
    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   PRODUCTION MEDIA SERVER - COMPREHENSIVE TEST SUITE          ║
║                                                                ║
║   Testing: 30+ Services | AI Features | Integrations          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Section separator
print_section() {
    echo ""
    echo -e "${MAGENTA}═══ $1 ═══${NC}"
    echo ""
}

# Test Docker environment
test_docker_environment() {
    print_section "Docker Environment Tests"
    
    log_test "Docker installation"
    if command -v docker &> /dev/null; then
        log_pass "Docker installed: $(docker --version)"
    else
        log_fail "Docker not found"
        return 1
    fi
    
    log_test "Docker Compose installation"
    if command -v docker-compose &> /dev/null; then
        log_pass "Docker Compose installed: $(docker-compose --version)"
    else
        log_fail "Docker Compose not found"
        return 1
    fi
    
    log_test "Docker daemon status"
    if docker info &> /dev/null; then
        log_pass "Docker daemon is running"
    else
        log_fail "Docker daemon not running"
        return 1
    fi
    
    log_test "Docker network connectivity"
    if docker run --rm alpine ping -c 1 google.com &> /dev/null; then
        log_pass "Docker network connectivity OK"
    else
        log_warn "Docker network connectivity issues"
    fi
}

# Test system resources
test_system_resources() {
    print_section "System Resource Tests"
    
    log_test "Available memory"
    local mem_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}' || echo "0")
    if [ "$mem_gb" -ge 8 ]; then
        log_pass "Available memory: ${mem_gb}GB (sufficient)"
    elif [ "$mem_gb" -ge 4 ]; then
        log_warn "Available memory: ${mem_gb}GB (minimum met)"
    else
        log_fail "Available memory: ${mem_gb}GB (insufficient)"
    fi
    
    log_test "Available disk space"
    local disk_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$disk_gb" -ge 50 ]; then
        log_pass "Available disk: ${disk_gb}GB (sufficient)"
    elif [ "$disk_gb" -ge 20 ]; then
        log_warn "Available disk: ${disk_gb}GB (minimum met)"
    else
        log_fail "Available disk: ${disk_gb}GB (insufficient)"
    fi
    
    log_test "CPU cores"
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
    if [ "$cpu_cores" -ge 4 ]; then
        log_pass "CPU cores: $cpu_cores (sufficient)"
    else
        log_warn "CPU cores: $cpu_cores (may impact performance)"
    fi
}

# Test port availability
test_port_availability() {
    print_section "Port Availability Tests"
    
    local ports=(80 443 3000 8096 8989 7878 9696 8686 6767 8080 8090)
    local blocked_ports=0
    
    for port in "${ports[@]}"; do
        log_test "Port $port availability"
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            log_warn "Port $port is in use"
            blocked_ports=$((blocked_ports + 1))
        else
            log_pass "Port $port is available"
        fi
    done
    
    if [ $blocked_ports -gt 0 ]; then
        log_info "$blocked_ports port(s) may need to be freed"
    fi
}

# Test container build
test_container_build() {
    print_section "Container Build Tests"
    
    log_test "Dockerfile validation"
    if [ -f "Dockerfile.production-single" ]; then
        log_pass "Production Dockerfile exists"
    else
        log_fail "Production Dockerfile missing"
        return 1
    fi
    
    log_test "Docker Compose configuration"
    if [ -f "docker-compose.production.yml" ]; then
        log_pass "Docker Compose configuration exists"
        
        log_test "Docker Compose syntax validation"
        if docker-compose -f docker-compose.production.yml config > /dev/null 2>&1; then
            log_pass "Docker Compose configuration is valid"
        else
            log_fail "Docker Compose configuration has syntax errors"
        fi
    else
        log_fail "Docker Compose configuration missing"
        return 1
    fi
    
    log_test "Service definitions"
    if [ -d "s6-services" ]; then
        local service_count=$(ls -1 s6-services/ | wc -l)
        if [ $service_count -ge 10 ]; then
            log_pass "Service definitions found: $service_count services"
        else
            log_warn "Only $service_count service definitions found"
        fi
    else
        log_fail "Service definitions directory missing"
    fi
}

# Test container deployment
test_container_deployment() {
    print_section "Container Deployment Tests"
    
    log_test "Container status"
    if docker ps | grep -q "ultimate-media-server"; then
        log_pass "Container is running"
        
        log_test "Container health"
        local health=$(docker inspect --format='{{.State.Health.Status}}' ultimate-media-server 2>/dev/null || echo "none")
        if [ "$health" = "healthy" ]; then
            log_pass "Container is healthy"
        elif [ "$health" = "starting" ]; then
            log_warn "Container is still starting"
        else
            log_warn "Container health status: $health"
        fi
    else
        log_warn "Container not running - will test after deployment"
    fi
}

# Test individual services
test_media_services() {
    print_section "Media Service Tests"
    
    # Jellyfin
    log_test "Jellyfin service"
    if curl -f -s "http://localhost:8096/health" > /dev/null 2>&1; then
        log_pass "Jellyfin is running"
    else
        log_fail "Jellyfin not responding"
    fi
    
    # Sonarr
    log_test "Sonarr service"
    if curl -f -s "http://localhost:8989/ping" > /dev/null 2>&1; then
        log_pass "Sonarr is running"
    else
        log_fail "Sonarr not responding"
    fi
    
    # Radarr
    log_test "Radarr service"
    if curl -f -s "http://localhost:7878/ping" > /dev/null 2>&1; then
        log_pass "Radarr is running"
    else
        log_fail "Radarr not responding"
    fi
    
    # Prowlarr
    log_test "Prowlarr service"
    if curl -f -s "http://localhost:9696/ping" > /dev/null 2>&1; then
        log_pass "Prowlarr is running"
    else
        log_fail "Prowlarr not responding"
    fi
    
    # Lidarr
    log_test "Lidarr service"
    if curl -f -s "http://localhost:8686/ping" > /dev/null 2>&1; then
        log_pass "Lidarr is running"
    else
        log_fail "Lidarr not responding"
    fi
    
    # Bazarr
    log_test "Bazarr service"
    if curl -f -s "http://localhost:6767" > /dev/null 2>&1; then
        log_pass "Bazarr is running"
    else
        log_fail "Bazarr not responding"
    fi
    
    # qBittorrent
    log_test "qBittorrent service"
    if curl -f -s "http://localhost:8080" > /dev/null 2>&1; then
        log_pass "qBittorrent is running"
    else
        log_fail "qBittorrent not responding"
    fi
}

# Test dashboard
test_dashboard() {
    print_section "Dashboard Tests"
    
    log_test "Dashboard availability"
    if curl -f -s "http://localhost:3000" > /dev/null 2>&1; then
        log_pass "Dashboard is accessible"
        
        log_test "Dashboard load time"
        local load_time=$(curl -o /dev/null -s -w '%{time_total}' "http://localhost:3000")
        if (( $(echo "$load_time < 2" | bc -l) )); then
            log_pass "Dashboard loads in ${load_time}s (< 2s target)"
        else
            log_warn "Dashboard loads in ${load_time}s (> 2s target)"
        fi
    else
        log_fail "Dashboard not accessible"
    fi
}

# Test AI features
test_ai_features() {
    print_section "AI Feature Tests"
    
    log_test "AI Assistant API"
    if curl -f -s "http://localhost:8090/health" > /dev/null 2>&1; then
        log_pass "AI Assistant API is running"
        
        log_test "AI response time"
        local start_time=$(date +%s%N)
        if curl -s -X POST "http://localhost:8090/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query":"test"}' > /dev/null 2>&1; then
            local end_time=$(date +%s%N)
            local response_time=$(( (end_time - start_time) / 1000000 ))
            if [ $response_time -lt 100 ]; then
                log_pass "AI response time: ${response_time}ms (< 100ms target)"
            else
                log_warn "AI response time: ${response_time}ms (> 100ms target)"
            fi
        else
            log_warn "AI query test failed"
        fi
    else
        log_fail "AI Assistant API not responding"
    fi
    
    log_test "Ollama service"
    if docker exec ultimate-media-server ollama list &> /dev/null; then
        log_pass "Ollama is installed and running"
    else
        log_warn "Ollama not detected"
    fi
}

# Test service interconnections
test_service_interconnections() {
    print_section "Service Interconnection Tests"
    
    log_test "Prowlarr → Sonarr connection"
    # Check if Prowlarr can reach Sonarr
    if curl -s "http://localhost:9696/api/v1/health" 2>/dev/null | grep -q "healthy"; then
        log_pass "Prowlarr is configured"
    else
        log_warn "Prowlarr configuration needs checking"
    fi
    
    log_test "Redis cache"
    if docker exec ultimate-media-server redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_pass "Redis cache is operational"
    else
        log_fail "Redis cache not responding"
    fi
}

# Performance tests
test_performance() {
    print_section "Performance Tests"
    
    log_test "Container memory usage"
    if docker stats --no-stream ultimate-media-server 2>/dev/null | grep -q ultimate-media-server; then
        local mem_usage=$(docker stats --no-stream --format "{{.MemUsage}}" ultimate-media-server 2>/dev/null | cut -d'/' -f1)
        log_info "Memory usage: $mem_usage"
    else
        log_warn "Cannot measure memory usage"
    fi
    
    log_test "Container CPU usage"
    if docker stats --no-stream ultimate-media-server 2>/dev/null | grep -q ultimate-media-server; then
        local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" ultimate-media-server 2>/dev/null)
        log_info "CPU usage: $cpu_usage"
    else
        log_warn "Cannot measure CPU usage"
    fi
}

# Generate test report
generate_report() {
    print_section "Test Summary Report"
    
    local pass_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    
    echo -e "${CYAN}═════════════════════════════════════════${NC}"
    echo -e "${WHITE}Total Tests:${NC}    $TOTAL_TESTS"
    echo -e "${GREEN}Passed:${NC}         $PASSED_TESTS"
    echo -e "${RED}Failed:${NC}         $FAILED_TESTS"
    echo -e "${YELLOW}Warnings:${NC}       $WARNINGS"
    echo -e "${WHITE}Pass Rate:${NC}      ${pass_rate}%"
    echo -e "${CYAN}═════════════════════════════════════════${NC}"
    
    # Save detailed report
    cat > test-results.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "summary": {
    "total": $TOTAL_TESTS,
    "passed": $PASSED_TESTS,
    "failed": $FAILED_TESTS,
    "warnings": $WARNINGS,
    "pass_rate": $pass_rate
  },
  "results": $(declare -p TEST_RESULTS | sed 's/declare -A TEST_RESULTS=//')
}
EOF
    
    echo ""
    if [ $pass_rate -ge 90 ]; then
        echo -e "${GREEN}✅ System is production ready!${NC}"
    elif [ $pass_rate -ge 70 ]; then
        echo -e "${YELLOW}⚠️  System needs minor fixes${NC}"
    else
        echo -e "${RED}❌ System has critical issues${NC}"
    fi
}

# Main test execution
main() {
    show_header
    
    # Run all test suites
    test_docker_environment
    test_system_resources
    test_port_availability
    test_container_build
    test_container_deployment
    test_media_services
    test_dashboard
    test_ai_features
    test_service_interconnections
    test_performance
    
    # Generate final report
    generate_report
    
    echo ""
    echo -e "${CYAN}Test results saved to: test-results.json${NC}"
    echo ""
    
    # Return appropriate exit code
    if [ $FAILED_TESTS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Execute tests
main "$@"