#!/bin/bash

# ============================================================================
# COMPREHENSIVE MEDIA SERVER TEST SUITE
# ============================================================================
# Tests all components of the fixed media server stack
# Author: QA Engineer Agent
# Date: August 2025
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Test report file
REPORT_FILE="test-results-$(date +%Y%m%d-%H%M%S).md"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
    echo "✅ PASS: $1" >> "$REPORT_FILE"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    echo "❌ FAIL: $1" >> "$REPORT_FILE"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
    echo "⚠️ SKIP: $1" >> "$REPORT_FILE"
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

check_port() {
    if nc -z localhost $1 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

test_api_endpoint() {
    local endpoint=$1
    local expected_status=${2:-200}
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:3000${endpoint}")
    if [ "$response" = "$expected_status" ]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# INITIALIZE TEST REPORT
# ============================================================================

echo "# Media Server Test Results - $(date)" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## Test Environment" >> "$REPORT_FILE"
echo "- Date: $(date)" >> "$REPORT_FILE"
echo "- System: $(uname -s)" >> "$REPORT_FILE"
echo "- Docker: $(docker --version 2>/dev/null || echo 'Not installed')" >> "$REPORT_FILE"
echo "- Node.js: $(node --version 2>/dev/null || echo 'Not installed')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## Test Results" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# ============================================================================
# TEST SUITE 1: DOCKER CONFIGURATION
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 1: DOCKER CONFIGURATION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Test 1.1: Docker installation
log_test "Checking Docker installation..."
if check_command docker; then
    log_pass "Docker is installed"
else
    log_fail "Docker is not installed"
fi

# Test 1.2: Docker Compose installation
log_test "Checking Docker Compose installation..."
if check_command docker-compose || docker compose version &>/dev/null; then
    log_pass "Docker Compose is installed"
else
    log_fail "Docker Compose is not installed"
fi

# Test 1.3: Docker daemon running
log_test "Checking Docker daemon status..."
if docker info &>/dev/null; then
    log_pass "Docker daemon is running"
else
    log_fail "Docker daemon is not running"
fi

# Test 1.4: Fixed Docker Compose file exists
log_test "Checking for fixed Docker Compose file..."
if check_file "docker-compose.fixed.yml"; then
    log_pass "docker-compose.fixed.yml exists"
else
    log_fail "docker-compose.fixed.yml not found"
fi

# Test 1.5: Environment template exists
log_test "Checking for environment template..."
if check_file ".env.fixed.template"; then
    log_pass ".env.fixed.template exists"
else
    log_fail ".env.fixed.template not found"
fi

# Test 1.6: Validate Docker Compose syntax
log_test "Validating Docker Compose syntax..."
if docker-compose -f docker-compose.fixed.yml config &>/dev/null; then
    log_pass "Docker Compose syntax is valid"
else
    log_fail "Docker Compose syntax has errors"
fi

# ============================================================================
# TEST SUITE 2: API SERVER
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 2: API SERVER${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Test 2.1: API server files exist
log_test "Checking API server files..."
if check_file "api/server.js"; then
    log_pass "api/server.js exists"
else
    log_fail "api/server.js not found"
fi

# Test 2.2: DockerManager service exists
log_test "Checking DockerManager service..."
if check_file "api/services/DockerManager.js"; then
    log_pass "DockerManager.js exists"
else
    log_fail "DockerManager.js not found"
fi

# Test 2.3: Authentication middleware exists
log_test "Checking authentication middleware..."
if check_file "api/middleware/AuthMiddleware.js"; then
    log_pass "AuthMiddleware.js exists"
else
    log_fail "AuthMiddleware.js not found"
fi

# Test 2.4: Node.js dependencies
log_test "Checking Node.js dependencies..."
if check_file "package.json"; then
    if [ -d "node_modules" ]; then
        log_pass "Node modules are installed"
    else
        log_skip "Node modules not installed (run npm install)"
    fi
else
    log_fail "package.json not found"
fi

# Test 2.5: API server port availability
log_test "Checking API port 3000..."
if ! check_port 3000; then
    log_pass "Port 3000 is available"
else
    log_skip "Port 3000 is in use"
fi

# ============================================================================
# TEST SUITE 3: DASHBOARD UI
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 3: DASHBOARD UI${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Test 3.1: Dashboard HTML exists
log_test "Checking dashboard HTML..."
if check_file "dashboard/modern-media-dashboard.html"; then
    log_pass "modern-media-dashboard.html exists"
else
    log_fail "modern-media-dashboard.html not found"
fi

# Test 3.2: WebSocket client exists
log_test "Checking WebSocket client..."
if check_file "dashboard/websocket-client.js"; then
    log_pass "websocket-client.js exists"
else
    log_fail "websocket-client.js not found"
fi

# Test 3.3: Authentication service exists
log_test "Checking authentication service..."
if check_file "dashboard/auth-service.js"; then
    log_pass "auth-service.js exists"
else
    log_fail "auth-service.js not found"
fi

# Test 3.4: Performance monitor exists
log_test "Checking performance monitor..."
if check_file "dashboard/performance-monitor.js"; then
    log_pass "performance-monitor.js exists"
else
    log_fail "performance-monitor.js not found"
fi

# Test 3.5: Mobile app exists
log_test "Checking mobile PWA app..."
if check_file "dashboard/mobile-app.html"; then
    log_pass "mobile-app.html exists"
else
    log_fail "mobile-app.html not found"
fi

# ============================================================================
# TEST SUITE 4: SERVICE PORTS
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 4: SERVICE PORTS AVAILABILITY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Define service ports
declare -A SERVICE_PORTS=(
    ["Traefik HTTP"]=80
    ["Traefik HTTPS"]=443
    ["Traefik Dashboard"]=8080
    ["API Server"]=3000
    ["Jellyfin"]=8096
    ["Plex"]=32400
    ["Emby"]=8097
    ["Sonarr"]=8989
    ["Radarr"]=7878
    ["Lidarr"]=8686
    ["Readarr"]=8787
    ["Bazarr"]=6767
    ["Prowlarr"]=9696
    ["qBittorrent"]=8081
    ["SABnzbd"]=8082
    ["Transmission"]=9091
    ["Overseerr"]=5055
    ["Tautulli"]=8181
    ["Prometheus"]=9090
    ["Grafana"]=3001
    ["Portainer"]=9000
)

# Test each service port
for service in "${!SERVICE_PORTS[@]}"; do
    port=${SERVICE_PORTS[$service]}
    log_test "Checking $service port $port..."
    if ! check_port $port; then
        log_pass "$service port $port is available"
    else
        log_skip "$service port $port is in use"
    fi
done

# ============================================================================
# TEST SUITE 5: SECURITY CHECKS
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 5: SECURITY CHECKS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Test 5.1: Check for hardcoded passwords
log_test "Checking for hardcoded passwords..."
if ! grep -r "password.*=.*admin" --include="*.js" --include="*.yml" . 2>/dev/null | grep -v ".env.template" | grep -v "test-" > /dev/null; then
    log_pass "No hardcoded passwords found"
else
    log_fail "Hardcoded passwords detected"
fi

# Test 5.2: Check for exposed API keys
log_test "Checking for exposed API keys..."
if ! grep -r "apikey.*=.*[a-z0-9]{32}" --include="*.js" . 2>/dev/null | grep -v ".env" > /dev/null; then
    log_pass "No exposed API keys found"
else
    log_fail "Exposed API keys detected"
fi

# Test 5.3: Check SSL configuration
log_test "Checking SSL configuration..."
if check_file "traefik/acme.json"; then
    log_pass "SSL certificate storage configured"
else
    log_skip "SSL certificate storage not configured yet"
fi

# Test 5.4: Check rate limiting
log_test "Checking rate limiting configuration..."
if grep -q "rateLimit" api/middleware/APIValidator.js 2>/dev/null; then
    log_pass "Rate limiting is configured"
else
    log_fail "Rate limiting not configured"
fi

# ============================================================================
# TEST SUITE 6: PERFORMANCE CHECKS
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 6: PERFORMANCE CHECKS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Test 6.1: Check resource limits in Docker Compose
log_test "Checking Docker resource limits..."
if grep -q "deploy:" docker-compose.fixed.yml 2>/dev/null; then
    log_pass "Resource limits are configured"
else
    log_fail "Resource limits not configured"
fi

# Test 6.2: Check caching configuration
log_test "Checking caching configuration..."
if grep -q "redis" docker-compose.fixed.yml 2>/dev/null; then
    log_pass "Redis caching is configured"
else
    log_skip "Redis caching not configured"
fi

# Test 6.3: Check hardware acceleration
log_test "Checking hardware acceleration..."
if grep -q "/dev/dri" docker-compose.fixed.yml 2>/dev/null; then
    log_pass "Hardware acceleration is configured"
else
    log_skip "Hardware acceleration not configured"
fi

# ============================================================================
# TEST SUITE 7: INTEGRATION TESTS
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUITE 7: INTEGRATION TESTS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Test 7.1: Start API server test
log_test "Testing API server startup..."
if check_file "start-api-server.js"; then
    # Try to start the server in background for testing
    timeout 5 node start-api-server.js &>/dev/null &
    API_PID=$!
    sleep 2
    
    if kill -0 $API_PID 2>/dev/null; then
        log_pass "API server starts successfully"
        kill $API_PID 2>/dev/null
    else
        log_fail "API server failed to start"
    fi
else
    log_skip "start-api-server.js not found"
fi

# Test 7.2: Backend test suite
log_test "Running backend test suite..."
if check_file "test-backend-fixes.js"; then
    if timeout 10 node test-backend-fixes.js &>/dev/null; then
        log_pass "Backend tests passed"
    else
        log_fail "Backend tests failed"
    fi
else
    log_skip "test-backend-fixes.js not found"
fi

# ============================================================================
# TEST SUMMARY
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

echo -e "Total Tests: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${TESTS_SKIPPED}${NC}"

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))
    echo -e "\nPass Rate: ${PASS_RATE}%"
fi

# Add summary to report
echo "" >> "$REPORT_FILE"
echo "## Summary" >> "$REPORT_FILE"
echo "- Total Tests: ${TOTAL_TESTS}" >> "$REPORT_FILE"
echo "- Passed: ${TESTS_PASSED}" >> "$REPORT_FILE"
echo "- Failed: ${TESTS_FAILED}" >> "$REPORT_FILE"
echo "- Skipped: ${TESTS_SKIPPED}" >> "$REPORT_FILE"
echo "- Pass Rate: ${PASS_RATE}%" >> "$REPORT_FILE"

# Determine overall status
if [ $TESTS_FAILED -eq 0 ] && [ $TESTS_PASSED -gt 0 ]; then
    echo -e "\n${GREEN}✅ ALL CRITICAL TESTS PASSED!${NC}"
    echo -e "\n## Status: ✅ SUCCESS" >> "$REPORT_FILE"
    EXIT_CODE=0
elif [ $TESTS_FAILED -lt 5 ]; then
    echo -e "\n${YELLOW}⚠️ MOSTLY PASSING - Minor issues detected${NC}"
    echo -e "\n## Status: ⚠️ PARTIAL SUCCESS" >> "$REPORT_FILE"
    EXIT_CODE=1
else
    echo -e "\n${RED}❌ CRITICAL FAILURES DETECTED${NC}"
    echo -e "\n## Status: ❌ FAILURE" >> "$REPORT_FILE"
    EXIT_CODE=2
fi

echo -e "\nTest report saved to: ${REPORT_FILE}"

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}RECOMMENDATIONS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

echo "## Recommendations" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ $TESTS_FAILED -gt 0 ] || [ $TESTS_SKIPPED -gt 0 ]; then
    echo "Based on test results, here are the recommended next steps:"
    echo ""
    
    if ! check_command docker; then
        echo "1. Install Docker: brew install --cask docker"
        echo "1. Install Docker" >> "$REPORT_FILE"
    fi
    
    if ! [ -d "node_modules" ]; then
        echo "2. Install Node dependencies: npm install"
        echo "2. Install Node dependencies" >> "$REPORT_FILE"
    fi
    
    if ! check_file ".env"; then
        echo "3. Create environment file: cp .env.fixed.template .env"
        echo "3. Create environment file" >> "$REPORT_FILE"
    fi
    
    if [ $TESTS_FAILED -gt 0 ]; then
        echo "4. Fix critical issues before deployment"
        echo "4. Fix critical issues" >> "$REPORT_FILE"
    fi
    
    echo "5. Run setup script: ./traefik-setup.sh"
    echo "5. Run setup script" >> "$REPORT_FILE"
    echo "6. Start services: docker-compose -f docker-compose.fixed.yml up -d"
    echo "6. Start services" >> "$REPORT_FILE"
else
    echo "✅ System is ready for deployment!"
    echo "✅ Ready for deployment" >> "$REPORT_FILE"
    echo ""
    echo "To deploy:"
    echo "1. cp .env.fixed.template .env"
    echo "2. Edit .env with your settings"
    echo "3. ./traefik-setup.sh"
    echo "4. docker-compose -f docker-compose.fixed.yml up -d"
fi

exit $EXIT_CODE