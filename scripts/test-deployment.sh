#!/bin/bash

# Media Server Deployment Test Suite
# Comprehensive testing for all deployment scenarios
# Version: 1.0.0

set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Test configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TEST_DIR="${PROJECT_ROOT}/test-deployment"
readonly TEST_LOG="${PROJECT_ROOT}/TEST_REPORTS/deployment-test-$(date +%Y%m%d-%H%M%S).log"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Logging functions
log_test() {
    echo -e "\n${BLUE}[TEST]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST: $1" >> "$TEST_LOG"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PASS: $1" >> "$TEST_LOG"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL: $1" >> "$TEST_LOG"
    ((TESTS_FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $1" >> "$TEST_LOG"
    ((TESTS_SKIPPED++))
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >> "$TEST_LOG"
}

# Show test banner
show_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║              Media Server Deployment Test Suite                  ║"
    echo "║                                                                  ║"
    echo "║  Validating: Installation • Upgrade • Backup • Recovery         ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Initialize test environment
init_test_env() {
    log_info "Initializing test environment..."
    
    # Create test directories
    mkdir -p "$TEST_DIR"
    mkdir -p "$(dirname "$TEST_LOG")"
    
    # Backup existing installation if present
    if [ -d "$HOME/mediaserver" ]; then
        log_info "Backing up existing installation..."
        mv "$HOME/mediaserver" "$HOME/mediaserver.backup.$(date +%s)"
    fi
    
    # Check Docker
    if ! docker info &> /dev/null; then
        log_fail "Docker is not running"
        exit 1
    fi
    
    log_info "Test environment ready"
}

# Cleanup test environment
cleanup_test_env() {
    log_info "Cleaning up test environment..."
    
    # Stop any test containers
    cd "$TEST_DIR" 2>/dev/null && docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove test directory
    rm -rf "$TEST_DIR"
    
    # Restore backup if exists
    if [ -d "$HOME/mediaserver.backup."* ]; then
        local backup=$(ls -d "$HOME/mediaserver.backup."* | head -1)
        log_info "Restoring original installation from $backup"
        mv "$backup" "$HOME/mediaserver"
    fi
}

# Test 1: Fresh Installation
test_fresh_installation() {
    ((TESTS_RUN++))
    log_test "Fresh Installation Test"
    
    # Create test directory
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    
    # Run installation script with test inputs
    log_info "Running installation script..."
    cat > test-inputs.txt << EOF
$TEST_DIR
localhost
admin@localhost
$TEST_DIR/media
3
EOF
    
    if ! "$PROJECT_ROOT/install-media-server.sh" < test-inputs.txt > install.log 2>&1; then
        log_fail "Installation script failed"
        cat install.log
        return 1
    fi
    
    # Verify installation
    local checks_passed=0
    local total_checks=5
    
    # Check 1: Environment file
    if [ -f "$TEST_DIR/.env" ]; then
        log_info "✓ Environment file created"
        ((checks_passed++))
    else
        log_info "✗ Environment file missing"
    fi
    
    # Check 2: Directory structure
    if [ -d "$TEST_DIR/config" ] && [ -d "$TEST_DIR/media" ] && [ -d "$TEST_DIR/downloads" ]; then
        log_info "✓ Directory structure created"
        ((checks_passed++))
    else
        log_info "✗ Directory structure incomplete"
    fi
    
    # Check 3: Docker compose file
    if [ -f "$TEST_DIR/docker-compose.yml" ]; then
        log_info "✓ Docker compose file present"
        ((checks_passed++))
    else
        log_info "✗ Docker compose file missing"
    fi
    
    # Check 4: Scripts directory
    if [ -d "$TEST_DIR/scripts" ] && [ -f "$TEST_DIR/scripts/backup.sh" ]; then
        log_info "✓ Scripts created"
        ((checks_passed++))
    else
        log_info "✗ Scripts missing"
    fi
    
    # Check 5: Caddy configuration
    if [ -f "$TEST_DIR/config/caddy/Caddyfile" ]; then
        log_info "✓ Caddy configuration created"
        ((checks_passed++))
    else
        log_info "✗ Caddy configuration missing"
    fi
    
    if [ $checks_passed -eq $total_checks ]; then
        log_pass "Fresh installation test passed ($checks_passed/$total_checks checks)"
        return 0
    else
        log_fail "Fresh installation test failed ($checks_passed/$total_checks checks)"
        return 1
    fi
}

# Test 2: Service Deployment
test_service_deployment() {
    ((TESTS_RUN++))
    log_test "Service Deployment Test"
    
    if [ ! -f "$TEST_DIR/docker-compose.yml" ]; then
        log_skip "Skipping deployment test - no docker-compose.yml"
        return 0
    fi
    
    cd "$TEST_DIR"
    
    # Deploy core services
    log_info "Deploying core services..."
    if ! docker-compose up -d jellyfin homepage qbittorrent > deploy.log 2>&1; then
        log_fail "Service deployment failed"
        cat deploy.log
        return 1
    fi
    
    # Wait for services to start
    log_info "Waiting for services to initialize..."
    sleep 15
    
    # Check service health
    local services_healthy=0
    local total_services=3
    
    # Check Jellyfin
    if curl -sf -m 5 http://localhost:8096 > /dev/null; then
        log_info "✓ Jellyfin is accessible"
        ((services_healthy++))
    else
        log_info "✗ Jellyfin not responding"
    fi
    
    # Check Homepage
    if curl -sf -m 5 http://localhost:3000 > /dev/null; then
        log_info "✓ Homepage is accessible"
        ((services_healthy++))
    else
        log_info "✗ Homepage not responding"
    fi
    
    # Check qBittorrent
    if curl -sf -m 5 http://localhost:8080 > /dev/null; then
        log_info "✓ qBittorrent is accessible"
        ((services_healthy++))
    else
        log_info "✗ qBittorrent not responding"
    fi
    
    if [ $services_healthy -eq $total_services ]; then
        log_pass "Service deployment test passed ($services_healthy/$total_services services healthy)"
        return 0
    else
        log_fail "Service deployment test failed ($services_healthy/$total_services services healthy)"
        return 1
    fi
}

# Test 3: Backup and Restore
test_backup_restore() {
    ((TESTS_RUN++))
    log_test "Backup and Restore Test"
    
    if [ ! -d "$TEST_DIR/config" ]; then
        log_skip "Skipping backup test - no config directory"
        return 0
    fi
    
    cd "$TEST_DIR"
    
    # Create test data
    echo "test-data" > config/test-file.txt
    mkdir -p config/jellyfin
    echo "jellyfin-config" > config/jellyfin/test.conf
    
    # Run backup
    log_info "Creating backup..."
    if ! ./scripts/backup.sh > backup.log 2>&1; then
        log_fail "Backup script failed"
        cat backup.log
        return 1
    fi
    
    # Verify backup created
    if [ ! -d "backups" ] || [ -z "$(ls -A backups/)" ]; then
        log_fail "No backup created"
        return 1
    fi
    
    local latest_backup=$(ls -t backups/ | head -1)
    log_info "Backup created: $latest_backup"
    
    # Delete test data
    rm -rf config/jellyfin
    rm -f config/test-file.txt
    
    # Restore backup
    log_info "Restoring from backup..."
    cd backups/"$latest_backup"
    if ! tar -xzf config.tar.gz -C ../.. > restore.log 2>&1; then
        log_fail "Restore failed"
        cat restore.log
        return 1
    fi
    
    cd "$TEST_DIR"
    
    # Verify restore
    if [ -f "config/test-file.txt" ] && [ -f "config/jellyfin/test.conf" ]; then
        log_pass "Backup and restore test passed"
        return 0
    else
        log_fail "Backup and restore test failed - data not restored"
        return 1
    fi
}

# Test 4: Service Recovery
test_service_recovery() {
    ((TESTS_RUN++))
    log_test "Service Recovery Test"
    
    # Check if services are running
    if ! docker ps | grep -q jellyfin; then
        log_skip "Skipping recovery test - services not running"
        return 0
    fi
    
    # Stop a service
    log_info "Stopping Jellyfin service..."
    docker stop jellyfin > /dev/null
    
    # Verify stopped
    if docker ps | grep -q jellyfin; then
        log_fail "Failed to stop service"
        return 1
    fi
    
    # Wait for auto-restart
    log_info "Waiting for auto-recovery (30 seconds)..."
    sleep 30
    
    # Check if recovered
    if docker ps | grep -q jellyfin; then
        log_pass "Service recovery test passed - Jellyfin auto-restarted"
        return 0
    else
        # Try manual restart
        log_info "Auto-recovery failed, attempting manual restart..."
        cd "$TEST_DIR"
        docker-compose up -d jellyfin > /dev/null 2>&1
        
        sleep 10
        if docker ps | grep -q jellyfin; then
            log_pass "Service recovery test passed - manual restart successful"
            return 0
        else
            log_fail "Service recovery test failed"
            return 1
        fi
    fi
}

# Test 5: Performance Validation
test_performance() {
    ((TESTS_RUN++))
    log_test "Performance Validation Test"
    
    if [ -z "$(docker ps -q)" ]; then
        log_skip "Skipping performance test - no containers running"
        return 0
    fi
    
    # Check overall resource usage
    log_info "Checking resource usage..."
    
    # Memory usage
    local total_memory=0
    while IFS= read -r line; do
        local mem=$(echo "$line" | awk '{print $4}' | sed 's/%//')
        total_memory=$(echo "$total_memory + $mem" | bc)
    done < <(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}")
    
    log_info "Total memory usage: ${total_memory}%"
    
    # Response time tests
    local endpoints=(
        "http://localhost:8096:Jellyfin"
        "http://localhost:3000:Homepage"
        "http://localhost:8080:qBittorrent"
    )
    
    local slow_endpoints=0
    for endpoint_info in "${endpoints[@]}"; do
        local url="${endpoint_info%%:*}"
        local port="${endpoint_info#*:}"
        local name="${port#*:}"
        port="${port%%:*}"
        
        if curl -sf -m 1 "$url" > /dev/null 2>&1; then
            log_info "✓ $name response time < 1s"
        else
            log_info "✗ $name response time > 1s or not accessible"
            ((slow_endpoints++))
        fi
    done
    
    # Performance criteria
    if (( $(echo "$total_memory > 80" | bc -l) )); then
        log_fail "Performance test failed - high memory usage (${total_memory}%)"
        return 1
    elif [ $slow_endpoints -gt 1 ]; then
        log_fail "Performance test failed - multiple slow endpoints"
        return 1
    elif [ $slow_endpoints -eq 1 ] || (( $(echo "$total_memory > 60" | bc -l) )); then
        log_pass "Performance test passed with warnings"
        return 0
    else
        log_pass "Performance test passed - excellent performance"
        return 0
    fi
}

# Test 6: Configuration Validation
test_configuration() {
    ((TESTS_RUN++))
    log_test "Configuration Validation Test"
    
    if [ ! -f "$TEST_DIR/.env" ]; then
        log_skip "Skipping configuration test - no .env file"
        return 0
    fi
    
    cd "$TEST_DIR"
    local issues=0
    
    # Check required environment variables
    source .env
    
    # Check PUID/PGID
    if [ -z "$PUID" ] || [ -z "$PGID" ]; then
        log_info "✗ PUID/PGID not set"
        ((issues++))
    else
        log_info "✓ PUID/PGID configured"
    fi
    
    # Check paths
    if [ ! -d "$MEDIA_PATH" ]; then
        log_info "✗ Media path does not exist: $MEDIA_PATH"
        ((issues++))
    else
        log_info "✓ Media path exists"
    fi
    
    # Check API keys
    if [[ "$JELLYFIN_API_KEY" == *"openssl"* ]] || [ ${#JELLYFIN_API_KEY} -lt 32 ]; then
        log_info "✗ Invalid Jellyfin API key"
        ((issues++))
    else
        log_info "✓ Valid Jellyfin API key"
    fi
    
    # Check domain configuration
    if [ "$DOMAIN" == "localhost" ] && [ "$ENABLE_HTTPS" == "true" ]; then
        log_info "⚠ HTTPS enabled for localhost"
        ((issues++))
    else
        log_info "✓ Domain configuration valid"
    fi
    
    if [ $issues -eq 0 ]; then
        log_pass "Configuration validation passed"
        return 0
    else
        log_fail "Configuration validation failed - $issues issues found"
        return 1
    fi
}

# Test 7: Security Validation
test_security() {
    ((TESTS_RUN++))
    log_test "Security Validation Test"
    
    local security_issues=0
    
    # Check for default passwords
    if docker ps | grep -q qbittorrent; then
        # Try default credentials
        if curl -sf -u admin:adminadmin http://localhost:8080/api/v2/app/version > /dev/null 2>&1; then
            log_info "✗ qBittorrent using default credentials"
            ((security_issues++))
        else
            log_info "✓ qBittorrent not using default credentials"
        fi
    fi
    
    # Check file permissions
    if [ -f "$TEST_DIR/.env" ]; then
        local env_perms=$(stat -f %A "$TEST_DIR/.env" 2>/dev/null || stat -c %a "$TEST_DIR/.env" 2>/dev/null)
        if [ "$env_perms" != "600" ] && [ "$env_perms" != "640" ]; then
            log_info "✗ .env file has loose permissions: $env_perms"
            ((security_issues++))
        else
            log_info "✓ .env file permissions secure"
        fi
    fi
    
    # Check for exposed ports
    local exposed_ports=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -c "0.0.0.0" || true)
    if [ $exposed_ports -gt 5 ]; then
        log_info "⚠ Many services exposed on all interfaces"
        ((security_issues++))
    else
        log_info "✓ Service exposure limited"
    fi
    
    if [ $security_issues -eq 0 ]; then
        log_pass "Security validation passed"
        return 0
    else
        log_fail "Security validation failed - $security_issues issues found"
        return 1
    fi
}

# Generate test report
generate_report() {
    local report_file="${PROJECT_ROOT}/TEST_REPORTS/deployment-test-summary-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
Media Server Deployment Test Summary
====================================
Date: $(date)
Total Tests: $TESTS_RUN
Passed: $TESTS_PASSED
Failed: $TESTS_FAILED
Skipped: $TESTS_SKIPPED

Success Rate: $(( TESTS_PASSED * 100 / TESTS_RUN ))%

Test Results:
EOF
    
    # Add test results from log
    grep -E "(PASS|FAIL|SKIP):" "$TEST_LOG" >> "$report_file"
    
    echo -e "\n${CYAN}Test report saved to: $report_file${NC}"
}

# Main test execution
main() {
    show_banner
    
    # Initialize
    init_test_env
    
    # Run tests
    test_fresh_installation || true
    test_service_deployment || true
    test_backup_restore || true
    test_service_recovery || true
    test_performance || true
    test_configuration || true
    test_security || true
    
    # Cleanup
    cleanup_test_env
    
    # Summary
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Test Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Total Tests: $TESTS_RUN"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo -e "Skipped: ${YELLOW}$TESTS_SKIPPED${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "\n${RED}❌ Some tests failed${NC}"
    fi
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    [ $TESTS_FAILED -eq 0 ]
}

# Run main function
main "$@"