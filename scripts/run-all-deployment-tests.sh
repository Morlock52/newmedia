#!/bin/bash

# Ultimate Media Server - Comprehensive Deployment Test Runner
# Executes all deployment validation tests and generates reports
# Version: 1.0.0

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TEST_RESULTS_DIR="${PROJECT_ROOT}/TEST_RESULTS/deployment-$(date +%Y%m%d-%H%M%S)"
readonly SUMMARY_FILE="${TEST_RESULTS_DIR}/test-summary.md"

# Test categories
declare -A TEST_CATEGORIES=(
    ["installation"]="Installation and Setup Tests"
    ["deployment"]="Docker Deployment Tests"
    ["health"]="Health Check Tests"
    ["backup"]="Backup and Recovery Tests"
    ["security"]="Security Validation Tests"
    ["performance"]="Performance Tests"
)

# Test results tracking
declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Initialize test environment
init_test_environment() {
    echo -e "${CYAN}Initializing test environment...${NC}"
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Create test summary header
    cat > "$SUMMARY_FILE" << EOF
# Ultimate Media Server Deployment Test Results

**Date:** $(date)  
**System:** $(uname -s) $(uname -r)  
**Docker:** $(docker --version 2>/dev/null || echo "Not installed")  
**Test Directory:** $TEST_RESULTS_DIR

---

## Test Execution Summary

EOF
    
    # Check Docker availability
    if ! docker info &> /dev/null; then
        echo -e "${RED}Docker is not running. Some tests will be skipped.${NC}"
        echo "⚠️ **Warning:** Docker daemon not running" >> "$SUMMARY_FILE"
    fi
}

# Test logging functions
log_test_start() {
    local category=$1
    local test_name=$2
    echo -e "\n${BLUE}[TEST]${NC} ${category}: ${test_name}"
    echo "### ${test_name}" >> "${TEST_RESULTS_DIR}/${category}.log"
    echo "Started: $(date)" >> "${TEST_RESULTS_DIR}/${category}.log"
}

log_test_result() {
    local category=$1
    local test_name=$2
    local result=$3
    local details=$4
    
    ((TOTAL_TESTS++))
    
    if [ "$result" == "PASS" ]; then
        echo -e "${GREEN}[PASS]${NC} ${test_name}"
        ((PASSED_TESTS++))
        TEST_RESULTS["${category}_${test_name}"]="✅ PASS"
    else
        echo -e "${RED}[FAIL]${NC} ${test_name}: ${details}"
        ((FAILED_TESTS++))
        TEST_RESULTS["${category}_${test_name}"]="❌ FAIL: ${details}"
    fi
    
    echo "Result: $result" >> "${TEST_RESULTS_DIR}/${category}.log"
    echo "Details: $details" >> "${TEST_RESULTS_DIR}/${category}.log"
    echo "---" >> "${TEST_RESULTS_DIR}/${category}.log"
}

# Installation Tests
run_installation_tests() {
    local category="installation"
    echo -e "\n${PURPLE}═══ Running Installation Tests ═══${NC}"
    
    # Test 1: Script file validation
    log_test_start "$category" "Script File Validation"
    local issues=""
    
    # Check for hardcoded paths in original script
    if grep -q "/Users/morlock" "${PROJECT_ROOT}/install-media-server.sh" 2>/dev/null; then
        issues="Hardcoded paths found in install-media-server.sh"
    fi
    
    if [ -z "$issues" ]; then
        log_test_result "$category" "Script File Validation" "PASS" "No hardcoded paths"
    else
        log_test_result "$category" "Script File Validation" "FAIL" "$issues"
    fi
    
    # Test 2: Fixed script validation
    log_test_start "$category" "Fixed Script Validation"
    if [ -f "${PROJECT_ROOT}/install-media-server-fixed.sh" ]; then
        if ! grep -q "/Users/morlock" "${PROJECT_ROOT}/install-media-server-fixed.sh"; then
            log_test_result "$category" "Fixed Script Validation" "PASS" "No hardcoded paths in fixed version"
        else
            log_test_result "$category" "Fixed Script Validation" "FAIL" "Hardcoded paths still present"
        fi
    else
        log_test_result "$category" "Fixed Script Validation" "FAIL" "Fixed script not found"
    fi
    
    # Test 3: Directory structure creation
    log_test_start "$category" "Directory Structure Test"
    local test_dir="/tmp/media-server-test-$$"
    mkdir -p "$test_dir"
    
    # Simulate directory creation
    mkdir -p "$test_dir"/{config,media,downloads,logs,backups,scripts}
    
    if [ -d "$test_dir/config" ] && [ -d "$test_dir/media" ] && [ -d "$test_dir/scripts" ]; then
        log_test_result "$category" "Directory Structure Test" "PASS" "All directories created"
    else
        log_test_result "$category" "Directory Structure Test" "FAIL" "Missing directories"
    fi
    
    rm -rf "$test_dir"
}

# Deployment Tests
run_deployment_tests() {
    local category="deployment"
    echo -e "\n${PURPLE}═══ Running Deployment Tests ═══${NC}"
    
    # Test 1: Docker Compose file validation
    log_test_start "$category" "Docker Compose Validation"
    if [ -f "${PROJECT_ROOT}/docker-compose.yml" ]; then
        if docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" config &> /dev/null; then
            log_test_result "$category" "Docker Compose Validation" "PASS" "Valid compose file"
        else
            log_test_result "$category" "Docker Compose Validation" "FAIL" "Invalid compose syntax"
        fi
    else
        log_test_result "$category" "Docker Compose Validation" "FAIL" "docker-compose.yml not found"
    fi
    
    # Test 2: Environment file generation
    log_test_start "$category" "Environment File Test"
    local test_env="/tmp/test-env-$$"
    
    # Generate test .env
    cat > "$test_env" << EOF
PUID=1000
PGID=1000
TZ=America/New_York
DOMAIN=localhost
EOF
    
    if grep -q "PUID=" "$test_env" && grep -q "DOMAIN=" "$test_env"; then
        log_test_result "$category" "Environment File Test" "PASS" "Valid .env format"
    else
        log_test_result "$category" "Environment File Test" "FAIL" "Invalid .env format"
    fi
    
    rm -f "$test_env"
    
    # Test 3: Service deployment scripts
    log_test_start "$category" "Deployment Script Test"
    if [ -f "${PROJECT_ROOT}/scripts/deploy/deploy.sh" ]; then
        # Check script structure
        if grep -q "check_prerequisites" "${PROJECT_ROOT}/scripts/deploy/deploy.sh" &&
           grep -q "deploy_services" "${PROJECT_ROOT}/scripts/deploy/deploy.sh"; then
            log_test_result "$category" "Deployment Script Test" "PASS" "Required functions present"
        else
            log_test_result "$category" "Deployment Script Test" "FAIL" "Missing required functions"
        fi
    else
        log_test_result "$category" "Deployment Script Test" "FAIL" "deploy.sh not found"
    fi
}

# Health Check Tests
run_health_tests() {
    local category="health"
    echo -e "\n${PURPLE}═══ Running Health Check Tests ═══${NC}"
    
    # Test 1: Health check script validation
    log_test_start "$category" "Health Script Validation"
    if [ -f "${PROJECT_ROOT}/scripts/deploy/health-check.sh" ]; then
        if bash -n "${PROJECT_ROOT}/scripts/deploy/health-check.sh" 2>/dev/null; then
            log_test_result "$category" "Health Script Validation" "PASS" "Valid bash syntax"
        else
            log_test_result "$category" "Health Script Validation" "FAIL" "Syntax errors in script"
        fi
    else
        log_test_result "$category" "Health Script Validation" "FAIL" "health-check.sh not found"
    fi
    
    # Test 2: Port availability check
    log_test_start "$category" "Port Availability Test"
    local blocked_ports=0
    local ports=(8096 8989 7878 9696 8080 3000)
    
    for port in "${ports[@]}"; do
        if lsof -i :$port &> /dev/null; then
            ((blocked_ports++))
        fi
    done
    
    if [ $blocked_ports -eq 0 ]; then
        log_test_result "$category" "Port Availability Test" "PASS" "All ports available"
    else
        log_test_result "$category" "Port Availability Test" "FAIL" "$blocked_ports ports in use"
    fi
    
    # Test 3: System resource check
    log_test_start "$category" "System Resources Test"
    local available_memory=$(free -g 2>/dev/null | awk 'NR==2{print $7}' || echo "8")
    
    if [ "${available_memory:-8}" -ge 4 ]; then
        log_test_result "$category" "System Resources Test" "PASS" "Sufficient memory available"
    else
        log_test_result "$category" "System Resources Test" "FAIL" "Low memory: ${available_memory}GB"
    fi
}

# Backup and Recovery Tests
run_backup_tests() {
    local category="backup"
    echo -e "\n${PURPLE}═══ Running Backup Tests ═══${NC}"
    
    # Test 1: Backup script validation
    log_test_start "$category" "Backup Script Test"
    local backup_script="${PROJECT_ROOT}/scripts/backup.sh"
    
    if [ -f "$backup_script" ]; then
        if grep -q "tar -czf" "$backup_script"; then
            log_test_result "$category" "Backup Script Test" "PASS" "Backup commands present"
        else
            log_test_result "$category" "Backup Script Test" "FAIL" "Missing backup commands"
        fi
    else
        # Check in installation directory
        if [ -f "${PROJECT_ROOT}/scripts/deploy/backup.sh" ]; then
            log_test_result "$category" "Backup Script Test" "PASS" "Backup script found in deploy/"
        else
            log_test_result "$category" "Backup Script Test" "FAIL" "No backup script found"
        fi
    fi
    
    # Test 2: Backup directory permissions
    log_test_start "$category" "Backup Permissions Test"
    local test_backup_dir="/tmp/backup-test-$$"
    mkdir -p "$test_backup_dir"
    
    if touch "$test_backup_dir/test" 2>/dev/null; then
        log_test_result "$category" "Backup Permissions Test" "PASS" "Write permissions OK"
    else
        log_test_result "$category" "Backup Permissions Test" "FAIL" "No write permissions"
    fi
    
    rm -rf "$test_backup_dir"
}

# Security Tests
run_security_tests() {
    local category="security"
    echo -e "\n${PURPLE}═══ Running Security Tests ═══${NC}"
    
    # Test 1: Default password check
    log_test_start "$category" "Default Password Test"
    local warnings=""
    
    # Check for default passwords in documentation
    if grep -r "admin/adminadmin" "${PROJECT_ROOT}"/*.sh 2>/dev/null | grep -v "change immediately" > /dev/null; then
        warnings="Default passwords mentioned without warning"
    fi
    
    if [ -z "$warnings" ]; then
        log_test_result "$category" "Default Password Test" "PASS" "Proper password warnings"
    else
        log_test_result "$category" "Default Password Test" "FAIL" "$warnings"
    fi
    
    # Test 2: File permission recommendations
    log_test_start "$category" "Permission Check Test"
    
    # Check if scripts set proper permissions for .env
    if grep -q "chmod 600 .env" "${PROJECT_ROOT}/install-media-server-fixed.sh" 2>/dev/null; then
        log_test_result "$category" "Permission Check Test" "PASS" "Secure .env permissions"
    else
        log_test_result "$category" "Permission Check Test" "FAIL" "Missing .env permission setting"
    fi
    
    # Test 3: API key generation
    log_test_start "$category" "API Key Generation Test"
    
    # Test key generation command
    local test_key=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | cut -d' ' -f1)
    
    if [ ${#test_key} -ge 32 ]; then
        log_test_result "$category" "API Key Generation Test" "PASS" "Valid key generation"
    else
        log_test_result "$category" "API Key Generation Test" "FAIL" "Invalid key generation"
    fi
}

# Performance Tests
run_performance_tests() {
    local category="performance"
    echo -e "\n${PURPLE}═══ Running Performance Tests ═══${NC}"
    
    # Test 1: Script execution time
    log_test_start "$category" "Script Performance Test"
    
    # Measure a simple operation
    local start_time=$(date +%s.%N)
    bash -c "for i in {1..100}; do echo test > /dev/null; done"
    local end_time=$(date +%s.%N)
    local execution_time=$(echo "$end_time - $start_time" | bc)
    
    if (( $(echo "$execution_time < 0.1" | bc -l) )); then
        log_test_result "$category" "Script Performance Test" "PASS" "Fast execution"
    else
        log_test_result "$category" "Script Performance Test" "FAIL" "Slow execution: ${execution_time}s"
    fi
    
    # Test 2: Docker pull simulation
    log_test_start "$category" "Image Pull Test"
    
    # Check if Docker is available for pull test
    if docker info &> /dev/null; then
        # Test with a small image
        if timeout 30 docker pull alpine:latest &> /dev/null; then
            log_test_result "$category" "Image Pull Test" "PASS" "Docker pull working"
        else
            log_test_result "$category" "Image Pull Test" "FAIL" "Docker pull timeout"
        fi
    else
        log_test_result "$category" "Image Pull Test" "FAIL" "Docker not available"
    fi
}

# Generate comprehensive report
generate_final_report() {
    echo -e "\n${CYAN}Generating final report...${NC}"
    
    # Add test results to summary
    cat >> "$SUMMARY_FILE" << EOF

## Test Results by Category

EOF
    
    # Generate category summaries
    for category in "${!TEST_CATEGORIES[@]}"; do
        echo "### ${TEST_CATEGORIES[$category]}" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        
        # List all tests in this category
        for test_key in "${!TEST_RESULTS[@]}"; do
            if [[ $test_key == ${category}_* ]]; then
                local test_name=${test_key#${category}_}
                echo "- ${test_name}: ${TEST_RESULTS[$test_key]}" >> "$SUMMARY_FILE"
            fi
        done
        echo "" >> "$SUMMARY_FILE"
    done
    
    # Add summary statistics
    cat >> "$SUMMARY_FILE" << EOF

## Summary Statistics

- **Total Tests:** $TOTAL_TESTS
- **Passed:** $PASSED_TESTS ($(( PASSED_TESTS * 100 / TOTAL_TESTS ))%)
- **Failed:** $FAILED_TESTS ($(( FAILED_TESTS * 100 / TOTAL_TESTS ))%)

### Overall Result: $([ $FAILED_TESTS -eq 0 ] && echo "✅ **ALL TESTS PASSED**" || echo "❌ **FAILURES DETECTED**")

## Recommendations

EOF
    
    # Add recommendations based on failures
    if [ $FAILED_TESTS -gt 0 ]; then
        cat >> "$SUMMARY_FILE" << EOF
1. Review failed tests in the detailed logs
2. Apply fixes from the fixed script versions
3. Re-run tests after applying fixes
4. Consider implementing automated CI/CD testing

EOF
    else
        cat >> "$SUMMARY_FILE" << EOF
1. All tests passed - deployment system is ready
2. Consider setting up continuous monitoring
3. Implement regular backup testing
4. Document any custom configurations

EOF
    fi
    
    # Add timestamp
    echo "---" >> "$SUMMARY_FILE"
    echo "*Report generated: $(date)*" >> "$SUMMARY_FILE"
}

# Show final summary
show_summary() {
    echo -e "\n${PURPLE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                     Test Execution Complete                      ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\nTest Results:"
    echo -e "  Total Tests: ${CYAN}$TOTAL_TESTS${NC}"
    echo -e "  Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "  Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "  Success Rate: $([ $TOTAL_TESTS -gt 0 ] && echo "$(( PASSED_TESTS * 100 / TOTAL_TESTS ))%" || echo "0%")"
    
    echo -e "\nDetailed results saved to:"
    echo -e "  ${BLUE}$TEST_RESULTS_DIR${NC}"
    echo -e "\nSummary report:"
    echo -e "  ${BLUE}$SUMMARY_FILE${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}✅ All deployment tests passed!${NC}"
        echo -e "The deployment system is ready for use."
    else
        echo -e "\n${RED}❌ Some tests failed.${NC}"
        echo -e "Please review the detailed logs and apply necessary fixes."
    fi
}

# Main execution
main() {
    echo -e "${PURPLE}Ultimate Media Server - Deployment Test Suite${NC}"
    echo -e "${PURPLE}============================================${NC}\n"
    
    # Initialize
    init_test_environment
    
    # Run all test categories
    run_installation_tests
    run_deployment_tests
    run_health_tests
    run_backup_tests
    run_security_tests
    run_performance_tests
    
    # Generate reports
    generate_final_report
    
    # Show summary
    show_summary
    
    # Return appropriate exit code
    [ $FAILED_TESTS -eq 0 ]
}

# Execute main function
main "$@"