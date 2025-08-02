#!/bin/bash

# Comprehensive test runner for NewMedia Docker containers
# Orchestrates integration, performance, and security testing

set -euo pipefail

# Configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${TEST_DIR}/reports"
TEST_TYPE="${1:-all}"
ENVIRONMENT="${2:-local}"
PARALLEL="${3:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
INTEGRATION_TIMEOUT=1800  # 30 minutes
PERFORMANCE_TIMEOUT=3600  # 1 hour
SECURITY_TIMEOUT=900      # 15 minutes

# Function to log with timestamp and color
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [TEST_TYPE] [ENVIRONMENT] [PARALLEL]

TEST_TYPE:
  all           - Run all tests (default)
  integration   - Run integration tests only
  performance   - Run performance tests only
  security      - Run security tests only
  smoke         - Run quick smoke tests
  api           - Run API tests only

ENVIRONMENT:
  local         - Local development (default)
  ci            - CI environment
  staging       - Staging environment
  production    - Production environment (read-only tests)

PARALLEL:
  true          - Run tests in parallel where possible
  false         - Run tests sequentially (default)

Examples:
  $0                              # Run all tests locally
  $0 integration local true       # Run integration tests in parallel
  $0 performance ci               # Run performance tests in CI
  $0 security                     # Run security tests only
  $0 smoke production             # Run smoke tests on production

Environment Variables:
  BASE_URL              - Base URL for testing (default: http://localhost)
  SONARR_API_KEY        - Sonarr API key for authenticated tests
  RADARR_API_KEY        - Radarr API key for authenticated tests
  PROWLARR_API_KEY      - Prowlarr API key for authenticated tests
  GRAFANA_USER          - Grafana username (default: admin)
  GRAFANA_PASSWORD      - Grafana password (default: admin)
  SKIP_SETUP            - Skip test environment setup (default: false)
  CLEANUP_AFTER         - Cleanup test environment after tests (default: true)
EOF
}

# Function to check prerequisites
check_prerequisites() {
    log "${BLUE}Checking prerequisites...${NC}"
    
    local missing_tools=()
    
    # Check required tools
    local tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    # Check optional tools based on test type
    if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "performance" ]]; then
        if ! command -v k6 &> /dev/null; then
            log "${YELLOW}⚠ k6 not found - performance tests will use Docker${NC}"
        fi
    fi
    
    if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "security" ]]; then
        if ! command -v trivy &> /dev/null; then
            log "${YELLOW}⚠ Trivy not found - will use Docker for security scans${NC}"
        fi
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log "${RED}✗ Missing required tools: ${missing_tools[*]}${NC}"
        exit 1
    fi
    
    log "${GREEN}✓ All prerequisites satisfied${NC}"
}

# Function to setup test environment
setup_test_environment() {
    if [[ "${SKIP_SETUP:-false}" == "true" ]]; then
        log "${YELLOW}Skipping test environment setup${NC}"
        return 0
    fi
    
    log "${BLUE}Setting up test environment...${NC}"
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"/{integration,performance,security,logs}
    
    # Set environment variables
    export NODE_ENV=test
    export BASE_URL="${BASE_URL:-http://localhost}"
    
    # Check if main services are running
    if [[ "$ENVIRONMENT" == "local" ]]; then
        log "${YELLOW}Checking if services are running...${NC}"
        
        # Start services if not running
        if ! docker ps | grep -q jellyfin; then
            log "${YELLOW}Starting services with docker-compose...${NC}"
            cd "${TEST_DIR}/.."
            docker-compose up -d
            
            # Wait for services to be ready
            log "${YELLOW}Waiting for services to be ready...${NC}"
            sleep 30
        fi
    fi
    
    # Setup test-specific containers
    log "${YELLOW}Starting test infrastructure...${NC}"
    cd "$TEST_DIR"
    docker-compose -f docker-compose.test.yml up -d test-postgres test-redis mock-tmdb mock-indexer
    
    # Wait for test services
    local max_retries=30
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if docker-compose -f docker-compose.test.yml ps | grep -q "Up"; then
            log "${GREEN}✓ Test services are ready${NC}"
            break
        fi
        
        log "${YELLOW}Waiting for test services... ($((retry + 1))/$max_retries)${NC}"
        sleep 5
        ((retry++))
    done
    
    if [ $retry -eq $max_retries ]; then
        log "${RED}✗ Test services failed to start${NC}"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    log "${CYAN}=== Running Integration Tests ===${NC}"
    
    local start_time=$(date +%s)
    local success=true
    
    cd "$TEST_DIR"
    
    # Run Jest integration tests
    if timeout $INTEGRATION_TIMEOUT npm run test:integration; then
        log "${GREEN}✓ Integration tests passed${NC}"
    else
        log "${RED}✗ Integration tests failed${NC}"
        success=false
    fi
    
    # Run Newman API tests if collection exists
    if [ -f "integration/postman/collection.json" ]; then
        log "${YELLOW}Running Newman API tests...${NC}"
        if timeout 300 npm run test:api; then
            log "${GREEN}✓ API tests passed${NC}"
        else
            log "${RED}✗ API tests failed${NC}"
            success=false
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "${BLUE}Integration tests completed in ${duration}s${NC}"
    
    if [[ "$success" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to run performance tests
run_performance_tests() {
    log "${CYAN}=== Running Performance Tests ===${NC}"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "${YELLOW}Skipping performance tests in production environment${NC}"
        return 0
    fi
    
    local start_time=$(date +%s)
    local success=true
    
    cd "$TEST_DIR"
    
    # Start monitoring infrastructure
    log "${YELLOW}Starting performance monitoring...${NC}"
    docker-compose -f docker-compose.test.yml up -d influxdb grafana-k6
    sleep 10
    
    # Run different performance test types
    local tests=("load-test" "stress-test")
    
    if [[ "$TEST_TYPE" == "performance" ]]; then
        tests+=("spike-test" "soak-test")
    fi
    
    for test in "${tests[@]}"; do
        log "${YELLOW}Running ${test}...${NC}"
        
        if command -v k6 &> /dev/null; then
            # Use local k6
            if timeout $PERFORMANCE_TIMEOUT k6 run \
                --out influxdb=http://localhost:8086/k6 \
                "performance/${test}.js"; then
                log "${GREEN}✓ ${test} passed${NC}"
            else
                log "${RED}✗ ${test} failed${NC}"
                success=false
            fi
        else
            # Use Docker k6
            if timeout $PERFORMANCE_TIMEOUT docker-compose -f docker-compose.test.yml run --rm k6 \
                run --out influxdb=http://influxdb:8086/k6 "/scripts/${test}.js"; then
                log "${GREEN}✓ ${test} passed${NC}"
            else
                log "${RED}✗ ${test} failed${NC}"
                success=false
            fi
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "${BLUE}Performance tests completed in ${duration}s${NC}"
    log "${BLUE}View results at: http://localhost:3030${NC}"
    
    if [[ "$success" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to run security tests
run_security_tests() {
    log "${CYAN}=== Running Security Tests ===${NC}"
    
    local start_time=$(date +%s)
    local success=true
    
    cd "$TEST_DIR"
    
    # Run Trivy vulnerability scans
    log "${YELLOW}Running vulnerability scans...${NC}"
    if timeout $SECURITY_TIMEOUT ./security/trivy-scan.sh "$REPORTS_DIR/security"; then
        log "${GREEN}✓ Vulnerability scans passed${NC}"
    else
        log "${RED}✗ Vulnerability scans found issues${NC}"
        success=false
    fi
    
    # Run network security tests
    log "${YELLOW}Running network security tests...${NC}"
    if timeout $SECURITY_TIMEOUT ./security/network-test.sh "$REPORTS_DIR/security"; then
        log "${GREEN}✓ Network security tests passed${NC}"
    else
        log "${RED}✗ Network security tests failed${NC}"
        success=false
    fi
    
    # Run permission audit
    log "${YELLOW}Running permission audit...${NC}"
    if timeout $SECURITY_TIMEOUT ./security/permission-audit.sh "$REPORTS_DIR/security"; then
        log "${GREEN}✓ Permission audit passed${NC}"
    else
        log "${RED}✗ Permission audit found issues${NC}"
        success=false
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "${BLUE}Security tests completed in ${duration}s${NC}"
    
    if [[ "$success" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to run smoke tests
run_smoke_tests() {
    log "${CYAN}=== Running Smoke Tests ===${NC}"
    
    local start_time=$(date +%s)
    local success=true
    
    # Quick health checks
    local services=(
        "localhost:8096:Jellyfin"
        "localhost:3001:Homepage"
        "localhost:8989:Sonarr"
        "localhost:7878:Radarr"
        "localhost:5055:Overseerr"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r host port name <<< "$service"
        
        log "${YELLOW}Testing ${name} at ${host}:${port}...${NC}"
        
        if curl -f -s -m 10 "http://${host}:${port}" >/dev/null; then
            log "${GREEN}✓ ${name} is accessible${NC}"
        else
            log "${RED}✗ ${name} is not accessible${NC}"
            success=false
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "${BLUE}Smoke tests completed in ${duration}s${NC}"
    
    if [[ "$success" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to cleanup test environment
cleanup_test_environment() {
    if [[ "${CLEANUP_AFTER:-true}" == "false" ]]; then
        log "${YELLOW}Skipping test environment cleanup${NC}"
        return 0
    fi
    
    log "${BLUE}Cleaning up test environment...${NC}"
    
    cd "$TEST_DIR"
    
    # Stop test containers
    docker-compose -f docker-compose.test.yml down -v
    
    # Clean up any leftover containers
    docker container prune -f
    
    log "${GREEN}✓ Test environment cleaned up${NC}"
}

# Function to generate test report
generate_test_report() {
    local total_duration="$1"
    local test_results="$2"
    
    log "${BLUE}Generating test report...${NC}"
    
    local report_file="${REPORTS_DIR}/test-summary.json"
    local html_file="${REPORTS_DIR}/test-report.html"
    
    # Generate JSON report
    cat > "$report_file" << EOF
{
  "test_run": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "test_type": "$TEST_TYPE",
    "environment": "$ENVIRONMENT",
    "duration_seconds": $total_duration,
    "parallel": $PARALLEL
  },
  "results": $test_results,
  "reports_location": "$REPORTS_DIR"
}
EOF
    
    # Generate HTML report
    cat > "$html_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>NewMedia Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .success { color: #4caf50; }
        .failure { color: #f44336; }
        .warning { color: #ff9800; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #2196f3; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NewMedia Docker Test Report</h1>
        <p><strong>Test Type:</strong> $TEST_TYPE</p>
        <p><strong>Environment:</strong> $ENVIRONMENT</p>
        <p><strong>Date:</strong> $(date)</p>
        <p><strong>Duration:</strong> ${total_duration}s</p>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <p>Detailed results are available in the individual test reports.</p>
        <p><strong>Reports Directory:</strong> $REPORTS_DIR</p>
    </div>
    
    <div class="section">
        <h2>Quick Links</h2>
        <ul>
            <li><a href="integration-test-report.html">Integration Test Results</a></li>
            <li><a href="http://localhost:3030">Performance Dashboard</a> (if running)</li>
            <li><a href="security/">Security Reports</a></li>
        </ul>
    </div>
</body>
</html>
EOF
    
    log "${GREEN}✓ Test report generated: ${html_file}${NC}"
}

# Main execution
main() {
    local start_time=$(date +%s)
    local overall_success=true
    local test_results='{"integration": "skipped", "performance": "skipped", "security": "skipped", "smoke": "skipped"}'
    
    # Show header
    log "${GREEN}=== NewMedia Docker Test Suite ===${NC}"
    log "${BLUE}Test Type: $TEST_TYPE${NC}"
    log "${BLUE}Environment: $ENVIRONMENT${NC}"
    log "${BLUE}Parallel: $PARALLEL${NC}"
    log "${BLUE}Reports: $REPORTS_DIR${NC}"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup test environment
    setup_test_environment
    
    # Run tests based on type
    case "$TEST_TYPE" in
        "all")
            if run_integration_tests; then
                test_results=$(echo "$test_results" | jq '.integration = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.integration = "failed"')
                overall_success=false
            fi
            
            if run_performance_tests; then
                test_results=$(echo "$test_results" | jq '.performance = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.performance = "failed"')
                overall_success=false
            fi
            
            if run_security_tests; then
                test_results=$(echo "$test_results" | jq '.security = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.security = "failed"')
                overall_success=false
            fi
            ;;
        "integration")
            if run_integration_tests; then
                test_results=$(echo "$test_results" | jq '.integration = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.integration = "failed"')
                overall_success=false
            fi
            ;;
        "performance")
            if run_performance_tests; then
                test_results=$(echo "$test_results" | jq '.performance = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.performance = "failed"')
                overall_success=false
            fi
            ;;
        "security")
            if run_security_tests; then
                test_results=$(echo "$test_results" | jq '.security = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.security = "failed"')
                overall_success=false
            fi
            ;;
        "smoke")
            if run_smoke_tests; then
                test_results=$(echo "$test_results" | jq '.smoke = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.smoke = "failed"')
                overall_success=false
            fi
            ;;
        "api")
            if run_integration_tests; then
                test_results=$(echo "$test_results" | jq '.integration = "passed"')
            else
                test_results=$(echo "$test_results" | jq '.integration = "failed"')
                overall_success=false
            fi
            ;;
        *)
            log "${RED}Unknown test type: $TEST_TYPE${NC}"
            show_usage
            exit 1
            ;;
    esac
    
    # Cleanup
    cleanup_test_environment
    
    # Calculate total duration
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Generate report
    generate_test_report "$total_duration" "$test_results"
    
    # Final summary
    log "${GREEN}=== Test Summary ===${NC}"
    log "Total duration: ${total_duration}s"
    log "Results: $(echo "$test_results" | jq -r 'to_entries | map("\(.key): \(.value)") | join(", ")')"
    
    if [[ "$overall_success" == "true" ]]; then
        log "${GREEN}✓ All tests passed successfully!${NC}"
        exit 0
    else
        log "${RED}✗ Some tests failed. Check the reports for details.${NC}"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    "-h"|"--help"|"help")
        show_usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac