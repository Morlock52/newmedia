#!/bin/bash

# Ultimate Media Server 2025 - Master Test Runner
# Orchestrates all testing scenarios with comprehensive reporting and CI/CD integration

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
TEST_CONFIG_DIR="${SCRIPT_DIR}/test-config"
RESULTS_DIR="${SCRIPT_DIR}/test-results"
LOG_FILE="${RESULTS_DIR}/master-test-run-$(date +%Y%m%d_%H%M%S).log"
FINAL_REPORT="${RESULTS_DIR}/final-test-report.json"
HTML_REPORT="${RESULTS_DIR}/test-report.html"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Test suite tracking (use bash 4+ associative arrays if available)
if [[ ${BASH_VERSION%%.*} -ge 4 ]]; then
    declare -A SUITE_RESULTS=()
    declare -A SUITE_DURATION=()
else
    # Fallback for older bash versions
    SUITE_RESULTS=""
    SUITE_DURATION=""
fi
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0
WARNING_SUITES=0

# Configuration flags
RUN_CONTAINER_TESTS=true
RUN_API_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_PERFORMANCE_TESTS=false
RUN_SECURITY_TESTS=false
PARALLEL_EXECUTION=false
QUICK_MODE=false
ENVIRONMENT="local"
CLEANUP_AFTER=true
GENERATE_HTML_REPORT=true

# Ensure directories exist
mkdir -p "$RESULTS_DIR" "$TEST_CONFIG_DIR"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] [$level] $*" | tee -a "$LOG_FILE"
}

# Display usage information
usage() {
    cat << EOF
Ultimate Media Server 2025 - Master Test Runner

USAGE:
    $0 [OPTIONS] [TEST_TYPES...]

TEST TYPES:
    container       Run container isolation tests
    api            Run API connectivity tests
    integration    Run service integration tests
    performance    Run performance tests (k6)
    security       Run security tests (trivy, nuclei)
    smoke          Run quick smoke tests only
    all            Run all test types (default)

OPTIONS:
    --environment ENV       Set environment (local, ci, staging, production) [default: local]
    --parallel             Run test suites in parallel
    --quick                Run quick tests only (smoke tests)
    --no-cleanup           Don't cleanup after tests
    --no-html              Don't generate HTML report
    --config-dir DIR       Custom configuration directory
    --results-dir DIR      Custom results directory
    --timeout SECONDS      Global timeout for test suites [default: 1800]
    --compose-file FILE    Custom docker-compose file
    --help                 Show this help message

EXAMPLES:
    # Run all tests in local environment
    $0

    # Run only API and integration tests
    $0 api integration

    # Run quick smoke tests for CI
    $0 --quick --environment ci smoke

    # Run tests in parallel for faster execution
    $0 --parallel all

    # Run security tests only
    $0 security

    # Run performance tests with custom timeout
    $0 --timeout 3600 performance

ENVIRONMENT CONFIGURATIONS:
    local       Full test suite with detailed logging
    ci          Optimized for CI/CD pipelines
    staging     Production-like testing
    production  Read-only tests for production validation

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL_EXECUTION=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --no-cleanup)
                CLEANUP_AFTER=false
                shift
                ;;
            --no-html)
                GENERATE_HTML_REPORT=false
                shift
                ;;
            --config-dir)
                TEST_CONFIG_DIR="$2"
                shift 2
                ;;
            --results-dir)
                RESULTS_DIR="$2"
                LOG_FILE="${RESULTS_DIR}/master-test-run-$(date +%Y%m%d_%H%M%S).log"
                FINAL_REPORT="${RESULTS_DIR}/final-test-report.json"
                HTML_REPORT="${RESULTS_DIR}/test-report.html"
                mkdir -p "$RESULTS_DIR"
                shift 2
                ;;
            --timeout)
                GLOBAL_TIMEOUT="$2"
                shift 2
                ;;
            --compose-file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            container|api|integration|performance|security|smoke|all)
                # Test type arguments - handle later
                break
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Parse test types
    if [[ $# -gt 0 ]]; then
        RUN_CONTAINER_TESTS=false
        RUN_API_TESTS=false
        RUN_INTEGRATION_TESTS=false
        RUN_PERFORMANCE_TESTS=false
        RUN_SECURITY_TESTS=false
        
        for test_type in "$@"; do
            case $test_type in
                container)
                    RUN_CONTAINER_TESTS=true
                    ;;
                api)
                    RUN_API_TESTS=true
                    ;;
                integration)
                    RUN_INTEGRATION_TESTS=true
                    ;;
                performance)
                    RUN_PERFORMANCE_TESTS=true
                    ;;
                security)
                    RUN_SECURITY_TESTS=true
                    ;;
                smoke)
                    QUICK_MODE=true
                    RUN_CONTAINER_TESTS=true
                    RUN_API_TESTS=true
                    ;;
                all)
                    RUN_CONTAINER_TESTS=true
                    RUN_API_TESTS=true
                    RUN_INTEGRATION_TESTS=true
                    if [[ "$ENVIRONMENT" != "production" ]]; then
                        RUN_PERFORMANCE_TESTS=true
                        RUN_SECURITY_TESTS=true
                    fi
                    ;;
                *)
                    echo "Unknown test type: $test_type"
                    usage
                    exit 1
                    ;;
            esac
        done
    fi
}

# Environment-specific configuration
configure_environment() {
    log "INFO" "Configuring for environment: $ENVIRONMENT"
    
    case "$ENVIRONMENT" in
        "ci")
            PARALLEL_EXECUTION=true
            CLEANUP_AFTER=true
            export CI_MODE=true
            ;;
        "staging")
            CLEANUP_AFTER=false
            export STAGING_MODE=true
            ;;
        "production")
            RUN_PERFORMANCE_TESTS=false
            RUN_SECURITY_TESTS=false
            CLEANUP_AFTER=false
            export PRODUCTION_MODE=true
            export READ_ONLY_TESTS=true
            ;;
    esac
    
    if [[ "$QUICK_MODE" == "true" ]]; then
        RUN_PERFORMANCE_TESTS=false
        RUN_SECURITY_TESTS=false
        export QUICK_TEST_MODE=true
    fi
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites"
    
    local missing_tools=()
    
    # Required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    
    # Optional tools
    if [[ "$RUN_API_TESTS" == "true" ]]; then
        command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    fi
    
    if [[ "$RUN_PERFORMANCE_TESTS" == "true" ]]; then
        command -v k6 >/dev/null 2>&1 || log "WARN" "k6 not found - will use Docker version"
    fi
    
    if [[ "$RUN_SECURITY_TESTS" == "true" ]]; then
        command -v trivy >/dev/null 2>&1 || log "WARN" "trivy not found - will use Docker version"
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        echo -e "${RED}Please install the missing tools and try again.${NC}"
        exit 1
    fi
    
    # Check Docker service
    if ! docker info >/dev/null 2>&1; then
        log "ERROR" "Docker daemon is not running"
        exit 1
    fi
    
    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log "ERROR" "Docker compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log "INFO" "Prerequisites check completed"
}

# Setup test environment
setup_test_environment() {
    log "INFO" "Setting up test environment"
    
    # Create test configuration files
    create_test_configs
    
    # Ensure services are running
    if [[ "$ENVIRONMENT" != "production" ]]; then
        log "INFO" "Starting services with docker-compose"
        docker-compose -f "$COMPOSE_FILE" up -d
        
        # Wait for critical services
        wait_for_critical_services
    fi
    
    log "INFO" "Test environment setup completed"
}

# Create test configuration files
create_test_configs() {
    log "INFO" "Creating test configuration files"
    
    # API test configuration
    cat > "$TEST_CONFIG_DIR/api-test-config.yaml" << EOF
services:
  jellyfin:
    api_key: "${JELLYFIN_API_KEY:-}"
    timeout: 30
  sonarr:
    api_key: "${SONARR_API_KEY:-}"
    timeout: 30
  radarr:
    api_key: "${RADARR_API_KEY:-}"
    timeout: 30
  prowlarr:
    api_key: "${PROWLARR_API_KEY:-}"
    timeout: 30
  grafana:
    username: "${GRAFANA_USERNAME:-admin}"
    password: "${GRAFANA_PASSWORD:-admin}"
    timeout: 30

global:
  base_url: "http://localhost"
  verify_ssl: false
  quick_mode: ${QUICK_MODE}
EOF
    
    # Performance test configuration
    cat > "$TEST_CONFIG_DIR/performance-config.js" << EOF
export let options = {
  scenarios: {
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '30s',
    },
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },
        { duration: '5m', target: 10 },
        { duration: '2m', target: 0 },
      ],
    },
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 0 },
      ],
    },
  },
};

export let baseUrl = '${BASE_URL:-http://localhost}';
export let quickMode = ${QUICK_MODE};
EOF
}

# Wait for critical services
wait_for_critical_services() {
    log "INFO" "Waiting for critical services to be ready"
    
    local services=("jellyfin:8096" "prometheus:9090" "grafana:3000")
    local max_attempts=30
    
    for service_info in "${services[@]}"; do
        local service=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        
        log "INFO" "Waiting for $service on port $port"
        
        for i in $(seq 1 $max_attempts); do
            if timeout 5 nc -z localhost "$port" 2>/dev/null; then
                log "INFO" "$service is ready"
                break
            fi
            
            if [[ $i -eq $max_attempts ]]; then
                log "WARN" "$service did not become ready after $max_attempts attempts"
            fi
            
            sleep 2
        done
    done
}

# Run test suite
run_test_suite() {
    local suite_name="$1"
    local test_command="$2"
    local timeout="${3:-1800}"
    
    log "INFO" "Starting test suite: $suite_name"
    
    local start_time=$(date +%s)
    local suite_log="${RESULTS_DIR}/${suite_name}-$(date +%Y%m%d_%H%M%S).log"
    
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    
    # Run the test command with timeout
    if timeout "$timeout" bash -c "$test_command" > "$suite_log" 2>&1; then
        local exit_code=0
    else
        local exit_code=$?
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    SUITE_DURATION["$suite_name"]=$duration
    
    # Determine result
    if [[ $exit_code -eq 0 ]]; then
        SUITE_RESULTS["$suite_name"]="PASS"
        PASSED_SUITES=$((PASSED_SUITES + 1))
        echo -e "${GREEN}✓ $suite_name completed successfully (${duration}s)${NC}"
    elif [[ $exit_code -eq 124 ]]; then  # timeout
        SUITE_RESULTS["$suite_name"]="TIMEOUT"
        FAILED_SUITES=$((FAILED_SUITES + 1))
        echo -e "${RED}✗ $suite_name timed out after ${timeout}s${NC}"
    else
        # Check if it's a warning (exit code 2) or failure
        if [[ $exit_code -eq 2 ]]; then
            SUITE_RESULTS["$suite_name"]="WARN"
            WARNING_SUITES=$((WARNING_SUITES + 1))
            echo -e "${YELLOW}⚠ $suite_name completed with warnings (${duration}s)${NC}"
        else
            SUITE_RESULTS["$suite_name"]="FAIL"
            FAILED_SUITES=$((FAILED_SUITES + 1))
            echo -e "${RED}✗ $suite_name failed (${duration}s)${NC}"
        fi
    fi
    
    log "INFO" "Test suite $suite_name completed with result: ${SUITE_RESULTS[$suite_name]}"
}

# Run all selected test suites
run_test_suites() {
    log "INFO" "Running selected test suites"
    
    # Prepare test commands
    local container_cmd="${SCRIPT_DIR}/test-container-isolation.sh"
    local api_cmd="cd ${SCRIPT_DIR} && python3 test-api-connectivity.py --config ${TEST_CONFIG_DIR}/api-test-config.yaml"
    local integration_cmd="${SCRIPT_DIR}/test-service-integrations.sh"
    
    if [[ "$QUICK_MODE" == "true" ]]; then
        api_cmd+=" --quick"
    fi
    
    # Run tests based on configuration
    if [[ "$PARALLEL_EXECUTION" == "true" ]]; then
        log "INFO" "Running test suites in parallel"
        
        local pids=()
        
        if [[ "$RUN_CONTAINER_TESTS" == "true" ]]; then
            run_test_suite "container-isolation" "$container_cmd" 900 &
            pids+=($!)
        fi
        
        if [[ "$RUN_API_TESTS" == "true" ]]; then
            run_test_suite "api-connectivity" "$api_cmd" 600 &
            pids+=($!)
        fi
        
        if [[ "$RUN_INTEGRATION_TESTS" == "true" ]]; then
            run_test_suite "service-integration" "$integration_cmd" 1200 &
            pids+=($!)
        fi
        
        # Wait for parallel tests to complete
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        
        # Run sequential tests (performance and security)
        if [[ "$RUN_PERFORMANCE_TESTS" == "true" ]]; then
            run_performance_tests
        fi
        
        if [[ "$RUN_SECURITY_TESTS" == "true" ]]; then
            run_security_tests
        fi
        
    else
        log "INFO" "Running test suites sequentially"
        
        if [[ "$RUN_CONTAINER_TESTS" == "true" ]]; then
            run_test_suite "container-isolation" "$container_cmd" 900
        fi
        
        if [[ "$RUN_API_TESTS" == "true" ]]; then
            run_test_suite "api-connectivity" "$api_cmd" 600
        fi
        
        if [[ "$RUN_INTEGRATION_TESTS" == "true" ]]; then
            run_test_suite "service-integration" "$integration_cmd" 1200
        fi
        
        if [[ "$RUN_PERFORMANCE_TESTS" == "true" ]]; then
            run_performance_tests
        fi
        
        if [[ "$RUN_SECURITY_TESTS" == "true" ]]; then
            run_security_tests
        fi
    fi
}

# Run performance tests
run_performance_tests() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "INFO" "Skipping performance tests in production environment"
        return
    fi
    
    log "INFO" "Running performance tests"
    
    # Create k6 performance test script
    local k6_script="${TEST_CONFIG_DIR}/performance-test.js"
    
    cat > "$k6_script" << "K6_SCRIPT"
import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

export let errorRate = new Rate('errors');

export default function () {
    // Test main services
    let services = [
        'http://localhost:8096/health',      // Jellyfin
        'http://localhost:9090/-/healthy',   // Prometheus
        'http://localhost:3000/api/health',  // Grafana
    ];
    
    for (let url of services) {
        let response = http.get(url, { timeout: '10s' });
        
        let success = check(response, {
            'status is 200': (r) => r.status === 200,
            'response time < 5000ms': (r) => r.timings.duration < 5000,
        });
        
        errorRate.add(!success);
    }
}
K6_SCRIPT
    
    # Run k6 tests
    local k6_cmd=""
    if command -v k6 >/dev/null 2>&1; then
        k6_cmd="k6 run --config ${TEST_CONFIG_DIR}/performance-config.js ${k6_script}"
    else
        k6_cmd="docker run --rm -v ${TEST_CONFIG_DIR}:/scripts grafana/k6:latest run /scripts/performance-test.js"
    fi
    
    run_test_suite "performance" "$k6_cmd" 1800
}

# Run security tests
run_security_tests() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "INFO" "Skipping security tests in production environment"
        return
    fi
    
    log "INFO" "Running security tests"
    
    # Container vulnerability scanning
    local security_cmd="docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v ${RESULTS_DIR}:/reports aquasec/trivy:latest image --format json --output /reports/trivy-scan.json jellyfin/jellyfin:latest"
    
    # Add network security tests
    security_cmd+=" && docker run --rm --network container:jellyfin nicolaka/netshoot nmap -sT -O localhost"
    
    run_test_suite "security" "$security_cmd" 1200
}

# Generate comprehensive test report
generate_final_report() {
    log "INFO" "Generating final test report"
    
    local total_duration=0
    for duration in "${SUITE_DURATION[@]}"; do
        total_duration=$((total_duration + duration))
    done
    
    local success_rate=0
    if [[ $TOTAL_SUITES -gt 0 ]]; then
        success_rate=$(( (PASSED_SUITES + WARNING_SUITES) * 100 / TOTAL_SUITES ))
    fi
    
    # Generate JSON report (simplified to avoid quote issues)
    cat > "$FINAL_REPORT" << EOF
{
  "timestamp": "$(date -u "+%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$ENVIRONMENT",
  "quick_mode": $([ "$QUICK_MODE" == "true" ] && echo "true" || echo "false"),
  "parallel_execution": $([ "$PARALLEL_EXECUTION" == "true" ] && echo "true" || echo "false"),
  "summary": {
    "total_suites": $TOTAL_SUITES,
    "passed_suites": $PASSED_SUITES,
    "warning_suites": $WARNING_SUITES,
    "failed_suites": $FAILED_SUITES,
    "success_rate": $success_rate,
    "total_duration": $total_duration
  },
  "configuration": {
    "container_tests": $([ "$RUN_CONTAINER_TESTS" == "true" ] && echo "true" || echo "false"),
    "api_tests": $([ "$RUN_API_TESTS" == "true" ] && echo "true" || echo "false"),
    "integration_tests": $([ "$RUN_INTEGRATION_TESTS" == "true" ] && echo "true" || echo "false"),
    "performance_tests": $([ "$RUN_PERFORMANCE_TESTS" == "true" ] && echo "true" || echo "false"),
    "security_tests": $([ "$RUN_SECURITY_TESTS" == "true" ] && echo "true" || echo "false")
  },
  "artifacts": ["$LOG_FILE"]
}
EOF
    
    # Generate HTML report if requested
    if [[ "$GENERATE_HTML_REPORT" == "true" ]]; then
        generate_html_report
    fi
    
    log "INFO" "Final report generated: $FINAL_REPORT"
}

# Generate HTML test report
generate_html_report() {
    local current_date=$(date "+%Y-%m-%d %H:%M:%S %Z")
    local mode_text="Full"
    if [[ "$QUICK_MODE" == "true" ]]; then
        mode_text="Quick"
    fi
    
    # Create basic HTML report
    cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Media Server 2025 - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .suite-results { margin-top: 30px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Ultimate Media Server 2025 - Test Report</h1>
            <p>Generated: $current_date</p>
            <p>Environment: $ENVIRONMENT | Mode: $mode_text</p>
        </div>
        
        <div class="summary">
            <div class="stat-card">
                <h3>Total Suites: $TOTAL_SUITES</h3>
                <h3>Passed: $PASSED_SUITES</h3>
                <h3>Warnings: $WARNING_SUITES</h3>
                <h3>Failed: $FAILED_SUITES</h3>
            </div>
        </div>
        
        <div class="suite-results">
            <h2>Test Suite Results</h2>
            <p>Detailed results available in log files</p>
        </div>
        
        <div style="margin-top: 40px; text-align: center;">
            <p>Detailed logs and reports available in: $RESULTS_DIR</p>
        </div>
    </div>
</body>
</html>
EOF

    log "INFO" "HTML report generated: $HTML_REPORT"
}

# Cleanup function
cleanup() {
    log "INFO" "Performing cleanup"
    
    if [[ "$CLEANUP_AFTER" == "true" ]] && [[ "$ENVIRONMENT" != "production" ]]; then
        log "INFO" "Stopping test containers"
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    fi
    
    # Clean up temporary files
    find "$TEST_CONFIG_DIR" -name "*.tmp" -delete 2>/dev/null || true
}

# Main execution function
main() {
    echo -e "${BLUE}🚀 Ultimate Media Server 2025 - Master Test Runner${NC}"
    echo "==============================================="
    
    log "INFO" "Starting master test execution"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Quick mode: $QUICK_MODE"
    log "INFO" "Parallel execution: $PARALLEL_EXECUTION"
    
    # Run all phases
    check_prerequisites
    configure_environment
    setup_test_environment
    run_test_suites
    generate_final_report
    
    # Display final summary
    echo -e "\n${BLUE}📊 Final Test Summary${NC}"
    echo "======================="
    echo -e "Environment: ${CYAN}$ENVIRONMENT${NC}"
    echo -e "Total Suites: $TOTAL_SUITES"
    echo -e "${GREEN}Passed: $PASSED_SUITES${NC}"
    echo -e "${YELLOW}Warnings: $WARNING_SUITES${NC}"
    echo -e "${RED}Failed: $FAILED_SUITES${NC}"
    echo -e "Success Rate: $(( (PASSED_SUITES + WARNING_SUITES) * 100 / TOTAL_SUITES ))%"
    echo -e "Total Duration: $(( $(date +%s) - START_TIME ))s"
    
    echo -e "\n${CYAN}📋 Reports Generated:${NC}"
    echo -e "• JSON Report: $FINAL_REPORT"
    if [[ "$GENERATE_HTML_REPORT" == "true" ]]; then
        echo -e "• HTML Report: $HTML_REPORT"
    fi
    echo -e "• Log File: $LOG_FILE"
    
    # Suite breakdown
    echo -e "\n${CYAN}🔍 Suite Breakdown:${NC}"
    for suite in "${!SUITE_RESULTS[@]}"; do
        local status="${SUITE_RESULTS[$suite]}"
        local duration="${SUITE_DURATION[$suite]}"
        local icon=""
        
        case "$status" in
            "PASS") icon="${GREEN}✓${NC}" ;;
            "WARN") icon="${YELLOW}⚠${NC}" ;;
            *) icon="${RED}✗${NC}" ;;
        esac
        
        echo -e "  $icon $suite (${duration}s)"
    done
    
    # Exit with appropriate code
    if [[ $FAILED_SUITES -gt 0 ]]; then
        echo -e "\n${RED}💥 Some test suites failed. Check the reports for details.${NC}"
        exit 1
    elif [[ $WARNING_SUITES -gt 0 ]]; then
        echo -e "\n${YELLOW}⚠️  Some test suites had warnings. Review recommended.${NC}"
        exit 2
    else
        echo -e "\n${GREEN}🎉 All test suites passed successfully!${NC}"
        exit 0
    fi
}

# Set up signal handlers
START_TIME=$(date +%s)
trap cleanup EXIT

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi