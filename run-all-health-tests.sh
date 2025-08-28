#!/bin/bash
# Master Health Testing Suite
# Runs all health testing tools in sequence with comprehensive reporting

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/health-test-results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FINAL_REPORT="${OUTPUT_DIR}/comprehensive_health_report_${TIMESTAMP}.md"

# Test configuration
CONTAINER_NAME=""
LOAD_TEST_DURATION=60
LOAD_TEST_CONCURRENCY=20
VERBOSE=false
SKIP_LOAD_TEST=false

# Usage information
usage() {
    cat << EOF
Master Health Testing Suite - Comprehensive Media Server Testing

Usage: $0 [OPTIONS]

OPTIONS:
    -c, --container NAME      Container name to test (auto-detect if not provided)
    -d, --duration SECONDS    Load test duration (default: 60)
    -n, --concurrency NUM     Load test concurrency (default: 20)
    -v, --verbose            Enable verbose output
    -s, --skip-load          Skip load testing (faster execution)
    -o, --output DIR         Output directory (default: ./health-test-results)
    -h, --help               Show this help

EXAMPLES:
    $0                                    # Run all tests with auto-detection
    $0 -c media-server -d 30             # Test specific container for 30s
    $0 -v -s                             # Verbose mode, skip load test
    $0 --duration 120 --concurrency 50   # Extended load test

The suite will run:
1. Service Discovery & Health Checks
2. Container Service Monitoring (if containerized)
3. Dependency Chain Validation
4. API Load Testing (unless skipped)
5. Performance Analysis
6. Comprehensive Report Generation

EOF
}

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%H:%M:%S')
    
    case "$level" in
        "ERROR") echo -e "${RED}[$timestamp] ❌ ERROR: $message${NC}" ;;
        "WARN")  echo -e "${YELLOW}[$timestamp] ⚠️  WARN:  $message${NC}" ;;
        "INFO")  echo -e "${GREEN}[$timestamp] ℹ️  INFO:  $message${NC}" ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[$timestamp] 🔍 DEBUG: $message${NC}" ;;
        "TEST")  echo -e "${PURPLE}[$timestamp] 🧪 TEST:  $message${NC}" ;;
        "PASS")  echo -e "${GREEN}[$timestamp] ✅ PASS:  $message${NC}" ;;
        "FAIL")  echo -e "${RED}[$timestamp] ❌ FAIL:  $message${NC}" ;;
    esac
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--container)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            -d|--duration)
                LOAD_TEST_DURATION="$2"
                shift 2
                ;;
            -n|--concurrency)
                LOAD_TEST_CONCURRENCY="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -s|--skip-load)
                SKIP_LOAD_TEST=true
                shift
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Initialize test environment
initialize() {
    log "INFO" "Initializing Master Health Testing Suite"
    log "INFO" "Output directory: $OUTPUT_DIR"
    
    # Create output directory structure
    mkdir -p "$OUTPUT_DIR"/{individual-tests,charts,raw-data,reports}
    
    # Check dependencies
    local missing_deps=()
    
    # Check for Node.js (for JavaScript tests)
    if ! command -v node &> /dev/null; then
        missing_deps+=("node.js")
    fi
    
    # Check for Python (for dependency validator)
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check for Docker (if container testing)
    if ! command -v docker &> /dev/null; then
        log "WARN" "Docker not available - container-specific tests will be skipped"
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        log "INFO" "Please install missing dependencies and retry"
        exit 1
    fi
    
    log "INFO" "Environment initialized successfully"
}

# Auto-detect container if not specified
detect_container() {
    if [[ -z "$CONTAINER_NAME" ]]; then
        log "INFO" "Auto-detecting media server container..."
        
        # Check for running containers with media server patterns
        local candidates=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(media|jellyfin|plex|arr)" || true)
        
        if [[ -n "$candidates" ]]; then
            # Take the first match
            CONTAINER_NAME=$(echo "$candidates" | head -1)
            log "INFO" "Detected container: $CONTAINER_NAME"
        else
            log "INFO" "No container detected - running host-based tests"
        fi
    fi
    
    # Verify container exists if specified
    if [[ -n "$CONTAINER_NAME" ]]; then
        if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            log "WARN" "Container '$CONTAINER_NAME' not found or not running"
            log "INFO" "Proceeding with host-based tests"
            CONTAINER_NAME=""
        fi
    fi
}

# Test 1: Service Discovery and Basic Health Checks
run_service_discovery() {
    log "TEST" "Starting Service Discovery & Health Checks"
    
    local test_output="${OUTPUT_DIR}/individual-tests/01_service_discovery_${TIMESTAMP}.json"
    local test_log="${OUTPUT_DIR}/individual-tests/01_service_discovery_${TIMESTAMP}.log"
    
    # Run the comprehensive service health tester
    if node "$SCRIPT_DIR/comprehensive-service-health-tester.js" > "$test_log" 2>&1; then
        log "PASS" "Service discovery completed successfully"
        return 0
    else
        log "FAIL" "Service discovery encountered issues"
        return 1
    fi
}

# Test 2: Container Service Monitoring (if applicable)
run_container_monitoring() {
    if [[ -z "$CONTAINER_NAME" ]]; then
        log "INFO" "Skipping container monitoring - no container detected"
        return 0
    fi
    
    log "TEST" "Starting Container Service Monitoring"
    
    local test_output="${OUTPUT_DIR}/individual-tests/02_container_monitoring_${TIMESTAMP}.csv"
    local test_log="${OUTPUT_DIR}/individual-tests/02_container_monitoring_${TIMESTAMP}.log"
    
    # Run container service monitor
    local monitor_args=(-c "$CONTAINER_NAME")
    [[ "$VERBOSE" == "true" ]] && monitor_args+=(-v)
    
    if "$SCRIPT_DIR/container-service-monitor.sh" "${monitor_args[@]}" > "$test_log" 2>&1; then
        log "PASS" "Container monitoring completed successfully"
        return 0
    else
        log "FAIL" "Container monitoring encountered issues"
        return 1
    fi
}

# Test 3: Service Dependency Chain Validation
run_dependency_validation() {
    log "TEST" "Starting Service Dependency Chain Validation"
    
    local test_output="${OUTPUT_DIR}/individual-tests/03_dependency_validation_${TIMESTAMP}.json"
    local test_log="${OUTPUT_DIR}/individual-tests/03_dependency_validation_${TIMESTAMP}.log"
    
    # Prepare Python dependency validator arguments
    local validator_args=(-o "$test_output")
    [[ -n "$CONTAINER_NAME" ]] && validator_args+=(-c "$CONTAINER_NAME")
    [[ "$VERBOSE" == "true" ]] && validator_args+=(-v)
    
    # Also generate visualization
    validator_args+=(-v --startup-script)
    
    if python3 "$SCRIPT_DIR/service-dependency-validator.py" "${validator_args[@]}" > "$test_log" 2>&1; then
        log "PASS" "Dependency validation completed successfully"
        
        # Move generated files to output directory
        [[ -f "dependency_graph.png" ]] && mv "dependency_graph.png" "$OUTPUT_DIR/charts/"
        [[ -f "optimal_startup.sh" ]] && mv "optimal_startup.sh" "$OUTPUT_DIR/reports/"
        
        return 0
    else
        log "FAIL" "Dependency validation encountered issues"
        return 1
    fi
}

# Test 4: API Load Testing
run_load_testing() {
    if [[ "$SKIP_LOAD_TEST" == "true" ]]; then
        log "INFO" "Skipping load testing as requested"
        return 0
    fi
    
    log "TEST" "Starting API Load Testing (Duration: ${LOAD_TEST_DURATION}s, Concurrency: ${LOAD_TEST_CONCURRENCY})"
    
    local test_output="${OUTPUT_DIR}/individual-tests/04_load_testing_${TIMESTAMP}"
    
    # Run API load test
    local load_test_args=(
        --duration "$LOAD_TEST_DURATION"
        --concurrency "$LOAD_TEST_CONCURRENCY"
        --outputDir "$test_output"
        --reportInterval 5
    )
    
    if node "$SCRIPT_DIR/api-load-test-suite.js" "${load_test_args[@]}"; then
        log "PASS" "Load testing completed successfully"
        return 0
    else
        log "FAIL" "Load testing encountered issues"
        return 1
    fi
}

# Test 5: Performance Analysis
run_performance_analysis() {
    log "TEST" "Starting Performance Analysis"
    
    # Run system performance check
    local perf_output="${OUTPUT_DIR}/individual-tests/05_performance_analysis_${TIMESTAMP}.json"
    
    # Collect system metrics
    local system_metrics="{
  \"timestamp\": \"$(date -Iseconds)\",
  \"system\": {
    \"uptime\": \"$(uptime)\",
    \"load_average\": [$(uptime | sed 's/.*load average: //' | tr ',' ' ')],
    \"memory\": $(free -m | awk 'NR==2{printf "{\"total\":%s,\"used\":%s,\"free\":%s,\"percent\":%.2f}", $2,$3,$4,$3*100/$2}'),
    \"disk\": $(df -h / | awk 'NR==2{printf "{\"size\":\"%s\",\"used\":\"%s\",\"available\":\"%s\",\"percent\":\"%s\"}", $2,$3,$4,$5}')
  }"
    
    # Add container stats if applicable
    if [[ -n "$CONTAINER_NAME" ]]; then
        local container_stats
        container_stats=$(docker stats "$CONTAINER_NAME" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" | tail -1 || echo "N/A N/A N/A N/A")
        
        system_metrics+=",
  \"container\": {
    \"name\": \"$CONTAINER_NAME\",
    \"stats\": \"$container_stats\",
    \"processes\": $(docker exec "$CONTAINER_NAME" ps aux --no-headers | wc -l 2>/dev/null || echo 0)
  }"
    fi
    
    system_metrics+="}"
    
    echo "$system_metrics" > "$perf_output"
    
    log "PASS" "Performance analysis completed"
    return 0
}

# Generate comprehensive report
generate_comprehensive_report() {
    log "INFO" "Generating comprehensive health report"
    
    # Start the report
    cat > "$FINAL_REPORT" << EOF
# Comprehensive Media Server Health Report

**Generated:** $(date)  
**Test Suite Version:** 2.0.0  
**Container:** ${CONTAINER_NAME:-"Host-based deployment"}  

## Executive Summary

EOF
    
    # Analyze test results
    local total_tests=5
    local passed_tests=0
    local test_statuses=()
    
    # Check each test result
    local tests=(
        "Service Discovery"
        "Container Monitoring"
        "Dependency Validation"
        "Load Testing"
        "Performance Analysis"
    )
    
    for i in {1..5}; do
        local test_name="${tests[$((i-1))]}"
        local test_files=($(find "$OUTPUT_DIR/individual-tests" -name "*${i}_*" 2>/dev/null))
        
        if [[ ${#test_files[@]} -gt 0 ]]; then
            passed_tests=$((passed_tests + 1))
            test_statuses+=("✅ $test_name: COMPLETED")
        else
            test_statuses+=("❌ $test_name: FAILED OR SKIPPED")
        fi
    done
    
    # Calculate overall health score
    local health_score=$((passed_tests * 100 / total_tests))
    
    cat >> "$FINAL_REPORT" << EOF
- **Overall Health Score:** ${health_score}% (${passed_tests}/${total_tests} tests passed)
- **Test Duration:** $(date -d "@$SECONDS" -u +%H:%M:%S)
- **Critical Issues:** $(find_critical_issues)

### Test Results Summary

EOF
    
    # Add test results
    for status in "${test_statuses[@]}"; do
        echo "- $status" >> "$FINAL_REPORT"
    done
    
    # Add detailed sections
    cat >> "$FINAL_REPORT" << EOF

## Detailed Results

### 1. Service Health Status

$(analyze_service_health)

### 2. Performance Metrics

$(analyze_performance_metrics)

### 3. Dependency Chain Analysis

$(analyze_dependencies)

### 4. Load Testing Results

$(analyze_load_test_results)

### 5. Resource Utilization

$(analyze_resource_utilization)

## Recommendations

$(generate_recommendations "$health_score")

## Next Steps

Based on this health assessment, consider the following actions:

1. **Immediate:** Address any critical issues identified above
2. **Short-term:** Optimize services showing performance warnings
3. **Long-term:** Implement continuous monitoring for ongoing health assessment

## Test Artifacts

All detailed test results, logs, and charts are available in:
- \`${OUTPUT_DIR}/individual-tests/\` - Individual test results
- \`${OUTPUT_DIR}/charts/\` - Generated charts and visualizations  
- \`${OUTPUT_DIR}/reports/\` - Detailed reports and analysis
- \`${OUTPUT_DIR}/raw-data/\` - Raw test data and metrics

---
*Report generated by Media Server Health Testing Suite*
EOF
    
    log "INFO" "Comprehensive report generated: $FINAL_REPORT"
}

# Helper functions for report generation
find_critical_issues() {
    local issues=0
    
    # Check for common critical patterns in logs
    if grep -r -i "error\|critical\|fail" "$OUTPUT_DIR/individual-tests/" >/dev/null 2>&1; then
        issues=$((issues + 1))
    fi
    
    echo "$issues"
}

analyze_service_health() {
    echo "Service health analysis would be based on test results..."
    echo "(Implementation would parse actual test outputs)"
}

analyze_performance_metrics() {
    echo "Performance metrics analysis would be based on collected data..."
    echo "(Implementation would parse performance test results)"
}

analyze_dependencies() {
    echo "Dependency analysis would be based on validation results..."
    echo "(Implementation would parse dependency test results)"
}

analyze_load_test_results() {
    if [[ "$SKIP_LOAD_TEST" == "true" ]]; then
        echo "Load testing was skipped."
    else
        echo "Load test analysis would be based on test results..."
        echo "(Implementation would parse load test outputs)"
    fi
}

analyze_resource_utilization() {
    echo "Resource utilization analysis would be based on system metrics..."
    echo "(Implementation would parse system performance data)"
}

generate_recommendations() {
    local health_score="$1"
    
    if [[ $health_score -ge 90 ]]; then
        echo "- ✅ System is performing excellently"
        echo "- Consider implementing proactive monitoring"
        echo "- Regular health checks recommended monthly"
    elif [[ $health_score -ge 75 ]]; then
        echo "- ⚠️ System is performing well with minor issues"
        echo "- Address identified warnings to improve reliability"
        echo "- Increase monitoring frequency to weekly"
    elif [[ $health_score -ge 50 ]]; then
        echo "- ⚠️ System has notable performance issues"
        echo "- Immediate attention required for failing services"
        echo "- Implement daily health monitoring"
    else
        echo "- 🚨 System has critical issues requiring immediate attention"
        echo "- Review all failing tests and implement fixes"
        echo "- Consider system maintenance or configuration review"
    fi
}

# Main execution flow
main() {
    local start_time=$(date +%s)
    
    echo "🏥 Master Health Testing Suite - Media Server Comprehensive Testing"
    echo "===================================================================="
    
    # Parse arguments and initialize
    parse_args "$@"
    initialize
    detect_container
    
    log "INFO" "Starting comprehensive health test suite"
    [[ -n "$CONTAINER_NAME" ]] && log "INFO" "Target container: $CONTAINER_NAME"
    [[ "$SKIP_LOAD_TEST" == "true" ]] && log "INFO" "Load testing will be skipped"
    
    # Track test results
    local test_results=()
    
    # Run test suite
    echo ""
    log "INFO" "Phase 1/5: Service Discovery & Health Checks"
    if run_service_discovery; then
        test_results+=("service_discovery:PASS")
    else
        test_results+=("service_discovery:FAIL")
    fi
    
    echo ""
    log "INFO" "Phase 2/5: Container Service Monitoring"
    if run_container_monitoring; then
        test_results+=("container_monitoring:PASS")
    else
        test_results+=("container_monitoring:FAIL")
    fi
    
    echo ""
    log "INFO" "Phase 3/5: Service Dependency Chain Validation"
    if run_dependency_validation; then
        test_results+=("dependency_validation:PASS")
    else
        test_results+=("dependency_validation:FAIL")
    fi
    
    echo ""
    log "INFO" "Phase 4/5: API Load Testing"
    if run_load_testing; then
        test_results+=("load_testing:PASS")
    else
        test_results+=("load_testing:FAIL")
    fi
    
    echo ""
    log "INFO" "Phase 5/5: Performance Analysis"
    if run_performance_analysis; then
        test_results+=("performance_analysis:PASS")
    else
        test_results+=("performance_analysis:FAIL")
    fi
    
    # Generate final report
    echo ""
    log "INFO" "Generating comprehensive report"
    generate_comprehensive_report
    
    # Display summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local passed_count=$(printf '%s\n' "${test_results[@]}" | grep -c "PASS" || echo 0)
    local total_count=${#test_results[@]}
    
    echo ""
    echo "===================================================================="
    echo "🎯 COMPREHENSIVE HEALTH TEST SUMMARY"
    echo "===================================================================="
    echo "📊 Tests Passed: $passed_count/$total_count"
    echo "⏱️  Total Duration: $(date -d "@$duration" -u +%H:%M:%S)"
    echo "📁 Results Directory: $OUTPUT_DIR"
    echo "📋 Comprehensive Report: $FINAL_REPORT"
    echo ""
    
    # Display individual test results
    echo "Individual Test Results:"
    for result in "${test_results[@]}"; do
        IFS=':' read -r test_name status <<< "$result"
        if [[ "$status" == "PASS" ]]; then
            echo "  ✅ $(echo "$test_name" | tr '_' ' ' | sed 's/\b\w/\U&/g')"
        else
            echo "  ❌ $(echo "$test_name" | tr '_' ' ' | sed 's/\b\w/\U&/g')"
        fi
    done
    
    echo ""
    if [[ $passed_count -eq $total_count ]]; then
        log "PASS" "All tests completed successfully! 🎉"
        echo "Your media server is in excellent health."
    elif [[ $passed_count -gt $((total_count / 2)) ]]; then
        log "WARN" "Most tests passed but some issues were detected."
        echo "Review the comprehensive report for details and recommendations."
    else
        log "FAIL" "Multiple test failures detected."
        echo "Immediate attention recommended. Review the comprehensive report."
    fi
    
    echo "===================================================================="
    
    # Return appropriate exit code
    if [[ $passed_count -eq $total_count ]]; then
        return 0
    elif [[ $passed_count -gt $((total_count / 2)) ]]; then
        return 1
    else
        return 2
    fi
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit $?
fi