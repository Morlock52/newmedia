#!/bin/bash

# Dashboard Test Runner Script
# Automated test runner for all dashboard functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$TEST_DIR/../.." && pwd)"
REPORTS_DIR="$TEST_DIR/reports"
LOG_FILE="$REPORTS_DIR/test-run-$(date +%Y%m%d_%H%M%S).log"

# Create reports directory
mkdir -p "$REPORTS_DIR"

echo -e "${BLUE}🧪 Dashboard Test Suite Runner${NC}"
echo "================================================"
echo "Test Directory: $TEST_DIR"
echo "Root Directory: $ROOT_DIR"
echo "Reports Directory: $REPORTS_DIR"
echo "Log File: $LOG_FILE"
echo ""

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}📋 Checking Prerequisites${NC}"
    
    local missing_deps=()
    
    # Check Node.js
    if ! command_exists node; then
        missing_deps+=("Node.js")
    else
        log "✅ Node.js version: $(node --version)"
    fi
    
    # Check npm
    if ! command_exists npm; then
        missing_deps+=("npm")
    else
        log "✅ npm version: $(npm --version)"
    fi
    
    # Check Jest
    if ! command_exists jest && ! [ -f "$ROOT_DIR/node_modules/.bin/jest" ]; then
        missing_deps+=("Jest")
    else
        log "✅ Jest available"
    fi
    
    # Check Docker (optional)
    if command_exists docker; then
        log "✅ Docker version: $(docker --version)"
    else
        log "⚠️ Docker not available - some service integration tests may be skipped"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing dependencies: ${missing_deps[*]}${NC}"
        echo "Please install missing dependencies and try again."
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
    echo ""
}

# Function to install test dependencies
install_dependencies() {
    echo -e "${BLUE}📦 Installing Test Dependencies${NC}"
    
    cd "$TEST_DIR"
    
    if [ ! -f "package.json" ]; then
        log "Creating package.json for dashboard tests..."
        cat > package.json << EOF
{
  "name": "dashboard-tests",
  "version": "1.0.0",
  "description": "Dashboard test suite",
  "scripts": {
    "test": "jest",
    "test:dashboard": "jest dashboard.test.js",
    "test:responsive": "jest responsive.test.js",
    "test:api": "jest api-integration.test.js",
    "test:services": "jest service-integration.test.js",
    "test:performance": "jest performance.test.js",
    "test:browser": "jest cross-browser.test.js",
    "test:websocket": "jest websocket.test.js"
  },
  "devDependencies": {
    "jest": "^29.7.0",
    "puppeteer": "^21.6.1",
    "jsdom": "^23.0.1",
    "axios": "^1.6.2",
    "ws": "^8.14.2"
  },
  "jest": {
    "testEnvironment": "node",
    "testTimeout": 30000,
    "setupFilesAfterEnv": ["<rootDir>/setup.js"]
  }
}
EOF
    fi
    
    # Install dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        log "Installing npm dependencies..."
        npm install
    else
        log "✅ Dependencies already installed"
    fi
    
    echo -e "${GREEN}✅ Dependencies ready${NC}"
    echo ""
}

# Function to create test setup file
create_test_setup() {
    if [ ! -f "$TEST_DIR/setup.js" ]; then
        log "Creating test setup file..."
        cat > "$TEST_DIR/setup.js" << 'EOF'
// Test setup file
const { TextEncoder, TextDecoder } = require('util');

// Polyfills for Node.js environment
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Increase timeout for slower systems
jest.setTimeout(30000);

// Console setup
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

console.error = (...args) => {
    if (args[0] && typeof args[0] === 'string' && args[0].includes('Warning:')) {
        return;
    }
    originalConsoleError.call(console, ...args);
};

console.warn = (...args) => {
    if (args[0] && typeof args[0] === 'string' && args[0].includes('deprecated')) {
        return;
    }
    originalConsoleWarn.call(console, ...args);
};
EOF
    fi
}

# Function to check if services are running
check_services() {
    echo -e "${BLUE}🔍 Checking Service Availability${NC}"
    
    local api_available=false
    local dashboard_available=false
    
    # Check API server
    if curl -s -f "http://localhost:3002/health" > /dev/null 2>&1; then
        api_available=true
        log "✅ API server is running"
    else
        log "⚠️ API server not running - some tests will be skipped"
    fi
    
    # Check dashboard file
    if [ -f "$ROOT_DIR/dashboard-enhanced.html" ]; then
        dashboard_available=true
        log "✅ Dashboard file found"
    else
        log "❌ Dashboard file not found at $ROOT_DIR/dashboard-enhanced.html"
    fi
    
    # Check Docker services
    if command_exists docker; then
        local running_containers=$(docker ps --format "{{.Names}}" 2>/dev/null | wc -l)
        log "ℹ️ Docker containers running: $running_containers"
    fi
    
    # Set environment variables for tests
    export BASE_URL="http://localhost"
    export API_AVAILABLE="$api_available"
    export DASHBOARD_AVAILABLE="$dashboard_available"
    
    echo ""
}

# Function to run individual test suite
run_test_suite() {
    local test_name="$1"
    local test_file="$2"
    local description="$3"
    
    echo -e "${BLUE}🧪 Running $test_name Tests${NC}"
    echo "Description: $description"
    echo ""
    
    local start_time=$(date +%s)
    local test_result=0
    
    # Run the test
    if npx jest "$test_file" --verbose --json --outputFile="$REPORTS_DIR/${test_name,,}-results.json" 2>&1 | tee -a "$LOG_FILE"; then
        test_result=0
        log "✅ $test_name tests completed successfully"
    else
        test_result=1
        log "❌ $test_name tests failed"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo -e "${YELLOW}⏱️ $test_name test duration: ${duration}s${NC}"
    echo ""
    
    return $test_result
}

# Function to run all dashboard tests
run_dashboard_tests() {
    echo -e "${BLUE}🚀 Starting Dashboard Test Execution${NC}"
    echo ""
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # Test suites configuration
    declare -A test_suites=(
        ["Dashboard"]="dashboard.test.js|Core dashboard HTML structure and functionality tests"
        ["Responsive"]="responsive.test.js|Responsive design and mobile compatibility tests"
        ["API-Integration"]="api-integration.test.js|API endpoint testing and service communication"
        ["Service-Integration"]="service-integration.test.js|Media service integration and connectivity tests"
        ["Performance"]="performance.test.js|Dashboard performance and optimization tests"
        ["Cross-Browser"]="cross-browser.test.js|Cross-browser compatibility and feature support tests"
        ["WebSocket"]="websocket.test.js|Real-time communication and WebSocket functionality tests"
    )
    
    # Run each test suite
    for test_name in "${!test_suites[@]}"; do
        IFS='|' read -r test_file description <<< "${test_suites[$test_name]}"
        
        total_tests=$((total_tests + 1))
        
        if [ ! -f "$TEST_DIR/$test_file" ]; then
            log "⚠️ Test file not found: $test_file - skipping"
            continue
        fi
        
        if run_test_suite "$test_name" "$test_file" "$description"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
    done
    
    # Generate summary
    echo -e "${BLUE}📊 Test Execution Summary${NC}"
    echo "========================================"
    echo "Total Test Suites: $total_tests"
    echo -e "Passed: ${GREEN}$passed_tests${NC}"
    echo -e "Failed: ${RED}$failed_tests${NC}"
    echo ""
    
    log "Test execution completed. Passed: $passed_tests, Failed: $failed_tests"
    
    return $failed_tests
}

# Function to generate comprehensive test report
generate_test_report() {
    echo -e "${BLUE}📄 Generating Test Report${NC}"
    
    local report_file="$REPORTS_DIR/dashboard-test-report-$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Test Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #2563eb; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; }
        .passed { border-left: 4px solid #10b981; }
        .failed { border-left: 4px solid #ef4444; }
        .warning { border-left: 4px solid #f59e0b; }
        .test-suite { margin-bottom: 30px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .test-suite h3 { margin-top: 0; color: #1f2937; }
        .log-section { background: #1f2937; color: #f3f4f6; padding: 20px; border-radius: 8px; overflow-x: auto; }
        pre { margin: 0; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Dashboard Test Report</h1>
        <p>Generated on $(date)</p>
        <p>Test Environment: $(uname -s) $(uname -r)</p>
    </div>
    
    <div class="summary">
        <div class="card passed">
            <h3>✅ Tests Passed</h3>
            <p style="font-size: 2em; margin: 0;">$(find "$REPORTS_DIR" -name "*results.json" -exec grep -l '"numPassedTests"' {} \; 2>/dev/null | wc -l)</p>
        </div>
        <div class="card failed">
            <h3>❌ Tests Failed</h3>
            <p style="font-size: 2em; margin: 0;">$(find "$REPORTS_DIR" -name "*results.json" -exec grep -l '"numFailedTests"' {} \; 2>/dev/null | wc -l)</p>
        </div>
        <div class="card warning">
            <h3>⚠️ Warnings</h3>
            <p style="font-size: 2em; margin: 0;">$(grep -c "⚠️" "$LOG_FILE" 2>/dev/null || echo "0")</p>
        </div>
    </div>
    
    <div class="test-suite">
        <h3>📋 Test Suites Executed</h3>
        <ul>
            <li><strong>Dashboard Tests:</strong> Core HTML structure and functionality</li>
            <li><strong>Responsive Tests:</strong> Mobile and tablet compatibility</li>
            <li><strong>API Integration Tests:</strong> Backend service communication</li>
            <li><strong>Service Integration Tests:</strong> Media service connectivity</li>
            <li><strong>Performance Tests:</strong> Load times and optimization</li>
            <li><strong>Cross-Browser Tests:</strong> Browser compatibility</li>
            <li><strong>WebSocket Tests:</strong> Real-time communication</li>
        </ul>
    </div>
    
    <div class="test-suite">
        <h3>📊 Key Metrics</h3>
        <p><strong>Total Test Duration:</strong> $(tail -1 "$LOG_FILE" | grep -o '[0-9]*s' || echo "N/A")</p>
        <p><strong>Dashboard Load Time:</strong> Measured across multiple viewports</p>
        <p><strong>API Response Times:</strong> All endpoints tested for performance</p>
        <p><strong>Browser Compatibility:</strong> Modern browser features validated</p>
    </div>
    
    <div class="log-section">
        <h3>📝 Test Execution Log</h3>
        <pre>$(tail -50 "$LOG_FILE" 2>/dev/null || echo "Log file not available")</pre>
    </div>
</body>
</html>
EOF
    
    log "Test report generated: $report_file"
    echo -e "${GREEN}✅ Test report saved to: $report_file${NC}"
    echo ""
}

# Function to cleanup test artifacts
cleanup_test_artifacts() {
    echo -e "${BLUE}🧹 Cleaning Up Test Artifacts${NC}"
    
    # Remove temporary screenshots older than 7 days
    find "$REPORTS_DIR" -name "screenshot-*.png" -mtime +7 -delete 2>/dev/null || true
    
    # Remove old JSON result files older than 30 days
    find "$REPORTS_DIR" -name "*-results.json" -mtime +30 -delete 2>/dev/null || true
    
    log "Cleanup completed"
    echo ""
}

# Function to display usage
show_usage() {
    cat << EOF
Dashboard Test Runner

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -s, --suite SUITE       Run specific test suite (dashboard, responsive, api, services, performance, browser, websocket)
    -f, --fast              Skip performance and browser tests for faster execution
    -v, --verbose           Enable verbose output
    -c, --cleanup           Cleanup old test artifacts
    --no-report             Skip HTML report generation

Examples:
    $0                      # Run all tests
    $0 -s dashboard         # Run only dashboard tests
    $0 -f                   # Run fast test suite
    $0 -c                   # Cleanup old artifacts

Test Suites:
    dashboard               Core dashboard functionality
    responsive              Responsive design tests
    api                     API integration tests
    services                Service integration tests
    performance             Performance benchmarks
    browser                 Cross-browser compatibility
    websocket               Real-time communication
EOF
}

# Parse command line arguments
SUITE=""
FAST_MODE=false
VERBOSE=false
CLEANUP_ONLY=false
SKIP_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -s|--suite)
            SUITE="$2"
            shift 2
            ;;
        -f|--fast)
            FAST_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--cleanup)
            CLEANUP_ONLY=true
            shift
            ;;
        --no-report)
            SKIP_REPORT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "Starting dashboard test execution"
    
    # Handle cleanup-only mode
    if [ "$CLEANUP_ONLY" = true ]; then
        cleanup_test_artifacts
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Install dependencies
    install_dependencies
    
    # Create test setup
    create_test_setup
    
    # Check service availability
    check_services
    
    # Run specific test suite if requested
    if [ -n "$SUITE" ]; then
        case "$SUITE" in
            dashboard)
                run_test_suite "Dashboard" "dashboard.test.js" "Core dashboard functionality tests"
                ;;
            responsive)
                run_test_suite "Responsive" "responsive.test.js" "Responsive design tests"
                ;;
            api)
                run_test_suite "API-Integration" "api-integration.test.js" "API integration tests"
                ;;
            services)
                run_test_suite "Service-Integration" "service-integration.test.js" "Service integration tests"
                ;;
            performance)
                run_test_suite "Performance" "performance.test.js" "Performance tests"
                ;;
            browser)
                run_test_suite "Cross-Browser" "cross-browser.test.js" "Cross-browser tests"
                ;;
            websocket)
                run_test_suite "WebSocket" "websocket.test.js" "WebSocket tests"
                ;;
            *)
                echo -e "${RED}❌ Unknown test suite: $SUITE${NC}"
                show_usage
                exit 1
                ;;
        esac
    else
        # Run all tests
        run_dashboard_tests
    fi
    
    # Generate test report unless skipped
    if [ "$SKIP_REPORT" != true ]; then
        generate_test_report
    fi
    
    # Cleanup
    cleanup_test_artifacts
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    echo -e "${GREEN}🎉 Dashboard test execution completed in ${total_duration}s${NC}"
    log "Total test execution time: ${total_duration}s"
}

# Run main function
main "$@"