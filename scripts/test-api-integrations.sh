#!/bin/bash

# Media Server API Integration Test Suite
# Tests all API endpoints and service integrations
# Author: API Integration Specialist
# Date: $(date)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load API keys if available
if [ -f "./api-keys.env" ]; then
    source "./api-keys.env"
fi

# Service URLs
EXTERNAL_BASE="${EXTERNAL_BASE:-http://localhost}"
JELLYFIN_URL="${EXTERNAL_BASE}:8096"
PLEX_URL="${EXTERNAL_BASE}:32400"
SONARR_URL="${EXTERNAL_BASE}:8989"
RADARR_URL="${EXTERNAL_BASE}:7878"
LIDARR_URL="${EXTERNAL_BASE}:8686"
PROWLARR_URL="${EXTERNAL_BASE}:9696"
BAZARR_URL="${EXTERNAL_BASE}:6767"
JELLYSEERR_URL="${EXTERNAL_BASE}:5055"
OVERSEERR_URL="${EXTERNAL_BASE}:5056"
QBITTORRENT_URL="${EXTERNAL_BASE}:8080"
TRANSMISSION_URL="${EXTERNAL_BASE}:9091"
SABNZBD_URL="${EXTERNAL_BASE}:8081"

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

test_result() {
    local test_name="$1"
    local result="$2"
    local message="$3"
    
    ((TOTAL_TESTS++))
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC} - $test_name: $message"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ FAIL${NC} - $test_name: $message"
        ((FAILED_TESTS++))
    fi
}

# Generic API test function
test_api_endpoint() {
    local service_name="$1"
    local url="$2"
    local api_key="$3"
    local endpoint="$4"
    local expected_status="${5:-200}"
    
    local full_url="${url}${endpoint}"
    local headers=""
    
    if [ -n "$api_key" ]; then
        headers="-H X-Api-Key:$api_key"
    fi
    
    local response
    local status_code
    
    if response=$(curl -s -w "%{http_code}" $headers "$full_url" 2>/dev/null); then
        status_code="${response: -3}"
        response="${response%???}"
        
        if [ "$status_code" = "$expected_status" ]; then
            test_result "$service_name API" "PASS" "$endpoint responded with $status_code"
            return 0
        else
            test_result "$service_name API" "FAIL" "$endpoint responded with $status_code (expected $expected_status)"
            return 1
        fi
    else
        test_result "$service_name API" "FAIL" "$endpoint is not accessible"
        return 1
    fi
}

# Test service connectivity
test_service_connectivity() {
    local service_name="$1"
    local url="$2"
    
    if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
        test_result "$service_name Connectivity" "PASS" "Service is accessible at $url"
        return 0
    else
        test_result "$service_name Connectivity" "FAIL" "Service is not accessible at $url"
        return 1
    fi
}

# Test *arr service APIs
test_arr_services() {
    log "Testing *arr service APIs"
    
    # Test Sonarr
    if [ -n "${SONARR_API_KEY:-}" ]; then
        test_api_endpoint "Sonarr" "$SONARR_URL" "$SONARR_API_KEY" "/api/v3/system/status"
        test_api_endpoint "Sonarr" "$SONARR_URL" "$SONARR_API_KEY" "/api/v3/series" 200
        test_api_endpoint "Sonarr" "$SONARR_URL" "$SONARR_API_KEY" "/api/v3/downloadclient"
        test_api_endpoint "Sonarr" "$SONARR_URL" "$SONARR_API_KEY" "/api/v3/indexer"
    else
        test_result "Sonarr API" "FAIL" "API key not available"
    fi
    
    # Test Radarr
    if [ -n "${RADARR_API_KEY:-}" ]; then
        test_api_endpoint "Radarr" "$RADARR_URL" "$RADARR_API_KEY" "/api/v3/system/status"
        test_api_endpoint "Radarr" "$RADARR_URL" "$RADARR_API_KEY" "/api/v3/movie" 200
        test_api_endpoint "Radarr" "$RADARR_URL" "$RADARR_API_KEY" "/api/v3/downloadclient"
        test_api_endpoint "Radarr" "$RADARR_URL" "$RADARR_API_KEY" "/api/v3/indexer"
    else
        test_result "Radarr API" "FAIL" "API key not available"
    fi
    
    # Test Lidarr
    if [ -n "${LIDARR_API_KEY:-}" ]; then
        test_api_endpoint "Lidarr" "$LIDARR_URL" "$LIDARR_API_KEY" "/api/v1/system/status"
        test_api_endpoint "Lidarr" "$LIDARR_URL" "$LIDARR_API_KEY" "/api/v1/artist" 200
    else
        test_result "Lidarr API" "FAIL" "API key not available"
    fi
    
    # Test Prowlarr
    if [ -n "${PROWLARR_API_KEY:-}" ]; then
        test_api_endpoint "Prowlarr" "$PROWLARR_URL" "$PROWLARR_API_KEY" "/api/v1/system/status"
        test_api_endpoint "Prowlarr" "$PROWLARR_URL" "$PROWLARR_API_KEY" "/api/v1/indexer"
        test_api_endpoint "Prowlarr" "$PROWLARR_URL" "$PROWLARR_API_KEY" "/api/v1/applications"
    else
        test_result "Prowlarr API" "FAIL" "API key not available"
    fi
    
    # Test Bazarr
    if [ -n "${BAZARR_API_KEY:-}" ]; then
        test_api_endpoint "Bazarr" "$BAZARR_URL" "$BAZARR_API_KEY" "/api/system/status"
        test_api_endpoint "Bazarr" "$BAZARR_URL" "$BAZARR_API_KEY" "/api/series"
        test_api_endpoint "Bazarr" "$BAZARR_URL" "$BAZARR_API_KEY" "/api/movies"
    else
        test_result "Bazarr API" "FAIL" "API key not available"
    fi
}

# Test media server APIs
test_media_servers() {
    log "Testing media server APIs"
    
    # Test Jellyfin
    test_service_connectivity "Jellyfin" "$JELLYFIN_URL"
    test_api_endpoint "Jellyfin" "$JELLYFIN_URL" "" "/System/Info/Public" 200
    
    if [ -n "${JELLYFIN_API_KEY:-}" ]; then
        test_api_endpoint "Jellyfin" "$JELLYFIN_URL" "$JELLYFIN_API_KEY" "/System/Info" 200
        test_api_endpoint "Jellyfin" "$JELLYFIN_URL" "$JELLYFIN_API_KEY" "/Users" 200
    else
        warn "Jellyfin API key not available for authenticated tests"
    fi
    
    # Test Plex
    test_service_connectivity "Plex" "$PLEX_URL"
    test_api_endpoint "Plex" "$PLEX_URL" "" "/identity" 200
    
    if [ -n "${PLEX_TOKEN:-}" ]; then
        test_api_endpoint "Plex" "$PLEX_URL" "" "/library/sections?X-Plex-Token=$PLEX_TOKEN" 200
    else
        warn "Plex token not available for authenticated tests"
    fi
}

# Test download clients
test_download_clients() {
    log "Testing download client APIs"
    
    # Test qBittorrent
    test_service_connectivity "qBittorrent" "$QBITTORRENT_URL"
    test_api_endpoint "qBittorrent" "$QBITTORRENT_URL" "" "/api/v2/app/version" 200
    
    # Test transmission (through VPN)
    test_service_connectivity "Transmission" "$TRANSMISSION_URL"
    
    # Test SABnzbd
    test_service_connectivity "SABnzbd" "$SABNZBD_URL"
    test_api_endpoint "SABnzbd" "$SABNZBD_URL" "" "/api?mode=version&output=json" 200
}

# Test request management services
test_request_services() {
    log "Testing request management services"
    
    # Test Jellyseerr
    test_service_connectivity "Jellyseerr" "$JELLYSEERR_URL"
    test_api_endpoint "Jellyseerr" "$JELLYSEERR_URL" "" "/api/v1/status" 200
    
    # Test Overseerr
    test_service_connectivity "Overseerr" "$OVERSEERR_URL"
    test_api_endpoint "Overseerr" "$OVERSEERR_URL" "" "/api/v1/status" 200
}

# Test service integrations
test_integrations() {
    log "Testing service integrations"
    
    # Test Prowlarr applications
    if [ -n "${PROWLARR_API_KEY:-}" ]; then
        local apps_response
        if apps_response=$(curl -s -H "X-Api-Key:$PROWLARR_API_KEY" "$PROWLARR_URL/api/v1/applications" 2>/dev/null); then
            local app_count=$(echo "$apps_response" | jq '. | length' 2>/dev/null || echo "0")
            if [ "$app_count" -gt 0 ]; then
                test_result "Prowlarr Integration" "PASS" "$app_count applications configured"
            else
                test_result "Prowlarr Integration" "FAIL" "No applications configured"
            fi
        else
            test_result "Prowlarr Integration" "FAIL" "Cannot retrieve applications"
        fi
    fi
    
    # Test download client connections
    if [ -n "${SONARR_API_KEY:-}" ]; then
        local dl_response
        if dl_response=$(curl -s -H "X-Api-Key:$SONARR_API_KEY" "$SONARR_URL/api/v3/downloadclient" 2>/dev/null); then
            local dl_count=$(echo "$dl_response" | jq '. | length' 2>/dev/null || echo "0")
            if [ "$dl_count" -gt 0 ]; then
                test_result "Sonarr Download Clients" "PASS" "$dl_count download clients configured"
            else
                test_result "Sonarr Download Clients" "FAIL" "No download clients configured"
            fi
        else
            test_result "Sonarr Download Clients" "FAIL" "Cannot retrieve download clients"
        fi
    fi
}

# Test system health
test_system_health() {
    log "Testing system health"
    
    # Test Docker containers
    local containers=("jellyfin" "plex" "sonarr" "radarr" "prowlarr" "qbittorrent" "bazarr")
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" --filter "name=$container" | grep -q "$container"; then
            test_result "Container Health" "PASS" "$container is running"
        else
            test_result "Container Health" "FAIL" "$container is not running"
        fi
    done
    
    # Test network connectivity between services
    if docker exec sonarr ping -c 1 prowlarr >/dev/null 2>&1; then
        test_result "Network Connectivity" "PASS" "Sonarr can reach Prowlarr"
    else
        test_result "Network Connectivity" "FAIL" "Sonarr cannot reach Prowlarr"
    fi
    
    if docker exec radarr ping -c 1 qbittorrent >/dev/null 2>&1; then
        test_result "Network Connectivity" "PASS" "Radarr can reach qBittorrent"
    else
        test_result "Network Connectivity" "FAIL" "Radarr cannot reach qBittorrent"
    fi
}

# Generate test report
generate_report() {
    local report_file="./logs/api-integration-test-$(date +%Y%m%d-%H%M%S).json"
    local html_report="./logs/api-integration-test-$(date +%Y%m%d-%H%M%S).html"
    
    # JSON Report
    cat > "$report_file" << EOF
{
    "test_run": {
        "timestamp": "$(date -Iseconds)",
        "total_tests": $TOTAL_TESTS,
        "passed_tests": $PASSED_TESTS,
        "failed_tests": $FAILED_TESTS,
        "success_rate": "$(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    },
    "services_tested": [
        {"name": "Jellyfin", "url": "$JELLYFIN_URL"},
        {"name": "Sonarr", "url": "$SONARR_URL"},
        {"name": "Radarr", "url": "$RADARR_URL"},
        {"name": "Prowlarr", "url": "$PROWLARR_URL"},
        {"name": "qBittorrent", "url": "$QBITTORRENT_URL"},
        {"name": "Jellyseerr", "url": "$JELLYSEERR_URL"}
    ],
    "api_keys_configured": {
        "sonarr": "$([ -n "${SONARR_API_KEY:-}" ] && echo true || echo false)",
        "radarr": "$([ -n "${RADARR_API_KEY:-}" ] && echo true || echo false)",
        "prowlarr": "$([ -n "${PROWLARR_API_KEY:-}" ] && echo true || echo false)",
        "bazarr": "$([ -n "${BAZARR_API_KEY:-}" ] && echo true || echo false)"
    }
}
EOF

    # HTML Report
    cat > "$html_report" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>API Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 8px; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .summary { margin: 20px 0; padding: 15px; background: #e9ecef; border-radius: 5px; }
        .service { margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>API Integration Test Report</h1>
        <p><strong>Generated:</strong> $(date)</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p><strong>Total Tests:</strong> $TOTAL_TESTS</p>
        <p><strong>Passed:</strong> <span class="pass">$PASSED_TESTS</span></p>
        <p><strong>Failed:</strong> <span class="fail">$FAILED_TESTS</span></p>
        <p><strong>Success Rate:</strong> $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%</p>
    </div>
    
    <h2>Service URLs</h2>
    <table>
        <tr><th>Service</th><th>URL</th><th>API Key Configured</th></tr>
        <tr><td>Jellyfin</td><td><a href="$JELLYFIN_URL">$JELLYFIN_URL</a></td><td>$([ -n "${JELLYFIN_API_KEY:-}" ] && echo "Yes" || echo "No")</td></tr>
        <tr><td>Sonarr</td><td><a href="$SONARR_URL">$SONARR_URL</a></td><td>$([ -n "${SONARR_API_KEY:-}" ] && echo "Yes" || echo "No")</td></tr>
        <tr><td>Radarr</td><td><a href="$RADARR_URL">$RADARR_URL</a></td><td>$([ -n "${RADARR_API_KEY:-}" ] && echo "Yes" || echo "No")</td></tr>
        <tr><td>Prowlarr</td><td><a href="$PROWLARR_URL">$PROWLARR_URL</a></td><td>$([ -n "${PROWLARR_API_KEY:-}" ] && echo "Yes" || echo "No")</td></tr>
        <tr><td>qBittorrent</td><td><a href="$QBITTORRENT_URL">$QBITTORRENT_URL</a></td><td>N/A</td></tr>
        <tr><td>Jellyseerr</td><td><a href="$JELLYSEERR_URL">$JELLYSEERR_URL</a></td><td>N/A</td></tr>
    </table>
</body>
</html>
EOF

    log "Test report generated: $report_file"
    log "HTML report generated: $html_report"
}

# Main test function
main() {
    echo -e "${BLUE}=== MEDIA SERVER API INTEGRATION TEST SUITE ===${NC}"
    echo -e "${BLUE}Starting comprehensive API integration tests...${NC}\n"
    
    # Create logs directory
    mkdir -p ./logs
    
    # Run test suites
    test_arr_services
    echo ""
    test_media_servers
    echo ""
    test_download_clients
    echo ""
    test_request_services
    echo ""
    test_integrations
    echo ""
    test_system_health
    echo ""
    
    # Display results
    echo -e "\n${BLUE}=== TEST RESULTS SUMMARY ===${NC}"
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All tests passed! Your media server integration is working correctly.${NC}"
    else
        echo -e "\n${YELLOW}⚠️  Some tests failed. Check the output above for details.${NC}"
    fi
    
    # Calculate success rate
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "Success Rate: ${success_rate}%\n"
    
    # Generate reports
    generate_report
    
    # Return appropriate exit code
    [ $FAILED_TESTS -eq 0 ] && exit 0 || exit 1
}

# Check dependencies
if ! command -v curl &> /dev/null; then
    error "curl is required but not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    error "docker is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    warn "jq is not installed - some tests may be limited"
fi

# Run main function
main "$@"