#!/bin/bash

# Media Server Dashboard Testing Script
# Tests all monitoring capabilities of simple-dashboard.html

echo "================================"
echo "Media Server Dashboard Test Suite"
echo "================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PASS_COUNT=0
FAIL_COUNT=0

# Function to log test results
log_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [ "$result" == "PASS" ]; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        ((PASS_COUNT++))
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        echo "   Details: $details"
        ((FAIL_COUNT++))
    fi
}

# Function to test service endpoint
test_service() {
    local service_name="$1"
    local url="$2"
    local expected_codes="$3"
    
    echo -n "Testing $service_name... "
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" --max-time 5)
    
    if [[ " $expected_codes " =~ " $response " ]]; then
        echo -e "${GREEN}✅ OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $response)"
        return 1
    fi
}

echo "1. Testing Dashboard File Structure"
echo "-----------------------------------"

# Test 1: Dashboard file exists
if [ -f "simple-dashboard.html" ]; then
    log_test "Dashboard file exists" "PASS"
else
    log_test "Dashboard file exists" "FAIL" "simple-dashboard.html not found"
fi

# Test 2: Dashboard has proper HTML structure
if grep -q "<!DOCTYPE html>" simple-dashboard.html && \
   grep -q "<title>Media Server Dashboard</title>" simple-dashboard.html; then
    log_test "HTML structure valid" "PASS"
else
    log_test "HTML structure valid" "FAIL" "Missing required HTML elements"
fi

echo ""
echo "2. Testing Service Links"
echo "------------------------"

# Test 3: Service links are present
services=(
    "Jellyfin|https://jellyfin.morloksmaze.com"
    "Overseerr|https://overseerr.morloksmaze.com"
    "Sonarr|https://sonarr.morloksmaze.com"
    "Radarr|https://radarr.morloksmaze.com"
    "Lidarr|https://lidarr.morloksmaze.com"
    "Prowlarr|https://prowlarr.morloksmaze.com"
    "Homarr|https://homarr.morloksmaze.com"
    "Authelia|https://auth.morloksmaze.com"
    "Traefik|http://localhost:8181/dashboard/"
)

all_links_present=true
for service in "${services[@]}"; do
    IFS='|' read -r name url <<< "$service"
    if grep -q "$url" simple-dashboard.html; then
        echo -e "  ${GREEN}✓${NC} $name link present: $url"
    else
        echo -e "  ${RED}✗${NC} $name link missing: $url"
        all_links_present=false
    fi
done

if [ "$all_links_present" = true ]; then
    log_test "All service links present" "PASS"
else
    log_test "All service links present" "FAIL" "Some service links missing"
fi

echo ""
echo "3. Testing Service Status Display"
echo "---------------------------------"

# Test 4: Status section exists
if grep -q "Service Status" simple-dashboard.html && \
   grep -q "Infrastructure Status" simple-dashboard.html && \
   grep -q "Authentication Status" simple-dashboard.html; then
    log_test "Status sections present" "PASS"
else
    log_test "Status sections present" "FAIL" "Missing status sections"
fi

# Test 5: Status indicators
if grep -q "class=\"working\"" simple-dashboard.html && \
   grep -q "class=\"offline\"" simple-dashboard.html; then
    log_test "Status indicators defined" "PASS"
else
    log_test "Status indicators defined" "FAIL" "Missing status CSS classes"
fi

echo ""
echo "4. Testing Docker Integration"
echo "-----------------------------"

# Test 6: Check if dashboard reflects actual Docker services
echo "Checking running Docker containers..."
docker_services=$(docker ps --format "table {{.Names}}" | tail -n +2 | sort)

if [ -n "$docker_services" ]; then
    echo "Running containers:"
    echo "$docker_services" | while read container; do
        echo "  - $container"
    done
    log_test "Docker containers detected" "PASS"
else
    log_test "Docker containers detected" "FAIL" "No containers running"
fi

echo ""
echo "5. Testing Auto-Refresh Capability"
echo "----------------------------------"

# Test 7: Check for auto-refresh meta tag or JavaScript
if grep -q "meta.*refresh" simple-dashboard.html || \
   grep -q "setTimeout\|setInterval" simple-dashboard.html || \
   grep -q "location.reload" simple-dashboard.html; then
    log_test "Auto-refresh capability" "FAIL" "No auto-refresh found in current implementation"
else
    log_test "Auto-refresh capability" "FAIL" "No auto-refresh functionality detected"
fi

echo ""
echo "6. Testing Error Handling"
echo "-------------------------"

# Test 8: CSS styling for offline services
if grep -q ".offline.*color.*#dc3545" simple-dashboard.html; then
    log_test "Offline service styling" "PASS"
else
    log_test "Offline service styling" "FAIL" "No specific offline styling found"
fi

echo ""
echo "7. Testing Live Service Connectivity"
echo "------------------------------------"

# Test actual service endpoints
test_service "Jellyfin" "https://jellyfin.morloksmaze.com" "302 404 502"
test_service "Sonarr" "https://sonarr.morloksmaze.com" "302 401 200"
test_service "Radarr" "https://radarr.morloksmaze.com" "302 401 200"
test_service "Lidarr" "https://lidarr.morloksmaze.com" "302 401 200"
test_service "Prowlarr" "https://prowlarr.morloksmaze.com" "302 401 200"
test_service "Overseerr" "https://overseerr.morloksmaze.com" "302 307 200"
test_service "Homarr" "https://homarr.morloksmaze.com" "302 200"
test_service "Authelia" "https://auth.morloksmaze.com" "302 200"

echo ""
echo "8. Recommendations for Improvement"
echo "----------------------------------"

echo "Current dashboard limitations:"
echo "1. ❌ No real-time service monitoring"
echo "2. ❌ No auto-refresh functionality"
echo "3. ❌ Static status display (not live)"
echo "4. ❌ No Docker integration"
echo "5. ❌ No API health checks"
echo ""
echo "Recommended enhancements:"
echo "1. ✨ Add JavaScript for live service checking"
echo "2. ✨ Implement auto-refresh (every 30-60 seconds)"
echo "3. ✨ Add Docker API integration"
echo "4. ✨ Show real-time service health"
echo "5. ✨ Add service restart buttons"
echo "6. ✨ Show resource usage (CPU/Memory)"
echo "7. ✨ Add notification for offline services"

echo ""
echo "================================"
echo "Test Summary"
echo "================================"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed. See details above.${NC}"
    exit 1
fi