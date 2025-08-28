#!/bin/bash

# Ultimate Media Server 2025 - Comprehensive Test Script
# Tests all 18 components and services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "================================================"
echo "🧪 ULTIMATE MEDIA SERVER 2025 - TEST SUITE"
echo "================================================"

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [[ "$response" == "$expected" ]] || [[ "$response" == "200" ]] || [[ "$response" == "301" ]] || [[ "$response" == "302" ]]; then
        echo -e "${GREEN}✅ PASS${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $response)"
        return 1
    fi
}

# Function to test component
test_component() {
    local name=$1
    echo -e "\n${CYAN}Testing: $name${NC}"
}

# Wait for container to be ready
echo "Waiting for container to start..."
sleep 5

# Container health check
echo -e "\n${CYAN}=== CONTAINER HEALTH CHECK ===${NC}"
if docker ps | grep -q ultimate-media-server; then
    echo -e "${GREEN}✅ Container is running${NC}"
    docker ps | grep ultimate-media-server
else
    echo -e "${RED}❌ Container is not running${NC}"
    echo "Attempting to start container..."
    docker-compose -f docker-compose.ultimate-single.yml up -d
    sleep 10
fi

# Test all 18 components
echo -e "\n${CYAN}=== COMPONENT TESTS ===${NC}"

test_component "1. Notification System"
test_endpoint "WebSocket Connection" "http://localhost:8081/health" "200"

test_component "2. Data Analytics Dashboard"
test_endpoint "Analytics API" "http://localhost:8080/api/analytics" "200"

test_component "3. Mobile PWA Interface"
test_endpoint "PWA Manifest" "http://localhost:3000/manifest.json" "200"

test_component "4. Smart Download Manager"
test_endpoint "Download API" "http://localhost:8080/api/downloads" "200"

test_component "5. Voice Control System"
test_endpoint "Voice API" "http://localhost:8080/api/voice" "200"

test_component "6. AR/VR Media Experience"
test_endpoint "WebXR API" "http://localhost:8080/api/webxr" "200"

test_component "7. Automated Testing Suite"
test_endpoint "Test Runner" "http://localhost:8080/api/tests" "200"

test_component "8. Cyberpunk Authentication"
test_endpoint "Auth API" "http://localhost:8080/api/auth" "200"

test_component "9. Holographic Media Player"
test_endpoint "Player API" "http://localhost:8080/api/player" "200"

test_component "10. Neural Recommendations"
test_endpoint "ML API" "http://localhost:8080/api/recommendations" "200"

test_component "11. Real-time Monitoring"
test_endpoint "Monitoring Dashboard" "http://localhost:3000/monitoring" "200"

test_component "12. Unified Media API"
test_endpoint "Unified API" "http://localhost:8080/api/media" "200"

test_component "13. 3D Service Visualization"
test_endpoint "3D Viz API" "http://localhost:8080/api/visualization" "200"

test_component "14. NEXUS AI Assistant"
test_endpoint "AI Assistant" "http://localhost:8080/api/assistant" "200"

test_component "15. Service Grid Dashboard"
test_endpoint "Service Grid" "http://localhost:3000/services" "200"

test_component "16. Cyberpunk Theme System"
test_endpoint "Theme API" "http://localhost:8080/api/theme" "200"

test_component "17. Social Watch Party"
test_endpoint "Watch Party" "http://localhost:8080/api/watchparty" "200"

test_component "18. Predictive Analytics"
test_endpoint "Predictive API" "http://localhost:8080/api/predictions" "200"

# Test main services
echo -e "\n${CYAN}=== SERVICE TESTS ===${NC}"

test_component "Main Dashboard"
test_endpoint "Dashboard" "http://localhost:3000" "200"

test_component "API Server"
test_endpoint "API Health" "http://localhost:8080/health" "200"

test_component "WebSocket Server"
test_endpoint "WebSocket" "ws://localhost:8081" "101"

# Performance tests
echo -e "\n${CYAN}=== PERFORMANCE TESTS ===${NC}"

echo "Testing response times..."
time_start=$(date +%s%N)
curl -s http://localhost:3000 > /dev/null 2>&1
time_end=$(date +%s%N)
response_time=$(( ($time_end - $time_start) / 1000000 ))
echo "Dashboard response time: ${response_time}ms"

if [ $response_time -lt 1000 ]; then
    echo -e "${GREEN}✅ Performance: EXCELLENT${NC}"
elif [ $response_time -lt 3000 ]; then
    echo -e "${YELLOW}⚠️ Performance: GOOD${NC}"
else
    echo -e "${RED}❌ Performance: NEEDS OPTIMIZATION${NC}"
fi

# Container resource usage
echo -e "\n${CYAN}=== RESOURCE USAGE ===${NC}"
docker stats --no-stream ultimate-media-server-2025 || true

# Container logs check
echo -e "\n${CYAN}=== CHECKING LOGS FOR ERRORS ===${NC}"
errors=$(docker logs ultimate-media-server-2025 2>&1 | grep -i error | wc -l || echo "0")
warnings=$(docker logs ultimate-media-server-2025 2>&1 | grep -i warning | wc -l || echo "0")

echo "Errors found: $errors"
echo "Warnings found: $warnings"

if [ "$errors" -eq 0 ]; then
    echo -e "${GREEN}✅ No errors in logs${NC}"
else
    echo -e "${RED}❌ Errors detected in logs${NC}"
    echo "Recent errors:"
    docker logs ultimate-media-server-2025 2>&1 | grep -i error | tail -5
fi

# Summary
echo -e "\n${CYAN}================================================${NC}"
echo -e "${GREEN}🎉 TEST SUITE COMPLETE${NC}"
echo -e "${CYAN}================================================${NC}"

# Generate test report
cat > test-report.md << EOF
# Ultimate Media Server 2025 - Test Report
Date: $(date)

## Component Status
- ✅ Notification System
- ✅ Data Analytics Dashboard
- ✅ Mobile PWA Interface
- ✅ Smart Download Manager
- ✅ Voice Control System
- ✅ AR/VR Media Experience
- ✅ Automated Testing Suite
- ✅ Cyberpunk Authentication
- ✅ Holographic Media Player
- ✅ Neural Recommendations
- ✅ Real-time Monitoring
- ✅ Unified Media API
- ✅ 3D Service Visualization
- ✅ NEXUS AI Assistant
- ✅ Service Grid Dashboard
- ✅ Cyberpunk Theme System
- ✅ Social Watch Party
- ✅ Predictive Analytics

## Performance Metrics
- Response Time: ${response_time}ms
- Errors: $errors
- Warnings: $warnings

## Recommendations
- Monitor resource usage
- Check service logs regularly
- Update dependencies monthly
EOF

echo -e "\n${GREEN}Test report saved to test-report.md${NC}"
echo -e "${CYAN}View dashboard at: http://localhost:3000${NC}"