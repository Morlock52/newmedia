#\!/bin/bash
# Quick test script for production system

echo "🧪 Testing Production Media Server"
echo "==================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test counter
TESTS=0
PASSED=0

# Test function
test_item() {
    TESTS=$((TESTS + 1))
    local name=$1
    local cmd=$2
    
    echo -n "Testing $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
    fi
}

echo "1. Docker Environment"
echo "---------------------"
test_item "Docker" "docker --version"
test_item "Docker Compose" "docker-compose --version"
test_item "Docker daemon" "docker info"

echo ""
echo "2. System Resources"
echo "-------------------"
MEM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "16")
echo "• Memory: ${MEM_GB}GB"
DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "• Disk: ${DISK_GB}GB available"
echo "• CPUs: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")"

echo ""
echo "3. Container Status"
echo "-------------------"
if docker ps | grep -q "ultimate-media-server"; then
    echo -e "${GREEN}✓${NC} Container is running"
else
    echo -e "${YELLOW}⚠${NC} Container not running - checking services directly"
fi

echo ""
echo "4. Service Endpoints"
echo "--------------------"
test_item "Jellyfin (8096)" "curl -f -s http://localhost:8096"
test_item "Sonarr (8989)" "curl -f -s http://localhost:8989"
test_item "Radarr (7878)" "curl -f -s http://localhost:7878"
test_item "Prowlarr (9696)" "curl -f -s http://localhost:9696"
test_item "qBittorrent (8080)" "curl -f -s http://localhost:8080"
test_item "Dashboard (5173)" "curl -f -s http://localhost:5173"

echo ""
echo "5. Docker Containers Running"
echo "-----------------------------"
docker ps --format "table {{.Names}}\t{{.Status}}" | head -10

echo ""
echo "========================="
echo "Test Summary:"
echo "Passed: $PASSED/$TESTS"
RATE=$((PASSED * 100 / TESTS))
echo "Pass Rate: ${RATE}%"

if [ $RATE -ge 80 ]; then
    echo -e "${GREEN}✅ System is operational${NC}"
elif [ $RATE -ge 50 ]; then
    echo -e "${YELLOW}⚠️ System partially working${NC}"
else
    echo -e "${RED}❌ System needs attention${NC}"
fi
