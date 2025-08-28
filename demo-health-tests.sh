#!/bin/bash
# Demo Health Testing Suite
# Quick demonstration of the health testing capabilities

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🏥 Media Server Health Testing Suite - Demo${NC}"
echo "=============================================="
echo

# Check if container is running
echo -e "${YELLOW}🔍 Checking for running containers...${NC}"
CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers found")
echo "$CONTAINERS"
echo

# Test 1: Basic service discovery (without container requirement)
echo -e "${BLUE}🧪 Test 1: Service Discovery Demo${NC}"
echo "This would normally check all 30+ services..."

# Simulate service checks
services=("jellyfin:8096" "sonarr:8989" "radarr:7878" "prowlarr:9696" "qbittorrent:8080")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    
    # Check if port is listening locally
    if nc -z localhost "$port" 2>/dev/null; then
        echo -e "  ✅ ${name}: Listening on port ${port}"
    else
        echo -e "  ❌ ${name}: Port ${port} not accessible"
    fi
done
echo

# Test 2: Basic health endpoint checks
echo -e "${BLUE}🧪 Test 2: Health Endpoint Demo${NC}"
echo "Testing accessible health endpoints..."

health_checks=(
    "jellyfin:8096:/health"
    "sonarr:8989:/ping"
    "radarr:7878:/ping"
    "prowlarr:9696:/ping"
)

for check in "${health_checks[@]}"; do
    IFS=':' read -r name port endpoint <<< "$check"
    
    if nc -z localhost "$port" 2>/dev/null; then
        response=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://localhost:${port}${endpoint}" 2>/dev/null || echo "000")
        
        if [[ "$response" =~ ^[2-3][0-9][0-9]$ ]]; then
            echo -e "  ✅ ${name}: Health endpoint OK (HTTP ${response})"
        else
            echo -e "  ⚠️  ${name}: Health endpoint returned HTTP ${response}"
        fi
    else
        echo -e "  ❌ ${name}: Service not accessible"
    fi
done
echo

# Test 3: System resource check
echo -e "${BLUE}🧪 Test 3: System Resource Demo${NC}"
echo "Current system resources:"

# Memory
echo -n "  💾 Memory: "
if command -v free >/dev/null; then
    free -h | awk 'NR==2{printf "Used: %s/%s (%.1f%%)\n", $3,$2,$3*100/$2}'
else
    echo "Unable to check memory usage"
fi

# Load average
echo -n "  🖥️  Load: "
uptime | awk -F'load average:' '{ print $2 }'

# Disk space
echo -n "  💿 Disk: "
df -h / | awk 'NR==2{printf "Used: %s/%s (%s)\n", $3,$2,$5}'
echo

# Test 4: Docker stats (if available)
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    echo -e "${BLUE}🧪 Test 4: Docker Stats Demo${NC}"
    
    running_containers=$(docker ps --format "{{.Names}}" | wc -l)
    echo "  📊 Running containers: $running_containers"
    
    if [[ $running_containers -gt 0 ]]; then
        echo "  🔢 Container resource usage:"
        docker stats --no-stream --format "table   {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -6
    fi
else
    echo -e "${YELLOW}⚠️  Docker not available - skipping container stats${NC}"
fi
echo

# Test 5: Network connectivity test
echo -e "${BLUE}🧪 Test 5: Network Connectivity Demo${NC}"
echo "Testing external connectivity:"

urls=("https://google.com" "https://github.com" "https://docker.io")

for url in "${urls[@]}"; do
    if curl -s -m 5 --head "$url" >/dev/null; then
        echo -e "  ✅ ${url}: Accessible"
    else
        echo -e "  ❌ ${url}: Not accessible"
    fi
done
echo

# Summary
echo -e "${GREEN}📋 Demo Summary${NC}"
echo "=============="
echo "This demonstration shows basic health checking capabilities."
echo
echo "🚀 Full Test Suite Features:"
echo "  • Service discovery for 30+ services"
echo "  • s6-overlay process monitoring"
echo "  • Dependency chain validation" 
echo "  • API load testing with performance metrics"
echo "  • Resource usage tracking"
echo "  • Comprehensive HTML reports"
echo
echo "📖 Usage Examples:"
echo "  ./run-all-health-tests.sh                    # Full test suite"
echo "  ./container-service-monitor.sh               # Container monitoring"
echo "  ./service-dependency-validator.py            # Dependency analysis"
echo "  ./api-load-test-suite.js --duration 30       # API load testing"
echo
echo "🎯 To run the complete health test suite:"
echo "  ./run-all-health-tests.sh -v --duration 60"
echo

echo -e "${GREEN}✅ Demo completed!${NC}"