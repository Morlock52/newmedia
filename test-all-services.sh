#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 ULTIMATE MEDIA HUB 2025 - COMPREHENSIVE TEST SUITE"
echo "===================================================="
echo

# Test function
test_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [[ "$response" == "$expected" ]] || [[ "$response" == "302" ]] || [[ "$response" == "303" ]] || [[ "$response" == "307" ]] || [[ "$response" == "200" ]]; then
        echo -e "${GREEN}✅ $name${NC} - Running (HTTP $response)"
        return 0
    else
        echo -e "${RED}❌ $name${NC} - Not responding (HTTP $response)"
        return 1
    fi
}

# Count results
total=0
passed=0

echo "📡 TESTING SERVICE CONNECTIVITY"
echo "==============================="
echo

# Test each service
test_service "Jellyfin Media Server" "http://localhost:8096" "302" && ((passed++))
((total++))

test_service "Overseerr Requests" "http://localhost:5055" "307" && ((passed++))
((total++))

test_service "Sonarr TV Automation" "http://localhost:8989" "307" && ((passed++))
((total++))

test_service "Radarr Movie Automation" "http://localhost:7878" "307" && ((passed++))
((total++))

test_service "Prowlarr Indexers" "http://localhost:9696" "307" && ((passed++))
((total++))

test_service "Lidarr Music" "http://localhost:8686" "307" && ((passed++))
((total++))

test_service "Bazarr Subtitles" "http://localhost:6767" "200" && ((passed++))
((total++))

test_service "Tautulli Analytics" "http://localhost:8181" "303" && ((passed++))
((total++))

test_service "SABnzbd Usenet" "http://localhost:8081" "403" && ((passed++))
((total++))

test_service "Homepage Dashboard" "http://localhost:3001" "200" && ((passed++))
((total++))

test_service "Portainer Management" "http://localhost:9000" "200" && ((passed++))
((total++))

test_service "Grafana Monitoring" "http://localhost:3000" "302" && ((passed++))
((total++))

test_service "Prometheus Metrics" "http://localhost:9090" "302" && ((passed++))
((total++))

test_service "Traefik Proxy" "http://localhost:8082" "200" && ((passed++))
((total++))

echo
echo "🐳 DOCKER CONTAINER STATUS"
echo "=========================="
echo

# Show container status
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(jellyfin|overseerr|sonarr|radarr|prowlarr|lidarr|bazarr|tautulli|sabnzbd|homepage|portainer|grafana|prometheus|traefik)" | head -15

echo
echo "📊 TEST RESULTS"
echo "==============="
echo
echo -e "Total Services Tested: ${total}"
echo -e "Services Running: ${GREEN}${passed}${NC}"
echo -e "Services Failed: ${RED}$((total - passed))${NC}"
echo

if [ $passed -eq $total ]; then
    echo -e "${GREEN}🎉 ALL SERVICES OPERATIONAL!${NC}"
    echo "Your Ultimate Media Hub 2025 is fully functional!"
else
    echo -e "${YELLOW}⚠️  Some services need attention${NC}"
    echo "Check the failed services above for troubleshooting"
fi

echo
echo "🌐 ACCESS YOUR SERVICES"
echo "======================"
echo
echo "🎯 Request Content: http://localhost:5055 (Overseerr)"
echo "🎬 Watch Media: http://localhost:8096 (Jellyfin)"
echo "🏠 Dashboard: http://localhost:3001 (Homepage)"
echo "📊 Monitoring: http://localhost:3000 (Grafana)"
echo "🐳 Management: http://localhost:9000 (Portainer)"
echo
echo "📖 Full Dashboard: Open ULTIMATE_MEDIA_HUB_2025.html"
echo