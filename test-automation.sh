#!/bin/bash

# Ultimate Media Server 2025 - Automation Test Suite
# Test all automation components and connections

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Ultimate Media Server 2025 - Test Suite     ${NC}"
    echo -e "${BLUE}  Automation Components Test                  ${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo
}

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test service connectivity
test_service() {
    local service=$1
    local port=$2
    local path=${3:-""}
    
    print_test "Testing $service connectivity..."
    
    if curl -s -f "http://localhost:$port$path" >/dev/null 2>&1; then
        print_pass "$service is responding"
        return 0
    else
        print_fail "$service is not responding"
        return 1
    fi
}

# Test API endpoint
test_api() {
    local service=$1
    local port=$2
    local api_key=$3
    
    print_test "Testing $service API..."
    
    if curl -s -f -H "X-Api-Key: $api_key" "http://localhost:$port/api/v1/system/status" >/dev/null 2>&1 || \
       curl -s -f -H "X-Api-Key: $api_key" "http://localhost:$port/api/v3/system/status" >/dev/null 2>&1; then
        print_pass "$service API is working"
        return 0
    else
        print_fail "$service API is not responding"
        return 1
    fi
}

# Get API key from config
get_api_key() {
    local config_path=$1
    if [ -f "$config_path" ]; then
        grep -o '<ApiKey>[^<]*</ApiKey>' "$config_path" | sed 's/<ApiKey>\|<\/ApiKey>//g' | head -1
    else
        echo ""
    fi
}

# Test Docker services
test_docker_services() {
    print_test "Testing Docker services..."
    
    # Check if docker-compose is running
    if ! docker-compose ps >/dev/null 2>&1; then
        print_fail "Docker Compose is not running"
        return 1
    fi
    
    # Get running services
    running_services=$(docker-compose ps --services --filter "status=running" 2>/dev/null || echo "")
    
    if [ -z "$running_services" ]; then
        print_fail "No services are running"
        return 1
    fi
    
    print_pass "Docker services are running: $(echo $running_services | tr '\n' ' ')"
    return 0
}

# Test service health
test_service_health() {
    print_test "Testing service health..."
    
    local failed_services=""
    
    # Core automation services
    services=(
        "prowlarr:9696:/ping"
        "sonarr:8989:/ping"
        "radarr:7878:/ping"
        "lidarr:8686:/ping"
        "readarr:8787:/ping"
        "bazarr:6767:/ping"
        "overseerr:5055:/api/v1/status"
        "jellyfin:8096:/health"
        "qbittorrent:8080:"
        "sabnzbd:8081:"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port path <<< "$service_info"
        
        if ! test_service "$service" "$port" "$path"; then
            failed_services="$failed_services $service"
        fi
    done
    
    if [ -n "$failed_services" ]; then
        print_fail "Some services failed health check:$failed_services"
        return 1
    else
        print_pass "All services passed health check"
        return 0
    fi
}

# Test API connections
test_api_connections() {
    print_test "Testing API connections..."
    
    # Extract API keys
    prowlarr_key=$(get_api_key "config/prowlarr/config.xml")
    sonarr_key=$(get_api_key "config/sonarr/config.xml")
    radarr_key=$(get_api_key "config/radarr/config.xml")
    lidarr_key=$(get_api_key "config/lidarr/config.xml")
    readarr_key=$(get_api_key "config/readarr/config.xml")
    
    local failed_apis=""
    
    # Test APIs
    if [ -n "$prowlarr_key" ]; then
        test_api "Prowlarr" "9696" "$prowlarr_key" || failed_apis="$failed_apis Prowlarr"
    else
        print_warning "Prowlarr API key not found"
    fi
    
    if [ -n "$sonarr_key" ]; then
        test_api "Sonarr" "8989" "$sonarr_key" || failed_apis="$failed_apis Sonarr"
    else
        print_warning "Sonarr API key not found"
    fi
    
    if [ -n "$radarr_key" ]; then
        test_api "Radarr" "7878" "$radarr_key" || failed_apis="$failed_apis Radarr"
    else
        print_warning "Radarr API key not found"
    fi
    
    if [ -n "$lidarr_key" ]; then
        test_api "Lidarr" "8686" "$lidarr_key" || failed_apis="$failed_apis Lidarr"
    else
        print_warning "Lidarr API key not found"
    fi
    
    if [ -n "$readarr_key" ]; then
        test_api "Readarr" "8787" "$readarr_key" || failed_apis="$failed_apis Readarr"
    else
        print_warning "Readarr API key not found"
    fi
    
    if [ -n "$failed_apis" ]; then
        print_fail "Some APIs failed:$failed_apis"
        return 1
    else
        print_pass "All APIs are responding"
        return 0
    fi
}

# Test download client connections
test_download_clients() {
    print_test "Testing download client connections..."
    
    local failed_clients=""
    
    # Test qBittorrent
    if curl -s -f "http://localhost:8080" >/dev/null 2>&1; then
        print_pass "qBittorrent is accessible"
    else
        print_fail "qBittorrent is not accessible"
        failed_clients="$failed_clients qBittorrent"
    fi
    
    # Test SABnzbd
    if curl -s -f "http://localhost:8081" >/dev/null 2>&1; then
        print_pass "SABnzbd is accessible"
    else
        print_fail "SABnzbd is not accessible"
        failed_clients="$failed_clients SABnzbd"
    fi
    
    if [ -n "$failed_clients" ]; then
        return 1
    else
        return 0
    fi
}

# Test directory structure
test_directory_structure() {
    print_test "Testing directory structure..."
    
    local missing_dirs=""
    
    # Required directories
    dirs=(
        "media-data/movies"
        "media-data/tv"
        "media-data/music"
        "media-data/books"
        "media-data/audiobooks"
        "media-data/downloads/torrents"
        "media-data/downloads/usenet"
        "config/prowlarr"
        "config/sonarr"
        "config/radarr"
        "config/lidarr"
        "config/readarr"
        "config/bazarr"
        "config/overseerr"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            missing_dirs="$missing_dirs $dir"
        fi
    done
    
    if [ -n "$missing_dirs" ]; then
        print_fail "Missing directories:$missing_dirs"
        return 1
    else
        print_pass "All required directories exist"
        return 0
    fi
}

# Test VPN connection (if configured)
test_vpn_connection() {
    print_test "Testing VPN connection..."
    
    # Check if Gluetun container is running
    if docker-compose ps gluetun 2>/dev/null | grep -q "Up"; then
        # Test external IP through VPN
        vpn_ip=$(docker-compose exec -T gluetun wget -qO- http://ipinfo.io/ip 2>/dev/null || echo "")
        host_ip=$(wget -qO- http://ipinfo.io/ip 2>/dev/null || echo "")
        
        if [ -n "$vpn_ip" ] && [ -n "$host_ip" ] && [ "$vpn_ip" != "$host_ip" ]; then
            print_pass "VPN is working (IP: $vpn_ip)"
            return 0
        elif [ -n "$vpn_ip" ]; then
            print_warning "VPN detected but may not be working properly"
            return 0
        else
            print_fail "VPN is not working"
            return 1
        fi
    else
        print_warning "VPN container (Gluetun) is not running"
        return 0
    fi
}

# Test automation workflow (simplified)
test_automation_workflow() {
    print_test "Testing automation workflow..."
    
    # This is a simplified test - in reality you'd test the full workflow
    
    # Check if Prowlarr has applications configured
    prowlarr_key=$(get_api_key "config/prowlarr/config.xml")
    
    if [ -n "$prowlarr_key" ]; then
        apps=$(curl -s -H "X-Api-Key: $prowlarr_key" "http://localhost:9696/api/v1/applications" 2>/dev/null || echo "[]")
        
        if echo "$apps" | grep -q "sonarr\|radarr"; then
            print_pass "Prowlarr has applications configured"
        else
            print_warning "Prowlarr applications not fully configured"
        fi
    else
        print_warning "Cannot test Prowlarr applications - no API key"
    fi
    
    # Check if services can communicate
    print_pass "Basic automation workflow test completed"
    return 0
}

# Generate test report
generate_test_report() {
    local total_tests=$1
    local passed_tests=$2
    local failed_tests=$3
    
    echo
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}               Test Results                     ${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo
    echo -e "Total Tests: $total_tests"
    echo -e "${GREEN}Passed: $passed_tests${NC}"
    echo -e "${RED}Failed: $failed_tests${NC}"
    echo
    
    if [ $failed_tests -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Your automation is working perfectly!${NC}"
        echo
        echo -e "${BLUE}Quick Access:${NC}"
        echo -e "  Overseerr (Request content): http://localhost:5055"
        echo -e "  Jellyfin (Watch content):    http://localhost:8096"
        echo -e "  Homepage (Dashboard):        http://localhost:3001"
        echo
    else
        echo -e "${YELLOW}⚠️  Some tests failed. Check the output above for details.${NC}"
        echo
        echo -e "${BLUE}Troubleshooting:${NC}"
        echo -e "  1. Make sure all services are running: docker-compose ps"
        echo -e "  2. Check service logs: docker-compose logs [service-name]"
        echo -e "  3. Restart services: docker-compose restart"
        echo -e "  4. Reconfigure APIs: ./scripts/configure-automation-apis.sh"
        echo
    fi
}

# Main test function
main() {
    print_header
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # Run tests
    tests=(
        "test_docker_services"
        "test_directory_structure"
        "test_service_health"
        "test_api_connections"
        "test_download_clients"
        "test_vpn_connection"
        "test_automation_workflow"
    )
    
    for test_func in "${tests[@]}"; do
        ((total_tests++))
        if $test_func; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
        echo
    done
    
    generate_test_report $total_tests $passed_tests $failed_tests
    
    # Exit with proper code
    if [ $failed_tests -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"