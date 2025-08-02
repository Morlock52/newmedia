#!/bin/bash

# Ultimate Media Server 2025 - Service Integration Test Suite
# Tests service-to-service communication, data flow, and integration points

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
LOG_FILE="${SCRIPT_DIR}/test-results/service-integrations-$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="${SCRIPT_DIR}/test-results/service-integrations-report.json"
TIMEOUT=300

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNING_TESTS=0
if [[ ${BASH_VERSION%%.*} -ge 4 ]]; then
    declare -A TEST_RESULTS=()
    declare -A INTEGRATION_MATRIX=()
else
    TEST_RESULTS=""
    INTEGRATION_MATRIX=""
fi

# Ensure reports directory exists
mkdir -p "${SCRIPT_DIR}/test-results"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Test result tracking
record_test() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    TEST_RESULTS["$test_name"]="$status|$details"
    
    case "$status" in
        "PASS")
            PASSED_TESTS=$((PASSED_TESTS + 1))
            echo -e "${GREEN}✓ $test_name${NC}"
            ;;
        "WARN")
            WARNING_TESTS=$((WARNING_TESTS + 1))
            echo -e "${YELLOW}⚠ $test_name${NC}"
            ;;
        "FAIL")
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo -e "${RED}✗ $test_name${NC}"
            log "ERROR" "$test_name failed: $details"
            ;;
    esac
}

# API Helper Functions
get_api_key() {
    local service="$1"
    local container_name="$2"
    
    case "$service" in
        "sonarr"|"radarr"|"lidarr"|"readarr"|"bazarr"|"prowlarr")
            # Try to get API key from config.xml
            docker exec "$container_name" cat /config/config.xml 2>/dev/null | grep -o '<ApiKey>[^<]*</ApiKey>' | sed 's/<[^>]*>//g' 2>/dev/null || echo ""
            ;;
        *)
            echo ""
            ;;
    esac
}

make_api_request() {
    local url="$1"
    local api_key="$2"
    local method="${3:-GET}"
    
    if [[ -n "$api_key" ]]; then
        curl -s -m 10 -X "$method" -H "X-Api-Key: $api_key" "$url" 2>/dev/null || echo ""
    else
        curl -s -m 10 -X "$method" "$url" 2>/dev/null || echo ""
    fi
}

wait_for_service() {
    local service_name="$1"
    local port="$2"
    local max_attempts="${3:-30}"
    
    log "INFO" "Waiting for $service_name on port $port"
    
    for i in $(seq 1 $max_attempts); do
        if timeout 5 nc -z localhost "$port" 2>/dev/null; then
            log "INFO" "$service_name is ready"
            return 0
        fi
        sleep 2
    done
    
    log "ERROR" "$service_name failed to become ready after $max_attempts attempts"
    return 1
}

# ARR Suite Integration Tests
test_arr_prowlarr_sync() {
    log "INFO" "Testing ARR services synchronization with Prowlarr"
    
    local prowlarr_api_key=$(get_api_key "prowlarr" "prowlarr")
    if [[ -z "$prowlarr_api_key" ]]; then
        record_test "ARR-Prowlarr Sync" "WARN" "Prowlarr API key not available"
        return
    fi
    
    # Get applications from Prowlarr
    local apps_response=$(make_api_request "http://localhost:9696/api/v1/applications" "$prowlarr_api_key")
    
    if [[ -n "$apps_response" ]] && echo "$apps_response" | jq -e '.[]' >/dev/null 2>&1; then
        local app_count=$(echo "$apps_response" | jq '. | length')
        record_test "ARR-Prowlarr Sync" "PASS" "Found $app_count connected applications"
        
        # Test indexer sync to each application
        local indexers_response=$(make_api_request "http://localhost:9696/api/v1/indexer" "$prowlarr_api_key")
        if [[ -n "$indexers_response" ]]; then
            local indexer_count=$(echo "$indexers_response" | jq '. | length')
            record_test "Prowlarr Indexers" "PASS" "Found $indexer_count configured indexers"
        else
            record_test "Prowlarr Indexers" "WARN" "No indexers configured"
        fi
    else
        record_test "ARR-Prowlarr Sync" "FAIL" "No applications connected to Prowlarr"
    fi
}

test_arr_download_clients() {
    local arr_services=("sonarr:8989" "radarr:7878" "lidarr:8686" "readarr:8787")
    
    log "INFO" "Testing ARR services download client connections"
    
    for service_info in "${arr_services[@]}"; do
        local service=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        
        # Skip if service not running
        if ! docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            record_test "$service Download Clients" "WARN" "Service not running"
            continue
        fi
        
        local api_key=$(get_api_key "$service" "$service")
        if [[ -z "$api_key" ]]; then
            record_test "$service Download Clients" "WARN" "API key not available"
            continue
        fi
        
        # Test download client connections
        local clients_response=$(make_api_request "http://localhost:$port/api/v3/downloadclient" "$api_key")
        
        if [[ -n "$clients_response" ]] && echo "$clients_response" | jq -e '.[]' >/dev/null 2>&1; then
            local client_count=$(echo "$clients_response" | jq '. | length')
            local enabled_count=$(echo "$clients_response" | jq '[.[] | select(.enable == true)] | length')
            record_test "$service Download Clients" "PASS" "$enabled_count/$client_count clients enabled"
            
            # Test each enabled client
            echo "$clients_response" | jq -r '.[] | select(.enable == true) | .name' | while read -r client_name; do
                record_test "$service Client: $client_name" "PASS" "Client configured"
            done
        else
            record_test "$service Download Clients" "FAIL" "No download clients configured"
        fi
    done
}

test_media_server_integrations() {
    log "INFO" "Testing media server integrations"
    
    # Test Jellyfin library access
    if docker ps --format "{{.Names}}" | grep -q "^jellyfin$"; then
        local jellyfin_info=$(curl -s "http://localhost:8096/System/Info/Public" 2>/dev/null || echo "")
        if [[ -n "$jellyfin_info" ]]; then
            local version=$(echo "$jellyfin_info" | jq -r '.Version // "unknown"')
            record_test "Jellyfin Info" "PASS" "Version: $version"
            
            # Test library access
            local libraries=$(curl -s "http://localhost:8096/Library/VirtualFolders" 2>/dev/null || echo "")
            if [[ -n "$libraries" ]] && echo "$libraries" | jq -e '.[]' >/dev/null 2>&1; then
                local library_count=$(echo "$libraries" | jq '. | length')
                record_test "Jellyfin Libraries" "PASS" "Found $library_count libraries"
            else
                record_test "Jellyfin Libraries" "WARN" "No libraries configured"
            fi
        else
            record_test "Jellyfin Info" "FAIL" "Cannot access public info endpoint"
        fi
    fi
    
    # Test Plex
    if docker ps --format "{{.Names}}" | grep -q "^plex$"; then
        local plex_identity=$(curl -s "http://localhost:32400/identity" 2>/dev/null || echo "")
        if [[ -n "$plex_identity" ]]; then
            record_test "Plex Identity" "PASS" "Server accessible"
        else
            record_test "Plex Identity" "FAIL" "Cannot access identity endpoint"
        fi
    fi
    
    # Test Emby
    if docker ps --format "{{.Names}}" | grep -q "^emby$"; then
        local emby_info=$(curl -s "http://localhost:8097/System/Info/Public" 2>/dev/null || echo "")
        if [[ -n "$emby_info" ]]; then
            record_test "Emby Info" "PASS" "Server accessible"
        else
            record_test "Emby Info" "FAIL" "Cannot access public info endpoint"
        fi
    fi
}

test_request_service_integrations() {
    log "INFO" "Testing request service integrations"
    
    local request_services=("jellyseerr:5055" "overseerr:5056" "ombi:3579")
    
    for service_info in "${request_services[@]}"; do
        local service=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        
        # Skip if service not running
        if ! docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            record_test "$service Integration" "WARN" "Service not running"
            continue
        fi
        
        # Test service status
        local status_endpoint=""
        case "$service" in
            "jellyseerr"|"overseerr") status_endpoint="/api/v1/status" ;;
            "ombi") status_endpoint="/api/v1/Status" ;;
        esac
        
        local status_response=$(curl -s "http://localhost:$port$status_endpoint" 2>/dev/null || echo "")
        if [[ -n "$status_response" ]]; then
            record_test "$service Status" "PASS" "Service responding"
            
            # Test settings endpoint for media server connections
            local settings_endpoint=""
            case "$service" in
                "jellyseerr") settings_endpoint="/api/v1/settings/jellyfin" ;;
                "overseerr") settings_endpoint="/api/v1/settings/plex" ;;
                "ombi") settings_endpoint="/api/v1/Settings" ;;
            esac
            
            if [[ -n "$settings_endpoint" ]]; then
                local settings_response=$(curl -s "http://localhost:$port$settings_endpoint" 2>/dev/null || echo "")
                if [[ -n "$settings_response" ]]; then
                    record_test "$service Media Server Config" "PASS" "Configuration accessible"
                else
                    record_test "$service Media Server Config" "WARN" "Configuration not accessible"
                fi
            fi
        else
            record_test "$service Status" "FAIL" "Service not responding"
        fi
    done
}

test_download_client_connectivity() {
    log "INFO" "Testing download client connectivity"
    
    # Test qBittorrent
    if docker ps --format "{{.Names}}" | grep -q "^qbittorrent$"; then
        # qBittorrent should be accessible through VPN container
        local qbt_version=$(curl -s "http://localhost:8080/api/v2/app/version" 2>/dev/null || echo "")
        if [[ -n "$qbt_version" ]]; then
            record_test "qBittorrent API" "PASS" "Version: $qbt_version"
            
            # Test torrent list
            local torrents=$(curl -s "http://localhost:8080/api/v2/torrents/info" 2>/dev/null || echo "")
            if [[ -n "$torrents" ]]; then
                record_test "qBittorrent Torrents" "PASS" "Torrent list accessible"
            else
                record_test "qBittorrent Torrents" "WARN" "No torrents or authentication required"
            fi
        else
            record_test "qBittorrent API" "FAIL" "Cannot access API"
        fi
    fi
    
    # Test Transmission
    if docker ps --format "{{.Names}}" | grep -q "^transmission$"; then
        local transmission_stats=$(curl -s "http://localhost:9091/transmission/rpc" -H "X-Transmission-Session-Id: test" 2>/dev/null || echo "")
        if [[ -n "$transmission_stats" ]]; then
            record_test "Transmission RPC" "PASS" "RPC accessible"
        else
            record_test "Transmission RPC" "FAIL" "Cannot access RPC"
        fi
    fi
    
    # Test SABnzbd
    if docker ps --format "{{.Names}}" | grep -q "^sabnzbd$"; then
        local sab_version=$(curl -s "http://localhost:8081/sabnzbd/api?mode=version&output=json" 2>/dev/null || echo "")
        if [[ -n "$sab_version" ]] && echo "$sab_version" | jq -e '.version' >/dev/null 2>&1; then
            local version=$(echo "$sab_version" | jq -r '.version')
            record_test "SABnzbd API" "PASS" "Version: $version"
        else
            record_test "SABnzbd API" "FAIL" "Cannot access API"
        fi
    fi
}

test_monitoring_integrations() {
    log "INFO" "Testing monitoring system integrations"
    
    # Test Prometheus targets
    if docker ps --format "{{.Names}}" | grep -q "^prometheus$"; then
        local targets=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null || echo "")
        if [[ -n "$targets" ]] && echo "$targets" | jq -e '.data.activeTargets' >/dev/null 2>&1; then
            local active_targets=$(echo "$targets" | jq '.data.activeTargets | length')
            local healthy_targets=$(echo "$targets" | jq '[.data.activeTargets[] | select(.health == "up")] | length')
            record_test "Prometheus Targets" "PASS" "$healthy_targets/$active_targets targets healthy"
            
            # Test specific queries
            local up_query=$(curl -s "http://localhost:9090/api/v1/query?query=up" 2>/dev/null || echo "")
            if [[ -n "$up_query" ]]; then
                record_test "Prometheus Queries" "PASS" "Query engine working"
            else
                record_test "Prometheus Queries" "FAIL" "Query engine not responding"
            fi
        else
            record_test "Prometheus Targets" "FAIL" "Cannot access targets API"
        fi
    fi
    
    # Test Grafana data sources
    if docker ps --format "{{.Names}}" | grep -q "^grafana$"; then
        local datasources=$(curl -s -u "admin:admin" "http://localhost:3000/api/datasources" 2>/dev/null || echo "")
        if [[ -n "$datasources" ]] && echo "$datasources" | jq -e '.[]' >/dev/null 2>&1; then
            local ds_count=$(echo "$datasources" | jq '. | length')
            record_test "Grafana DataSources" "PASS" "Found $ds_count data sources"
        else
            record_test "Grafana DataSources" "WARN" "No data sources configured or auth required"
        fi
    fi
    
    # Test Loki
    if docker ps --format "{{.Names}}" | grep -q "^loki$"; then
        local loki_ready=$(curl -s "http://localhost:3100/ready" 2>/dev/null || echo "")
        if [[ "$loki_ready" == "ready" ]]; then
            record_test "Loki Ready" "PASS" "Log aggregation ready"
        else
            record_test "Loki Ready" "FAIL" "Log aggregation not ready"
        fi
    fi
}

test_database_integrations() {
    log "INFO" "Testing database integrations"
    
    # Test PostgreSQL connections
    if docker ps --format "{{.Names}}" | grep -q "^postgres$"; then
        # Test basic connectivity
        if docker exec postgres pg_isready -U postgres >/dev/null 2>&1; then
            record_test "PostgreSQL Health" "PASS" "Database ready"
            
            # Test database existence for services
            local databases=$(docker exec postgres psql -U postgres -t -c "SELECT datname FROM pg_database WHERE datistemplate = false;" 2>/dev/null | tr -d ' ' | grep -v '^$' || echo "")
            if [[ -n "$databases" ]]; then
                local db_count=$(echo "$databases" | wc -l)
                record_test "PostgreSQL Databases" "PASS" "Found $db_count databases"
            else
                record_test "PostgreSQL Databases" "WARN" "No application databases found"
            fi
        else
            record_test "PostgreSQL Health" "FAIL" "Database not ready"
        fi
    fi
    
    # Test Redis connections
    if docker ps --format "{{.Names}}" | grep -q "^redis$"; then
        if docker exec redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
            record_test "Redis Health" "PASS" "Cache ready"
            
            # Test key count
            local key_count=$(docker exec redis redis-cli dbsize 2>/dev/null || echo "0")
            record_test "Redis Keys" "PASS" "Database size: $key_count keys"
        else
            record_test "Redis Health" "FAIL" "Cache not ready"
        fi
    fi
    
    # Test MariaDB if present
    if docker ps --format "{{.Names}}" | grep -q "^mariadb$"; then
        if docker exec mariadb mysqladmin ping -h localhost -u root -p"${MYSQL_ROOT_PASSWORD:-root}" 2>/dev/null | grep -q "alive"; then
            record_test "MariaDB Health" "PASS" "Database ready"
        else
            record_test "MariaDB Health" "FAIL" "Database not ready"
        fi
    fi
}

test_vpn_network_isolation() {
    log "INFO" "Testing VPN network isolation"
    
    # Test Gluetun VPN container
    if docker ps --format "{{.Names}}" | grep -q "^gluetun$"; then
        # Check if VPN is connected
        local vpn_status=$(docker exec gluetun wget -qO- https://ipinfo.io/ip 2>/dev/null || echo "")
        if [[ -n "$vpn_status" ]]; then
            record_test "VPN Connection" "PASS" "External IP: $vpn_status"
            
            # Test download clients using VPN
            local qbt_network=$(docker inspect qbittorrent --format='{{.HostConfig.NetworkMode}}' 2>/dev/null || echo "")
            if [[ "$qbt_network" == "service:gluetun" ]]; then
                record_test "qBittorrent VPN" "PASS" "Using VPN network"
            else
                record_test "qBittorrent VPN" "WARN" "Not using VPN network: $qbt_network"
            fi
            
            local transmission_network=$(docker inspect transmission --format='{{.HostConfig.NetworkMode}}' 2>/dev/null || echo "")
            if [[ "$transmission_network" == "service:gluetun" ]]; then
                record_test "Transmission VPN" "PASS" "Using VPN network"
            else
                record_test "Transmission VPN" "WARN" "Not using VPN network: $transmission_network"
            fi
        else
            record_test "VPN Connection" "FAIL" "Cannot determine external IP"
        fi
    else
        record_test "VPN Connection" "WARN" "Gluetun container not running"
    fi
}

test_volume_data_flow() {
    log "INFO" "Testing volume data flow between services"
    
    # Check shared volume mounts
    local media_volume_containers=$(docker ps --format "{{.Names}}" | xargs -I {} docker inspect {} --format='{{.Name}}: {{range .Mounts}}{{if eq .Destination "/media"}}{{.Source}}{{end}}{{end}}' 2>/dev/null | grep -v ": $" || echo "")
    
    if [[ -n "$media_volume_containers" ]]; then
        local container_count=$(echo "$media_volume_containers" | wc -l)
        record_test "Media Volume Sharing" "PASS" "$container_count containers share media volume"
        
        # Test write permissions for ARR services
        for arr_service in sonarr radarr lidarr readarr; do
            if docker ps --format "{{.Names}}" | grep -q "^$arr_service$"; then
                if docker exec "$arr_service" touch /media/test_write_$$_"$arr_service" 2>/dev/null && docker exec "$arr_service" rm /media/test_write_$$_"$arr_service" 2>/dev/null; then
                    record_test "$arr_service Volume Write" "PASS" "Can write to media volume"
                else
                    record_test "$arr_service Volume Write" "FAIL" "Cannot write to media volume"
                fi
            fi
        done
    else
        record_test "Media Volume Sharing" "WARN" "No shared media volumes detected"
    fi
    
    # Check downloads volume
    local downloads_volume_containers=$(docker ps --format "{{.Names}}" | xargs -I {} docker inspect {} --format='{{.Name}}: {{range .Mounts}}{{if eq .Destination "/downloads"}}{{.Source}}{{end}}{{end}}' 2>/dev/null | grep -v ": $" || echo "")
    
    if [[ -n "$downloads_volume_containers" ]]; then
        local container_count=$(echo "$downloads_volume_containers" | wc -l)
        record_test "Downloads Volume Sharing" "PASS" "$container_count containers share downloads volume"
    else
        record_test "Downloads Volume Sharing" "WARN" "No shared downloads volumes detected"
    fi
}

test_dashboard_integrations() {
    log "INFO" "Testing dashboard integrations"
    
    # Test Homepage
    if docker ps --format "{{.Names}}" | grep -q "^homepage$"; then
        local homepage_response=$(curl -s "http://localhost:3003" 2>/dev/null || echo "")
        if [[ -n "$homepage_response" ]]; then
            record_test "Homepage Dashboard" "PASS" "Dashboard accessible"
        else
            record_test "Homepage Dashboard" "FAIL" "Dashboard not accessible"
        fi
    fi
    
    # Test Homarr
    if docker ps --format "{{.Names}}" | grep -q "^homarr$"; then
        local homarr_response=$(curl -s "http://localhost:7575" 2>/dev/null || echo "")
        if [[ -n "$homarr_response" ]]; then
            record_test "Homarr Dashboard" "PASS" "Dashboard accessible"
        else
            record_test "Homarr Dashboard" "FAIL" "Dashboard not accessible"
        fi
    fi
    
    # Test Dashy if present
    if docker ps --format "{{.Names}}" | grep -q "^dashy$"; then
        local dashy_response=$(curl -s "http://localhost:4000" 2>/dev/null || echo "")
        if [[ -n "$dashy_response" ]]; then
            record_test "Dashy Dashboard" "PASS" "Dashboard accessible"
        else
            record_test "Dashy Dashboard" "FAIL" "Dashboard not accessible"
        fi
    fi
}

test_reverse_proxy_integrations() {
    log "INFO" "Testing reverse proxy integrations"
    
    # Test Nginx Proxy Manager if present
    if docker ps --format "{{.Names}}" | grep -q "^nginx-proxy-manager$"; then
        local npm_response=$(curl -s "http://localhost:81" 2>/dev/null || echo "")
        if [[ -n "$npm_response" ]]; then
            record_test "Nginx Proxy Manager" "PASS" "Admin interface accessible"
        else
            record_test "Nginx Proxy Manager" "FAIL" "Admin interface not accessible"
        fi
    fi
    
    # Test Traefik if present
    if docker ps --format "{{.Names}}" | grep -q "traefik"; then
        local traefik_response=$(curl -s "http://localhost:8080/api/version" 2>/dev/null || echo "")
        if [[ -n "$traefik_response" ]]; then
            record_test "Traefik API" "PASS" "API accessible"
        else
            record_test "Traefik API" "FAIL" "API not accessible"
        fi
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}🔗 Ultimate Media Server 2025 - Service Integration Tests${NC}"
    echo "==========================================================="
    
    log "INFO" "Starting service integration tests"
    log "INFO" "Docker Compose file: $COMPOSE_FILE"
    log "INFO" "Log file: $LOG_FILE"
    
    # Wait for critical services to be ready
    echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
    
    # Core services
    wait_for_service "Prometheus" "9090" 15
    wait_for_service "Grafana" "3000" 15
    
    # Run integration tests
    echo -e "\n${YELLOW}Running integration tests...${NC}"
    
    echo -e "\n${CYAN}Testing ARR Suite Integrations${NC}"
    test_arr_prowlarr_sync
    test_arr_download_clients
    
    echo -e "\n${CYAN}Testing Media Server Integrations${NC}"
    test_media_server_integrations
    
    echo -e "\n${CYAN}Testing Request Service Integrations${NC}"
    test_request_service_integrations
    
    echo -e "\n${CYAN}Testing Download Client Connectivity${NC}"
    test_download_client_connectivity
    
    echo -e "\n${CYAN}Testing Monitoring Integrations${NC}"
    test_monitoring_integrations
    
    echo -e "\n${CYAN}Testing Database Integrations${NC}"
    test_database_integrations
    
    echo -e "\n${CYAN}Testing VPN Network Isolation${NC}"
    test_vpn_network_isolation
    
    echo -e "\n${CYAN}Testing Volume Data Flow${NC}"
    test_volume_data_flow
    
    echo -e "\n${CYAN}Testing Dashboard Integrations${NC}"
    test_dashboard_integrations
    
    echo -e "\n${CYAN}Testing Reverse Proxy Integrations${NC}"
    test_reverse_proxy_integrations
    
    # Generate test report
    generate_report
    
    # Summary
    echo -e "\n${BLUE}Test Summary${NC}"
    echo "============="
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${YELLOW}Warnings: $WARNING_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo -e "Success Rate: $(( (PASSED_TESTS + WARNING_TESTS) * 100 / TOTAL_TESTS ))%"
    echo -e "\nDetailed report: $REPORT_FILE"
    echo -e "Log file: $LOG_FILE"
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        echo -e "\n${RED}⚠️  Some integration tests failed. Check the logs for details.${NC}"
        exit 1
    elif [[ $WARNING_TESTS -gt 0 ]]; then
        echo -e "\n${YELLOW}⚠️  Some tests generated warnings. Review for potential issues.${NC}"
        exit 0
    else
        echo -e "\n${GREEN}✅ All service integration tests passed!${NC}"
        exit 0
    fi
}

# Generate JSON report
generate_report() {
    local report_data="{"
    report_data+='"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",'
    report_data+='"test_suite":"Service Integration Tests",'
    report_data+='"total_tests":'$TOTAL_TESTS','
    report_data+='"passed_tests":'$PASSED_TESTS','
    report_data+='"warning_tests":'$WARNING_TESTS','
    report_data+='"failed_tests":'$FAILED_TESTS','
    report_data+='"success_rate":'$(( (PASSED_TESTS + WARNING_TESTS) * 100 / TOTAL_TESTS ))','
    report_data+='"test_results":{'
    
    local first=true
    for test in "${!TEST_RESULTS[@]}"; do
        if [[ $first == true ]]; then
            first=false
        else
            report_data+=','
        fi
        local status=$(echo "${TEST_RESULTS[$test]}" | cut -d'|' -f1)
        local details=$(echo "${TEST_RESULTS[$test]}" | cut -d'|' -f2-)
        report_data+='"'$test'":{"status":"'$status'","details":"'$details'"}'
    done
    
    report_data+='},'
    report_data+='"integration_matrix":{'
    
    first=true
    for integration in "${!INTEGRATION_MATRIX[@]}"; do
        if [[ $first == true ]]; then
            first=false
        else
            report_data+=','
        fi
        report_data+='"'$integration'":"'${INTEGRATION_MATRIX[$integration]}'"'
    done
    
    report_data+='}}'
    
    echo "$report_data" | jq '.' > "$REPORT_FILE" 2>/dev/null || echo "$report_data" > "$REPORT_FILE"
    log "INFO" "Report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    log "INFO" "Service integration tests completed"
}

# Set up signal handlers
trap cleanup EXIT

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    command -v docker >/dev/null 2>&1 || missing_deps+=("docker")
    command -v curl >/dev/null 2>&1 || missing_deps+=("curl")
    command -v jq >/dev/null 2>&1 || missing_deps+=("jq")
    command -v nc >/dev/null 2>&1 || missing_deps+=("netcat")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "${RED}Missing dependencies: ${missing_deps[*]}${NC}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main "$@"
fi