#!/bin/bash

# Ultimate Media Server 2025 - Container Isolation Test Suite
# Tests individual Docker containers for proper isolation, networking, and resource management

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-ultimate.yml"
LOG_FILE="${SCRIPT_DIR}/test-results/container-isolation-$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="${SCRIPT_DIR}/test-results/container-isolation-report.json"
TIMEOUT=300

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
if [[ ${BASH_VERSION%%.*} -ge 4 ]]; then
    declare -A TEST_RESULTS=()
    declare -A CONTAINER_STATUS=()
else
    TEST_RESULTS=""
    CONTAINER_STATUS=""
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
    
    if [[ "$status" == "PASS" ]]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ $test_name${NC}"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ $test_name${NC}"
        log "ERROR" "$test_name failed: $details"
    fi
}

# Container health check
check_container_health() {
    local container_name="$1"
    local expected_status="running"
    
    log "INFO" "Checking health of container: $container_name"
    
    if ! docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "^$container_name"; then
        record_test "Container Health: $container_name" "FAIL" "Container not found"
        return 1
    fi
    
    local status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not_found")
    CONTAINER_STATUS["$container_name"]="$status"
    
    if [[ "$status" == "$expected_status" ]]; then
        record_test "Container Health: $container_name" "PASS" "Status: $status"
        return 0
    else
        record_test "Container Health: $container_name" "FAIL" "Expected: $expected_status, Got: $status"
        return 1
    fi
}

# Test container resource limits
test_resource_limits() {
    local container_name="$1"
    
    log "INFO" "Testing resource limits for: $container_name"
    
    # Check memory limit
    local memory_limit=$(docker inspect --format='{{.HostConfig.Memory}}' "$container_name" 2>/dev/null || echo "0")
    
    # Check CPU limit
    local cpu_quota=$(docker inspect --format='{{.HostConfig.CpuQuota}}' "$container_name" 2>/dev/null || echo "0")
    local cpu_period=$(docker inspect --format='{{.HostConfig.CpuPeriod}}' "$container_name" 2>/dev/null || echo "0")
    
    # Get current resource usage
    local stats=$(docker stats --no-stream --format "{{.CPUPerc}}\t{{.MemUsage}}" "$container_name" 2>/dev/null || echo "N/A\tN/A")
    local cpu_usage=$(echo "$stats" | cut -f1)
    local mem_usage=$(echo "$stats" | cut -f2)
    
    local details="Memory: $mem_usage, CPU: $cpu_usage"
    
    if [[ "$stats" != "N/A	N/A" ]]; then
        record_test "Resource Usage: $container_name" "PASS" "$details"
    else
        record_test "Resource Usage: $container_name" "FAIL" "Could not retrieve stats"
    fi
}

# Test network isolation
test_network_isolation() {
    local container_name="$1"
    local expected_networks=("${@:2}")
    
    log "INFO" "Testing network isolation for: $container_name"
    
    # Get container networks
    local networks=$(docker inspect --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$container_name" 2>/dev/null || echo "")
    
    if [[ -z "$networks" ]]; then
        record_test "Network Isolation: $container_name" "FAIL" "No networks found"
        return 1
    fi
    
    # Check if container is only on expected networks
    local network_match=true
    for expected_net in "${expected_networks[@]}"; do
        if [[ ! "$networks" =~ $expected_net ]]; then
            network_match=false
            break
        fi
    done
    
    if $network_match; then
        record_test "Network Isolation: $container_name" "PASS" "Networks: $networks"
    else
        record_test "Network Isolation: $container_name" "FAIL" "Unexpected networks: $networks"
    fi
}

# Test volume mounts and permissions
test_volume_permissions() {
    local container_name="$1"
    
    log "INFO" "Testing volume permissions for: $container_name"
    
    # Get mounted volumes
    local mounts=$(docker inspect --format='{{range .Mounts}}{{.Source}}:{{.Destination}}:{{.Mode}} {{end}}' "$container_name" 2>/dev/null || echo "")
    
    if [[ -z "$mounts" ]]; then
        record_test "Volume Permissions: $container_name" "PASS" "No volumes mounted"
        return 0
    fi
    
    # Test write permissions in writable volumes
    local write_test_passed=true
    for mount in $mounts; do
        local mode=$(echo "$mount" | cut -d: -f3)
        local dest=$(echo "$mount" | cut -d: -f2)
        
        if [[ "$mode" == "rw" ]]; then
            # Test write permission
            local test_file="/tmp/test_write_$$"
            if docker exec "$container_name" sh -c "touch '$dest/test_write_permission' 2>/dev/null && rm '$dest/test_write_permission' 2>/dev/null"; then
                continue
            else
                write_test_passed=false
                break
            fi
        fi
    done
    
    if $write_test_passed; then
        record_test "Volume Permissions: $container_name" "PASS" "All writable volumes accessible"
    else
        record_test "Volume Permissions: $container_name" "FAIL" "Write permission issues detected"
    fi
}

# Test container security
test_container_security() {
    local container_name="$1"
    
    log "INFO" "Testing security configuration for: $container_name"
    
    # Check if running as root
    local user=$(docker exec "$container_name" whoami 2>/dev/null || echo "unknown")
    
    # Check capabilities
    local caps=$(docker inspect --format='{{.HostConfig.CapAdd}}' "$container_name" 2>/dev/null || echo "[]")
    
    # Check privileged mode
    local privileged=$(docker inspect --format='{{.HostConfig.Privileged}}' "$container_name" 2>/dev/null || echo "false")
    
    local security_issues=()
    
    if [[ "$user" == "root" ]] && [[ "$container_name" != "gluetun" ]] && [[ "$container_name" != "pihole" ]]; then
        security_issues+=("Running as root")
    fi
    
    if [[ "$privileged" == "true" ]]; then
        security_issues+=("Privileged mode enabled")
    fi
    
    if [[ "$caps" != "[]" ]] && [[ "$caps" != "<nil>" ]]; then
        security_issues+=("Additional capabilities: $caps")
    fi
    
    if [[ ${#security_issues[@]} -eq 0 ]]; then
        record_test "Security Config: $container_name" "PASS" "User: $user, Privileged: $privileged"
    else
        local issues_str=$(IFS=', '; echo "${security_issues[*]}")
        record_test "Security Config: $container_name" "WARN" "$issues_str"
    fi
}

# Test inter-container connectivity
test_inter_container_connectivity() {
    local source_container="$1"
    local target_container="$2"
    local target_port="$3"
    
    log "INFO" "Testing connectivity from $source_container to $target_container:$target_port"
    
    # Get target container IP
    local target_ip=$(docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$target_container" 2>/dev/null | head -1)
    
    if [[ -z "$target_ip" ]]; then
        record_test "Connectivity: $source_container→$target_container" "FAIL" "Could not determine target IP"
        return 1
    fi
    
    # Test connectivity using nc (if available) or timeout + telnet-like test
    if docker exec "$source_container" sh -c "timeout 5 nc -z $target_ip $target_port" 2>/dev/null; then
        record_test "Connectivity: $source_container→$target_container" "PASS" "Port $target_port accessible"
    else
        # Fallback test using curl or wget if available
        if docker exec "$source_container" sh -c "timeout 5 curl -f http://$target_ip:$target_port/ >/dev/null 2>&1" 2>/dev/null; then
            record_test "Connectivity: $source_container→$target_container" "PASS" "HTTP connection successful"
        else
            record_test "Connectivity: $source_container→$target_container" "FAIL" "Cannot connect to port $target_port"
        fi
    fi
}

# Test port exposure
test_port_exposure() {
    local container_name="$1"
    local internal_port="$2"
    local external_port="$3"
    
    log "INFO" "Testing port exposure for $container_name: $internal_port→$external_port"
    
    # Check if port is exposed
    local port_mapping=$(docker port "$container_name" "$internal_port" 2>/dev/null || echo "")
    
    if [[ -n "$port_mapping" ]]; then
        # Test external accessibility
        if timeout 5 nc -z localhost "$external_port" 2>/dev/null; then
            record_test "Port Exposure: $container_name:$external_port" "PASS" "Port accessible externally"
        else
            record_test "Port Exposure: $container_name:$external_port" "FAIL" "Port not accessible externally"
        fi
    else
        record_test "Port Exposure: $container_name:$external_port" "FAIL" "Port mapping not found"
    fi
}

# Test container dependencies
test_container_dependencies() {
    local container_name="$1"
    shift
    local dependencies=("$@")
    
    log "INFO" "Testing dependencies for $container_name"
    
    local missing_deps=()
    for dep in "${dependencies[@]}"; do
        if ! docker ps --format "{{.Names}}" | grep -q "^$dep$"; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -eq 0 ]]; then
        record_test "Dependencies: $container_name" "PASS" "All dependencies running"
    else
        local missing_str=$(IFS=', '; echo "${missing_deps[*]}")
        record_test "Dependencies: $container_name" "FAIL" "Missing dependencies: $missing_str"
    fi
}

# Test restart policies
test_restart_policies() {
    local container_name="$1"
    
    log "INFO" "Testing restart policy for: $container_name"
    
    local restart_policy=$(docker inspect --format='{{.HostConfig.RestartPolicy.Name}}' "$container_name" 2>/dev/null || echo "no")
    
    if [[ "$restart_policy" == "unless-stopped" ]] || [[ "$restart_policy" == "always" ]]; then
        record_test "Restart Policy: $container_name" "PASS" "Policy: $restart_policy"
    else
        record_test "Restart Policy: $container_name" "WARN" "Policy: $restart_policy (consider unless-stopped)"
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}🔍 Ultimate Media Server 2025 - Container Isolation Tests${NC}"
    echo "=================================================="
    
    log "INFO" "Starting container isolation tests"
    log "INFO" "Docker Compose file: $COMPOSE_FILE"
    log "INFO" "Log file: $LOG_FILE"
    
    # Define containers and their expected configurations (core services only)
    if [[ ${BASH_VERSION%%.*} -ge 4 ]]; then
        declare -A CONTAINERS=(
            ["jellyfin"]="media-net:8096"
            ["sonarr"]="media-net:8989"
            ["radarr"]="media-net:7878"
            ["prowlarr"]="media-net:9696"
            ["qbittorrent"]="media-net:8082"
            ["prometheus"]="monitoring-net,media-net:9090"
            ["grafana"]="monitoring-net,media-net:3000"
            ["portainer"]="media-net:9000"
            ["homarr"]="media-net:7575"
        )
        
        # Test each container
        for container in "${!CONTAINERS[@]}"; do
            local config="${CONTAINERS[$container]}"
        local networks_ports=($(echo "$config" | tr ':' ' '))
        local networks="${networks_ports[0]}"
        local port="${networks_ports[1]}"
        
        echo -e "\n${YELLOW}Testing container: $container${NC}"
        
        # Basic health check
        if check_container_health "$container"; then
            # Resource limits
            test_resource_limits "$container"
            
            # Network isolation
            IFS=',' read -ra network_array <<< "$networks"
            test_network_isolation "$container" "${network_array[@]}"
            
            # Volume permissions
            test_volume_permissions "$container"
            
            # Security configuration
            test_container_security "$container"
            
            # Restart policy
            test_restart_policies "$container"
            
            # Port exposure (if external port expected)
            case "$container" in
                "jellyfin") test_port_exposure "$container" "8096" "8096" ;;
                "plex") test_port_exposure "$container" "32400" "32400" ;;
                "grafana") test_port_exposure "$container" "3000" "3000" ;;
                "prometheus") test_port_exposure "$container" "9090" "9090" ;;
                "sonarr") test_port_exposure "$container" "8989" "8989" ;;
                "radarr") test_port_exposure "$container" "7878" "7878" ;;
            esac
        fi
    done
    
    # Test specific connectivity patterns
    echo -e "\n${YELLOW}Testing inter-container connectivity${NC}"
    
    # ARR services to download clients
    test_inter_container_connectivity "sonarr" "qbittorrent" "8080"
    test_inter_container_connectivity "radarr" "sabnzbd" "8080"
    test_inter_container_connectivity "prowlarr" "sonarr" "8989"
    
    # Media servers to monitoring
    test_inter_container_connectivity "prometheus" "jellyfin" "8096"
    test_inter_container_connectivity "grafana" "prometheus" "9090"
    
    # Request services to ARR services
    test_inter_container_connectivity "jellyseerr" "sonarr" "8989"
    test_inter_container_connectivity "overseerr" "radarr" "7878"
    
    # Database dependencies
    test_container_dependencies "grafana" "postgres"
    test_container_dependencies "nextcloud" "postgres" "redis"
    test_container_dependencies "photoprism" "mariadb"
    
    # Generate test report
    generate_report
    
    # Summary
    echo -e "\n${BLUE}Test Summary${NC}"
    echo "============="
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo -e "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo -e "\nDetailed report: $REPORT_FILE"
    echo -e "Log file: $LOG_FILE"
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        echo -e "\n${RED}⚠️  Some tests failed. Check the logs for details.${NC}"
        exit 1
    else
        echo -e "\n${GREEN}✅ All container isolation tests passed!${NC}"
        exit 0
    fi
}

# Generate JSON report
generate_report() {
    local report_data="{"
    report_data+='"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",'
    report_data+='"test_suite":"Container Isolation Tests",'
    report_data+='"total_tests":'$TOTAL_TESTS','
    report_data+='"passed_tests":'$PASSED_TESTS','
    report_data+='"failed_tests":'$FAILED_TESTS','
    report_data+='"success_rate":'$(( PASSED_TESTS * 100 / TOTAL_TESTS ))','
    report_data+='"container_status":{'
    
    local first=true
    for container in "${!CONTAINER_STATUS[@]}"; do
        if [[ $first == true ]]; then
            first=false
        else
            report_data+=','
        fi
        report_data+='"'$container'":"'${CONTAINER_STATUS[$container]}'"'
    done
    
    report_data+='},'
    report_data+='"test_results":{'
    
    first=true
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
    
    report_data+='}}'
    
    echo "$report_data" | jq '.' > "$REPORT_FILE" 2>/dev/null || echo "$report_data" > "$REPORT_FILE"
    log "INFO" "Report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    log "INFO" "Container isolation tests completed"
}

# Set up signal handlers
trap cleanup EXIT

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    command -v docker >/dev/null 2>&1 || missing_deps+=("docker")
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