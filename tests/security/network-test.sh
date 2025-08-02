#!/bin/bash

# Network security and isolation testing
# Tests network segmentation, firewall rules, and service exposure

set -euo pipefail

# Configuration
REPORT_DIR="${1:-./reports/security}"
TEST_TIMEOUT="${2:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure report directory exists
mkdir -p "${REPORT_DIR}"

# Function to log with timestamp
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to test port accessibility
test_port() {
    local host="$1"
    local port="$2"
    local expected="$3"  # "open" or "closed"
    local service="$4"
    
    if timeout "$TEST_TIMEOUT" bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
        exec 3<&-
        exec 3>&-
        if [ "$expected" = "open" ]; then
            log "${GREEN}✓ ${service} port ${port} is accessible as expected${NC}"
            return 0
        else
            log "${RED}✗ ${service} port ${port} is unexpectedly accessible${NC}"
            return 1
        fi
    else
        if [ "$expected" = "closed" ]; then
            log "${GREEN}✓ ${service} port ${port} is properly blocked${NC}"
            return 0
        else
            log "${RED}✗ ${service} port ${port} is unexpectedly blocked${NC}"
            return 1
        fi
    fi
}

# Function to test network isolation
test_network_isolation() {
    local container1="$1"
    local container2="$2"
    local should_communicate="$3"  # "yes" or "no"
    
    log "${BLUE}Testing network isolation: ${container1} -> ${container2}${NC}"
    
    # Get container IPs
    local ip1=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container1" 2>/dev/null | head -1)
    local ip2=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container2" 2>/dev/null | head -1)
    
    if [ -z "$ip1" ] || [ -z "$ip2" ]; then
        log "${YELLOW}⚠ Could not get IPs for containers (may not be running)${NC}"
        return 0
    fi
    
    # Test ping connectivity
    if docker exec "$container1" ping -c 1 -W 2 "$ip2" >/dev/null 2>&1; then
        if [ "$should_communicate" = "yes" ]; then
            log "${GREEN}✓ ${container1} can reach ${container2} as expected${NC}"
            return 0
        else
            log "${RED}✗ ${container1} can reach ${container2} but shouldn't${NC}"
            return 1
        fi
    else
        if [ "$should_communicate" = "no" ]; then
            log "${GREEN}✓ ${container1} cannot reach ${container2} as expected${NC}"
            return 0
        else
            log "${RED}✗ ${container1} cannot reach ${container2} but should${NC}"
            return 1
        fi
    fi
}

# Function to scan for open ports
scan_open_ports() {
    local target="$1"
    local name="$2"
    
    log "${BLUE}Scanning open ports on ${name}${NC}"
    
    local open_ports=()
    for port in {80,443,8080,8081,8082,8096,8181,8989,7878,9696,5055,3000,3001,9000,9090,22,21,23,25,53,110,143,993,995}; do
        if timeout 2 bash -c "exec 3<>/dev/tcp/$target/$port" 2>/dev/null; then
            exec 3<&-
            exec 3>&-
            open_ports+=("$port")
        fi
    done
    
    if [ ${#open_ports[@]} -gt 0 ]; then
        log "${YELLOW}Open ports on ${name}: ${open_ports[*]}${NC}"
    else
        log "${GREEN}No unexpected ports found open on ${name}${NC}"
    fi
    
    # Save to report
    echo "${name}: ${open_ports[*]}" >> "${REPORT_DIR}/open-ports.txt"
}

# Function to test VPN isolation
test_vpn_isolation() {
    log "${BLUE}Testing VPN network isolation${NC}"
    
    # Check if VPN container exists
    if ! docker ps --format "table {{.Names}}" | grep -q "^vpn$"; then
        log "${YELLOW}⚠ VPN container not running, skipping VPN tests${NC}"
        return 0
    fi
    
    # Test that qBittorrent traffic goes through VPN
    local vpn_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' vpn 2>/dev/null | head -1)
    
    if [ -n "$vpn_ip" ]; then
        # Check qBittorrent can reach VPN
        if docker exec qbittorrent ping -c 1 -W 2 "$vpn_ip" >/dev/null 2>&1; then
            log "${GREEN}✓ qBittorrent can reach VPN container${NC}"
        else
            log "${RED}✗ qBittorrent cannot reach VPN container${NC}"
            return 1
        fi
        
        # Check external IP through VPN
        local external_ip=$(docker exec qbittorrent wget -qO- http://ipinfo.io/ip 2>/dev/null || echo "")
        if [ -n "$external_ip" ]; then
            log "${GREEN}✓ qBittorrent external IP: ${external_ip}${NC}"
        else
            log "${YELLOW}⚠ Could not determine qBittorrent external IP${NC}"
        fi
    else
        log "${YELLOW}⚠ Could not get VPN container IP${NC}"
    fi
}

# Function to test SSL/TLS configuration
test_ssl_tls() {
    log "${BLUE}Testing SSL/TLS configurations${NC}"
    
    local services=(
        "localhost:443:traefik"
        "localhost:9443:portainer"
        "localhost:8920:jellyfin"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r host port name <<< "$service"
        
        if timeout 5 openssl s_client -connect "$host:$port" -verify_return_error >/dev/null 2>&1; then
            log "${GREEN}✓ ${name} SSL/TLS configuration valid${NC}"
        else
            log "${YELLOW}⚠ ${name} SSL/TLS not configured or invalid${NC}"
        fi
    done
}

# Function to check for default credentials
check_default_credentials() {
    log "${BLUE}Checking for default credentials${NC}"
    
    local credentials=(
        "localhost:3000:admin:admin:Grafana"
        "localhost:9000:admin::Portainer"
        "localhost:8080:admin:adminpass:qBittorrent"
    )
    
    for cred in "${credentials[@]}"; do
        IFS=':' read -r host port user pass service <<< "$cred"
        
        local auth_header=""
        if [ -n "$user" ] && [ -n "$pass" ]; then
            auth_header="Authorization: Basic $(echo -n "$user:$pass" | base64)"
        elif [ -n "$user" ]; then
            auth_header="Authorization: Basic $(echo -n "$user:" | base64)"
        fi
        
        if [ -n "$auth_header" ]; then
            if curl -s -H "$auth_header" "http://$host:$port/api" >/dev/null 2>&1; then
                log "${RED}✗ ${service} may be using default credentials${NC}"
            else
                log "${GREEN}✓ ${service} default credentials not working${NC}"
            fi
        fi
    done
}

# Function to test Docker socket security
test_docker_socket_security() {
    log "${BLUE}Testing Docker socket security${NC}"
    
    # Check which containers have Docker socket access
    local containers_with_docker=()
    
    for container in $(docker ps --format "{{.Names}}"); do
        if docker inspect "$container" | jq -r '.[].Mounts[].Source' | grep -q "/var/run/docker.sock"; then
            containers_with_docker+=("$container")
        fi
    done
    
    if [ ${#containers_with_docker[@]} -gt 0 ]; then
        log "${YELLOW}Containers with Docker socket access: ${containers_with_docker[*]}${NC}"
        
        # Check if access is read-only
        for container in "${containers_with_docker[@]}"; do
            local mode=$(docker inspect "$container" | jq -r '.[].Mounts[] | select(.Source=="/var/run/docker.sock") | .Mode')
            if [ "$mode" = "ro" ]; then
                log "${GREEN}✓ ${container} has read-only Docker socket access${NC}"
            else
                log "${YELLOW}⚠ ${container} has read-write Docker socket access${NC}"
            fi
        done
    else
        log "${GREEN}✓ No containers have Docker socket access${NC}"
    fi
}

# Main execution
log "${GREEN}=== Starting Network Security Tests ===${NC}"

# Initialize report
cat > "${REPORT_DIR}/network-security-report.json" << EOF
{
  "scan_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tests": {
    "port_accessibility": {},
    "network_isolation": {},
    "vpn_isolation": {},
    "ssl_tls": {},
    "default_credentials": {},
    "docker_socket": {}
  },
  "summary": {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0
  }
}
EOF

test_count=0
passed_count=0
failed_count=0

# Test 1: Port accessibility
log "${YELLOW}=== Testing Port Accessibility ===${NC}"
services=(
    "localhost:80:open:HTTP"
    "localhost:443:open:HTTPS"
    "localhost:8096:open:Jellyfin"
    "localhost:8989:open:Sonarr"
    "localhost:7878:open:Radarr"
    "localhost:9696:open:Prowlarr"
    "localhost:22:closed:SSH"
    "localhost:3306:closed:MySQL"
    "localhost:5432:closed:PostgreSQL"
)

for service in "${services[@]}"; do
    IFS=':' read -r host port expected name <<< "$service"
    if test_port "$host" "$port" "$expected" "$name"; then
        ((passed_count++))
    else
        ((failed_count++))
    fi
    ((test_count++))
done

# Test 2: Network isolation
log "${YELLOW}=== Testing Network Isolation ===${NC}"
isolation_tests=(
    "jellyfin:qbittorrent:no"
    "sonarr:radarr:yes"
    "homepage:grafana:yes"
)

for test in "${isolation_tests[@]}"; do
    IFS=':' read -r container1 container2 should_communicate <<< "$test"
    if test_network_isolation "$container1" "$container2" "$should_communicate"; then
        ((passed_count++))
    else
        ((failed_count++))
    fi
    ((test_count++))
done

# Test 3: VPN isolation
log "${YELLOW}=== Testing VPN Isolation ===${NC}"
if test_vpn_isolation; then
    ((passed_count++))
else
    ((failed_count++))
fi
((test_count++))

# Test 4: Port scanning
log "${YELLOW}=== Scanning for Open Ports ===${NC}"
scan_open_ports "localhost" "localhost"

# Test 5: SSL/TLS
log "${YELLOW}=== Testing SSL/TLS ===${NC}"
test_ssl_tls
((test_count++))

# Test 6: Default credentials
log "${YELLOW}=== Checking Default Credentials ===${NC}"
check_default_credentials
((test_count++))

# Test 7: Docker socket security
log "${YELLOW}=== Testing Docker Socket Security ===${NC}"
test_docker_socket_security
((test_count++))

# Update report summary
tmp_file=$(mktemp)
jq ".summary.total_tests = $test_count | .summary.passed = $passed_count | .summary.failed = $failed_count" \
   "${REPORT_DIR}/network-security-report.json" > "$tmp_file"
mv "$tmp_file" "${REPORT_DIR}/network-security-report.json"

# Final summary
log "${GREEN}=== Network Security Test Summary ===${NC}"
log "Total tests: $test_count"
log "Passed: $passed_count"
log "Failed: $failed_count"
log "Success rate: $(( passed_count * 100 / test_count ))%"

if [ $failed_count -gt 0 ]; then
    log "${RED}⚠ Some security tests failed. Review the reports.${NC}"
    exit 1
else
    log "${GREEN}✓ All network security tests passed${NC}"
    exit 0
fi