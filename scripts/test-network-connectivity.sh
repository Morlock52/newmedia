#!/bin/bash

# ============================================================================
# MEDIA SERVER NETWORK CONNECTIVITY TEST SUITE
# ============================================================================
# Tests network configuration, DNS resolution, and inter-service communication
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
LOG_FILE="${PROJECT_ROOT}/logs/network-test.log"

# Ensure log directory exists
mkdir -p "${PROJECT_ROOT}/logs"

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

success() {
    log "${GREEN}✓ $1${NC}"
}

warning() {
    log "${YELLOW}⚠ $1${NC}"
}

error() {
    log "${RED}✗ $1${NC}"
}

info() {
    log "${BLUE}ℹ $1${NC}"
}

# Test functions
test_docker_networks() {
    info "Testing Docker network configuration..."
    
    local networks=(
        "media-net"
        "downloads-net"
        "vpn-net"
        "monitoring-net"
        "management-net"
    )
    
    for network in "${networks[@]}"; do
        if docker network ls | grep -q "$network"; then
            success "Network $network exists"
            
            # Get network details
            local subnet=$(docker network inspect "$network" --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || echo "N/A")
            local gateway=$(docker network inspect "$network" --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || echo "N/A")
            
            info "  └── Subnet: $subnet, Gateway: $gateway"
        else
            warning "Network $network does not exist"
        fi
    done
}

test_container_connectivity() {
    info "Testing container-to-container connectivity..."
    
    # Define service pairs to test
    declare -A service_pairs=(
        ["sonarr"]="prowlarr:9696"
        ["radarr"]="prowlarr:9696"
        ["jellyfin"]="sonarr:8989"
        ["prowlarr"]="qbittorrent:8080"
        ["uptime-kuma"]="jellyfin:8096"
    )
    
    for source_service in "${!service_pairs[@]}"; do
        local target="${service_pairs[$source_service]}"
        local target_host="${target%:*}"
        local target_port="${target#*:}"
        
        if docker ps | grep -q "$source_service"; then
            info "Testing $source_service → $target_host:$target_port"
            
            # Test DNS resolution
            if docker exec "$source_service" nslookup "$target_host" &>/dev/null; then
                success "DNS resolution: $source_service can resolve $target_host"
            else
                # Fallback to getent if nslookup is not available
                if docker exec "$source_service" getent hosts "$target_host" &>/dev/null; then
                    success "DNS resolution: $source_service can resolve $target_host (via getent)"
                else
                    warning "DNS resolution failed: $source_service cannot resolve $target_host"
                fi
            fi
            
            # Test port connectivity
            if timeout 10 docker exec "$source_service" sh -c "echo > /dev/tcp/$target_host/$target_port" &>/dev/null; then
                success "Port connectivity: $source_service can reach $target_host:$target_port"
            else
                warning "Port connectivity failed: $source_service cannot reach $target_host:$target_port"
            fi
        else
            warning "Source container $source_service is not running"
        fi
    done
}

test_service_health() {
    info "Testing service health endpoints..."
    
    declare -A health_endpoints=(
        ["jellyfin"]="http://jellyfin:8096/health"
        ["sonarr"]="http://sonarr:8989/ping"
        ["radarr"]="http://radarr:7878/ping"
        ["prowlarr"]="http://prowlarr:9696/ping"
        ["qbittorrent"]="http://qbittorrent:8080/api/v2/app/version"
        ["uptime-kuma"]="http://uptime-kuma:3001"
    )
    
    for service in "${!health_endpoints[@]}"; do
        local endpoint="${health_endpoints[$service]}"
        
        if docker ps | grep -q "$service"; then
            info "Testing health endpoint for $service: $endpoint"
            
            # Use a generic container to test the endpoint
            if docker run --rm --network newmedia_media-net curlimages/curl:latest \
                curl -sf --connect-timeout 10 --max-time 30 "$endpoint" &>/dev/null; then
                success "Health check passed for $service"
            else
                warning "Health check failed for $service at $endpoint"
            fi
        else
            warning "Service $service is not running"
        fi
    done
}

test_dns_resolution() {
    info "Testing DNS resolution within networks..."
    
    local services=("jellyfin" "sonarr" "radarr" "prowlarr" "qbittorrent")
    
    for source in "${services[@]}"; do
        if docker ps | grep -q "$source"; then
            for target in "${services[@]}"; do
                if [ "$source" != "$target" ] && docker ps | grep -q "$target"; then
                    # Test DNS resolution using a simple approach
                    if docker exec "$source" sh -c "getent hosts $target > /dev/null 2>&1"; then
                        success "DNS: $source can resolve $target"
                    else
                        warning "DNS: $source cannot resolve $target"
                    fi
                fi
            done
        fi
    done
}

test_network_performance() {
    info "Testing network performance between containers..."
    
    # Test with iperf3 if available
    if docker run --rm --network newmedia_media-net nicolaka/netshoot iperf3 --version &>/dev/null; then
        info "Running network performance tests..."
        
        # Start iperf3 server
        docker run -d --name iperf-server --network newmedia_media-net nicolaka/netshoot iperf3 -s
        sleep 2
        
        # Run client test
        if docker run --rm --network newmedia_media-net nicolaka/netshoot \
            iperf3 -c iperf-server -t 10 -f M | grep -E "sender|receiver"; then
            success "Network performance test completed"
        else
            warning "Network performance test failed"
        fi
        
        # Cleanup
        docker stop iperf-server &>/dev/null || true
        docker rm iperf-server &>/dev/null || true
    else
        info "iperf3 not available, skipping performance tests"
    fi
}

test_external_connectivity() {
    info "Testing external connectivity through VPN..."
    
    if docker ps | grep -q gluetun; then
        # Test external IP through VPN
        local external_ip=$(docker exec gluetun sh -c "curl -s ipinfo.io/ip" 2>/dev/null || echo "failed")
        
        if [ "$external_ip" != "failed" ]; then
            success "VPN external IP: $external_ip"
        else
            warning "Failed to get external IP through VPN"
        fi
        
        # Test if download clients can reach external sites
        if docker exec qbittorrent sh -c "curl -s --connect-timeout 10 google.com > /dev/null 2>&1"; then
            success "Download client can reach external sites"
        else
            warning "Download client cannot reach external sites"
        fi
    else
        warning "VPN container (gluetun) is not running"
    fi
}

generate_network_report() {
    info "Generating network configuration report..."
    
    local report_file="${PROJECT_ROOT}/logs/network-report.txt"
    
    cat > "$report_file" << EOF
# ============================================================================
# MEDIA SERVER NETWORK CONFIGURATION REPORT
# ============================================================================
# Generated: $(date)
# ============================================================================

## Docker Networks
$(docker network ls | grep -E "(media-net|downloads-net|vpn-net|monitoring-net|management-net)" || echo "No custom networks found")

## Network Details
EOF
    
    for network in media-net downloads-net vpn-net monitoring-net management-net; do
        if docker network ls | grep -q "$network"; then
            echo "### Network: $network" >> "$report_file"
            docker network inspect "$network" --format '{{json .}}' | jq -r '
                "Subnet: " + (.IPAM.Config[0].Subnet // "N/A") + "\n" +
                "Gateway: " + (.IPAM.Config[0].Gateway // "N/A") + "\n" +
                "Driver: " + .Driver + "\n" +
                "MTU: " + (.Options["com.docker.network.mtu"] // "default") + "\n"
            ' 2>/dev/null >> "$report_file" || echo "Details not available" >> "$report_file"
            echo "" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## Container Network Assignments
$(docker ps --format "table {{.Names}}\t{{.Networks}}" | grep -v "NETWORKS")

## Service Endpoints
- Jellyfin: http://jellyfin:8096
- Sonarr: http://sonarr:8989
- Radarr: http://radarr:7878
- Prowlarr: http://prowlarr:9696
- qBittorrent: http://qbittorrent:8080
- Uptime Kuma: http://uptime-kuma:3001

EOF
    
    success "Network report generated: $report_file"
}

optimize_network_settings() {
    info "Applying network optimizations..."
    
    # Set kernel network parameters for better performance
    if [ "$(uname)" = "Linux" ]; then
        info "Applying Linux network optimizations..."
        
        # These would typically go in /etc/sysctl.conf for permanent settings
        sysctl -w net.core.rmem_max=134217728 2>/dev/null || warning "Could not set rmem_max"
        sysctl -w net.core.wmem_max=134217728 2>/dev/null || warning "Could not set wmem_max"
        sysctl -w net.ipv4.tcp_rmem="4096 131072 134217728" 2>/dev/null || warning "Could not set tcp_rmem"
        sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728" 2>/dev/null || warning "Could not set tcp_wmem"
        
        success "Network optimizations applied"
    else
        info "Non-Linux system detected, skipping kernel optimizations"
    fi
}

# Main execution
main() {
    log "${BLUE}============================================================================"
    log "STARTING MEDIA SERVER NETWORK CONNECTIVITY TESTS"
    log "============================================================================${NC}"
    
    test_docker_networks
    echo
    
    test_container_connectivity
    echo
    
    test_service_health
    echo
    
    test_dns_resolution
    echo
    
    test_network_performance
    echo
    
    test_external_connectivity
    echo
    
    generate_network_report
    echo
    
    optimize_network_settings
    
    log "${BLUE}============================================================================"
    log "NETWORK TESTS COMPLETED"
    log "============================================================================${NC}"
    log "Results logged to: $LOG_FILE"
    log "Network report: ${PROJECT_ROOT}/logs/network-report.txt"
}

# Command line options
case "${1:-}" in
    --networks-only)
        test_docker_networks
        ;;
    --connectivity-only)
        test_container_connectivity
        ;;
    --health-only)
        test_service_health
        ;;
    --dns-only)
        test_dns_resolution
        ;;
    --performance-only)
        test_network_performance
        ;;
    --report-only)
        generate_network_report
        ;;
    --optimize-only)
        optimize_network_settings
        ;;
    *)
        main
        ;;
esac