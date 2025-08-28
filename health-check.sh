#\!/bin/bash
# Ultimate Media Server 2025 - Comprehensive Health Check
# Checks all services and reports overall system health

set -e

# Health check for main services
check_service() {
    local service=$1
    local port=$2
    
    # Check if port is listening
    if netstat -ln 2>/dev/null | grep -q ":${port} " || ss -ln 2>/dev/null | grep -q ":${port} "; then
        echo "✓ $service (port $port): healthy"
        return 0
    else
        echo "✗ $service (port $port): unhealthy"
        return 1
    fi
}

# Main health check
main() {
    echo "=== Ultimate Media Server 2025 Health Check ==="
    echo "Timestamp: $(date)"
    echo ""
    
    failed=0
    
    # Check core services
    check_service "Caddy" 80 || failed=$((failed + 1))
    check_service "Jellyfin" 8096 || failed=$((failed + 1))
    check_service "Sonarr" 8989 || failed=$((failed + 1))
    check_service "Radarr" 7878 || failed=$((failed + 1))
    check_service "Lidarr" 8686 || failed=$((failed + 1))
    check_service "Prowlarr" 9696 || failed=$((failed + 1))
    check_service "Bazarr" 6767 || failed=$((failed + 1))
    check_service "qBittorrent" 8080 || failed=$((failed + 1))
    check_service "SABnzbd" 8085 || failed=$((failed + 1))
    check_service "Transmission" 9091 || failed=$((failed + 1))
    check_service "API Server" 3002 || failed=$((failed + 1))
    check_service "Uptime Kuma" 3001 || failed=$((failed + 1))
    
    echo ""
    if [ $failed -eq 0 ]; then
        echo "Overall Status: HEALTHY"
        exit 0
    elif [ $failed -le 2 ]; then
        echo "Overall Status: DEGRADED ($failed services down)"
        exit 0  # Still pass for minor issues
    else
        echo "Overall Status: UNHEALTHY ($failed services down)"
        exit 1
    fi
}

# Create logs directory if it doesn't exist
mkdir -p /logs

# Run main function
main "$@" | tee -a /logs/health-check.log
EOF < /dev/null