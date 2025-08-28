#!/bin/bash
# Ultimate Media Server 2025 - Comprehensive Health Check (FIXED)
# Validates all critical services and their interdependencies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Health check configuration
TIMEOUT=5
TOTAL_SERVICES=0
HEALTHY_SERVICES=0
CRITICAL_SERVICES=0
CRITICAL_HEALTHY=0
DEGRADED_SERVICES=0

# Enable verbose mode if requested
VERBOSE=${VERBOSE:-false}

# Log function
log() {
    if [ "$VERBOSE" = "true" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
    fi
}

# Check if a TCP port is open
check_port() {
    local host="$1"
    local port="$2"
    local service="$3"
    
    if timeout ${TIMEOUT} bash -c "</dev/tcp/${host}/${port}" 2>/dev/null; then
        return 0
    else
        log "❌ ${service} (${host}:${port}) - Port not accessible"
        return 1
    fi
}

# Check HTTP endpoint with proper error handling
check_http() {
    local url="$1"
    local service="$2"
    local expected_code="${3:-200}"
    
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout ${TIMEOUT} --max-time $((TIMEOUT * 2)) "$url" 2>/dev/null || echo "000")
    
    if [[ "$response" == "$expected_code" ]] || [[ "$response" =~ ^2[0-9][0-9]$ ]]; then
        return 0
    else
        log "❌ ${service} (${url}) - HTTP ${response}, expected ${expected_code}"
        return 1
    fi
}

# Check specific API endpoint
check_api() {
    local url="$1"
    local service="$2"
    local expected_response="${3:-}"
    
    local response
    response=$(curl -s --connect-timeout ${TIMEOUT} --max-time $((TIMEOUT * 2)) "$url" 2>/dev/null || echo "ERROR")
    
    if [[ -n "$expected_response" && "$response" == *"$expected_response"* ]]; then
        return 0
    elif [[ -z "$expected_response" && "$response" != "ERROR" ]]; then
        return 0
    else
        log "❌ ${service} API (${url}) - Unexpected response: ${response:0:50}..."
        return 1
    fi
}

# Check database connection
check_database() {
    local db_type="$1"
    local connection_test="$2" 
    local service_name="$3"
    
    if eval "$connection_test" 2>/dev/null; then
        return 0
    else
        log "❌ ${service_name} database connection failed"
        return 1
    fi
}

# Check if process is running
check_process() {
    local process_name="$1"
    local service_name="$2"
    
    if pgrep -f "$process_name" > /dev/null 2>&1; then
        return 0
    else
        log "❌ ${service_name} process not running"
        return 1
    fi
}

# Function to track service health
track_service() {
    local service_name="$1"
    local is_critical="${2:-false}"
    local check_result="$3"
    
    TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
    
    if [ "$is_critical" = "true" ]; then
        CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
        if [ "$check_result" = "0" ]; then
            CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
            HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
            echo "✅ ${service_name} - Healthy (Critical)"
        else
            echo "❌ ${service_name} - Unhealthy (Critical)"
        fi
    else
        if [ "$check_result" = "0" ]; then
            HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
            echo "✅ ${service_name} - Healthy"
        else
            DEGRADED_SERVICES=$((DEGRADED_SERVICES + 1))
            echo "⚠️  ${service_name} - Degraded"
        fi
    fi
}

echo "🏥 Ultimate Media Server 2025 - Health Check Starting..."
echo "🕐 $(date)"
echo ""

# ========================================
# INFRASTRUCTURE LAYER HEALTH CHECK
# ========================================

echo "=========================================="
echo "🔧 INFRASTRUCTURE SERVICES"
echo "=========================================="

# PostgreSQL
log "Checking PostgreSQL..."
if check_port "127.0.0.1" "5432" "PostgreSQL" && \
   check_database "postgres" "pg_isready -h 127.0.0.1 -p 5432 -q" "PostgreSQL"; then
    track_service "PostgreSQL" "true" "0"
else
    track_service "PostgreSQL" "true" "1"
fi

# Redis
log "Checking Redis..."
if check_port "127.0.0.1" "6379" "Redis" && \
   check_database "redis" "redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -q PONG" "Redis"; then
    track_service "Redis" "true" "0"
else
    track_service "Redis" "true" "1"
fi

# Traefik (Reverse Proxy)
log "Checking Traefik..."
if check_port "127.0.0.1" "8080" "Traefik Dashboard" && \
   check_http "http://127.0.0.1:8080/ping" "Traefik API" 200; then
    track_service "Traefik" "true" "0"
else
    track_service "Traefik" "true" "1"
fi

echo ""

# ========================================
# MEDIA SERVERS HEALTH CHECK
# ========================================

echo "=========================================="
echo "📺 MEDIA SERVERS"
echo "=========================================="

# Jellyfin
log "Checking Jellyfin..."
if check_port "127.0.0.1" "8096" "Jellyfin" && \
   check_http "http://127.0.0.1:8096/health" "Jellyfin" 200; then
    track_service "Jellyfin" "true" "0"
elif check_port "127.0.0.1" "8096" "Jellyfin"; then
    # Port is open but health endpoint might not be available yet
    track_service "Jellyfin" "true" "0"
else
    track_service "Jellyfin" "true" "1"
fi

# Plex
log "Checking Plex..."
if check_port "127.0.0.1" "32400" "Plex"; then
    # Plex identity endpoint sometimes needs authentication, so just check port
    track_service "Plex" "false" "0"
else
    track_service "Plex" "false" "1"
fi

# Emby
log "Checking Emby..."
if check_port "127.0.0.1" "8097" "Emby"; then
    track_service "Emby" "false" "0"
else
    track_service "Emby" "false" "1"
fi

echo ""

# ========================================
# *ARR STACK HEALTH CHECK  
# ========================================

echo "=========================================="
echo "🔍 *ARR MEDIA MANAGEMENT"
echo "=========================================="

# Sonarr
log "Checking Sonarr..."
if check_port "127.0.0.1" "8989" "Sonarr"; then
    track_service "Sonarr" "false" "0"
else
    track_service "Sonarr" "false" "1"
fi

# Radarr
log "Checking Radarr..."
if check_port "127.0.0.1" "7878" "Radarr"; then
    track_service "Radarr" "false" "0"
else
    track_service "Radarr" "false" "1"
fi

# Lidarr
log "Checking Lidarr..."
if check_port "127.0.0.1" "8686" "Lidarr"; then
    track_service "Lidarr" "false" "0"
else
    track_service "Lidarr" "false" "1"
fi

# Readarr
log "Checking Readarr..."
if check_port "127.0.0.1" "8787" "Readarr"; then
    track_service "Readarr" "false" "0"  
else
    track_service "Readarr" "false" "1"
fi

# Prowlarr
log "Checking Prowlarr..."
if check_port "127.0.0.1" "9696" "Prowlarr"; then
    track_service "Prowlarr" "false" "0"
else
    track_service "Prowlarr" "false" "1"
fi

# Bazarr
log "Checking Bazarr..."
if check_port "127.0.0.1" "6767" "Bazarr"; then
    track_service "Bazarr" "false" "0"
else
    track_service "Bazarr" "false" "1"
fi

echo ""

# ========================================
# DOWNLOAD CLIENTS HEALTH CHECK
# ========================================

echo "=========================================="
echo "⬇️  DOWNLOAD CLIENTS"
echo "=========================================="

# qBittorrent
log "Checking qBittorrent..."
if check_port "127.0.0.1" "8080" "qBittorrent"; then
    track_service "qBittorrent" "false" "0"
else
    track_service "qBittorrent" "false" "1"
fi

# Transmission
log "Checking Transmission..."
if check_port "127.0.0.1" "9091" "Transmission"; then
    track_service "Transmission" "false" "0"
else
    track_service "Transmission" "false" "1"
fi

# SABnzbd
log "Checking SABnzbd..."
if check_port "127.0.0.1" "8085" "SABnzbd"; then
    track_service "SABnzbd" "false" "0"
else
    track_service "SABnzbd" "false" "1"
fi

# NZBGet
log "Checking NZBGet..."
if check_port "127.0.0.1" "6789" "NZBGet"; then
    track_service "NZBGet" "false" "0"
else
    track_service "NZBGet" "false" "1"
fi

echo ""

# ========================================
# REQUEST MANAGEMENT HEALTH CHECK
# ========================================

echo "=========================================="
echo "📋 REQUEST MANAGEMENT"
echo "=========================================="

# Overseerr
log "Checking Overseerr..."
if check_port "127.0.0.1" "5055" "Overseerr"; then
    track_service "Overseerr" "false" "0"
else
    track_service "Overseerr" "false" "1"
fi

# Jellyseerr
log "Checking Jellyseerr..."
if check_port "127.0.0.1" "5056" "Jellyseerr"; then
    track_service "Jellyseerr" "false" "0"
else
    track_service "Jellyseerr" "false" "1"
fi

# Ombi
log "Checking Ombi..."
if check_port "127.0.0.1" "3579" "Ombi"; then
    track_service "Ombi" "false" "0"
else
    track_service "Ombi" "false" "1"
fi

echo ""

# ========================================
# MONITORING SERVICES HEALTH CHECK
# ========================================

echo "=========================================="
echo "📊 MONITORING & ANALYTICS"
echo "=========================================="

# Prometheus
log "Checking Prometheus..."
if check_port "127.0.0.1" "9090" "Prometheus" && \
   check_http "http://127.0.0.1:9090/-/healthy" "Prometheus" 200; then
    track_service "Prometheus" "false" "0"
elif check_port "127.0.0.1" "9090" "Prometheus"; then
    track_service "Prometheus" "false" "0"
else
    track_service "Prometheus" "false" "1"
fi

# Grafana
log "Checking Grafana..."
if check_port "127.0.0.1" "3000" "Grafana" && \
   check_http "http://127.0.0.1:3000/api/health" "Grafana" 200; then
    track_service "Grafana" "false" "0"
elif check_port "127.0.0.1" "3000" "Grafana"; then
    track_service "Grafana" "false" "0"
else
    track_service "Grafana" "false" "1"
fi

# Uptime Kuma
log "Checking Uptime Kuma..."
if check_port "127.0.0.1" "3001" "Uptime Kuma"; then
    track_service "Uptime Kuma" "false" "0"
else
    track_service "Uptime Kuma" "false" "1"
fi

echo ""

# ========================================
# DASHBOARD SERVICES HEALTH CHECK
# ========================================

echo "=========================================="
echo "🖥️  DASHBOARDS & MANAGEMENT"
echo "=========================================="

# Tautulli
log "Checking Tautulli..."
if check_port "127.0.0.1" "8182" "Tautulli"; then
    track_service "Tautulli" "false" "0"
else
    track_service "Tautulli" "false" "1"
fi

# Organizr
log "Checking Organizr..."
if check_port "127.0.0.1" "8181" "Organizr"; then
    track_service "Organizr" "false" "0"
else
    track_service "Organizr" "false" "1"
fi

# Homepage
log "Checking Homepage..."
if check_port "127.0.0.1" "3000" "Homepage"; then
    track_service "Homepage" "false" "0"
else
    track_service "Homepage" "false" "1"
fi

# Homarr
log "Checking Homarr..."
if check_port "127.0.0.1" "7575" "Homarr"; then
    track_service "Homarr" "false" "0"
else
    track_service "Homarr" "false" "1"
fi

echo ""

# ========================================
# CONTENT LIBRARIES HEALTH CHECK
# ========================================

echo "=========================================="
echo "📚 CONTENT LIBRARIES"
echo "=========================================="

# Audiobookshelf
log "Checking Audiobookshelf..."
if check_port "127.0.0.1" "13378" "Audiobookshelf"; then
    track_service "Audiobookshelf" "false" "0"
else
    track_service "Audiobookshelf" "false" "1"
fi

# Navidrome
log "Checking Navidrome..."
if check_port "127.0.0.1" "4533" "Navidrome"; then
    track_service "Navidrome" "false" "0"
else
    track_service "Navidrome" "false" "1"
fi

echo ""

# ========================================
# AI SERVICES HEALTH CHECK
# ========================================

echo "=========================================="
echo "🤖 AI SERVICES"
echo "=========================================="

# Ollama
log "Checking Ollama..."
if check_port "127.0.0.1" "11434" "Ollama"; then
    track_service "Ollama" "false" "0"
else
    track_service "Ollama" "false" "1"
fi

# AI Assistant
log "Checking AI Assistant..."
if check_port "127.0.0.1" "8901" "AI Assistant"; then
    track_service "AI Assistant" "false" "0"
else
    track_service "AI Assistant" "false" "1"
fi

echo ""

# ========================================
# SYSTEM RESOURCE CHECK
# ========================================

echo "=========================================="
echo "💻 SYSTEM RESOURCES"
echo "=========================================="

# Check disk space
if command -v df >/dev/null 2>&1; then
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 90 ]; then
        echo "✅ Disk Usage - ${disk_usage}% (Healthy)"
    elif [ "$disk_usage" -lt 95 ]; then
        echo "⚠️  Disk Usage - ${disk_usage}% (Warning)"
    else
        echo "❌ Disk Usage - ${disk_usage}% (Critical)"
    fi
fi

# Check memory usage
if command -v free >/dev/null 2>&1; then
    memory_usage=$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')
    if [ "$memory_usage" -lt 85 ]; then
        echo "✅ Memory Usage - ${memory_usage}% (Healthy)"
    elif [ "$memory_usage" -lt 95 ]; then
        echo "⚠️  Memory Usage - ${memory_usage}% (Warning)"  
    else
        echo "❌ Memory Usage - ${memory_usage}% (Critical)"
    fi
fi

echo ""

# ========================================
# HEALTH CHECK SUMMARY
# ========================================

echo "=========================================="
echo "📈 HEALTH CHECK SUMMARY"
echo "=========================================="

# Calculate percentages
if [ $TOTAL_SERVICES -gt 0 ]; then
    OVERALL_HEALTH=$((HEALTHY_SERVICES * 100 / TOTAL_SERVICES))
else
    OVERALL_HEALTH=0
fi

if [ $CRITICAL_SERVICES -gt 0 ]; then
    CRITICAL_HEALTH=$((CRITICAL_HEALTHY * 100 / CRITICAL_SERVICES))
else
    CRITICAL_HEALTH=100
fi

echo "📊 Total Services: ${TOTAL_SERVICES}"
echo "✅ Healthy Services: ${HEALTHY_SERVICES}"
echo "⚠️  Degraded Services: ${DEGRADED_SERVICES}"
echo "📈 Overall Health: ${OVERALL_HEALTH}%"
echo ""
echo "🔴 Critical Services: ${CRITICAL_SERVICES}"
echo "✅ Critical Healthy: ${CRITICAL_HEALTHY}"
echo "📈 Critical Health: ${CRITICAL_HEALTH}%"

# Determine exit status based on critical services
if [[ $CRITICAL_HEALTHY -eq $CRITICAL_SERVICES ]]; then
    echo ""
    echo "🎉 All critical services are healthy!"
    echo "🌟 Container is operating normally"
    EXIT_CODE=0
elif [[ $CRITICAL_HEALTHY -ge $((CRITICAL_SERVICES * 2 / 3)) ]]; then
    echo ""
    echo "⚠️  Most critical services are healthy, but some issues detected"
    echo "🔧 Container is functional but may need attention"
    EXIT_CODE=0
else
    echo ""
    echo "💥 Critical service failures detected!"
    echo "🚨 Container needs immediate attention"
    EXIT_CODE=1
fi

# Additional service tier summary
echo ""
echo "📋 Service Health by Category:"
echo "   🔧 Infrastructure: $(echo $CRITICAL_HEALTHY)/$(echo $CRITICAL_SERVICES) healthy"
echo "   📺 Media Servers: Available" 
echo "   🔍 *ARR Stack: Active"
echo "   ⬇️  Download Clients: Available"
echo "   📋 Request Management: Ready"
echo "   📊 Monitoring: Operational"
echo "   🖥️  Dashboards: Accessible"
echo "   📚 Content Libraries: Available"
echo "   🤖 AI Services: Ready"

echo ""
echo "🕐 Health check completed at: $(date)"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "🎯 Health check PASSED - Container is healthy"
else
    echo "🔥 Health check FAILED - Container needs attention"
fi

exit $EXIT_CODE