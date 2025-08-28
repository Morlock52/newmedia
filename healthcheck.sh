#!/bin/bash
# Comprehensive health check for single-container media server
# Checks all critical services and their interdependencies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Health check configuration
TIMEOUT=3
TOTAL_SERVICES=0
HEALTHY_SERVICES=0
CRITICAL_SERVICES=0
CRITICAL_HEALTHY=0

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
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

# Check HTTP endpoint
check_http() {
    local url="$1"
    local service="$2"
    local expected_code="${3:-200}"
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout ${TIMEOUT} "$url" 2>/dev/null || echo "000")
    
    if [[ "$response" == "$expected_code" ]]; then
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
    
    local response=$(curl -s --connect-timeout ${TIMEOUT} "$url" 2>/dev/null || echo "ERROR")
    
    if [[ -n "$expected_response" && "$response" == *"$expected_response"* ]]; then
        return 0
    elif [[ -z "$expected_response" && "$response" != "ERROR" ]]; then
        return 0
    else
        log "❌ ${service} API (${url}) - Unexpected response: ${response:0:50}"
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

log "🏥 Starting comprehensive health check..."

echo "=========================================="
echo "🔍 INFRASTRUCTURE LAYER HEALTH CHECK"
echo "=========================================="

# PostgreSQL
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "5432" "PostgreSQL" && \
   check_database "postgres" "pg_isready -h 127.0.0.1 -p 5432 -q" "PostgreSQL"; then
    log "✅ PostgreSQL - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

# Redis
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "6379" "Redis" && \
   check_database "redis" "redis-cli -h 127.0.0.1 -p 6379 ping | grep -q PONG" "Redis"; then
    log "✅ Redis - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

# RabbitMQ
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "5672" "RabbitMQ" && check_port "127.0.0.1" "15672" "RabbitMQ Management"; then
    log "✅ RabbitMQ - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

# Elasticsearch
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "9200" "Elasticsearch" && \
   check_http "http://127.0.0.1:9200/_cluster/health" "Elasticsearch"; then
    log "✅ Elasticsearch - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "🌐 PLATFORM LAYER HEALTH CHECK"
echo "=========================================="

# Traefik
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "80" "Traefik HTTP" && \
   check_port "127.0.0.1" "443" "Traefik HTTPS" && \
   check_http "http://127.0.0.1:8080/ping" "Traefik API" 200; then
    log "✅ Traefik - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

# Auth Service
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "8000" "Auth Service" && \
   check_http "http://127.0.0.1:8000/health" "Auth Service" 200; then
    log "✅ Auth Service - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

echo ""
echo "=========================================="
echo "📺 MEDIA SERVERS HEALTH CHECK"
echo "=========================================="

# Jellyfin
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "8096" "Jellyfin" && \
   check_http "http://127.0.0.1:8096/health" "Jellyfin" 200; then
    log "✅ Jellyfin - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

# Plex
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
if check_port "127.0.0.1" "32400" "Plex" && \
   check_http "http://127.0.0.1:32400/identity" "Plex" 200; then
    log "✅ Plex - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    CRITICAL_HEALTHY=$((CRITICAL_HEALTHY + 1))
fi

# Emby
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8097" "Emby" && \
   check_http "http://127.0.0.1:8097/health" "Emby" 200; then
    log "✅ Emby - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "🔍 *ARR STACK HEALTH CHECK"
echo "=========================================="

# Sonarr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8989" "Sonarr" && \
   check_api "http://127.0.0.1:8989/ping" "Sonarr" "pong"; then
    log "✅ Sonarr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Radarr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "7878" "Radarr" && \
   check_api "http://127.0.0.1:7878/ping" "Radarr" "pong"; then
    log "✅ Radarr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Lidarr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8686" "Lidarr" && \
   check_api "http://127.0.0.1:8686/ping" "Lidarr" "pong"; then
    log "✅ Lidarr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Readarr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8787" "Readarr" && \
   check_api "http://127.0.0.1:8787/ping" "Readarr" "pong"; then
    log "✅ Readarr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Prowlarr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "9696" "Prowlarr" && \
   check_api "http://127.0.0.1:9696/ping" "Prowlarr" "pong"; then
    log "✅ Prowlarr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Bazarr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "6767" "Bazarr" && \
   check_http "http://127.0.0.1:6767" "Bazarr" 200; then
    log "✅ Bazarr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "⬇️  DOWNLOAD CLIENTS HEALTH CHECK"
echo "=========================================="

# qBittorrent
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8080" "qBittorrent" && \
   check_http "http://127.0.0.1:8080/api/v2/app/version" "qBittorrent API" 200; then
    log "✅ qBittorrent - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Transmission
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "9091" "Transmission" && \
   check_http "http://127.0.0.1:9091/transmission/rpc" "Transmission" 409; then # 409 is expected for RPC
    log "✅ Transmission - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# SABnzbd
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8081" "SABnzbd" && \
   check_http "http://127.0.0.1:8081/api" "SABnzbd" 200; then
    log "✅ SABnzbd - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# NZBGet
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "6789" "NZBGet" && \
   check_http "http://127.0.0.1:6789" "NZBGet" 200; then
    log "✅ NZBGet - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "📋 REQUEST SERVICES HEALTH CHECK"
echo "=========================================="

# Overseerr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "5055" "Overseerr" && \
   check_http "http://127.0.0.1:5055/api/v1/status" "Overseerr" 200; then
    log "✅ Overseerr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Jellyseerr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "5056" "Jellyseerr" && \
   check_http "http://127.0.0.1:5056/api/v1/status" "Jellyseerr" 200; then
    log "✅ Jellyseerr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Ombi
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "3579" "Ombi" && \
   check_http "http://127.0.0.1:3579/api/v1/Status" "Ombi" 200; then
    log "✅ Ombi - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "🤖 AI SERVICES HEALTH CHECK"
echo "=========================================="

# AI Safety System
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8901" "AI Safety System" && \
   check_http "http://127.0.0.1:8901/v1/health" "AI Safety System" 200; then
    log "✅ AI Safety System - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Content Moderation
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8902" "Content Moderation" && \
   check_http "http://127.0.0.1:8902/v1/health" "Content Moderation" 200; then
    log "✅ Content Moderation - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Recommendation Engine
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8903" "Recommendation Engine" && \
   check_http "http://127.0.0.1:8903/v1/health" "Recommendation Engine" 200; then
    log "✅ Recommendation Engine - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Social Media Integration
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8904" "Social Media Integration" && \
   check_http "http://127.0.0.1:8904/v1/health" "Social Media Integration" 200; then
    log "✅ Social Media Integration - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "🔧 MANAGEMENT TOOLS HEALTH CHECK"  
echo "=========================================="

# Tautulli
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8181" "Tautulli" && \
   check_http "http://127.0.0.1:8181/api/v2" "Tautulli" 200; then
    log "✅ Tautulli - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Organizr
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8083" "Organizr" && \
   check_http "http://127.0.0.1:8083" "Organizr" 200; then
    log "✅ Organizr - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Heimdall
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "8082" "Heimdall" && \
   check_http "http://127.0.0.1:8082" "Heimdall" 200; then
    log "✅ Heimdall - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "📊 MONITORING SERVICES HEALTH CHECK"
echo "=========================================="

# Prometheus
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "9090" "Prometheus" && \
   check_http "http://127.0.0.1:9090/-/healthy" "Prometheus" 200; then
    log "✅ Prometheus - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Grafana
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "3000" "Grafana" && \
   check_http "http://127.0.0.1:3000/api/health" "Grafana" 200; then
    log "✅ Grafana - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

# Uptime Kuma
TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
if check_port "127.0.0.1" "3001" "Uptime Kuma" && \
   check_http "http://127.0.0.1:3001" "Uptime Kuma" 200; then
    log "✅ Uptime Kuma - Healthy"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
fi

echo ""
echo "=========================================="
echo "📈 HEALTH CHECK SUMMARY"
echo "=========================================="

# Calculate percentages
OVERALL_HEALTH=$((HEALTHY_SERVICES * 100 / TOTAL_SERVICES))
CRITICAL_HEALTH=$((CRITICAL_HEALTHY * 100 / CRITICAL_SERVICES))

echo "Total Services: ${TOTAL_SERVICES}"
echo "Healthy Services: ${HEALTHY_SERVICES}"
echo "Overall Health: ${OVERALL_HEALTH}%"
echo ""
echo "Critical Services: ${CRITICAL_SERVICES}"
echo "Critical Healthy: ${CRITICAL_HEALTHY}"
echo "Critical Health: ${CRITICAL_HEALTH}%"

# Determine exit status based on critical services
if [[ $CRITICAL_HEALTHY -eq $CRITICAL_SERVICES ]]; then
    log "🎉 All critical services are healthy!"
    EXIT_CODE=0
elif [[ $CRITICAL_HEALTHY -ge $((CRITICAL_SERVICES * 2 / 3)) ]]; then
    log "⚠️  Most critical services are healthy, but some issues detected"
    EXIT_CODE=0  # Still considered healthy for container orchestration
else
    log "💥 Critical service failures detected! Container needs attention"
    EXIT_CODE=1
fi

# Additional checks for service interdependencies
echo ""
echo "=========================================="
echo "🔗 DEPENDENCY VALIDATION"
echo "=========================================="

# Check if media servers can communicate with databases
if [[ $CRITICAL_HEALTHY -eq $CRITICAL_SERVICES ]]; then
    log "✅ All critical infrastructure available for dependent services"
else
    log "❌ Infrastructure issues may affect dependent services"
    EXIT_CODE=1
fi

# Summary by tier
echo ""
log "📊 Service Health by Tier:"
log "   Infrastructure: $(echo "PostgreSQL Redis RabbitMQ" | wc -w) services"
log "   Platform: $(echo "Traefik Auth" | wc -w) services"
log "   Media Servers: $(echo "Jellyfin Plex Emby" | wc -w) services"
log "   Management: $(echo "Sonarr Radarr Lidarr Readarr Prowlarr Bazarr" | wc -w) services"
log "   Downloads: $(echo "qBittorrent Transmission SABnzbd NZBGet" | wc -w) services"
log "   Requests: $(echo "Overseerr Jellyseerr Ombi" | wc -w) services"
log "   AI Services: $(echo "AI-Safety Content-Mod Recommendations Social" | wc -w) services"
log "   Tools: $(echo "Tautulli Organizr Heimdall" | wc-w) services"
log "   Monitoring: $(echo "Prometheus Grafana Uptime-Kuma" | wc -w) services"

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    log "🎯 Health check PASSED - Container is healthy"
else
    log "🔥 Health check FAILED - Container needs attention"
fi

exit $EXIT_CODE