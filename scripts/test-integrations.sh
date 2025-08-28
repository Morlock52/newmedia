#!/bin/bash

# Ultimate Media Server 2025 - Integration Test Suite
# This script tests all service integrations and validates the user flow

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

# Header
echo "========================================"
echo "Ultimate Media Server 2025"
echo "Integration Test Suite"
echo "========================================"
echo

# Test 1: Docker Services Health
log_test "Checking Docker services health..."
SERVICES=(
    "jellyfin:8096"
    "sonarr:8989"
    "radarr:7878"
    "prowlarr:9696"
    "qbittorrent:8080"
    "homepage:3001"
    "grafana:3000"
    "caddy:80"
)

for service in "${SERVICES[@]}"; do
    name="${service%%:*}"
    port="${service##*:}"
    
    if docker ps | grep -q "$name"; then
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -qE "200|301|302"; then
            log_pass "$name is running and responding on port $port"
        else
            log_fail "$name is running but not responding on port $port"
        fi
    else
        log_fail "$name container is not running"
    fi
done

echo

# Test 2: API Connectivity
log_test "Testing API endpoints..."

# Jellyfin API
if curl -s "http://localhost:8096/System/Info/Public" | grep -q "ServerName"; then
    log_pass "Jellyfin API is accessible"
else
    log_fail "Jellyfin API is not accessible"
fi

# Sonarr API
if curl -s "http://localhost:8989/api/v3/system/status" -H "X-Api-Key: test" | grep -q "error"; then
    log_warn "Sonarr API requires valid API key (expected)"
else
    log_fail "Sonarr API is not responding"
fi

echo

# Test 3: Service Integration Flow
log_test "Testing service integration flow..."

# Check if Prowlarr can reach Sonarr/Radarr
PROWLARR_UP=$(docker exec prowlarr wget -q -O- http://sonarr:8989/ping 2>/dev/null || echo "FAIL")
if [[ "$PROWLARR_UP" != "FAIL" ]]; then
    log_pass "Prowlarr can communicate with Sonarr"
else
    log_fail "Prowlarr cannot reach Sonarr (network isolation issue)"
fi

# Check if qBittorrent is accessible
QB_UP=$(docker exec sonarr wget -q -O- http://qbittorrent:8080 2>/dev/null || echo "FAIL")
if [[ "$QB_UP" != "FAIL" ]]; then
    log_pass "Sonarr can communicate with qBittorrent"
else
    log_warn "Sonarr cannot reach qBittorrent (may be behind VPN)"
fi

echo

# Test 4: Media Paths
log_test "Checking media path configurations..."

PATHS=(
    "./media-data/movies"
    "./media-data/tv"
    "./media-data/music"
    "./downloads/complete"
    "./config"
)

for path in "${PATHS[@]}"; do
    if [[ -d "$path" ]]; then
        log_pass "Path exists: $path"
    else
        log_fail "Path missing: $path"
    fi
done

echo

# Test 5: Performance Metrics
log_test "Testing performance endpoints..."

# Homepage dashboard
START_TIME=$(date +%s%N)
curl -s -o /dev/null "http://localhost:3001"
END_TIME=$(date +%s%N)
LOAD_TIME=$(( (END_TIME - START_TIME) / 1000000 ))

if [[ $LOAD_TIME -lt 1000 ]]; then
    log_pass "Homepage loads in ${LOAD_TIME}ms (target: <1000ms)"
else
    log_warn "Homepage loads in ${LOAD_TIME}ms (target: <1000ms)"
fi

echo

# Test 6: SSL/TLS Configuration
log_test "Checking SSL/TLS configuration..."

if curl -s -k "https://localhost" > /dev/null 2>&1; then
    log_pass "HTTPS is configured on Caddy"
else
    log_warn "HTTPS is not configured (using HTTP)"
fi

echo

# Test 7: Database Connectivity
log_test "Checking database connections..."

# Check if Redis is running
if docker ps | grep -q "redis"; then
    if docker exec redis redis-cli ping | grep -q "PONG"; then
        log_pass "Redis is running and responsive"
    else
        log_fail "Redis is running but not responsive"
    fi
else
    log_warn "Redis container not found (optional service)"
fi

echo

# Test 8: Mobile PWA Features
log_test "Checking PWA configuration..."

# Check for manifest.json
if [[ -f "./homepage-config/manifest.json" ]]; then
    log_pass "PWA manifest found"
else
    log_warn "PWA manifest missing - mobile install won't work"
fi

# Check for service worker
if [[ -f "./homepage-config/sw.js" ]]; then
    log_pass "Service worker found"
else
    log_warn "Service worker missing - offline mode won't work"
fi

echo

# Test 9: Monitoring Stack
log_test "Checking monitoring services..."

# Prometheus
if curl -s "http://localhost:9090/-/healthy" | grep -q "Prometheus is Healthy"; then
    log_pass "Prometheus is healthy"
else
    log_warn "Prometheus health check failed"
fi

# Grafana
if curl -s "http://localhost:3000/api/health" | grep -q "ok"; then
    log_pass "Grafana is healthy"
else
    log_warn "Grafana health check failed"
fi

echo

# Test 10: Security Headers
log_test "Checking security headers..."

HEADERS=$(curl -s -I "http://localhost:3001")

if echo "$HEADERS" | grep -q "X-Frame-Options"; then
    log_pass "X-Frame-Options header present"
else
    log_warn "X-Frame-Options header missing"
fi

if echo "$HEADERS" | grep -q "X-Content-Type-Options"; then
    log_pass "X-Content-Type-Options header present"
else
    log_warn "X-Content-Type-Options header missing"
fi

echo
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${RED}Failed:${NC} $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All critical tests passed! 🎉${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the logs above.${NC}"
    exit 1
fi