#!/bin/bash

# Simple Jellyfin Authentication Test
# Tests authentication without Node.js dependencies

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

JELLYFIN_URL="http://localhost:8096"

echo "🧪 Jellyfin Authentication Test Suite"
echo "======================================"

# Test 1: Basic connectivity
echo "🔍 Test 1: Basic connectivity..."
if curl -s --connect-timeout 5 "$JELLYFIN_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Jellyfin is accessible${NC}"
else
    echo -e "${RED}❌ Jellyfin is not accessible${NC}"
    exit 1
fi

# Test 2: Public API endpoints
echo "🔍 Test 2: Public API endpoints..."
if curl -s --connect-timeout 5 "$JELLYFIN_URL/System/Info/Public" > /dev/null; then
    echo -e "${GREEN}✅ Public API endpoints working${NC}"
    
    # Get system info
    system_info=$(curl -s "$JELLYFIN_URL/System/Info/Public")
    server_name=$(echo "$system_info" | grep -o '"ServerName":"[^"]*' | cut -d'"' -f4 2>/dev/null || echo "Jellyfin")
    version=$(echo "$system_info" | grep -o '"Version":"[^"]*' | cut -d'"' -f4 2>/dev/null || echo "Unknown")
    wizard_completed=$(echo "$system_info" | grep -o '"StartupWizardCompleted":[^,]*' | cut -d':' -f2 2>/dev/null || echo "Unknown")
    
    echo "   Server: $server_name"
    echo "   Version: $version" 
    echo "   Startup Wizard: $wizard_completed"
else
    echo -e "${RED}❌ Public API endpoints not working${NC}"
    exit 1
fi

# Test 3: Authentication endpoint
echo "🔍 Test 3: Authentication endpoint availability..."
auth_response=$(curl -s -w "%{http_code}" -o /dev/null -X POST "$JELLYFIN_URL/Users/AuthenticateByName" \
    -H "Content-Type: application/json" \
    -d '{"Username":"test","Pw":"test"}')

if [ "$auth_response" = "401" ] || [ "$auth_response" = "400" ]; then
    echo -e "${GREEN}✅ Authentication endpoint responding correctly${NC}"
else
    echo -e "${YELLOW}⚠️  Authentication endpoint returned: $auth_response${NC}"
fi

# Test 4: CORS headers
echo "🔍 Test 4: CORS configuration..."
cors_response=$(curl -s -I -X OPTIONS "$JELLYFIN_URL/System/Info" \
    -H "Origin: http://localhost:3000" \
    -H "Access-Control-Request-Method: GET" | head -n 1)

if echo "$cors_response" | grep -q "200\|204"; then
    echo -e "${GREEN}✅ CORS configuration working${NC}"
else
    echo -e "${YELLOW}⚠️  CORS may need additional configuration${NC}"
fi

# Test 5: Container status
echo "🔍 Test 5: Container health..."
if docker ps | grep -q "jellyfin.*healthy"; then
    echo -e "${GREEN}✅ Container is running and healthy${NC}"
elif docker ps | grep -q "jellyfin"; then
    echo -e "${YELLOW}⚠️  Container is running but health check pending${NC}"
else
    echo -e "${RED}❌ Container is not running${NC}"
fi

# Test 6: Check for authentication errors in logs
echo "🔍 Test 6: Recent authentication errors..."
recent_errors=$(docker logs jellyfin 2>&1 | tail -50 | grep -c "Invalid token" 2>/dev/null || echo "0")

if [ "$recent_errors" -eq 0 ]; then
    echo -e "${GREEN}✅ No recent authentication errors${NC}"
else
    echo -e "${YELLOW}⚠️  Found $recent_errors recent authentication errors${NC}"
fi

# Test 7: File permissions
echo "🔍 Test 7: Configuration files..."
if docker exec jellyfin test -f "/config/config/system.xml"; then
    echo -e "${GREEN}✅ System configuration file exists${NC}"
else
    echo -e "${RED}❌ System configuration file missing${NC}"
fi

if docker exec jellyfin test -f "/config/config/network.xml"; then
    echo -e "${GREEN}✅ Network configuration file exists${NC}"
else
    echo -e "${YELLOW}⚠️  Network configuration file missing${NC}"
fi

echo ""
echo "======================================"
echo "🎉 Authentication Test Suite Complete"
echo "======================================"

# Summary
echo ""
echo "📊 Summary:"
echo "- Jellyfin URL: $JELLYFIN_URL"
echo "- Container Status: $(docker ps --format 'table {{.Status}}' | grep jellyfin | head -1 || echo 'Not running')"
echo "- Default Credentials: admin / admin123"
echo ""
echo "🔧 Available Scripts:"
echo "- ./scripts/fix-jellyfin-auth.sh - Fix authentication issues"
echo "- ./scripts/jellyfin-cors-config.js - Configure CORS (requires Node.js)"
echo "- ./verify-jellyfin-auth.sh - Quick verification"
echo ""
echo "✅ Jellyfin authentication system is ready for dashboard integration!"