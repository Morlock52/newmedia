#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Cloudflare Tunnel Error 1033 - Diagnostic & Fix"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    echo -e "${RED}❌ .env file not found${NC}"
    exit 1
fi

echo -e "${BLUE}Domain:${NC} $DOMAIN"
echo -e "${BLUE}Current Tunnel Token:${NC} ${CLOUDFLARE_TUNNEL_TOKEN:0:30}..."
echo ""

echo -e "${GREEN}Step 1: Check Current Tunnel Status${NC}"
echo "===================================="

# Check if cloudflared container is running
if docker ps | grep -q cloudflared; then
    echo "✅ Cloudflared container is running"
    
    echo ""
    echo "Recent tunnel logs:"
    echo "=================="
    docker logs cloudflared --tail 20
    
    echo ""
    echo -e "${YELLOW}Analyzing logs...${NC}"
    
    if docker logs cloudflared --tail 20 | grep -q "error"; then
        echo "❌ Errors found in tunnel logs"
        echo ""
        echo "Common error patterns:"
        docker logs cloudflared --tail 20 | grep -i "error\|failed\|denied" || echo "No specific error patterns found"
    fi
    
    if docker logs cloudflared --tail 20 | grep -q "expired\|invalid"; then
        echo "🔑 Token appears to be expired or invalid"
    fi
    
else
    echo "❌ Cloudflared container not running"
    echo "Starting cloudflared container..."
    docker-compose up -d cloudflared
    sleep 10
    echo ""
    echo "New container logs:"
    docker logs cloudflared --tail 10
fi

echo ""
echo -e "${GREEN}Step 2: Test Tunnel Connectivity${NC}"
echo "=================================="

# Try to reach Cloudflare's API
echo -n "Cloudflare API connectivity: "
if curl -s --connect-timeout 5 https://api.cloudflare.com/client/v4/user/tokens/verify >/dev/null; then
    echo "✅ Can reach Cloudflare API"
else
    echo "❌ Cannot reach Cloudflare API"
fi

# Check if tunnel is responding
echo -n "Tunnel health endpoint: "
if curl -s --connect-timeout 5 http://localhost:8080 >/dev/null 2>&1; then
    echo "✅ Local services responding"
else
    echo "⚠️  Local services may not be fully ready"
fi

echo ""
echo -e "${GREEN}Step 3: Tunnel Token Analysis${NC}"
echo "================================="

# Decode the tunnel token (it's base64 JSON)
echo "Decoding tunnel token..."
if command -v base64 >/dev/null && command -v jq >/dev/null; then
    token_data=$(echo "$CLOUDFLARE_TUNNEL_TOKEN" | base64 -d 2>/dev/null | jq . 2>/dev/null || echo "Failed to decode")
    
    if [[ "$token_data" != "Failed to decode" ]]; then
        echo "✅ Token structure is valid"
        echo "Token details:"
        echo "$token_data" | jq -r '"Account ID: " + .a'
        echo "$token_data" | jq -r '"Tunnel ID: " + .t'
    else
        echo "❌ Token appears to be malformed"
    fi
else
    echo "⚠️  Cannot decode token (missing base64 or jq)"
fi

echo ""
echo -e "${GREEN}Step 4: Fix Recommendations${NC}"
echo "============================"

echo ""
echo "🔧 Immediate Fixes to Try:"
echo "=========================="

echo ""
echo "1️⃣  Restart Tunnel with Fresh Connection:"
echo "   docker-compose restart cloudflared"
echo "   docker logs cloudflared --tail 10"

echo ""
echo "2️⃣  Clear and Restart Everything:"
echo "   docker-compose down"
echo "   docker-compose up -d"

echo ""
echo "3️⃣  Get New Tunnel Token (if current is expired):"
echo "   • Go to: https://one.dash.cloudflare.com/"
echo "   • Navigate to: Networks → Tunnels"
echo "   • Find: morloksmaze-media-tunnel"
echo "   • Click: Configure → Copy token"
echo "   • Update CLOUDFLARE_TUNNEL_TOKEN in .env"

echo ""
echo "4️⃣  Create Brand New Tunnel:"
echo "   • Delete existing tunnel in Cloudflare dashboard"
echo "   • Create new tunnel with same name"
echo "   • Update token in .env file"
echo "   • Reconfigure all public hostnames"

echo ""
echo -e "${BLUE}🚀 Quick Fix Commands:${NC}"
echo "====================="

echo ""
echo "# Try restarting tunnel first:"
echo "docker-compose restart cloudflared && sleep 10 && docker logs cloudflared --tail 10"

echo ""
echo "# If that doesn't work, full restart:"
echo "docker-compose down && docker-compose up -d cloudflared"

echo ""
echo "# Check if tunnel connects:"
echo "docker logs cloudflared --follow"

echo ""
echo -e "${YELLOW}⚠️  If the tunnel keeps failing:${NC}"
echo "================================="

echo ""
echo "The token might be expired. Here's how to get a new one:"
echo ""
echo "1. Open Cloudflare Zero Trust: https://one.dash.cloudflare.com/"
echo "2. Go to Networks → Tunnels"
echo "3. Find 'morloksmaze-media-tunnel'"
echo "4. Click 'Configure'"
echo "5. Copy the tunnel token"
echo "6. Update your .env file:"
echo "   CLOUDFLARE_TUNNEL_TOKEN=your-new-token-here"
echo "7. Restart: docker-compose restart cloudflared"

echo ""
echo -e "${GREEN}🧪 Test After Fix:${NC}"
echo "=================="
echo ""
echo "1. Check tunnel logs: docker logs cloudflared"
echo "2. Test a service: curl -I https://jellyfin.$DOMAIN"
echo "3. Open in browser: https://jellyfin.$DOMAIN"
echo ""

echo "💡 The tunnel should show 'Registered tunnel connection' when working"

echo ""
read -p "Would you like me to restart the tunnel now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Restarting Cloudflare Tunnel..."
    docker-compose restart cloudflared
    
    echo "Waiting for tunnel to connect..."
    sleep 15
    
    echo ""
    echo "New tunnel status:"
    docker logs cloudflared --tail 15
    
    echo ""
    if docker logs cloudflared --tail 10 | grep -q "Registered tunnel connection"; then
        echo -e "${GREEN}✅ Tunnel appears to be connected!${NC}"
        echo ""
        echo "Test your services:"
        echo "https://jellyfin.$DOMAIN"
    else
        echo -e "${RED}❌ Tunnel still not connecting${NC}"
        echo ""
        echo "You likely need to get a new tunnel token from Cloudflare."
        echo "Follow the steps above to get a fresh token."
    fi
fi
