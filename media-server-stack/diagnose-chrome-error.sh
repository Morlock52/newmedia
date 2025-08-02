#!/usr/bin/env bash
set -euo pipefail

echo "🔍 TUNNEL DIAGNOSTIC - Chrome Error Check"
echo "=========================================="

# Load environment
source .env

echo "Domain: $DOMAIN"
echo "Current time: $(date)"
echo ""

echo "1. 🐳 Container Status:"
echo "======================"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(cloudflared|traefik|jellyfin)" || echo "No matching containers found"

echo ""
echo "2. 🔗 Tunnel Connection:"
echo "======================="
if docker ps | grep -q cloudflared; then
    echo "✅ Cloudflared container is running"
    
    echo ""
    echo "Recent tunnel logs:"
    docker logs cloudflared --tail 15
    
    echo ""
    if docker logs cloudflared --tail 10 | grep -q "Registered tunnel connection"; then
        echo "✅ Tunnel shows successful connection"
    elif docker logs cloudflared --tail 10 | grep -i "error\|failed"; then
        echo "❌ Tunnel has errors"
    else
        echo "⏳ Tunnel status unclear"
    fi
else
    echo "❌ Cloudflared container not running"
fi

echo ""
echo "3. 🌐 DNS & Connectivity:"
echo "========================"

echo -n "Internet connectivity: "
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo "✅ Online"
else
    echo "❌ Offline"
fi

echo -n "Domain resolution: "
if nslookup jellyfin.$DOMAIN >/dev/null 2>&1; then
    echo "✅ jellyfin.$DOMAIN resolves"
else
    echo "❌ DNS issue"
fi

echo ""
echo "4. 🏥 Local Service Health:"
echo "=========================="

services=("jellyfin:8096" "traefik:8080" "sonarr:8989" "radarr:7878")

for service_port in "${services[@]}"; do
    service="${service_port%:*}"
    port="${service_port#*:}"
    
    echo -n "$service (port $port): "
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -q "200\|302\|401"; then
        echo "✅ Responding"
    else
        echo "❌ Not responding"
    fi
done

echo ""
echo "5. 🧪 External Access Test:"
echo "=========================="

echo "Testing https://jellyfin.$DOMAIN..."

response=$(curl -s -I "https://jellyfin.$DOMAIN" 2>&1 || echo "FAILED")

if echo "$response" | grep -q "HTTP/[12]"; then
    echo "✅ Getting HTTP response"
    echo "Response headers:"
    echo "$response" | head -5
elif echo "$response" | grep -qi "cloudflare"; then
    echo "🔐 Cloudflare is responding (may need authentication)"
elif echo "$response" | grep -qi "timeout\|failed"; then
    echo "❌ Connection timeout/failed"
else
    echo "❓ Unclear response:"
    echo "$response" | head -3
fi

echo ""
echo "6. 🎯 Common Issues & Solutions:"
echo "==============================="

# Check for common issues
if ! docker ps | grep -q cloudflared; then
    echo "❌ Tunnel container not running"
    echo "   Fix: docker-compose up -d cloudflared"
fi

if ! docker logs cloudflared --tail 10 | grep -q "Registered tunnel connection"; then
    echo "❌ Tunnel not properly connected"
    echo "   Fix: Check tunnel token or restart tunnel"
fi

if ! curl -s -o /dev/null -w "%{http_code}" "http://localhost:8096" | grep -q "200\|302"; then
    echo "❌ Jellyfin not responding locally"
    echo "   Fix: docker-compose restart jellyfin"
fi

echo ""
echo "🆘 Tell me what error you see in Chrome and I'll help fix it!"
