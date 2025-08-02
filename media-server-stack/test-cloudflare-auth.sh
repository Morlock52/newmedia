#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Cloudflare Tunnel Status Check"
echo "================================="

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    echo "❌ .env file not found"
    exit 1
fi

echo "Domain: $DOMAIN"
echo "Tunnel Token: ${CLOUDFLARE_TUNNEL_TOKEN:0:20}..."
echo ""

# Check if cloudflared container is running
echo "1. Checking Cloudflare Tunnel Container..."
if docker ps | grep -q cloudflared; then
    echo "✅ Cloudflared container is running"
    
    # Show tunnel logs
    echo ""
    echo "Recent tunnel logs:"
    echo "==================="
    docker logs cloudflared --tail 10
    echo ""
    
    # Check if tunnel is connected
    if docker logs cloudflared --tail 20 | grep -q "Registered tunnel connection"; then
        echo "✅ Tunnel connection established"
    else
        echo "⚠️  Tunnel may not be fully connected"
    fi
else
    echo "❌ Cloudflared container not running"
    echo "Starting cloudflared..."
    docker-compose up -d cloudflared
    sleep 10
fi

echo ""
echo "2. Testing Internal Service Connectivity..."

services=("jellyfin:8096" "sonarr:8989" "radarr:7878" "prowlarr:9696" "overseerr:5055" "traefik:8080")

for service_port in "${services[@]}"; do
    service="${service_port%:*}"
    port="${service_port#*:}"
    
    echo -n "Testing $service (port $port)... "
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -q "200\|302\|401"; then
        echo "✅ OK"
    else
        echo "❌ Not responding"
    fi
done

echo ""
echo "3. Testing External DNS Resolution..."

for service in jellyfin sonarr radarr prowlarr overseerr traefik; do
    echo -n "Testing $service.$DOMAIN DNS... "
    
    if nslookup "$service.$DOMAIN" >/dev/null 2>&1; then
        echo "✅ Resolves"
    else
        echo "❌ No DNS record"
    fi
done

echo ""
echo "4. Quick External Access Test..."
echo "Opening Jellyfin in browser for testing..."

# Open Jellyfin for testing
osascript -e "tell application \"Google Chrome\" to open location \"https://jellyfin.$DOMAIN\"" &

echo ""
echo "📋 Expected Test Result:"
echo "1. Browser opens https://jellyfin.$DOMAIN"
echo "2. Shows Cloudflare Access login page"
echo "3. Enter email: $EMAIL" 
echo "4. Receive PIN code via email"
echo "5. Enter PIN to access Jellyfin"
echo ""

read -p "Did the authentication flow work correctly? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎉 Cloudflare authentication is working!"
    echo ""
    echo "Your services are now securely accessible at:"
    echo "• https://jellyfin.$DOMAIN"
    echo "• https://sonarr.$DOMAIN"
    echo "• https://radarr.$DOMAIN" 
    echo "• https://prowlarr.$DOMAIN"
    echo "• https://overseerr.$DOMAIN"
else
    echo "🔧 Troubleshooting needed. Common issues:"
    echo ""
    echo "1. Tunnel not connected properly:"
    echo "   docker logs cloudflared"
    echo ""
    echo "2. Access application not configured:"
    echo "   Check Zero Trust → Access → Applications"
    echo ""
    echo "3. DNS records missing:"
    echo "   Check Cloudflare DNS for CNAME records"
    echo ""
    echo "4. Services not running:"
    echo "   docker-compose ps"
    echo ""
    echo "5. Wrong email in policy:"
    echo "   Verify email matches: $EMAIL"
fi

echo ""
echo "🔗 Useful Links:"
echo "• Zero Trust Dashboard: https://one.dash.cloudflare.com/"
echo "• Cloudflare DNS: https://dash.cloudflare.com/$DOMAIN/dns"
echo "• Tunnel Management: https://one.dash.cloudflare.com/ → Networks → Tunnels"
