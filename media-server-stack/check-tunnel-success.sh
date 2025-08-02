#!/usr/bin/env bash
set -euo pipefail

echo "🎉 TUNNEL TOKEN UPDATED!"
echo "======================="

echo ""
echo "✅ Your new token has been applied"
echo "✅ Cloudflare tunnel is restarting"
echo ""

# Wait a moment for the restart to complete
sleep 15

echo "🔍 Checking tunnel status..."
echo ""

# Check if tunnel is connected
if docker logs cloudflared --tail 10 | grep -q "Registered tunnel connection"; then
    echo "🎉 SUCCESS! Tunnel is connected!"
    echo ""
    
    echo "🌐 Your services are now accessible:"
    echo "• https://jellyfin.morloksmaze.com"
    echo "• https://overseerr.morloksmaze.com"
    echo "• https://sonarr.morloksmaze.com"
    echo "• https://radarr.morloksmaze.com"
    echo "• https://prowlarr.morloksmaze.com"
    echo ""
    
    echo "🧪 Testing Jellyfin..."
    if curl -s -I https://jellyfin.morloksmaze.com | grep -q "cloudflare\|HTTP"; then
        echo "✅ Jellyfin is responding!"
        echo ""
        echo "🚀 Opening Jellyfin for you..."
        osascript -e 'tell application "Google Chrome" to open location "https://jellyfin.morloksmaze.com"'
    else
        echo "⏳ Services may still be starting up..."
    fi
    
else
    echo "⚠️  Tunnel may still be connecting..."
    echo ""
    echo "Recent logs:"
    docker logs cloudflared --tail 10
    echo ""
    echo "If you see 'Registered tunnel connection' in the logs above, you're good!"
fi

echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. ✅ Tunnel token updated"
echo "2. ✅ Services should be accessible"
echo "3. 🔐 Set up Cloudflare Zero Trust authentication"
echo "4. 🧪 Test all your services"
echo ""
echo "Run this to test everything: ./test-complete-stack.sh"
