#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Update Cloudflare Tunnel Token"
echo "================================="

echo "Backing up current .env file..."
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

echo ""
echo "Current tunnel token (first 50 chars):"
grep "CLOUDFLARE_TUNNEL_TOKEN" .env | cut -c1-70

echo ""
echo "Please paste your new tunnel token from Cloudflare dashboard:"
echo "(It should start with 'eyJ' and be quite long)"
echo ""
read -p "New tunnel token: " new_token

if [[ -n "$new_token" && "$new_token" =~ ^eyJ ]]; then
    # Update the token in .env file
    sed -i.bak "s/CLOUDFLARE_TUNNEL_TOKEN=.*/CLOUDFLARE_TUNNEL_TOKEN=$new_token/" .env
    
    echo ""
    echo "✅ Token updated successfully!"
    echo ""
    echo "New token (first 50 chars):"
    echo "${new_token:0:50}..."
    
    echo ""
    echo "🔄 Restarting Cloudflare Tunnel..."
    docker-compose restart cloudflared
    
    echo ""
    echo "⏳ Waiting for tunnel to connect..."
    sleep 10
    
    echo ""
    echo "📋 Tunnel logs:"
    echo "=============="
    docker logs cloudflared --tail 15
    
    echo ""
    if docker logs cloudflared --tail 10 | grep -q "Registered tunnel connection"; then
        echo "🎉 SUCCESS! Tunnel is connected!"
        echo ""
        echo "Your services should now be accessible:"
        echo "• https://jellyfin.morloksmaze.com"
        echo "• https://overseerr.morloksmaze.com"
        echo "• https://sonarr.morloksmaze.com"
        echo "• https://radarr.morloksmaze.com"
        echo ""
        echo "Test one now:"
        read -p "Open Jellyfin for testing? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            osascript -e 'tell application "Google Chrome" to open location "https://jellyfin.morloksmaze.com"'
        fi
    else
        echo "⚠️  Tunnel may still be connecting. Check logs again in a moment:"
        echo "docker logs cloudflared --tail 10"
    fi
    
else
    echo "❌ Invalid token format. Token should start with 'eyJ'"
    echo "Please run this script again with the correct token."
fi
