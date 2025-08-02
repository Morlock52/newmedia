#!/usr/bin/env bash

echo "🔧 SIMPLE TUNNEL TOKEN UPDATE"
echo "============================="
echo ""
echo "Step 1: Get the token from Cloudflare"
echo "Step 2: Paste it below"
echo "Step 3: I'll fix everything!"
echo ""

read -p "Paste your tunnel token here: " token

if [[ -n "$token" ]]; then
    echo ""
    echo "✅ Got token! Updating your system..."
    
    # Backup .env
    cp .env .env.backup.$(date +%s)
    
    # Update token
    sed -i.bak "s/CLOUDFLARE_TUNNEL_TOKEN=.*/CLOUDFLARE_TUNNEL_TOKEN=$token/" .env
    
    echo "✅ Updated .env file"
    echo "✅ Restarting tunnel..."
    
    docker-compose restart cloudflared
    
    echo "⏳ Waiting for connection..."
    sleep 10
    
    echo ""
    echo "📋 Tunnel Status:"
    docker logs cloudflared --tail 10
    
    echo ""
    echo "🎉 Done! Your tunnel should be working now!"
    echo ""
    echo "Test it: https://jellyfin.morloksmaze.com"
    
else
    echo "❌ No token provided. Please run the script again."
fi
