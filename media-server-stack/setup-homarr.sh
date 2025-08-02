#!/bin/bash

# Homarr Dashboard Auto-Setup Script
echo "🏠 Setting up Homarr dashboard automatically..."

HOMARR_URL="http://localhost:8080"
HOST_HEADER="Host: home.morloksmaze.com"

# Wait for Homarr to be ready
echo "⏳ Waiting for Homarr to be accessible..."
for i in {1..30}; do
    if curl -s -H "$HOST_HEADER" "$HOMARR_URL" > /dev/null 2>&1; then
        echo "✅ Homarr is accessible!"
        break
    fi
    sleep 2
    echo "Attempt $i/30..."
done

# Test the connection
echo "🧪 Testing Homarr access..."
curl -I -H "$HOST_HEADER" "$HOMARR_URL"

echo ""
echo "🎯 Manual Setup Required:"
echo "1. Open https://home.morloksmaze.com in your browser"
echo "2. Login with Authelia (morlock/changeme123)"
echo "3. Complete Homarr onboarding wizard"
echo "4. Add these services to your dashboard:"
echo ""
echo "📋 Services to Add:"
echo "- Jellyfin: https://jellyfin.morloksmaze.com"
echo "- Overseerr: https://overseerr.morloksmaze.com"
echo "- Sonarr: https://sonarr.morloksmaze.com"
echo "- Radarr: https://radarr.morloksmaze.com"
echo "- Lidarr: https://lidarr.morloksmaze.com"
echo "- Prowlarr: https://prowlarr.morloksmaze.com"
echo "- Authelia: https://auth.morloksmaze.com"
echo ""
echo "✨ All services should show green status indicators when working properly"