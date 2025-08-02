#!/usr/bin/env bash
set -euo pipefail

echo "🆕 Create New Cloudflare Tunnel"
echo "==============================="

# Load environment
source .env

echo "Domain: $DOMAIN"
echo ""

echo "📋 Steps to Create New Tunnel:"
echo "=============================="

echo ""
echo "1. 🗑️  Delete Old Tunnel (if needed):"
echo "   • Go to: https://one.dash.cloudflare.com/"
echo "   • Networks → Tunnels"
echo "   • Find: morloksmaze-media-tunnel"
echo "   • Click three dots → Delete"

echo ""
echo "2. ➕ Create New Tunnel:"
echo "   • Click: 'Create a tunnel'"
echo "   • Name: morloksmaze-media-tunnel"
echo "   • Save tunnel"
echo "   • Copy the tunnel token"

echo ""
echo "3. 🔧 Update Configuration:"
echo "   • Update CLOUDFLARE_TUNNEL_TOKEN in .env file"
echo "   • Run: docker-compose restart cloudflared"

echo ""
echo "4. 🌐 Configure Public Hostnames:"
echo "   Add these hostnames (ALL point to HTTP://traefik:80):"

services=("jellyfin" "sonarr" "radarr" "lidarr" "readarr" "bazarr" "prowlarr" "overseerr" "tautulli" "mylar" "podgrab" "youtube-dl" "photoprism" "grafana" "prometheus" "traefik")

for service in "${services[@]}"; do
    echo "   • $service.$DOMAIN → HTTP://traefik:80"
done

echo ""
echo "5. 🔐 Set Up Access Applications:"
echo "   • Go to: Access → Applications"
echo "   • Create application for each service"
echo "   • Domain: [service].$DOMAIN"
echo "   • Policy: Allow → Emails → $EMAIL"

echo ""
echo "🚀 Quick Commands After New Token:"
echo "================================="
echo ""
echo "# Update .env with new token, then:"
echo "docker-compose down"
echo "docker-compose up -d"
echo "docker logs cloudflared --follow"

echo ""
echo "💡 Look for 'Registered tunnel connection' in the logs"

echo ""
read -p "Open Cloudflare dashboard to create new tunnel? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Opening Cloudflare Zero Trust dashboard..."
    osascript -e 'tell application "Google Chrome" to open location "https://one.dash.cloudflare.com/"'
    
    echo ""
    echo "Follow the steps above to create a new tunnel."
    echo "After you get the new token, update your .env file and restart the services."
fi
