#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Fix Error 1033 - Configure Tunnel Public Hostnames"
echo "======================================================"

# Load environment
source .env

echo "Domain: $DOMAIN"
echo "Issue: Tunnel connected but public hostnames not configured"
echo ""

echo "🎯 SOLUTION: Add Public Hostnames to Your Tunnel"
echo "==============================================="

echo ""
echo "You need to configure these public hostnames in Cloudflare:"

# All services that need hostnames
services=(
    "jellyfin:Media Server"
    "sonarr:TV Management" 
    "radarr:Movie Management"
    "lidarr:Music Management"
    "readarr:Book Management"
    "bazarr:Subtitle Management"
    "prowlarr:Indexer Management"
    "overseerr:Request Management"
    "tautulli:Jellyfin Analytics"
    "mylar:Comic Management"
    "podgrab:Podcast Management"
    "youtube-dl:Video Downloader"
    "photoprism:Photo Management"
    "grafana:Monitoring Dashboards"
    "prometheus:Metrics Collection"
    "traefik:Reverse Proxy Dashboard"
)

echo ""
echo "📋 PUBLIC HOSTNAMES TO ADD:"
echo "============================"

for service_desc in "${services[@]}"; do
    IFS=':' read -r service description <<< "$service_desc"
    echo "🔗 $service.$DOMAIN"
    echo "   Service Type: HTTP"
    echo "   URL: traefik:80"
    echo "   Additional Headers: Host → $service.$DOMAIN"
    echo ""
done

echo "🚀 STEP-BY-STEP INSTRUCTIONS:"
echo "============================="

echo ""
echo "1. 🌐 Open Cloudflare Zero Trust Dashboard:"
echo "   https://one.dash.cloudflare.com/"

echo ""
echo "2. 📍 Navigate to Your Tunnel:"
echo "   • Click 'Networks' → 'Tunnels'"
echo "   • Click 'morloksmaze-media-tunnel'"

echo ""
echo "3. ➕ Add Public Hostnames:"
echo "   • Click 'Public Hostnames' tab"
echo "   • Click 'Add a public hostname' for each service"

echo ""
echo "4. 📝 For EACH service, enter:"
echo "   • Subdomain: [service-name] (e.g., 'prowlarr')"
echo "   • Domain: morloksmaze.com"
echo "   • Service Type: HTTP"
echo "   • URL: traefik:80"

echo ""
echo "5. 🔧 IMPORTANT Settings:"
echo "   • Service Type: HTTP (not HTTPS)"
echo "   • URL: traefik:80 (this routes through your reverse proxy)"
echo "   • Additional headers: Host → [service].morloksmaze.com"

echo ""
echo "🎯 PRIORITY SERVICES (Add these first):"
echo "======================================="
echo "1. prowlarr.morloksmaze.com → HTTP://traefik:80"
echo "2. jellyfin.morloksmaze.com → HTTP://traefik:80"
echo "3. overseerr.morloksmaze.com → HTTP://traefik:80"
echo "4. sonarr.morloksmaze.com → HTTP://traefik:80" 
echo "5. radarr.morloksmaze.com → HTTP://traefik:80"

echo ""
echo "⚡ QUICK TEST:"
echo "============="
echo "After adding prowlarr.morloksmaze.com hostname:"
echo "• Wait 30 seconds"
echo "• Try https://prowlarr.morloksmaze.com again"
echo "• Should work immediately!"

echo ""
echo "🆘 ALTERNATIVE - Let me open the exact page:"

read -p "Open Cloudflare tunnel configuration now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Opening Cloudflare tunnel configuration..."
    osascript -e 'tell application "Google Chrome" to open location "https://one.dash.cloudflare.com/"'
    
    echo ""
    echo "📍 Navigate to: Networks → Tunnels → morloksmaze-media-tunnel → Public Hostnames"
    echo ""
    echo "✅ Add prowlarr.morloksmaze.com → HTTP://traefik:80 first"
    echo "✅ Then test: https://prowlarr.morloksmaze.com"
fi

echo ""
echo "💡 Why this works:"
echo "=================="
echo "• Your tunnel is connected ✅"
echo "• Your services are running ✅" 
echo "• You just need to tell Cloudflare how to route the domains"
echo "• All services route through traefik:80 which handles the rest"
