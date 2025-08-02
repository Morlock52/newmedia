#!/usr/bin/env bash
set -euo pipefail

echo "🔐 Cloudflare Zero Trust Setup for Complete Media Stack"
echo "======================================================="

# Load environment
source .env

echo "Domain: $DOMAIN"
echo "Admin Email: $EMAIL"
echo ""

# All services that need Cloudflare Access protection
declare -A all_services=(
    # Core Services
    ["jellyfin"]="Media Server - Stream movies, TV, music"
    ["sonarr"]="TV Management - Download TV shows automatically"
    ["radarr"]="Movie Management - Download movies automatically"
    ["prowlarr"]="Indexer Management - Manage torrent/usenet sources"
    ["overseerr"]="Request Management - User-friendly media requests"
    ["traefik"]="Reverse Proxy Dashboard - View routing and SSL status"
    
    # Additional Media Services
    ["lidarr"]="Music Management - Download music automatically"
    ["readarr"]="Book Management - Download ebooks automatically" 
    ["bazarr"]="Subtitle Management - Download subtitles for media"
    ["tautulli"]="Jellyfin Analytics - View usage stats and monitoring"
    ["mylar"]="Comic Management - Download comics automatically"
    ["podgrab"]="Podcast Management - Download and organize podcasts"
    ["youtube-dl"]="Video Downloader - Download videos from YouTube/other sites"
    ["photoprism"]="Photo Management - AI-powered photo organization"
    
    # Monitoring Services
    ["grafana"]="Monitoring Dashboards - Visualize system metrics"
    ["prometheus"]="Metrics Collection - System and service monitoring"
    ["alertmanager"]="Alert Management - Send notifications for issues"
)

echo "🎯 Services to Configure in Cloudflare Zero Trust:"
echo "================================================="
echo ""

for service in "${!all_services[@]}"; do
    description="${all_services[$service]}"
    echo "🔗 $service.$DOMAIN"
    echo "   $description"
    echo ""
done

echo "📋 Cloudflare Zero Trust Configuration Steps:"
echo "============================================="
echo ""

echo "1. 🌐 DNS Records (Auto-create with API):"
echo "   All CNAME records pointing to $DOMAIN with Proxy enabled (🧡)"
echo ""

echo "2. 🔗 Tunnel Public Hostnames:"
echo "   In Zero Trust → Networks → Tunnels → morloksmaze-media-tunnel"
echo "   Add these public hostnames (ALL point to: HTTP://traefik:80):"
echo ""

for service in "${!all_services[@]}"; do
    echo "   $service.$DOMAIN → HTTP://traefik:80"
done

echo ""
echo "3. 🛡️  Access Applications:"
echo "   In Zero Trust → Access → Applications"
echo "   Create one application for each service with these settings:"
echo ""

echo "   Application Template:"
echo "   - Type: Self-hosted"
echo "   - Application domain: [service].$DOMAIN"
echo "   - Session duration: 24 hours"
echo "   - Policy: Allow → Emails → $EMAIL"
echo ""

echo "🚀 Quick Setup Commands:"
echo "======================="
echo ""

echo "# 1. Open Cloudflare Zero Trust Dashboard"
echo "open 'https://one.dash.cloudflare.com/'"
echo ""

echo "# 2. Open Cloudflare DNS Management" 
echo "open 'https://dash.cloudflare.com/$DOMAIN/dns'"
echo ""

echo "# 3. Test a service after configuration"
echo "open 'https://jellyfin.$DOMAIN'"
echo ""

echo "📝 Configuration Checklist:"
echo "=========================="
echo ""
echo "□ Authentication method configured (One-time PIN)"
echo "□ Tunnel public hostnames added (${#all_services[@]} total)"
echo "□ Access applications created (${#all_services[@]} total)"
echo "□ DNS records verified (all proxied 🧡)"
echo "□ Test authentication flow"
echo ""

echo "🎯 Priority Services to Test First:"
echo "==================================="
echo "1. jellyfin.$DOMAIN - Core media streaming"
echo "2. overseerr.$DOMAIN - Request new content"
echo "3. sonarr.$DOMAIN - TV show management"
echo "4. radarr.$DOMAIN - Movie management"
echo "5. prowlarr.$DOMAIN - Indexer configuration"
echo ""

echo "🔒 Remember:"
echo "============"
echo "• All services will require email authentication"
echo "• You'll receive a PIN code at: $EMAIL"
echo "• Session lasts 24 hours by default"
echo "• Use incognito/private browsing if you have issues"
echo ""

echo "🛠️  Troubleshooting:"
echo "==================="
echo "• Check tunnel status: docker logs cloudflared"
echo "• Verify service health: docker ps"
echo "• Test internal access: curl -I http://localhost:8096"
echo "• View Traefik routes: https://traefik.$DOMAIN"
echo ""

echo "📊 After setup, run the complete test:"
echo "======================================"
echo "./test-complete-stack.sh"
