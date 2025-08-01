#!/bin/bash

echo "🎬 Media Server Status Dashboard"
echo "================================"
echo ""

echo "📊 Running Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(jellyfin|sonarr|radarr|prowlarr|overseerr|qbittorrent|NAMES)"

echo ""
echo "🌐 Access Your Media Apps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test connectivity and show status
check_service() {
    local name=$1
    local url=$2
    local port=$3
    
    if curl -s --connect-timeout 2 "$url" > /dev/null 2>&1; then
        echo "✅ $name: $url"
    else
        echo "⏳ $name: $url (starting up...)"
    fi
}

check_service "Jellyfin (Media Server)" "http://localhost:8096" "8096"
check_service "Prowlarr (Indexers)" "http://localhost:9696" "9696"  
check_service "Sonarr (TV Shows)" "http://localhost:8989" "8989"
check_service "Radarr (Movies)" "http://localhost:7878" "7878"
check_service "Overseerr (Requests)" "http://localhost:5055" "5055"
check_service "qBittorrent (Downloads)" "http://localhost:8080" "8080"

echo ""
echo "🔐 Default Credentials:"
echo "   qBittorrent: admin / adminadmin (change on first login)"
echo ""
echo "📁 Data Locations:"
echo "   Config: $(pwd)/config/"
echo "   Media:  $(pwd)/data/media/"
echo "   Downloads: $(pwd)/data/downloads/"
echo ""
echo "💡 Quick Commands:"
echo "   📊 Status: docker ps"
echo "   📜 Logs:   docker logs [container-name]"
echo "   🔄 Restart: docker restart [container-name]"
echo "   🛑 Stop:   docker stop [container-name]"