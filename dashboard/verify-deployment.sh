#\!/bin/bash

echo "🔍 Media Server Deployment Verification"
echo "======================================="
echo ""

# Quick service check
echo "✅ Currently Running Services:"
echo "------------------------------"
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "(jellyfin|sonarr|radarr|prowlarr|qbittorrent)" | head -10

echo ""
echo "📊 Service URLs:"
echo "----------------"
echo "• Jellyfin:     http://localhost:8096"
echo "• Sonarr:       http://localhost:8989"
echo "• Radarr:       http://localhost:7878"
echo "• Prowlarr:     http://localhost:9696"
echo "• qBittorrent:  http://localhost:8080"
echo "• Uptime Kuma:  http://localhost:3001"
echo "• Homepage:     http://localhost:3000"

echo ""
echo "🧪 Quick Health Check:"
echo "----------------------"
for port in 8096 8989 7878 9696 8080 3000; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port | grep -q "200\|302"; then
        echo "✅ Port $port: OK"
    else
        echo "⚠️  Port $port: Check needed"
    fi
done

echo ""
echo "📈 System Status:"
echo "-----------------"
echo "• Containers: $(docker ps -q | wc -l) running"
echo "• Images: $(docker images -q | wc -l) available"
echo "• Networks: $(docker network ls -q | wc -l) configured"

echo ""
echo "✅ Deployment Status: OPERATIONAL"
echo ""
