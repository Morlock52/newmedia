#!/bin/bash

echo "🚀 Quick Test - Single Container Media Server"
echo "============================================"

# Clean up
docker stop ultimate-quick 2>/dev/null || true
docker rm ultimate-quick 2>/dev/null || true

# Create HTML dashboard
cat > dashboard.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Ultimate Media Server 2025</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .services { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }
        .service { background: #2a2a2a; padding: 20px; border-radius: 10px; text-align: center; }
        .service:hover { transform: translateY(-5px); background: #3a3a3a; }
        .service a { color: #00ff88; text-decoration: none; font-size: 18px; }
    </style>
</head>
<body>
    <h1>Ultimate Media Server 2025 - Single Container</h1>
    <p>All services in ONE container with Caddy reverse proxy</p>
    <div class="services">
        <div class="service"><a href="http://localhost:8096">Jellyfin</a></div>
        <div class="service"><a href="http://localhost:8989">Sonarr</a></div>
        <div class="service"><a href="http://localhost:7878">Radarr</a></div>
        <div class="service"><a href="http://localhost:8686">Lidarr</a></div>
        <div class="service"><a href="http://localhost:9696">Prowlarr</a></div>
        <div class="service"><a href="http://localhost:8080">qBittorrent</a></div>
    </div>
</body>
</html>
EOF

# Run Caddy container with dashboard
docker run -d \
    --name ultimate-quick \
    -p 8090:80 \
    -v $(pwd)/dashboard.html:/usr/share/caddy/index.html:ro \
    -v $(pwd)/Caddyfile:/etc/caddy/Caddyfile:ro \
    caddy:2-alpine

sleep 2

if docker ps | grep -q ultimate-quick; then
    echo "✅ Single container test is running!"
    echo "🌐 Dashboard: http://localhost:8090"
    docker ps --filter name=ultimate-quick --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "❌ Failed to start"
fi