#!/bin/bash

echo "🚀 Testing Ultimate Single Container Setup"
echo "=========================================="

# Stop any existing ultimate container
docker stop ultimate-test 2>/dev/null || true
docker rm ultimate-test 2>/dev/null || true

# Create a simple test container with Caddy and dashboard
cat > Dockerfile.test << 'EOF'
FROM caddy:2-alpine

# Install Node.js for dashboard
RUN apk add --no-cache nodejs npm python3

# Create directories
RUN mkdir -p /opt/dashboard /config /data

# Create simple dashboard
RUN echo '<!DOCTYPE html>
<html>
<head>
    <title>Ultimate Media Server 2025</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .services { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }
        .service { background: #2a2a2a; padding: 20px; border-radius: 10px; text-align: center; transition: transform 0.2s; }
        .service:hover { transform: translateY(-5px); background: #3a3a3a; }
        .service a { color: #00ff88; text-decoration: none; font-size: 18px; }
        .status { color: #ffaa00; font-size: 12px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🎬 Ultimate Media Server 2025</h1>
    <p>Single Container Solution with All Services</p>
    <div class="services">
        <div class="service">
            <a href="http://localhost:8096">📺 Jellyfin</a>
            <div class="status">Port 8096</div>
        </div>
        <div class="service">
            <a href="http://localhost:8989">📡 Sonarr</a>
            <div class="status">Port 8989</div>
        </div>
        <div class="service">
            <a href="http://localhost:7878">🎥 Radarr</a>
            <div class="status">Port 7878</div>
        </div>
        <div class="service">
            <a href="http://localhost:8686">🎵 Lidarr</a>
            <div class="status">Port 8686</div>
        </div>
        <div class="service">
            <a href="http://localhost:9696">🔍 Prowlarr</a>
            <div class="status">Port 9696</div>
        </div>
        <div class="service">
            <a href="http://localhost:6767">📝 Bazarr</a>
            <div class="status">Port 6767</div>
        </div>
        <div class="service">
            <a href="http://localhost:8080">💾 qBittorrent</a>
            <div class="status">Port 8080</div>
        </div>
        <div class="service">
            <a href="http://localhost:5173">🎨 Dashboard</a>
            <div class="status">Port 5173</div>
        </div>
    </div>
    <p style="margin-top: 40px; color: #888;">All services accessible through single container on port 80</p>
</body>
</html>' > /opt/dashboard/index.html

# Caddyfile for routing
RUN echo ':80 {
    root * /opt/dashboard
    file_server
    
    handle /health {
        respond "OK" 200
    }
}' > /etc/caddy/Caddyfile

EXPOSE 80
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile"]
EOF

# Build test container
echo "🏗️  Building test container..."
docker build -f Dockerfile.test -t ultimate-test:latest .

# Run test container
echo "🚀 Starting test container..."
docker run -d \
    --name ultimate-test \
    -p 8090:80 \
    ultimate-test:latest

# Wait for startup
sleep 2

# Check if running
if docker ps | grep -q ultimate-test; then
    echo ""
    echo "✅ SUCCESS! Single container is running!"
    echo ""
    echo "🌐 Access the dashboard at: http://localhost:8090"
    echo ""
    echo "📊 Container status:"
    docker ps --filter name=ultimate-test --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "This demonstrates the single container concept."
    echo "The full version would include all media services."
else
    echo "❌ Container failed to start"
    docker logs ultimate-test
fi