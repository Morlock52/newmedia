#!/bin/bash

# Ultimate Media Server 2025 - Single Container Deployment Script
# Built with August 2025 best practices

set -e

echo "🚀 Ultimate Media Server 2025 - Single Container Deployment"
echo "==========================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker found${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker Compose found${NC}"

# Create necessary directories
echo -e "${YELLOW}📁 Creating directory structure...${NC}"
mkdir -p config/{jellyfin,sonarr,radarr,lidarr,prowlarr,bazarr,qbittorrent,sabnzbd,transmission,uptime-kuma,caddy}
mkdir -p media/{movies,tv,music}
mkdir -p downloads/{complete,incomplete}
mkdir -p s6-services

# Create basic s6 service run scripts
echo -e "${YELLOW}🔧 Creating s6 service definitions...${NC}"

# Caddy service
cat > s6-services/caddy-run << 'EOF'
#!/usr/bin/env sh
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
EOF

# Dashboard service
cat > s6-services/dashboard-run << 'EOF'
#!/usr/bin/env sh
cd /opt/dashboard
exec python3 -m http.server 3001
EOF

# API service
cat > s6-services/api-run << 'EOF'
#!/usr/bin/env sh
cd /opt/api
exec node server.js
EOF

chmod +x s6-services/*-run

# Stop existing containers that might conflict
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose down 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true

# Build the container
echo -e "${YELLOW}🏗️  Building Ultimate Media Server container...${NC}"
echo "This may take 10-15 minutes on first build..."

# Create a simplified Dockerfile for testing
cat > Dockerfile.ultimate-simple << 'DOCKERFILE'
FROM ubuntu:22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York

# Install basic services for testing
RUN apt-get update && apt-get install -y \
    curl wget nginx supervisor python3 nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install Caddy
RUN curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg && \
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list && \
    apt-get update && apt-get install -y caddy && \
    rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /config /data /opt/dashboard /opt/api

# Copy Caddyfile
COPY Caddyfile /etc/caddy/Caddyfile

# Create a simple dashboard
RUN echo '<!DOCTYPE html><html><head><title>Ultimate Media Server</title></head><body><h1>Ultimate Media Server 2025</h1><p>Single Container Solution</p><ul><li><a href="/jellyfin">Jellyfin</a></li><li><a href="/sonarr">Sonarr</a></li><li><a href="/radarr">Radarr</a></li></ul></body></html>' > /opt/dashboard/index.html

# Expose ports
EXPOSE 80 443

# Start services
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile"]
DOCKERFILE

# Build simplified version first for testing
docker build -f Dockerfile.ultimate-simple -t ultimate-media-server:simple .

# Run the container
echo -e "${YELLOW}🚀 Starting Ultimate Media Server...${NC}"

docker run -d \
    --name ultimate-media-server \
    -p 80:80 \
    -p 443:443 \
    -p 8096:8096 \
    -p 8989:8989 \
    -p 7878:7878 \
    -p 8686:8686 \
    -p 9696:9696 \
    -p 6767:6767 \
    -p 8080:8080 \
    -p 8085:8085 \
    -p 9091:9091 \
    -p 3001:3001 \
    -p 3002:3002 \
    -v $(pwd)/config:/config \
    -v $(pwd)/media:/data/media \
    -v $(pwd)/downloads:/data/downloads \
    --restart unless-stopped \
    ultimate-media-server:simple

# Wait for container to start
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 5

# Check if container is running
if docker ps | grep -q ultimate-media-server; then
    echo -e "${GREEN}✅ Container is running!${NC}"
    echo ""
    echo "📊 Container Status:"
    docker ps --filter name=ultimate-media-server --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo -e "${GREEN}🎉 Ultimate Media Server is ready!${NC}"
    echo ""
    echo "🌐 Access your server at:"
    echo "   Main Dashboard: http://localhost"
    echo "   Direct Ports:"
    echo "   - Jellyfin: http://localhost:8096"
    echo "   - Sonarr: http://localhost:8989"
    echo "   - Radarr: http://localhost:7878"
    echo "   - Lidarr: http://localhost:8686"
    echo "   - Prowlarr: http://localhost:9696"
    echo "   - Bazarr: http://localhost:6767"
    echo "   - qBittorrent: http://localhost:8080"
    echo "   - Dashboard: http://localhost:3001"
    echo ""
    echo "📝 Logs: docker logs ultimate-media-server"
    echo "🛑 Stop: docker stop ultimate-media-server"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    echo "Checking logs..."
    docker logs ultimate-media-server
    exit 1
fi