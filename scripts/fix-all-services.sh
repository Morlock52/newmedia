#!/bin/bash

# Comprehensive Service Fix Script
# Author: Media Server Team
# Date: August 15, 2025

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Starting comprehensive service fixes...${NC}\n"

# 1. Stop all running containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
docker-compose down 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true

# 2. Clean up Docker resources
echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
docker system prune -f --volumes 2>/dev/null || true

# 3. Fix network configuration
echo -e "${BLUE}Fixing Docker networks...${NC}"
docker network rm media-net downloads-net vpn-net monitoring-net management-net 2>/dev/null || true

docker network create --driver bridge \
    --subnet=172.30.0.0/16 \
    --gateway=172.30.0.1 \
    --opt com.docker.network.bridge.name=media-br0 \
    media-net

docker network create --driver bridge \
    --subnet=172.31.0.0/16 \
    --gateway=172.31.0.1 \
    --opt com.docker.network.bridge.name=downloads-br0 \
    downloads-net

docker network create --driver bridge \
    --subnet=172.33.0.0/16 \
    --gateway=172.33.0.1 \
    --opt com.docker.network.bridge.name=monitoring-br0 \
    monitoring-net

echo -e "${GREEN}✅ Networks fixed${NC}\n"

# 4. Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p {config,data,media,downloads,logs,backups}/{jellyfin,sonarr,radarr,prowlarr,qbittorrent,prometheus,grafana}
mkdir -p media/{movies,tv,music,books}
mkdir -p downloads/{complete,incomplete,torrents,usenet}
chmod -R 755 config data media downloads

echo -e "${GREEN}✅ Directories created${NC}\n"

# 5. Setup environment
echo -e "${BLUE}Setting up environment...${NC}"
if [ ! -f .env ]; then
    cp .env.fixed .env 2>/dev/null || cp .env.template .env
    
    # Generate secure passwords
    sed -i.bak "s/changeThisSecurePassword123!/$(openssl rand -base64 32)/" .env
    sed -i.bak "s/changeThisSecurePassword456!/$(openssl rand -base64 32)/" .env
    sed -i.bak "s/changeThisSecurePassword789!/$(openssl rand -base64 32)/" .env
    sed -i.bak "s/your-secure-api-key-here-change-in-production/$(openssl rand -hex 32)/" .env
    sed -i.bak "s/your-jwt-secret-key-here-change-in-production/$(openssl rand -hex 32)/" .env
fi

echo -e "${GREEN}✅ Environment configured${NC}\n"

# 6. Fix Docker Compose configuration
echo -e "${BLUE}Updating Docker Compose configuration...${NC}"
cat > docker-compose-fixed.yml << 'EOF'
version: '3.9'

networks:
  media-net:
    external: true
  downloads-net:
    external: true
  monitoring-net:
    external: true

volumes:
  postgres-data:
  redis-data:
  jellyfin-config:
  sonarr-config:
  radarr-config:
  prowlarr-config:
  qbittorrent-config:

services:
  # Database Services
  postgres:
    image: postgres:16-alpine
    container_name: postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-mediaserver}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - media-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: redis
    command: redis-server --save 60 1 --loglevel warning
    volumes:
      - redis-data:/data
    networks:
      - media-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Media Server
  jellyfin:
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
    volumes:
      - jellyfin-config:/config
      - ./media:/media
      - /dev/dri:/dev/dri
    ports:
      - "8096:8096"
      - "8920:8920"
      - "7359:7359/udp"
      - "1900:1900/udp"
    networks:
      - media-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8096/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Content Management
  sonarr:
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
    volumes:
      - sonarr-config:/config
      - ./media:/media
      - ./downloads:/downloads
    ports:
      - "8989:8989"
    networks:
      - media-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8989/ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  radarr:
    image: lscr.io/linuxserver/radarr:latest
    container_name: radarr
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
    volumes:
      - radarr-config:/config
      - ./media:/media
      - ./downloads:/downloads
    ports:
      - "7878:7878"
    networks:
      - media-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7878/ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prowlarr:
    image: lscr.io/linuxserver/prowlarr:latest
    container_name: prowlarr
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
    volumes:
      - prowlarr-config:/config
    ports:
      - "9696:9696"
    networks:
      - media-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9696/ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Download Client
  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
      - WEBUI_PORT=8080
    volumes:
      - qbittorrent-config:/config
      - ./downloads:/downloads
    ports:
      - "8080:8080"
      - "6881:6881"
      - "6881:6881/udp"
    networks:
      - media-net
      - downloads-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080"]
      interval: 30s
      timeout: 10s
      retries: 3

  # API Server
  api-server:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: api-server
    environment:
      - NODE_ENV=production
      - API_PORT=3002
      - CORS_ORIGIN=http://localhost:3030
      - API_KEY=${API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - POSTGRES_HOST=postgres
      - POSTGRES_USER=${POSTGRES_USER:-postgres}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}
      - POSTGRES_DB=${POSTGRES_DB:-mediaserver}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./api:/app
    ports:
      - "3002:3002"
    networks:
      - media-net
      - monitoring-net
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Dashboard
  media-dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    container_name: media-dashboard
    environment:
      - NODE_ENV=production
      - API_BASE_URL=http://api-server:3002
      - WS_URL=ws://api-server:3002
      - NEXT_PUBLIC_API_URL=http://localhost:3002
      - NEXT_PUBLIC_WS_URL=ws://localhost:3002
    ports:
      - "3030:3000"
    networks:
      - media-net
    depends_on:
      - api-server
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

echo -e "${GREEN}✅ Docker Compose configuration fixed${NC}\n"

# 7. Start services
echo -e "${BLUE}Starting services...${NC}"
docker-compose -f docker-compose-fixed.yml up -d postgres redis
sleep 10
docker-compose -f docker-compose-fixed.yml up -d jellyfin sonarr radarr prowlarr qbittorrent
sleep 10
docker-compose -f docker-compose-fixed.yml up -d api-server media-dashboard

echo -e "${GREEN}✅ Services started${NC}\n"

# 8. Health check
echo -e "${BLUE}Performing health checks...${NC}"
sleep 15

services=(
    "Jellyfin:8096/health"
    "Sonarr:8989/ping"
    "Radarr:7878/ping"
    "Prowlarr:9696/ping"
    "qBittorrent:8080"
    "API Server:3002/health"
)

for service in "${services[@]}"; do
    IFS=':' read -r name endpoint <<< "$service"
    if curl -s "http://localhost:$endpoint" &>/dev/null; then
        echo -e "${GREEN}✅ $name is healthy${NC}"
    else
        echo -e "${RED}❌ $name health check failed${NC}"
    fi
done

echo -e "\n${GREEN}🎉 Service fixes complete!${NC}"
echo -e "\n${BLUE}Access your services at:${NC}"
echo "  Jellyfin:    http://localhost:8096"
echo "  Sonarr:      http://localhost:8989"
echo "  Radarr:      http://localhost:7878"
echo "  Prowlarr:    http://localhost:9696"
echo "  qBittorrent: http://localhost:8080"
echo "  Dashboard:   http://localhost:3030"
echo "  API:         http://localhost:3002"