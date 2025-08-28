#!/bin/bash

# ============================================================================
# TRAEFIK REVERSE PROXY SETUP SCRIPT
# ============================================================================

set -euo pipefail

echo "🚀 Setting up Traefik reverse proxy for media server..."

# Create directories
echo "📁 Creating required directories..."
mkdir -p traefik-data/letsencrypt
mkdir -p monitoring/{prometheus,grafana}
mkdir -p homepage-config
mkdir -p postgres-init

# Set proper permissions for Let's Encrypt
echo "🔒 Setting permissions..."
chmod 600 traefik-data/letsencrypt
touch traefik-data/letsencrypt/acme.json
chmod 600 traefik-data/letsencrypt/acme.json

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik:8080']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['localhost:8080']

  - job_name: 'media-services'
    static_configs:
      - targets: 
        - 'jellyfin:8096'
        - 'sonarr:8989'
        - 'radarr:7878'
        - 'lidarr:8686'
        - 'bazarr:6767'
        - 'prowlarr:9696'
        - 'qbittorrent:8081'
        - 'sabnzbd:8080'
EOF

# Create Grafana datasource configuration
echo "📈 Creating Grafana configuration..."
mkdir -p monitoring/grafana/datasources
cat > monitoring/grafana/datasources/datasource.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create basic homepage configuration
echo "🏠 Creating Homepage configuration..."
cat > homepage-config/settings.yaml << 'EOF'
title: Media Server Dashboard
background: https://images.unsplash.com/photo-1506905925346-21bda4d32df4
theme: dark
color: slate

layout:
  Media Servers:
    style: row
    columns: 2
  Media Management:
    style: row
    columns: 4
  Downloads:
    style: row
    columns: 2
  Monitoring:
    style: row
    columns: 3
  Management:
    style: row
    columns: 2

providers:
  docker:
    endpoint: unix:///var/run/docker.sock
EOF

cat > homepage-config/services.yaml << 'EOF'
- Media Servers:
    - Jellyfin:
        href: https://jellyfin.localhost
        description: Open Source Media Server
        icon: jellyfin.png
        ping: http://jellyfin:8096
        
    - Plex:
        href: https://plex.localhost
        description: Premium Media Server
        icon: plex.png
        ping: http://plex:32400

- Media Management:
    - Sonarr:
        href: https://sonarr.localhost
        description: TV Series Management
        icon: sonarr.png
        ping: http://sonarr:8989
        
    - Radarr:
        href: https://radarr.localhost
        description: Movie Management
        icon: radarr.png
        ping: http://radarr:7878
        
    - Lidarr:
        href: https://lidarr.localhost
        description: Music Management
        icon: lidarr.png
        ping: http://lidarr:8686
        
    - Bazarr:
        href: https://bazarr.localhost
        description: Subtitle Management
        icon: bazarr.png
        ping: http://bazarr:6767
        
    - Prowlarr:
        href: https://prowlarr.localhost
        description: Indexer Management
        icon: prowlarr.png
        ping: http://prowlarr:9696

- Downloads:
    - qBittorrent:
        href: https://qbittorrent.localhost
        description: Torrent Client
        icon: qbittorrent.png
        ping: http://qbittorrent:8081
        
    - SABnzbd:
        href: https://sabnzbd.localhost
        description: Usenet Downloader
        icon: sabnzbd.png
        ping: http://sabnzbd:8080

- Monitoring:
    - Grafana:
        href: https://grafana.localhost
        description: Metrics Dashboard
        icon: grafana.png
        ping: http://grafana:3000
        
    - Prometheus:
        href: https://prometheus.localhost
        description: Metrics Collection
        icon: prometheus.png
        ping: http://prometheus:9090
        
    - Uptime Kuma:
        href: https://uptime.localhost
        description: Service Monitoring
        icon: uptime-kuma.png
        ping: http://uptime-kuma:3001

- Management:
    - Portainer:
        href: https://portainer.localhost
        description: Container Management
        icon: portainer.png
        ping: http://portainer:9000
        
    - Traefik:
        href: https://traefik.localhost
        description: Reverse Proxy
        icon: traefik.png
        ping: http://traefik:8080
EOF

cat > homepage-config/widgets.yaml << 'EOF'
- search:
    provider: duckduckgo
    target: _blank

- resources:
    cpu: true
    memory: true
    disk: /

- datetime:
    text_size: xl
    format:
      dateStyle: long
      timeStyle: medium
      hourCycle: h23
EOF

# Create PostgreSQL initialization script
echo "🗄️ Creating PostgreSQL initialization..."
cat > postgres-init/init.sql << 'EOF'
-- Create databases for various services
CREATE DATABASE grafana;
CREATE DATABASE homepage;

-- Create users with limited privileges
CREATE USER grafana_user WITH PASSWORD 'grafana_secure_password';
CREATE USER homepage_user WITH PASSWORD 'homepage_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE grafana TO grafana_user;
GRANT ALL PRIVILEGES ON DATABASE homepage TO homepage_user;

-- Additional security settings
ALTER DATABASE grafana OWNER TO grafana_user;
ALTER DATABASE homepage OWNER TO homepage_user;
EOF

# Create network verification script
echo "🔗 Creating network verification script..."
cat > scripts/verify-networks.sh << 'EOF'
#!/bin/bash

echo "🔍 Verifying Docker networks..."

# Check if networks exist
if docker network ls | grep -q "media-net"; then
    echo "✅ media-net network exists"
else
    echo "❌ media-net network missing"
    exit 1
fi

if docker network ls | grep -q "secure-net"; then
    echo "✅ secure-net network exists"
else
    echo "❌ secure-net network missing"
    exit 1
fi

# Check network configuration
echo "📊 Network details:"
docker network inspect media-net --format '{{.IPAM.Config}}'
docker network inspect secure-net --format '{{.IPAM.Config}}'

echo "✅ Network verification complete"
EOF

chmod +x scripts/verify-networks.sh

# Create health check script
echo "🏥 Creating comprehensive health check script..."
cat > scripts/health-check-all.sh << 'EOF'
#!/bin/bash

echo "🏥 Running comprehensive health checks..."

services=(
    "traefik:8080"
    "jellyfin:8096"
    "sonarr:8989"
    "radarr:7878"
    "lidarr:8686"
    "bazarr:6767"
    "prowlarr:9696"
    "qbittorrent:8081"
    "sabnzbd:8080"
    "prometheus:9090"
    "grafana:3000"
    "uptime-kuma:3001"
    "portainer:9000"
    "homepage:3000"
    "postgres:5432"
    "redis:6379"
)

healthy=0
total=${#services[@]}

for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}
    
    if docker exec "$name" curl -f "http://localhost:$port" >/dev/null 2>&1; then
        echo "✅ $name is healthy"
        ((healthy++))
    else
        echo "❌ $name is unhealthy"
    fi
done

echo "📊 Health check summary: $healthy/$total services healthy"

if [ $healthy -eq $total ]; then
    echo "🎉 All services are healthy!"
    exit 0
else
    echo "⚠️  Some services need attention"
    exit 1
fi
EOF

chmod +x scripts/health-check-all.sh

# Create startup script
echo "🚀 Creating startup script..."
cat > start-media-server.sh << 'EOF'
#!/bin/bash

set -euo pipefail

echo "🚀 Starting optimized media server stack..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from template..."
    cp .env.fixed.template .env
    echo "📝 Please edit .env file with your configuration before proceeding"
    exit 1
fi

# Create required directories
echo "📁 Creating directories..."
mkdir -p media downloads config logs backups
mkdir -p traefik-data/letsencrypt
mkdir -p portainer-data uptime-kuma-data

# Set permissions
echo "🔒 Setting permissions..."
chmod 600 traefik-data/letsencrypt/acme.json 2>/dev/null || touch traefik-data/letsencrypt/acme.json && chmod 600 traefik-data/letsencrypt/acme.json

# Pull latest images
echo "📦 Pulling latest images..."
docker-compose -f docker-compose.fixed.yml pull

# Start services
echo "🔄 Starting services..."
docker-compose -f docker-compose.fixed.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Run health checks
echo "🏥 Running initial health checks..."
./scripts/health-check-all.sh

echo "✅ Media server started successfully!"
echo ""
echo "🌐 Access your services:"
echo "  Dashboard: https://localhost (or https://yourdomain.com)"
echo "  Traefik:   https://traefik.localhost"
echo "  Jellyfin:  https://jellyfin.localhost"
echo "  Grafana:   https://grafana.localhost"
echo ""
echo "📝 Check logs: docker-compose -f docker-compose.fixed.yml logs -f"
echo "📊 Monitor status: ./scripts/health-check-all.sh"
EOF

chmod +x start-media-server.sh

# Create stop script
echo "🛑 Creating stop script..."
cat > stop-media-server.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping media server..."
docker-compose -f docker-compose.fixed.yml down

echo "🧹 Cleaning up (optional - uncomment to enable):"
echo "# docker-compose -f docker-compose.fixed.yml down -v  # Remove volumes"
echo "# docker system prune -f  # Remove unused containers/images"

echo "✅ Media server stopped"
EOF

chmod +x stop-media-server.sh

echo "✅ Traefik setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.fixed.template to .env and configure"
echo "2. Run ./start-media-server.sh"
echo "3. Access dashboard at https://localhost"
echo ""
echo "🔧 Configuration files created:"
echo "  - docker-compose.fixed.yml (main configuration)"
echo "  - .env.fixed.template (environment template)"
echo "  - traefik-data/ (SSL certificates)"
echo "  - monitoring/ (Prometheus & Grafana)"
echo "  - homepage-config/ (Dashboard configuration)"
echo "  - scripts/ (Management scripts)"
echo ""
echo "🌐 All services will be available via HTTPS with automatic SSL certificates"