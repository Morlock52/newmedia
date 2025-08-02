#!/bin/bash

# Auto-configuration runner script
# This script executes the media server auto-configuration process

set -e

echo "🚀 Starting Media Server Auto-Configuration"
echo "=========================================="
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Make scripts executable
echo "📝 Making scripts executable..."
chmod +x scripts/auto-configure-all-services.sh
chmod +x scripts/quick-setup.sh
chmod +x scripts/quick-setup-arr-stack.sh
chmod +x scripts/auto-configure-arr-stack.sh

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check for docker-compose file
if [ -f "docker-compose-demo.yml" ]; then
    echo "📄 Found docker-compose-demo.yml"
    COMPOSE_FILE="docker-compose-demo.yml"
elif [ -f "docker-compose.yml" ]; then
    echo "📄 Found docker-compose.yml"
    COMPOSE_FILE="docker-compose.yml"
else
    echo "❌ Error: No docker-compose file found!"
    exit 1
fi

echo ""
echo "🔍 Checking current container status..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

echo ""
echo "📦 Checking essential services..."
# Check which services are already running
RUNNING_SERVICES=$(docker ps --format "{{.Names}}")
TO_START=()

for service in prowlarr sonarr radarr jellyfin qbittorrent homarr; do
    if echo "$RUNNING_SERVICES" | grep -q "^${service}$"; then
        echo "✅ $service is already running"
    else
        echo "🚀 Will start $service"
        TO_START+=("$service")
    fi
done

# Only start services that aren't already running
if [ ${#TO_START[@]} -gt 0 ]; then
    echo ""
    echo "📦 Starting services: ${TO_START[*]}..."
    docker-compose -f "$COMPOSE_FILE" up -d ${TO_START[*]} 2>&1 | tail -20
else
    echo ""
    echo "✅ All essential services are already running"
fi

echo ""
echo "⏳ Waiting for services to initialize (30 seconds)..."
sleep 30

echo ""
echo "🔧 Running quick setup script..."
./scripts/quick-setup.sh

echo ""
echo "🔧 Running full auto-configuration script..."
./scripts/auto-configure-all-services.sh "$COMPOSE_FILE"

echo ""
echo "✅ Auto-configuration complete!"
echo ""
echo "📊 Service Status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(prowlarr|sonarr|radarr|jellyfin|qbittorrent|homarr)" || echo "No services found"

echo ""
echo "🌐 Service URLs:"
echo "  - Homarr Dashboard: http://localhost:7575"
echo "  - Jellyfin: http://localhost:8096"
echo "  - Prowlarr: http://localhost:9696"
echo "  - Sonarr: http://localhost:8989"
echo "  - Radarr: http://localhost:7878"
echo "  - qBittorrent: http://localhost:8090 (admin/adminadmin)"
echo ""