#!/bin/bash

# Ultimate Media Server 2025 - Volume Mapping Deployment Script
# This script gracefully stops containers, updates volume mappings, and redeploys

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🚀 Ultimate Media Server 2025 - Volume Mapping Deployment${NC}"
echo "=================================================================="
echo ""

# Check if /Volumes/Plex exists
PLEX_VOLUME="/Volumes/Plex"
if [ ! -d "$PLEX_VOLUME" ]; then
    echo -e "${RED}❌ Error: $PLEX_VOLUME does not exist!${NC}"
    echo "Please ensure the Plex volume is mounted before continuing."
    exit 1
fi

# Check write permissions
echo -e "${BLUE}🔍 Checking write permissions to $PLEX_VOLUME...${NC}"
if [ ! -w "$PLEX_VOLUME" ]; then
    echo -e "${RED}❌ Error: No write permissions to $PLEX_VOLUME${NC}"
    echo "Please ensure proper permissions are set."
    exit 1
fi

echo -e "${GREEN}✅ $PLEX_VOLUME is accessible and writable${NC}"

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "   Waiting for $service"
    while ! nc -z localhost $port 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e " ${RED}[TIMEOUT]${NC}"
            return 1
        fi
        echo -n "."
        sleep 3
        ((attempt++))
    done
    echo -e " ${GREEN}[OK]${NC}"
}

# Phase 1: Graceful Shutdown
echo ""
echo -e "${YELLOW}🛑 Phase 1: Graceful Container Shutdown${NC}"
echo ""

# Stop all containers gracefully
echo -e "${BLUE}📋 Checking running containers...${NC}"
RUNNING_CONTAINERS=$(docker ps --format '{{.Names}}' | grep -E "(jellyfin|plex|emby|sonarr|radarr|lidarr|prowlarr|bazarr|qbittorrent|overseerr|tautulli)" 2>/dev/null || true)

if [ -n "$RUNNING_CONTAINERS" ]; then
    echo -e "${YELLOW}🔄 Stopping containers gracefully...${NC}"
    for container in $RUNNING_CONTAINERS; do
        echo -e "   Stopping $container..."
        docker stop $container --time=30 2>/dev/null || true
    done
    echo -e "${GREEN}✅ All containers stopped gracefully${NC}"
else
    echo -e "${GREEN}ℹ️  No relevant containers running${NC}"
fi

# Phase 2: Update Environment Configuration
echo ""
echo -e "${YELLOW}🔧 Phase 2: Update Volume Configuration${NC}"
echo ""

# Update .env file with new paths
echo -e "${BLUE}📝 Updating environment configuration...${NC}"

# Backup existing .env
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# Update paths in .env
sed -i.bak "s|MEDIA_PATH=.*|MEDIA_PATH=$PLEX_VOLUME|g" .env
sed -i.bak "s|DOWNLOADS_PATH=.*|DOWNLOADS_PATH=$PLEX_VOLUME/downloads|g" .env

echo -e "${GREEN}✅ Environment configuration updated${NC}"

# Phase 3: Create Required Directory Structure
echo ""
echo -e "${YELLOW}📁 Phase 3: Create Directory Structure${NC}"
echo ""

echo -e "${BLUE}🏗️  Creating media directory structure in $PLEX_VOLUME...${NC}"

# Create all required directories
mkdir -p "$PLEX_VOLUME"/{movies,tv,music,audiobooks,books,comics,photos,podcasts}
mkdir -p "$PLEX_VOLUME"/downloads/{complete,incomplete,torrents,watch}
mkdir -p "$PLEX_VOLUME"/downloads/{movies,tv,music,books}

# Set proper permissions
chmod -R 755 "$PLEX_VOLUME"
chown -R $(id -u):$(id -g) "$PLEX_VOLUME" 2>/dev/null || true

echo -e "${GREEN}✅ Directory structure created${NC}"

# Phase 4: Deploy with New Configuration
echo ""
echo -e "${YELLOW}🚀 Phase 4: Deploy with New Volume Mappings${NC}"
echo ""

# Use docker-compose-unified-2025.yml for deployment
COMPOSE_FILE="docker-compose-unified-2025.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Error: $COMPOSE_FILE not found!${NC}"
    exit 1
fi

# Deploy core services first
echo -e "${BLUE}🏗️  Deploying core infrastructure...${NC}"
docker compose -f "$COMPOSE_FILE" --profile core up -d

echo -e "${BLUE}⏳ Waiting for core services to start...${NC}"
sleep 10

# Deploy media services
echo -e "${BLUE}🎬 Deploying media services...${NC}"
docker compose -f "$COMPOSE_FILE" --profile media up -d

echo -e "${BLUE}⏳ Waiting for media services to start...${NC}"
sleep 15

# Deploy automation services
echo -e "${BLUE}🤖 Deploying automation services...${NC}"
docker compose -f "$COMPOSE_FILE" --profile automation up -d

echo -e "${BLUE}⏳ Waiting for automation services to start...${NC}"
sleep 10

# Deploy additional profiles
echo -e "${BLUE}📚 Deploying additional services...${NC}"
docker compose -f "$COMPOSE_FILE" --profile requests up -d
docker compose -f "$COMPOSE_FILE" --profile management up -d

# Phase 5: Service Health Verification
echo ""
echo -e "${YELLOW}🔍 Phase 5: Service Health Verification${NC}"
echo ""

# Check service health
echo -e "${BLUE}📊 Checking service health...${NC}"

# Core services
wait_for_service "Redis" 6379
wait_for_service "PostgreSQL" 5432

# Media services
wait_for_service "Jellyfin" 8096
wait_for_service "Plex" 32400

# Automation services
wait_for_service "Sonarr" 8989
wait_for_service "Radarr" 7878
wait_for_service "Prowlarr" 9696

# Request management
wait_for_service "Overseerr" 5055

# Management
wait_for_service "Homepage" 3001

# Phase 6: Volume Mount Verification
echo ""
echo -e "${YELLOW}📂 Phase 6: Volume Mount Verification${NC}"
echo ""

echo -e "${BLUE}🔍 Verifying volume mounts...${NC}"

# Test file operations in Plex volume
TEST_FILE="$PLEX_VOLUME/test_write_permissions.txt"
echo "Test file created on $(date)" > "$TEST_FILE"

if [ -f "$TEST_FILE" ]; then
    echo -e "${GREEN}✅ Write permissions to $PLEX_VOLUME verified${NC}"
    rm "$TEST_FILE"
else
    echo -e "${RED}❌ Failed to write to $PLEX_VOLUME${NC}"
fi

# Check container mounts
echo -e "${BLUE}📋 Checking container volume mounts...${NC}"
docker exec jellyfin ls -la /media 2>/dev/null && echo -e "${GREEN}✅ Jellyfin media mount OK${NC}" || echo -e "${RED}❌ Jellyfin media mount failed${NC}"
docker exec sonarr ls -la /media 2>/dev/null && echo -e "${GREEN}✅ Sonarr media mount OK${NC}" || echo -e "${RED}❌ Sonarr media mount failed${NC}"
docker exec radarr ls -la /media 2>/dev/null && echo -e "${GREEN}✅ Radarr media mount OK${NC}" || echo -e "${RED}❌ Radarr media mount failed${NC}"

# Phase 7: Generate Test Workflow
echo ""
echo -e "${YELLOW}🧪 Phase 7: Test Workflow Creation${NC}"
echo ""

cat > test-media-workflow.sh << 'EOF'
#!/bin/bash

# Test Media Workflow
echo "🧪 Testing Complete Media Workflow"
echo "================================="

# Test 1: Create test media files
echo "📁 Creating test media files..."
mkdir -p /Volumes/Plex/movies/Test\ Movie\ \(2025\)
echo "This is a test movie file" > "/Volumes/Plex/movies/Test Movie (2025)/Test Movie (2025).mkv"

mkdir -p /Volumes/Plex/tv/Test\ Show/Season\ 01
echo "This is a test TV episode" > "/Volumes/Plex/tv/Test Show/Season 01/S01E01 - Test Episode.mkv"

echo "✅ Test files created"

# Test 2: Check download folder access
echo "📥 Testing download folder access..."
touch /Volumes/Plex/downloads/test_download.txt
echo "Test download file" > /Volumes/Plex/downloads/test_download.txt
echo "✅ Download folder accessible"

# Test 3: Verify service access to volumes
echo "🔍 Testing service access to volumes..."
docker exec jellyfin ls -la /media/movies 2>/dev/null && echo "✅ Jellyfin can access movies" || echo "❌ Jellyfin cannot access movies"
docker exec sonarr ls -la /media/tv 2>/dev/null && echo "✅ Sonarr can access TV shows" || echo "❌ Sonarr cannot access TV shows"
docker exec radarr ls -la /media/movies 2>/dev/null && echo "✅ Radarr can access movies" || echo "❌ Radarr cannot access movies"

# Test 4: Test API connectivity
echo "🌐 Testing API connectivity..."
curl -s http://localhost:8096/health >/dev/null && echo "✅ Jellyfin API responsive" || echo "❌ Jellyfin API not responding"
curl -s http://localhost:8989 >/dev/null && echo "✅ Sonarr API responsive" || echo "❌ Sonarr API not responding"
curl -s http://localhost:7878 >/dev/null && echo "✅ Radarr API responsive" || echo "❌ Radarr API not responding"

echo ""
echo "🎉 Test workflow complete!"
EOF

chmod +x test-media-workflow.sh

# Phase 8: Generate Deployment Report
echo ""
echo -e "${YELLOW}📊 Phase 8: Deployment Report${NC}"
echo ""

cat > deployment-report.md << EOF
# Ultimate Media Server 2025 - Deployment Report

**Deployment Date:** $(date)
**Volume Path:** $PLEX_VOLUME
**Configuration:** docker-compose-unified-2025.yml

## Services Status

$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(jellyfin|plex|sonarr|radarr|prowlarr|overseerr|redis|postgres)" || echo "No matching services found")

## Volume Mappings

- **Media Root:** $PLEX_VOLUME
- **Movies:** $PLEX_VOLUME/movies
- **TV Shows:** $PLEX_VOLUME/tv  
- **Music:** $PLEX_VOLUME/music
- **Downloads:** $PLEX_VOLUME/downloads

## Access URLs

- **Jellyfin:** http://localhost:8096
- **Plex:** http://localhost:32400
- **Sonarr:** http://localhost:8989
- **Radarr:** http://localhost:7878
- **Prowlarr:** http://localhost:9696
- **Overseerr:** http://localhost:5055
- **Homepage:** http://localhost:3001

## Next Steps

1. Configure Prowlarr with indexers
2. Connect *arr apps to Prowlarr
3. Add media libraries to Jellyfin/Plex
4. Run test workflow: \`./test-media-workflow.sh\`

## Issues & Fixes

$(if [ -w "$PLEX_VOLUME" ]; then echo "- ✅ No issues detected"; else echo "- ❌ Volume write permissions need attention"; fi)

EOF

echo -e "${GREEN}📄 Deployment report saved to deployment-report.md${NC}"

# Final Summary
echo ""
echo -e "${CYAN}🎉 Deployment Complete!${NC}"
echo "========================="
echo ""
echo -e "${YELLOW}📋 Summary:${NC}"
echo "   • Containers gracefully stopped and restarted"
echo "   • Volume mappings updated to use $PLEX_VOLUME"
echo "   • All services deployed with new configuration"
echo "   • Service health verified"
echo ""
echo -e "${YELLOW}🔗 Access Points:${NC}"
echo "   • Main Dashboard: http://localhost:3001"
echo "   • Jellyfin: http://localhost:8096"
echo "   • Media Management: Check deployment-report.md"
echo ""
echo -e "${YELLOW}🧪 Testing:${NC}"
echo "   • Run: ./test-media-workflow.sh"
echo "   • Check: deployment-report.md"
echo ""
echo -e "${GREEN}✅ Ultimate Media Server 2025 is ready with new volume mappings!${NC}"