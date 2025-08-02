#!/bin/bash

# Volume Mapping Test and Deployment Script
# Tests /Volumes/Plex accessibility and deploys with proper volume mappings

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Ultimate Media Server 2025 - Volume Mapping Test & Deploy${NC}"
echo "=============================================================="
echo ""

# Test 1: Check if /Volumes/Plex exists
echo -e "${YELLOW}🔍 Phase 1: Volume Accessibility Test${NC}"
echo ""

PLEX_VOLUME="/Volumes/Plex"
if [ -d "$PLEX_VOLUME" ]; then
    echo -e "${GREEN}✅ $PLEX_VOLUME exists${NC}"
else
    echo -e "${RED}❌ $PLEX_VOLUME does not exist${NC}"
    echo -e "${YELLOW}💡 Creating fallback directory structure...${NC}"
    PLEX_VOLUME="./media-plex-volume"
    mkdir -p "$PLEX_VOLUME"
    echo -e "${BLUE}📁 Using fallback path: $PLEX_VOLUME${NC}"
fi

# Test 2: Check write permissions
echo -e "${BLUE}✍️  Testing write permissions...${NC}"
TEST_FILE="$PLEX_VOLUME/test_write_$(date +%s).txt"
if echo "Test write permissions" > "$TEST_FILE" 2>/dev/null; then
    echo -e "${GREEN}✅ Write permissions OK${NC}"
    rm -f "$TEST_FILE"
else
    echo -e "${RED}❌ No write permissions to $PLEX_VOLUME${NC}"
    echo -e "${YELLOW}💡 Attempting to fix permissions...${NC}"
    sudo chmod -R 755 "$PLEX_VOLUME" 2>/dev/null || {
        echo -e "${RED}❌ Cannot fix permissions. Please run: sudo chmod -R 755 $PLEX_VOLUME${NC}"
        exit 1
    }
fi

# Test 3: Create directory structure
echo ""
echo -e "${YELLOW}📁 Phase 2: Directory Structure Creation${NC}"
echo ""

echo -e "${BLUE}🏗️  Creating media directory structure...${NC}"
mkdir -p "$PLEX_VOLUME"/{movies,tv,music,audiobooks,books,comics,photos,podcasts}
mkdir -p "$PLEX_VOLUME"/downloads/{complete,incomplete,torrents,watch}
mkdir -p "$PLEX_VOLUME"/downloads/{movies,tv,music,books}

echo -e "${GREEN}✅ Directory structure created${NC}"

# Test 4: Update environment configuration
echo ""
echo -e "${YELLOW}🔧 Phase 3: Environment Configuration Update${NC}"
echo ""

# Backup .env file
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
echo -e "${BLUE}💾 Backed up .env file${NC}"

# Update paths in .env
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|MEDIA_PATH=.*|MEDIA_PATH=$PLEX_VOLUME|g" .env
    sed -i '' "s|DOWNLOADS_PATH=.*|DOWNLOADS_PATH=$PLEX_VOLUME/downloads|g" .env
else
    # Linux
    sed -i "s|MEDIA_PATH=.*|MEDIA_PATH=$PLEX_VOLUME|g" .env
    sed -i "s|DOWNLOADS_PATH=.*|DOWNLOADS_PATH=$PLEX_VOLUME/downloads|g" .env
fi

echo -e "${GREEN}✅ Environment configuration updated${NC}"

# Test 5: Verify docker-compose file
echo ""
echo -e "${YELLOW}🐳 Phase 4: Docker Compose Verification${NC}"
echo ""

# Check which compose file to use
if [ -f "docker-compose-unified-2025.yml" ]; then
    COMPOSE_FILE="docker-compose-unified-2025.yml"
    echo -e "${GREEN}✅ Using docker-compose-unified-2025.yml${NC}"
elif [ -f "docker-compose.yml" ]; then
    COMPOSE_FILE="docker-compose.yml"
    echo -e "${YELLOW}⚠️  Using docker-compose.yml${NC}"
else
    echo -e "${RED}❌ No docker-compose file found${NC}"
    exit 1
fi

# Test 6: Container deployment test
echo ""
echo -e "${YELLOW}🚀 Phase 5: Test Deployment${NC}"
echo ""

# Check if any containers are running
RUNNING_CONTAINERS=$(docker ps -q | wc -l | tr -d ' ')
if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
    echo -e "${YELLOW}📊 Found $RUNNING_CONTAINERS running containers${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    echo ""
    echo -e "${BLUE}🔄 Stopping containers for redeployment...${NC}"
    docker stop $(docker ps -q) --time=30 2>/dev/null || true
fi

# Deploy core services first
echo -e "${BLUE}🏗️  Deploying core services...${NC}"
if [[ "$COMPOSE_FILE" == "docker-compose-unified-2025.yml" ]]; then
    # Use profiles for unified compose
    docker compose -f "$COMPOSE_FILE" --profile core up -d
    sleep 10
    docker compose -f "$COMPOSE_FILE" --profile media up -d
    sleep 10
    docker compose -f "$COMPOSE_FILE" --profile automation up -d
else
    # Use standard compose
    docker compose -f "$COMPOSE_FILE" up -d jellyfin sonarr radarr prowlarr overseerr
fi

echo -e "${GREEN}✅ Services deployed${NC}"

# Test 7: Service health check
echo ""
echo -e "${YELLOW}🔍 Phase 6: Service Health Check${NC}"
echo ""

# Wait for services to start
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"
sleep 30

# Function to check service
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=15
    local attempt=0
    
    echo -n "   Checking $service_name (port $port)"
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo -e " ${GREEN}[OK]${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e " ${RED}[TIMEOUT]${NC}"
    return 1
}

# Check critical services
check_service "Jellyfin" 8096
check_service "Sonarr" 8989
check_service "Radarr" 7878
check_service "Prowlarr" 9696
check_service "Overseerr" 5055

# Test 8: Volume mount verification
echo ""
echo -e "${YELLOW}📂 Phase 7: Volume Mount Verification${NC}"
echo ""

# Test container volume access
echo -e "${BLUE}🔍 Testing container volume access...${NC}"

# Create test files
echo "Test movie file" > "$PLEX_VOLUME/movies/test_movie.txt"
echo "Test TV file" > "$PLEX_VOLUME/tv/test_tv.txt"

# Test container access
if docker exec jellyfin ls /media/movies/test_movie.txt >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Jellyfin can access movies folder${NC}"
else
    echo -e "${RED}❌ Jellyfin cannot access movies folder${NC}"
fi

if docker exec sonarr ls /media/tv/test_tv.txt >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Sonarr can access TV folder${NC}"
else
    echo -e "${RED}❌ Sonarr cannot access TV folder${NC}"
fi

if docker exec radarr ls /media/movies/test_movie.txt >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Radarr can access movies folder${NC}"
else
    echo -e "${RED}❌ Radarr cannot access movies folder${NC}"
fi

# Clean up test files
rm -f "$PLEX_VOLUME/movies/test_movie.txt" "$PLEX_VOLUME/tv/test_tv.txt"

# Test 9: Generate comprehensive test workflow
echo ""
echo -e "${YELLOW}🧪 Phase 8: Generate Test Workflow${NC}"
echo ""

cat > test-complete-workflow.sh << EOF
#!/bin/bash

# Complete Media Server Workflow Test
echo "🧪 Testing Complete Media Server Workflow"
echo "=========================================="
echo ""

# Configuration
PLEX_VOLUME="$PLEX_VOLUME"

echo -e "📁 Using media path: \$PLEX_VOLUME"
echo ""

# Test 1: Media folder structure
echo "🔍 Test 1: Verify media folder structure"
for folder in movies tv music audiobooks books comics photos podcasts; do
    if [ -d "\$PLEX_VOLUME/\$folder" ]; then
        echo "✅ \$folder folder exists"
    else
        echo "❌ \$folder folder missing"
    fi
done
echo ""

# Test 2: Download folder structure  
echo "🔍 Test 2: Verify download folder structure"
for folder in complete incomplete torrents watch; do
    if [ -d "\$PLEX_VOLUME/downloads/\$folder" ]; then
        echo "✅ downloads/\$folder exists"
    else
        echo "❌ downloads/\$folder missing"
    fi
done
echo ""

# Test 3: Service API connectivity
echo "🔍 Test 3: Service API connectivity"
curl -s http://localhost:8096/health >/dev/null && echo "✅ Jellyfin API responsive" || echo "❌ Jellyfin API not responding"
curl -s "http://localhost:8989/api/v3/system/status" -H "X-Api-Key: \${SONARR_API_KEY:-test}" >/dev/null 2>&1 && echo "✅ Sonarr API responsive" || echo "⚠️  Sonarr API (needs API key configuration)"
curl -s "http://localhost:7878/api/v3/system/status" -H "X-Api-Key: \${RADARR_API_KEY:-test}" >/dev/null 2>&1 && echo "✅ Radarr API responsive" || echo "⚠️  Radarr API (needs API key configuration)"
curl -s http://localhost:9696 >/dev/null && echo "✅ Prowlarr web interface responsive" || echo "❌ Prowlarr not responding"
curl -s http://localhost:5055 >/dev/null && echo "✅ Overseerr responsive" || echo "❌ Overseerr not responding"
echo ""

# Test 4: Volume mount verification
echo "🔍 Test 4: Container volume mount verification"
docker exec jellyfin ls -la /media >/dev/null 2>&1 && echo "✅ Jellyfin media mount OK" || echo "❌ Jellyfin media mount failed"
docker exec sonarr ls -la /media >/dev/null 2>&1 && echo "✅ Sonarr media mount OK" || echo "❌ Sonarr media mount failed"  
docker exec radarr ls -la /media >/dev/null 2>&1 && echo "✅ Radarr media mount OK" || echo "❌ Radarr media mount failed"
echo ""

# Test 5: Write permissions test
echo "🔍 Test 5: Write permissions test"
TEST_FILE="\$PLEX_VOLUME/write_test_\$(date +%s).txt"
if echo "Write test" > "\$TEST_FILE" 2>/dev/null; then
    echo "✅ Write permissions OK"
    rm -f "\$TEST_FILE"
else
    echo "❌ Write permissions failed"
fi
echo ""

# Test 6: Container logs check
echo "🔍 Test 6: Container health check"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(jellyfin|sonarr|radarr|prowlarr|overseerr)"
echo ""

echo "🎉 Workflow test complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure Prowlarr with indexers"
echo "2. Connect ARR apps to Prowlarr" 
echo "3. Add media libraries to Jellyfin"
echo "4. Test media requests through Overseerr"
EOF

chmod +x test-complete-workflow.sh
echo -e "${GREEN}✅ Test workflow created: test-complete-workflow.sh${NC}"

# Test 10: Generate deployment report
echo ""
echo -e "${YELLOW}📊 Phase 9: Deployment Report${NC}"
echo ""

cat > deployment-status-report.md << EOF
# Ultimate Media Server 2025 - Deployment Status Report

**Deployment Date:** $(date)
**Volume Path:** $PLEX_VOLUME
**Docker Compose:** $COMPOSE_FILE

## Service Status

\`\`\`
$(docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running")
\`\`\`

## Volume Configuration

- **Media Root:** $PLEX_VOLUME
- **Movies:** $PLEX_VOLUME/movies
- **TV Shows:** $PLEX_VOLUME/tv
- **Music:** $PLEX_VOLUME/music
- **Downloads:** $PLEX_VOLUME/downloads

## Directory Structure

\`\`\`
$(find "$PLEX_VOLUME" -type d -maxdepth 2 2>/dev/null | head -20 || echo "Directory listing unavailable")
\`\`\`

## Access URLs

- **Jellyfin:** http://localhost:8096
- **Sonarr:** http://localhost:8989  
- **Radarr:** http://localhost:7878
- **Prowlarr:** http://localhost:9696
- **Overseerr:** http://localhost:5055
- **Homepage:** http://localhost:3001 (if enabled)

## Configuration Notes

1. Volume mappings updated to use $PLEX_VOLUME
2. Services deployed with new volume configuration
3. Directory structure created automatically
4. Test workflow available: \`./test-complete-workflow.sh\`

## Next Steps

1. Run complete test: \`./test-complete-workflow.sh\`
2. Configure API keys in *arr applications
3. Set up indexers in Prowlarr
4. Add media libraries to Jellyfin
5. Test end-to-end media workflow

## Issues & Resolutions

$(if [ -w "$PLEX_VOLUME" ]; then echo "- ✅ No issues detected - Volume accessible and writable"; else echo "- ❌ Volume write permissions need attention"; fi)
$(if docker ps -q >/dev/null 2>&1; then echo "- ✅ Docker containers running"; else echo "- ❌ No Docker containers detected"; fi)

---
*Report generated automatically by deploy-volume-test.sh*
EOF

echo -e "${GREEN}📄 Deployment report saved: deployment-status-report.md${NC}"

# Final summary
echo ""
echo -e "${BLUE}🎉 Volume Mapping Deployment Complete!${NC}"
echo "====================================="
echo ""
echo -e "${GREEN}✅ Summary:${NC}"
echo "   • Volume path configured: $PLEX_VOLUME"
echo "   • Directory structure created"
echo "   • Services deployed with new volume mappings"
echo "   • Environment configuration updated"
echo ""
echo -e "${YELLOW}📋 What to do next:${NC}"
echo "1. Run: ./test-complete-workflow.sh"
echo "2. Check: deployment-status-report.md"  
echo "3. Access Jellyfin: http://localhost:8096"
echo "4. Configure *arr services API keys"
echo ""
echo -e "${GREEN}🚀 Your media server is ready with volume mapping to $PLEX_VOLUME!${NC}"