#!/bin/bash

# Ultimate Media Server 2025 - Troubleshooting Script
# This script helps diagnose and fix common deployment issues

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo "🔍 Ultimate Media Server 2025 - Troubleshooting"
echo "=============================================="
echo ""

# Check Docker status
echo -e "${BLUE}🐳 Checking Docker status...${NC}"
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running or not installed${NC}"
    echo "   Please start Docker Desktop or install Docker"
    exit 1
else
    echo -e "${GREEN}✅ Docker is running${NC}"
    docker version --format 'Docker version {{.Server.Version}}'
fi

# Check for problematic containers
echo ""
echo -e "${BLUE}🔍 Checking for failed containers...${NC}"
FAILED_CONTAINERS=$(docker ps -a --filter "status=exited" --filter "status=created" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | tail -n +2)

if [ -n "$FAILED_CONTAINERS" ]; then
    echo -e "${YELLOW}⚠️  Found containers that aren't running:${NC}"
    echo "$FAILED_CONTAINERS"
    echo ""
    
    # Offer to remove failed containers
    read -p "Remove failed containers? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker ps -a --filter "status=exited" --filter "status=created" -q | xargs -r docker rm
        echo -e "${GREEN}✅ Failed containers removed${NC}"
    fi
else
    echo -e "${GREEN}✅ No failed containers found${NC}"
fi

# Check for image pull errors
echo ""
echo -e "${BLUE}🔍 Checking for problematic images...${NC}"

# List of known problematic images
PROBLEMATIC_IMAGES=(
    "ghcr.io/media-platform/ai-recommendations"
    "quantum-security"
    "neural-dashboard"
    "ml-processor"
)

for image in "${PROBLEMATIC_IMAGES[@]}"; do
    if docker images | grep -q "$image"; then
        echo -e "${YELLOW}⚠️  Found problematic image: $image${NC}"
        read -p "Remove this image? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker rmi "$image" 2>/dev/null || true
            echo -e "${GREEN}✅ Removed $image${NC}"
        fi
    fi
done

# Check disk space
echo ""
echo -e "${BLUE}💾 Checking disk space...${NC}"
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo -e "${RED}❌ Low disk space (${DISK_USAGE}% used)${NC}"
    echo "   Consider cleaning up Docker resources:"
    echo "   docker system prune -a"
else
    echo -e "${GREEN}✅ Disk space OK (${DISK_USAGE}% used)${NC}"
fi

# Check for port conflicts
echo ""
echo -e "${BLUE}🔌 Checking for port conflicts...${NC}"
PORTS=(8096 8989 7878 9696 8080 8081 5055 8181 3000 9000 3001)
CONFLICTS=0

for port in "${PORTS[@]}"; do
    if lsof -i ":$port" >/dev/null 2>&1; then
        SERVICE=$(lsof -i ":$port" | awk 'NR==2 {print $1}')
        if [[ "$SERVICE" != "com.docke" ]] && [[ "$SERVICE" != "Docker" ]]; then
            echo -e "${YELLOW}⚠️  Port $port is in use by: $SERVICE${NC}"
            CONFLICTS=$((CONFLICTS + 1))
        fi
    fi
done

if [ $CONFLICTS -eq 0 ]; then
    echo -e "${GREEN}✅ No port conflicts detected${NC}"
fi

# Check Docker Compose file
echo ""
echo -e "${BLUE}📄 Checking Docker Compose files...${NC}"
if [ -f "docker-compose-simplified-2025.yml" ]; then
    echo -e "${GREEN}✅ Simplified compose file found${NC}"
    
    # Validate the compose file
    if docker compose -f docker-compose-simplified-2025.yml config >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Compose file is valid${NC}"
    else
        echo -e "${RED}❌ Compose file has errors${NC}"
        docker compose -f docker-compose-simplified-2025.yml config 2>&1 | head -10
    fi
else
    echo -e "${YELLOW}⚠️  Simplified compose file not found${NC}"
fi

# Check for common issues
echo ""
echo -e "${BLUE}🔧 Checking for common issues...${NC}"

# Check if running on Mac with ARM processor
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        echo -e "${YELLOW}⚠️  Running on Apple Silicon (M1/M2)${NC}"
        echo "   Some containers may need platform specification"
        echo "   The simplified deployment handles this automatically"
    fi
fi

# Check for VPN interference
if docker network ls | grep -q "vpn"; then
    echo -e "${YELLOW}⚠️  VPN network detected${NC}"
    echo "   This might interfere with container networking"
fi

# Generate diagnostic report
echo ""
echo -e "${BLUE}📊 Generating diagnostic report...${NC}"

cat > deployment-diagnostic.txt << EOF
Ultimate Media Server 2025 - Diagnostic Report
Generated: $(date)

System Information:
- OS: $(uname -s) $(uname -m)
- Docker: $(docker version --format '{{.Server.Version}}')
- Compose: $(docker compose version 2>/dev/null || echo "Not found")

Running Containers:
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20)

Failed Containers:
$(docker ps -a --filter "status=exited" --filter "status=created" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}")

Docker Networks:
$(docker network ls)

Disk Usage:
$(df -h /)

Memory Usage:
$(free -h 2>/dev/null || vm_stat 2>/dev/null || echo "Not available")
EOF

echo -e "${GREEN}✅ Diagnostic report saved to: deployment-diagnostic.txt${NC}"

# Provide recommendations
echo ""
echo -e "${PURPLE}💡 Recommendations:${NC}"
echo ""

if [ $CONFLICTS -gt 0 ]; then
    echo "1. Stop conflicting services or change ports in .env file"
fi

echo "1. Use the simplified deployment: ./deploy-simplified-2025.sh"
echo "2. Deploy services in phases to identify issues"
echo "3. Check container logs: docker logs [container-name]"
echo "4. Monitor resources: docker stats"
echo "5. Clean up if needed: docker system prune -a"
echo ""
echo -e "${CYAN}📚 Quick Fixes:${NC}"
echo "   Restart all:     docker compose -f docker-compose-simplified-2025.yml restart"
echo "   View logs:       docker compose -f docker-compose-simplified-2025.yml logs -f"
echo "   Clean start:     docker compose -f docker-compose-simplified-2025.yml down && ./deploy-simplified-2025.sh"
echo ""

echo -e "${GREEN}✅ Troubleshooting complete!${NC}"