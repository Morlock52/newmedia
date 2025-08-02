#!/bin/bash

# ============================================================================
# Media Server Configuration Launcher
# ============================================================================
# This script handles container conflicts and runs the configuration
# ============================================================================

set -e

echo "🚀 Media Server Configuration Launcher"
echo "====================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Change to script directory
cd "$(dirname "$0")"

# Step 1: Check Docker
echo -e "${BLUE}1️⃣ Checking Docker status...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running!${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Step 2: Fix any container conflicts
echo -e "${BLUE}2️⃣ Checking for container conflicts...${NC}"
if [ -f "scripts/fix-container-conflicts.sh" ]; then
    ./scripts/fix-container-conflicts.sh
else
    echo -e "${YELLOW}⚠️  Container conflict checker not found, skipping...${NC}"
fi
echo ""

# Step 3: Choose configuration method
echo -e "${BLUE}3️⃣ Choose configuration method:${NC}"
echo "   1) Smart Auto-Configure (Recommended) - Handles existing containers intelligently"
echo "   2) Quick Setup - Fast configuration for essential services only"
echo "   3) Full Auto-Configure - Complete configuration of all services"
echo "   4) Manual Configuration - Just start services, configure manually"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${PURPLE}Running Smart Auto-Configure...${NC}"
        echo ""
        if [ -f "scripts/smart-auto-configure.sh" ]; then
            ./scripts/smart-auto-configure.sh
        else
            echo -e "${RED}❌ smart-auto-configure.sh not found${NC}"
            exit 1
        fi
        ;;
    2)
        echo -e "${PURPLE}Running Quick Setup...${NC}"
        echo ""
        if [ -f "scripts/quick-setup.sh" ]; then
            ./scripts/quick-setup.sh
        else
            echo -e "${RED}❌ quick-setup.sh not found${NC}"
            exit 1
        fi
        ;;
    3)
        echo -e "${PURPLE}Running Full Auto-Configure...${NC}"
        echo ""
        if [ -f "scripts/auto-configure-all-services.sh" ]; then
            ./scripts/auto-configure-all-services.sh
        else
            echo -e "${RED}❌ auto-configure-all-services.sh not found${NC}"
            exit 1
        fi
        ;;
    4)
        echo -e "${PURPLE}Starting services only...${NC}"
        echo ""
        # Find compose file
        if [ -f "docker-compose-demo.yml" ]; then
            COMPOSE_FILE="docker-compose-demo.yml"
        elif [ -f "docker-compose.yml" ]; then
            COMPOSE_FILE="docker-compose.yml"
        else
            echo -e "${RED}❌ No docker-compose file found${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Starting all services...${NC}"
        docker-compose -f "$COMPOSE_FILE" up -d
        
        echo ""
        echo -e "${GREEN}✅ Services started!${NC}"
        echo -e "${YELLOW}Configure them manually at:${NC}"
        echo "  - Homarr: http://localhost:7575"
        echo "  - Jellyfin: http://localhost:8096"
        echo "  - Prowlarr: http://localhost:9696"
        echo "  - Sonarr: http://localhost:8989"
        echo "  - Radarr: http://localhost:7878"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Configuration complete!${NC}"
echo ""
echo -e "${CYAN}📌 Quick Links:${NC}"
echo "  - Service Status: ./service-status.html"
echo "  - Environment Manager: ./env-settings-manager.html"
echo "  - Container Status: docker ps"
echo "  - View Logs: docker logs <container_name>"
echo ""
echo -e "${YELLOW}💡 Troubleshooting:${NC}"
echo "  - If a service fails: docker logs <service_name>"
echo "  - To restart a service: docker restart <service_name>"
echo "  - To recreate a service: docker-compose up -d <service_name>"
echo "  - For help: Check ./docs/ or GitHub issues"
echo ""