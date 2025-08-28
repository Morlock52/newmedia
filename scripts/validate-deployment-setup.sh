#!/bin/bash

# Quick Deployment Setup Validation
# Checks current state of deployment scripts and configuration

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Media Server Deployment Validation${NC}"
echo "=================================="

# Check Docker
echo -e "\n${BLUE}Docker Status:${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker installed: $(docker --version)"
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker daemon running"
    else
        echo -e "${RED}✗${NC} Docker daemon not running"
    fi
else
    echo -e "${RED}✗${NC} Docker not installed"
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose v2: $(docker compose version)"
elif command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose v1: $(docker-compose --version)"
else
    echo -e "${RED}✗${NC} Docker Compose not found"
fi

# Check deployment scripts
echo -e "\n${BLUE}Deployment Scripts:${NC}"
SCRIPTS=(
    "install-media-server.sh"
    "install-media-server-fixed.sh"
    "scripts/deploy/deploy.sh"
    "scripts/deploy/health-check.sh"
    "scripts/deploy/quick-start.sh"
    "scripts/test-deployment.sh"
    "scripts/run-all-deployment-tests.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo -e "${GREEN}✓${NC} $script (executable)"
        else
            echo -e "${YELLOW}⚠${NC} $script (not executable)"
        fi
    else
        echo -e "${RED}✗${NC} $script (missing)"
    fi
done

# Check docker-compose files
echo -e "\n${BLUE}Docker Compose Files:${NC}"
compose_count=$(find . -name "docker-compose*.yml" -type f | wc -l)
echo "Found $compose_count docker-compose files"

# Check for issues
echo -e "\n${BLUE}Known Issues:${NC}"
if grep -q "/Users/morlock" install-media-server.sh 2>/dev/null; then
    echo -e "${RED}✗${NC} Hardcoded paths in install-media-server.sh"
else
    echo -e "${GREEN}✓${NC} No hardcoded paths in install-media-server.sh"
fi

if [ -f ".env" ]; then
    perms=$(stat -f %A .env 2>/dev/null || stat -c %a .env 2>/dev/null || echo "unknown")
    if [ "$perms" == "600" ] || [ "$perms" == "640" ]; then
        echo -e "${GREEN}✓${NC} .env has secure permissions: $perms"
    else
        echo -e "${YELLOW}⚠${NC} .env has loose permissions: $perms"
    fi
else
    echo -e "${YELLOW}⚠${NC} No .env file present"
fi

# Port availability
echo -e "\n${BLUE}Port Availability:${NC}"
ports=(8096 8989 7878 9696 8080 3000)
blocked=0
for port in "${ports[@]}"; do
    if lsof -i :$port &> /dev/null; then
        echo -e "${RED}✗${NC} Port $port is in use"
        ((blocked++))
    fi
done
if [ $blocked -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All required ports are available"
fi

# Recommendations
echo -e "\n${BLUE}Recommendations:${NC}"
echo "1. Use install-media-server-fixed.sh for new installations"
echo "2. Run ./scripts/run-all-deployment-tests.sh for full validation"
echo "3. Check TEST_REPORTS/deployment-validation-report.md for detailed findings"

echo -e "\n${GREEN}Validation complete!${NC}"