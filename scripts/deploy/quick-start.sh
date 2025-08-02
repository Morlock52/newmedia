#!/bin/bash

# Media Server - Quick Start Script
# One-command deployment for new users
# Version: 1.0.0

set -euo pipefail

# Color codes
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Show banner
echo -e "${BLUE}"
cat << "EOF"
    __  ___         ___         _____                           
   /  |/  /__  ____/ (_)___ _  / ___/___  ______   _____  _____
  / /|_/ / _ \/ __  / / __ `/ \__ \/ _ \/ ___/ | / / _ \/ ___/
 / /  / /  __/ /_/ / / /_/ / ___/ /  __/ /   | |/ /  __/ /    
/_/  /_/\___/\__,_/_/\__,_/ /____/\___/_/    |___/\___/_/     
                                                               
EOF
echo -e "${NC}"
echo -e "${CYAN}Quick Start - Deploying your media server in one command!${NC}\n"

# Run setup
echo -e "${GREEN}Step 1/3: Running initial setup...${NC}"
"${SCRIPT_DIR}/setup.sh"

echo -e "\n${GREEN}Step 2/3: Deploying core services...${NC}"
sleep 2

# Deploy core services
"${SCRIPT_DIR}/deploy.sh" core

echo -e "\n${GREEN}Step 3/3: Running health check...${NC}"
sleep 5

# Run health check
"${SCRIPT_DIR}/health-check.sh" || true

# Show final message
echo -e "\n${GREEN}✅ Quick start completed!${NC}\n"
echo -e "${CYAN}Your media server is now running!${NC}"
echo -e "Access your services at:"
echo -e "  📊 Dashboard: ${BLUE}http://localhost:3000${NC}"
echo -e "  📺 Jellyfin:  ${BLUE}http://localhost:8096${NC}"
echo -e "  ⬇️  Downloads: ${BLUE}http://localhost:8080${NC}"
echo
echo -e "${CYAN}To deploy additional services:${NC}"
echo -e "  ./scripts/deploy/deploy.sh full"
echo
echo -e "${CYAN}For help:${NC}"
echo -e "  ./scripts/deploy/service-control.sh help"