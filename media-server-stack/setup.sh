#!/bin/bash

# Media Server Stack Environment Setup
# This script helps configure environment variables for the media server stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Media Server Stack Environment Setup${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo -e "${YELLOW}Please install Node.js (version 18 or higher) to continue${NC}"
    echo -e "${YELLOW}Visit: https://nodejs.org/${NC}"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ Node.js version 18 or higher is required${NC}"
    echo -e "${YELLOW}Current version: $(node -v)${NC}"
    echo -e "${YELLOW}Please upgrade Node.js and try again${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Node.js $(node -v) detected${NC}"

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    echo -e "${YELLOW}Please install npm to continue${NC}"
    exit 1
fi

echo -e "${GREEN}✓ npm $(npm -v) detected${NC}\n"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    npm install
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}\n"
fi

# Run the environment setup tool
echo -e "${BLUE}🚀 Starting environment configuration tool...${NC}\n"
node setup-env.js

# Check if .env was created successfully
if [ -f ".env" ]; then
    echo -e "\n${GREEN}✅ Environment configuration completed!${NC}"
    echo -e "${BLUE}Your .env file has been created with the following variables:${NC}"
    echo -e "${YELLOW}$(grep -c "^[A-Z]" .env) environment variables configured${NC}"
    
    # Show next steps
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "${GREEN}1.${NC} Review your .env file: ${YELLOW}cat .env${NC}"
    echo -e "${GREEN}2.${NC} Start the stack: ${YELLOW}cd compose && docker compose up -d${NC}"
    echo -e "${GREEN}3.${NC} Access Web UI: ${YELLOW}http://localhost:3000${NC}"
    
else
    echo -e "\n${RED}❌ Environment setup failed${NC}"
    echo -e "${YELLOW}Please check the error messages above and try again${NC}"
    exit 1
fi
