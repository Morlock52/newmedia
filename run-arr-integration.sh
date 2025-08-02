#!/bin/bash

# ARR Integration Runner Script
# This script sets up the Python environment and runs the ARR integration

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 ARR Services Integration Setup${NC}"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  pip3 not found, trying to install requests manually${NC}"
    python3 -c "import requests" 2>/dev/null || {
        echo -e "${RED}❌ requests library not available. Please install it with: pip3 install requests${NC}"
        exit 1
    }
else
    # Install requests if not available
    python3 -c "import requests" 2>/dev/null || {
        echo -e "${YELLOW}📦 Installing requests library...${NC}"
        pip3 install requests --quiet
    }
fi

# Make the Python script executable
chmod +x arr-integration-script.py

echo -e "${GREEN}✅ Environment ready${NC}"
echo ""

# Run the integration script
echo -e "${BLUE}🔄 Running ARR Integration...${NC}"
python3 arr-integration-script.py

echo ""
echo -e "${GREEN}🎉 Integration process completed!${NC}"