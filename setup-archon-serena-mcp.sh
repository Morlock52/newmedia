#!/bin/bash

# Setup Archon and Serena MCP Servers
# Author: MCP Integration Team
# Date: August 15, 2025

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔧 Setting up Archon and Serena MCP Servers...${NC}\n"

# 1. Check if Archon containers are running
echo -e "${BLUE}Checking Archon containers...${NC}"
if docker ps | grep -q "Archon"; then
    echo -e "${GREEN}✅ Archon containers are running${NC}"
    docker ps | grep Archon
else
    echo -e "${YELLOW}⚠️ Archon containers not found. Starting Archon...${NC}"
    
    # Check if Archon is installed
    if [ -d ~/archon ]; then
        cd ~/archon
        docker-compose up -d
    else
        echo -e "${RED}❌ Archon not found. Installing...${NC}"
        git clone https://github.com/Archon-AI/archon.git ~/archon
        cd ~/archon
        docker-compose up -d
    fi
fi

# 2. Test Archon MCP endpoint
echo -e "\n${BLUE}Testing Archon MCP endpoint...${NC}"
if curl -s http://localhost:8051/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Archon MCP is healthy${NC}"
else
    echo -e "${YELLOW}⚠️ Archon MCP not responding. Checking logs...${NC}"
    docker logs Archon-MCP --tail 20
fi

# 3. Install Serena if not installed
echo -e "\n${BLUE}Setting up Serena...${NC}"
if command -v uvx &> /dev/null; then
    echo -e "${GREEN}✅ uvx is installed${NC}"
else
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Test Serena
echo -e "${BLUE}Testing Serena MCP...${NC}"
uvx --from git+https://github.com/oraios/serena serena --version 2>/dev/null || {
    echo -e "${YELLOW}Installing Serena...${NC}"
    pip install git+https://github.com/oraios/serena
}

# 4. Update Claude Desktop configuration
echo -e "\n${BLUE}Updating Claude Desktop configuration...${NC}"

CONFIG_FILE="$HOME/Library/Application Support/Claude/claude_desktop_config.json"

# Create backup
cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true

# Update configuration
cat > "$CONFIG_FILE" << 'EOF'
{
  "mcpServers": {
    "archon": {
      "name": "archon",
      "transport": "http",
      "url": "http://localhost:8051/mcp",
      "description": "Archon AI Knowledge Management"
    },
    "serena": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server"],
      "description": "Serena Voice & Audio Processing"
    },
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start"],
      "env": {
        "OPENAI_API_KEY": ""
      },
      "description": "Claude Flow Swarm Orchestration"
    },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"],
      "description": "RUV Swarm Coordination"
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration updated${NC}"

# 5. Test MCP connections
echo -e "\n${BLUE}Testing MCP connections...${NC}"

# Test Archon
echo -e "${CYAN}Testing Archon...${NC}"
curl -s -X POST http://localhost:8051/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}' || echo "Archon test failed"

# 6. Restart Claude Desktop
echo -e "\n${YELLOW}⚠️ You need to restart Claude Desktop to apply changes${NC}"
echo -e "${CYAN}Steps to complete setup:${NC}"
echo "1. Quit Claude Desktop completely (Cmd+Q)"
echo "2. Start Claude Desktop again"
echo "3. Check the MCP icon in Claude Desktop - you should see Archon and Serena"
echo ""
echo -e "${GREEN}Available MCP Servers:${NC}"
echo "  • Archon - Knowledge management and task tracking"
echo "  • Serena - Voice and audio processing"
echo "  • Claude Flow - Swarm orchestration"
echo "  • RUV Swarm - Distributed coordination"

# 7. Display Archon access info
echo -e "\n${BLUE}Archon Services:${NC}"
echo "  • Archon UI:     http://localhost:3737"
echo "  • Archon Server: http://localhost:8181"
echo "  • Archon MCP:    http://localhost:8051"
echo "  • Archon Agents: http://localhost:8052"

echo -e "\n${GREEN}✅ Setup complete!${NC}"