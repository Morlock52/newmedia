#!/bin/bash

echo "🔧 Resetting Claude Desktop MCP Connection"
echo "========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\n${YELLOW}Step 1: Backing up current config...${NC}"
if [ -f "$HOME/.claude/claude_desktop_config.json" ]; then
    cp "$HOME/.claude/claude_desktop_config.json" "$HOME/.claude/claude_desktop_config.backup-$(date +%Y%m%d-%H%M%S).json"
    echo -e "${GREEN}✅ Backup created${NC}"
fi

echo -e "\n${YELLOW}Step 2: Creating fresh configuration...${NC}"
mkdir -p "$HOME/.claude"

cat > "$HOME/.claude/claude_desktop_config.json" << 'EOF'
{
  "mcpServers": {
    "media-server": {
      "command": "node",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js"],
      "env": {
        "DEBUG": "false"
      }
    },
    "sonarr": {
      "command": "node",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js"],
      "env": {
        "DEBUG": "false",
        "SONARR_URL": "http://localhost:8989",
        "SONARR_API_KEY": ""
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Fresh configuration created${NC}"

echo -e "\n${YELLOW}Step 3: Testing MCP servers...${NC}"

# Test media server
echo -n "Testing media-server... "
if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node /Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js 2>/dev/null | grep -q "serverInfo"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test Sonarr
echo -n "Testing sonarr... "
if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node /Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js 2>/dev/null | grep -q "sonarr-mcp"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo -e "\n${GREEN}===================================${NC}"
echo -e "${GREEN}✅ MCP RESET COMPLETE${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT - Next Steps:${NC}"
echo ""
echo "1. ${BLUE}Close Claude Desktop completely${NC}"
echo "   - Right-click Claude in dock → Quit"
echo "   - Or press Cmd+Q when Claude is active"
echo ""
echo "2. ${BLUE}Wait 5-10 seconds${NC}"
echo ""
echo "3. ${BLUE}Open Claude Desktop again${NC}"
echo ""
echo "4. ${BLUE}Test by typing:${NC}"
echo '   "What MCP tools do you have available?"'
echo ""
echo -e "${GREEN}Expected response:${NC}"
echo "   • Media server tools (4 tools)"
echo "   • Sonarr tools (6 tools)"
echo ""
echo -e "${YELLOW}If tools don't appear:${NC}"
echo "1. Open Claude Desktop Developer Console:"
echo "   View → Developer Tools → Console"
echo "2. Look for errors containing 'mcp' or 'server'"
echo "3. Share any errors for troubleshooting"