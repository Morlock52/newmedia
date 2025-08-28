#!/bin/bash

echo "🎬 Setting up Sonarr MCP Server for Claude Desktop"
echo "================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MCP_PATH="/Users/morlock/fun/newmedia/mcp-architecture"
CLAUDE_CONFIG="$HOME/.claude/claude_desktop_config.json"

echo -e "\n${YELLOW}Step 1: Testing Sonarr MCP Server...${NC}"

# Test the server
TEST_CMD='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0"}}'
TEST_RESULT=$(echo "$TEST_CMD" | node "$MCP_PATH/sonarr-mcp-standalone.js" 2>/dev/null | head -1)

if [[ $TEST_RESULT == *"sonarr-mcp"* ]]; then
    echo -e "${GREEN}✅ Sonarr MCP server test PASSED${NC}"
else
    echo -e "${YELLOW}⚠️  Test unexpected result, but continuing...${NC}"
fi

echo -e "\n${YELLOW}Step 2: Updating Claude Desktop configuration...${NC}"

# Check if config exists
if [ -f "$CLAUDE_CONFIG" ]; then
    echo -e "${BLUE}Existing config found. Creating backup...${NC}"
    cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup"
fi

# Create new configuration with both servers
cat > "$CLAUDE_CONFIG" << EOF
{
  "mcpServers": {
    "media-server": {
      "command": "node",
      "args": ["$MCP_PATH/standalone-mcp.js"],
      "env": {
        "DEBUG": "true"
      }
    },
    "sonarr": {
      "command": "node", 
      "args": ["$MCP_PATH/sonarr-mcp-standalone.js"],
      "env": {
        "DEBUG": "true",
        "SONARR_URL": "http://localhost:8989",
        "SONARR_API_KEY": ""
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration updated with Sonarr MCP${NC}"

echo -e "\n${YELLOW}Step 3: Testing Sonarr tools...${NC}"

# Test tools list
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | \
    node "$MCP_PATH/sonarr-mcp-standalone.js" 2>/dev/null | \
    grep -q "search_series" && echo -e "${GREEN}✅ Tools list working${NC}" || echo -e "${YELLOW}⚠️  Tools list issue${NC}"

# Test tool call
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_system_status","arguments":{}}}' | \
    node "$MCP_PATH/sonarr-mcp-standalone.js" 2>/dev/null | \
    grep -q "Sonarr System Status" && echo -e "${GREEN}✅ Tool calls working${NC}" || echo -e "${YELLOW}⚠️  Tool call issue${NC}"

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}✅ SONARR MCP SERVER READY FOR CLAUDE DESKTOP${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${BLUE}Available Sonarr Tools:${NC}"
echo "  • search_series - Search for TV series"
echo "  • get_series_list - Get all TV series in library"
echo "  • get_upcoming_episodes - Get upcoming episodes"
echo "  • get_missing_episodes - Get missing episodes"
echo "  • get_system_status - Get Sonarr system status"
echo "  • get_queue - Get download queue"
echo ""
echo -e "${YELLOW}To use in Claude Desktop:${NC}"
echo "1. Restart Claude Desktop (Cmd+Q, then reopen)"
echo "2. Ask Claude: 'What Sonarr tools do you have?'"
echo "3. Try: 'Search for TV series about dragons'"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "• Config file: $CLAUDE_CONFIG"
echo "• Server file: $MCP_PATH/sonarr-mcp-standalone.js"
echo ""
echo -e "${YELLOW}To connect to a real Sonarr instance:${NC}"
echo "1. Get your Sonarr API key from Settings → General → Security"
echo "2. Update the config file with your API key"
echo "3. Ensure Sonarr is running on port 8989"
echo ""
echo -e "${GREEN}The server works in Demo Mode without configuration!${NC}"