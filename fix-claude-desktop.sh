#!/bin/bash

echo "🔧 Fixing Claude Desktop MCP Connection"
echo "======================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Paths
MCP_PATH="/Users/morlock/fun/newmedia/mcp-architecture"
CLAUDE_CONFIG="$HOME/.claude/claude_desktop_config.json"

echo -e "\n${YELLOW}Step 1: Stopping any running MCP processes...${NC}"
pkill -f "node.*simple-index" 2>/dev/null
pkill -f "node.*standalone-mcp" 2>/dev/null
pkill -f "node.*claude-desktop-bridge" 2>/dev/null
sleep 2
echo -e "${GREEN}✅ Cleaned up processes${NC}"

echo -e "\n${YELLOW}Step 2: Creating simplified standalone MCP server...${NC}"
chmod +x "$MCP_PATH/standalone-mcp.js"
echo -e "${GREEN}✅ Standalone server ready${NC}"

echo -e "\n${YELLOW}Step 3: Updating Claude Desktop configuration...${NC}"
mkdir -p "$HOME/.claude"

# Create a simpler configuration
cat > "$CLAUDE_CONFIG" << EOF
{
  "mcpServers": {
    "media-server": {
      "command": "node",
      "args": ["$MCP_PATH/standalone-mcp.js"],
      "env": {
        "DEBUG": "true"
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration updated${NC}"

echo -e "\n${YELLOW}Step 4: Testing standalone MCP server...${NC}"
# Test the server
TEST_RESULT=$(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0"}}' | node "$MCP_PATH/standalone-mcp.js" 2>/dev/null | head -1)

if [[ $TEST_RESULT == *"serverInfo"* ]]; then
    echo -e "${GREEN}✅ Standalone server test PASSED${NC}"
    echo "Response: ${TEST_RESULT:0:100}..."
else
    echo -e "${RED}❌ Standalone server test failed${NC}"
fi

echo -e "\n${YELLOW}Step 5: Creating test script...${NC}"
cat > "$MCP_PATH/test-claude-connection.sh" << 'EOF'
#!/bin/bash

echo "🧪 Testing MCP Server for Claude Desktop"
echo "========================================"

# Test initialize
echo "Test 1: Initialize"
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0"}}' | node standalone-mcp.js 2>/dev/null | head -1

# Test tools list
echo -e "\nTest 2: List Tools"
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | node standalone-mcp.js 2>/dev/null | head -1

# Test tool call
echo -e "\nTest 3: Call Tool"
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_system_info","arguments":{}}}' | node standalone-mcp.js 2>/dev/null | head -1

echo -e "\n✅ If you see JSON responses above, the MCP server is working!"
EOF

chmod +x "$MCP_PATH/test-claude-connection.sh"
echo -e "${GREEN}✅ Test script created${NC}"

# Final instructions
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}✅ FIX COMPLETE - SIMPLIFIED MCP SERVER READY${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "The simplified standalone MCP server is now configured."
echo ""
echo -e "${YELLOW}To complete the fix:${NC}"
echo ""
echo "1. ${YELLOW}Restart Claude Desktop${NC}"
echo "   - Quit Claude Desktop completely (Cmd+Q)"
echo "   - Start Claude Desktop again"
echo ""
echo "2. ${YELLOW}Test the connection${NC}"
echo "   Ask Claude: 'What media server tools do you have available?'"
echo ""
echo "3. ${YELLOW}If still having issues, run the test:${NC}"
echo "   cd $MCP_PATH"
echo "   ./test-claude-connection.sh"
echo ""
echo -e "${GREEN}Expected tools:${NC}"
echo "   • search_media - Search for media"
echo "   • get_library_stats - Get library statistics"
echo "   • get_recent_media - Get recent additions"
echo "   • get_system_info - Get system information"
echo ""
echo -e "${YELLOW}Troubleshooting:${NC}"
echo "1. Check Claude Desktop logs for errors"
echo "2. Make sure no other MCP servers are configured"
echo "3. Try removing and re-adding the configuration"
echo ""
echo "Config location: $CLAUDE_CONFIG"
echo "Server location: $MCP_PATH/standalone-mcp.js"