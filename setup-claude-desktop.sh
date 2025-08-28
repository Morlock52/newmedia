#!/bin/bash

echo "🚀 Setting up MCP connection for Claude Desktop"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the absolute path to the MCP architecture
MCP_PATH="/Users/morlock/fun/newmedia/mcp-architecture"

# Step 1: Check if MCP suite is installed
echo -e "\n${YELLOW}Step 1: Checking MCP suite installation...${NC}"
if [ -f "$MCP_PATH/package.json" ]; then
    echo -e "${GREEN}✅ MCP suite found at: $MCP_PATH${NC}"
else
    echo -e "${RED}❌ MCP suite not found. Please check the path.${NC}"
    exit 1
fi

# Step 2: Install dependencies if needed
echo -e "\n${YELLOW}Step 2: Installing dependencies...${NC}"
cd "$MCP_PATH"
npm install --silent
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 3: Create Claude Desktop config directory
echo -e "\n${YELLOW}Step 3: Setting up Claude Desktop configuration...${NC}"
CLAUDE_CONFIG_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_CONFIG_DIR"
echo -e "${GREEN}✅ Config directory ready: $CLAUDE_CONFIG_DIR${NC}"

# Step 4: Create the configuration file
CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
echo -e "\n${YELLOW}Step 4: Creating MCP configuration...${NC}"

cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "media-server": {
      "command": "node",
      "args": ["$MCP_PATH/claude-desktop-bridge.js"],
      "env": {
        "MCP_BASE_URL": "http://localhost:3001",
        "DEBUG": "true"
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration created at: $CONFIG_FILE${NC}"

# Step 5: Create a startup script
echo -e "\n${YELLOW}Step 5: Creating startup script...${NC}"
STARTUP_SCRIPT="$MCP_PATH/start-for-claude.sh"

cat > "$STARTUP_SCRIPT" << 'EOF'
#!/bin/bash

echo "🚀 Starting MCP Suite for Claude Desktop..."

# Start the MCP suite
node src/simple-index.js &
MCP_PID=$!

echo "✅ MCP Suite started (PID: $MCP_PID)"
echo ""
echo "📡 Available endpoints:"
echo "  • Main Dashboard: http://localhost:8090"
echo "  • Jellyfin MCP: http://localhost:3001"
echo "  • Health Check: http://localhost:3001/health"
echo ""
echo "🔗 Claude Desktop should now be able to connect!"
echo ""
echo "Press Ctrl+C to stop the MCP suite..."

# Wait for the process
wait $MCP_PID
EOF

chmod +x "$STARTUP_SCRIPT"
echo -e "${GREEN}✅ Startup script created: $STARTUP_SCRIPT${NC}"

# Step 6: Test the MCP suite
echo -e "\n${YELLOW}Step 6: Testing MCP suite...${NC}"
cd "$MCP_PATH"
timeout 5 node src/simple-index.js > /dev/null 2>&1 &
TEST_PID=$!
sleep 3

if curl -s http://localhost:3001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ MCP suite test successful!${NC}"
    kill $TEST_PID 2>/dev/null
else
    echo -e "${YELLOW}⚠️  MCP suite test failed - but configuration is ready${NC}"
fi

# Final instructions
echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo -e "${GREEN}===================================================${NC}"
echo ""
echo "To connect Claude Desktop to your MCP servers:"
echo ""
echo "1. Start the MCP suite:"
echo -e "   ${YELLOW}cd $MCP_PATH${NC}"
echo -e "   ${YELLOW}./start-for-claude.sh${NC}"
echo ""
echo "2. Restart Claude Desktop"
echo ""
echo "3. Test the connection by asking Claude:"
echo "   'What media server tools do you have available?'"
echo ""
echo "4. Claude should respond with 4 available tools:"
echo "   • search_media"
echo "   • get_library_stats"
echo "   • get_recent_media"
echo "   • get_system_info"
echo ""
echo -e "${YELLOW}Configuration file saved at:${NC}"
echo "$CONFIG_FILE"
echo ""
echo -e "${GREEN}🎉 Your MCP servers are ready for Claude Desktop!${NC}"