#!/bin/bash

echo "🔍 MCP Connection Diagnostic Tool"
echo "================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Claude Desktop config
echo -e "\n${YELLOW}1. Checking Claude Desktop Configuration...${NC}"
CONFIG_FILE="$HOME/.claude/claude_desktop_config.json"

if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✅ Config file exists${NC}"
    echo -e "${BLUE}Current configuration:${NC}"
    cat "$CONFIG_FILE" | python3 -m json.tool 2>/dev/null || cat "$CONFIG_FILE"
else
    echo -e "${RED}❌ Config file not found at $CONFIG_FILE${NC}"
    echo "Creating config directory..."
    mkdir -p "$HOME/.claude"
fi

# Test MCP servers
echo -e "\n${YELLOW}2. Testing MCP Servers...${NC}"

# Test media server
echo -e "\n${BLUE}Testing media-server MCP:${NC}"
TEST_CMD='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0"}}'
MEDIA_RESULT=$(echo "$TEST_CMD" | timeout 5 node /Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js 2>&1 | head -1)

if [[ $MEDIA_RESULT == *"serverInfo"* ]]; then
    echo -e "${GREEN}✅ Media server MCP is working${NC}"
else
    echo -e "${RED}❌ Media server MCP failed${NC}"
    echo "Error: $MEDIA_RESULT"
fi

# Test Sonarr server
echo -e "\n${BLUE}Testing Sonarr MCP:${NC}"
SONARR_RESULT=$(echo "$TEST_CMD" | timeout 5 node /Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js 2>&1 | head -1)

if [[ $SONARR_RESULT == *"sonarr-mcp"* ]]; then
    echo -e "${GREEN}✅ Sonarr MCP is working${NC}"
else
    echo -e "${RED}❌ Sonarr MCP failed${NC}"
    echo "Error: $SONARR_RESULT"
fi

# Check for common issues
echo -e "\n${YELLOW}3. Checking Common Issues...${NC}"

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js installed: $NODE_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js not found${NC}"
fi

# Check file permissions
echo -e "\n${BLUE}Checking file permissions:${NC}"
if [ -r "/Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js" ]; then
    echo -e "${GREEN}✅ Media MCP script is readable${NC}"
else
    echo -e "${RED}❌ Cannot read Media MCP script${NC}"
fi

if [ -r "/Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js" ]; then
    echo -e "${GREEN}✅ Sonarr MCP script is readable${NC}"
else
    echo -e "${RED}❌ Cannot read Sonarr MCP script${NC}"
fi

# Check if scripts are executable
if [ -x "/Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js" ]; then
    echo -e "${GREEN}✅ Media MCP script is executable${NC}"
else
    echo -e "${YELLOW}⚠️  Media MCP script not executable (fixing...)${NC}"
    chmod +x "/Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js"
fi

# Fix suggestions
echo -e "\n${YELLOW}4. Fix Suggestions:${NC}"

if [[ $MEDIA_RESULT != *"serverInfo"* ]] || [[ $SONARR_RESULT != *"sonarr-mcp"* ]]; then
    echo -e "${BLUE}Try these fixes:${NC}"
    echo ""
    echo "1. ${YELLOW}Restart Claude Desktop:${NC}"
    echo "   - Quit Claude Desktop completely (Cmd+Q)"
    echo "   - Wait 5 seconds"
    echo "   - Open Claude Desktop again"
    echo ""
    echo "2. ${YELLOW}Reset MCP configuration:${NC}"
    echo "   rm ~/.claude/claude_desktop_config.json"
    echo "   Then run: ./setup-sonarr-mcp.sh"
    echo ""
    echo "3. ${YELLOW}Check Claude Desktop logs:${NC}"
    echo "   - Open Claude Desktop"
    echo "   - View → Developer Tools → Console"
    echo "   - Look for MCP errors"
    echo ""
    echo "4. ${YELLOW}Try simplified config:${NC}"
    cat > /tmp/simple-claude-config.json << 'EOF'
{
  "mcpServers": {
    "test-server": {
      "command": "echo",
      "args": ["{'jsonrpc':'2.0','result':{'test':'working'}}"]
    }
  }
}
EOF
    echo "   cp /tmp/simple-claude-config.json ~/.claude/claude_desktop_config.json"
    echo "   (This tests if Claude Desktop reads configs at all)"
else
    echo -e "${GREEN}✅ All MCP servers are functional!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Restart Claude Desktop"
    echo "2. Ask Claude: 'What MCP tools do you have available?'"
    echo "3. If no tools show, check Developer Console for errors"
fi

echo -e "\n${BLUE}5. Quick Test Commands:${NC}"
echo "Test media server:"
echo '  echo '"'"'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'"'"' | node /Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js'
echo ""
echo "Test Sonarr server:"
echo '  echo '"'"'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'"'"' | node /Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js'

echo -e "\n${GREEN}=================================${NC}"
echo -e "${GREEN}Diagnostic complete!${NC}"
echo -e "${GREEN}=================================${NC}"