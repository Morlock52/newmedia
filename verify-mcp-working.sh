#!/bin/bash

echo "🔍 MCP Server Verification Script"
echo "================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test each server
echo -e "${YELLOW}Testing all 5 MCP servers...${NC}"
echo ""

servers=(
    "fixed-standalone-mcp.js:media-server"
    "fixed-sonarr-mcp-standalone.js:sonarr"
    "fixed-jellyfin-mcp-standalone.js:jellyfin"
    "fixed-radarr-mcp-standalone.js:radarr"
    "fixed-prowlarr-mcp-standalone.js:prowlarr"
)

passed=0
failed=0

for server_info in "${servers[@]}"; do
    IFS=':' read -r file name <<< "$server_info"
    echo -n "Testing $name... "
    
    # Test with shell wrapper (as configured)
    result=$(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
             /bin/zsh -c "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/$file" 2>/dev/null)
    
    if echo "$result" | grep -q '"protocolVersion":"1.0"'; then
        echo -e "${GREEN}✅ Working${NC}"
        ((passed++))
    else
        echo -e "${RED}❌ Failed${NC}"
        ((failed++))
    fi
done

echo ""
echo -e "${BLUE}Results:${NC}"
echo -e "  ${GREEN}Passed: $passed${NC}"
echo -e "  ${RED}Failed: $failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All MCP servers are working correctly!${NC}"
    echo ""
    echo "Claude Desktop should now show:"
    echo "  • media-server (4 tools)"
    echo "  • sonarr (6 tools)"
    echo "  • jellyfin (2 tools)"
    echo "  • radarr (6 tools)"
    echo "  • prowlarr (6 tools)"
    echo "  Total: 24 tools"
else
    echo -e "${RED}⚠️  Some servers are still failing${NC}"
    echo ""
    echo "Try enabling debug mode:"
    echo "1. Edit ~/.claude/claude_desktop_config.json"
    echo "2. Change MCP_DEBUG from 'false' to 'true'"
    echo "3. Restart Claude Desktop"
    echo "4. Check Developer Console for errors"
fi

echo ""
echo "Configuration location: ~/.claude/claude_desktop_config.json"
echo "Server location: /Users/morlock/fun/newmedia/mcp-architecture/"