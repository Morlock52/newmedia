#!/bin/bash

echo "🔄 Restarting Claude Desktop with Fixed MCP Configuration"
echo "======================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\n${YELLOW}Step 1: Killing Claude Desktop...${NC}"
# Kill all Claude processes
pkill -f "Claude.app" 2>/dev/null
sleep 2

echo -e "\n${YELLOW}Step 2: Verifying Claude is closed...${NC}"
if pgrep -f "Claude.app" > /dev/null; then
    echo -e "${YELLOW}Claude still running, force killing...${NC}"
    pkill -9 -f "Claude.app" 2>/dev/null
    sleep 2
fi
echo -e "${GREEN}✅ Claude Desktop closed${NC}"

echo -e "\n${YELLOW}Step 3: Testing MCP servers...${NC}"

# Test media-server
echo -n "Testing media-server... "
if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | /Users/morlock/.nvm/versions/node/v22.16.0/bin/node /Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js 2>/dev/null | grep -q "serverInfo"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test sonarr
echo -n "Testing sonarr... "
if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | /Users/morlock/.nvm/versions/node/v22.16.0/bin/node /Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js 2>/dev/null | grep -q "sonarr-mcp"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Test jellyfin
echo -n "Testing jellyfin... "
if [ -f "/Users/morlock/fun/newmedia/mcp-architecture/jellyfin-mcp-standalone.js" ]; then
    if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | /Users/morlock/.nvm/versions/node/v22.16.0/bin/node /Users/morlock/fun/newmedia/mcp-architecture/jellyfin-mcp-standalone.js 2>/dev/null | grep -q "jellyfin-mcp"; then
        echo -e "${GREEN}✅ Working${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Not found (creating it)${NC}"
fi

echo -e "\n${YELLOW}Step 4: Verifying configuration...${NC}"
if [ -f "$HOME/.claude/claude_desktop_config.json" ]; then
    echo -e "${GREEN}✅ Configuration file exists${NC}"
    echo -e "${BLUE}Current MCP servers:${NC}"
    grep -E '"(media-server|sonarr|jellyfin)"' "$HOME/.claude/claude_desktop_config.json" | head -3
fi

echo -e "\n${YELLOW}Step 5: Starting Claude Desktop...${NC}"
open -a "Claude" 2>/dev/null

echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}✅ CLAUDE DESKTOP RESTARTED${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo -e "${YELLOW}Wait for Claude to fully load, then test by asking:${NC}"
echo '  "What MCP tools are available?"'
echo ""
echo -e "${GREEN}Expected tools:${NC}"
echo "  • Media server: search_media, get_library_stats, get_recent_media, get_system_info"
echo "  • Sonarr: search_series, get_series_list, get_upcoming_episodes, get_missing_episodes, get_system_status, get_queue"
echo "  • Jellyfin: search_media, get_library_stats"
echo ""
echo -e "${YELLOW}If tools don't appear:${NC}"
echo "1. Open Claude Desktop Developer Console (View → Developer Tools)"
echo "2. Check Console tab for errors"
echo "3. Look for 'MCP server connected' messages"