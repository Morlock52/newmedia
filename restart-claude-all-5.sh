#!/bin/bash

echo "🔄 Restarting Claude Desktop with All 5 MCP Servers"
echo "==================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

echo -e "\n${YELLOW}Step 3: Testing all 5 MCP servers...${NC}"

# Array of servers to test
declare -A servers=(
    ["media-server"]="standalone-mcp.js"
    ["sonarr"]="sonarr-mcp-standalone.js"
    ["jellyfin"]="jellyfin-mcp-standalone.js"
    ["radarr"]="radarr-mcp-standalone.js"
    ["prowlarr"]="prowlarr-mcp-standalone.js"
)

# Test each server
passed=0
failed=0
for name in media-server sonarr jellyfin radarr prowlarr; do
    file="${servers[$name]}"
    echo -n "Testing $name... "
    if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | /Users/morlock/.nvm/versions/node/v22.16.0/bin/node "/Users/morlock/fun/newmedia/mcp-architecture/$file" 2>/dev/null | grep -q "serverInfo"; then
        echo -e "${GREEN}✅ Working${NC}"
        ((passed++))
    else
        echo -e "${RED}❌ Failed${NC}"
        ((failed++))
    fi
done

echo -e "\n${BLUE}Test Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"

echo -e "\n${YELLOW}Step 4: Verifying configuration...${NC}"
if [ -f "$HOME/.claude/claude_desktop_config.json" ]; then
    echo -e "${GREEN}✅ Configuration file exists${NC}"
    echo -e "${BLUE}Configured MCP servers:${NC}"
    grep -E '"(media-server|sonarr|jellyfin|radarr|prowlarr)"' "$HOME/.claude/claude_desktop_config.json" | head -5
fi

echo -e "\n${YELLOW}Step 5: Starting Claude Desktop...${NC}"
open -a "Claude" 2>/dev/null

echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}✅ CLAUDE DESKTOP RESTARTED${NC}"
echo -e "${GREEN}✅ ALL 5 MCP SERVERS CONFIGURED${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo -e "${YELLOW}Wait for Claude to fully load, then test by asking:${NC}"
echo '  "What MCP tools are available?"'
echo ""
echo -e "${GREEN}Expected MCP servers (5 total):${NC}"
echo "  1. ${BLUE}media-server${NC}: General media management (4 tools)"
echo "     • search_media, get_library_stats, get_recent_media, get_system_info"
echo "  2. ${BLUE}sonarr${NC}: TV series management (6 tools)"
echo "     • search_series, get_series_list, get_upcoming_episodes, etc."
echo "  3. ${BLUE}jellyfin${NC}: Media library access (2 tools)"
echo "     • search_media, get_library_stats"
echo "  4. ${BLUE}radarr${NC}: Movie management (6 tools)"
echo "     • search_movies, get_movie_list, get_upcoming_movies, etc."
echo "  5. ${BLUE}prowlarr${NC}: Indexer management (6 tools)"
echo "     • search_indexers, get_indexer_list, test_indexers, etc."
echo ""
echo -e "${YELLOW}Total: 24 tools across 5 MCP servers${NC}"
echo ""
echo -e "${YELLOW}If tools don't appear:${NC}"
echo "1. Open Claude Desktop Developer Console (View → Developer Tools)"
echo "2. Check Console tab for MCP errors"
echo "3. Look for 'MCP server connected' messages (should see 5)"