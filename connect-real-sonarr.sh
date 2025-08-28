#!/bin/bash

echo "🔗 Connecting to Real Sonarr Instance"
echo "====================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Check if Sonarr is running
echo -e "\n${YELLOW}Step 1: Checking if Sonarr is running on port 8989...${NC}"

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8989 | grep -q "200\|301\|302"; then
    echo -e "${GREEN}✅ Sonarr is running on port 8989${NC}"
    SONARR_RUNNING=true
else
    echo -e "${RED}❌ Sonarr is not accessible on port 8989${NC}"
    echo -e "${YELLOW}Make sure Sonarr is running. You can start it with:${NC}"
    echo "  docker start sonarr"
    echo "  or"
    echo "  sudo systemctl start sonarr"
    SONARR_RUNNING=false
fi

# Step 2: Show how to get API key
echo -e "\n${YELLOW}Step 2: Getting Sonarr API Key${NC}"
echo -e "${BLUE}To get your Sonarr API key:${NC}"
echo "1. Open Sonarr in your browser: http://localhost:8989"
echo "2. Go to: Settings → General"
echo "3. Look for 'Security' section"
echo "4. Find 'API Key' - click 'Show' to reveal it"
echo "5. Copy the API key (32 characters)"
echo ""
echo -e "${YELLOW}Enter your Sonarr API key (or press Enter to skip): ${NC}"
read -r API_KEY

if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}Skipping API key configuration${NC}"
else
    # Step 3: Update configuration
    echo -e "\n${YELLOW}Step 3: Updating Claude Desktop configuration...${NC}"
    
    CONFIG_FILE="$HOME/.claude/claude_desktop_config.json"
    
    # Create backup
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${BLUE}Created backup of config file${NC}"
    
    # Update the configuration with the API key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/\"SONARR_API_KEY\": \"\"/\"SONARR_API_KEY\": \"$API_KEY\"/" "$CONFIG_FILE"
    else
        # Linux
        sed -i "s/\"SONARR_API_KEY\": \"\"/\"SONARR_API_KEY\": \"$API_KEY\"/" "$CONFIG_FILE"
    fi
    
    echo -e "${GREEN}✅ Configuration updated with API key${NC}"
    
    # Step 4: Test the connection
    echo -e "\n${YELLOW}Step 4: Testing Sonarr connection...${NC}"
    
    # Test with the real API
    TEST_CMD='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_system_status","arguments":{}}}'
    TEST_RESULT=$(echo "$TEST_CMD" | SONARR_API_KEY="$API_KEY" node /Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js 2>/dev/null | head -1)
    
    if [[ $TEST_RESULT == *"Version"* ]] && [[ $TEST_RESULT != *"Demo Mode"* ]]; then
        echo -e "${GREEN}✅ Successfully connected to real Sonarr instance!${NC}"
        
        # Extract version info
        VERSION=$(echo "$TEST_RESULT" | grep -o 'Version: [^\\]*' | cut -d' ' -f2)
        echo -e "${BLUE}Sonarr Version: $VERSION${NC}"
    else
        echo -e "${RED}❌ Could not connect to Sonarr with provided API key${NC}"
        echo -e "${YELLOW}Please verify:${NC}"
        echo "  • API key is correct"
        echo "  • Sonarr is running on port 8989"
        echo "  • No authentication is blocking access"
    fi
fi

# Show current configuration
echo -e "\n${YELLOW}Current Configuration:${NC}"
echo -e "${BLUE}Config file:${NC} $HOME/.claude/claude_desktop_config.json"
grep -A3 "sonarr" "$HOME/.claude/claude_desktop_config.json" | sed 's/^/  /'

# Final instructions
echo -e "\n${GREEN}===============================================${NC}"
echo -e "${GREEN}✅ SONARR MCP CONFIGURATION COMPLETE${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Restart Claude Desktop (Cmd+Q, then reopen)"
echo "2. Test with real data:"
echo "   • 'Show me my TV series library'"
echo "   • 'What episodes are airing this week?'"
echo "   • 'Search for The Last of Us'"
echo "   • 'What's in my download queue?'"
echo ""

if [ "$SONARR_RUNNING" = true ] && [ ! -z "$API_KEY" ]; then
    echo -e "${GREEN}🎉 You're connected to your real Sonarr instance!${NC}"
else
    echo -e "${YELLOW}📝 Running in Demo Mode (configure API key for real data)${NC}"
fi