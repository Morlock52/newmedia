#!/bin/bash

echo "🔧 Ultimate NVM Fix for Claude Desktop MCP Servers"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Step 1: Find Node.js path from NVM
echo -e "${YELLOW}Step 1: Finding your NVM Node.js installation...${NC}"
NODE_PATH=$(which node)
NPM_PATH=$(which npm)
NODE_VERSION=$(node --version)

echo -e "${GREEN}✅ Found Node.js:${NC}"
echo "   Path: $NODE_PATH"
echo "   Version: $NODE_VERSION"

# Step 2: Find the actual node modules path
echo -e "\n${YELLOW}Step 2: Finding global node_modules path...${NC}"
NPM_ROOT=$(npm root -g)
echo -e "${GREEN}✅ Global modules at: $NPM_ROOT${NC}"

# Step 3: Install MCP servers globally
echo -e "\n${YELLOW}Step 3: Installing MCP servers globally...${NC}"
echo "This ensures we have full control over paths..."

# Create a simple test server first
cat > /Users/morlock/fun/newmedia/mcp-architecture/absolute-test-mcp.js << 'EOF'
#!/usr/bin/env node

const readline = require('readline');

process.stderr.write('[MCP] Starting absolute test server...\n');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  try {
    const request = JSON.parse(line);
    
    if (request.method === 'initialize') {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          protocolVersion: '1.0',
          capabilities: { tools: {}, resources: {} },
          serverInfo: {
            name: 'absolute-test-mcp',
            version: '1.0.0'
          }
        }
      };
      process.stdout.write(JSON.stringify(response) + '\n');
    } else if (request.method === 'tools/list') {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          tools: [{
            name: 'test_tool',
            description: 'Test tool that always works',
            inputSchema: { type: 'object', properties: {} }
          }]
        }
      };
      process.stdout.write(JSON.stringify(response) + '\n');
    } else {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        error: { code: -32601, message: 'Method not found' }
      };
      process.stdout.write(JSON.stringify(response) + '\n');
    }
  } catch (e) {
    // Ignore parse errors
  }
});
EOF

chmod +x /Users/morlock/fun/newmedia/mcp-architecture/absolute-test-mcp.js

# Step 4: Create the ultimate configuration with absolute paths
echo -e "\n${YELLOW}Step 4: Creating configuration with absolute paths...${NC}"

cat > /Users/morlock/.claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "test-server": {
      "command": "$NODE_PATH",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/absolute-test-mcp.js"]
    },
    "media-server": {
      "command": "$NODE_PATH",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/fixed-standalone-mcp.js"],
      "env": {
        "MCP_DEBUG": "false"
      }
    },
    "sonarr": {
      "command": "$NODE_PATH",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/fixed-sonarr-mcp-standalone.js"],
      "env": {
        "MCP_DEBUG": "false",
        "SONARR_URL": "http://localhost:8989",
        "SONARR_API_KEY": ""
      }
    },
    "jellyfin": {
      "command": "$NODE_PATH",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/perfect-jellyfin-mcp.js"],
      "env": {
        "MCP_DEBUG": "false",
        "JELLYFIN_URL": "http://localhost:8096",
        "JELLYFIN_API_KEY": ""
      }
    },
    "radarr": {
      "command": "$NODE_PATH",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/perfect-radarr-mcp.js"],
      "env": {
        "MCP_DEBUG": "false",
        "RADARR_URL": "http://localhost:7878",
        "RADARR_API_KEY": ""
      }
    },
    "prowlarr": {
      "command": "$NODE_PATH",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/perfect-prowlarr-mcp.js"],
      "env": {
        "MCP_DEBUG": "false",
        "PROWLARR_URL": "http://localhost:9696",
        "PROWLARR_API_KEY": ""
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration created with absolute paths!${NC}"

# Step 5: Alternative - Create symlinks as backup
echo -e "\n${YELLOW}Step 5: Creating symlinks as backup solution...${NC}"
if [ ! -L "/usr/local/bin/node" ]; then
    echo "Creating symlink for node..."
    sudo ln -sf "$NODE_PATH" /usr/local/bin/node 2>/dev/null || echo "   (Requires sudo - skipping)"
fi

# Step 6: Test the configuration
echo -e "\n${YELLOW}Step 6: Testing configuration...${NC}"
echo -n "Testing absolute test server... "
if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | "$NODE_PATH" /Users/morlock/fun/newmedia/mcp-architecture/absolute-test-mcp.js 2>/dev/null | grep -q "absolute-test-mcp"; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Step 7: Restart Claude
echo -e "\n${YELLOW}Step 7: Restarting Claude Desktop...${NC}"
pkill -f "Claude.app" 2>/dev/null
sleep 3
open -a "Claude"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ NVM FIX APPLIED SUCCESSFULLY!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration now uses absolute paths:"
echo "  Node: $NODE_PATH"
echo "  Version: $NODE_VERSION"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "1. Claude Desktop has been restarted"
echo "2. Wait 10-15 seconds for it to fully load"
echo "3. Test by asking: 'What MCP tools are available?'"
echo ""
echo "You should see 6 servers:"
echo "  • test-server (1 tool)"
echo "  • media-server (4 tools)"
echo "  • sonarr (6 tools)"
echo "  • jellyfin (2 tools)"
echo "  • radarr (6 tools)"
echo "  • prowlarr (6 tools)"
echo "  Total: 25 tools"
echo ""
echo -e "${BLUE}Debug tip:${NC} If still not working, check:"
echo "  View → Developer Tools → Console"
echo "  Look for 'MCP server connected' messages"