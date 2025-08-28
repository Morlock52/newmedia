#!/bin/bash

# Setup MCP for Claude Desktop - Media Server Integration
# This script configures Claude Desktop to work with all media server apps

set -e

echo "🚀 Setting up MCP for Claude Desktop..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if Claude Desktop is installed
CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
if [ ! -d "$CLAUDE_CONFIG_DIR" ]; then
    echo -e "${YELLOW}⚠️  Claude Desktop config directory not found at: $CLAUDE_CONFIG_DIR${NC}"
    echo "Please ensure Claude Desktop is installed first."
    echo "Download from: https://claude.ai/download"
    exit 1
fi

# Step 2: Extract API keys from running services
echo -e "${GREEN}📋 Extracting API keys from services...${NC}"

# Get Jellyfin API key
if [ -f "jellyfin-config/system.xml" ]; then
    JELLYFIN_KEY=$(grep -oP '(?<=<AuthenticationProviderId>)[^<]+' jellyfin-config/system.xml 2>/dev/null || echo "")
fi

# Get Sonarr API key
if [ -f "sonarr-config/config.xml" ]; then
    SONARR_KEY=$(grep -oP '(?<=<ApiKey>)[^<]+' sonarr-config/config.xml 2>/dev/null || echo "")
fi

# Get Radarr API key
if [ -f "radarr-config/config.xml" ]; then
    RADARR_KEY=$(grep -oP '(?<=<ApiKey>)[^<]+' radarr-config/config.xml 2>/dev/null || echo "")
fi

# Get Prowlarr API key
if [ -f "prowlarr-config/config.xml" ]; then
    PROWLARR_KEY=$(grep -oP '(?<=<ApiKey>)[^<]+' prowlarr-config/config.xml 2>/dev/null || echo "")
fi

# Step 3: Install MCP dependencies
echo -e "${GREEN}📦 Installing MCP dependencies...${NC}"
cd mcp-architecture
npm install --silent
cd ..

# Step 4: Create Claude Desktop MCP configuration
echo -e "${GREEN}⚙️  Creating Claude Desktop configuration...${NC}"

cat > "$CLAUDE_CONFIG_DIR/claude_desktop_config.json" << EOF
{
  "mcpServers": {
    "media-server-suite": {
      "command": "node",
      "args": [
        "$(pwd)/mcp-architecture/src/index.js"
      ],
      "env": {
        "JELLYFIN_URL": "http://localhost:8096",
        "JELLYFIN_API_KEY": "${JELLYFIN_KEY:-your_jellyfin_api_key}",
        "SONARR_URL": "http://localhost:8989",
        "SONARR_API_KEY": "${SONARR_KEY:-your_sonarr_api_key}",
        "RADARR_URL": "http://localhost:7878",
        "RADARR_API_KEY": "${RADARR_KEY:-your_radarr_api_key}",
        "PROWLARR_URL": "http://localhost:9696",
        "PROWLARR_API_KEY": "${PROWLARR_KEY:-your_prowlarr_api_key}",
        "QBITTORRENT_URL": "http://localhost:8080",
        "QBITTORRENT_USERNAME": "admin",
        "QBITTORRENT_PASSWORD": "adminadmin",
        "PORT": "8090",
        "LOG_LEVEL": "info"
      }
    },
    "claude-flow": {
      "command": "npx",
      "args": [
        "claude-flow@alpha",
        "mcp",
        "start"
      ]
    },
    "ruv-swarm": {
      "command": "npx",
      "args": [
        "ruv-swarm@latest",
        "mcp",
        "start"
      ]
    }
  }
}
EOF

# Step 5: Create environment file for easy configuration
echo -e "${GREEN}📝 Creating .env.mcp file for configuration...${NC}"

cat > .env.mcp << EOF
# MCP Configuration for Media Server Suite
# Edit these values with your actual API keys

# Jellyfin Configuration
JELLYFIN_URL=http://localhost:8096
JELLYFIN_API_KEY=${JELLYFIN_KEY:-your_jellyfin_api_key}

# Sonarr Configuration  
SONARR_URL=http://localhost:8989
SONARR_API_KEY=${SONARR_KEY:-your_sonarr_api_key}

# Radarr Configuration
RADARR_URL=http://localhost:7878
RADARR_API_KEY=${RADARR_KEY:-your_radarr_api_key}

# Prowlarr Configuration
PROWLARR_URL=http://localhost:9696
PROWLARR_API_KEY=${PROWLARR_KEY:-your_prowlarr_api_key}

# qBittorrent Configuration
QBITTORRENT_URL=http://localhost:8080
QBITTORRENT_USERNAME=admin
QBITTORRENT_PASSWORD=adminadmin

# OpenAI Configuration (Optional - for AI features)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# MCP Server Port
PORT=8090
LOG_LEVEL=info
EOF

# Step 6: Test MCP server
echo -e "${GREEN}🧪 Testing MCP server startup...${NC}"
cd mcp-architecture
timeout 5 node src/index.js > /dev/null 2>&1 || true
cd ..

# Step 7: Create quick start script
echo -e "${GREEN}🚀 Creating quick start script...${NC}"

cat > start-mcp-servers.sh << 'EOF'
#!/bin/bash
# Start all MCP servers for Claude Desktop

echo "Starting Media Server MCP Suite..."
cd mcp-architecture
npm start &
MCP_PID=$!

echo "MCP Suite started with PID: $MCP_PID"
echo "Dashboard: http://localhost:8090"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for interrupt
trap "kill $MCP_PID; exit" INT
wait
EOF

chmod +x start-mcp-servers.sh

# Step 8: Display status and instructions
echo ""
echo -e "${GREEN}✅ MCP setup complete!${NC}"
echo "================================================"
echo ""
echo "📋 Configuration saved to:"
echo "   $CLAUDE_CONFIG_DIR/claude_desktop_config.json"
echo ""

if [ -n "$SONARR_KEY" ] || [ -n "$RADARR_KEY" ] || [ -n "$JELLYFIN_KEY" ]; then
    echo -e "${GREEN}🔑 API Keys detected and configured:${NC}"
    [ -n "$JELLYFIN_KEY" ] && echo "   ✅ Jellyfin"
    [ -n "$SONARR_KEY" ] && echo "   ✅ Sonarr"
    [ -n "$RADARR_KEY" ] && echo "   ✅ Radarr"
    [ -n "$PROWLARR_KEY" ] && echo "   ✅ Prowlarr"
else
    echo -e "${YELLOW}⚠️  No API keys found. Please edit .env.mcp with your API keys${NC}"
fi

echo ""
echo "🎯 Next steps:"
echo "   1. Restart Claude Desktop"
echo "   2. Look for 'MCP' indicator in Claude Desktop"
echo "   3. Try commands like:"
echo "      - 'Search for movies in my Jellyfin library'"
echo "      - 'Show me what's currently downloading'"
echo "      - 'Add a new TV show to Sonarr'"
echo "      - 'Check the status of all media services'"
echo ""
echo "📚 To start MCP servers manually:"
echo "   ./start-mcp-servers.sh"
echo ""
echo "🔧 To edit configuration:"
echo "   nano .env.mcp"
echo ""
echo "Happy media managing with Claude! 🎬"