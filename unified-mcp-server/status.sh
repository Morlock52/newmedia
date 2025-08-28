#!/bin/bash
# Unified MCP Server Status Script

echo "📊 Unified MCP Server Status"
echo "============================"

# Check if process is running
if pgrep -f "node.*server.js" > /dev/null; then
    echo "✅ Server is running"
    echo "🔧 Process ID: $(pgrep -f 'node.*server.js')"
else
    echo "❌ Server is not running"
fi

# Check Claude Desktop configuration
CLAUDE_CONFIG="$HOME/.claude.json"
if [ -f "$CLAUDE_CONFIG" ]; then
    echo "✅ Claude Desktop config exists"
    if grep -q "unified-media" "$CLAUDE_CONFIG"; then
        echo "✅ Unified MCP server configured in Claude Desktop"
    else
        echo "⚠️  Unified MCP server not found in Claude Desktop config"
    fi
else
    echo "❌ Claude Desktop config not found"
fi

# Check dependencies
cd "/Users/morlock/fun/newmedia/unified-mcp-server"
if [ -d "node_modules" ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Dependencies missing - run 'npm install'"
fi
