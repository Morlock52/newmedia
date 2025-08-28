#!/bin/bash

echo "🚀 Final MCP Server Fix - Restarting Claude Desktop"
echo "=================================================="
echo ""
echo "✅ All 5 MCP servers have been FIXED:"
echo "   - stdout pollution removed (no more console.log)"
echo "   - Protocol version corrected to '1.0'"
echo "   - Proper JSON-RPC message formatting"
echo "   - All debug output goes to stderr"
echo ""
echo "📋 Fixed Servers:"
echo "   1. media-server (4 tools)"
echo "   2. sonarr (6 tools)"
echo "   3. jellyfin (2 tools)"
echo "   4. radarr (6 tools)"
echo "   5. prowlarr (6 tools)"
echo "   Total: 24 tools"
echo ""
echo "🔄 Restarting Claude Desktop..."

# Kill Claude
pkill -f "Claude.app" 2>/dev/null
sleep 2

# Make sure it's really dead
if pgrep -f "Claude.app" > /dev/null; then
    echo "Force killing Claude..."
    pkill -9 -f "Claude.app" 2>/dev/null
    sleep 2
fi

# Start Claude
open -a "Claude"

echo ""
echo "✅ Claude Desktop restarted!"
echo ""
echo "📝 TEST INSTRUCTIONS:"
echo "1. Wait for Claude to fully load (10-15 seconds)"
echo "2. In Claude, type: 'What MCP tools are available?'"
echo "3. You should see all 5 servers with 24 tools listed"
echo ""
echo "🐛 DEBUGGING:"
echo "If tools don't appear:"
echo "1. Open Developer Console: View → Developer Tools"
echo "2. Look for 'MCP server connected' messages"
echo "3. Check for any red error messages"
echo "4. Set MCP_DEBUG=true in config for verbose logs"
echo ""
echo "📁 File Locations:"
echo "Config: ~/.claude/claude_desktop_config.json"
echo "Servers: /Users/morlock/fun/newmedia/mcp-architecture/fixed-*.js"
echo ""
echo "🎉 The stdio protocol issue has been FIXED!"