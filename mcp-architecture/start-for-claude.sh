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
