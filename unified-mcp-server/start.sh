#!/bin/bash
# Unified MCP Server Startup Script

echo "🚀 Starting Unified MCP Server..."

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    exit 1
fi

# Navigate to server directory
cd "/Users/morlock/fun/newmedia/unified-mcp-server"

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the server
echo "✅ Starting server..."
node server.js
