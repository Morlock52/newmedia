#!/bin/bash
# Unified MCP Server Stop Script

echo "🛑 Stopping Unified MCP Server..."

# Find and kill the process
pkill -f "node.*server.js" || echo "No server process found"

echo "✅ Server stopped"
