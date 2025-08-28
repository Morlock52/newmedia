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
