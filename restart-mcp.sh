#!/bin/bash
# Quick restart script for MCP Dashboard

echo "🔄 Restarting MCP Dashboard..."
cd /Users/morlock/fun/newmedia/mcp-architecture
docker-compose -f docker-compose.simple.yml restart mcp-dashboard
echo "✅ MCP Dashboard restarted!"
echo ""
echo "📊 Check status at: http://localhost:8090"
echo "🤖 MCP Server at: http://localhost:3000/health"