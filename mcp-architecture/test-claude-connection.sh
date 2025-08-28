#!/bin/bash

echo "🧪 Testing MCP Server for Claude Desktop"
echo "========================================"

# Test initialize
echo "Test 1: Initialize"
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0"}}' | node standalone-mcp.js 2>/dev/null | head -1

# Test tools list
echo -e "\nTest 2: List Tools"
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | node standalone-mcp.js 2>/dev/null | head -1

# Test tool call
echo -e "\nTest 3: Call Tool"
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_system_info","arguments":{}}}' | node standalone-mcp.js 2>/dev/null | head -1

echo -e "\n✅ If you see JSON responses above, the MCP server is working!"
