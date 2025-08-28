#!/bin/bash
# MCP Server Wrapper for ruv-swarm
echo "Starting ruv-swarm MCP server..." >&2
exec node /Users/morlock/fun/newmedia/mcp-architecture/bulletproof-mcp.js "$@"