#!/bin/bash

# Fix for MaxListenersExceededWarning when running claude-flow MCP server
# This sets the max listeners to a higher value to prevent warnings

echo "Starting claude-flow with increased max listeners..."

# Set Node.js options to increase max listeners
export NODE_OPTIONS="--max-listeners=20"

# Optional: Increase memory allocation if needed
# export NODE_OPTIONS="--max-listeners=20 --max-old-space-size=4096"

# Start claude-flow MCP server
npx claude-flow@alpha mcp start

echo "claude-flow MCP server started with proper configuration"