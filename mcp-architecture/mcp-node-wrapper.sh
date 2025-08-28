#\!/bin/bash

# MCP Node.js Wrapper Script
# This ensures proper Node.js environment for Claude Desktop MCP servers
# Based on proven solutions from the MCP community

# Source nvm if it exists
if [ -f "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
fi

# Use specific Node.js version if available
if command -v "$HOME/.nvm/versions/node/v22.16.0/bin/node" &> /dev/null; then
    exec "$HOME/.nvm/versions/node/v22.16.0/bin/node" "$@"
elif command -v node &> /dev/null; then
    exec node "$@"
else
    echo "Error: Node.js not found" >&2
    exit 1
fi
EOF < /dev/null