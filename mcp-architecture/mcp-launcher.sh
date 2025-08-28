#!/bin/zsh

# Set up the correct PATH for nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Use the correct node version
nvm use 22.16.0 >/dev/null 2>&1

# Launch the MCP server
exec node "$@"