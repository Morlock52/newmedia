#!/bin/bash
echo "Testing all fixed MCP servers..."
for server in fixed-*.js; do
  echo -n "Testing $server... "
  if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node "$server" 2>/dev/null | grep -q '"protocolVersion"'; then
    echo "✅ Working"
  else
    echo "❌ Failed"
  fi
done
