#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Fixes MCP servers to properly handle stdio protocol
 */

function fixMCPServer(inputFile, outputFile) {
  console.log(`Fixing ${inputFile} -> ${outputFile}`);
  
  let content = fs.readFileSync(inputFile, 'utf8');
  
  // Replace console.log with proper stdout write
  content = content.replace(
    /console\.log\(JSON\.stringify\((response|errorResponse)\)\);?/g,
    'process.stdout.write(JSON.stringify($1) + \'\\n\');'
  );
  
  // Ensure protocol version is correct
  content = content.replace(
    /protocolVersion:\s*['"]0\.1\.0['"]/g,
    'protocolVersion: \'1.0\''
  );
  
  // Fix the initialize response to match expected format
  content = content.replace(
    /case 'initialize':\s*return\s*{[^}]+}/,
    `case 'initialize':
          return {
            protocolVersion: '1.0',
            capabilities: this.serverInfo.capabilities || { tools: {}, resources: {} },
            serverInfo: {
              name: this.serverInfo.name,
              version: this.serverInfo.version
            }
          }`
  );
  
  // Ensure debugging goes to stderr
  content = content.replace(
    /console\.error/g,
    'process.stderr.write'
  );
  
  // Ensure all log functions use stderr
  content = content.replace(
    /process\.env\.DEBUG === 'true'/g,
    'process.env.MCP_DEBUG === \'true\''
  );
  
  fs.writeFileSync(outputFile, content);
  fs.chmodSync(outputFile, '755');
  
  return outputFile;
}

// Fix all MCP servers
const servers = [
  'standalone-mcp.js',
  'sonarr-mcp-standalone.js',
  'jellyfin-mcp-standalone.js',
  'radarr-mcp-standalone.js',
  'prowlarr-mcp-standalone.js'
];

const mcpDir = path.dirname(process.argv[1]);

servers.forEach(server => {
  const inputFile = path.join(mcpDir, server);
  const outputFile = path.join(mcpDir, `fixed-${server}`);
  
  if (fs.existsSync(inputFile)) {
    fixMCPServer(inputFile, outputFile);
    console.log(`✅ Fixed ${server}`);
  } else {
    console.log(`❌ ${server} not found`);
  }
});

// Create test script
const testScript = `#!/bin/bash
echo "Testing all fixed MCP servers..."
for server in fixed-*.js; do
  echo -n "Testing $server... "
  if echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node "$server" 2>/dev/null | grep -q '"protocolVersion"'; then
    echo "✅ Working"
  else
    echo "❌ Failed"
  fi
done
`;

fs.writeFileSync(path.join(mcpDir, 'test-fixed-servers.sh'), testScript);
fs.chmodSync(path.join(mcpDir, 'test-fixed-servers.sh'), '755');

console.log('\n✅ All servers fixed!');
console.log('Run ./test-fixed-servers.sh to test all servers');