#!/usr/bin/env node

const readline = require('readline');

process.stderr.write('[MCP] Starting absolute test server...\n');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  try {
    const request = JSON.parse(line);
    
    if (request.method === 'initialize') {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          protocolVersion: '1.0',
          capabilities: { tools: {}, resources: {} },
          serverInfo: {
            name: 'absolute-test-mcp',
            version: '1.0.0'
          }
        }
      };
      process.stdout.write(JSON.stringify(response) + '\n');
    } else if (request.method === 'tools/list') {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          tools: [{
            name: 'test_tool',
            description: 'Test tool that always works',
            inputSchema: { type: 'object', properties: {} }
          }]
        }
      };
      process.stdout.write(JSON.stringify(response) + '\n');
    } else {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        error: { code: -32601, message: 'Method not found' }
      };
      process.stdout.write(JSON.stringify(response) + '\n');
    }
  } catch (e) {
    // Ignore parse errors
  }
});
