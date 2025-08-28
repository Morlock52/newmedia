#!/usr/bin/env node

// Ultra-simple MCP server for testing
const readline = require('readline');

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
          protocolVersion: '0.1.0',
          serverInfo: {
            name: 'test-mcp',
            version: '1.0.0',
            protocolVersion: '0.1.0',
            capabilities: {
              tools: {},
              resources: {}
            }
          }
        }
      };
      console.log(JSON.stringify(response));
    } else if (request.method === 'tools/list') {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          tools: [{
            name: 'test_tool',
            description: 'A test tool',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          }]
        }
      };
      console.log(JSON.stringify(response));
    } else {
      const response = {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: -32601,
          message: 'Method not found'
        }
      };
      console.log(JSON.stringify(response));
    }
  } catch (error) {
    // Invalid JSON, ignore
  }
});