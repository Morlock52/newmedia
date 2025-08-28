#!/usr/bin/env node

/**
 * Minimal Working MCP Server - Full 2025 Specification
 * Based on latest MCP troubleshooting research
 */

const readline = require('readline');

class MinimalMCPServer {
  constructor() {
    this.protocolVersion = '2025-06-18';
    this.serverInfo = {
      name: 'minimal-test-mcp',
      version: '1.0.0'
    };
  }

  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      console.error(`[minimal-mcp] ${new Date().toISOString()} ${message}`);
    }
  }

  async handleRequest(request) {
    this.log(`Handling: ${request.method}`);
    
    switch (request.method) {
      case 'initialize':
        return {
          protocolVersion: this.protocolVersion,
          capabilities: {
            tools: {},
            resources: {},
            prompts: { listChanged: true }
          },
          serverInfo: this.serverInfo
        };
        
      case 'tools/list':
        return {
          tools: [{
            name: 'test_connection',
            description: 'Test MCP connection',
            inputSchema: { type: 'object', properties: {} }
          }]
        };
        
      case 'tools/call':
        if (request.params.name === 'test_connection') {
          return {
            content: [{
              type: 'text',
              text: '🎉 MCP Server is working perfectly\!\n\nConnection: ✅ Successful\nProtocol: 2025-06-18\nTime: ' + new Date().toLocaleString()
            }]
          };
        }
        throw new Error(`Unknown tool: ${request.params.name}`);
        
      case 'resources/list':
        return { resources: [] };
        
      case 'resources/read':
        throw new Error(`No resources available`);
        
      case 'prompts/list':
        return {
          prompts: [{
            name: 'test_prompt',
            description: 'Test prompt for MCP',
            arguments: []
          }]
        };
        
      case 'prompts/get':
        if (request.params.name === 'test_prompt') {
          return {
            messages: [{
              role: 'user',
              content: {
                type: 'text',
                text: 'This is a test prompt to verify MCP is working correctly.'
              }
            }]
          };
        }
        throw new Error(`Unknown prompt: ${request.params.name}`);
        
      default:
        throw new Error(`Unknown method: ${request.method}`);
    }
  }

  start() {
    console.error('[minimal-mcp] Starting Minimal MCP Server...');
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    // Keep alive interval
    const keepAlive = setInterval(() => {}, 60000);

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        this.log(`Request: ${JSON.stringify(request)}`);
        
        const result = await this.handleRequest(request);
        const response = {
          jsonrpc: '2.0',
          id: request.id,
          result
        };
        
        process.stdout.write(JSON.stringify(response) + '\n');
        this.log(`Response: ${JSON.stringify(response)}`);
      } catch (error) {
        const errorRequest = JSON.parse(line);
        const errorResponse = {
          jsonrpc: '2.0',
          id: errorRequest.id,
          error: {
            code: -32603,
            message: error.message
          }
        };
        process.stdout.write(JSON.stringify(errorResponse) + '\n');
        this.log(`Error: ${error.message}`);
      }
    });

    rl.on('close', () => {
      clearInterval(keepAlive);
      process.exit(0);
    });

    process.on('SIGINT', () => {
      clearInterval(keepAlive);
      process.exit(0);
    });

    console.error('[minimal-mcp] Server ready');
  }
}

const server = new MinimalMCPServer();
server.start();
EOF < /dev/null