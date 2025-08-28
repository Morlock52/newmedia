#!/usr/bin/env node

const readline = require('readline');

class ProwlarrMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'prowlarr-mcp',
      version: '1.0.0',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    this.tools = [
      {
            "name": "search_indexers",
            "description": "Search indexers",
            "inputSchema": {
                  "type": "object",
                  "properties": {
                        "query": {
                              "type": "string"
                        }
                  },
                  "required": [
                        "query"
                  ]
            }
      },
      {
            "name": "get_indexer_list",
            "description": "Get all indexers",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "get_indexer_stats",
            "description": "Get indexer statistics",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "test_indexers",
            "description": "Test indexer connections",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "get_system_status",
            "description": "Get Prowlarr status",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "sync_apps",
            "description": "Sync to connected apps",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      }
];
  }

  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[MCP] ${new Date().toISOString()} ${message}\n`);
    }
  }

  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    // Demo responses for each tool
    const responses = {
      'search_indexers': {
        content: [{
          type: 'text',
          text: 'Demo response for search_indexers'
        }]
      },
      'get_indexer_list': {
        content: [{
          type: 'text',
          text: 'Demo response for get_indexer_list'
        }]
      },
      'get_indexer_stats': {
        content: [{
          type: 'text',
          text: 'Demo response for get_indexer_stats'
        }]
      },
      'test_indexers': {
        content: [{
          type: 'text',
          text: 'Demo response for test_indexers'
        }]
      },
      'get_system_status': {
        content: [{
          type: 'text',
          text: 'Demo response for get_system_status'
        }]
      },
      'sync_apps': {
        content: [{
          type: 'text',
          text: 'Demo response for sync_apps'
        }]
      }
    };
    
    return responses[name] || {
      content: [{
        type: 'text',
        text: `Demo result for ${name}`
      }]
    };
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: '1.0',
            capabilities: this.serverInfo.capabilities,
            serverInfo: {
              name: this.serverInfo.name,
              version: this.serverInfo.version
            }
          };
          
        case 'tools/list':
          return { tools: this.tools };
          
        case 'tools/call':
          return await this.handleToolCall(
            request.params.name,
            request.params.arguments || {}
          );
          
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      this.log(`Request error: ${error.message}`);
      throw error;
    }
  }

  start() {
    this.log('Starting prowlarr-mcp MCP Server...');
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        this.log(`Received: ${JSON.stringify(request)}`);
        
        const result = await this.handleRequest(request);
        
        const response = {
          jsonrpc: '2.0',
          id: request.id,
          result
        };
        
        process.stdout.write(JSON.stringify(response) + '\n');
        this.log(`Sent: ${JSON.stringify(response)}`);
      } catch (error) {
        const errorResponse = {
          jsonrpc: '2.0',
          id: line ? JSON.parse(line).id : null,
          error: {
            code: -32603,
            message: error.message
          }
        };
        process.stdout.write(JSON.stringify(errorResponse) + '\n');
        this.log(`Error: ${JSON.stringify(errorResponse)}`);
      }
    });

    rl.on('close', () => {
      this.log('Server closing...');
      process.exit(0);
    });
  }
}

const server = new ProwlarrMCPServer();
server.start();
