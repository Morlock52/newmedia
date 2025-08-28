#!/usr/bin/env node

const readline = require('readline');

class RadarrMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'radarr-mcp',
      version: '1.0.0',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    this.tools = [
      {
            "name": "search_movies",
            "description": "Search for movies",
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
            "name": "get_movie_list",
            "description": "Get all movies",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "get_upcoming_movies",
            "description": "Get upcoming movies",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "get_missing_movies",
            "description": "Get missing movies",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "get_system_status",
            "description": "Get Radarr status",
            "inputSchema": {
                  "type": "object",
                  "properties": {}
            }
      },
      {
            "name": "get_download_queue",
            "description": "Get download queue",
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
      'search_movies': {
        content: [{
          type: 'text',
          text: 'Demo response for search_movies'
        }]
      },
      'get_movie_list': {
        content: [{
          type: 'text',
          text: 'Demo response for get_movie_list'
        }]
      },
      'get_upcoming_movies': {
        content: [{
          type: 'text',
          text: 'Demo response for get_upcoming_movies'
        }]
      },
      'get_missing_movies': {
        content: [{
          type: 'text',
          text: 'Demo response for get_missing_movies'
        }]
      },
      'get_system_status': {
        content: [{
          type: 'text',
          text: 'Demo response for get_system_status'
        }]
      },
      'get_download_queue': {
        content: [{
          type: 'text',
          text: 'Demo response for get_download_queue'
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
    this.log('Starting radarr-mcp MCP Server...');
    
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

const server = new RadarrMCPServer();
server.start();
