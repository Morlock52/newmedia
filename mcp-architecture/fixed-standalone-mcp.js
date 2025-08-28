#!/usr/bin/env node

/**
 * Standalone MCP Server for Claude Desktop
 * Single file implementation with built-in HTTP bridge
 */

const readline = require('readline');
const http = require('http');
const axios = require('axios');

class StandaloneMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'media-server-mcp',
      version: '1.0.0',
      protocolVersion: '1.0',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    this.tools = [
      {
        name: 'search_media',
        description: 'Search for movies, TV shows, music, and other media',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            type: { type: 'string', description: 'Media type (Movie, Series, Audio, etc.)' },
            limit: { type: 'number', description: 'Maximum number of results' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_library_stats',
        description: 'Get media library statistics',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      },
      {
        name: 'get_recent_media',
        description: 'Get recently added media items',
        inputSchema: {
          type: 'object',
          properties: {
            limit: { type: 'number', description: 'Number of items to return' }
          }
        }
      },
      {
        name: 'get_system_info',
        description: 'Get Jellyfin system information',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      }
    ];
    
    this.resources = [
      { uri: 'jellyfin://libraries', name: 'Media Libraries', mimeType: 'application/json' },
      { uri: 'jellyfin://system', name: 'System Info', mimeType: 'application/json' }
    ];
  }

  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[MCP] ${new Date().toISOString()} ${message}\n`);
    }
  }

  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    try {
      switch (name) {
        case 'search_media':
          return {
            content: [{
              type: 'text',
              text: `🎬 Search results for "${args.query}":\n\n• The Matrix (1999) [Movie]\n  A computer hacker learns about the true nature of reality\n\n• Star Wars (1977) [Movie]\n  A young farm boy becomes a hero in a galaxy far, far away\n\n• Breaking Bad (2008) [Series]\n  A chemistry teacher turns to crime\n\nNote: This is demo data. Connect to a real Jellyfin server for actual results.`
            }]
          };
          
        case 'get_library_stats':
          return {
            content: [{
              type: 'text',
              text: `📊 Media Library Statistics:\n\n🖥️ Server: Demo Server\n📁 Libraries: 3\n\n• Movies: 1,234 items\n• TV Shows: 567 items\n• Music: 8,901 items\n\nTotal: 10,702 media items\n\nNote: This is demo data. Connect to a real Jellyfin server for actual statistics.`
            }]
          };
          
        case 'get_recent_media':
          const limit = args.limit || 5;
          return {
            content: [{
              type: 'text',
              text: `🆕 Recently Added Media (last ${limit} items):\n\n• Oppenheimer (2023) [Movie]\n  Added: 2 hours ago\n\n• The Last of Us S01E09 [Series]\n  Added: 1 day ago\n\n• Dune: Part Two (2024) [Movie]\n  Added: 2 days ago\n\n• Ted Lasso S03E12 [Series]\n  Added: 3 days ago\n\n• Spider-Man: Across the Spider-Verse (2023) [Movie]\n  Added: 5 days ago\n\nNote: This is demo data.`
            }]
          };
          
        case 'get_system_info':
          return {
            content: [{
              type: 'text',
              text: `🖥️ Jellyfin System Information:\n\nServer Name: Demo Media Server\nVersion: 10.8.13\nOperating System: Linux\nArchitecture: x64\nServer ID: demo-server-001\n\nStatus: Online (Demo Mode)\n\nNote: This is demo data. Connect to a real Jellyfin server for actual system info.`
            }]
          };
          
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      this.log(`Tool call error: ${error.message}`);
      throw error;
    }
  }

  async handleResourceRead(uri) {
    this.log(`Resource read: ${uri}`);
    
    switch (uri) {
      case 'jellyfin://libraries':
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify({
              libraries: [
                { id: '1', name: 'Movies', type: 'Movie', itemCount: 1234 },
                { id: '2', name: 'TV Shows', type: 'Series', itemCount: 567 },
                { id: '3', name: 'Music', type: 'Audio', itemCount: 8901 }
              ]
            }, null, 2)
          }]
        };
        
      case 'jellyfin://system':
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify({
              serverName: 'Demo Media Server',
              version: '10.8.13',
              os: 'Linux',
              architecture: 'x64',
              id: 'demo-server-001'
            }, null, 2)
          }]
        };
        
      default:
        throw new Error(`Unknown resource: ${uri}`);
    }
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: '1.0',
            capabilities: this.serverInfo.capabilities || { tools: {}, resources: {} },
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
          
        case 'resources/list':
          return { resources: this.resources };
          
        case 'resources/read':
          return await this.handleResourceRead(request.params.uri);
          
        case 'completion/complete':
          return { completion: { values: [] } };
          
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      this.log(`Request error: ${error.message}`);
      throw error;
    }
  }

  start() {
    this.log('Starting Standalone MCP Server...');
    
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
          id: JSON.parse(line).id,
          error: {
            code: -32603,
            message: error.message,
            data: error.stack
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
    
    // Handle errors gracefully
    process.on('uncaughtException', (error) => {
      this.log(`Uncaught exception: ${error.message}`);
      process.stderr.write(`MCP Server Error: ${error.message}\n`);
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      this.log(`Unhandled rejection: ${reason}`);
      process.stderr.write(`MCP Server Unhandled Rejection: ${reason}\n`);
    });
  }
}

// Start the server
const server = new StandaloneMCPServer();
server.start();