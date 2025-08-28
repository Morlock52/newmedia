#!/usr/bin/env node

const readline = require('readline');

class JellyfinMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'jellyfin-mcp',
      version: '1.0.0',
      protocolVersion: '0.1.0',
      capabilities: { tools: {}, resources: {} }
    };
    
    this.tools = [
      {
        name: 'search_media',
        description: 'Search for movies, TV shows, music in Jellyfin',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            type: { type: 'string', description: 'Media type' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_library_stats',
        description: 'Get Jellyfin library statistics',
        inputSchema: { type: 'object', properties: {} }
      }
    ];
  }

  async handleRequest(request) {
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: this.serverInfo.protocolVersion,
            serverInfo: this.serverInfo
          };
        case 'tools/list':
          return { tools: this.tools };
        case 'tools/call':
          return {
            content: [{
              type: 'text',
              text: `Demo result for ${request.params.name}`
            }]
          };
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      throw error;
    }
  }

  start() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        const result = await this.handleRequest(request);
        console.log(JSON.stringify({
          jsonrpc: '2.0',
          id: request.id,
          result
        }));
      } catch (error) {
        console.log(JSON.stringify({
          jsonrpc: '2.0',
          id: JSON.parse(line).id,
          error: { code: -32603, message: error.message }
        }));
      }
    });
  }
}

new JellyfinMCPServer().start();
