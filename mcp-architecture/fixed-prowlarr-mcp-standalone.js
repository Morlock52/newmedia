#!/usr/bin/env node

const readline = require('readline');

class ProwlarrMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'prowlarr-mcp',
      version: '1.0.0',
      protocolVersion: '1.0',
      capabilities: { tools: {}, resources: {} }
    };
    
    this.tools = [
      {
        name: 'search_indexers',
        description: 'Search across all configured indexers',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            category: { type: 'string', description: 'Category: movies, tv, music, books' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_indexer_list',
        description: 'Get list of all configured indexers',
        inputSchema: {
          type: 'object',
          properties: {
            enabled_only: { type: 'boolean', description: 'Show only enabled indexers' }
          }
        }
      },
      {
        name: 'get_indexer_stats',
        description: 'Get statistics for indexers',
        inputSchema: {
          type: 'object',
          properties: {
            indexer_id: { type: 'number', description: 'Specific indexer ID (optional)' }
          }
        }
      },
      {
        name: 'test_indexers',
        description: 'Test indexer connections',
        inputSchema: {
          type: 'object',
          properties: {
            indexer_id: { type: 'number', description: 'Test specific indexer (optional)' }
          }
        }
      },
      {
        name: 'get_system_status',
        description: 'Get Prowlarr system status',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'sync_apps',
        description: 'Sync indexers to connected apps (Sonarr, Radarr, etc)',
        inputSchema: { type: 'object', properties: {} }
      }
    ];
  }

  async handleRequest(request) {
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
          }, resources: {} },
            serverInfo: {
              name: this.serverInfo.name,
              version: this.serverInfo.version
            }
          };
        
        case 'tools/list':
          return { tools: this.tools };
        
        case 'tools/call':
          return await this.handleToolCall(request.params);
        
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      throw error;
    }
  }

  async handleToolCall(params) {
    const { name, arguments: args } = params;
    
    // Demo mode responses
    const demoResponses = {
      search_indexers: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            results: [
              { 
                indexer: 'NZBgeek',
                title: 'Movie.2023.1080p.BluRay.x264',
                size: '8.5 GB',
                seeders: 45,
                category: 'Movies/HD'
              },
              {
                indexer: 'Usenet-Crawler',
                title: 'Movie.2023.2160p.UHD.BluRay.x265',
                size: '15.2 GB',
                seeders: 23,
                category: 'Movies/UHD'
              }
            ],
            message: 'Found 2 results across indexers'
          }, null, 2)
        }]
      },
      get_indexer_list: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            indexers: [
              { id: 1, name: 'NZBgeek', enabled: true, priority: 0, type: 'usenet' },
              { id: 2, name: 'Usenet-Crawler', enabled: true, priority: 10, type: 'usenet' },
              { id: 3, name: 'RARBG', enabled: false, priority: 20, type: 'torrent' }
            ],
            total: 3,
            enabled: 2,
            message: '3 indexers configured, 2 enabled'
          }, null, 2)
        }]
      },
      get_indexer_stats: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            stats: {
              total_queries: 1523,
              successful_queries: 1456,
              failed_queries: 67,
              average_response_time: '1.2s',
              last_24h_queries: 45
            },
            message: 'Indexer statistics retrieved'
          }, null, 2)
        }]
      },
      test_indexers: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            tests: [
              { indexer: 'NZBgeek', status: 'success', response_time: '0.8s' },
              { indexer: 'Usenet-Crawler', status: 'success', response_time: '1.1s' }
            ],
            message: 'All indexers tested successfully'
          }, null, 2)
        }]
      },
      get_system_status: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: 'running',
            version: '1.8.1.3884',
            uptime: '5 days',
            connected_apps: ['Sonarr', 'Radarr', 'Lidarr'],
            message: 'Prowlarr is running normally'
          }, null, 2)
        }]
      },
      sync_apps: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            synced: [
              { app: 'Sonarr', indexers_synced: 2, status: 'success' },
              { app: 'Radarr', indexers_synced: 2, status: 'success' }
            ],
            message: 'Indexers synced to all connected apps'
          }, null, 2)
        }]
      }
    };

    return demoResponses[name] || {
      content: [{
        type: 'text',
        text: `Demo result for ${name} with args: ${JSON.stringify(args)}`
      }]
    };
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
          error: {
            code: -32603,
            message: error.message
          }
        }));
      }
    });
  }
}

// Start the server
new ProwlarrMCPServer().start();