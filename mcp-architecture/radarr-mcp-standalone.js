#!/usr/bin/env node

const readline = require('readline');

class RadarrMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'radarr-mcp',
      version: '1.0.0',
      protocolVersion: '0.1.0',
      capabilities: { tools: {}, resources: {} }
    };
    
    this.tools = [
      {
        name: 'search_movies',
        description: 'Search for movies in Radarr database or add new ones',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Movie title to search' },
            year: { type: 'number', description: 'Release year (optional)' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_movie_list',
        description: 'Get list of all movies in Radarr library',
        inputSchema: {
          type: 'object',
          properties: {
            status: { type: 'string', description: 'Filter by status: all, monitored, unmonitored' }
          }
        }
      },
      {
        name: 'get_upcoming_movies',
        description: 'Get movies releasing soon',
        inputSchema: {
          type: 'object',
          properties: {
            days: { type: 'number', description: 'Number of days to look ahead' }
          }
        }
      },
      {
        name: 'get_missing_movies',
        description: 'Get monitored movies that are missing',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_system_status',
        description: 'Get Radarr system status and health',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_download_queue',
        description: 'Get current download queue',
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
      search_movies: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            results: [
              { title: 'The Matrix', year: 1999, id: 'tt0133093', status: 'downloaded' },
              { title: 'Inception', year: 2010, id: 'tt1375666', status: 'monitored' }
            ],
            message: 'Found 2 movies matching your search'
          }, null, 2)
        }]
      },
      get_movie_list: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            movies: [
              { title: 'The Dark Knight', year: 2008, status: 'downloaded', quality: '1080p' },
              { title: 'Interstellar', year: 2014, status: 'downloaded', quality: '4K' },
              { title: 'Dune', year: 2021, status: 'monitored', quality: 'Not Available' }
            ],
            total: 3,
            message: 'Showing 3 movies from library'
          }, null, 2)
        }]
      },
      get_upcoming_movies: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            upcoming: [
              { title: 'Dune: Part Two', releaseDate: '2024-03-01', status: 'announced' }
            ],
            message: '1 movie releasing soon'
          }, null, 2)
        }]
      },
      get_missing_movies: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            missing: [
              { title: 'Blade Runner 2049', year: 2017, monitored: true }
            ],
            total: 1,
            message: '1 monitored movie is missing'
          }, null, 2)
        }]
      },
      get_system_status: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: 'running',
            version: '4.7.5.7809',
            health: 'healthy',
            diskSpace: { free: '500GB', total: '1TB' },
            message: 'Radarr is running normally'
          }, null, 2)
        }]
      },
      get_download_queue: {
        content: [{
          type: 'text',
          text: JSON.stringify({
            queue: [
              { title: 'Avatar: The Way of Water', progress: 45, eta: '25 minutes' }
            ],
            message: '1 movie downloading'
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
new RadarrMCPServer().start();