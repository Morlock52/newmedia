#!/usr/bin/env node

/**
 * Standalone Sonarr MCP Server for Claude Desktop
 * Direct stdio implementation - no external dependencies needed
 */

const readline = require('readline');
const http = require('http');
const https = require('https');
const url = require('url');

class SonarrMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'sonarr-mcp',
      version: '1.0.0',
      protocolVersion: '1.0',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    // Get Sonarr connection details from environment
    this.sonarrUrl = process.env.SONARR_URL || 'http://localhost:8989';
    this.sonarrApiKey = process.env.SONARR_API_KEY || '';
    
    this.tools = [
      {
        name: 'search_series',
        description: 'Search for TV series',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query for series name' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_series_list',
        description: 'Get list of all TV series in Sonarr',
        inputSchema: {
          type: 'object',
          properties: {
            limit: { type: 'number', description: 'Maximum number of series to return' }
          }
        }
      },
      {
        name: 'get_upcoming_episodes',
        description: 'Get upcoming episodes',
        inputSchema: {
          type: 'object',
          properties: {
            days: { type: 'number', description: 'Number of days to look ahead' }
          }
        }
      },
      {
        name: 'get_missing_episodes',
        description: 'Get missing episodes from monitored series',
        inputSchema: {
          type: 'object',
          properties: {
            limit: { type: 'number', description: 'Maximum number of episodes to return' }
          }
        }
      },
      {
        name: 'get_system_status',
        description: 'Get Sonarr system status and version',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      },
      {
        name: 'get_queue',
        description: 'Get current download queue',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      }
    ];
    
    this.resources = [
      { uri: 'sonarr://series', name: 'All TV Series', mimeType: 'application/json' },
      { uri: 'sonarr://calendar', name: 'Episode Calendar', mimeType: 'application/json' },
      { uri: 'sonarr://queue', name: 'Download Queue', mimeType: 'application/json' },
      { uri: 'sonarr://system', name: 'System Status', mimeType: 'application/json' }
    ];
  }

  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[Sonarr MCP] ${new Date().toISOString()} ${message}\n`);
    }
  }

  async makeRequest(endpoint, method = 'GET') {
    return new Promise((resolve, reject) => {
      try {
        const parsedUrl = url.parse(`${this.sonarrUrl}/api/v3${endpoint}`);
        const options = {
          hostname: parsedUrl.hostname,
          port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
          path: parsedUrl.path,
          method: method,
          headers: {
            'X-Api-Key': this.sonarrApiKey,
            'Content-Type': 'application/json'
          }
        };

        const httpModule = parsedUrl.protocol === 'https:' ? https : http;
        
        const req = httpModule.request(options, (res) => {
          let data = '';
          res.on('data', (chunk) => data += chunk);
          res.on('end', () => {
            try {
              if (res.statusCode >= 200 && res.statusCode < 300) {
                resolve(JSON.parse(data));
              } else {
                reject(new Error(`Sonarr API error: ${res.statusCode}`));
              }
            } catch (e) {
              reject(new Error(`Failed to parse response: ${e.message}`));
            }
          });
        });

        req.on('error', reject);
        req.end();
      } catch (error) {
        reject(error);
      }
    });
  }

  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    // If no API key is configured, return demo data
    if (!this.sonarrApiKey) {
      return this.getDemoData(name, args);
    }
    
    try {
      switch (name) {
        case 'search_series':
          const searchResults = await this.makeRequest(`/series/lookup?term=${encodeURIComponent(args.query)}`);
          const seriesList = searchResults.slice(0, 5).map(s => 
            `• ${s.title} (${s.year})\n  ${s.overview?.substring(0, 100)}...`
          ).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `📺 Search results for "${args.query}":\n\n${seriesList || 'No series found.'}`
            }]
          };
          
        case 'get_series_list':
          const series = await this.makeRequest('/series');
          const limit = args.limit || 10;
          const seriesInfo = series.slice(0, limit).map(s => 
            `• ${s.title} (${s.year}) - ${s.episodeFileCount}/${s.episodeCount} episodes`
          ).join('\n');
          
          return {
            content: [{
              type: 'text',
              text: `📚 TV Series Library (${series.length} total):\n\n${seriesInfo}`
            }]
          };
          
        case 'get_upcoming_episodes':
          const days = args.days || 7;
          const startDate = new Date().toISOString();
          const endDate = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString();
          const calendar = await this.makeRequest(`/calendar?start=${startDate}&end=${endDate}`);
          
          const upcoming = calendar.map(ep => 
            `• ${ep.series.title} - S${ep.seasonNumber}E${ep.episodeNumber}: ${ep.title}\n  Air Date: ${new Date(ep.airDateUtc).toLocaleDateString()}`
          ).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `📅 Upcoming Episodes (next ${days} days):\n\n${upcoming || 'No upcoming episodes.'}`
            }]
          };
          
        case 'get_missing_episodes':
          const missing = await this.makeRequest('/wanted/missing');
          const limit2 = args.limit || 10;
          const missingList = missing.records?.slice(0, limit2).map(ep => 
            `• ${ep.series.title} - S${ep.seasonNumber}E${ep.episodeNumber}: ${ep.title}\n  Air Date: ${new Date(ep.airDateUtc).toLocaleDateString()}`
          ).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `❌ Missing Episodes (${missing.totalRecords || 0} total):\n\n${missingList || 'No missing episodes.'}`
            }]
          };
          
        case 'get_system_status':
          const status = await this.makeRequest('/system/status');
          return {
            content: [{
              type: 'text',
              text: `🖥️ Sonarr System Status:\n\nVersion: ${status.version}\nBranch: ${status.branch}\nStartup Time: ${new Date(status.startTime).toLocaleString()}\nOS: ${status.osName} ${status.osVersion}\nRuntime: ${status.runtimeName} ${status.runtimeVersion}`
            }]
          };
          
        case 'get_queue':
          const queue = await this.makeRequest('/queue');
          const queueItems = queue.records?.map(item => 
            `• ${item.series.title} - ${item.episode.title}\n  Status: ${item.status}\n  Progress: ${Math.round(item.sizeleft / item.size * 100)}%`
          ).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `📥 Download Queue (${queue.totalRecords || 0} items):\n\n${queueItems || 'Queue is empty.'}`
            }]
          };
          
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      this.log(`Tool call error: ${error.message}`);
      // Return demo data on error
      return this.getDemoData(name, args);
    }
  }

  getDemoData(name, args) {
    switch (name) {
      case 'search_series':
        return {
          content: [{
            type: 'text',
            text: `📺 Search results for "${args.query}" (Demo Mode):\n\n• Breaking Bad (2008)\n  A high school chemistry teacher turned methamphetamine producer\n\n• Better Call Saul (2015)\n  The story of Jimmy McGill's transformation into Saul Goodman\n\n• The Wire (2002)\n  Baltimore drug scene seen through the eyes of dealers and law enforcement\n\nNote: Connect to a real Sonarr instance for actual results.`
          }]
        };
        
      case 'get_series_list':
        return {
          content: [{
            type: 'text',
            text: `📚 TV Series Library (Demo Mode):\n\n• The Last of Us (2023) - 9/9 episodes\n• House of the Dragon (2022) - 10/10 episodes\n• Wednesday (2022) - 8/8 episodes\n• The Mandalorian (2019) - 24/24 episodes\n• Stranger Things (2016) - 42/42 episodes\n\nNote: This is demo data. Configure SONARR_API_KEY for real data.`
          }]
        };
        
      case 'get_upcoming_episodes':
        return {
          content: [{
            type: 'text',
            text: `📅 Upcoming Episodes (Demo Mode):\n\n• The Mandalorian - S4E1: Chapter 33\n  Air Date: ${new Date(Date.now() + 86400000).toLocaleDateString()}\n\n• House of the Dragon - S2E1: TBA\n  Air Date: ${new Date(Date.now() + 172800000).toLocaleDateString()}\n\nNote: This is demo data.`
          }]
        };
        
      case 'get_missing_episodes':
        return {
          content: [{
            type: 'text',
            text: `❌ Missing Episodes (Demo Mode):\n\n• The Last of Us - S1E9: Look for the Light\n  Air Date: 3/12/2023\n\n• Wednesday - S1E8: A Murder of Woes\n  Air Date: 11/23/2022\n\nNote: This is demo data.`
          }]
        };
        
      case 'get_system_status':
        return {
          content: [{
            type: 'text',
            text: `🖥️ Sonarr System Status (Demo Mode):\n\nVersion: 4.0.0.731\nBranch: main\nStartup Time: ${new Date().toLocaleString()}\nOS: Linux 5.15\nRuntime: .NET 6.0.25\n\nStatus: Demo Mode (Configure SONARR_API_KEY for real connection)`
          }]
        };
        
      case 'get_queue':
        return {
          content: [{
            type: 'text',
            text: `📥 Download Queue (Demo Mode):\n\n• The Last of Us - Episode 9\n  Status: Downloading\n  Progress: 73%\n\n• House of the Dragon - Episode 10\n  Status: Queued\n  Progress: 0%\n\nNote: This is demo data.`
          }]
        };
        
      default:
        return {
          content: [{
            type: 'text',
            text: `Error: Unknown tool ${name}`
          }]
        };
    }
  }

  async handleResourceRead(uri) {
    this.log(`Resource read: ${uri}`);
    
    if (!this.sonarrApiKey) {
      return this.getDemoResourceData(uri);
    }
    
    try {
      switch (uri) {
        case 'sonarr://series':
          const series = await this.makeRequest('/series');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(series, null, 2)
            }]
          };
          
        case 'sonarr://calendar':
          const calendar = await this.makeRequest('/calendar');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(calendar, null, 2)
            }]
          };
          
        case 'sonarr://queue':
          const queue = await this.makeRequest('/queue');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(queue, null, 2)
            }]
          };
          
        case 'sonarr://system':
          const status = await this.makeRequest('/system/status');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(status, null, 2)
            }]
          };
          
        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    } catch (error) {
      return this.getDemoResourceData(uri);
    }
  }

  getDemoResourceData(uri) {
    const demoData = {
      'sonarr://series': {
        series: [
          { id: 1, title: 'Breaking Bad', year: 2008, episodeCount: 62, episodeFileCount: 62 },
          { id: 2, title: 'The Wire', year: 2002, episodeCount: 60, episodeFileCount: 60 }
        ]
      },
      'sonarr://system': {
        version: '4.0.0.731',
        branch: 'main',
        osName: 'Linux',
        mode: 'Demo'
      }
    };
    
    return {
      contents: [{
        uri,
        mimeType: 'application/json',
        text: JSON.stringify(demoData[uri] || {}, null, 2)
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
    this.log('Starting Sonarr MCP Server...');
    this.log(`Sonarr URL: ${this.sonarrUrl}`);
    this.log(`API Key: ${this.sonarrApiKey ? 'Configured' : 'Not configured (Demo mode)'}`);
    
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
      process.stderr.write(`Sonarr MCP Error: ${error.message}\n`);
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      this.log(`Unhandled rejection: ${reason}`);
      process.stderr.write(`Sonarr MCP Unhandled Rejection: ${reason}\n`);
    });
  }
}

// Start the server
const server = new SonarrMCPServer();
server.start();