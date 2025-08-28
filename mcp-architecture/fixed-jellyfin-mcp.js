#!/usr/bin/env node

/**
 * Fixed Jellyfin MCP Server
 * Properly implements stdio protocol for MCP
 */

const readline = require('readline');
const axios = require('axios');

class FixedJellyfinMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'jellyfin-mcp-fixed',
      version: '1.0.0',
      protocolVersion: '2024.11',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    // Get Jellyfin configuration from environment
    this.jellyfinUrl = process.env.JELLYFIN_URL || 'http://localhost:8096';
    this.jellyfinApiKey = process.env.JELLYFIN_API_KEY || '';
    
    this.tools = [
      {
        name: 'search_media',
        description: 'Search for movies, TV shows, music, and other media in Jellyfin',
        inputSchema: {
          type: 'object',
          properties: {
            query: { 
              type: 'string', 
              description: 'Search query' 
            },
            type: { 
              type: 'string', 
              description: 'Media type filter (Movie, Series, Audio, etc.)',
              enum: ['Movie', 'Series', 'Audio', 'Book', 'Photo', 'Video']
            },
            limit: { 
              type: 'number', 
              description: 'Maximum number of results (default: 10)',
              minimum: 1,
              maximum: 50
            }
          },
          required: ['query']
        }
      },
      {
        name: 'get_library_stats',
        description: 'Get Jellyfin library statistics and overview',
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
            limit: { 
              type: 'number', 
              description: 'Number of items to return (default: 10)',
              minimum: 1,
              maximum: 50
            },
            type: {
              type: 'string',
              description: 'Filter by media type',
              enum: ['Movie', 'Series', 'Audio', 'Book', 'Photo', 'Video']
            }
          }
        }
      },
      {
        name: 'get_system_info',
        description: 'Get Jellyfin system information and status',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      },
      {
        name: 'get_playing_now',
        description: 'Get currently playing media sessions',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      }
    ];
    
    this.resources = [
      {
        uri: 'jellyfin://libraries',
        name: 'Media Libraries',
        mimeType: 'application/json',
        description: 'List of all Jellyfin media libraries'
      },
      {
        uri: 'jellyfin://system',
        name: 'System Info',
        mimeType: 'application/json',
        description: 'Jellyfin system information'
      },
      {
        uri: 'jellyfin://users',
        name: 'User List',
        mimeType: 'application/json',
        description: 'List of Jellyfin users'
      }
    ];
  }

  // ALL logging must go to stderr
  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[JELLYFIN-MCP] ${new Date().toISOString()} ${message}\n`);
    }
  }

  async makeJellyfinRequest(endpoint, params = {}) {
    try {
      const response = await axios.get(`${this.jellyfinUrl}${endpoint}`, {
        params: {
          ...params,
          api_key: this.jellyfinApiKey
        },
        timeout: 10000,
        validateStatus: () => true
      });
      
      if (response.status !== 200) {
        throw new Error(`Jellyfin API error: ${response.status} ${response.statusText}`);
      }
      
      return response.data;
    } catch (error) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to Jellyfin server. Is it running?');
      }
      throw error;
    }
  }

  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    try {
      switch (name) {
        case 'search_media': {
          if (!this.jellyfinApiKey) {
            return {
              content: [{
                type: 'text',
                text: 'Jellyfin API key not configured. Set JELLYFIN_API_KEY environment variable.'
              }]
            };
          }
          
          const searchParams = {
            searchTerm: args.query,
            limit: args.limit || 10,
            fields: 'Overview,Path,Genres,DateCreated',
            recursive: true,
            includeItemTypes: args.type || undefined
          };
          
          const results = await this.makeJellyfinRequest('/Items', searchParams);
          
          if (!results.Items || results.Items.length === 0) {
            return {
              content: [{
                type: 'text',
                text: `No results found for "${args.query}"`
              }]
            };
          }
          
          const formattedResults = results.Items.map(item => {
            const year = item.ProductionYear ? ` (${item.ProductionYear})` : '';
            const genres = item.Genres ? item.Genres.join(', ') : 'N/A';
            const overview = item.Overview ? 
              item.Overview.substring(0, 150) + (item.Overview.length > 150 ? '...' : '') : 
              'No overview available';
            
            return `• ${item.Name}${year} [${item.Type}]\n  ${overview}\n  Genres: ${genres}`;
          }).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `🔍 Search results for "${args.query}":\n\n${formattedResults}\n\nTotal results: ${results.TotalRecordCount}`
            }]
          };
        }
          
        case 'get_library_stats': {
          if (!this.jellyfinApiKey) {
            return {
              content: [{
                type: 'text',
                text: 'Jellyfin API key not configured. Set JELLYFIN_API_KEY environment variable.'
              }]
            };
          }
          
          const libraries = await this.makeJellyfinRequest('/Library/MediaFolders');
          const systemInfo = await this.makeJellyfinRequest('/System/Info');
          
          const libraryDetails = await Promise.all(
            libraries.Items.map(async (lib) => {
              const items = await this.makeJellyfinRequest('/Items', {
                parentId: lib.Id,
                recursive: true
              });
              return `• ${lib.Name}: ${items.TotalRecordCount} items`;
            })
          );
          
          return {
            content: [{
              type: 'text',
              text: `📊 Jellyfin Library Statistics:\n\n🖥️ Server: ${systemInfo.ServerName} v${systemInfo.Version}\n📁 Libraries: ${libraries.Items.length}\n\n${libraryDetails.join('\n')}`
            }]
          };
        }
          
        case 'get_recent_media': {
          if (!this.jellyfinApiKey) {
            return {
              content: [{
                type: 'text',
                text: 'Jellyfin API key not configured. Set JELLYFIN_API_KEY environment variable.'
              }]
            };
          }
          
          const params = {
            sortBy: 'DateCreated',
            sortOrder: 'Descending',
            limit: args.limit || 10,
            recursive: true,
            fields: 'Overview,DateCreated',
            includeItemTypes: args.type || undefined
          };
          
          const results = await this.makeJellyfinRequest('/Items', params);
          
          const formattedResults = results.Items.map(item => {
            const date = new Date(item.DateCreated);
            const dateStr = date.toLocaleDateString();
            const year = item.ProductionYear ? ` (${item.ProductionYear})` : '';
            
            return `• ${item.Name}${year} [${item.Type}]\n  Added: ${dateStr}`;
          }).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `🆕 Recently Added Media:\n\n${formattedResults}`
            }]
          };
        }
          
        case 'get_system_info': {
          if (!this.jellyfinApiKey) {
            return {
              content: [{
                type: 'text',
                text: 'Jellyfin API key not configured. Set JELLYFIN_API_KEY environment variable.'
              }]
            };
          }
          
          const info = await this.makeJellyfinRequest('/System/Info');
          
          return {
            content: [{
              type: 'text',
              text: `🖥️ Jellyfin System Information:\n\nServer Name: ${info.ServerName}\nVersion: ${info.Version}\nOperating System: ${info.OperatingSystem}\nArchitecture: ${info.SystemArchitecture}\nServer ID: ${info.Id}\n\nStatus: Online ✅`
            }]
          };
        }
          
        case 'get_playing_now': {
          if (!this.jellyfinApiKey) {
            return {
              content: [{
                type: 'text',
                text: 'Jellyfin API key not configured. Set JELLYFIN_API_KEY environment variable.'
              }]
            };
          }
          
          const sessions = await this.makeJellyfinRequest('/Sessions');
          
          const activeSessions = sessions.filter(s => s.NowPlayingItem);
          
          if (activeSessions.length === 0) {
            return {
              content: [{
                type: 'text',
                text: '🎵 No media currently playing'
              }]
            };
          }
          
          const formattedSessions = activeSessions.map(session => {
            const item = session.NowPlayingItem;
            const user = session.UserName || 'Unknown User';
            const client = session.Client || 'Unknown Client';
            const device = session.DeviceName || 'Unknown Device';
            
            return `• ${user} on ${device} (${client})\n  Playing: ${item.Name} [${item.Type}]`;
          }).join('\n\n');
          
          return {
            content: [{
              type: 'text',
              text: `🎵 Currently Playing:\n\n${formattedSessions}`
            }]
          };
        }
          
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      this.log(`Tool call error: ${error.message}`);
      return {
        content: [{
          type: 'text',
          text: `Error: ${error.message}`
        }]
      };
    }
  }

  async handleResourceRead(uri) {
    this.log(`Resource read: ${uri}`);
    
    try {
      switch (uri) {
        case 'jellyfin://libraries': {
          if (!this.jellyfinApiKey) {
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({ error: 'API key not configured' }, null, 2)
              }]
            };
          }
          
          const libraries = await this.makeJellyfinRequest('/Library/MediaFolders');
          
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(libraries, null, 2)
            }]
          };
        }
          
        case 'jellyfin://system': {
          if (!this.jellyfinApiKey) {
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({ error: 'API key not configured' }, null, 2)
              }]
            };
          }
          
          const info = await this.makeJellyfinRequest('/System/Info');
          
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(info, null, 2)
            }]
          };
        }
          
        case 'jellyfin://users': {
          if (!this.jellyfinApiKey) {
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({ error: 'API key not configured' }, null, 2)
              }]
            };
          }
          
          const users = await this.makeJellyfinRequest('/Users');
          
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(users, null, 2)
            }]
          };
        }
          
        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    } catch (error) {
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify({ error: error.message }, null, 2)
        }]
      };
    }
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    switch (request.method) {
      case 'initialize':
        return {
          protocolVersion: this.serverInfo.protocolVersion,
          serverInfo: this.serverInfo
        };
        
      case 'tools/list':
        return { tools: this.tools };
        
      case 'tools/call':
        if (!request.params || !request.params.name) {
          throw new Error('Missing tool name');
        }
        return await this.handleToolCall(
          request.params.name,
          request.params.arguments || {}
        );
        
      case 'resources/list':
        return { resources: this.resources };
        
      case 'resources/read':
        if (!request.params || !request.params.uri) {
          throw new Error('Missing resource URI');
        }
        return await this.handleResourceRead(request.params.uri);
        
      case 'completion/complete':
        return { 
          completion: { 
            values: [],
            hasMore: false
          } 
        };
        
      default:
        throw new Error(`Unknown method: ${request.method}`);
    }
  }

  sendResponse(id, result) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      result: result
    };
    
    // CRITICAL: Only send JSON to stdout
    process.stdout.write(JSON.stringify(response) + '\n');
    this.log(`Sent response for request ${id}`);
  }

  sendError(id, code, message, data = null) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      error: {
        code: code,
        message: message
      }
    };
    
    if (data) {
      response.error.data = data;
    }
    
    process.stdout.write(JSON.stringify(response) + '\n');
    this.log(`Sent error for request ${id}: ${message}`);
  }

  start() {
    this.log('Starting Fixed Jellyfin MCP Server...');
    this.log(`Jellyfin URL: ${this.jellyfinUrl}`);
    this.log(`API Key configured: ${this.jellyfinApiKey ? 'Yes' : 'No'}`);
    
    // Create readline interface
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    // Handle incoming JSON-RPC requests
    rl.on('line', async (line) => {
      let request;
      
      try {
        request = JSON.parse(line);
        this.log(`Received request: ${JSON.stringify(request)}`);
        
        // Validate JSON-RPC
        if (!request.jsonrpc || request.jsonrpc !== '2.0') {
          this.sendError(request.id || null, -32600, 'Invalid JSON-RPC version');
          return;
        }
        
        if (!request.method) {
          this.sendError(request.id, -32600, 'Missing method');
          return;
        }
        
        // Handle the request
        const result = await this.handleRequest(request);
        
        // Send response
        if (request.id !== undefined) {
          this.sendResponse(request.id, result);
        }
        
      } catch (parseError) {
        if (parseError instanceof SyntaxError) {
          this.sendError(null, -32700, 'Parse error', parseError.message);
        } else if (request && request.id !== undefined) {
          this.sendError(
            request.id,
            -32603,
            parseError.message || 'Internal error',
            process.env.MCP_DEBUG === 'true' ? parseError.stack : undefined
          );
        }
      }
    });

    // Handle process lifecycle
    rl.on('close', () => {
      this.log('Readline closed, exiting...');
      process.exit(0);
    });
    
    process.on('SIGINT', () => {
      this.log('Received SIGINT');
      rl.close();
    });

    process.on('SIGTERM', () => {
      this.log('Received SIGTERM');
      rl.close();
    });

    // Log ready status to stderr
    this.log('Jellyfin MCP Server is ready');
  }
}

// Start the server
if (require.main === module) {
  const server = new FixedJellyfinMCPServer();
  server.start();
}

module.exports = FixedJellyfinMCPServer;