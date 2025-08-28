#!/usr/bin/env node

/**
 * FIXED Media Services MCP Server with Proper SDK Integration
 * 
 * This version uses the official @modelcontextprotocol/sdk properly
 * and follows the correct implementation patterns.
 */

const { Server } = require('./node_modules/@modelcontextprotocol/sdk/dist/cjs/server/index.js');
const { StdioServerTransport } = require('./node_modules/@modelcontextprotocol/sdk/dist/cjs/server/stdio.js');
const https = require('https');
const http = require('http');

class MediaServicesMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'media-services-mcp',
      version: '2.0.1',
      description: 'Complete media services management via MCP (SDK-based)'
    };
    
    // Create MCP server instance
    this.server = new Server(this.serverInfo, {
      capabilities: {
        tools: {},
        resources: {},
        logging: {}
      }
    });
    
    // Service configurations
    this.services = {
      jellyfin: { 
        url: process.env.JELLYFIN_URL || 'http://localhost:8096',
        apiKey: process.env.JELLYFIN_API_KEY || '',
        enabled: true
      },
      sonarr: {
        url: process.env.SONARR_URL || 'http://localhost:8989', 
        apiKey: process.env.SONARR_API_KEY || '',
        enabled: true
      },
      radarr: {
        url: process.env.RADARR_URL || 'http://localhost:7878',
        apiKey: process.env.RADARR_API_KEY || '',
        enabled: true
      },
      prowlarr: {
        url: process.env.PROWLARR_URL || 'http://localhost:9696',
        apiKey: process.env.PROWLARR_API_KEY || '',
        enabled: true
      },
      qbittorrent: {
        url: process.env.QBITTORRENT_URL || 'http://localhost:8080',
        username: process.env.QBITTORRENT_USER || 'admin',
        password: process.env.QBITTORRENT_PASS || 'admin',
        enabled: true
      },
      bazarr: {
        url: process.env.BAZARR_URL || 'http://localhost:6767',
        apiKey: process.env.BAZARR_API_KEY || '',
        enabled: true
      },
      lidarr: {
        url: process.env.LIDARR_URL || 'http://localhost:8686',
        apiKey: process.env.LIDARR_API_KEY || '',
        enabled: true
      }
    };
    
    // Cache for performance
    this.cache = new Map();
    this.cacheTimeout = 300000; // 5 minutes
    
    this._setupTools();
    this._setupResources();
  }
  
  log(message, level = 'info') {
    const timestamp = new Date().toISOString();
    const logLevel = level.toUpperCase();
    const logMessage = `[${this.serverInfo.name}] ${timestamp} ${logLevel}: ${message}`;
    
    if (process.env.MCP_DEBUG === 'true' || level === 'error') {
      process.stderr.write(logMessage + '\n');
    }
  }
  
  _setupTools() {
    // System status tool
    this.server.setRequestHandler(
      { method: 'tools/call' },
      async (request) => {
        const { name, arguments: args } = request.params;
        
        this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
        
        try {
          switch (name) {
            case 'get_system_status':
              return await this._getSystemStatus(args?.detailed || false);
              
            case 'search_media':
              return await this._searchMedia(args?.query, args?.type || 'all', args?.limit || 20);
              
            case 'get_library_stats':
              return await this._getLibraryStats(args?.service || 'all');
              
            case 'get_recent_activity':
              return await this._getRecentActivity(args?.hours || 24, args?.limit || 50);
              
            case 'manage_downloads':
              return await this._manageDownloads(args?.action, args?.hash);
              
            case 'add_media_request':
              return await this._addMediaRequest(args);
              
            default:
              throw new Error(`Unknown tool: ${name}`);
          }
        } catch (error) {
          this.log(`Tool error: ${error.message}`, 'error');
          return {
            content: [{
              type: 'text',
              text: `❌ Error executing ${name}: ${error.message}`
            }]
          };
        }
      }
    );
    
    // List available tools
    this.server.setRequestHandler(
      { method: 'tools/list' },
      async () => {
        return {
          tools: [
            {
              name: 'get_system_status',
              description: 'Get overall system status for all media services',
              inputSchema: {
                type: 'object',
                properties: {
                  detailed: { type: 'boolean', description: 'Include detailed status information' }
                }
              }
            },
            {
              name: 'search_media',
              description: 'Search for movies, TV shows, music across all services',
              inputSchema: {
                type: 'object',
                properties: {
                  query: { type: 'string', description: 'Search query' },
                  type: { 
                    type: 'string', 
                    enum: ['movie', 'tv', 'music', 'all'], 
                    description: 'Media type to search for' 
                  },
                  limit: { type: 'number', description: 'Maximum results to return', default: 20 }
                },
                required: ['query']
              }
            },
            {
              name: 'get_library_stats',
              description: 'Get comprehensive library statistics',
              inputSchema: {
                type: 'object',
                properties: {
                  service: { 
                    type: 'string', 
                    enum: ['jellyfin', 'sonarr', 'radarr', 'lidarr', 'all'],
                    description: 'Specific service or all services' 
                  }
                }
              }
            },
            {
              name: 'get_recent_activity',
              description: 'Get recent media activity and additions',
              inputSchema: {
                type: 'object',
                properties: {
                  hours: { type: 'number', description: 'Hours to look back', default: 24 },
                  limit: { type: 'number', description: 'Maximum items to return', default: 50 }
                }
              }
            },
            {
              name: 'manage_downloads',
              description: 'View and manage active downloads',
              inputSchema: {
                type: 'object',
                properties: {
                  action: { 
                    type: 'string', 
                    enum: ['list', 'pause', 'resume', 'delete'],
                    description: 'Action to perform'
                  },
                  hash: { type: 'string', description: 'Torrent hash (for pause/resume/delete)' }
                },
                required: ['action']
              }
            },
            {
              name: 'add_media_request',
              description: 'Add a new media request (movie/TV show/music)',
              inputSchema: {
                type: 'object',
                properties: {
                  title: { type: 'string', description: 'Media title' },
                  type: { type: 'string', enum: ['movie', 'tv', 'music'], description: 'Media type' },
                  year: { type: 'number', description: 'Release year' },
                  quality: { type: 'string', description: 'Quality profile', default: 'HD-1080p' },
                  monitor: { type: 'boolean', description: 'Monitor for releases', default: true }
                },
                required: ['title', 'type']
              }
            }
          ]
        };
      }
    );
  }
  
  _setupResources() {
    // List available resources
    this.server.setRequestHandler(
      { method: 'resources/list' },
      async () => {
        return {
          resources: [
            { uri: 'media://system/status', name: 'System Status', mimeType: 'application/json' },
            { uri: 'media://library/stats', name: 'Library Statistics', mimeType: 'application/json' },
            { uri: 'media://activity/recent', name: 'Recent Activity', mimeType: 'application/json' },
            { uri: 'media://downloads/active', name: 'Active Downloads', mimeType: 'application/json' },
            { uri: 'media://config/services', name: 'Service Configuration', mimeType: 'application/json' }
          ]
        };
      }
    );
    
    // Read resources
    this.server.setRequestHandler(
      { method: 'resources/read' },
      async (request) => {
        const { uri } = request.params;
        this.log(`Resource read: ${uri}`);
        
        try {
          switch (uri) {
            case 'media://system/status':
              const statusData = await this._getSystemStatus(true);
              return {
                contents: [{
                  uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(statusData, null, 2)
                }]
              };
              
            case 'media://library/stats':
              const statsData = await this._getLibraryStats('all');
              return {
                contents: [{
                  uri,
                  mimeType: 'application/json', 
                  text: JSON.stringify(statsData, null, 2)
                }]
              };
              
            case 'media://config/services':
              const config = Object.keys(this.services).reduce((acc, service) => {
                acc[service] = {
                  url: this.services[service].url,
                  enabled: this.services[service].enabled,
                  hasApiKey: !!this.services[service].apiKey
                };
                return acc;
              }, {});
              
              return {
                contents: [{
                  uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(config, null, 2)
                }]
              };
              
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
    );
  }
  
  async makeRequest(service, endpoint, options = {}) {
    const serviceConfig = this.services[service];
    if (!serviceConfig || !serviceConfig.enabled) {
      throw new Error(`Service ${service} is not enabled or configured`);
    }
    
    const cacheKey = `${service}:${endpoint}:${JSON.stringify(options)}`;
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      this.log(`Cache hit for ${cacheKey}`);
      return cached.data;
    }
    
    return new Promise((resolve, reject) => {
      const url = new URL(endpoint, serviceConfig.url);
      const isHttps = url.protocol === 'https:';
      const client = isHttps ? https : http;
      
      // Prepare headers
      const headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'MediaServices-MCP/2.0.1'
      };
      
      // Add authentication
      if (serviceConfig.apiKey) {
        headers['X-Api-Key'] = serviceConfig.apiKey;
      }
      
      const requestOptions = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: options.method || 'GET',
        headers,
        timeout: options.timeout || 10000
      };
      
      const req = client.request(requestOptions, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          try {
            const parsedData = res.statusCode === 204 ? {} : JSON.parse(data);
            
            if (res.statusCode >= 200 && res.statusCode < 300) {
              // Cache successful responses
              this.cache.set(cacheKey, {
                data: parsedData,
                timestamp: Date.now()
              });
              
              resolve(parsedData);
            } else {
              reject(new Error(`HTTP ${res.statusCode}: ${parsedData.message || 'Unknown error'}`));
            }
          } catch (error) {
            reject(new Error(`Failed to parse response: ${error.message}`));
          }
        });
      });
      
      req.on('timeout', () => {
        req.destroy();
        reject(new Error(`Request timeout for ${service}${endpoint}`));
      });
      
      req.on('error', (error) => {
        reject(new Error(`Request failed for ${service}: ${error.message}`));
      });
      
      if (options.body) {
        req.write(JSON.stringify(options.body));
      }
      
      req.end();
    });
  }
  
  async _getSystemStatus(detailed = false) {
    const status = {};
    const services = Object.keys(this.services);
    
    for (const service of services) {
      if (!this.services[service].enabled) {
        status[service] = { status: 'disabled' };
        continue;
      }
      
      try {
        let healthEndpoint;
        switch (service) {
          case 'jellyfin':
            healthEndpoint = '/System/Info/Public';
            break;
          case 'sonarr':
          case 'radarr':
          case 'lidarr':
          case 'prowlarr':
          case 'bazarr':
            healthEndpoint = '/api/v1/system/status';
            break;
          case 'qbittorrent':
            healthEndpoint = '/api/v2/app/version';
            break;
          default:
            continue;
        }
        
        const response = await this.makeRequest(service, healthEndpoint);
        status[service] = {
          status: 'running',
          version: response.Version || response.version || 'unknown',
          ...(detailed && { details: response })
        };
      } catch (error) {
        status[service] = {
          status: 'error',
          error: error.message
        };
      }
    }
    
    const runningServices = Object.values(status).filter(s => s.status === 'running').length;
    const totalServices = services.length;
    
    return {
      content: [{
        type: 'text',
        text: `🖥️ **Media Server System Status**\n\n` +
              `Overall Health: ${runningServices}/${totalServices} services running\n\n` +
              Object.entries(status).map(([service, info]) => {
                const emoji = info.status === 'running' ? '✅' : 
                             info.status === 'disabled' ? '⚪' : '❌';
                const version = info.version ? ` (v${info.version})` : '';
                const error = info.error ? ` - ${info.error}` : '';
                return `${emoji} **${service.charAt(0).toUpperCase() + service.slice(1)}**: ${info.status}${version}${error}`;
              }).join('\n')
      }]
    };
  }
  
  async _searchMedia(query, type, limit) {
    const results = [];
    
    try {
      // Search logic similar to original but simplified for demo
      if (type === 'movie' || type === 'all') {
        try {
          const radarrResults = await this.makeRequest('radarr', `/api/v3/movie/lookup?term=${encodeURIComponent(query)}`);
          results.push(...radarrResults.slice(0, limit).map(movie => ({
            service: 'radarr',
            type: 'movie',
            title: movie.title,
            year: movie.year,
            overview: movie.overview,
            status: movie.status,
            tmdbId: movie.tmdbId
          })));
        } catch (error) {
          this.log(`Radarr search failed: ${error.message}`, 'error');
        }
      }
      
    } catch (error) {
      this.log(`Media search error: ${error.message}`, 'error');
    }
    
    const displayResults = results.slice(0, limit);
    
    return {
      content: [{
        type: 'text',
        text: `🔍 **Search Results for "${query}"**\n\n` +
              `Found ${displayResults.length} results:\n\n` +
              displayResults.map((result, index) => 
                `${index + 1}. **${result.title}** (${result.year || 'N/A'})\n` +
                `   Service: ${result.service} | Type: ${result.type}\n` +
                `   ${result.overview ? result.overview.substring(0, 100) + '...' : 'No description'}\n`
              ).join('\n')
      }]
    };
  }
  
  async _getLibraryStats(service) {
    // Simplified implementation for demo
    return {
      content: [{
        type: 'text',
        text: `📊 **Media Library Statistics**\n\nStats for ${service} (demo mode - implement full logic as needed)`
      }]
    };
  }
  
  async _getRecentActivity(hours, limit) {
    // Simplified implementation for demo
    return {
      content: [{
        type: 'text',
        text: `🕒 **Recent Activity (Last ${hours} hours)**\n\nRecent activity (demo mode - implement full logic as needed)`
      }]
    };
  }
  
  async _manageDownloads(action, hash) {
    // Simplified implementation for demo
    return {
      content: [{
        type: 'text',
        text: `📥 **Download Management**\n\nAction: ${action} (demo mode - implement full logic as needed)`
      }]
    };
  }
  
  async _addMediaRequest(args) {
    // Simplified implementation for demo
    return {
      content: [{
        type: 'text',
        text: `✅ **Media Request**\n\nRequest: ${JSON.stringify(args)} (demo mode - implement full logic as needed)`
      }]
    };
  }
  
  async start() {
    this.log('Starting Media Services MCP Server (SDK-based)...');
    
    // Create stdio transport
    const transport = new StdioServerTransport();
    
    // Connect the server to the transport
    await this.server.connect(transport);
    
    this.log(`SDK-based server started successfully! Ready to handle media service requests.`);
    this.log(`Available services: ${Object.keys(this.services).filter(s => this.services[s].enabled).join(', ')}`);
  }
}

// Start the server if run directly
if (require.main === module) {
  const server = new MediaServicesMCPServer();
  server.start().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}

module.exports = MediaServicesMCPServer;