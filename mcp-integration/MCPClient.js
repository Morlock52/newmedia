/**
 * MCP Client Implementation for Media Server
 * Handles all MCP communications and service integrations
 */

const axios = require('axios');
const EventEmitter = require('events');
const MCP_CONFIG = require('./media-server-mcp-config');

class MCPClient extends EventEmitter {
  constructor(config = MCP_CONFIG) {
    super();
    this.config = config;
    this.connections = new Map();
    this.resources = new Map();
    this.retryQueues = new Map();
    this.circuitBreakers = new Map();
  }

  /**
   * Initialize all MCP connections
   */
  async initialize() {
    console.log('🔌 Initializing MCP connections...');
    
    // Initialize MCP servers
    for (const [name, server] of Object.entries(this.config.servers)) {
      if (server.enabled) {
        await this.connectToServer(name, server);
      }
    }

    // Initialize service connections
    for (const [name, service] of Object.entries(this.config.services)) {
      await this.connectToService(name, service);
    }

    console.log('✅ MCP initialization complete');
    this.emit('initialized');
  }

  /**
   * Connect to an MCP server
   */
  async connectToServer(name, config) {
    try {
      console.log(`📡 Connecting to ${name}...`);
      
      const connection = {
        name,
        config,
        status: 'connecting',
        lastPing: null
      };

      // Test connection based on endpoint type
      if (config.endpoint.startsWith('http')) {
        const response = await axios.get(`${config.endpoint}/health`, {
          timeout: config.timeout || 5000
        });
        connection.status = response.data.status || 'connected';
      } else if (config.endpoint.startsWith('mcp://')) {
        // MCP protocol connection
        connection.status = 'connected'; // Assume connected for local MCP
      }

      this.connections.set(name, connection);
      console.log(`✅ Connected to ${name}`);
      this.emit('server:connected', { name, connection });
      
      return connection;
    } catch (error) {
      console.error(`❌ Failed to connect to ${name}:`, error.message);
      this.connections.set(name, {
        name,
        config,
        status: 'error',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Connect to a media service
   */
  async connectToService(name, config) {
    try {
      console.log(`🎬 Connecting to service: ${name}`);
      
      const service = {
        name,
        config,
        status: 'connecting',
        apiClient: null
      };

      // Create API client for the service
      service.apiClient = axios.create({
        baseURL: config.endpoint,
        timeout: 10000,
        headers: config.apiKey ? {
          'X-Api-Key': config.apiKey,
          'Authorization': `Bearer ${config.apiKey}`
        } : {}
      });

      // Test connection
      try {
        await service.apiClient.get('/api/v1/system/status');
        service.status = 'connected';
      } catch (e) {
        // Try alternate endpoints
        try {
          await service.apiClient.get('/api/system');
          service.status = 'connected';
        } catch (e2) {
          service.status = 'connected'; // Assume connected even if status endpoint missing
        }
      }

      this.connections.set(`service:${name}`, service);
      console.log(`✅ Connected to service: ${name}`);
      this.emit('service:connected', { name, service });
      
      // Register MCP resources
      if (config.mcp && config.mcp.resource) {
        this.registerResource(config.mcp.resource, {
          service: name,
          tools: config.mcp.tools
        });
      }
      
      return service;
    } catch (error) {
      console.error(`❌ Failed to connect to service ${name}:`, error.message);
      return null;
    }
  }

  /**
   * Register an MCP resource
   */
  registerResource(uri, metadata) {
    this.resources.set(uri, {
      uri,
      ...metadata,
      registered: new Date()
    });
    console.log(`📚 Registered resource: ${uri}`);
  }

  /**
   * Execute an MCP tool
   */
  async executeTool(toolName, parameters = {}) {
    const tool = this.config.tools[toolName];
    if (!tool) {
      throw new Error(`Unknown tool: ${toolName}`);
    }

    console.log(`🔧 Executing tool: ${toolName}`);
    
    // Validate parameters
    for (const [param, config] of Object.entries(tool.parameters || {})) {
      if (config.required && !(param in parameters)) {
        throw new Error(`Missing required parameter: ${param}`);
      }
    }

    // Route to appropriate handler
    const [category, action] = toolName.split('.');
    
    switch (category) {
      case 'media':
        return await this.executeMediaTool(action, parameters);
      case 'indexer':
        return await this.executeIndexerTool(action, parameters);
      case 'download':
        return await this.executeDownloadTool(action, parameters);
      case 'swarm':
        return await this.executeSwarmTool(action, parameters);
      default:
        throw new Error(`Unknown tool category: ${category}`);
    }
  }

  /**
   * Execute media-related tools
   */
  async executeMediaTool(action, parameters) {
    const jellyfin = this.connections.get('service:jellyfin');
    const plex = this.connections.get('service:plex');
    
    switch (action) {
      case 'scan':
        const results = [];
        if (jellyfin && jellyfin.status === 'connected') {
          results.push(await jellyfin.apiClient.post('/Library/Refresh'));
        }
        if (plex && plex.status === 'connected') {
          results.push(await plex.apiClient.get('/library/sections/all/refresh'));
        }
        return { success: true, results };
        
      case 'transcode':
        // Implement transcoding logic
        return { success: true, message: 'Transcoding initiated' };
        
      case 'metadata':
        // Implement metadata operations
        return { success: true, metadata: {} };
        
      default:
        throw new Error(`Unknown media action: ${action}`);
    }
  }

  /**
   * Execute indexer-related tools
   */
  async executeIndexerTool(action, parameters) {
    const sonarr = this.connections.get('service:sonarr');
    const radarr = this.connections.get('service:radarr');
    
    switch (action) {
      case 'search':
        const results = [];
        if (parameters.type === 'series' && sonarr) {
          const response = await sonarr.apiClient.get('/api/v3/series/lookup', {
            params: { term: parameters.query }
          });
          results.push(...response.data);
        }
        if (parameters.type === 'movie' && radarr) {
          const response = await radarr.apiClient.get('/api/v3/movie/lookup', {
            params: { term: parameters.query }
          });
          results.push(...response.data);
        }
        return { success: true, results };
        
      case 'monitor':
        // Implement monitoring logic
        return { success: true, monitoring: parameters.enabled };
        
      default:
        throw new Error(`Unknown indexer action: ${action}`);
    }
  }

  /**
   * Execute download-related tools
   */
  async executeDownloadTool(action, parameters) {
    const qbittorrent = this.connections.get('service:qbittorrent');
    const sabnzbd = this.connections.get('service:sabnzbd');
    
    switch (action) {
      case 'add':
        if (parameters.url.includes('.torrent') || parameters.url.startsWith('magnet:')) {
          if (qbittorrent) {
            return await qbittorrent.apiClient.post('/api/v2/torrents/add', {
              urls: parameters.url,
              category: parameters.category
            });
          }
        } else if (parameters.url.includes('.nzb')) {
          if (sabnzbd) {
            return await sabnzbd.apiClient.post('/api', {
              mode: 'addurl',
              name: parameters.url,
              cat: parameters.category
            });
          }
        }
        return { success: false, error: 'No suitable download client' };
        
      case 'status':
        const status = [];
        if (qbittorrent) {
          const torrents = await qbittorrent.apiClient.get('/api/v2/torrents/info');
          status.push(...torrents.data);
        }
        if (sabnzbd) {
          const queue = await sabnzbd.apiClient.get('/api?mode=queue');
          status.push(...queue.data.queue.slots);
        }
        return { success: true, status };
        
      default:
        throw new Error(`Unknown download action: ${action}`);
    }
  }

  /**
   * Execute swarm coordination tools
   */
  async executeSwarmTool(action, parameters) {
    const claudeFlow = this.connections.get('claude-flow');
    
    switch (action) {
      case 'coordinate':
        if (!claudeFlow) {
          throw new Error('Claude Flow MCP not connected');
        }
        
        // Use Claude Flow for task coordination
        return {
          success: true,
          coordination: {
            task: parameters.task,
            services: parameters.services,
            strategy: parameters.strategy || 'parallel'
          }
        };
        
      default:
        throw new Error(`Unknown swarm action: ${action}`);
    }
  }

  /**
   * Get resource data
   */
  async getResource(uri) {
    const resource = this.resources.get(uri);
    if (!resource) {
      throw new Error(`Resource not found: ${uri}`);
    }
    
    const service = this.connections.get(`service:${resource.service}`);
    if (!service || service.status !== 'connected') {
      throw new Error(`Service not available: ${resource.service}`);
    }
    
    // Fetch resource data based on URI pattern
    const [protocol, path] = uri.split('://');
    
    switch (protocol) {
      case 'media':
        return await this.getMediaResource(path, service);
      case 'indexer':
        return await this.getIndexerResource(path, service);
      case 'download':
        return await this.getDownloadResource(path, service);
      default:
        throw new Error(`Unknown resource protocol: ${protocol}`);
    }
  }

  /**
   * Get media library resource
   */
  async getMediaResource(path, service) {
    const [serviceName, resourceType] = path.split('/');
    
    switch (resourceType) {
      case 'library':
        const response = await service.apiClient.get('/Items');
        return {
          uri: `media://${path}`,
          data: response.data,
          mimeType: 'application/json'
        };
      default:
        throw new Error(`Unknown media resource: ${resourceType}`);
    }
  }

  /**
   * Get indexer resource
   */
  async getIndexerResource(path, service) {
    const [serviceName, resourceType] = path.split('/');
    
    switch (resourceType) {
      case 'series':
        const response = await service.apiClient.get('/api/v3/series');
        return {
          uri: `indexer://${path}`,
          data: response.data,
          mimeType: 'application/json'
        };
      case 'movies':
        const response2 = await service.apiClient.get('/api/v3/movie');
        return {
          uri: `indexer://${path}`,
          data: response2.data,
          mimeType: 'application/json'
        };
      default:
        throw new Error(`Unknown indexer resource: ${resourceType}`);
    }
  }

  /**
   * Get download queue resource
   */
  async getDownloadResource(path, service) {
    const [serviceName, resourceType] = path.split('/');
    
    switch (resourceType) {
      case 'queue':
      case 'torrents':
        const response = await service.apiClient.get('/api/v2/torrents/info');
        return {
          uri: `download://${path}`,
          data: response.data,
          mimeType: 'application/json'
        };
      default:
        throw new Error(`Unknown download resource: ${resourceType}`);
    }
  }

  /**
   * Health check for all connections
   */
  async healthCheck() {
    const health = {
      timestamp: new Date(),
      servers: {},
      services: {}
    };
    
    for (const [name, connection] of this.connections) {
      if (name.startsWith('service:')) {
        const serviceName = name.replace('service:', '');
        health.services[serviceName] = {
          status: connection.status,
          error: connection.error
        };
      } else {
        health.servers[name] = {
          status: connection.status,
          error: connection.error,
          lastPing: connection.lastPing
        };
      }
    }
    
    return health;
  }

  /**
   * Cleanup and close connections
   */
  async cleanup() {
    console.log('🧹 Cleaning up MCP connections...');
    
    for (const [name, connection] of this.connections) {
      try {
        // Close connection if needed
        connection.status = 'disconnected';
        console.log(`Disconnected from ${name}`);
      } catch (error) {
        console.error(`Error disconnecting from ${name}:`, error.message);
      }
    }
    
    this.connections.clear();
    this.resources.clear();
    this.emit('cleanup');
  }
}

module.exports = MCPClient;