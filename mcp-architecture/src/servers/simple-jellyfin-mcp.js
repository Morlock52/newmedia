/**
 * Simple Jellyfin MCP Server (No external MCP SDK dependency)
 * Provides HTTP API interface for Jellyfin media server
 */

const axios = require('axios');

class SimpleJellyfinMCP {
  constructor(options = {}) {
    this.port = options.port || 3001;
    this.jellyfinUrl = options.jellyfinUrl || process.env.JELLYFIN_URL || 'http://localhost:8096';
    this.apiKey = options.apiKey || process.env.JELLYFIN_API_KEY;
    this.io = options.io;
    
    this.tools = new Map();
    this.resources = new Map();
    
    this.setupTools();
    this.setupResources();
  }

  setupTools() {
    this.tools.set('search_media', {
      description: 'Search for movies, TV shows, music, and other media',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query' },
          type: { type: 'string', description: 'Media type (Movie, Series, Audio, etc.)' },
          limit: { type: 'number', description: 'Maximum number of results' }
        },
        required: ['query']
      },
      handler: this.searchMedia.bind(this)
    });

    this.tools.set('get_library_stats', {
      description: 'Get media library statistics',
      parameters: { type: 'object', properties: {} },
      handler: this.getLibraryStats.bind(this)
    });

    this.tools.set('get_recent_media', {
      description: 'Get recently added media items',
      parameters: {
        type: 'object',
        properties: {
          limit: { type: 'number', description: 'Number of items to return' }
        }
      },
      handler: this.getRecentMedia.bind(this)
    });

    this.tools.set('get_system_info', {
      description: 'Get Jellyfin system information',
      parameters: { type: 'object', properties: {} },
      handler: this.getSystemInfo.bind(this)
    });
  }

  setupResources() {
    this.resources.set('jellyfin://libraries', {
      description: 'All media libraries in Jellyfin',
      mimeType: 'application/json'
    });

    this.resources.set('jellyfin://users', {
      description: 'All users in Jellyfin',
      mimeType: 'application/json'
    });

    this.resources.set('jellyfin://system', {
      description: 'System information',
      mimeType: 'application/json'
    });
  }

  async apiRequest(endpoint, method = 'GET', data = null) {
    try {
      const config = {
        method,
        url: `${this.jellyfinUrl}${endpoint}`,
        headers: {
          'X-Emby-Token': this.apiKey,
          'Content-Type': 'application/json'
        }
      };

      if (data) {
        config.data = data;
      }

      const response = await axios(config);
      return response.data;
    } catch (error) {
      console.error(`Jellyfin API error (${method} ${endpoint}):`, error.message);
      throw new Error(`API request failed: ${error.response?.data || error.message}`);
    }
  }

  async searchMedia({ query, type, limit = 20 }) {
    try {
      const searchUrl = `/Users/Public/Items?searchTerm=${encodeURIComponent(query)}&limit=${limit}`;
      const results = await this.apiRequest(searchUrl);
      
      let filteredResults = results.Items || [];
      if (type) {
        filteredResults = filteredResults.filter(item => item.Type === type);
      }

      const mediaText = filteredResults.map(item => {
        const year = item.ProductionYear ? ` (${item.ProductionYear})` : '';
        const type = item.Type || 'Unknown';
        const overview = item.Overview ? item.Overview.substring(0, 100) + '...' : 'No description';
        
        return `• ${item.Name}${year} [${type}]\n  ${overview}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🎬 Found ${filteredResults.length} items matching "${query}":\n\n${mediaText}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `❌ Error searching media: ${error.message}`
        }]
      };
    }
  }

  async getLibraryStats() {
    try {
      const libraries = await this.apiRequest('/Library/VirtualFolders');
      const systemInfo = await this.apiRequest('/System/Info/Public');
      
      let statsText = `📊 Jellyfin Library Statistics:\n\n`;
      statsText += `🖥️ Server: ${systemInfo.ServerName || 'Unknown'}\n`;
      statsText += `📁 Libraries: ${libraries.length}\n\n`;
      
      for (const library of libraries) {
        const itemsResponse = await this.apiRequest(`/Users/Public/Items?ParentId=${library.ItemId}&Recursive=true`).catch(() => ({ TotalRecordCount: 0 }));
        statsText += `• ${library.Name}: ${itemsResponse.TotalRecordCount || 0} items\n`;
      }

      return {
        content: [{
          type: 'text',
          text: statsText
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `❌ Error getting library stats: ${error.message}`
        }]
      };
    }
  }

  async getRecentMedia({ limit = 10 }) {
    try {
      const recentItems = await this.apiRequest(`/Users/Public/Items/Latest?Limit=${limit}`);
      
      if (!recentItems || recentItems.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📭 No recent media found'
          }]
        };
      }

      const mediaText = recentItems.map(item => {
        const year = item.ProductionYear ? ` (${item.ProductionYear})` : '';
        const type = item.Type || 'Unknown';
        const date = item.DateCreated ? new Date(item.DateCreated).toLocaleDateString() : 'Unknown date';
        
        return `• ${item.Name}${year} [${type}]\n  Added: ${date}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🆕 Recently Added Media:\n\n${mediaText}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `❌ Error getting recent media: ${error.message}`
        }]
      };
    }
  }

  async getSystemInfo() {
    try {
      const systemInfo = await this.apiRequest('/System/Info/Public');
      
      return {
        content: [{
          type: 'text',
          text: `🖥️ Jellyfin System Information:

Server Name: ${systemInfo.ServerName || 'Unknown'}
Version: ${systemInfo.Version || 'Unknown'}
Operating System: ${systemInfo.OperatingSystem || 'Unknown'}
Architecture: ${systemInfo.SystemArchitecture || 'Unknown'}
Server ID: ${systemInfo.Id || 'Unknown'}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `❌ Error getting system info: ${error.message}`
        }]
      };
    }
  }

  getTools() {
    return Array.from(this.tools.entries()).map(([name, tool]) => ({
      name,
      description: tool.description,
      inputSchema: tool.parameters
    }));
  }

  getResources() {
    return Array.from(this.resources.keys()).map(uri => ({ uri }));
  }

  async callTool(name, args) {
    const tool = this.tools.get(name);
    if (!tool) {
      throw new Error(`Tool ${name} not found`);
    }
    return await tool.handler(args);
  }
}

module.exports = SimpleJellyfinMCP;