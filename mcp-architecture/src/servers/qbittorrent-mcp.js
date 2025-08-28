/**
 * qBittorrent MCP Server
 * Provides MCP interface for qBittorrent torrent client API
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { ListResourcesRequestSchema, ReadResourceRequestSchema, ListToolsRequestSchema, CallToolRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const axios = require('axios');

class QBittorrentMCP {
  constructor(options = {}) {
    this.port = options.port || 3005;
    this.qbUrl = options.qbUrl || process.env.QBITTORRENT_URL || 'http://localhost:8080';
    this.username = options.username || process.env.QBITTORRENT_USERNAME || 'admin';
    this.password = options.password || process.env.QBITTORRENT_PASSWORD || 'adminadmin';
    this.io = options.io; // Socket.io instance for real-time updates
    this.sessionCookie = null;
    
    this.server = new McpServer(
      {
        name: 'qbittorrent-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupTools();
    this.setupResources();
    this.setupRequestHandlers();
  }

  setupTools() {
    // Get Torrents
    this.server.tool('get_torrents', {
      description: 'Get all torrents with their status',
      parameters: {
        type: 'object',
        properties: {
          filter: {
            type: 'string',
            description: 'Filter torrents (all, downloading, seeding, completed, paused, active, inactive)',
            enum: ['all', 'downloading', 'seeding', 'completed', 'paused', 'active', 'inactive']
          },
          category: {
            type: 'string',
            description: 'Filter by category'
          }
        }
      }
    }, this.getTorrents.bind(this));

    // Get Torrent Info
    this.server.tool('get_torrent_info', {
      description: 'Get detailed information about a specific torrent',
      parameters: {
        type: 'object',
        properties: {
          hash: {
            type: 'string',
            description: 'Torrent hash'
          }
        },
        required: ['hash']
      }
    }, this.getTorrentInfo.bind(this));

    // Pause/Resume Torrents
    this.server.tool('pause_torrents', {
      description: 'Pause one or more torrents',
      parameters: {
        type: 'object',
        properties: {
          hashes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of torrent hashes (or "all" for all torrents)'
          }
        },
        required: ['hashes']
      }
    }, this.pauseTorrents.bind(this));

    this.server.tool('resume_torrents', {
      description: 'Resume one or more torrents',
      parameters: {
        type: 'object',
        properties: {
          hashes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of torrent hashes (or "all" for all torrents)'
          }
        },
        required: ['hashes']
      }
    }, this.resumeTorrents.bind(this));

    // Delete Torrents
    this.server.tool('delete_torrents', {
      description: 'Delete one or more torrents',
      parameters: {
        type: 'object',
        properties: {
          hashes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of torrent hashes'
          },
          deleteFiles: {
            type: 'boolean',
            description: 'Whether to delete files as well (default: false)'
          }
        },
        required: ['hashes']
      }
    }, this.deleteTorrents.bind(this));

    // Add Torrent
    this.server.tool('add_torrent', {
      description: 'Add a new torrent via magnet link or torrent file URL',
      parameters: {
        type: 'object',
        properties: {
          urls: {
            type: 'string',
            description: 'Magnet link or torrent file URL'
          },
          category: {
            type: 'string',
            description: 'Category to assign to the torrent'
          },
          paused: {
            type: 'boolean',
            description: 'Add torrent in paused state (default: false)'
          },
          savepath: {
            type: 'string',
            description: 'Download path for the torrent'
          }
        },
        required: ['urls']
      }
    }, this.addTorrent.bind(this));

    // Get Categories
    this.server.tool('get_categories', {
      description: 'Get all torrent categories',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getCategories.bind(this));

    // Set Category
    this.server.tool('set_category', {
      description: 'Set category for torrents',
      parameters: {
        type: 'object',
        properties: {
          hashes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of torrent hashes'
          },
          category: {
            type: 'string',
            description: 'Category name'
          }
        },
        required: ['hashes', 'category']
      }
    }, this.setCategory.bind(this));

    // Get Global Stats
    this.server.tool('get_global_stats', {
      description: 'Get global transfer statistics',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getGlobalStats.bind(this));

    // Get Preferences
    this.server.tool('get_preferences', {
      description: 'Get qBittorrent preferences/settings',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getPreferences.bind(this));

    // Set Priority
    this.server.tool('set_priority', {
      description: 'Set priority for torrents',
      parameters: {
        type: 'object',
        properties: {
          hashes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of torrent hashes'
          },
          priority: {
            type: 'string',
            description: 'Priority level',
            enum: ['increase', 'decrease', 'maxPrio', 'minPrio']
          }
        },
        required: ['hashes', 'priority']
      }
    }, this.setPriority.bind(this));
  }

  setupResources() {
    this.server.resource('qbittorrent://torrents', {
      description: 'All torrents in qBittorrent',
      mimeType: 'application/json'
    });

    this.server.resource('qbittorrent://categories', {
      description: 'All torrent categories',
      mimeType: 'application/json'
    });

    this.server.resource('qbittorrent://stats', {
      description: 'Global transfer statistics',
      mimeType: 'application/json'
    });

    this.server.resource('qbittorrent://preferences', {
      description: 'qBittorrent settings and preferences',
      mimeType: 'application/json'
    });
  }

  setupRequestHandlers() {
    this.server.request(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'qbittorrent://torrents',
            name: 'Torrents',
            description: 'All torrents in qBittorrent with their status',
            mimeType: 'application/json'
          },
          {
            uri: 'qbittorrent://categories',
            name: 'Categories',
            description: 'All configured torrent categories',
            mimeType: 'application/json'
          },
          {
            uri: 'qbittorrent://stats',
            name: 'Statistics',
            description: 'Global transfer and session statistics',
            mimeType: 'application/json'
          },
          {
            uri: 'qbittorrent://preferences',
            name: 'Preferences',
            description: 'qBittorrent configuration and settings',
            mimeType: 'application/json'
          }
        ]
      };
    });

    this.server.request(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        await this.ensureAuthenticated();

        switch (uri) {
          case 'qbittorrent://torrents':
            const torrents = await this.apiRequest('/api/v2/torrents/info');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(torrents, null, 2)
              }]
            };

          case 'qbittorrent://categories':
            const categories = await this.apiRequest('/api/v2/torrents/categories');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(categories, null, 2)
              }]
            };

          case 'qbittorrent://stats':
            const stats = await this.apiRequest('/api/v2/transfer/info');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(stats, null, 2)
              }]
            };

          case 'qbittorrent://preferences':
            const preferences = await this.apiRequest('/api/v2/app/preferences');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(preferences, null, 2)
              }]
            };

          default:
            throw new Error(`Unknown resource: ${uri}`);
        }
      } catch (error) {
        throw new Error(`Failed to read resource ${uri}: ${error.message}`);
      }
    });
  }

  async ensureAuthenticated() {
    if (!this.sessionCookie) {
      await this.login();
    }
  }

  async login() {
    try {
      const response = await axios.post(`${this.qbUrl}/api/v2/auth/login`, 
        `username=${encodeURIComponent(this.username)}&password=${encodeURIComponent(this.password)}`,
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }
      );

      if (response.headers['set-cookie']) {
        this.sessionCookie = response.headers['set-cookie'][0];
      } else {
        throw new Error('Authentication failed - no session cookie received');
      }
    } catch (error) {
      console.error('qBittorrent login error:', error.message);
      throw new Error(`Authentication failed: ${error.message}`);
    }
  }

  async apiRequest(endpoint, method = 'GET', data = null) {
    try {
      await this.ensureAuthenticated();

      const config = {
        method,
        url: `${this.qbUrl}${endpoint}`,
        headers: {
          'Cookie': this.sessionCookie
        }
      };

      if (data) {
        if (method === 'POST') {
          config.headers['Content-Type'] = 'application/x-www-form-urlencoded';
          config.data = new URLSearchParams(data).toString();
        } else {
          config.data = data;
        }
      }

      const response = await axios(config);
      return response.data;
    } catch (error) {
      // If we get a 403, try re-authenticating
      if (error.response?.status === 403) {
        this.sessionCookie = null;
        await this.ensureAuthenticated();
        return this.apiRequest(endpoint, method, data);
      }
      
      console.error(`qBittorrent API error (${method} ${endpoint}):`, error.message);
      throw new Error(`API request failed: ${error.response?.data?.message || error.message}`);
    }
  }

  // Tool implementations
  async getTorrents({ filter = 'all', category }) {
    try {
      let url = '/api/v2/torrents/info';
      const params = [];
      
      if (filter !== 'all') {
        params.push(`filter=${filter}`);
      }
      
      if (category) {
        params.push(`category=${encodeURIComponent(category)}`);
      }
      
      if (params.length > 0) {
        url += `?${params.join('&')}`;
      }

      const torrents = await this.apiRequest(url);
      
      if (torrents.length === 0) {
        return {
          content: [{
            type: 'text',
            text: `📭 No torrents found${filter !== 'all' ? ` with filter "${filter}"` : ''}${category ? ` in category "${category}"` : ''}`
          }]
        };
      }

      const torrentSummary = torrents.map(torrent => {
        const progress = (torrent.progress * 100).toFixed(1);
        const status = this.getTorrentStatusIcon(torrent.state);
        
        return `${status} ${torrent.name}
  Progress: ${progress}% | Size: ${this.formatBytes(torrent.size)}
  Speed: ↓${this.formatBytes(torrent.dlspeed)}/s ↑${this.formatBytes(torrent.upspeed)}/s
  ETA: ${torrent.eta === 8640000 ? 'Unknown' : this.formatTime(torrent.eta)}
  Ratio: ${torrent.ratio.toFixed(2)} | Seeds: ${torrent.num_seeds} | Peers: ${torrent.num_leechs}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🌊 Torrents (${torrents.length} total)${filter !== 'all' ? ` - Filter: ${filter}` : ''}:

${torrentSummary}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting torrents: ${error.message}`
        }]
      };
    }
  }

  async getTorrentInfo({ hash }) {
    try {
      const torrents = await this.apiRequest(`/api/v2/torrents/info?hashes=${hash}`);
      
      if (torrents.length === 0) {
        return {
          content: [{
            type: 'text',
            text: `❌ Torrent with hash ${hash} not found`
          }]
        };
      }

      const torrent = torrents[0];
      const status = this.getTorrentStatusIcon(torrent.state);
      const progress = (torrent.progress * 100).toFixed(1);

      return {
        content: [{
          type: 'text',
          text: `${status} ${torrent.name}

📊 Progress & Status:
• Progress: ${progress}%
• State: ${torrent.state}
• Priority: ${torrent.priority}
• Category: ${torrent.category || 'None'}

📏 Size Information:
• Total Size: ${this.formatBytes(torrent.size)}
• Downloaded: ${this.formatBytes(torrent.downloaded)}
• Uploaded: ${this.formatBytes(torrent.uploaded)}
• Remaining: ${this.formatBytes(torrent.amount_left)}

🚀 Speed & Performance:
• Download Speed: ${this.formatBytes(torrent.dlspeed)}/s
• Upload Speed: ${this.formatBytes(torrent.upspeed)}/s
• ETA: ${torrent.eta === 8640000 ? 'Unknown' : this.formatTime(torrent.eta)}
• Ratio: ${torrent.ratio.toFixed(2)}

👥 Peers & Seeds:
• Seeds: ${torrent.num_seeds} (${torrent.num_complete} total)
• Peers: ${torrent.num_leechs} (${torrent.num_incomplete} total)

📅 Dates:
• Added: ${new Date(torrent.added_on * 1000).toLocaleString()}
• Completed: ${torrent.completed > 0 ? new Date(torrent.completed * 1000).toLocaleString() : 'Not completed'}

📍 Location:
• Save Path: ${torrent.save_path}

🔗 Hash: ${torrent.hash}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting torrent info: ${error.message}`
        }]
      };
    }
  }

  async pauseTorrents({ hashes }) {
    try {
      const hashList = Array.isArray(hashes) ? hashes.join('|') : hashes;
      await this.apiRequest('/api/v2/torrents/pause', 'POST', { hashes: hashList });
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('qbittorrent-pause', { hashes: Array.isArray(hashes) ? hashes : [hashes] });
      }

      return {
        content: [{
          type: 'text',
          text: `⏸️ Successfully paused ${Array.isArray(hashes) ? hashes.length : 1} torrent(s)`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error pausing torrents: ${error.message}`
        }]
      };
    }
  }

  async resumeTorrents({ hashes }) {
    try {
      const hashList = Array.isArray(hashes) ? hashes.join('|') : hashes;
      await this.apiRequest('/api/v2/torrents/resume', 'POST', { hashes: hashList });
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('qbittorrent-resume', { hashes: Array.isArray(hashes) ? hashes : [hashes] });
      }

      return {
        content: [{
          type: 'text',
          text: `▶️ Successfully resumed ${Array.isArray(hashes) ? hashes.length : 1} torrent(s)`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error resuming torrents: ${error.message}`
        }]
      };
    }
  }

  async deleteTorrents({ hashes, deleteFiles = false }) {
    try {
      const hashList = Array.isArray(hashes) ? hashes.join('|') : hashes;
      await this.apiRequest('/api/v2/torrents/delete', 'POST', { 
        hashes: hashList,
        deleteFiles: deleteFiles.toString()
      });
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('qbittorrent-delete', { 
          hashes: Array.isArray(hashes) ? hashes : [hashes], 
          deleteFiles 
        });
      }

      return {
        content: [{
          type: 'text',
          text: `🗑️ Successfully deleted ${Array.isArray(hashes) ? hashes.length : 1} torrent(s)${deleteFiles ? ' and their files' : ''}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error deleting torrents: ${error.message}`
        }]
      };
    }
  }

  async addTorrent({ urls, category, paused = false, savepath }) {
    try {
      const data = { urls };
      
      if (category) data.category = category;
      if (paused) data.paused = 'true';
      if (savepath) data.savepath = savepath;

      await this.apiRequest('/api/v2/torrents/add', 'POST', data);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('qbittorrent-add', { urls, category, paused });
      }

      return {
        content: [{
          type: 'text',
          text: `✅ Successfully added torrent${category ? ` to category "${category}"` : ''}${paused ? ' (paused)' : ''}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error adding torrent: ${error.message}`
        }]
      };
    }
  }

  async getCategories() {
    try {
      const categories = await this.apiRequest('/api/v2/torrents/categories');
      
      if (Object.keys(categories).length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📁 No categories configured'
          }]
        };
      }

      const categoryList = Object.entries(categories).map(([name, info]) => {
        return `• ${name}: ${info.savePath}`;
      }).join('\n');

      return {
        content: [{
          type: 'text',
          text: `📁 Configured Categories:

${categoryList}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting categories: ${error.message}`
        }]
      };
    }
  }

  async setCategory({ hashes, category }) {
    try {
      const hashList = Array.isArray(hashes) ? hashes.join('|') : hashes;
      await this.apiRequest('/api/v2/torrents/setCategory', 'POST', { 
        hashes: hashList,
        category: category
      });

      return {
        content: [{
          type: 'text',
          text: `📁 Successfully set category "${category}" for ${Array.isArray(hashes) ? hashes.length : 1} torrent(s)`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error setting category: ${error.message}`
        }]
      };
    }
  }

  async getGlobalStats() {
    try {
      const stats = await this.apiRequest('/api/v2/transfer/info');
      
      return {
        content: [{
          type: 'text',
          text: `📊 Global Transfer Statistics:

🚀 Current Session:
• Download Speed: ${this.formatBytes(stats.dl_info_speed)}/s
• Upload Speed: ${this.formatBytes(stats.up_info_speed)}/s
• Downloaded: ${this.formatBytes(stats.dl_info_data)}
• Uploaded: ${this.formatBytes(stats.up_info_data)}

📈 All-Time Statistics:
• Total Downloaded: ${this.formatBytes(stats.alltime_dl)}
• Total Uploaded: ${this.formatBytes(stats.alltime_ul)}
• Global Ratio: ${(stats.alltime_ul / Math.max(stats.alltime_dl, 1)).toFixed(2)}

🌐 Connection:
• Status: ${stats.connection_status}
• DHT Nodes: ${stats.dht_nodes}

📊 Limits:
• Download Limit: ${stats.dl_rate_limit > 0 ? this.formatBytes(stats.dl_rate_limit) + '/s' : 'Unlimited'}
• Upload Limit: ${stats.up_rate_limit > 0 ? this.formatBytes(stats.up_rate_limit) + '/s' : 'Unlimited'}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting global stats: ${error.message}`
        }]
      };
    }
  }

  async getPreferences() {
    try {
      const prefs = await this.apiRequest('/api/v2/app/preferences');
      
      return {
        content: [{
          type: 'text',
          text: `⚙️ qBittorrent Preferences:

📁 Paths:
• Downloads: ${prefs.save_path || 'Not set'}
• Incomplete: ${prefs.temp_path || 'Not set'}

🌐 Connection:
• Port: ${prefs.listen_port || 'Auto'}
• UPnP: ${prefs.upnp ? 'Enabled' : 'Disabled'}
• Encryption: ${prefs.encryption || 'Prefer'}

📊 Limits:
• Download Limit: ${prefs.dl_limit > 0 ? this.formatBytes(prefs.dl_limit) + '/s' : 'Unlimited'}
• Upload Limit: ${prefs.up_limit > 0 ? this.formatBytes(prefs.up_limit) + '/s' : 'Unlimited'}
• Max Connections: ${prefs.max_connec || 'Unlimited'}
• Max Uploads: ${prefs.max_uploads || 'Unlimited'}

🔄 Queue:
• Max Active Downloads: ${prefs.max_active_downloads || 'Unlimited'}
• Max Active Uploads: ${prefs.max_active_uploads || 'Unlimited'}
• Max Active Torrents: ${prefs.max_active_torrents || 'Unlimited'}

🌱 Seeding:
• Share Ratio Limit: ${prefs.max_ratio > 0 ? prefs.max_ratio : 'Unlimited'}
• Seeding Time Limit: ${prefs.max_seeding_time > 0 ? prefs.max_seeding_time + ' minutes' : 'Unlimited'}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting preferences: ${error.message}`
        }]
      };
    }
  }

  async setPriority({ hashes, priority }) {
    try {
      const hashList = Array.isArray(hashes) ? hashes.join('|') : hashes;
      const endpoint = `/api/v2/torrents/${priority}`;
      
      await this.apiRequest(endpoint, 'POST', { hashes: hashList });

      const priorityNames = {
        'increase': 'increased',
        'decrease': 'decreased',
        'maxPrio': 'set to maximum',
        'minPrio': 'set to minimum'
      };

      return {
        content: [{
          type: 'text',
          text: `📶 Successfully ${priorityNames[priority]} priority for ${Array.isArray(hashes) ? hashes.length : 1} torrent(s)`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error setting priority: ${error.message}`
        }]
      };
    }
  }

  getTorrentStatusIcon(state) {
    const statusIcons = {
      'downloading': '📥',
      'uploading': '📤',
      'pausedDL': '⏸️',
      'pausedUP': '⏸️',
      'queuedDL': '⏳',
      'queuedUP': '⏳',
      'stalledDL': '🐌',
      'stalledUP': '🐌',
      'checkingDL': '🔍',
      'checkingUP': '🔍',
      'error': '❌',
      'missingFiles': '📁❌',
      'allocating': '💾',
      'metaDL': '📋'
    };
    
    return statusIcons[state] || '❓';
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  formatTime(seconds) {
    if (seconds === 0 || seconds === 8640000) return 'Unknown';
    
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log(`qBittorrent MCP Server running on port ${this.port}`);
  }
}

module.exports = QBittorrentMCP;