/**
 * Prowlarr MCP Server
 * Provides MCP interface for Prowlarr indexer management API
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { ListResourcesRequestSchema, ReadResourceRequestSchema, ListToolsRequestSchema, CallToolRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const axios = require('axios');

class ProwlarrMCP {
  constructor(options = {}) {
    this.port = options.port || 3004;
    this.prowlarrUrl = options.prowlarrUrl || process.env.PROWLARR_URL || 'http://localhost:9696';
    this.apiKey = options.apiKey || process.env.PROWLARR_API_KEY;
    this.io = options.io; // Socket.io instance for real-time updates
    
    this.server = new McpServer(
      {
        name: 'prowlarr-mcp',
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
    // Get Indexers
    this.server.tool('get_indexers', {
      description: 'Get all configured indexers',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getIndexers.bind(this));

    // Test Indexer
    this.server.tool('test_indexer', {
      description: 'Test a specific indexer connection',
      parameters: {
        type: 'object',
        properties: {
          indexerId: {
            type: 'number',
            description: 'Prowlarr indexer ID'
          }
        },
        required: ['indexerId']
      }
    }, this.testIndexer.bind(this));

    // Search Indexers
    this.server.tool('search_indexers', {
      description: 'Search across all enabled indexers',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query'
          },
          categories: {
            type: 'array',
            items: { type: 'number' },
            description: 'Category IDs to search (optional)'
          },
          limit: {
            type: 'number',
            description: 'Maximum number of results (default: 100)'
          }
        },
        required: ['query']
      }
    }, this.searchIndexers.bind(this));

    // Get Indexer Stats
    this.server.tool('get_indexer_stats', {
      description: 'Get statistics for indexers',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getIndexerStats.bind(this));

    // Enable/Disable Indexer
    this.server.tool('toggle_indexer', {
      description: 'Enable or disable a specific indexer',
      parameters: {
        type: 'object',
        properties: {
          indexerId: {
            type: 'number',
            description: 'Prowlarr indexer ID'
          },
          enable: {
            type: 'boolean',
            description: 'Whether to enable the indexer'
          }
        },
        required: ['indexerId', 'enable']
      }
    }, this.toggleIndexer.bind(this));

    // Get Applications
    this.server.tool('get_applications', {
      description: 'Get all configured applications (Sonarr, Radarr, etc.)',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getApplications.bind(this));

    // Sync with Applications
    this.server.tool('sync_applications', {
      description: 'Sync indexers with configured applications',
      parameters: {
        type: 'object',
        properties: {
          applicationId: {
            type: 'number',
            description: 'Application ID to sync (optional, syncs all if not provided)'
          }
        }
      }
    }, this.syncApplications.bind(this));

    // Get System Status
    this.server.tool('get_system_status', {
      description: 'Get Prowlarr system status and information',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getSystemStatus.bind(this));

    // Get History
    this.server.tool('get_history', {
      description: 'Get indexer search history',
      parameters: {
        type: 'object',
        properties: {
          page: {
            type: 'number',
            description: 'Page number (default: 1)'
          },
          pageSize: {
            type: 'number',
            description: 'Items per page (default: 20)'
          }
        }
      }
    }, this.getHistory.bind(this));
  }

  setupResources() {
    this.server.resource('prowlarr://indexers', {
      description: 'All configured indexers',
      mimeType: 'application/json'
    });

    this.server.resource('prowlarr://applications', {
      description: 'All configured applications',
      mimeType: 'application/json'
    });

    this.server.resource('prowlarr://history', {
      description: 'Search and indexer history',
      mimeType: 'application/json'
    });

    this.server.resource('prowlarr://system', {
      description: 'System status and information',
      mimeType: 'application/json'
    });

    this.server.resource('prowlarr://stats', {
      description: 'Indexer statistics and performance',
      mimeType: 'application/json'
    });
  }

  setupRequestHandlers() {
    this.server.request(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'prowlarr://indexers',
            name: 'Indexers',
            description: 'All configured indexers and their status',
            mimeType: 'application/json'
          },
          {
            uri: 'prowlarr://applications',
            name: 'Applications',
            description: 'Connected applications (Sonarr, Radarr, etc.)',
            mimeType: 'application/json'
          },
          {
            uri: 'prowlarr://history',
            name: 'Search History',
            description: 'Recent search and indexer activity',
            mimeType: 'application/json'
          },
          {
            uri: 'prowlarr://system',
            name: 'System Status',
            description: 'Prowlarr system information and status',
            mimeType: 'application/json'
          },
          {
            uri: 'prowlarr://stats',
            name: 'Statistics',
            description: 'Indexer performance and usage statistics',
            mimeType: 'application/json'
          }
        ]
      };
    });

    this.server.request(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        switch (uri) {
          case 'prowlarr://indexers':
            const indexers = await this.apiRequest('/api/v1/indexer');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(indexers, null, 2)
              }]
            };

          case 'prowlarr://applications':
            const applications = await this.apiRequest('/api/v1/applications');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(applications, null, 2)
              }]
            };

          case 'prowlarr://history':
            const history = await this.apiRequest('/api/v1/history');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(history, null, 2)
              }]
            };

          case 'prowlarr://system':
            const status = await this.apiRequest('/api/v1/system/status');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(status, null, 2)
              }]
            };

          case 'prowlarr://stats':
            const stats = await this.apiRequest('/api/v1/indexerstats');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(stats, null, 2)
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

  async apiRequest(endpoint, method = 'GET', data = null) {
    try {
      const config = {
        method,
        url: `${this.prowlarrUrl}${endpoint}`,
        headers: {
          'X-Api-Key': this.apiKey,
          'Content-Type': 'application/json'
        }
      };

      if (data) {
        config.data = data;
      }

      const response = await axios(config);
      return response.data;
    } catch (error) {
      console.error(`Prowlarr API error (${method} ${endpoint}):`, error.message);
      throw new Error(`API request failed: ${error.response?.data?.message || error.message}`);
    }
  }

  // Tool implementations
  async getIndexers() {
    try {
      const indexers = await this.apiRequest('/api/v1/indexer');
      
      const indexerSummary = indexers.map(indexer => {
        return `• ${indexer.name} (ID: ${indexer.id})
  Status: ${indexer.enable ? '🟢 Enabled' : '🔴 Disabled'}
  Implementation: ${indexer.implementation}
  Categories: ${indexer.capabilities?.categories?.length || 0}
  Priority: ${indexer.priority}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🔍 Configured Indexers (${indexers.length} total):

${indexerSummary}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting indexers: ${error.message}`
        }]
      };
    }
  }

  async testIndexer({ indexerId }) {
    try {
      const indexer = await this.apiRequest(`/api/v1/indexer/${indexerId}`);
      
      // Test the indexer connection
      const testResult = await this.apiRequest(`/api/v1/indexer/test/${indexerId}`, 'POST');
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('prowlarr-test-indexer', { 
          indexerId, 
          name: indexer.name, 
          success: testResult.isValid 
        });
      }

      return {
        content: [{
          type: 'text',
          text: `🧪 Test Results for "${indexer.name}":
${testResult.isValid ? '✅ Connection successful' : '❌ Connection failed'}
${testResult.validationFailures?.map(failure => `⚠️ ${failure.propertyName}: ${failure.errorMessage}`).join('\n') || ''}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error testing indexer: ${error.message}`
        }]
      };
    }
  }

  async searchIndexers({ query, categories = [], limit = 100 }) {
    try {
      let searchUrl = `/api/v1/search?query=${encodeURIComponent(query)}&limit=${limit}`;
      
      if (categories.length > 0) {
        searchUrl += `&categories=${categories.join(',')}`;
      }

      const results = await this.apiRequest(searchUrl);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('prowlarr-search', { 
          query, 
          categories, 
          results: results.length 
        });
      }

      if (results.length === 0) {
        return {
          content: [{
            type: 'text',
            text: `🔍 No results found for "${query}"`
          }]
        };
      }

      const resultSummary = results.slice(0, 10).map(result => {
        return `• ${result.title}
  Indexer: ${result.indexer}
  Size: ${this.formatBytes(result.size)}
  Seeders: ${result.seeders || 'N/A'} | Leechers: ${result.leechers || 'N/A'}
  Category: ${result.categories?.join(', ') || 'Unknown'}
  Age: ${result.age ? `${Math.floor(result.age / 24)}d ${result.age % 24}h` : 'Unknown'}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🔍 Search Results for "${query}" (showing ${Math.min(10, results.length)} of ${results.length}):

${resultSummary}

${results.length > 10 ? `\n... and ${results.length - 10} more results` : ''}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error searching indexers: ${error.message}`
        }]
      };
    }
  }

  async getIndexerStats() {
    try {
      const stats = await this.apiRequest('/api/v1/indexerstats');
      
      if (!stats || stats.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📊 No indexer statistics available'
          }]
        };
      }

      const statsSummary = stats.map(stat => {
        return `• ${stat.indexerName} (ID: ${stat.indexerId})
  Queries: ${stat.numberOfQueries || 0}
  Grabs: ${stat.numberOfGrabs || 0}
  Average Response Time: ${stat.averageResponseTime || 0}ms
  Success Rate: ${stat.numberOfQueries > 0 ? ((stat.numberOfQueries - (stat.numberOfFailedQueries || 0)) / stat.numberOfQueries * 100).toFixed(1) : 0}%`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `📊 Indexer Statistics:

${statsSummary}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting indexer stats: ${error.message}`
        }]
      };
    }
  }

  async toggleIndexer({ indexerId, enable }) {
    try {
      const indexer = await this.apiRequest(`/api/v1/indexer/${indexerId}`);
      indexer.enable = enable;
      
      await this.apiRequest(`/api/v1/indexer/${indexerId}`, 'PUT', indexer);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('prowlarr-toggle-indexer', { 
          indexerId, 
          name: indexer.name, 
          enabled: enable 
        });
      }

      return {
        content: [{
          type: 'text',
          text: `✅ Successfully ${enable ? 'enabled' : 'disabled'} indexer "${indexer.name}"`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error toggling indexer: ${error.message}`
        }]
      };
    }
  }

  async getApplications() {
    try {
      const applications = await this.apiRequest('/api/v1/applications');
      
      if (applications.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📱 No applications configured'
          }]
        };
      }

      const appSummary = applications.map(app => {
        return `• ${app.name} (${app.implementation})
  Status: ${app.enable ? '🟢 Enabled' : '🔴 Disabled'}
  Sync Level: ${app.syncLevel}
  Tags: ${app.tags?.join(', ') || 'None'}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `📱 Configured Applications (${applications.length} total):

${appSummary}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting applications: ${error.message}`
        }]
      };
    }
  }

  async syncApplications({ applicationId }) {
    try {
      if (applicationId) {
        // Sync specific application
        await this.apiRequest('/api/v1/command', 'POST', {
          name: 'ApplicationSync',
          applicationId: applicationId
        });
        
        const app = await this.apiRequest(`/api/v1/applications/${applicationId}`);
        
        return {
          content: [{
            type: 'text',
            text: `🔄 Sync triggered for application "${app.name}"`
          }]
        };
      } else {
        // Sync all applications
        await this.apiRequest('/api/v1/command', 'POST', {
          name: 'ApplicationSync'
        });
        
        // Emit real-time update
        if (this.io) {
          this.io.emit('prowlarr-sync-apps', { all: true });
        }
        
        return {
          content: [{
            type: 'text',
            text: '🔄 Sync triggered for all applications'
          }]
        };
      }
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error syncing applications: ${error.message}`
        }]
      };
    }
  }

  async getSystemStatus() {
    try {
      const status = await this.apiRequest('/api/v1/system/status');
      const health = await this.apiRequest('/api/v1/health').catch(() => []);
      
      const healthIssues = health.filter(h => h.type !== 'ok');
      
      return {
        content: [{
          type: 'text',
          text: `🖥️ Prowlarr System Status:

Version: ${status.version}
Build Date: ${new Date(status.buildTime).toLocaleDateString()}
Runtime Version: ${status.runtimeVersion}
Database Version: ${status.databaseVersion}

🔧 Configuration:
• Start Time: ${new Date(status.startTime).toLocaleString()}
• App Data: ${status.appData}
• OS Name: ${status.osName}
• Is Production: ${status.isProduction ? 'Yes' : 'No'}

🏥 Health Status:
${healthIssues.length === 0 ? '✅ All systems healthy' : 
  healthIssues.map(issue => `⚠️ ${issue.source}: ${issue.message}`).join('\n')}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting system status: ${error.message}`
        }]
      };
    }
  }

  async getHistory({ page = 1, pageSize = 20 }) {
    try {
      const history = await this.apiRequest(`/api/v1/history?page=${page}&pageSize=${pageSize}&sortKey=date&sortDirection=descending`);
      
      if (!history.records || history.records.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📜 No history available'
          }]
        };
      }

      const historyText = history.records.map(record => {
        const eventTime = new Date(record.date).toLocaleString();
        return `• ${record.eventType} - ${eventTime}
  ${record.data?.query ? `Query: "${record.data.query}"` : ''}
  ${record.data?.indexer ? `Indexer: ${record.data.indexer}` : ''}
  ${record.data?.successful !== undefined ? `Success: ${record.data.successful ? 'Yes' : 'No'}` : ''}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `📜 Search History (Page ${page}/${Math.ceil(history.totalRecords / pageSize)}):

${historyText}

Total Records: ${history.totalRecords}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting history: ${error.message}`
        }]
      };
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log(`Prowlarr MCP Server running on port ${this.port}`);
  }
}

module.exports = ProwlarrMCP;