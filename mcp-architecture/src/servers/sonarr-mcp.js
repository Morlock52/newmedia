/**
 * Sonarr MCP Server
 * Provides MCP interface for Sonarr TV show management API
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { ListResourcesRequestSchema, ReadResourceRequestSchema, ListToolsRequestSchema, CallToolRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const axios = require('axios');

class SonarrMCP {
  constructor(options = {}) {
    this.port = options.port || 3002;
    this.sonarrUrl = options.sonarrUrl || process.env.SONARR_URL || 'http://localhost:8989';
    this.apiKey = options.apiKey || process.env.SONARR_API_KEY;
    this.io = options.io; // Socket.io instance for real-time updates
    
    this.server = new McpServer(
      {
        name: 'sonarr-mcp',
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
    // TV Show Search
    this.server.tool('search_tv_shows', {
      description: 'Search for TV shows in Sonarr',
      parameters: {
        type: 'object',
        properties: {
          term: {
            type: 'string',
            description: 'Search term for TV shows'
          }
        },
        required: ['term']
      }
    }, this.searchTVShows.bind(this));

    // Get Series Info
    this.server.tool('get_series', {
      description: 'Get detailed information about a specific series',
      parameters: {
        type: 'object',
        properties: {
          seriesId: {
            type: 'number',
            description: 'Sonarr series ID'
          }
        },
        required: ['seriesId']
      }
    }, this.getSeries.bind(this));

    // Monitor Series
    this.server.tool('monitor_series', {
      description: 'Set monitoring status for a series',
      parameters: {
        type: 'object',
        properties: {
          seriesId: {
            type: 'number',
            description: 'Sonarr series ID'
          },
          monitored: {
            type: 'boolean',
            description: 'Whether to monitor the series'
          }
        },
        required: ['seriesId', 'monitored']
      }
    }, this.monitorSeries.bind(this));

    // Get Queue
    this.server.tool('get_queue', {
      description: 'Get current download queue',
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
    }, this.getQueue.bind(this));

    // Get Calendar
    this.server.tool('get_calendar', {
      description: 'Get upcoming episodes',
      parameters: {
        type: 'object',
        properties: {
          start: {
            type: 'string',
            description: 'Start date (YYYY-MM-DD format)'
          },
          end: {
            type: 'string',
            description: 'End date (YYYY-MM-DD format)'
          }
        }
      }
    }, this.getCalendar.bind(this));

    // Trigger Series Refresh
    this.server.tool('refresh_series', {
      description: 'Trigger a refresh for specific series',
      parameters: {
        type: 'object',
        properties: {
          seriesId: {
            type: 'number',
            description: 'Sonarr series ID'
          }
        },
        required: ['seriesId']
      }
    }, this.refreshSeries.bind(this));

    // Get System Status
    this.server.tool('get_system_status', {
      description: 'Get Sonarr system status and information',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getSystemStatus.bind(this));

    // Search Episodes
    this.server.tool('search_episodes', {
      description: 'Search for specific episodes',
      parameters: {
        type: 'object',
        properties: {
          seriesId: {
            type: 'number',
            description: 'Sonarr series ID'
          },
          seasonNumber: {
            type: 'number',
            description: 'Season number (optional)'
          }
        },
        required: ['seriesId']
      }
    }, this.searchEpisodes.bind(this));
  }

  setupResources() {
    this.server.resource('sonarr://series', {
      description: 'All TV series in Sonarr',
      mimeType: 'application/json'
    });

    this.server.resource('sonarr://queue', {
      description: 'Current download queue',
      mimeType: 'application/json'
    });

    this.server.resource('sonarr://calendar', {
      description: 'Upcoming episodes calendar',
      mimeType: 'application/json'
    });

    this.server.resource('sonarr://system', {
      description: 'System status and information',
      mimeType: 'application/json'
    });
  }

  setupRequestHandlers() {
    this.server.request(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'sonarr://series',
            name: 'TV Series Library',
            description: 'All TV series managed by Sonarr',
            mimeType: 'application/json'
          },
          {
            uri: 'sonarr://queue',
            name: 'Download Queue',
            description: 'Current download queue status',
            mimeType: 'application/json'
          },
          {
            uri: 'sonarr://calendar',
            name: 'Episodes Calendar',
            description: 'Upcoming episodes and air dates',
            mimeType: 'application/json'
          },
          {
            uri: 'sonarr://system',
            name: 'System Status',
            description: 'Sonarr system information and status',
            mimeType: 'application/json'
          }
        ]
      };
    });

    this.server.request(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        switch (uri) {
          case 'sonarr://series':
            const series = await this.apiRequest('/api/v3/series');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(series, null, 2)
              }]
            };

          case 'sonarr://queue':
            const queue = await this.apiRequest('/api/v3/queue');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(queue, null, 2)
              }]
            };

          case 'sonarr://calendar':
            const today = new Date().toISOString().split('T')[0];
            const nextWeek = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            const calendar = await this.apiRequest(`/api/v3/calendar?start=${today}&end=${nextWeek}`);
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(calendar, null, 2)
              }]
            };

          case 'sonarr://system':
            const status = await this.apiRequest('/api/v3/system/status');
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
        throw new Error(`Failed to read resource ${uri}: ${error.message}`);
      }
    });
  }

  async apiRequest(endpoint, method = 'GET', data = null) {
    try {
      const config = {
        method,
        url: `${this.sonarrUrl}${endpoint}`,
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
      console.error(`Sonarr API error (${method} ${endpoint}):`, error.message);
      throw new Error(`API request failed: ${error.response?.data?.message || error.message}`);
    }
  }

  // Tool implementations
  async searchTVShows({ term }) {
    try {
      const results = await this.apiRequest(`/api/v3/series/lookup?term=${encodeURIComponent(term)}`);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('sonarr-search', { term, results: results.length });
      }

      return {
        content: [{
          type: 'text',
          text: `Found ${results.length} TV shows matching "${term}":
${results.slice(0, 10).map(show => 
  `• ${show.title} (${show.year}) - ${show.network || 'Unknown Network'}
    Status: ${show.status}, Seasons: ${show.seasonCount || 0}
    TVDB ID: ${show.tvdbId}`
).join('\n')}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error searching TV shows: ${error.message}`
        }]
      };
    }
  }

  async getSeries({ seriesId }) {
    try {
      const series = await this.apiRequest(`/api/v3/series/${seriesId}`);
      const episodes = await this.apiRequest(`/api/v3/episode?seriesId=${seriesId}`);
      
      const totalEpisodes = episodes.length;
      const downloadedEpisodes = episodes.filter(ep => ep.hasFile).length;
      const monitoredEpisodes = episodes.filter(ep => ep.monitored).length;

      return {
        content: [{
          type: 'text',
          text: `📺 ${series.title} (${series.year})

Network: ${series.network || 'Unknown'}
Status: ${series.status}
Rating: ${series.ratings?.value ? `${series.ratings.value}/10` : 'Not rated'}
Genres: ${series.genres?.join(', ') || 'Unknown'}

📊 Episode Statistics:
• Total Episodes: ${totalEpisodes}
• Downloaded: ${downloadedEpisodes}
• Monitored: ${monitoredEpisodes}
• Missing: ${monitoredEpisodes - downloadedEpisodes}

🔧 Settings:
• Monitored: ${series.monitored ? 'Yes' : 'No'}
• Season Folder: ${series.seasonFolder ? 'Yes' : 'No'}
• Quality Profile: ${series.qualityProfileId}

📍 Path: ${series.path}
🔗 TVDB ID: ${series.tvdbId}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting series information: ${error.message}`
        }]
      };
    }
  }

  async monitorSeries({ seriesId, monitored }) {
    try {
      const series = await this.apiRequest(`/api/v3/series/${seriesId}`);
      series.monitored = monitored;
      
      await this.apiRequest(`/api/v3/series/${seriesId}`, 'PUT', series);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('sonarr-monitor-change', { seriesId, monitored, title: series.title });
      }

      return {
        content: [{
          type: 'text',
          text: `✅ Successfully ${monitored ? 'enabled' : 'disabled'} monitoring for "${series.title}"`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error updating monitoring status: ${error.message}`
        }]
      };
    }
  }

  async getQueue({ page = 1, pageSize = 20 }) {
    try {
      const queue = await this.apiRequest(`/api/v3/queue?page=${page}&pageSize=${pageSize}`);
      
      if (!queue.records || queue.records.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📭 Download queue is empty'
          }]
        };
      }

      const queueText = queue.records.map(item => {
        const progress = item.size > 0 ? ((item.size - item.sizeleft) / item.size * 100).toFixed(1) : 0;
        return `• ${item.series?.title} - S${String(item.episode?.seasonNumber).padStart(2, '0')}E${String(item.episode?.episodeNumber).padStart(2, '0')}
  Status: ${item.status} | Progress: ${progress}%
  Quality: ${item.quality?.quality?.name} | Size: ${this.formatBytes(item.size)}
  ETA: ${item.timeleft || 'Unknown'}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `📥 Download Queue (Page ${page}/${Math.ceil(queue.totalRecords / pageSize)}):

${queueText}

Total Items: ${queue.totalRecords}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting queue: ${error.message}`
        }]
      };
    }
  }

  async getCalendar({ start, end }) {
    try {
      const today = start || new Date().toISOString().split('T')[0];
      const nextWeek = end || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
      
      const calendar = await this.apiRequest(`/api/v3/calendar?start=${today}&end=${nextWeek}`);
      
      if (calendar.length === 0) {
        return {
          content: [{
            type: 'text',
            text: `📅 No episodes scheduled between ${today} and ${nextWeek}`
          }]
        };
      }

      const groupedByDate = calendar.reduce((acc, episode) => {
        const airDate = episode.airDate;
        if (!acc[airDate]) acc[airDate] = [];
        acc[airDate].push(episode);
        return acc;
      }, {});

      const calendarText = Object.entries(groupedByDate)
        .sort(([a], [b]) => new Date(a) - new Date(b))
        .map(([date, episodes]) => {
          const formattedDate = new Date(date).toLocaleDateString();
          const episodesList = episodes.map(ep => 
            `  • ${ep.series.title} - S${String(ep.seasonNumber).padStart(2, '0')}E${String(ep.episodeNumber).padStart(2, '0')}: ${ep.title}
    ${ep.hasFile ? '✅ Downloaded' : '⏳ Not Downloaded'}`
          ).join('\n');
          
          return `📅 ${formattedDate}:\n${episodesList}`;
        }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `📺 Upcoming Episodes (${today} to ${nextWeek}):

${calendarText}

Total Episodes: ${calendar.length}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting calendar: ${error.message}`
        }]
      };
    }
  }

  async refreshSeries({ seriesId }) {
    try {
      await this.apiRequest('/api/v3/command', 'POST', {
        name: 'RefreshSeries',
        seriesId: seriesId
      });

      const series = await this.apiRequest(`/api/v3/series/${seriesId}`);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('sonarr-refresh', { seriesId, title: series.title });
      }

      return {
        content: [{
          type: 'text',
          text: `🔄 Refresh triggered for "${series.title}". Check the queue for progress.`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error triggering refresh: ${error.message}`
        }]
      };
    }
  }

  async getSystemStatus() {
    try {
      const status = await this.apiRequest('/api/v3/system/status');
      const diskSpace = await this.apiRequest('/api/v3/diskspace');
      
      return {
        content: [{
          type: 'text',
          text: `🖥️ Sonarr System Status:

Version: ${status.version}
Build Date: ${new Date(status.buildTime).toLocaleDateString()}
Runtime Version: ${status.runtimeVersion}
Database Version: ${status.databaseVersion}

🔧 Configuration:
• Start Time: ${new Date(status.startTime).toLocaleString()}
• App Data: ${status.appData}
• OS Name: ${status.osName}
• Is Production: ${status.isProduction ? 'Yes' : 'No'}

💾 Disk Space:
${diskSpace.map(disk => 
  `• ${disk.label}: ${this.formatBytes(disk.freeSpace)} free / ${this.formatBytes(disk.totalSpace)} total (${((disk.totalSpace - disk.freeSpace) / disk.totalSpace * 100).toFixed(1)}% used)`
).join('\n')}`
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

  async searchEpisodes({ seriesId, seasonNumber }) {
    try {
      const command = {
        name: 'EpisodeSearch',
        seriesId: seriesId
      };

      if (seasonNumber !== undefined) {
        command.name = 'SeasonSearch';
        command.seasonNumber = seasonNumber;
      }

      await this.apiRequest('/api/v3/command', 'POST', command);
      
      const series = await this.apiRequest(`/api/v3/series/${seriesId}`);
      const searchType = seasonNumber !== undefined ? `Season ${seasonNumber}` : 'all missing episodes';
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('sonarr-search-episodes', { seriesId, seasonNumber, title: series.title });
      }

      return {
        content: [{
          type: 'text',
          text: `🔍 Episode search triggered for "${series.title}" (${searchType}). Check the queue for progress.`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error triggering episode search: ${error.message}`
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
    console.log(`Sonarr MCP Server running on port ${this.port}`);
  }
}

module.exports = SonarrMCP;