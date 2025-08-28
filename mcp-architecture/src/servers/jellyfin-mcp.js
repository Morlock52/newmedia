/**
 * Jellyfin MCP Server
 * Provides Model Context Protocol interface for Jellyfin media server
 * 
 * Features:
 * - Media library management
 * - Playback control
 * - User management
 * - Statistics and analytics
 * - Real-time updates via WebSocket
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/index.js');
const axios = require('axios');
const winston = require('winston');

class JellyfinMCP {
  constructor(options = {}) {
    this.port = options.port || 3001;
    this.jellyfinUrl = options.jellyfinUrl || 'http://localhost:8096';
    this.apiKey = options.apiKey;
    this.io = options.io;
    this.isRunning = false;
    this.lastActivity = null;
    this.requestCount = 0;
    this.errorCount = 0;

    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.label({ label: 'JellyfinMCP' }),
        winston.format.json()
      ),
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/jellyfin-mcp.log' })
      ]
    });

    this.server = new McpServer(
      {
        name: 'jellyfin-mcp',
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
  }

  setupTools() {
    // Media Library Tools
    this.server.tool('search_media', {
      description: 'Search for movies, TV shows, music, and other media in Jellyfin library',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query for media content'
          },
          mediaType: {
            type: 'string',
            enum: ['Movie', 'Series', 'Episode', 'Audio', 'All'],
            description: 'Type of media to search for'
          },
          limit: {
            type: 'number',
            default: 20,
            description: 'Maximum number of results to return'
          }
        },
        required: ['query']
      }
    }, this.searchMedia.bind(this));

    this.server.tool('get_library_stats', {
      description: 'Get statistics about the Jellyfin media library',
      parameters: {
        type: 'object',
        properties: {
          userId: {
            type: 'string',
            description: 'User ID for personalized stats (optional)'
          }
        }
      }
    }, this.getLibraryStats.bind(this));

    this.server.tool('get_recently_added', {
      description: 'Get recently added media items',
      parameters: {
        type: 'object',
        properties: {
          limit: {
            type: 'number',
            default: 10,
            description: 'Number of items to return'
          },
          mediaType: {
            type: 'string',
            enum: ['Movie', 'Series', 'Episode', 'Audio'],
            description: 'Filter by media type'
          }
        }
      }
    }, this.getRecentlyAdded.bind(this));

    this.server.tool('get_playing_sessions', {
      description: 'Get currently active playback sessions',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getPlayingSessions.bind(this));

    this.server.tool('control_playback', {
      description: 'Control playback for a specific session',
      parameters: {
        type: 'object',
        properties: {
          sessionId: {
            type: 'string',
            description: 'Session ID to control'
          },
          command: {
            type: 'string',
            enum: ['Play', 'Pause', 'Stop', 'Next', 'Previous'],
            description: 'Playback command to execute'
          }
        },
        required: ['sessionId', 'command']
      }
    }, this.controlPlayback.bind(this));

    this.server.tool('get_user_activity', {
      description: 'Get user activity and viewing history',
      parameters: {
        type: 'object',
        properties: {
          userId: {
            type: 'string',
            description: 'User ID to get activity for'
          },
          days: {
            type: 'number',
            default: 30,
            description: 'Number of days to look back'
          }
        }
      }
    }, this.getUserActivity.bind(this));

    this.server.tool('manage_library', {
      description: 'Trigger library scan or refresh operations',
      parameters: {
        type: 'object',
        properties: {
          libraryId: {
            type: 'string',
            description: 'Library ID to scan (optional, scans all if not provided)'
          },
          operation: {
            type: 'string',
            enum: ['scan', 'refresh', 'identify'],
            description: 'Type of operation to perform'
          }
        },
        required: ['operation']
      }
    }, this.manageLibrary.bind(this));

    this.server.tool('get_media_info', {
      description: 'Get detailed information about a specific media item',
      parameters: {
        type: 'object',
        properties: {
          itemId: {
            type: 'string',
            description: 'Media item ID'
          },
          userId: {
            type: 'string',
            description: 'User ID for personalized info (optional)'
          }
        },
        required: ['itemId']
      }
    }, this.getMediaInfo.bind(this));
  }

  setupResources() {
    this.server.resource('jellyfin://libraries', {
      description: 'Jellyfin media libraries',
      mimeType: 'application/json'
    }, this.getLibraries.bind(this));

    this.server.resource('jellyfin://users', {
      description: 'Jellyfin users',
      mimeType: 'application/json'
    }, this.getUsers.bind(this));

    this.server.resource('jellyfin://server-info', {
      description: 'Jellyfin server information',
      mimeType: 'application/json'
    }, this.getServerInfo.bind(this));
  }

  async makeRequest(endpoint, options = {}) {
    try {
      this.requestCount++;
      this.lastActivity = new Date();

      const config = {
        baseURL: this.jellyfinUrl,
        url: endpoint,
        headers: {
          'X-Emby-Authorization': `MediaBrowser Token="${this.apiKey}"`,
          'Content-Type': 'application/json'
        },
        ...options
      };

      const response = await axios(config);
      
      // Emit activity to WebSocket clients
      if (this.io) {
        this.io.to('logs-jellyfin').emit('mcp-activity', {
          server: 'jellyfin',
          endpoint,
          method: options.method || 'GET',
          status: response.status,
          timestamp: new Date()
        });
      }

      return response.data;
    } catch (error) {
      this.errorCount++;
      this.logger.error('Jellyfin API request failed:', {
        endpoint,
        error: error.message,
        status: error.response?.status
      });
      
      if (this.io) {
        this.io.to('logs-jellyfin').emit('mcp-error', {
          server: 'jellyfin',
          endpoint,
          error: error.message,
          timestamp: new Date()
        });
      }
      
      throw error;
    }
  }

  async searchMedia({ query, mediaType = 'All', limit = 20 }) {
    const params = new URLSearchParams({
      searchTerm: query,
      limit: limit.toString(),
      IncludeItemTypes: mediaType === 'All' ? '' : mediaType
    });

    const data = await this.makeRequest(`/Items?${params}`);
    
    return {
      success: true,
      results: data.Items?.map(item => ({
        id: item.Id,
        name: item.Name,
        type: item.Type,
        year: item.ProductionYear,
        overview: item.Overview,
        genres: item.Genres,
        rating: item.CommunityRating,
        runtime: item.RunTimeTicks ? Math.round(item.RunTimeTicks / 600000000) : null,
        imageUrl: item.ImageTags?.Primary ? 
          `${this.jellyfinUrl}/Items/${item.Id}/Images/Primary` : null
      })) || [],
      total: data.TotalRecordCount || 0
    };
  }

  async getLibraryStats({ userId }) {
    const endpoint = userId ? `/Users/${userId}/Items/Counts` : '/Items/Counts';
    const counts = await this.makeRequest(endpoint);
    
    const libraries = await this.makeRequest('/Library/VirtualFolders');
    
    return {
      success: true,
      stats: {
        movies: counts.MovieCount || 0,
        series: counts.SeriesCount || 0,
        episodes: counts.EpisodeCount || 0,
        songs: counts.SongCount || 0,
        albums: counts.AlbumCount || 0,
        artists: counts.ArtistCount || 0,
        libraries: libraries.length || 0
      }
    };
  }

  async getRecentlyAdded({ limit = 10, mediaType }) {
    const params = new URLSearchParams({
      Limit: limit.toString(),
      Recursive: 'true',
      SortBy: 'DateCreated',
      SortOrder: 'Descending'
    });

    if (mediaType) {
      params.append('IncludeItemTypes', mediaType);
    }

    const data = await this.makeRequest(`/Items?${params}`);
    
    return {
      success: true,
      items: data.Items?.map(item => ({
        id: item.Id,
        name: item.Name,
        type: item.Type,
        dateAdded: item.DateCreated,
        year: item.ProductionYear,
        overview: item.Overview?.substring(0, 200) + '...',
        imageUrl: item.ImageTags?.Primary ? 
          `${this.jellyfinUrl}/Items/${item.Id}/Images/Primary` : null
      })) || []
    };
  }

  async getPlayingSessions() {
    const sessions = await this.makeRequest('/Sessions');
    
    const activeSessions = sessions.filter(session => 
      session.NowPlayingItem && session.PlayState
    );

    return {
      success: true,
      sessions: activeSessions.map(session => ({
        id: session.Id,
        userId: session.UserId,
        userName: session.UserName,
        deviceName: session.DeviceName,
        client: session.Client,
        nowPlaying: {
          id: session.NowPlayingItem.Id,
          name: session.NowPlayingItem.Name,
          type: session.NowPlayingItem.Type,
          year: session.NowPlayingItem.ProductionYear
        },
        playState: {
          isPaused: session.PlayState.IsPaused,
          positionTicks: session.PlayState.PositionTicks,
          canControl: session.SupportsRemoteControl
        }
      }))
    };
  }

  async controlPlayback({ sessionId, command }) {
    const endpoint = `/Sessions/${sessionId}/Playing/${command}`;
    
    await this.makeRequest(endpoint, { method: 'POST' });
    
    return {
      success: true,
      message: `Playback command '${command}' sent to session ${sessionId}`
    };
  }

  async getUserActivity({ userId, days = 30 }) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    const params = new URLSearchParams({
      UserId: userId,
      StartDate: startDate.toISOString(),
      Limit: '100'
    });

    const activity = await this.makeRequest(`/UserActivity?${params}`);
    
    return {
      success: true,
      activity: activity.Items?.map(item => ({
        date: item.Date,
        type: item.Type,
        itemName: item.Name,
        deviceName: item.DeviceName,
        userId: item.UserId
      })) || []
    };
  }

  async manageLibrary({ libraryId, operation }) {
    let endpoint;
    
    if (libraryId) {
      endpoint = `/Library/VirtualFolders/${libraryId}/${operation}`;
    } else {
      endpoint = `/Library/${operation}`;
    }
    
    await this.makeRequest(endpoint, { method: 'POST' });
    
    return {
      success: true,
      message: `Library ${operation} operation started${libraryId ? ` for library ${libraryId}` : ''}`
    };
  }

  async getMediaInfo({ itemId, userId }) {
    const endpoint = userId ? 
      `/Users/${userId}/Items/${itemId}` : 
      `/Items/${itemId}`;
    
    const item = await this.makeRequest(endpoint);
    
    return {
      success: true,
      item: {
        id: item.Id,
        name: item.Name,
        type: item.Type,
        overview: item.Overview,
        year: item.ProductionYear,
        genres: item.Genres,
        rating: item.CommunityRating,
        runtime: item.RunTimeTicks ? Math.round(item.RunTimeTicks / 600000000) : null,
        cast: item.People?.filter(p => p.Type === 'Actor').slice(0, 10),
        directors: item.People?.filter(p => p.Type === 'Director'),
        writers: item.People?.filter(p => p.Type === 'Writer'),
        studios: item.Studios,
        tags: item.Tags,
        officialRating: item.OfficialRating,
        playCount: item.UserData?.PlayCount || 0,
        isFavorite: item.UserData?.IsFavorite || false,
        dateCreated: item.DateCreated,
        path: item.Path
      }
    };
  }

  async getLibraries() {
    const libraries = await this.makeRequest('/Library/VirtualFolders');
    
    return {
      data: libraries.map(lib => ({
        id: lib.ItemId,
        name: lib.Name,
        type: lib.CollectionType,
        locations: lib.Locations
      }))
    };
  }

  async getUsers() {
    const users = await this.makeRequest('/Users');
    
    return {
      data: users.map(user => ({
        id: user.Id,
        name: user.Name,
        lastLoginDate: user.LastLoginDate,
        lastActivityDate: user.LastActivityDate,
        isAdministrator: user.Policy?.IsAdministrator || false,
        isDisabled: user.Policy?.IsDisabled || false
      }))
    };
  }

  async getServerInfo() {
    const info = await this.makeRequest('/System/Info');
    
    return {
      data: {
        serverName: info.ServerName,
        version: info.Version,
        operatingSystem: info.OperatingSystem,
        architecture: info.SystemArchitecture,
        localAddress: info.LocalAddress,
        webSocketPortNumber: info.WebSocketPortNumber,
        hasUpdateAvailable: info.HasUpdateAvailable,
        uptime: info.StartupWizardCompleted
      }
    };
  }

  async start() {
    try {
      if (!this.apiKey) {
        throw new Error('Jellyfin API key is required');
      }

      // Test connection
      await this.makeRequest('/System/Info');

      await this.server.listen({ port: this.port });
      this.isRunning = true;
      
      this.logger.info(`Jellyfin MCP server started on port ${this.port}`);
      
      return true;
    } catch (error) {
      this.logger.error('Failed to start Jellyfin MCP server:', error);
      throw error;
    }
  }

  async stop() {
    try {
      await this.server.close();
      this.isRunning = false;
      this.logger.info('Jellyfin MCP server stopped');
    } catch (error) {
      this.logger.error('Error stopping Jellyfin MCP server:', error);
      throw error;
    }
  }
}

module.exports = JellyfinMCP;