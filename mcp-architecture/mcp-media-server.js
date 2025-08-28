#!/usr/bin/env node

/**
 * Production-Ready Media Services MCP Server
 * 
 * A comprehensive MCP server that integrates with:
 * - Jellyfin (Media streaming)
 * - Sonarr (TV show management)
 * - Radarr (Movie management) 
 * - Prowlarr (Indexer management)
 * - qBittorrent (Download client)
 * - Bazarr (Subtitle management)
 * - Lidarr (Music management)
 * 
 * Features:
 * - Stdio transport for Claude Desktop
 * - Proper error handling and logging
 * - Authentication support
 * - Caching for performance
 * - Full MCP 1.0 protocol compliance
 */

const readline = require('readline');
const https = require('https');
const http = require('http');
const fs = require('fs').promises;
const path = require('path');

class MediaServicesMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'media-services-mcp',
      version: '2.0.0',
      description: 'Complete media services management via MCP'
    };
    
    this.protocolVersion = '1.0';
    this.capabilities = {
      tools: {},
      resources: {},
      logging: {}
    };
    
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
    
    // Keep-alive interval
    this.keepAliveInterval = null;
    
    // Define available tools
    this.tools = this._defineTools();
    this.resources = this._defineResources();
  }
  
  log(message, level = 'info') {
    const timestamp = new Date().toISOString();
    const logLevel = level.toUpperCase();
    const logMessage = `[${this.serverInfo.name}] ${timestamp} ${logLevel}: ${message}`;
    
    if (process.env.MCP_DEBUG === 'true' || level === 'error') {
      process.stderr.write(logMessage + '\n');
    }
  }
  
  _defineTools() {
    return [
      // System status and health
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
      
      // Media search and discovery
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
      
      // Library management
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
      
      // Recent activity
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
      
      // Download management
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
      
      // Media requests
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
      },
      
      // Subtitle management
      {
        name: 'manage_subtitles',
        description: 'Search and download subtitles for media',
        inputSchema: {
          type: 'object',
          properties: {
            action: { type: 'string', enum: ['search', 'download', 'wanted'], description: 'Action to perform' },
            mediaId: { type: 'string', description: 'Media ID for subtitle search' },
            language: { type: 'string', description: 'Subtitle language', default: 'en' }
          },
          required: ['action']
        }
      },
      
      // Indexer management
      {
        name: 'manage_indexers',
        description: 'View and manage torrent/usenet indexers',
        inputSchema: {
          type: 'object',
          properties: {
            action: { type: 'string', enum: ['list', 'test', 'stats'], description: 'Action to perform' },
            indexerId: { type: 'number', description: 'Specific indexer ID' }
          },
          required: ['action']
        }
      },
      
      // Calendar and upcoming
      {
        name: 'get_calendar',
        description: 'Get upcoming releases calendar',
        inputSchema: {
          type: 'object',
          properties: {
            days: { type: 'number', description: 'Days ahead to check', default: 7 },
            service: { type: 'string', enum: ['sonarr', 'radarr', 'lidarr', 'all'], default: 'all' }
          }
        }
      }
    ];
  }
  
  _defineResources() {
    return [
      { uri: 'media://system/status', name: 'System Status', mimeType: 'application/json' },
      { uri: 'media://library/stats', name: 'Library Statistics', mimeType: 'application/json' },
      { uri: 'media://activity/recent', name: 'Recent Activity', mimeType: 'application/json' },
      { uri: 'media://downloads/active', name: 'Active Downloads', mimeType: 'application/json' },
      { uri: 'media://calendar/upcoming', name: 'Upcoming Releases', mimeType: 'application/json' },
      { uri: 'media://indexers/status', name: 'Indexer Status', mimeType: 'application/json' },
      { uri: 'media://config/services', name: 'Service Configuration', mimeType: 'application/json' }
    ];
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
        'User-Agent': 'MediaServices-MCP/2.0.0'
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
  
  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    try {
      switch (name) {
        case 'get_system_status':
          return await this._getSystemStatus(args.detailed || false);
          
        case 'search_media':
          return await this._searchMedia(args.query, args.type || 'all', args.limit || 20);
          
        case 'get_library_stats':
          return await this._getLibraryStats(args.service || 'all');
          
        case 'get_recent_activity':
          return await this._getRecentActivity(args.hours || 24, args.limit || 50);
          
        case 'manage_downloads':
          return await this._manageDownloads(args.action, args.hash);
          
        case 'add_media_request':
          return await this._addMediaRequest(args);
          
        case 'manage_subtitles':
          return await this._manageSubtitles(args.action, args.mediaId, args.language || 'en');
          
        case 'manage_indexers':
          return await this._manageIndexers(args.action, args.indexerId);
          
        case 'get_calendar':
          return await this._getCalendar(args.days || 7, args.service || 'all');
          
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
      // Search across different services based on type
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
      
      if (type === 'tv' || type === 'all') {
        try {
          const sonarrResults = await this.makeRequest('sonarr', `/api/v3/series/lookup?term=${encodeURIComponent(query)}`);
          results.push(...sonarrResults.slice(0, limit).map(series => ({
            service: 'sonarr',
            type: 'tv',
            title: series.title,
            year: series.year,
            overview: series.overview,
            status: series.status,
            tvdbId: series.tvdbId
          })));
        } catch (error) {
          this.log(`Sonarr search failed: ${error.message}`, 'error');
        }
      }
      
      if (type === 'music' || type === 'all') {
        try {
          const lidarrResults = await this.makeRequest('lidarr', `/api/v1/artist/lookup?term=${encodeURIComponent(query)}`);
          results.push(...lidarrResults.slice(0, limit).map(artist => ({
            service: 'lidarr',
            type: 'music',
            title: artist.artistName,
            overview: artist.overview,
            status: artist.status,
            musicBrainzId: artist.foreignArtistId
          })));
        } catch (error) {
          this.log(`Lidarr search failed: ${error.message}`, 'error');
        }
      }
      
      // Also search Jellyfin library
      try {
        const jellyfinResults = await this.makeRequest('jellyfin', `/Items?searchTerm=${encodeURIComponent(query)}&limit=${limit}`);
        if (jellyfinResults.Items) {
          results.push(...jellyfinResults.Items.map(item => ({
            service: 'jellyfin',
            type: item.Type.toLowerCase(),
            title: item.Name,
            year: item.ProductionYear,
            overview: item.Overview,
            id: item.Id
          })));
        }
      } catch (error) {
        this.log(`Jellyfin search failed: ${error.message}`, 'error');
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
    const stats = {};
    
    try {
      if (service === 'jellyfin' || service === 'all') {
        try {
          const libraryStats = await this.makeRequest('jellyfin', '/Items/Counts');
          stats.jellyfin = {
            movies: libraryStats.MovieCount || 0,
            series: libraryStats.SeriesCount || 0,
            episodes: libraryStats.EpisodeCount || 0,
            songs: libraryStats.SongCount || 0,
            albums: libraryStats.AlbumCount || 0,
            artists: libraryStats.ArtistCount || 0
          };
        } catch (error) {
          stats.jellyfin = { error: error.message };
        }
      }
      
      if (service === 'sonarr' || service === 'all') {
        try {
          const sonarrSeries = await this.makeRequest('sonarr', '/api/v3/series');
          const sonarrEpisodes = await this.makeRequest('sonarr', '/api/v3/episode');
          stats.sonarr = {
            series: sonarrSeries.length,
            episodes: sonarrEpisodes.length,
            monitored: sonarrSeries.filter(s => s.monitored).length
          };
        } catch (error) {
          stats.sonarr = { error: error.message };
        }
      }
      
      if (service === 'radarr' || service === 'all') {
        try {
          const radarrMovies = await this.makeRequest('radarr', '/api/v3/movie');
          stats.radarr = {
            movies: radarrMovies.length,
            monitored: radarrMovies.filter(m => m.monitored).length,
            downloaded: radarrMovies.filter(m => m.hasFile).length
          };
        } catch (error) {
          stats.radarr = { error: error.message };
        }
      }
      
      if (service === 'lidarr' || service === 'all') {
        try {
          const lidarrArtists = await this.makeRequest('lidarr', '/api/v1/artist');
          const lidarrAlbums = await this.makeRequest('lidarr', '/api/v1/album');
          stats.lidarr = {
            artists: lidarrArtists.length,
            albums: lidarrAlbums.length,
            monitored: lidarrArtists.filter(a => a.monitored).length
          };
        } catch (error) {
          stats.lidarr = { error: error.message };
        }
      }
      
    } catch (error) {
      this.log(`Library stats error: ${error.message}`, 'error');
    }
    
    return {
      content: [{
        type: 'text',
        text: `📊 **Media Library Statistics**\n\n` +
              Object.entries(stats).map(([serviceName, data]) => {
                if (data.error) {
                  return `❌ **${serviceName}**: Error - ${data.error}`;
                }
                
                const items = [];
                Object.entries(data).forEach(([key, value]) => {
                  items.push(`${key}: ${value}`);
                });
                
                return `✅ **${serviceName.charAt(0).toUpperCase() + serviceName.slice(1)}**:\n   ${items.join(' | ')}`;
              }).join('\n\n')
      }]
    };
  }
  
  async _getRecentActivity(hours, limit) {
    const activities = [];
    const cutoffTime = new Date(Date.now() - (hours * 60 * 60 * 1000));
    
    try {
      // Get recent additions from Jellyfin
      try {
        const recentItems = await this.makeRequest('jellyfin', `/Items/Latest?limit=${limit}`);
        activities.push(...recentItems.map(item => ({
          service: 'jellyfin',
          type: 'added',
          title: item.Name,
          mediaType: item.Type,
          date: item.DateCreated,
          timestamp: new Date(item.DateCreated)
        })));
      } catch (error) {
        this.log(`Jellyfin recent activity failed: ${error.message}`, 'error');
      }
      
      // Get download history from Radarr
      try {
        const radarrHistory = await this.makeRequest('radarr', `/api/v3/history?pageSize=${limit}`);
        activities.push(...radarrHistory.records
          .filter(record => new Date(record.date) > cutoffTime)
          .map(record => ({
            service: 'radarr',
            type: record.eventType,
            title: record.movie?.title || 'Unknown Movie',
            mediaType: 'movie',
            date: record.date,
            timestamp: new Date(record.date)
          })));
      } catch (error) {
        this.log(`Radarr history failed: ${error.message}`, 'error');
      }
      
      // Get download history from Sonarr
      try {
        const sonarrHistory = await this.makeRequest('sonarr', `/api/v3/history?pageSize=${limit}`);
        activities.push(...sonarrHistory.records
          .filter(record => new Date(record.date) > cutoffTime)
          .map(record => ({
            service: 'sonarr',
            type: record.eventType,
            title: record.series?.title || 'Unknown Series',
            mediaType: 'tv',
            date: record.date,
            timestamp: new Date(record.date)
          })));
      } catch (error) {
        this.log(`Sonarr history failed: ${error.message}`, 'error');
      }
      
    } catch (error) {
      this.log(`Recent activity error: ${error.message}`, 'error');
    }
    
    // Sort by timestamp and limit
    const sortedActivities = activities
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit);
    
    return {
      content: [{
        type: 'text',
        text: `🕒 **Recent Activity (Last ${hours} hours)**\n\n` +
              (sortedActivities.length > 0 ? 
                sortedActivities.map((activity, index) => 
                  `${index + 1}. **${activity.title}** (${activity.mediaType})\n` +
                  `   ${activity.service} - ${activity.type} - ${new Date(activity.date).toLocaleString()}`
                ).join('\n\n') :
                'No recent activity found.')
      }]
    };
  }
  
  async _manageDownloads(action, hash) {
    try {
      switch (action) {
        case 'list':
          const torrents = await this.makeRequest('qbittorrent', '/api/v2/torrents/info');
          return {
            content: [{
              type: 'text',
              text: `📥 **Active Downloads**\n\n` +
                    torrents.map((torrent, index) => 
                      `${index + 1}. **${torrent.name}**\n` +
                      `   Progress: ${Math.round(torrent.progress * 100)}% | ` +
                      `Speed: ${this._formatBytes(torrent.dlspeed)}/s | ` +
                      `State: ${torrent.state}\n` +
                      `   Size: ${this._formatBytes(torrent.size)} | ` +
                      `ETA: ${torrent.eta > 0 ? this._formatTime(torrent.eta) : 'Unknown'}`
                    ).join('\n\n')
            }]
          };
          
        case 'pause':
          if (!hash) throw new Error('Hash required for pause action');
          await this.makeRequest('qbittorrent', `/api/v2/torrents/pause`, { method: 'POST', body: { hashes: hash }});
          return { content: [{ type: 'text', text: `⏸️ Download paused successfully` }] };
          
        case 'resume':
          if (!hash) throw new Error('Hash required for resume action');
          await this.makeRequest('qbittorrent', `/api/v2/torrents/resume`, { method: 'POST', body: { hashes: hash }});
          return { content: [{ type: 'text', text: `▶️ Download resumed successfully` }] };
          
        case 'delete':
          if (!hash) throw new Error('Hash required for delete action');
          await this.makeRequest('qbittorrent', `/api/v2/torrents/delete`, { method: 'POST', body: { hashes: hash, deleteFiles: false }});
          return { content: [{ type: 'text', text: `🗑️ Download deleted successfully` }] };
          
        default:
          throw new Error(`Unknown action: ${action}`);
      }
    } catch (error) {
      throw new Error(`Download management failed: ${error.message}`);
    }
  }
  
  async _addMediaRequest(args) {
    const { title, type, year, quality, monitor } = args;
    
    try {
      let service, endpoint, body;
      
      switch (type) {
        case 'movie':
          service = 'radarr';
          endpoint = '/api/v3/movie';
          
          // First lookup the movie
          const movieResults = await this.makeRequest('radarr', `/api/v3/movie/lookup?term=${encodeURIComponent(title)}`);
          const movie = movieResults.find(m => !year || m.year === year);
          
          if (!movie) throw new Error('Movie not found');
          
          body = {
            ...movie,
            monitored: monitor,
            qualityProfileId: 4, // HD-1080p default
            rootFolderPath: '/movies'
          };
          break;
          
        case 'tv':
          service = 'sonarr';
          endpoint = '/api/v3/series';
          
          // First lookup the series
          const seriesResults = await this.makeRequest('sonarr', `/api/v3/series/lookup?term=${encodeURIComponent(title)}`);
          const series = seriesResults.find(s => !year || s.year === year);
          
          if (!series) throw new Error('TV series not found');
          
          body = {
            ...series,
            monitored: monitor,
            qualityProfileId: 4, // HD-1080p default
            rootFolderPath: '/tv'
          };
          break;
          
        case 'music':
          service = 'lidarr';
          endpoint = '/api/v1/artist';
          
          // First lookup the artist
          const artistResults = await this.makeRequest('lidarr', `/api/v1/artist/lookup?term=${encodeURIComponent(title)}`);
          const artist = artistResults[0];
          
          if (!artist) throw new Error('Artist not found');
          
          body = {
            ...artist,
            monitored: monitor,
            qualityProfileId: 1,
            rootFolderPath: '/music'
          };
          break;
          
        default:
          throw new Error(`Unsupported media type: ${type}`);
      }
      
      await this.makeRequest(service, endpoint, { method: 'POST', body });
      
      return {
        content: [{
          type: 'text',
          text: `✅ **Media Request Added**\n\n` +
                `Title: ${title}\n` +
                `Type: ${type}\n` +
                `Year: ${year || 'N/A'}\n` +
                `Service: ${service}\n` +
                `Status: ${monitor ? 'Monitoring' : 'Added but not monitoring'}`
        }]
      };
      
    } catch (error) {
      throw new Error(`Failed to add media request: ${error.message}`);
    }
  }
  
  async _manageSubtitles(action, mediaId, language) {
    try {
      switch (action) {
        case 'search':
          if (!mediaId) throw new Error('Media ID required for subtitle search');
          const searchResults = await this.makeRequest('bazarr', `/api/episodes/${mediaId}/subtitles`);
          return {
            content: [{
              type: 'text',
              text: `🎬 **Available Subtitles for Media ID: ${mediaId}**\n\n` +
                    searchResults.map(sub => 
                      `Language: ${sub.name} | Provider: ${sub.provider} | Score: ${sub.score}`
                    ).join('\n')
            }]
          };
          
        case 'wanted':
          const wantedSubs = await this.makeRequest('bazarr', '/api/wanted');
          return {
            content: [{
              type: 'text',
              text: `📋 **Wanted Subtitles**\n\n` +
                    `Total: ${wantedSubs.total} missing subtitles\n` +
                    `Recent: ${wantedSubs.data.slice(0, 10).map(item => 
                      `${item.seriesTitle || item.movieTitle} - ${item.missing_subtitles.join(', ')}`
                    ).join('\n')}`
            }]
          };
          
        case 'download':
          if (!mediaId) throw new Error('Media ID required for subtitle download');
          await this.makeRequest('bazarr', `/api/episodes/${mediaId}/subtitles`, { 
            method: 'POST', 
            body: { language, forced: false, hi: false }
          });
          return {
            content: [{
              type: 'text',
              text: `📥 Subtitle download started for media ID: ${mediaId}, language: ${language}`
            }]
          };
          
        default:
          throw new Error(`Unknown subtitle action: ${action}`);
      }
    } catch (error) {
      throw new Error(`Subtitle management failed: ${error.message}`);
    }
  }
  
  async _manageIndexers(action, indexerId) {
    try {
      switch (action) {
        case 'list':
          const indexers = await this.makeRequest('prowlarr', '/api/v1/indexer');
          return {
            content: [{
              type: 'text',
              text: `🌐 **Available Indexers**\n\n` +
                    indexers.map(indexer => 
                      `${indexer.enable ? '✅' : '❌'} **${indexer.name}**\n` +
                      `   ID: ${indexer.id} | Protocol: ${indexer.protocol} | ` +
                      `Categories: ${indexer.capabilities?.categories?.length || 0}`
                    ).join('\n\n')
            }]
          };
          
        case 'test':
          if (!indexerId) throw new Error('Indexer ID required for testing');
          const testResult = await this.makeRequest('prowlarr', `/api/v1/indexer/test/${indexerId}`, { method: 'POST' });
          return {
            content: [{
              type: 'text',
              text: `🧪 **Indexer Test Result**\n\nID: ${indexerId}\nResult: ${testResult.isValid ? '✅ Success' : '❌ Failed'}\n${testResult.validationFailures?.map(f => f.errorMessage).join('\n') || ''}`
            }]
          };
          
        case 'stats':
          const stats = await this.makeRequest('prowlarr', '/api/v1/indexerstats');
          return {
            content: [{
              type: 'text',
              text: `📈 **Indexer Statistics**\n\n` +
                    stats.indexers.map(stat => 
                      `**${stat.indexerName}**:\n` +
                      `   Queries: ${stat.numberOfQueries} | Grabs: ${stat.numberOfGrabs} | ` +
                      `Average Response: ${stat.averageResponseTime}ms`
                    ).join('\n\n')
            }]
          };
          
        default:
          throw new Error(`Unknown indexer action: ${action}`);
      }
    } catch (error) {
      throw new Error(`Indexer management failed: ${error.message}`);
    }
  }
  
  async _getCalendar(days, service) {
    const endDate = new Date();
    endDate.setDate(endDate.getDate() + days);
    const startDate = new Date();
    
    const calendar = [];
    
    try {
      if (service === 'sonarr' || service === 'all') {
        try {
          const sonarrCalendar = await this.makeRequest('sonarr', 
            `/api/v3/calendar?start=${startDate.toISOString().split('T')[0]}&end=${endDate.toISOString().split('T')[0]}`);
          calendar.push(...sonarrCalendar.map(episode => ({
            service: 'sonarr',
            type: 'episode',
            title: `${episode.series.title} - S${episode.seasonNumber.toString().padStart(2, '0')}E${episode.episodeNumber.toString().padStart(2, '0')}`,
            subtitle: episode.title,
            airDate: episode.airDate,
            hasFile: episode.hasFile
          })));
        } catch (error) {
          this.log(`Sonarr calendar failed: ${error.message}`, 'error');
        }
      }
      
      if (service === 'radarr' || service === 'all') {
        try {
          const radarrCalendar = await this.makeRequest('radarr', 
            `/api/v3/calendar?start=${startDate.toISOString().split('T')[0]}&end=${endDate.toISOString().split('T')[0]}`);
          calendar.push(...radarrCalendar.map(movie => ({
            service: 'radarr',
            type: 'movie',
            title: movie.title,
            subtitle: `${movie.year}`,
            airDate: movie.inCinemas || movie.digitalRelease,
            hasFile: movie.hasFile
          })));
        } catch (error) {
          this.log(`Radarr calendar failed: ${error.message}`, 'error');
        }
      }
      
      if (service === 'lidarr' || service === 'all') {
        try {
          const lidarrCalendar = await this.makeRequest('lidarr', 
            `/api/v1/calendar?start=${startDate.toISOString().split('T')[0]}&end=${endDate.toISOString().split('T')[0]}`);
          calendar.push(...lidarrCalendar.map(album => ({
            service: 'lidarr',
            type: 'album',
            title: `${album.artist.artistName} - ${album.title}`,
            subtitle: '',
            airDate: album.releaseDate,
            hasFile: album.statistics?.trackFileCount > 0
          })));
        } catch (error) {
          this.log(`Lidarr calendar failed: ${error.message}`, 'error');
        }
      }
      
    } catch (error) {
      this.log(`Calendar error: ${error.message}`, 'error');
    }
    
    // Sort by air date
    const sortedCalendar = calendar
      .filter(item => item.airDate)
      .sort((a, b) => new Date(a.airDate) - new Date(b.airDate));
    
    return {
      content: [{
        type: 'text',
        text: `📅 **Upcoming Releases (Next ${days} days)**\n\n` +
              (sortedCalendar.length > 0 ?
                sortedCalendar.map(item => {
                  const date = new Date(item.airDate).toLocaleDateString();
                  const status = item.hasFile ? '✅ Downloaded' : '⏳ Waiting';
                  return `**${item.title}** ${item.subtitle ? `(${item.subtitle})` : ''}\n` +
                         `   ${date} | ${item.service} | ${status}`;
                }).join('\n\n') :
                'No upcoming releases found.')
      }]
    };
  }
  
  async handleResourceRead(uri) {
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
          
        case 'media://activity/recent':
          const activityData = await this._getRecentActivity(24, 50);
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(activityData, null, 2)
            }]
          };
          
        case 'media://downloads/active':
          const downloadData = await this._manageDownloads('list');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(downloadData, null, 2)
            }]
          };
          
        case 'media://calendar/upcoming':
          const calendarData = await this._getCalendar(7, 'all');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(calendarData, null, 2)
            }]
          };
          
        case 'media://indexers/status':
          const indexerData = await this._manageIndexers('list');
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(indexerData, null, 2)
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
  
  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: this.protocolVersion,
            capabilities: this.capabilities,
            serverInfo: this.serverInfo
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
      this.log(`Request error: ${error.message}`, 'error');
      throw error;
    }
  }
  
  // Utility functions
  _formatBytes(bytes) {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }
  
  _formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  }
  
  start() {
    this.log('Starting Media Services MCP Server...');
    
    // Create readline interface for stdio transport
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });
    
    // Handle incoming messages
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
        let requestId;
        try {
          requestId = JSON.parse(line).id;
        } catch {
          requestId = null;
        }
        
        const errorResponse = {
          jsonrpc: '2.0',
          id: requestId,
          error: {
            code: -32603,
            message: error.message,
            data: error.stack
          }
        };
        
        process.stdout.write(JSON.stringify(errorResponse) + '\n');
        this.log(`Error response: ${JSON.stringify(errorResponse)}`, 'error');
      }
    });
    
    // Handle cleanup
    rl.on('close', () => {
      this.log('Readline closed, shutting down...');
      this._cleanup();
    });
    
    // Handle signals
    process.on('SIGINT', () => {
      this.log('Received SIGINT, shutting down...');
      this._cleanup();
    });
    
    process.on('SIGTERM', () => {
      this.log('Received SIGTERM, shutting down...');
      this._cleanup();
    });
    
    // Handle errors
    process.on('uncaughtException', (error) => {
      this.log(`Uncaught exception: ${error.message}`, 'error');
      process.stderr.write(`MCP Fatal Error: ${error.message}\n`);
      this._cleanup(1);
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      this.log(`Unhandled rejection: ${reason}`, 'error');
      process.stderr.write(`MCP Unhandled Rejection: ${reason}\n`);
    });
    
    // Keep the process alive
    this.keepAliveInterval = setInterval(() => {
      this.log('Keep-alive tick (process active)');
    }, 30000);
    
    // Keep stdin open
    process.stdin.resume();
    
    this.log(`Server started successfully! Ready to handle media service requests.`);
    this.log(`Available services: ${Object.keys(this.services).filter(s => this.services[s].enabled).join(', ')}`);
  }
  
  _cleanup(exitCode = 0) {
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
      this.keepAliveInterval = null;
    }
    
    this.log('Server cleanup completed');
    process.exit(exitCode);
  }
}

// Start the server if run directly
if (require.main === module) {
  const server = new MediaServicesMCPServer();
  server.start();
}

module.exports = MediaServicesMCPServer;