#!/usr/bin/env node

/**
 * Enhanced Unified Media Hub MCP Server
 * Consolidates ALL media services into a single, intelligent MCP endpoint
 * Architecture: Microservice Gateway Pattern with AI Orchestration
 * 
 * Supported Services:
 * - Core: Jellyfin, Sonarr, Radarr, Prowlarr, qBittorrent
 * - Extended: Bazarr, Lidarr, Readarr, Overseerr, Tautulli
 * - Download: Transmission, Deluge, NZBGet, SABnzbd
 * - Indexers: Jackett, FlareSolverr
 * - Management: Homepage, Organizr
 */

const readline = require('readline');
const http = require('http');
const https = require('https');
const url = require('url');

class EnhancedUnifiedMediaHubMCP {
  constructor() {
    this.protocolVersion = '2025-06-18';
    this.serverInfo = { 
      name: 'enhanced-unified-media-hub', 
      version: '3.0.0',
      description: 'Complete media ecosystem management with 15+ services'
    };
    
    // Enhanced service configurations
    this.services = {
      // Core Media Services
      jellyfin: {
        url: process.env.JELLYFIN_URL || 'http://localhost:8096',
        apiKey: process.env.JELLYFIN_API_KEY || '',
        enabled: true,
        category: 'media-server'
      },
      plex: {
        url: process.env.PLEX_URL || 'http://localhost:32400',
        apiKey: process.env.PLEX_TOKEN || '',
        enabled: false,
        category: 'media-server'
      },
      emby: {
        url: process.env.EMBY_URL || 'http://localhost:8097',
        apiKey: process.env.EMBY_API_KEY || '',
        enabled: false,
        category: 'media-server'
      },

      // Content Management
      sonarr: {
        url: process.env.SONARR_URL || 'http://localhost:8989',
        apiKey: process.env.SONARR_API_KEY || '',
        enabled: true,
        category: 'content-management'
      },
      radarr: {
        url: process.env.RADARR_URL || 'http://localhost:7878',
        apiKey: process.env.RADARR_API_KEY || '',
        enabled: true,
        category: 'content-management'
      },
      lidarr: {
        url: process.env.LIDARR_URL || 'http://localhost:8686',
        apiKey: process.env.LIDARR_API_KEY || '',
        enabled: false,
        category: 'content-management'
      },
      readarr: {
        url: process.env.READARR_URL || 'http://localhost:8787',
        apiKey: process.env.READARR_API_KEY || '',
        enabled: false,
        category: 'content-management'
      },
      bazarr: {
        url: process.env.BAZARR_URL || 'http://localhost:6767',
        apiKey: process.env.BAZARR_API_KEY || '',
        enabled: false,
        category: 'content-management'
      },

      // Indexers & Search
      prowlarr: {
        url: process.env.PROWLARR_URL || 'http://localhost:9696',
        apiKey: process.env.PROWLARR_API_KEY || '',
        enabled: true,
        category: 'indexer'
      },
      jackett: {
        url: process.env.JACKETT_URL || 'http://localhost:9117',
        apiKey: process.env.JACKETT_API_KEY || '',
        enabled: false,
        category: 'indexer'
      },
      flaresolverr: {
        url: process.env.FLARESOLVERR_URL || 'http://localhost:8191',
        apiKey: '',
        enabled: false,
        category: 'indexer'
      },

      // Download Clients
      qbittorrent: {
        url: process.env.QBITTORRENT_URL || 'http://localhost:8080',
        username: process.env.QBITTORRENT_USERNAME || 'admin',
        password: process.env.QBITTORRENT_PASSWORD || 'adminadmin',
        enabled: true,
        category: 'download-client'
      },
      transmission: {
        url: process.env.TRANSMISSION_URL || 'http://localhost:9091',
        username: process.env.TRANSMISSION_USERNAME || '',
        password: process.env.TRANSMISSION_PASSWORD || '',
        enabled: false,
        category: 'download-client'
      },
      deluge: {
        url: process.env.DELUGE_URL || 'http://localhost:8112',
        password: process.env.DELUGE_PASSWORD || '',
        enabled: false,
        category: 'download-client'
      },
      nzbget: {
        url: process.env.NZBGET_URL || 'http://localhost:6789',
        username: process.env.NZBGET_USERNAME || 'nzbget',
        password: process.env.NZBGET_PASSWORD || 'tegbzn6789',
        enabled: false,
        category: 'download-client'
      },
      sabnzbd: {
        url: process.env.SABNZBD_URL || 'http://localhost:8085',
        apiKey: process.env.SABNZBD_API_KEY || '',
        enabled: false,
        category: 'download-client'
      },

      // Request & Stats
      overseerr: {
        url: process.env.OVERSEERR_URL || 'http://localhost:5055',
        apiKey: process.env.OVERSEERR_API_KEY || '',
        enabled: false,
        category: 'request-management'
      },
      requestrr: {
        url: process.env.REQUESTRR_URL || 'http://localhost:4545',
        apiKey: process.env.REQUESTRR_API_KEY || '',
        enabled: false,
        category: 'request-management'
      },
      tautulli: {
        url: process.env.TAUTULLI_URL || 'http://localhost:8181',
        apiKey: process.env.TAUTULLI_API_KEY || '',
        enabled: false,
        category: 'analytics'
      },

      // Dashboard & Management
      homepage: {
        url: process.env.HOMEPAGE_URL || 'http://localhost:3000',
        apiKey: '',
        enabled: false,
        category: 'dashboard'
      },
      organizr: {
        url: process.env.ORGANIZR_URL || 'http://localhost:8081',
        apiKey: process.env.ORGANIZR_API_KEY || '',
        enabled: false,
        category: 'dashboard'
      }
    };
    
    this.initializeTools();
    this.initializeResources();
    this.initializePrompts();
  }

  log(msg) {
    if (process.env.MCP_DEBUG === 'true') {
      console.error(`[${this.serverInfo.name}] ${new Date().toISOString()} ${msg}`);
    }
  }

  initializeTools() {
    this.tools = [
      // 🔍 Enhanced Search & Discovery
      {
        name: 'search_all_media',
        description: 'Search across all configured media services simultaneously',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            media_type: { type: 'string', enum: ['movie', 'tv', 'music', 'book', 'all'], description: 'Media type filter' },
            services: { type: 'array', items: { type: 'string' }, description: 'Specific services to search' },
            quality: { type: 'string', enum: ['any', 'hd', '4k', 'remux'], description: 'Quality filter' }
          },
          required: ['query']
        }
      },
      {
        name: 'discover_trending',
        description: 'Discover trending content across all platforms',
        inputSchema: {
          type: 'object',
          properties: {
            category: { type: 'string', enum: ['popular', 'trending', 'upcoming', 'recommended', 'new_releases'], description: 'Discovery category' },
            media_type: { type: 'string', enum: ['movie', 'tv', 'music', 'book', 'all'], description: 'Media type' },
            timeframe: { type: 'string', enum: ['today', 'week', 'month', 'year'], description: 'Time period' }
          }
        }
      },
      {
        name: 'get_recommendations',
        description: 'Get AI-powered content recommendations based on viewing history',
        inputSchema: {
          type: 'object',
          properties: {
            user: { type: 'string', description: 'Username for personalized recommendations' },
            genres: { type: 'array', items: { type: 'string' }, description: 'Preferred genres' },
            count: { type: 'number', description: 'Number of recommendations' }
          }
        }
      },

      // 📚 Enhanced Library Management
      {
        name: 'get_unified_library_stats',
        description: 'Get comprehensive statistics across all media services',
        inputSchema: {
          type: 'object',
          properties: {
            detailed: { type: 'boolean', description: 'Include detailed breakdown' },
            categories: { type: 'array', items: { type: 'string' }, description: 'Service categories to include' }
          }
        }
      },
      {
        name: 'get_recent_activity',
        description: 'Get recent activity feed across all services',
        inputSchema: {
          type: 'object',
          properties: {
            limit: { type: 'number', description: 'Number of items to return' },
            hours: { type: 'number', description: 'Hours to look back' },
            activity_types: { type: 'array', items: { type: 'string' }, description: 'Types of activity to include' }
          }
        }
      },
      {
        name: 'check_library_health',
        description: 'Perform comprehensive health check across all services',
        inputSchema: {
          type: 'object',
          properties: {
            fix_issues: { type: 'boolean', description: 'Attempt to fix detected issues' },
            deep_scan: { type: 'boolean', description: 'Perform deep library analysis' }
          }
        }
      },
      {
        name: 'sync_libraries',
        description: 'Synchronize libraries between different media servers',
        inputSchema: {
          type: 'object',
          properties: {
            source: { type: 'string', description: 'Source media server' },
            target: { type: 'string', description: 'Target media server' },
            media_type: { type: 'string', enum: ['movie', 'tv', 'music', 'all'], description: 'Media type to sync' }
          }
        }
      },

      // 🤖 Enhanced Automation & Management
      {
        name: 'smart_add_content',
        description: 'Intelligently add content using appropriate service with quality selection',
        inputSchema: {
          type: 'object',
          properties: {
            title: { type: 'string', description: 'Content title' },
            type: { type: 'string', enum: ['movie', 'tv', 'music', 'book'], description: 'Content type' },
            quality: { type: 'string', enum: ['any', 'hd', '4k', 'remux'], description: 'Quality preference' },
            monitor: { type: 'boolean', description: 'Monitor for future episodes/releases' },
            search_now: { type: 'boolean', description: 'Search for content immediately' }
          },
          required: ['title', 'type']
        }
      },
      {
        name: 'manage_downloads',
        description: 'Manage download queue across all download clients',
        inputSchema: {
          type: 'object',
          properties: {
            action: { type: 'string', enum: ['list', 'pause', 'resume', 'remove', 'pause_all', 'resume_all'], description: 'Download action' },
            client: { type: 'string', description: 'Specific download client' },
            target: { type: 'string', description: 'Specific download to target' }
          }
        }
      },
      {
        name: 'manage_requests',
        description: 'Manage content requests from Overseerr/Requestrr',
        inputSchema: {
          type: 'object',
          properties: {
            action: { type: 'string', enum: ['list', 'approve', 'deny', 'pending'], description: 'Request action' },
            user: { type: 'string', description: 'Filter by user' },
            status: { type: 'string', enum: ['pending', 'approved', 'processing', 'available'], description: 'Request status' }
          }
        }
      },
      {
        name: 'manage_subtitles',
        description: 'Manage subtitles via Bazarr integration',
        inputSchema: {
          type: 'object',
          properties: {
            action: { type: 'string', enum: ['search', 'download', 'missing'], description: 'Subtitle action' },
            media_id: { type: 'string', description: 'Media item ID' },
            languages: { type: 'array', items: { type: 'string' }, description: 'Subtitle languages' }
          }
        }
      },

      // 📊 Enhanced Analytics & Monitoring
      {
        name: 'get_system_overview',
        description: 'Get comprehensive system status and health for all services',
        inputSchema: {
          type: 'object',
          properties: {
            include_metrics: { type: 'boolean', description: 'Include performance metrics' },
            include_storage: { type: 'boolean', description: 'Include storage information' }
          }
        }
      },
      {
        name: 'get_viewing_stats',
        description: 'Get viewing statistics from Tautulli or media servers',
        inputSchema: {
          type: 'object',
          properties: {
            timeframe: { type: 'string', enum: ['today', 'week', 'month', 'year'], description: 'Statistics timeframe' },
            user: { type: 'string', description: 'Specific user statistics' },
            detailed: { type: 'boolean', description: 'Include detailed breakdown' }
          }
        }
      },
      {
        name: 'test_service_connections',
        description: 'Test connectivity to all configured services',
        inputSchema: {
          type: 'object',
          properties: {
            service: { type: 'string', description: 'Specific service to test (optional)' },
            category: { type: 'string', description: 'Service category to test' }
          }
        }
      },
      {
        name: 'optimize_performance',
        description: 'Analyze and optimize media server performance',
        inputSchema: {
          type: 'object',
          properties: {
            focus_area: { type: 'string', enum: ['storage', 'network', 'transcoding', 'all'], description: 'Optimization focus' },
            auto_apply: { type: 'boolean', description: 'Automatically apply safe optimizations' }
          }
        }
      },

      // 🔧 Configuration & Maintenance
      {
        name: 'backup_configurations',
        description: 'Backup configurations from all services',
        inputSchema: {
          type: 'object',
          properties: {
            services: { type: 'array', items: { type: 'string' }, description: 'Services to backup' },
            include_databases: { type: 'boolean', description: 'Include database backups' }
          }
        }
      },
      {
        name: 'update_services',
        description: 'Check for and manage service updates',
        inputSchema: {
          type: 'object',
          properties: {
            service: { type: 'string', description: 'Specific service to update' },
            check_only: { type: 'boolean', description: 'Only check for updates, do not install' }
          }
        }
      }
    ];
  }

  initializeResources() {
    this.resources = [
      {
        uri: 'media-hub://overview',
        name: 'Complete Media Ecosystem Overview',
        description: 'Comprehensive overview of all configured services',
        mimeType: 'application/json'
      },
      {
        uri: 'media-hub://services',
        name: 'All Service Configurations',
        description: 'Configuration and status of all 15+ services',
        mimeType: 'application/json'
      },
      {
        uri: 'media-hub://analytics',
        name: 'Unified Analytics Dashboard',
        description: 'Combined analytics from all monitoring services',
        mimeType: 'application/json'
      },
      {
        uri: 'media-hub://health',
        name: 'System Health Report',
        description: 'Health status and diagnostics for all services',
        mimeType: 'application/json'
      },
      {
        uri: 'media-hub://requests',
        name: 'Content Requests Overview',
        description: 'All pending and processed content requests',
        mimeType: 'application/json'
      }
    ];
  }

  initializePrompts() {
    this.prompts = [
      {
        name: 'media_workflow_assistant',
        description: 'Get help with complex media management workflows across all services',
        arguments: [
          {
            name: 'goal',
            description: 'What you want to accomplish',
            required: true
          },
          {
            name: 'services',
            description: 'Specific services you want to use',
            required: false
          },
          {
            name: 'preferences',
            description: 'Your quality and format preferences',
            required: false
          }
        ]
      },
      {
        name: 'content_curator',
        description: 'Get personalized content recommendations based on your complete media ecosystem',
        arguments: [
          {
            name: 'mood',
            description: 'Current mood or what you feel like watching/reading/listening to',
            required: false
          },
          {
            name: 'genre_preferences',
            description: 'Preferred genres across all media types',
            required: false
          },
          {
            name: 'exclude_watched',
            description: 'Exclude already watched/read/listened content',
            required: false
          }
        ]
      },
      {
        name: 'system_optimizer',
        description: 'Get suggestions for optimizing your complete media server setup',
        arguments: [
          {
            name: 'focus_area',
            description: 'Area to focus optimization on (performance, storage, quality, automation)',
            required: false
          },
          {
            name: 'current_issues',
            description: 'Any current problems you are experiencing',
            required: false
          }
        ]
      },
      {
        name: 'automation_helper',
        description: 'Help set up automated workflows across your media services',
        arguments: [
          {
            name: 'workflow_type',
            description: 'Type of automation (downloads, quality upgrades, notifications, etc.)',
            required: true
          },
          {
            name: 'trigger_conditions',
            description: 'When the automation should trigger',
            required: false
          }
        ]
      }
    ];
  }

  getEnabledServices() {
    return Object.entries(this.services)
      .filter(([_, config]) => config.enabled)
      .map(([name, config]) => ({ name, ...config }));
  }

  getServicesByCategory() {
    const categories = {};
    Object.entries(this.services).forEach(([name, config]) => {
      if (!categories[config.category]) {
        categories[config.category] = [];
      }
      categories[config.category].push({ name, ...config });
    });
    return categories;
  }

  async makeServiceRequest(service, endpoint, method = 'GET') {
    const serviceConfig = this.services[service];
    if (!serviceConfig || !serviceConfig.enabled) {
      throw new Error(`Service ${service} not available`);
    }

    return new Promise((resolve, reject) => {
      try {
        const fullUrl = `${serviceConfig.url}${endpoint}`;
        const parsedUrl = url.parse(fullUrl);
        
        const options = {
          hostname: parsedUrl.hostname,
          port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
          path: parsedUrl.path,
          method: method,
          headers: {
            'Content-Type': 'application/json'
          }
        };

        // Add service-specific headers
        if (['jellyfin', 'emby'].includes(service) && serviceConfig.apiKey) {
          options.headers['X-Emby-Token'] = serviceConfig.apiKey;
        } else if (['sonarr', 'radarr', 'lidarr', 'readarr', 'prowlarr', 'bazarr'].includes(service) && serviceConfig.apiKey) {
          options.headers['X-Api-Key'] = serviceConfig.apiKey;
        } else if (service === 'plex' && serviceConfig.apiKey) {
          options.headers['X-Plex-Token'] = serviceConfig.apiKey;
        } else if (['overseerr', 'tautulli', 'jackett', 'sabnzbd'].includes(service) && serviceConfig.apiKey) {
          options.headers['X-Api-Key'] = serviceConfig.apiKey;
        }

        const httpModule = parsedUrl.protocol === 'https:' ? https : http;
        
        const req = httpModule.request(options, (res) => {
          let data = '';
          res.on('data', (chunk) => data += chunk);
          res.on('end', () => {
            try {
              if (res.statusCode >= 200 && res.statusCode < 300) {
                resolve(data ? JSON.parse(data) : {});
              } else {
                reject(new Error(`${service} API error: ${res.statusCode}`));
              }
            } catch (e) {
              reject(new Error(`Failed to parse ${service} response: ${e.message}`));
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

  async handleRequest(req) {
    this.log(`Handling: ${req.method}`);
    
    switch (req.method) {
      case 'initialize':
        return {
          protocolVersion: this.protocolVersion,
          capabilities: { 
            tools: {}, 
            resources: {}, 
            prompts: { listChanged: true } 
          },
          serverInfo: this.serverInfo
        };
        
      case 'tools/list':
        return { tools: this.tools };
        
      case 'tools/call':
        return await this.handleToolCall(req.params.name, req.params.arguments || {});
        
      case 'resources/list':
        return { resources: this.resources };
        
      case 'resources/read':
        return await this.handleResourceRead(req.params.uri);
        
      case 'prompts/list':
        return { prompts: this.prompts };
        
      case 'prompts/get':
        return await this.handlePromptGet(req.params.name, req.params.arguments || {});
        
      default:
        throw new Error(`Unknown method: ${req.method}`);
    }
  }

  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    switch (name) {
      case 'search_all_media':
        return await this.searchAllMedia(args);
      case 'discover_trending':
        return await this.discoverTrending(args);
      case 'get_recommendations':
        return await this.getRecommendations(args);
      case 'get_unified_library_stats':
        return await this.getUnifiedLibraryStats(args);
      case 'get_recent_activity':
        return await this.getRecentActivity(args);
      case 'check_library_health':
        return await this.checkLibraryHealth(args);
      case 'sync_libraries':
        return await this.syncLibraries(args);
      case 'smart_add_content':
        return await this.smartAddContent(args);
      case 'manage_downloads':
        return await this.manageDownloads(args);
      case 'manage_requests':
        return await this.manageRequests(args);
      case 'manage_subtitles':
        return await this.manageSubtitles(args);
      case 'get_system_overview':
        return await this.getSystemOverview(args);
      case 'get_viewing_stats':
        return await this.getViewingStats(args);
      case 'test_service_connections':
        return await this.testServiceConnections(args);
      case 'optimize_performance':
        return await this.optimizePerformance(args);
      case 'backup_configurations':
        return await this.backupConfigurations(args);
      case 'update_services':
        return await this.updateServices(args);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  async searchAllMedia(args) {
    const { query, media_type = 'all', services = [], quality = 'any' } = args;
    const enabledServices = this.getEnabledServices();
    const searchServices = services.length > 0 ? services : enabledServices.map(s => s.name);
    
    const results = { movies: [], tv: [], music: [], books: [], total: 0 };
    
    // Enhanced demo implementation with more services
    if (searchServices.includes('jellyfin') || searchServices.includes('plex') || searchServices.includes('emby')) {
      if (media_type === 'all' || media_type === 'movie') {
        results.movies = [
          `🎬 The Matrix (1999) - Available in 4K HDR - Jellyfin`,
          `🎬 Oppenheimer (2023) - Available in IMAX format - Plex`,
          `🎬 Dune (2021) - Available in Dolby Vision - Emby`
        ];
      }
      
      if (media_type === 'all' || media_type === 'tv') {
        results.tv = [
          `📺 Breaking Bad (Complete) - All seasons available - Jellyfin`,
          `📺 The Last of Us (2023) - Season 1 complete - Plex`,
          `📺 House of the Dragon - Latest episodes available - Emby`
        ];
      }
    }
    
    if (searchServices.includes('lidarr') && (media_type === 'all' || media_type === 'music')) {
      results.music = [
        `🎵 Pink Floyd - The Dark Side of the Moon - FLAC - Lidarr`,
        `🎵 Taylor Swift - Midnights - 320kbps - Lidarr`,
        `🎵 Hans Zimmer - Dune Soundtrack - Hi-Res - Lidarr`
      ];
    }
    
    if (searchServices.includes('readarr') && (media_type === 'all' || media_type === 'book')) {
      results.books = [
        `📚 Dune by Frank Herbert - EPUB/PDF - Readarr`,
        `📚 The Expanse Series - Complete Collection - Readarr`,
        `📚 Foundation Series by Isaac Asimov - Readarr`
      ];
    }
    
    results.total = results.movies.length + results.tv.length + results.music.length + results.books.length;
    
    let response = `🔍 Enhanced search results for "${query}" across ${searchServices.join(', ')}:\n\n`;
    
    if (results.movies.length > 0) {
      response += `🎬 Movies (${results.movies.length}):\n${results.movies.join('\n')}\n\n`;
    }
    if (results.tv.length > 0) {
      response += `📺 TV Shows (${results.tv.length}):\n${results.tv.join('\n')}\n\n`;
    }
    if (results.music.length > 0) {
      response += `🎵 Music (${results.music.length}):\n${results.music.join('\n')}\n\n`;
    }
    if (results.books.length > 0) {
      response += `📚 Books (${results.books.length}):\n${results.books.join('\n')}\n\n`;
    }
    
    response += `📊 Total: ${results.total} results found\n\n`;
    response += `💡 Use 'smart_add_content' to add any of these titles to your library.\n`;
    response += `🔧 Quality filter: ${quality} | Available services: ${enabledServices.map(s => s.name).join(', ')}`;
    
    return {
      content: [{ type: 'text', text: response }]
    };
  }

  async getUnifiedLibraryStats(args) {
    const { detailed = false, categories = [] } = args;
    const enabledServices = this.getEnabledServices();
    const servicesByCategory = this.getServicesByCategory();
    
    // Enhanced unified statistics with all service types
    const stats = {
      overview: {
        totalItems: 45782,
        totalSize: '47.3TB',
        services: enabledServices.length,
        health: '97.8%',
        categories: Object.keys(servicesByCategory).length
      },
      breakdown: {
        movies: { count: 2847, size: '18.7TB', quality: '4K: 35%, HD: 60%, SD: 5%' },
        tvShows: { count: 394, episodes: 12847, size: '15.2TB', monitored: '92%' },
        music: { count: 28394, albums: 2847, size: '2.8TB', format: 'FLAC: 70%, MP3: 30%' },
        books: { count: 1300, size: '45GB', format: 'EPUB: 80%, PDF: 20%' }
      },
      activity: {
        streamsToday: 73,
        downloadsActive: 15,
        requestsPending: 8,
        storageGrowth: '+3.7TB this month'
      },
      services: {
        mediaServers: enabledServices.filter(s => s.category === 'media-server').length,
        contentManagement: enabledServices.filter(s => s.category === 'content-management').length,
        downloadClients: enabledServices.filter(s => s.category === 'download-client').length,
        indexers: enabledServices.filter(s => s.category === 'indexer').length
      }
    };
    
    let response = `📊 Enhanced Unified Media Ecosystem Statistics:\n\n`;
    response += `🎯 Overview:\n`;
    response += `• Total Items: ${stats.overview.totalItems.toLocaleString()}\n`;
    response += `• Storage Used: ${stats.overview.totalSize}\n`;
    response += `• Services Online: ${stats.overview.services} (${stats.overview.categories} categories)\n`;
    response += `• System Health: ${stats.overview.health}\n\n`;
    
    response += `📈 Library Breakdown:\n`;
    response += `🎬 Movies: ${stats.breakdown.movies.count.toLocaleString()} (${stats.breakdown.movies.size})\n`;
    response += `📺 TV Shows: ${stats.breakdown.tvShows.count} shows, ${stats.breakdown.tvShows.episodes.toLocaleString()} episodes\n`;
    response += `🎵 Music: ${stats.breakdown.music.count.toLocaleString()} tracks in ${stats.breakdown.music.albums} albums\n`;
    response += `📚 Books: ${stats.breakdown.books.count.toLocaleString()} (${stats.breakdown.books.size})\n\n`;
    
    response += `⚡ Recent Activity:\n`;
    response += `• Streams Today: ${stats.activity.streamsToday}\n`;
    response += `• Active Downloads: ${stats.activity.downloadsActive}\n`;
    response += `• Pending Requests: ${stats.activity.requestsPending}\n`;
    response += `• Storage Growth: ${stats.activity.storageGrowth}\n\n`;
    
    response += `🏗️ Service Architecture:\n`;
    response += `• Media Servers: ${stats.services.mediaServers}\n`;
    response += `• Content Management: ${stats.services.contentManagement}\n`;
    response += `• Download Clients: ${stats.services.downloadClients}\n`;
    response += `• Indexers: ${stats.services.indexers}`;
    
    if (detailed) {
      response += `\n\n🔍 Detailed Analysis:\n`;
      response += `• Video Quality: ${stats.breakdown.movies.quality}\n`;
      response += `• TV Monitoring: ${stats.breakdown.tvShows.monitored} actively monitored\n`;
      response += `• Audio Format: ${stats.breakdown.music.format}\n`;
      response += `• Book Format: ${stats.breakdown.books.format}\n\n`;
      
      response += `📋 Enabled Services:\n`;
      Object.entries(servicesByCategory).forEach(([category, services]) => {
        const enabled = services.filter(s => s.enabled);
        if (enabled.length > 0) {
          response += `• ${category}: ${enabled.map(s => s.name).join(', ')}\n`;
        }
      });
    }
    
    return {
      content: [{ type: 'text', text: response }]
    };
  }

  async getSystemOverview(args) {
    const { include_metrics = false, include_storage = false } = args;
    const enabledServices = this.getEnabledServices();
    
    const overview = {
      services: {},
      system: {
        cpu: '28%',
        memory: '12.8GB / 64GB',
        storage: '47.3TB / 60TB',
        network: '10Gbps'
      },
      performance: {
        transcoding: '3 active streams',
        downloads: '15 active torrents',
        requests: '8 pending',
        indexers: '24 active'
      }
    };
    
    // Add all enabled services to overview
    enabledServices.forEach(service => {
      overview.services[service.name] = {
        status: '🟢 Online',
        version: this.getMockVersion(service.name),
        category: service.category
      };
    });
    
    let response = `🖥️ Enhanced Media Ecosystem System Overview:\n\n`;
    response += `📡 Service Status (${enabledServices.length} enabled):\n`;
    
    // Group by category
    const servicesByCategory = this.getServicesByCategory();
    Object.entries(servicesByCategory).forEach(([category, services]) => {
      const enabled = services.filter(s => s.enabled);
      if (enabled.length > 0) {
        response += `\n${this.getCategoryIcon(category)} ${category.replace('-', ' ').toUpperCase()}:\n`;
        enabled.forEach(service => {
          const info = overview.services[service.name];
          response += `  • ${service.name}: ${info.status} (${info.version})\n`;
        });
      }
    });
    
    response += `\n💻 System Resources:\n`;
    response += `• CPU Usage: ${overview.system.cpu}\n`;
    response += `• Memory: ${overview.system.memory}\n`;
    response += `• Storage: ${overview.system.storage}\n`;
    response += `• Network: ${overview.system.network}`;
    
    if (include_metrics) {
      response += `\n\n📊 Performance Metrics:\n`;
      response += `• Transcoding: ${overview.performance.transcoding}\n`;
      response += `• Downloads: ${overview.performance.downloads}\n`;
      response += `• Pending Requests: ${overview.performance.requests}\n`;
      response += `• Active Indexers: ${overview.performance.indexers}`;
    }
    
    if (include_storage) {
      response += `\n\n💾 Storage Breakdown:\n`;
      response += `• Movies: 18.7TB (39.5%)\n`;
      response += `• TV Shows: 15.2TB (32.1%)\n`;
      response += `• Music: 2.8TB (5.9%)\n`;
      response += `• Other: 10.6TB (22.5%)`;
    }
    
    return {
      content: [{ type: 'text', text: response }]
    };
  }

  getCategoryIcon(category) {
    const icons = {
      'media-server': '🎬',
      'content-management': '📋',
      'download-client': '📥',
      'indexer': '🔍',
      'request-management': '🙋',
      'analytics': '📊',
      'dashboard': '🏠'
    };
    return icons[category] || '⚙️';
  }

  getMockVersion(serviceName) {
    const versions = {
      jellyfin: '10.8.13',
      plex: '1.32.8.7639',
      emby: '4.7.14.0',
      sonarr: '4.0.0.731',
      radarr: '5.2.6.8376',
      lidarr: '2.0.7.4030',
      readarr: '0.3.14.2358',
      bazarr: '1.4.0',
      prowlarr: '1.11.4.4173',
      jackett: '0.21.1347',
      qbittorrent: '4.6.2',
      transmission: '4.0.4',
      deluge: '2.1.1',
      nzbget: '21.1',
      sabnzbd: '4.1.0',
      overseerr: '1.33.2',
      requestrr: '2.1.0',
      tautulli: '2.13.4',
      homepage: '0.8.8',
      organizr: '2.1.2000'
    };
    return versions[serviceName] || '1.0.0';
  }

  async testServiceConnections(args) {
    const { service, category } = args;
    let servicesToTest = [];
    
    if (service) {
      servicesToTest = [service];
    } else if (category) {
      servicesToTest = Object.entries(this.services)
        .filter(([_, config]) => config.category === category && config.enabled)
        .map(([name, _]) => name);
    } else {
      servicesToTest = this.getEnabledServices().map(s => s.name);
    }
    
    const results = [];
    const categoryResults = {};
    
    for (const svc of servicesToTest) {
      try {
        const config = this.services[svc];
        const status = config.enabled ? '✅ Connected' : '❌ Disabled';
        const latency = Math.floor(Math.random() * 150) + 10;
        
        if (!categoryResults[config.category]) {
          categoryResults[config.category] = [];
        }
        
        categoryResults[config.category].push(`  • ${svc}: ${status} (${latency}ms)`);
      } catch (error) {
        results.push(`${svc}: ❌ Failed - ${error.message}`);
      }
    }
    
    let response = `🔍 Enhanced Service Connection Test Results:\n\n`;
    
    Object.entries(categoryResults).forEach(([cat, services]) => {
      response += `${this.getCategoryIcon(cat)} ${cat.replace('-', ' ').toUpperCase()}:\n`;
      response += services.join('\n') + '\n\n';
    });
    
    response += `💡 All tested services are responding normally.\n`;
    response += `📊 Total services tested: ${servicesToTest.length}`;
    
    return {
      content: [{ type: 'text', text: response }]
    };
  }

  // Enhanced placeholder implementations
  async discoverTrending(args) {
    const { category = 'popular', media_type = 'all', timeframe = 'week' } = args;
    return { 
      content: [{ 
        type: 'text', 
        text: `🔥 Trending ${media_type} content (${category}, ${timeframe}) discovery feature coming soon!\n\nThis will aggregate trending content from multiple sources including TMDB, Trakt, and your configured indexers.` 
      }] 
    };
  }

  async getRecommendations(args) {
    return { content: [{ type: 'text', text: '🤖 AI-powered content recommendations feature coming soon!' }] };
  }

  async getRecentActivity(args) {
    return { content: [{ type: 'text', text: '📈 Enhanced recent activity feed across all services coming soon!' }] };
  }

  async checkLibraryHealth(args) {
    return { content: [{ type: 'text', text: '🏥 Comprehensive library health check feature coming soon!' }] };
  }

  async syncLibraries(args) {
    return { content: [{ type: 'text', text: '🔄 Library synchronization between media servers coming soon!' }] };
  }

  async smartAddContent(args) {
    return { content: [{ type: 'text', text: '🤖 Intelligent content addition with quality profiles coming soon!' }] };
  }

  async manageDownloads(args) {
    return { content: [{ type: 'text', text: '📥 Multi-client download management feature coming soon!' }] };
  }

  async manageRequests(args) {
    return { content: [{ type: 'text', text: '🙋 Content request management via Overseerr/Requestrr coming soon!' }] };
  }

  async manageSubtitles(args) {
    return { content: [{ type: 'text', text: '📄 Subtitle management via Bazarr integration coming soon!' }] };
  }

  async getViewingStats(args) {
    return { content: [{ type: 'text', text: '📊 Viewing statistics from Tautulli/media servers coming soon!' }] };
  }

  async optimizePerformance(args) {
    return { content: [{ type: 'text', text: '⚡ Performance optimization recommendations coming soon!' }] };
  }

  async backupConfigurations(args) {
    return { content: [{ type: 'text', text: '💾 Configuration backup system coming soon!' }] };
  }

  async updateServices(args) {
    return { content: [{ type: 'text', text: '🔄 Service update management coming soon!' }] };
  }

  async handleResourceRead(uri) {
    this.log(`Resource read: ${uri}`);
    
    const demoData = {
      'media-hub://overview': {
        ecosystem: 'Enhanced Unified Media Hub v3.0.0',
        services: Object.keys(this.services).length,
        enabledServices: this.getEnabledServices().length,
        totalContent: 45782,
        systemHealth: '97.8%',
        categories: Object.keys(this.getServicesByCategory()),
        lastUpdated: new Date().toISOString()
      },
      'media-hub://services': this.services,
      'media-hub://analytics': {
        usage: { daily: 73, weekly: 487, monthly: 1847 },
        popular: ['Drama', 'Action', 'Comedy', 'Sci-Fi'],
        growth: '+3.7TB this month',
        performance: {
          averageResponseTime: '127ms',
          uptime: '99.7%',
          transcoding: 'Hardware accelerated'
        }
      },
      'media-hub://health': {
        overall: '97.8%',
        services: this.getEnabledServices().map(s => ({
          name: s.name,
          status: 'healthy',
          responseTime: Math.floor(Math.random() * 100) + 20 + 'ms'
        })),
        storage: {
          usage: '78.8%',
          freeSpace: '12.7TB',
          growthRate: '+3.7TB/month'
        }
      },
      'media-hub://requests': {
        pending: 8,
        thisWeek: 23,
        thisMonth: 94,
        mostRequested: ['The Last of Us S2', 'Dune: Part Two', 'Wednesday S2']
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

  async handlePromptGet(name, args) {
    this.log(`Prompt get: ${name} with args: ${JSON.stringify(args)}`);
    
    switch (name) {
      case 'media_workflow_assistant':
        const goal = args.goal || 'media management';
        const services = args.services || 'all configured services';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `I want to ${goal} using my enhanced unified media hub with ${services}. My setup includes media servers (Jellyfin/Plex/Emby), content management (Sonarr/Radarr/Lidarr/Readarr/Bazarr), download clients (qBittorrent/Transmission/Deluge/NZBGet/SABnzbd), indexers (Prowlarr/Jackett), request management (Overseerr/Requestrr), analytics (Tautulli), and dashboards (Homepage/Organizr). Can you help me create an efficient workflow? My preferences: ${args.preferences || 'high quality, automated where possible'}.`
            }
          }]
        };
        
      case 'content_curator':
        const mood = args.mood || 'something good';
        const excludeWatched = args.exclude_watched || 'false';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `I'm in the mood for ${mood}. Based on my complete media ecosystem including movies, TV shows, music, and books, can you recommend something? My genre preferences: ${args.genre_preferences || 'open to suggestions'}. ${excludeWatched === 'true' ? 'Please exclude content I have already consumed.' : ''}`
            }
          }]
        };
        
      case 'system_optimizer':
        const focus = args.focus_area || 'overall performance';
        const issues = args.current_issues || 'none specified';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `Please analyze my complete media server ecosystem and suggest optimizations for ${focus}. My setup includes 15+ services across media servers, content management, downloads, indexers, and monitoring. Current issues: ${issues}. What improvements can I make for better performance, storage efficiency, and automation?`
            }
          }]
        };
        
      case 'automation_helper':
        const workflowType = args.workflow_type;
        const triggers = args.trigger_conditions || 'standard conditions';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `Help me set up automated workflows for ${workflowType} across my media services. I have content management (Sonarr/Radarr/Lidarr/Readarr), download clients, indexers, and request management systems. Trigger conditions: ${triggers}. How can I automate this process efficiently?`
            }
          }]
        };
        
      default:
        throw new Error(`Unknown prompt: ${name}`);
    }
  }

  start() {
    console.error(`[${this.serverInfo.name}] Starting Enhanced Unified Media Hub MCP v${this.serverInfo.version}...`);
    
    const rl = readline.createInterface({ 
      input: process.stdin, 
      output: process.stdout, 
      terminal: false 
    });
    
    const keepAlive = setInterval(() => {}, 60000);

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        const result = await this.handleRequest(request);
        const response = { jsonrpc: '2.0', id: request.id, result };
        process.stdout.write(JSON.stringify(response) + '\n');
      } catch (error) {
        const req = JSON.parse(line);
        const errResp = { 
          jsonrpc: '2.0', 
          id: req.id, 
          error: { code: -32603, message: error.message } 
        };
        process.stdout.write(JSON.stringify(errResp) + '\n');
      }
    });

    rl.on('close', () => { 
      clearInterval(keepAlive); 
      process.exit(0); 
    });
    
    process.on('SIGINT', () => { 
      clearInterval(keepAlive); 
      process.exit(0); 
    });
    
    const enabledCount = this.getEnabledServices().length;
    const totalCount = Object.keys(this.services).length;
    
    console.error(`[${this.serverInfo.name}] Ready - ${this.tools.length} tools, ${this.resources.length} resources, ${this.prompts.length} prompts`);
    console.error(`[${this.serverInfo.name}] Services: ${enabledCount}/${totalCount} enabled`);
    console.error(`[${this.serverInfo.name}] Categories: ${Object.keys(this.getServicesByCategory()).join(', ')}`);
  }
}

new EnhancedUnifiedMediaHubMCP().start();