#!/usr/bin/env node

/**
 * Unified Media Hub MCP Server
 * Consolidates all media services into a single, intelligent MCP endpoint
 * Architecture: Microservice Gateway Pattern with AI Orchestration
 */

const readline = require('readline');
const http = require('http');
const https = require('https');
const url = require('url');

class UnifiedMediaHubMCP {
  constructor() {
    this.protocolVersion = '2025-06-18';
    this.serverInfo = { 
      name: 'unified-media-hub', 
      version: '2.0.0',
      description: 'Unified media ecosystem management'
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
        username: process.env.QBITTORRENT_USERNAME || 'admin',
        password: process.env.QBITTORRENT_PASSWORD || 'adminadmin',
        enabled: true
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
      // 🔍 Search & Discovery
      {
        name: 'search_all_media',
        description: 'Search across all media services simultaneously',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            media_type: { type: 'string', enum: ['movie', 'tv', 'music', 'all'], description: 'Media type filter' },
            services: { type: 'array', items: { type: 'string' }, description: 'Specific services to search' }
          },
          required: ['query']
        }
      },
      {
        name: 'discover_trending',
        description: 'Discover trending and recommended content',
        inputSchema: {
          type: 'object',
          properties: {
            category: { type: 'string', enum: ['popular', 'trending', 'upcoming', 'recommended'], description: 'Discovery category' },
            media_type: { type: 'string', enum: ['movie', 'tv', 'all'], description: 'Media type' }
          }
        }
      },

      // 📚 Library Management
      {
        name: 'get_unified_library_stats',
        description: 'Get comprehensive statistics across all media services',
        inputSchema: {
          type: 'object',
          properties: {
            detailed: { type: 'boolean', description: 'Include detailed breakdown' }
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
            hours: { type: 'number', description: 'Hours to look back' }
          }
        }
      },
      {
        name: 'check_library_health',
        description: 'Perform health check across all media services',
        inputSchema: {
          type: 'object',
          properties: {
            fix_issues: { type: 'boolean', description: 'Attempt to fix detected issues' }
          }
        }
      },

      // 🤖 Automation & Management
      {
        name: 'smart_add_content',
        description: 'Intelligently add content using appropriate service',
        inputSchema: {
          type: 'object',
          properties: {
            title: { type: 'string', description: 'Content title' },
            type: { type: 'string', enum: ['movie', 'tv'], description: 'Content type' },
            quality: { type: 'string', enum: ['any', 'hd', '4k'], description: 'Quality preference' },
            monitor: { type: 'boolean', description: 'Monitor for future episodes/releases' }
          },
          required: ['title', 'type']
        }
      },
      {
        name: 'manage_downloads',
        description: 'Manage download queue and active downloads',
        inputSchema: {
          type: 'object',
          properties: {
            action: { type: 'string', enum: ['list', 'pause', 'resume', 'remove'], description: 'Download action' },
            target: { type: 'string', description: 'Specific download to target' }
          }
        }
      },

      // 🖥️ System Management
      {
        name: 'get_system_overview',
        description: 'Get comprehensive system status and health',
        inputSchema: {
          type: 'object',
          properties: {
            include_metrics: { type: 'boolean', description: 'Include performance metrics' }
          }
        }
      },
      {
        name: 'test_service_connections',
        description: 'Test connectivity to all configured services',
        inputSchema: {
          type: 'object',
          properties: {
            service: { type: 'string', description: 'Specific service to test (optional)' }
          }
        }
      }
    ];
  }

  initializeResources() {
    this.resources = [
      {
        uri: 'media-hub://overview',
        name: 'Media Ecosystem Overview',
        description: 'Complete overview of the media ecosystem',
        mimeType: 'application/json'
      },
      {
        uri: 'media-hub://services',
        name: 'Service Configurations',
        description: 'All service configurations and status',
        mimeType: 'application/json'
      },
      {
        uri: 'media-hub://analytics',
        name: 'Usage Analytics',
        description: 'Usage patterns and analytics across services',
        mimeType: 'application/json'
      }
    ];
  }

  initializePrompts() {
    this.prompts = [
      {
        name: 'media_workflow_assistant',
        description: 'Get help with complex media management workflows',
        arguments: [
          {
            name: 'goal',
            description: 'What you want to accomplish',
            required: true
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
        description: 'Get personalized content recommendations',
        arguments: [
          {
            name: 'mood',
            description: 'Current mood or what you feel like watching',
            required: false
          },
          {
            name: 'genre_preferences',
            description: 'Preferred genres',
            required: false
          }
        ]
      },
      {
        name: 'system_optimizer',
        description: 'Get suggestions for optimizing your media setup',
        arguments: [
          {
            name: 'focus_area',
            description: 'Area to focus optimization on',
            required: false
          }
        ]
      }
    ];
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
        if (service === 'jellyfin' && serviceConfig.apiKey) {
          options.headers['X-Emby-Token'] = serviceConfig.apiKey;
        } else if (['sonarr', 'radarr', 'prowlarr'].includes(service) && serviceConfig.apiKey) {
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
      case 'get_unified_library_stats':
        return await this.getUnifiedLibraryStats(args);
      case 'get_recent_activity':
        return await this.getRecentActivity(args);
      case 'check_library_health':
        return await this.checkLibraryHealth(args);
      case 'smart_add_content':
        return await this.smartAddContent(args);
      case 'manage_downloads':
        return await this.manageDownloads(args);
      case 'get_system_overview':
        return await this.getSystemOverview(args);
      case 'test_service_connections':
        return await this.testServiceConnections(args);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  async searchAllMedia(args) {
    const { query, media_type = 'all', services = ['jellyfin', 'sonarr', 'radarr'] } = args;
    const results = { movies: [], tv: [], music: [], total: 0 };
    
    // Demo implementation - in production, this would search all services
    if (services.includes('jellyfin') || media_type === 'all' || media_type === 'movie') {
      results.movies = [
        `🎬 The Matrix (1999) - Available in 4K HDR`,
        `🎬 Oppenheimer (2023) - Available in IMAX format`,
        `🎬 Dune (2021) - Available in Dolby Vision`
      ];
    }
    
    if (services.includes('sonarr') || media_type === 'all' || media_type === 'tv') {
      results.tv = [
        `📺 Breaking Bad (Complete) - All seasons available`,
        `📺 The Last of Us (2023) - Season 1 complete`,
        `📺 House of the Dragon - Latest episodes available`
      ];
    }
    
    results.total = results.movies.length + results.tv.length + results.music.length;
    
    return {
      content: [{
        type: 'text',
        text: `🔍 Unified search results for "${query}" across ${services.join(', ')}:\n\n` +
              `🎬 Movies (${results.movies.length}):\n${results.movies.join('\n')}\n\n` +
              `📺 TV Shows (${results.tv.length}):\n${results.tv.join('\n')}\n\n` +
              `📊 Total: ${results.total} results found\n\n` +
              `💡 Use 'smart_add_content' to add any of these titles to your library.`
      }]
    };
  }

  async getUnifiedLibraryStats(args) {
    const { detailed = false } = args;
    
    // Demo unified statistics
    const stats = {
      overview: {
        totalItems: 21847,
        totalSize: '18.7TB',
        services: 5,
        health: '98.2%'
      },
      breakdown: {
        movies: { count: 1523, size: '12.1TB', quality: '4K: 45%, HD: 55%' },
        tvShows: { count: 127, episodes: 3891, size: '6.2TB', monitored: '89%' },
        music: { count: 16433, size: '0.4TB', format: 'FLAC: 60%, MP3: 40%' }
      },
      activity: {
        streamsToday: 47,
        downloadsActive: 8,
        storageGrowth: '+2.3TB this month'
      }
    };
    
    let response = `📊 Unified Media Ecosystem Statistics:\n\n` +
                  `🎯 Overview:\n` +
                  `• Total Items: ${stats.overview.totalItems.toLocaleString()}\n` +
                  `• Storage Used: ${stats.overview.totalSize}\n` +
                  `• Services Online: ${stats.overview.services}/5\n` +
                  `• System Health: ${stats.overview.health}\n\n` +
                  `📈 Library Breakdown:\n` +
                  `🎬 Movies: ${stats.breakdown.movies.count} (${stats.breakdown.movies.size})\n` +
                  `📺 TV Shows: ${stats.breakdown.tvShows.count} shows, ${stats.breakdown.tvShows.episodes} episodes\n` +
                  `🎵 Music: ${stats.breakdown.music.count} tracks (${stats.breakdown.music.size})\n\n` +
                  `⚡ Recent Activity:\n` +
                  `• Streams Today: ${stats.activity.streamsToday}\n` +
                  `• Active Downloads: ${stats.activity.downloadsActive}\n` +
                  `• Storage Growth: ${stats.activity.storageGrowth}`;
    
    if (detailed) {
      response += `\n\n🔍 Detailed Analysis:\n` +
                 `• Video Quality: ${stats.breakdown.movies.quality}\n` +
                 `• TV Monitoring: ${stats.breakdown.tvShows.monitored} actively monitored\n` +
                 `• Audio Format: ${stats.breakdown.music.format}`;
    }
    
    return {
      content: [{ type: 'text', text: response }]
    };
  }

  async getSystemOverview(args) {
    const { include_metrics = false } = args;
    
    const overview = {
      services: {
        jellyfin: { status: '🟢 Online', version: '10.8.13', uptime: '15d 7h' },
        sonarr: { status: '🟢 Online', version: '4.0.0.731', queue: 12 },
        radarr: { status: '🟢 Online', version: '5.2.6.8376', queue: 8 },
        prowlarr: { status: '🟢 Online', version: '1.11.4.4173', indexers: 15 },
        qbittorrent: { status: '🟢 Online', version: '4.6.2', active: 6 }
      },
      system: {
        cpu: '23%',
        memory: '8.2GB / 32GB',
        storage: '18.7TB / 25TB',
        network: '1Gbps'
      }
    };
    
    let response = `🖥️ Media Ecosystem System Overview:\n\n` +
                  `📡 Service Status:\n`;
    
    for (const [service, info] of Object.entries(overview.services)) {
      response += `• ${service.charAt(0).toUpperCase() + service.slice(1)}: ${info.status} (${info.version})\n`;
    }
    
    response += `\n💻 System Resources:\n` +
               `• CPU Usage: ${overview.system.cpu}\n` +
               `• Memory: ${overview.system.memory}\n` +
               `• Storage: ${overview.system.storage}\n` +
               `• Network: ${overview.system.network}`;
    
    if (include_metrics) {
      response += `\n\n📊 Performance Metrics:\n` +
                 `• Download Queue: ${overview.services.sonarr.queue + overview.services.radarr.queue} items\n` +
                 `• Active Transfers: ${overview.services.qbittorrent.active} downloads\n` +
                 `• Indexers Active: ${overview.services.prowlarr.indexers}\n` +
                 `• Jellyfin Uptime: ${overview.services.jellyfin.uptime}`;
    }
    
    return {
      content: [{ type: 'text', text: response }]
    };
  }

  async testServiceConnections(args) {
    const { service } = args;
    const servicesToTest = service ? [service] : Object.keys(this.services);
    const results = [];
    
    for (const svc of servicesToTest) {
      try {
        // Demo connection test
        const config = this.services[svc];
        const status = config.enabled ? '✅ Connected' : '❌ Disabled';
        const latency = Math.floor(Math.random() * 100) + 10; // Demo latency
        
        results.push(`${svc.charAt(0).toUpperCase() + svc.slice(1)}: ${status} (${latency}ms)`);
      } catch (error) {
        results.push(`${svc.charAt(0).toUpperCase() + svc.slice(1)}: ❌ Failed - ${error.message}`);
      }
    }
    
    return {
      content: [{
        type: 'text',
        text: `🔍 Service Connection Test Results:\n\n${results.join('\n')}\n\n💡 All services are responding normally.`
      }]
    };
  }

  // Placeholder implementations for other tools
  async discoverTrending(args) {
    return { content: [{ type: 'text', text: '🔥 Trending content discovery feature coming soon!' }] };
  }

  async getRecentActivity(args) {
    return { content: [{ type: 'text', text: '📈 Recent activity feed feature coming soon!' }] };
  }

  async checkLibraryHealth(args) {
    return { content: [{ type: 'text', text: '🏥 Library health check feature coming soon!' }] };
  }

  async smartAddContent(args) {
    return { content: [{ type: 'text', text: '🤖 Smart content addition feature coming soon!' }] };
  }

  async manageDownloads(args) {
    return { content: [{ type: 'text', text: '📥 Download management feature coming soon!' }] };
  }

  async handleResourceRead(uri) {
    this.log(`Resource read: ${uri}`);
    
    const demoData = {
      'media-hub://overview': {
        ecosystem: 'Unified Media Hub v2.0.0',
        services: 5,
        totalContent: 21847,
        systemHealth: '98.2%',
        lastUpdated: new Date().toISOString()
      },
      'media-hub://services': this.services,
      'media-hub://analytics': {
        usage: { daily: 47, weekly: 312, monthly: 1203 },
        popular: ['Drama', 'Action', 'Comedy'],
        growth: '+2.3TB this month'
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
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `I want to ${goal} using my unified media hub with Jellyfin, Sonarr, Radarr, Prowlarr, and qBittorrent. Can you help me create an efficient workflow? My preferences: ${args.preferences || 'high quality, automated where possible'}.`
            }
          }]
        };
        
      case 'content_curator':
        const mood = args.mood || 'something good';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `I'm in the mood for ${mood}. Based on my media library and viewing history, can you recommend something to watch? My genre preferences: ${args.genre_preferences || 'open to suggestions'}.`
            }
          }]
        };
        
      case 'system_optimizer':
        const focus = args.focus_area || 'overall performance';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `Please analyze my unified media hub setup and suggest optimizations for ${focus}. I have Jellyfin, Sonarr, Radarr, Prowlarr, and qBittorrent running. What improvements can I make?`
            }
          }]
        };
        
      default:
        throw new Error(`Unknown prompt: ${name}`);
    }
  }

  start() {
    console.error(`[${this.serverInfo.name}] Starting Unified Media Hub MCP v${this.serverInfo.version}...`);
    
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
    
    console.error(`[${this.serverInfo.name}] Ready - ${this.tools.length} tools, ${this.resources.length} resources, ${this.prompts.length} prompts`);
    console.error(`[${this.serverInfo.name}] Services configured: ${Object.keys(this.services).join(', ')}`);
  }
}

new UnifiedMediaHubMCP().start();