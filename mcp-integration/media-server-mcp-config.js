/**
 * Media Server MCP Integration Configuration
 * Manages external service connections through MCP interfaces
 */

const MCP_CONFIG = {
  // MCP Server Configurations
  servers: {
    'claude-flow': {
      enabled: true,
      endpoint: 'mcp://claude-flow',
      capabilities: ['swarm', 'memory', 'neural', 'performance'],
      timeout: 30000
    },
    'ruv-swarm': {
      enabled: true,
      endpoint: 'mcp://ruv-swarm',
      capabilities: ['distributed', 'consensus', 'coordination'],
      timeout: 30000
    },
    'media-unified': {
      enabled: true,
      endpoint: 'http://localhost:3737/mcp',
      capabilities: ['media', 'streaming', 'transcoding', 'metadata'],
      timeout: 60000
    }
  },

  // Media Service Integrations
  services: {
    jellyfin: {
      name: 'Jellyfin',
      type: 'media-server',
      endpoint: 'http://localhost:8096',
      apiKey: process.env.JELLYFIN_API_KEY,
      mcp: {
        resource: 'media://jellyfin/library',
        tools: ['scan', 'transcode', 'stream', 'metadata']
      }
    },
    plex: {
      name: 'Plex Media Server',
      type: 'media-server',
      endpoint: 'http://localhost:32400',
      token: process.env.PLEX_TOKEN,
      mcp: {
        resource: 'media://plex/library',
        tools: ['scan', 'optimize', 'share', 'sync']
      }
    },
    sonarr: {
      name: 'Sonarr',
      type: 'indexer',
      endpoint: 'http://localhost:8989',
      apiKey: process.env.SONARR_API_KEY,
      mcp: {
        resource: 'indexer://sonarr/series',
        tools: ['search', 'download', 'monitor', 'calendar']
      }
    },
    radarr: {
      name: 'Radarr',
      type: 'indexer',
      endpoint: 'http://localhost:7878',
      apiKey: process.env.RADARR_API_KEY,
      mcp: {
        resource: 'indexer://radarr/movies',
        tools: ['search', 'download', 'monitor', 'discover']
      }
    },
    lidarr: {
      name: 'Lidarr',
      type: 'indexer',
      endpoint: 'http://localhost:8686',
      apiKey: process.env.LIDARR_API_KEY,
      mcp: {
        resource: 'indexer://lidarr/music',
        tools: ['search', 'download', 'monitor', 'artist']
      }
    },
    prowlarr: {
      name: 'Prowlarr',
      type: 'indexer-manager',
      endpoint: 'http://localhost:9696',
      apiKey: process.env.PROWLARR_API_KEY,
      mcp: {
        resource: 'indexer://prowlarr/indexers',
        tools: ['manage', 'search', 'test', 'sync']
      }
    },
    qbittorrent: {
      name: 'qBittorrent',
      type: 'download-client',
      endpoint: 'http://localhost:8080',
      username: process.env.QBITTORRENT_USER || 'admin',
      password: process.env.QBITTORRENT_PASS,
      mcp: {
        resource: 'download://qbittorrent/torrents',
        tools: ['add', 'pause', 'resume', 'delete', 'status']
      }
    },
    sabnzbd: {
      name: 'SABnzbd',
      type: 'download-client',
      endpoint: 'http://localhost:8081',
      apiKey: process.env.SABNZBD_API_KEY,
      mcp: {
        resource: 'download://sabnzbd/queue',
        tools: ['add', 'pause', 'resume', 'delete', 'status']
      }
    },
    bazarr: {
      name: 'Bazarr',
      type: 'subtitle-manager',
      endpoint: 'http://localhost:6767',
      apiKey: process.env.BAZARR_API_KEY,
      mcp: {
        resource: 'subtitle://bazarr/subtitles',
        tools: ['search', 'download', 'sync', 'languages']
      }
    },
    overseerr: {
      name: 'Overseerr',
      type: 'request-manager',
      endpoint: 'http://localhost:5055',
      apiKey: process.env.OVERSEERR_API_KEY,
      mcp: {
        resource: 'request://overseerr/requests',
        tools: ['request', 'approve', 'deny', 'notify']
      }
    }
  },

  // MCP Tool Definitions
  tools: {
    // Media Management Tools
    'media.scan': {
      description: 'Scan media library for new content',
      parameters: {
        library: { type: 'string', required: true },
        deep: { type: 'boolean', default: false }
      }
    },
    'media.transcode': {
      description: 'Transcode media to different format',
      parameters: {
        source: { type: 'string', required: true },
        profile: { type: 'string', required: true },
        destination: { type: 'string' }
      }
    },
    'media.metadata': {
      description: 'Fetch or update media metadata',
      parameters: {
        mediaId: { type: 'string', required: true },
        action: { type: 'string', enum: ['fetch', 'update', 'refresh'] }
      }
    },

    // Indexer Tools
    'indexer.search': {
      description: 'Search for media across indexers',
      parameters: {
        query: { type: 'string', required: true },
        type: { type: 'string', enum: ['movie', 'series', 'music'] },
        quality: { type: 'string' }
      }
    },
    'indexer.monitor': {
      description: 'Monitor media for new releases',
      parameters: {
        mediaId: { type: 'string', required: true },
        enabled: { type: 'boolean', default: true }
      }
    },

    // Download Tools
    'download.add': {
      description: 'Add download to client',
      parameters: {
        url: { type: 'string', required: true },
        category: { type: 'string' },
        priority: { type: 'number', default: 0 }
      }
    },
    'download.status': {
      description: 'Get download status',
      parameters: {
        downloadId: { type: 'string' }
      }
    },

    // Coordination Tools
    'swarm.coordinate': {
      description: 'Coordinate tasks across services',
      parameters: {
        task: { type: 'string', required: true },
        services: { type: 'array', required: true },
        strategy: { type: 'string', enum: ['parallel', 'sequential'] }
      }
    }
  },

  // Resource Definitions
  resources: {
    'media://library': {
      description: 'Unified media library across all servers',
      mimeType: 'application/json',
      access: ['read', 'write']
    },
    'indexer://search': {
      description: 'Unified search across all indexers',
      mimeType: 'application/json',
      access: ['read']
    },
    'download://queue': {
      description: 'Unified download queue',
      mimeType: 'application/json',
      access: ['read', 'write']
    },
    'system://status': {
      description: 'System-wide status and health',
      mimeType: 'application/json',
      access: ['read']
    }
  },

  // Authentication Configuration
  auth: {
    method: 'bearer',
    tokenEndpoint: '/api/auth/token',
    refreshEndpoint: '/api/auth/refresh',
    scopes: ['media.read', 'media.write', 'admin']
  },

  // Error Handling
  errorHandling: {
    retries: 3,
    backoff: 'exponential',
    timeout: 30000,
    circuitBreaker: {
      threshold: 5,
      resetTimeout: 60000
    }
  },

  // Monitoring
  monitoring: {
    enabled: true,
    metricsEndpoint: '/api/metrics',
    healthEndpoint: '/api/health',
    logLevel: 'info'
  }
};

module.exports = MCP_CONFIG;