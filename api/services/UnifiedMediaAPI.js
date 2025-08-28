// Unified Media API - Integration layer for 30+ media services
import axios from 'axios';
import WebSocket from 'ws';
import EventEmitter from 'events';

class UnifiedMediaAPI extends EventEmitter {
  constructor() {
    super();
    this.services = this.initializeServices();
    this.connections = new Map();
    this.healthStatus = new Map();
    this.apiKeys = this.loadAPIKeys();
    this.wsConnections = new Map();
  }

  initializeServices() {
    return {
      // Media Servers
      jellyfin: {
        name: 'Jellyfin',
        baseUrl: process.env.JELLYFIN_URL || 'http://localhost:8096',
        apiKey: process.env.JELLYFIN_API_KEY,
        type: 'media-server',
        endpoints: {
          status: '/System/Info',
          libraries: '/Library/VirtualFolders',
          items: '/Items',
          playback: '/PlayingItems',
          users: '/Users'
        }
      },
      
      plex: {
        name: 'Plex',
        baseUrl: process.env.PLEX_URL || 'http://localhost:32400',
        token: process.env.PLEX_TOKEN,
        type: 'media-server',
        endpoints: {
          status: '/identity',
          libraries: '/library/sections',
          items: '/library/all',
          sessions: '/status/sessions'
        }
      },
      
      emby: {
        name: 'Emby',
        baseUrl: process.env.EMBY_URL || 'http://localhost:8096',
        apiKey: process.env.EMBY_API_KEY,
        type: 'media-server',
        endpoints: {
          status: '/emby/System/Info',
          libraries: '/emby/Library/VirtualFolders',
          items: '/emby/Items'
        }
      },
      
      // Content Management (*arr apps)
      sonarr: {
        name: 'Sonarr',
        baseUrl: process.env.SONARR_URL || 'http://localhost:8989',
        apiKey: process.env.SONARR_API_KEY,
        type: 'content-manager',
        endpoints: {
          status: '/api/v3/system/status',
          series: '/api/v3/series',
          calendar: '/api/v3/calendar',
          queue: '/api/v3/queue',
          wanted: '/api/v3/wanted/missing',
          indexers: '/api/v3/indexer'
        }
      },
      
      radarr: {
        name: 'Radarr',
        baseUrl: process.env.RADARR_URL || 'http://localhost:7878',
        apiKey: process.env.RADARR_API_KEY,
        type: 'content-manager',
        endpoints: {
          status: '/api/v3/system/status',
          movies: '/api/v3/movie',
          calendar: '/api/v3/calendar',
          queue: '/api/v3/queue',
          wanted: '/api/v3/wanted/missing'
        }
      },
      
      lidarr: {
        name: 'Lidarr',
        baseUrl: process.env.LIDARR_URL || 'http://localhost:8686',
        apiKey: process.env.LIDARR_API_KEY,
        type: 'content-manager',
        endpoints: {
          status: '/api/v1/system/status',
          artists: '/api/v1/artist',
          albums: '/api/v1/album',
          queue: '/api/v1/queue'
        }
      },
      
      readarr: {
        name: 'Readarr',
        baseUrl: process.env.READARR_URL || 'http://localhost:8787',
        apiKey: process.env.READARR_API_KEY,
        type: 'content-manager',
        endpoints: {
          status: '/api/v1/system/status',
          books: '/api/v1/book',
          authors: '/api/v1/author',
          queue: '/api/v1/queue'
        }
      },
      
      bazarr: {
        name: 'Bazarr',
        baseUrl: process.env.BAZARR_URL || 'http://localhost:6767',
        apiKey: process.env.BAZARR_API_KEY,
        type: 'subtitle-manager',
        endpoints: {
          status: '/api/system/status',
          series: '/api/series',
          movies: '/api/movies',
          wanted: '/api/wanted'
        }
      },
      
      prowlarr: {
        name: 'Prowlarr',
        baseUrl: process.env.PROWLARR_URL || 'http://localhost:9696',
        apiKey: process.env.PROWLARR_API_KEY,
        type: 'indexer-manager',
        endpoints: {
          status: '/api/v1/system/status',
          indexers: '/api/v1/indexer',
          stats: '/api/v1/indexerstats'
        }
      },
      
      // Download Clients
      qbittorrent: {
        name: 'qBittorrent',
        baseUrl: process.env.QBITTORRENT_URL || 'http://localhost:8080',
        username: process.env.QBITTORRENT_USER,
        password: process.env.QBITTORRENT_PASS,
        type: 'download-client',
        endpoints: {
          login: '/api/v2/auth/login',
          torrents: '/api/v2/torrents/info',
          add: '/api/v2/torrents/add',
          pause: '/api/v2/torrents/pause',
          resume: '/api/v2/torrents/resume',
          delete: '/api/v2/torrents/delete'
        }
      },
      
      sabnzbd: {
        name: 'SABnzbd',
        baseUrl: process.env.SABNZBD_URL || 'http://localhost:8085',
        apiKey: process.env.SABNZBD_API_KEY,
        type: 'download-client',
        endpoints: {
          status: '/sabnzbd/api',
          queue: '/sabnzbd/api?mode=queue',
          history: '/sabnzbd/api?mode=history',
          pause: '/sabnzbd/api?mode=pause',
          resume: '/sabnzbd/api?mode=resume'
        }
      },
      
      transmission: {
        name: 'Transmission',
        baseUrl: process.env.TRANSMISSION_URL || 'http://localhost:9091',
        username: process.env.TRANSMISSION_USER,
        password: process.env.TRANSMISSION_PASS,
        type: 'download-client',
        endpoints: {
          rpc: '/transmission/rpc',
          session: '/transmission/rpc/session-stats'
        }
      },
      
      // Request Management
      overseerr: {
        name: 'Overseerr',
        baseUrl: process.env.OVERSEERR_URL || 'http://localhost:5055',
        apiKey: process.env.OVERSEERR_API_KEY,
        type: 'request-manager',
        endpoints: {
          status: '/api/v1/status',
          requests: '/api/v1/request',
          users: '/api/v1/user',
          media: '/api/v1/media'
        }
      },
      
      jellyseerr: {
        name: 'Jellyseerr',
        baseUrl: process.env.JELLYSEERR_URL || 'http://localhost:5055',
        apiKey: process.env.JELLYSEERR_API_KEY,
        type: 'request-manager',
        endpoints: {
          status: '/api/v1/status',
          requests: '/api/v1/request',
          users: '/api/v1/user'
        }
      },
      
      // Monitoring & Analytics
      tautulli: {
        name: 'Tautulli',
        baseUrl: process.env.TAUTULLI_URL || 'http://localhost:8181',
        apiKey: process.env.TAUTULLI_API_KEY,
        type: 'monitoring',
        endpoints: {
          status: '/api/v2',
          activity: '/api/v2?cmd=get_activity',
          history: '/api/v2?cmd=get_history',
          statistics: '/api/v2?cmd=get_home_stats'
        }
      },
      
      grafana: {
        name: 'Grafana',
        baseUrl: process.env.GRAFANA_URL || 'http://localhost:3000',
        apiKey: process.env.GRAFANA_API_KEY,
        type: 'monitoring',
        endpoints: {
          health: '/api/health',
          dashboards: '/api/dashboards',
          datasources: '/api/datasources',
          alerts: '/api/alerts'
        }
      },
      
      prometheus: {
        name: 'Prometheus',
        baseUrl: process.env.PROMETHEUS_URL || 'http://localhost:9090',
        type: 'monitoring',
        endpoints: {
          health: '/-/healthy',
          query: '/api/v1/query',
          targets: '/api/v1/targets',
          alerts: '/api/v1/alerts'
        }
      },
      
      uptimeKuma: {
        name: 'Uptime Kuma',
        baseUrl: process.env.UPTIME_KUMA_URL || 'http://localhost:3001',
        type: 'monitoring',
        endpoints: {
          status: '/api/status-page/heartbeat',
          monitors: '/api/monitors'
        }
      },
      
      // Dashboard & Organization
      organizr: {
        name: 'Organizr',
        baseUrl: process.env.ORGANIZR_URL || 'http://localhost:80',
        apiKey: process.env.ORGANIZR_API_KEY,
        type: 'dashboard',
        endpoints: {
          status: '/api/v2/ping',
          users: '/api/v2/users',
          tabs: '/api/v2/tabs'
        }
      },
      
      heimdall: {
        name: 'Heimdall',
        baseUrl: process.env.HEIMDALL_URL || 'http://localhost:8080',
        type: 'dashboard',
        endpoints: {
          ping: '/ping',
          items: '/items'
        }
      },
      
      homer: {
        name: 'Homer',
        baseUrl: process.env.HOMER_URL || 'http://localhost:3000',
        type: 'dashboard',
        endpoints: {
          config: '/assets/config.yml'
        }
      },
      
      // Container Management
      portainer: {
        name: 'Portainer',
        baseUrl: process.env.PORTAINER_URL || 'http://localhost:9000',
        token: process.env.PORTAINER_TOKEN,
        type: 'container-manager',
        endpoints: {
          status: '/api/system/status',
          endpoints: '/api/endpoints',
          containers: '/api/endpoints/1/docker/containers/json',
          stacks: '/api/stacks'
        }
      },
      
      // Reverse Proxy
      nginxProxyManager: {
        name: 'Nginx Proxy Manager',
        baseUrl: process.env.NPM_URL || 'http://localhost:81',
        email: process.env.NPM_EMAIL,
        password: process.env.NPM_PASSWORD,
        type: 'reverse-proxy',
        endpoints: {
          login: '/api/tokens',
          hosts: '/api/nginx/proxy-hosts',
          certificates: '/api/nginx/certificates'
        }
      },
      
      // Auto-Update
      watchtower: {
        name: 'Watchtower',
        baseUrl: process.env.WATCHTOWER_URL || 'http://localhost:8080',
        type: 'auto-updater',
        endpoints: {
          metrics: '/v1/metrics'
        }
      },
      
      // Backup
      duplicati: {
        name: 'Duplicati',
        baseUrl: process.env.DUPLICATI_URL || 'http://localhost:8200',
        type: 'backup',
        endpoints: {
          status: '/api/v1/serverstate',
          backups: '/api/v1/backups'
        }
      },
      
      // Cloud Storage
      nextcloud: {
        name: 'Nextcloud',
        baseUrl: process.env.NEXTCLOUD_URL || 'http://localhost:8080',
        username: process.env.NEXTCLOUD_USER,
        password: process.env.NEXTCLOUD_PASS,
        type: 'cloud-storage',
        endpoints: {
          status: '/status.php',
          users: '/ocs/v1.php/cloud/users',
          files: '/remote.php/dav/files'
        }
      },
      
      syncthing: {
        name: 'Syncthing',
        baseUrl: process.env.SYNCTHING_URL || 'http://localhost:8384',
        apiKey: process.env.SYNCTHING_API_KEY,
        type: 'sync',
        endpoints: {
          ping: '/rest/system/ping',
          status: '/rest/system/status',
          folders: '/rest/config/folders',
          devices: '/rest/config/devices'
        }
      },
      
      // RSS & Reading
      freshrss: {
        name: 'FreshRSS',
        baseUrl: process.env.FRESHRSS_URL || 'http://localhost:80',
        apiKey: process.env.FRESHRSS_API_KEY,
        type: 'rss-reader',
        endpoints: {
          api: '/api/greader.php',
          feeds: '/api/greader.php/reader/api/0/subscription/list'
        }
      },
      
      calibreWeb: {
        name: 'Calibre-Web',
        baseUrl: process.env.CALIBRE_URL || 'http://localhost:8083',
        type: 'ebook-server',
        endpoints: {
          opds: '/opds',
          books: '/books'
        }
      },
      
      // Photo Management
      photoprism: {
        name: 'PhotoPrism',
        baseUrl: process.env.PHOTOPRISM_URL || 'http://localhost:2342',
        type: 'photo-manager',
        endpoints: {
          status: '/api/v1/status',
          photos: '/api/v1/photos',
          albums: '/api/v1/albums'
        }
      }
    };
  }

  loadAPIKeys() {
    return {
      jellyfin: process.env.JELLYFIN_API_KEY,
      sonarr: process.env.SONARR_API_KEY,
      radarr: process.env.RADARR_API_KEY,
      lidarr: process.env.LIDARR_API_KEY,
      readarr: process.env.READARR_API_KEY,
      bazarr: process.env.BAZARR_API_KEY,
      prowlarr: process.env.PROWLARR_API_KEY,
      overseerr: process.env.OVERSEERR_API_KEY,
      jellyseerr: process.env.JELLYSEERR_API_KEY,
      tautulli: process.env.TAUTULLI_API_KEY,
      grafana: process.env.GRAFANA_API_KEY,
      organizr: process.env.ORGANIZR_API_KEY,
      syncthing: process.env.SYNCTHING_API_KEY,
      freshrss: process.env.FRESHRSS_API_KEY,
      sabnzbd: process.env.SABNZBD_API_KEY
    };
  }

  // Initialize all service connections
  async initializeConnections() {
    const results = await Promise.allSettled(
      Object.entries(this.services).map(([key, service]) => 
        this.testConnection(key, service)
      )
    );
    
    results.forEach((result, index) => {
      const serviceKey = Object.keys(this.services)[index];
      if (result.status === 'fulfilled') {
        this.healthStatus.set(serviceKey, result.value);
      } else {
        this.healthStatus.set(serviceKey, { status: 'offline', error: result.reason });
      }
    });
    
    return this.healthStatus;
  }

  // Test individual service connection
  async testConnection(serviceKey, service) {
    try {
      const headers = this.getAuthHeaders(serviceKey, service);
      let endpoint = service.endpoints.status || service.endpoints.health || '/';
      
      const response = await axios.get(`${service.baseUrl}${endpoint}`, {
        headers,
        timeout: 5000
      });
      
      return {
        status: 'online',
        latency: response.headers['x-response-time'] || 0,
        version: response.data.version || 'unknown'
      };
    } catch (error) {
      console.error(`Failed to connect to ${service.name}:`, error.message);
      return {
        status: 'offline',
        error: error.message
      };
    }
  }

  // Get authentication headers for each service
  getAuthHeaders(serviceKey, service) {
    const headers = {};
    
    switch (serviceKey) {
      case 'jellyfin':
      case 'emby':
        if (service.apiKey) {
          headers['X-MediaBrowser-Token'] = service.apiKey;
        }
        break;
        
      case 'plex':
        if (service.token) {
          headers['X-Plex-Token'] = service.token;
        }
        break;
        
      case 'sonarr':
      case 'radarr':
      case 'lidarr':
      case 'readarr':
      case 'bazarr':
      case 'prowlarr':
        if (service.apiKey) {
          headers['X-Api-Key'] = service.apiKey;
        }
        break;
        
      case 'grafana':
        if (service.apiKey) {
          headers['Authorization'] = `Bearer ${service.apiKey}`;
        }
        break;
        
      case 'qbittorrent':
      case 'transmission':
      case 'nextcloud':
        if (service.username && service.password) {
          const auth = Buffer.from(`${service.username}:${service.password}`).toString('base64');
          headers['Authorization'] = `Basic ${auth}`;
        }
        break;
        
      case 'syncthing':
        if (service.apiKey) {
          headers['X-API-Key'] = service.apiKey;
        }
        break;
    }
    
    return headers;
  }

  // Unified search across all media servers
  async searchMedia(query, options = {}) {
    const { type = 'all', limit = 20 } = options;
    const results = [];
    
    const mediaServers = ['jellyfin', 'plex', 'emby'];
    
    await Promise.all(
      mediaServers.map(async (serverKey) => {
        const service = this.services[serverKey];
        if (this.healthStatus.get(serverKey)?.status !== 'online') return;
        
        try {
          const headers = this.getAuthHeaders(serverKey, service);
          let searchEndpoint = '';
          
          switch (serverKey) {
            case 'jellyfin':
            case 'emby':
              searchEndpoint = `${service.baseUrl}/Items?searchTerm=${query}&limit=${limit}&recursive=true`;
              break;
            case 'plex':
              searchEndpoint = `${service.baseUrl}/hubs/search?query=${query}&limit=${limit}`;
              break;
          }
          
          const response = await axios.get(searchEndpoint, { headers });
          
          results.push({
            server: serverKey,
            items: this.normalizeMediaItems(serverKey, response.data)
          });
        } catch (error) {
          console.error(`Search failed on ${serverKey}:`, error.message);
        }
      })
    );
    
    return results;
  }

  // Normalize media items across different servers
  normalizeMediaItems(serverKey, data) {
    switch (serverKey) {
      case 'jellyfin':
      case 'emby':
        return data.Items?.map(item => ({
          id: item.Id,
          title: item.Name,
          type: item.Type,
          year: item.ProductionYear,
          overview: item.Overview,
          rating: item.CommunityRating,
          thumbnail: `${this.services[serverKey].baseUrl}/Items/${item.Id}/Images/Primary`
        })) || [];
        
      case 'plex':
        return data.MediaContainer?.Metadata?.map(item => ({
          id: item.ratingKey,
          title: item.title,
          type: item.type,
          year: item.year,
          overview: item.summary,
          rating: item.rating,
          thumbnail: `${this.services.plex.baseUrl}${item.thumb}?X-Plex-Token=${this.services.plex.token}`
        })) || [];
        
      default:
        return [];
    }
  }

  // Get download queue from all download clients
  async getDownloadQueue() {
    const queue = [];
    const downloadClients = ['qbittorrent', 'sabnzbd', 'transmission'];
    
    await Promise.all(
      downloadClients.map(async (clientKey) => {
        const service = this.services[clientKey];
        if (this.healthStatus.get(clientKey)?.status !== 'online') return;
        
        try {
          const headers = this.getAuthHeaders(clientKey, service);
          let queueData;
          
          switch (clientKey) {
            case 'qbittorrent':
              // Login first for qBittorrent
              await axios.post(`${service.baseUrl}/api/v2/auth/login`, 
                `username=${service.username}&password=${service.password}`,
                { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
              );
              
              const torrentsResponse = await axios.get(`${service.baseUrl}/api/v2/torrents/info`, {
                withCredentials: true
              });
              queueData = torrentsResponse.data;
              break;
              
            case 'sabnzbd':
              const sabResponse = await axios.get(
                `${service.baseUrl}/sabnzbd/api?mode=queue&apikey=${service.apiKey}&output=json`
              );
              queueData = sabResponse.data.queue?.slots || [];
              break;
              
            case 'transmission':
              const transmissionResponse = await axios.post(
                `${service.baseUrl}/transmission/rpc`,
                {
                  method: 'torrent-get',
                  arguments: {
                    fields: ['id', 'name', 'status', 'percentDone', 'rateDownload', 'eta']
                  }
                },
                { headers }
              );
              queueData = transmissionResponse.data.arguments?.torrents || [];
              break;
          }
          
          queue.push({
            client: clientKey,
            items: this.normalizeQueueItems(clientKey, queueData)
          });
        } catch (error) {
          console.error(`Failed to get queue from ${clientKey}:`, error.message);
        }
      })
    );
    
    return queue;
  }

  // Normalize queue items across different download clients
  normalizeQueueItems(clientKey, data) {
    switch (clientKey) {
      case 'qbittorrent':
        return data.map(item => ({
          id: item.hash,
          name: item.name,
          size: item.size,
          progress: item.progress * 100,
          speed: item.dlspeed,
          eta: item.eta,
          status: item.state
        }));
        
      case 'sabnzbd':
        return data.map(item => ({
          id: item.nzo_id,
          name: item.filename,
          size: item.mb * 1024 * 1024,
          progress: parseFloat(item.percentage),
          speed: item.kbpersec * 1024,
          eta: item.timeleft,
          status: item.status
        }));
        
      case 'transmission':
        return data.map(item => ({
          id: item.id,
          name: item.name,
          size: item.totalSize,
          progress: item.percentDone * 100,
          speed: item.rateDownload,
          eta: item.eta,
          status: item.status === 4 ? 'downloading' : 'paused'
        }));
        
      default:
        return [];
    }
  }

  // Add media to download queue
  async addToQueue(mediaInfo, options = {}) {
    const { client = 'qbittorrent', priority = 'normal' } = options;
    const service = this.services[client];
    
    if (!service || this.healthStatus.get(client)?.status !== 'online') {
      throw new Error(`Download client ${client} is not available`);
    }
    
    try {
      const headers = this.getAuthHeaders(client, service);
      
      switch (client) {
        case 'qbittorrent':
          await axios.post(
            `${service.baseUrl}/api/v2/torrents/add`,
            {
              urls: mediaInfo.magnetLink || mediaInfo.torrentUrl,
              category: mediaInfo.category || 'media',
              paused: priority === 'low'
            },
            { headers, withCredentials: true }
          );
          break;
          
        case 'sabnzbd':
          await axios.post(
            `${service.baseUrl}/sabnzbd/api`,
            {
              mode: 'addurl',
              name: mediaInfo.nzbUrl,
              priority: priority === 'high' ? 2 : priority === 'low' ? -1 : 0,
              apikey: service.apiKey
            }
          );
          break;
          
        case 'transmission':
          await axios.post(
            `${service.baseUrl}/transmission/rpc`,
            {
              method: 'torrent-add',
              arguments: {
                filename: mediaInfo.magnetLink || mediaInfo.torrentUrl,
                paused: priority === 'low'
              }
            },
            { headers }
          );
          break;
      }
      
      return { success: true, client, mediaInfo };
    } catch (error) {
      throw new Error(`Failed to add to ${client}: ${error.message}`);
    }
  }

  // Get system statistics from all services
  async getSystemStatistics() {
    const stats = {
      mediaServers: {},
      contentManagers: {},
      downloadClients: {},
      monitoring: {},
      overall: {
        totalServices: Object.keys(this.services).length,
        onlineServices: 0,
        totalMedia: 0,
        activeDownloads: 0,
        diskUsage: 0,
        bandwidth: { in: 0, out: 0 }
      }
    };
    
    // Collect stats from each service type
    await Promise.all([
      this.getMediaServerStats(stats),
      this.getContentManagerStats(stats),
      this.getDownloadClientStats(stats),
      this.getMonitoringStats(stats)
    ]);
    
    // Calculate overall statistics
    stats.overall.onlineServices = Array.from(this.healthStatus.values())
      .filter(status => status.status === 'online').length;
    
    return stats;
  }

  async getMediaServerStats(stats) {
    const servers = ['jellyfin', 'plex', 'emby'];
    
    await Promise.all(
      servers.map(async (serverKey) => {
        if (this.healthStatus.get(serverKey)?.status !== 'online') return;
        
        try {
          const service = this.services[serverKey];
          const headers = this.getAuthHeaders(serverKey, service);
          
          // Get library statistics
          const response = await axios.get(
            `${service.baseUrl}${service.endpoints.libraries}`,
            { headers }
          );
          
          stats.mediaServers[serverKey] = {
            libraries: response.data.length || 0,
            totalItems: response.data.reduce((sum, lib) => sum + (lib.ItemCount || 0), 0)
          };
          
          stats.overall.totalMedia += stats.mediaServers[serverKey].totalItems;
        } catch (error) {
          console.error(`Failed to get stats from ${serverKey}:`, error.message);
        }
      })
    );
  }

  async getContentManagerStats(stats) {
    const managers = ['sonarr', 'radarr', 'lidarr', 'readarr'];
    
    await Promise.all(
      managers.map(async (managerKey) => {
        if (this.healthStatus.get(managerKey)?.status !== 'online') return;
        
        try {
          const service = this.services[managerKey];
          const headers = this.getAuthHeaders(managerKey, service);
          
          // Get queue statistics
          const response = await axios.get(
            `${service.baseUrl}${service.endpoints.queue}`,
            { headers }
          );
          
          stats.contentManagers[managerKey] = {
            queueSize: response.data.totalRecords || response.data.length || 0,
            monitoring: true
          };
        } catch (error) {
          console.error(`Failed to get stats from ${managerKey}:`, error.message);
        }
      })
    );
  }

  async getDownloadClientStats(stats) {
    const clients = ['qbittorrent', 'sabnzbd', 'transmission'];
    
    await Promise.all(
      clients.map(async (clientKey) => {
        if (this.healthStatus.get(clientKey)?.status !== 'online') return;
        
        try {
          const queue = await this.getDownloadQueue();
          const clientQueue = queue.find(q => q.client === clientKey);
          
          if (clientQueue) {
            stats.downloadClients[clientKey] = {
              activeDownloads: clientQueue.items.filter(i => i.status === 'downloading').length,
              totalDownloads: clientQueue.items.length,
              totalSpeed: clientQueue.items.reduce((sum, item) => sum + item.speed, 0)
            };
            
            stats.overall.activeDownloads += stats.downloadClients[clientKey].activeDownloads;
            stats.overall.bandwidth.in += stats.downloadClients[clientKey].totalSpeed;
          }
        } catch (error) {
          console.error(`Failed to get stats from ${clientKey}:`, error.message);
        }
      })
    );
  }

  async getMonitoringStats(stats) {
    const monitors = ['grafana', 'prometheus', 'uptimeKuma'];
    
    await Promise.all(
      monitors.map(async (monitorKey) => {
        if (this.healthStatus.get(monitorKey)?.status !== 'online') return;
        
        try {
          const service = this.services[monitorKey];
          const headers = this.getAuthHeaders(monitorKey, service);
          
          if (monitorKey === 'prometheus') {
            // Get Prometheus metrics
            const response = await axios.get(
              `${service.baseUrl}/api/v1/query?query=up`,
              { headers }
            );
            
            stats.monitoring[monitorKey] = {
              targetsUp: response.data.data?.result?.filter(r => r.value[1] === '1').length || 0,
              totalTargets: response.data.data?.result?.length || 0
            };
          }
        } catch (error) {
          console.error(`Failed to get stats from ${monitorKey}:`, error.message);
        }
      })
    );
  }

  // WebSocket connections for real-time updates
  establishWebSocketConnections() {
    // Jellyfin WebSocket
    if (this.services.jellyfin && this.healthStatus.get('jellyfin')?.status === 'online') {
      const jellyfinWs = new WebSocket(
        `ws://localhost:8096/socket?api_key=${this.services.jellyfin.apiKey}`
      );
      
      jellyfinWs.on('message', (data) => {
        const message = JSON.parse(data);
        this.emit('jellyfin:update', message);
      });
      
      this.wsConnections.set('jellyfin', jellyfinWs);
    }
    
    // Sonarr/Radarr SignalR connections
    ['sonarr', 'radarr'].forEach(service => {
      if (this.healthStatus.get(service)?.status === 'online') {
        // SignalR connection logic would go here
        // This is a simplified example
        this.emit(`${service}:connected`, { service });
      }
    });
  }

  // Batch operations across services
  async performBatchOperation(operation, targets) {
    const results = await Promise.allSettled(
      targets.map(target => this.executeOperation(operation, target))
    );
    
    return results.map((result, index) => ({
      target: targets[index],
      success: result.status === 'fulfilled',
      result: result.status === 'fulfilled' ? result.value : result.reason
    }));
  }

  async executeOperation(operation, target) {
    const { service, action, params } = target;
    const serviceConfig = this.services[service];
    
    if (!serviceConfig) {
      throw new Error(`Service ${service} not found`);
    }
    
    const headers = this.getAuthHeaders(service, serviceConfig);
    
    switch (action) {
      case 'restart':
        // Service restart logic
        return { restarted: true, service };
        
      case 'update':
        // Service update logic
        return { updated: true, service };
        
      case 'backup':
        // Service backup logic
        return { backed_up: true, service };
        
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  // Cleanup method
  async cleanup() {
    // Close WebSocket connections
    this.wsConnections.forEach(ws => ws.close());
    this.wsConnections.clear();
    
    // Clear health status
    this.healthStatus.clear();
    
    // Remove all event listeners
    this.removeAllListeners();
  }
}

export default UnifiedMediaAPI;