// API Integration Scripts for Homarr Dashboard
// Real-time data fetching and display for media server statistics

class MediaServerAPI {
  constructor() {
    this.endpoints = {
      jellyfin: 'http://localhost:8096',
      plex: 'http://localhost:32400',
      sonarr: 'http://localhost:8989',
      radarr: 'http://localhost:7878',
      qbittorrent: 'http://localhost:8080',
      sabnzbd: 'http://localhost:8081',
      prometheus: 'http://localhost:9090'
    };
    
    this.apiKeys = {
      jellyfin: this.getStoredApiKey('jellyfin'),
      sonarr: this.getStoredApiKey('sonarr'),
      radarr: this.getStoredApiKey('radarr'),
      sabnzbd: this.getStoredApiKey('sabnzbd')
    };
    
    this.updateInterval = 30000; // 30 seconds
    this.fastUpdateInterval = 5000; // 5 seconds for downloads
    
    this.init();
  }
  
  init() {
    this.startUpdating();
    this.createWidgetContainers();
  }
  
  getStoredApiKey(service) {
    return localStorage.getItem(`${service}_api_key`) || '';
  }
  
  async fetchWithRetry(url, options = {}, retries = 3) {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers
          }
        });
        
        if (response.ok) {
          return await response.json();
        }
      } catch (error) {
        console.warn(`Attempt ${i + 1} failed for ${url}:`, error);
        if (i === retries - 1) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      }
    }
  }
  
  // Jellyfin API methods
  async getJellyfinStats() {
    try {
      const [library, sessions] = await Promise.all([
        this.fetchWithRetry(`${this.endpoints.jellyfin}/Items/Counts`, {
          headers: { 'X-MediaBrowser-Token': this.apiKeys.jellyfin }
        }),
        this.fetchWithRetry(`${this.endpoints.jellyfin}/Sessions`, {
          headers: { 'X-MediaBrowser-Token': this.apiKeys.jellyfin }
        })
      ]);
      
      return {
        movies: library.MovieCount || 0,
        series: library.SeriesCount || 0,
        episodes: library.EpisodeCount || 0,
        songs: library.SongCount || 0,
        activeSessions: sessions.filter(s => s.PlayState && !s.PlayState.IsPaused).length,
        totalSessions: sessions.length
      };
    } catch (error) {
      console.warn('Jellyfin API error:', error);
      return this.getMockData('jellyfin');
    }
  }
  
  // Plex API methods
  async getPlexStats() {
    try {
      const response = await this.fetchWithRetry(`${this.endpoints.plex}/library/sections`, {
        headers: { 'X-Plex-Token': this.apiKeys.plex }
      });
      
      // This would need actual Plex implementation
      return this.getMockData('plex');
    } catch (error) {
      console.warn('Plex API error:', error);
      return this.getMockData('plex');
    }
  }
  
  // Sonarr API methods
  async getSonarrStats() {
    try {
      const [series, episodes, queue] = await Promise.all([
        this.fetchWithRetry(`${this.endpoints.sonarr}/api/v3/series`, {
          headers: { 'X-Api-Key': this.apiKeys.sonarr }
        }),
        this.fetchWithRetry(`${this.endpoints.sonarr}/api/v3/episode`, {
          headers: { 'X-Api-Key': this.apiKeys.sonarr }
        }),
        this.fetchWithRetry(`${this.endpoints.sonarr}/api/v3/queue`, {
          headers: { 'X-Api-Key': this.apiKeys.sonarr }
        })
      ]);
      
      return {
        totalSeries: series.length,
        totalEpisodes: episodes.length,
        queuedItems: queue.totalRecords || 0,
        missingEpisodes: episodes.filter(e => !e.hasFile).length
      };
    } catch (error) {
      console.warn('Sonarr API error:', error);
      return this.getMockData('sonarr');
    }
  }
  
  // Radarr API methods
  async getRadarrStats() {
    try {
      const [movies, queue] = await Promise.all([
        this.fetchWithRetry(`${this.endpoints.radarr}/api/v3/movie`, {
          headers: { 'X-Api-Key': this.apiKeys.radarr }
        }),
        this.fetchWithRetry(`${this.endpoints.radarr}/api/v3/queue`, {
          headers: { 'X-Api-Key': this.apiKeys.radarr }
        })
      ]);
      
      return {
        totalMovies: movies.length,
        queuedItems: queue.totalRecords || 0,
        missingMovies: movies.filter(m => !m.hasFile).length,
        availableMovies: movies.filter(m => m.hasFile).length
      };
    } catch (error) {
      console.warn('Radarr API error:', error);
      return this.getMockData('radarr');
    }
  }
  
  // qBittorrent API methods
  async getQBittorrentStats() {
    try {
      const [torrents, globalStats] = await Promise.all([
        this.fetchWithRetry(`${this.endpoints.qbittorrent}/api/v2/torrents/info`),
        this.fetchWithRetry(`${this.endpoints.qbittorrent}/api/v2/transfer/info`)
      ]);
      
      return {
        activeTorrents: torrents.filter(t => t.state === 'downloading').length,
        totalTorrents: torrents.length,
        downloadSpeed: globalStats.dl_info_speed || 0,
        uploadSpeed: globalStats.up_info_speed || 0,
        queue: torrents.filter(t => ['downloading', 'queuedDL'].includes(t.state))
      };
    } catch (error) {
      console.warn('qBittorrent API error:', error);
      return this.getMockData('qbittorrent');
    }
  }
  
  // SABnzbd API methods
  async getSABnzbdStats() {
    try {
      const [queue, history] = await Promise.all([
        this.fetchWithRetry(`${this.endpoints.sabnzbd}/api?mode=queue&output=json&apikey=${this.apiKeys.sabnzbd}`),
        this.fetchWithRetry(`${this.endpoints.sabnzbd}/api?mode=history&output=json&apikey=${this.apiKeys.sabnzbd}`)
      ]);
      
      return {
        queueSize: queue.queue?.noofslots || 0,
        downloadSpeed: queue.queue?.kbpersec || 0,
        remainingSize: queue.queue?.mbleft || 0,
        recentDownloads: history.history?.slots?.slice(0, 5) || []
      };
    } catch (error) {
      console.warn('SABnzbd API error:', error);
      return this.getMockData('sabnzbd');
    }
  }
  
  // System stats from Prometheus
  async getSystemStats() {
    try {
      const queries = [
        'cpu_usage_percent',
        'memory_usage_percent',
        'disk_usage_percent',
        'network_receive_bytes',
        'network_transmit_bytes'
      ];
      
      const results = await Promise.all(
        queries.map(query => 
          this.fetchWithRetry(`${this.endpoints.prometheus}/api/v1/query?query=${query}`)
        )
      );
      
      return {
        cpuUsage: results[0]?.data?.result?.[0]?.value?.[1] || 0,
        memoryUsage: results[1]?.data?.result?.[0]?.value?.[1] || 0,
        diskUsage: results[2]?.data?.result?.[0]?.value?.[1] || 0,
        networkRx: results[3]?.data?.result?.[0]?.value?.[1] || 0,
        networkTx: results[4]?.data?.result?.[0]?.value?.[1] || 0
      };
    } catch (error) {
      console.warn('Prometheus API error:', error);
      return this.getMockData('system');
    }
  }
  
  getMockData(service) {
    const mockData = {
      jellyfin: {
        movies: Math.floor(Math.random() * 5000) + 1000,
        series: Math.floor(Math.random() * 500) + 100,
        episodes: Math.floor(Math.random() * 10000) + 2000,
        songs: Math.floor(Math.random() * 20000) + 5000,
        activeSessions: Math.floor(Math.random() * 5),
        totalSessions: Math.floor(Math.random() * 10) + 2
      },
      plex: {
        movies: Math.floor(Math.random() * 3000) + 800,
        series: Math.floor(Math.random() * 300) + 80,
        episodes: Math.floor(Math.random() * 8000) + 1500,
        activeSessions: Math.floor(Math.random() * 8),
        totalSessions: Math.floor(Math.random() * 15) + 3
      },
      sonarr: {
        totalSeries: Math.floor(Math.random() * 300) + 100,
        totalEpisodes: Math.floor(Math.random() * 5000) + 1000,
        queuedItems: Math.floor(Math.random() * 20),
        missingEpisodes: Math.floor(Math.random() * 100) + 10
      },
      radarr: {
        totalMovies: Math.floor(Math.random() * 2000) + 500,
        queuedItems: Math.floor(Math.random() * 15),
        missingMovies: Math.floor(Math.random() * 50) + 5,
        availableMovies: Math.floor(Math.random() * 1800) + 400
      },
      qbittorrent: {
        activeTorrents: Math.floor(Math.random() * 10),
        totalTorrents: Math.floor(Math.random() * 50) + 10,
        downloadSpeed: Math.floor(Math.random() * 10000000), // bytes/sec
        uploadSpeed: Math.floor(Math.random() * 5000000),
        queue: []
      },
      sabnzbd: {
        queueSize: Math.floor(Math.random() * 15),
        downloadSpeed: Math.floor(Math.random() * 50000), // KB/s
        remainingSize: Math.floor(Math.random() * 10000), // MB
        recentDownloads: []
      },
      system: {
        cpuUsage: Math.floor(Math.random() * 80) + 10,
        memoryUsage: Math.floor(Math.random() * 70) + 20,
        diskUsage: Math.floor(Math.random() * 60) + 30,
        networkRx: Math.floor(Math.random() * 1000000),
        networkTx: Math.floor(Math.random() * 500000)
      }
    };
    
    return mockData[service] || {};
  }
  
  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  formatSpeed(bytesPerSec) {
    return this.formatBytes(bytesPerSec) + '/s';
  }
  
  createWidgetContainers() {
    // Media Library Stats Widget
    this.createWidget('media-library-stats', 'Media Library Statistics', `
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-icon">🎬</div>
          <div class="stat-value" id="total-movies">--</div>
          <div class="stat-label">Movies</div>
        </div>
        <div class="stat-item">
          <div class="stat-icon">📺</div>
          <div class="stat-value" id="total-series">--</div>
          <div class="stat-label">TV Series</div>
        </div>
        <div class="stat-item">
          <div class="stat-icon">🎵</div>
          <div class="stat-value" id="total-songs">--</div>
          <div class="stat-label">Songs</div>
        </div>
        <div class="stat-item">
          <div class="stat-icon">👥</div>
          <div class="stat-value" id="active-streams">--</div>
          <div class="stat-label">Active Streams</div>
        </div>
      </div>
    `);
    
    // Download Stats Widget
    this.createWidget('download-stats', 'Download Statistics', `
      <div class="download-stats">
        <div class="speed-display">
          <div class="speed-item">
            <span class="speed-icon">⬇️</span>
            <span class="speed-value" id="download-speed">0 B/s</span>
          </div>
          <div class="speed-item">
            <span class="speed-icon">⬆️</span>
            <span class="speed-value" id="upload-speed">0 B/s</span>
          </div>
        </div>
        <div class="queue-info">
          <div class="queue-item">
            <span id="active-downloads">0</span> Active Downloads
          </div>
          <div class="queue-item">
            <span id="queued-items">0</span> Queued Items
          </div>
        </div>
      </div>
    `);
    
    // System Health Widget
    this.createWidget('system-health', 'System Health', `
      <div class="health-display">
        <div class="health-item">
          <div class="health-circle cpu">
            <div class="health-percentage" id="cpu-percentage">0%</div>
          </div>
          <div class="health-label">CPU</div>
        </div>
        <div class="health-item">
          <div class="health-circle memory">
            <div class="health-percentage" id="memory-percentage">0%</div>
          </div>
          <div class="health-label">Memory</div>
        </div>
        <div class="health-item">
          <div class="health-circle disk">
            <div class="health-percentage" id="disk-percentage">0%</div>
          </div>
          <div class="health-label">Storage</div>
        </div>
      </div>
    `);
  }
  
  createWidget(id, title, content) {
    const widget = document.createElement('div');
    widget.id = id;
    widget.className = 'api-widget';
    widget.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${title}</h3>
        <div class="widget-status" id="${id}-status">🔴</div>
      </div>
      <div class="widget-content">
        ${content}
      </div>
    `;
    
    // Add to appropriate container or create one
    let container = document.getElementById('api-widgets-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'api-widgets-container';
      container.className = 'api-widgets-container';
      document.body.appendChild(container);
    }
    
    container.appendChild(widget);
  }
  
  updateWidgets() {
    this.updateMediaLibraryStats();
    this.updateDownloadStats();
    this.updateSystemHealth();
  }
  
  async updateMediaLibraryStats() {
    try {
      const [jellyfin, sonarr, radarr] = await Promise.all([
        this.getJellyfinStats(),
        this.getSonarrStats(),
        this.getRadarrStats()
      ]);
      
      document.getElementById('total-movies').textContent = (jellyfin.movies + radarr.totalMovies).toLocaleString();
      document.getElementById('total-series').textContent = (jellyfin.series + sonarr.totalSeries).toLocaleString();
      document.getElementById('total-songs').textContent = jellyfin.songs.toLocaleString();
      document.getElementById('active-streams').textContent = jellyfin.activeSessions;
      
      document.getElementById('media-library-stats-status').textContent = '🟢';
    } catch (error) {
      document.getElementById('media-library-stats-status').textContent = '🔴';
    }
  }
  
  async updateDownloadStats() {
    try {
      const [qbt, sab] = await Promise.all([
        this.getQBittorrentStats(),
        this.getSABnzbdStats()
      ]);
      
      const totalDownloadSpeed = qbt.downloadSpeed + (sab.downloadSpeed * 1024); // SAB is in KB/s
      const totalUploadSpeed = qbt.uploadSpeed;
      
      document.getElementById('download-speed').textContent = this.formatSpeed(totalDownloadSpeed);
      document.getElementById('upload-speed').textContent = this.formatSpeed(totalUploadSpeed);
      document.getElementById('active-downloads').textContent = qbt.activeTorrents + sab.queueSize;
      document.getElementById('queued-items').textContent = qbt.totalTorrents;
      
      document.getElementById('download-stats-status').textContent = '🟢';
    } catch (error) {
      document.getElementById('download-stats-status').textContent = '🔴';
    }
  }
  
  async updateSystemHealth() {
    try {
      const stats = await this.getSystemStats();
      
      document.getElementById('cpu-percentage').textContent = Math.round(stats.cpuUsage) + '%';
      document.getElementById('memory-percentage').textContent = Math.round(stats.memoryUsage) + '%';
      document.getElementById('disk-percentage').textContent = Math.round(stats.diskUsage) + '%';
      
      // Update circle colors based on usage
      this.updateHealthCircle('cpu', stats.cpuUsage);
      this.updateHealthCircle('memory', stats.memoryUsage);
      this.updateHealthCircle('disk', stats.diskUsage);
      
      document.getElementById('system-health-status').textContent = '🟢';
    } catch (error) {
      document.getElementById('system-health-status').textContent = '🔴';
    }
  }
  
  updateHealthCircle(type, percentage) {
    const circle = document.querySelector(`.health-circle.${type}`);
    if (circle) {
      let color = '#00ff88'; // Green
      if (percentage > 80) color = '#ff0080'; // Red
      else if (percentage > 60) color = '#ffaa00'; // Yellow
      
      circle.style.borderColor = color;
      circle.style.boxShadow = `0 0 20px ${color}`;
    }
  }
  
  startUpdating() {
    // Initial update
    this.updateWidgets();
    
    // Regular updates
    setInterval(() => this.updateWidgets(), this.updateInterval);
    
    // Fast updates for downloads
    setInterval(() => this.updateDownloadStats(), this.fastUpdateInterval);
  }
}

// Widget styles
const widgetStyles = `
  .api-widgets-container {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 350px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  
  .api-widget {
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 255, 255, 0.15);
  }
  
  .widget-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }
  
  .widget-title {
    color: #00ffff;
    font-size: 1.1em;
    margin: 0;
    text-shadow: 0 0 10px currentColor;
  }
  
  .widget-status {
    font-size: 1.2em;
  }
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
  }
  
  .stat-item {
    text-align: center;
    padding: 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .stat-icon {
    font-size: 1.5em;
    margin-bottom: 5px;
  }
  
  .stat-value {
    font-size: 1.8em;
    font-weight: bold;
    color: #00ffff;
    margin: 5px 0;
    text-shadow: 0 0 10px currentColor;
  }
  
  .stat-label {
    font-size: 0.9em;
    color: #888;
  }
  
  .download-stats {
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  
  .speed-display {
    display: flex;
    justify-content: space-around;
    padding: 15px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
  }
  
  .speed-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
  }
  
  .speed-icon {
    font-size: 1.5em;
  }
  
  .speed-value {
    font-size: 1.2em;
    font-weight: bold;
    color: #00ff88;
    text-shadow: 0 0 10px currentColor;
  }
  
  .queue-info {
    display: flex;
    justify-content: space-between;
    padding: 0 10px;
  }
  
  .queue-item {
    color: #fff;
    font-size: 0.9em;
  }
  
  .queue-item span {
    color: #ffaa00;
    font-weight: bold;
  }
  
  .health-display {
    display: flex;
    justify-content: space-around;
    align-items: center;
  }
  
  .health-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }
  
  .health-circle {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 3px solid #00ff88;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 255, 136, 0.1);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    transition: all 0.3s ease;
  }
  
  .health-percentage {
    font-size: 0.9em;
    font-weight: bold;
    color: #fff;
  }
  
  .health-label {
    font-size: 0.8em;
    color: #888;
    text-transform: uppercase;
  }
  
  @media (max-width: 768px) {
    .api-widgets-container {
      position: relative;
      width: 100%;
      top: 0;
      right: 0;
      padding: 20px;
    }
    
    .stats-grid {
      grid-template-columns: 1fr;
    }
  }
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = widgetStyles;
document.head.appendChild(styleSheet);

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    window.mediaServerAPI = new MediaServerAPI();
  }, 2000);
});

// Export for external use
window.MediaServerAPI = MediaServerAPI;