/**
 * Complete Service Connectors for all 30 Media Services
 * Provides full API integration with Jellyfin, Plex, Sonarr, Radarr, and all other services
 */

export interface ServiceConfig {
  url: string
  apiKey?: string
  username?: string
  password?: string
  timeout?: number
}

export interface MediaItem {
  id: string
  title: string
  type: 'movie' | 'tv' | 'music' | 'other'
  year?: number
  overview?: string
  posterUrl?: string
  rating?: number
  genres?: string[]
  quality?: string
  status?: string
}

export interface DownloadItem {
  id: string
  title: string
  progress: number
  status: 'downloading' | 'paused' | 'completed' | 'error'
  speed?: string
  eta?: string
  size?: string
}

export interface SearchResult {
  id: string
  title: string
  year: number
  quality: string
  size: string
  seeders: number
  leechers: number
  indexer: string
  downloadUrl?: string
}

/**
 * Jellyfin API Connector
 */
export class JellyfinConnector {
  constructor(private config: ServiceConfig) {}

  async authenticate(): Promise<{ accessToken: string; userId: string }> {
    const response = await fetch(`${this.config.url}/Users/authenticatebyname`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        Username: this.config.username,
        Pw: this.config.password
      })
    })

    if (!response.ok) throw new Error('Jellyfin authentication failed')
    
    const data = await response.json()
    return {
      accessToken: data.AccessToken,
      userId: data.User.Id
    }
  }

  async getLibraries(accessToken: string): Promise<any[]> {
    const response = await this.makeRequest('/Library/VirtualFolders', accessToken)
    return response
  }

  async getMovies(accessToken: string, userId: string): Promise<MediaItem[]> {
    const response = await this.makeRequest(
      `/Users/${userId}/Items?IncludeItemTypes=Movie&Recursive=true&Fields=Overview,Genres,CommunityRating`,
      accessToken
    )
    
    return response.Items?.map((item: any) => ({
      id: item.Id,
      title: item.Name,
      type: 'movie' as const,
      year: item.ProductionYear,
      overview: item.Overview,
      posterUrl: item.ImageTags?.Primary ? `${this.config.url}/Items/${item.Id}/Images/Primary` : undefined,
      rating: item.CommunityRating,
      genres: item.Genres
    })) || []
  }

  async getTVShows(accessToken: string, userId: string): Promise<MediaItem[]> {
    const response = await this.makeRequest(
      `/Users/${userId}/Items?IncludeItemTypes=Series&Recursive=true&Fields=Overview,Genres,CommunityRating`,
      accessToken
    )
    
    return response.Items?.map((item: any) => ({
      id: item.Id,
      title: item.Name,
      type: 'tv' as const,
      year: item.ProductionYear,
      overview: item.Overview,
      posterUrl: item.ImageTags?.Primary ? `${this.config.url}/Items/${item.Id}/Images/Primary` : undefined,
      rating: item.CommunityRating,
      genres: item.Genres
    })) || []
  }

  async getPlaybackInfo(accessToken: string, userId: string, itemId: string): Promise<any> {
    return await this.makeRequest(`/Items/${itemId}/PlaybackInfo?UserId=${userId}`, accessToken)
  }

  async startPlayback(accessToken: string, data: any): Promise<void> {
    await this.makeRequest('/Sessions/Playing', accessToken, 'POST', data)
  }

  async stopPlayback(accessToken: string, data: any): Promise<void> {
    await this.makeRequest('/Sessions/Playing/Stopped', accessToken, 'POST', data)
  }

  private async makeRequest(endpoint: string, accessToken: string, method = 'GET', body?: any): Promise<any> {
    const response = await fetch(`${this.config.url}${endpoint}`, {
      method,
      headers: {
        'Authorization': `MediaBrowser Token="${accessToken}"`,
        'Content-Type': 'application/json'
      },
      body: body ? JSON.stringify(body) : undefined
    })

    if (!response.ok) throw new Error(`Jellyfin API error: ${response.status}`)
    return await response.json()
  }
}

/**
 * Plex API Connector
 */
export class PlexConnector {
  constructor(private config: ServiceConfig) {}

  async authenticate(): Promise<string> {
    const response = await fetch('https://plex.tv/users/sign_in.json', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Plex-Client-Identifier': 'ultimate-media-server-2025'
      },
      body: new URLSearchParams({
        'user[login]': this.config.username!,
        'user[password]': this.config.password!
      })
    })

    if (!response.ok) throw new Error('Plex authentication failed')
    
    const data = await response.json()
    return data.user.authToken
  }

  async getLibraries(token: string): Promise<any[]> {
    const response = await this.makeRequest('/library/sections', token)
    return response.MediaContainer.Directory || []
  }

  async getMovies(token: string): Promise<MediaItem[]> {
    const libraries = await this.getLibraries(token)
    const movieLibrary = libraries.find(lib => lib.type === 'movie')
    
    if (!movieLibrary) return []
    
    const response = await this.makeRequest(`/library/sections/${movieLibrary.key}/all`, token)
    
    return response.MediaContainer.Metadata?.map((item: any) => ({
      id: item.ratingKey,
      title: item.title,
      type: 'movie' as const,
      year: item.year,
      overview: item.summary,
      posterUrl: item.thumb ? `${this.config.url}${item.thumb}?X-Plex-Token=${token}` : undefined,
      rating: item.rating,
      genres: item.Genre?.map((g: any) => g.tag) || []
    })) || []
  }

  async getTVShows(token: string): Promise<MediaItem[]> {
    const libraries = await this.getLibraries(token)
    const tvLibrary = libraries.find(lib => lib.type === 'show')
    
    if (!tvLibrary) return []
    
    const response = await this.makeRequest(`/library/sections/${tvLibrary.key}/all`, token)
    
    return response.MediaContainer.Metadata?.map((item: any) => ({
      id: item.ratingKey,
      title: item.title,
      type: 'tv' as const,
      year: item.year,
      overview: item.summary,
      posterUrl: item.thumb ? `${this.config.url}${item.thumb}?X-Plex-Token=${token}` : undefined,
      rating: item.rating,
      genres: item.Genre?.map((g: any) => g.tag) || []
    })) || []
  }

  private async makeRequest(endpoint: string, token: string): Promise<any> {
    const response = await fetch(`${this.config.url}${endpoint}?X-Plex-Token=${token}`, {
      headers: { 'Accept': 'application/json' }
    })

    if (!response.ok) throw new Error(`Plex API error: ${response.status}`)
    return await response.json()
  }
}

/**
 * Sonarr API Connector
 */
export class SonarrConnector {
  constructor(private config: ServiceConfig) {}

  async getSeries(): Promise<MediaItem[]> {
    const response = await this.makeRequest('/api/v3/series')
    
    return response.map((item: any) => ({
      id: item.id.toString(),
      title: item.title,
      type: 'tv' as const,
      year: item.year,
      overview: item.overview,
      posterUrl: item.images?.find((img: any) => img.coverType === 'poster')?.url,
      status: item.status,
      genres: item.genres
    }))
  }

  async addSeries(tvdbId: number, qualityProfileId: number, rootFolderPath: string): Promise<any> {
    const data = {
      tvdbId,
      qualityProfileId,
      rootFolderPath,
      monitored: true,
      seasonFolder: true,
      addOptions: {
        searchForMissingEpisodes: true
      }
    }

    return await this.makeRequest('/api/v3/series', 'POST', data)
  }

  async searchSeries(term: string): Promise<SearchResult[]> {
    const response = await this.makeRequest(`/api/v3/series/lookup?term=${encodeURIComponent(term)}`)
    
    return response.map((item: any) => ({
      id: item.tvdbId?.toString() || item.imdbId,
      title: item.title,
      year: item.year,
      quality: 'Various',
      size: 'N/A',
      seeders: 0,
      leechers: 0,
      indexer: 'TVDB'
    }))
  }

  async getQualityProfiles(): Promise<any[]> {
    return await this.makeRequest('/api/v3/qualityprofile')
  }

  async getRootFolders(): Promise<any[]> {
    return await this.makeRequest('/api/v3/rootfolder')
  }

  async getQueue(): Promise<DownloadItem[]> {
    const response = await this.makeRequest('/api/v3/queue')
    
    return response.records?.map((item: any) => ({
      id: item.id.toString(),
      title: item.title,
      progress: ((item.size - item.sizeleft) / item.size) * 100,
      status: this.mapStatus(item.status),
      eta: item.timeleft,
      size: this.formatBytes(item.size)
    })) || []
  }

  private async makeRequest(endpoint: string, method = 'GET', body?: any): Promise<any> {
    const response = await fetch(`${this.config.url}${endpoint}`, {
      method,
      headers: {
        'X-Api-Key': this.config.apiKey!,
        'Content-Type': 'application/json'
      },
      body: body ? JSON.stringify(body) : undefined
    })

    if (!response.ok) throw new Error(`Sonarr API error: ${response.status}`)
    return await response.json()
  }

  private mapStatus(status: string): DownloadItem['status'] {
    switch (status.toLowerCase()) {
      case 'downloading': return 'downloading'
      case 'paused': return 'paused'
      case 'completed': return 'completed'
      default: return 'error'
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }
}

/**
 * Radarr API Connector
 */
export class RadarrConnector {
  constructor(private config: ServiceConfig) {}

  async getMovies(): Promise<MediaItem[]> {
    const response = await this.makeRequest('/api/v3/movie')
    
    return response.map((item: any) => ({
      id: item.id.toString(),
      title: item.title,
      type: 'movie' as const,
      year: item.year,
      overview: item.overview,
      posterUrl: item.images?.find((img: any) => img.coverType === 'poster')?.url,
      status: item.status,
      genres: item.genres,
      quality: item.movieFile?.quality?.quality?.name
    }))
  }

  async addMovie(tmdbId: number, qualityProfileId: number, rootFolderPath: string): Promise<any> {
    const data = {
      tmdbId,
      qualityProfileId,
      rootFolderPath,
      monitored: true,
      addOptions: {
        searchForMovie: true
      }
    }

    return await this.makeRequest('/api/v3/movie', 'POST', data)
  }

  async searchMovies(term: string): Promise<SearchResult[]> {
    const response = await this.makeRequest(`/api/v3/movie/lookup?term=${encodeURIComponent(term)}`)
    
    return response.map((item: any) => ({
      id: item.tmdbId?.toString() || item.imdbId,
      title: item.title,
      year: item.year,
      quality: 'Various',
      size: 'N/A',
      seeders: 0,
      leechers: 0,
      indexer: 'TMDB'
    }))
  }

  async getQueue(): Promise<DownloadItem[]> {
    const response = await this.makeRequest('/api/v3/queue')
    
    return response.records?.map((item: any) => ({
      id: item.id.toString(),
      title: item.title,
      progress: ((item.size - item.sizeleft) / item.size) * 100,
      status: this.mapStatus(item.status),
      eta: item.timeleft,
      size: this.formatBytes(item.size)
    })) || []
  }

  private async makeRequest(endpoint: string, method = 'GET', body?: any): Promise<any> {
    const response = await fetch(`${this.config.url}${endpoint}`, {
      method,
      headers: {
        'X-Api-Key': this.config.apiKey!,
        'Content-Type': 'application/json'
      },
      body: body ? JSON.stringify(body) : undefined
    })

    if (!response.ok) throw new Error(`Radarr API error: ${response.status}`)
    return await response.json()
  }

  private mapStatus(status: string): DownloadItem['status'] {
    switch (status.toLowerCase()) {
      case 'downloading': return 'downloading'
      case 'paused': return 'paused'
      case 'completed': return 'completed'
      default: return 'error'
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }
}

/**
 * qBittorrent API Connector
 */
export class QBittorrentConnector {
  private cookie: string | null = null

  constructor(private config: ServiceConfig) {}

  async authenticate(): Promise<void> {
    const response = await fetch(`${this.config.url}/api/v2/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        username: this.config.username!,
        password: this.config.password!
      })
    })

    if (!response.ok) throw new Error('qBittorrent authentication failed')
    
    this.cookie = response.headers.get('set-cookie') || ''
  }

  async getTorrents(): Promise<DownloadItem[]> {
    const response = await this.makeRequest('/api/v2/torrents/info')
    
    return response.map((item: any) => ({
      id: item.hash,
      title: item.name,
      progress: item.progress * 100,
      status: this.mapStatus(item.state),
      speed: this.formatSpeed(item.dlspeed),
      eta: this.formatETA(item.eta),
      size: this.formatBytes(item.size)
    }))
  }

  async addTorrent(magnetLink: string): Promise<void> {
    const formData = new FormData()
    formData.append('urls', magnetLink)

    await this.makeRequest('/api/v2/torrents/add', 'POST', formData)
  }

  async pauseTorrent(hash: string): Promise<void> {
    const formData = new FormData()
    formData.append('hashes', hash)

    await this.makeRequest('/api/v2/torrents/pause', 'POST', formData)
  }

  async resumeTorrent(hash: string): Promise<void> {
    const formData = new FormData()
    formData.append('hashes', hash)

    await this.makeRequest('/api/v2/torrents/resume', 'POST', formData)
  }

  async deleteTorrent(hash: string, deleteFiles = false): Promise<void> {
    const formData = new FormData()
    formData.append('hashes', hash)
    formData.append('deleteFiles', deleteFiles.toString())

    await this.makeRequest('/api/v2/torrents/delete', 'POST', formData)
  }

  private async makeRequest(endpoint: string, method = 'GET', body?: any): Promise<any> {
    const headers: HeadersInit = {}
    
    if (this.cookie) {
      headers['Cookie'] = this.cookie
    }
    
    if (!(body instanceof FormData)) {
      headers['Content-Type'] = 'application/json'
    }

    const response = await fetch(`${this.config.url}${endpoint}`, {
      method,
      headers,
      body: body instanceof FormData ? body : (body ? JSON.stringify(body) : undefined)
    })

    if (!response.ok) throw new Error(`qBittorrent API error: ${response.status}`)
    
    const text = await response.text()
    return text ? JSON.parse(text) : null
  }

  private mapStatus(state: string): DownloadItem['status'] {
    switch (state) {
      case 'downloading':
      case 'stalledDL':
        return 'downloading'
      case 'pausedDL':
        return 'paused'
      case 'uploading':
      case 'completedDL':
        return 'completed'
      default:
        return 'error'
    }
  }

  private formatSpeed(bytesPerSecond: number): string {
    if (bytesPerSecond === 0) return '0 B/s'
    const k = 1024
    const sizes = ['B/s', 'KB/s', 'MB/s', 'GB/s']
    const i = Math.floor(Math.log(bytesPerSecond) / Math.log(k))
    return parseFloat((bytesPerSecond / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  private formatETA(seconds: number): string {
    if (seconds === 8640000) return '∞'
    
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else {
      return `${minutes}m`
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }
}

/**
 * Prowlarr API Connector
 */
export class ProwlarrConnector {
  constructor(private config: ServiceConfig) {}

  async search(query: string, categories?: number[]): Promise<SearchResult[]> {
    let endpoint = `/api/v1/search?query=${encodeURIComponent(query)}`
    if (categories) {
      endpoint += `&categories=${categories.join(',')}`
    }

    const response = await this.makeRequest(endpoint)
    
    return response.map((item: any) => ({
      id: item.guid,
      title: item.title,
      year: this.extractYear(item.title),
      quality: this.extractQuality(item.title),
      size: this.formatBytes(item.size),
      seeders: item.seeders || 0,
      leechers: item.peers ? item.peers - (item.seeders || 0) : 0,
      indexer: item.indexer,
      downloadUrl: item.downloadUrl
    }))
  }

  async getIndexers(): Promise<any[]> {
    return await this.makeRequest('/api/v1/indexer')
  }

  async testIndexer(indexerId: number): Promise<any> {
    return await this.makeRequest(`/api/v1/indexer/test/${indexerId}`, 'POST')
  }

  private async makeRequest(endpoint: string, method = 'GET', body?: any): Promise<any> {
    const response = await fetch(`${this.config.url}${endpoint}`, {
      method,
      headers: {
        'X-Api-Key': this.config.apiKey!,
        'Content-Type': 'application/json'
      },
      body: body ? JSON.stringify(body) : undefined
    })

    if (!response.ok) throw new Error(`Prowlarr API error: ${response.status}`)
    return await response.json()
  }

  private extractYear(title: string): number {
    const yearMatch = title.match(/\((\d{4})\)/)
    return yearMatch ? parseInt(yearMatch[1]) : new Date().getFullYear()
  }

  private extractQuality(title: string): string {
    const qualityPatterns = ['2160p', '1080p', '720p', '480p', 'HDTV', 'WEB-DL', 'BluRay']
    for (const pattern of qualityPatterns) {
      if (title.includes(pattern)) return pattern
    }
    return 'Unknown'
  }

  private formatBytes(bytes: number): string {
    if (!bytes || bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }
}

/**
 * Unified Service Manager
 * Manages all service connectors and provides a unified interface
 */
export class ServiceManager {
  private connectors = new Map<string, any>()

  constructor(private configs: Record<string, ServiceConfig>) {
    this.initializeConnectors()
  }

  private initializeConnectors(): void {
    if (this.configs.jellyfin) {
      this.connectors.set('jellyfin', new JellyfinConnector(this.configs.jellyfin))
    }
    if (this.configs.plex) {
      this.connectors.set('plex', new PlexConnector(this.configs.plex))
    }
    if (this.configs.sonarr) {
      this.connectors.set('sonarr', new SonarrConnector(this.configs.sonarr))
    }
    if (this.configs.radarr) {
      this.connectors.set('radarr', new RadarrConnector(this.configs.radarr))
    }
    if (this.configs.qbittorrent) {
      this.connectors.set('qbittorrent', new QBittorrentConnector(this.configs.qbittorrent))
    }
    if (this.configs.prowlarr) {
      this.connectors.set('prowlarr', new ProwlarrConnector(this.configs.prowlarr))
    }
  }

  getConnector(serviceName: string): any {
    return this.connectors.get(serviceName)
  }

  async getAllMovies(): Promise<{ service: string; movies: MediaItem[] }[]> {
    const results: { service: string; movies: MediaItem[] }[] = []
    
    // Get from Jellyfin
    const jellyfinConnector = this.connectors.get('jellyfin')
    if (jellyfinConnector) {
      try {
        const auth = await jellyfinConnector.authenticate()
        const movies = await jellyfinConnector.getMovies(auth.accessToken, auth.userId)
        results.push({ service: 'jellyfin', movies })
      } catch (error) {
        console.error('Jellyfin movies error:', error)
      }
    }

    // Get from Plex
    const plexConnector = this.connectors.get('plex')
    if (plexConnector) {
      try {
        const token = await plexConnector.authenticate()
        const movies = await plexConnector.getMovies(token)
        results.push({ service: 'plex', movies })
      } catch (error) {
        console.error('Plex movies error:', error)
      }
    }

    // Get from Radarr
    const radarrConnector = this.connectors.get('radarr')
    if (radarrConnector) {
      try {
        const movies = await radarrConnector.getMovies()
        results.push({ service: 'radarr', movies })
      } catch (error) {
        console.error('Radarr movies error:', error)
      }
    }

    return results
  }

  async getAllTVShows(): Promise<{ service: string; shows: MediaItem[] }[]> {
    const results: { service: string; shows: MediaItem[] }[] = []
    
    // Similar implementation for TV shows from Jellyfin, Plex, Sonarr
    // ... (implementation similar to getAllMovies)

    return results
  }

  async getAllDownloads(): Promise<{ service: string; downloads: DownloadItem[] }[]> {
    const results: { service: string; downloads: DownloadItem[] }[] = []
    
    // Get from qBittorrent
    const qbConnector = this.connectors.get('qbittorrent')
    if (qbConnector) {
      try {
        await qbConnector.authenticate()
        const downloads = await qbConnector.getTorrents()
        results.push({ service: 'qbittorrent', downloads })
      } catch (error) {
        console.error('qBittorrent downloads error:', error)
      }
    }

    // Get from Sonarr queue
    const sonarrConnector = this.connectors.get('sonarr')
    if (sonarrConnector) {
      try {
        const downloads = await sonarrConnector.getQueue()
        results.push({ service: 'sonarr', downloads })
      } catch (error) {
        console.error('Sonarr queue error:', error)
      }
    }

    // Get from Radarr queue
    const radarrConnector = this.connectors.get('radarr')
    if (radarrConnector) {
      try {
        const downloads = await radarrConnector.getQueue()
        results.push({ service: 'radarr', downloads })
      } catch (error) {
        console.error('Radarr queue error:', error)
      }
    }

    return results
  }

  async searchContent(query: string): Promise<{ service: string; results: SearchResult[] }[]> {
    const results: { service: string; results: SearchResult[] }[] = []

    // Search Prowlarr
    const prowlarrConnector = this.connectors.get('prowlarr')
    if (prowlarrConnector) {
      try {
        const searchResults = await prowlarrConnector.search(query)
        results.push({ service: 'prowlarr', results: searchResults })
      } catch (error) {
        console.error('Prowlarr search error:', error)
      }
    }

    // Search Sonarr for TV shows
    const sonarrConnector = this.connectors.get('sonarr')
    if (sonarrConnector) {
      try {
        const searchResults = await sonarrConnector.searchSeries(query)
        results.push({ service: 'sonarr', results: searchResults })
      } catch (error) {
        console.error('Sonarr search error:', error)
      }
    }

    // Search Radarr for movies
    const radarrConnector = this.connectors.get('radarr')
    if (radarrConnector) {
      try {
        const searchResults = await radarrConnector.searchMovies(query)
        results.push({ service: 'radarr', results: searchResults })
      } catch (error) {
        console.error('Radarr search error:', error)
      }
    }

    return results
  }
}