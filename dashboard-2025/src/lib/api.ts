// API client for connecting to real services
import axios from 'axios';

interface ServiceHealth {
  name: string;
  status: 'online' | 'offline' | 'error';
  responseTime: number;
  version?: string;
  info?: any;
}

// Service endpoints configuration
const SERVICES = {
  jellyfin: { url: 'http://localhost:8096', healthPath: '/health' },
  sonarr: { url: 'http://localhost:8989', healthPath: '/api/v3/system/status', apiKey: process.env.NEXT_PUBLIC_SONARR_API_KEY },
  radarr: { url: 'http://localhost:7878', healthPath: '/api/v3/system/status', apiKey: process.env.NEXT_PUBLIC_RADARR_API_KEY },
  prowlarr: { url: 'http://localhost:9696', healthPath: '/api/v1/health', apiKey: process.env.NEXT_PUBLIC_PROWLARR_API_KEY },
  qbittorrent: { url: 'http://localhost:8080', healthPath: '/api/v2/app/version' },
};

// Check individual service health
export async function checkServiceHealth(serviceName: string): Promise<ServiceHealth> {
  const service = SERVICES[serviceName as keyof typeof SERVICES];
  if (!service) {
    return { name: serviceName, status: 'offline', responseTime: 0 };
  }

  const startTime = Date.now();
  try {
    const headers: any = {};
    if ('apiKey' in service && service.apiKey) {
      headers['X-Api-Key'] = service.apiKey;
    }

    const response = await axios.get(`${service.url}${service.healthPath}`, {
      headers,
      timeout: 5000,
    });

    return {
      name: serviceName,
      status: 'online',
      responseTime: Date.now() - startTime,
      info: response.data,
    };
  } catch (error) {
    return {
      name: serviceName,
      status: 'error',
      responseTime: Date.now() - startTime,
    };
  }
}

// Get all services health
export async function getAllServicesHealth(): Promise<ServiceHealth[]> {
  const healthChecks = Object.keys(SERVICES).map(service => checkServiceHealth(service));
  return Promise.all(healthChecks);
}

// Jellyfin specific API calls
export async function getJellyfinInfo() {
  try {
    const response = await axios.get('http://localhost:8096/System/Info/Public');
    return response.data;
  } catch (error) {
    console.error('Failed to get Jellyfin info:', error);
    return null;
  }
}

// Sonarr API calls
export async function getSonarrSeries() {
  try {
    const response = await axios.get('http://localhost:8989/api/v3/series', {
      headers: { 'X-Api-Key': process.env.NEXT_PUBLIC_SONARR_API_KEY || '' }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to get Sonarr series:', error);
    return [];
  }
}

// Radarr API calls
export async function getRadarrMovies() {
  try {
    const response = await axios.get('http://localhost:7878/api/v3/movie', {
      headers: { 'X-Api-Key': process.env.NEXT_PUBLIC_RADARR_API_KEY || '' }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to get Radarr movies:', error);
    return [];
  }
}

// qBittorrent API calls
export async function getQBittorrentStats() {
  try {
    // First login to get cookie
    await axios.post('http://localhost:8080/api/v2/auth/login', 
      `username=admin&password=${process.env.NEXT_PUBLIC_QBITTORRENT_PASSWORD || 'adminadmin'}`,
      { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
    );
    
    const stats = await axios.get('http://localhost:8080/api/v2/transfer/info');
    return stats.data;
  } catch (error) {
    console.error('Failed to get qBittorrent stats:', error);
    return null;
  }
}

// Docker stats
export async function getDockerStats() {
  try {
    const response = await axios.get('http://localhost:2375/containers/json');
    return response.data;
  } catch (error) {
    console.error('Failed to get Docker stats:', error);
    return [];
  }
}