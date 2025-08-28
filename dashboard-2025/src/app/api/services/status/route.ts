import { NextResponse } from 'next/server';

// Service health check endpoints
const HEALTH_CHECKS = [
  { name: 'Jellyfin', url: 'http://localhost:8096/health', method: 'GET' },
  { name: 'Sonarr', url: 'http://localhost:8989/api/v3/health', method: 'GET', headers: { 'X-Api-Key': process.env.SONARR_API_KEY || '' } },
  { name: 'Radarr', url: 'http://localhost:7878/api/v3/health', method: 'GET', headers: { 'X-Api-Key': process.env.RADARR_API_KEY || '' } },
  { name: 'Prowlarr', url: 'http://localhost:9696/api/v1/health', method: 'GET', headers: { 'X-Api-Key': process.env.PROWLARR_API_KEY || '' } },
  { name: 'qBittorrent', url: 'http://localhost:8080/api/v2/app/version', method: 'GET' },
  { name: 'Archon UI', url: 'http://localhost:3737', method: 'HEAD' },
  { name: 'Archon Server', url: 'http://localhost:8181/health', method: 'GET' },
  { name: 'PostgreSQL', url: 'http://localhost:5432', method: 'HEAD' },
  { name: 'Redis', url: 'http://localhost:6379', method: 'HEAD' },
];

export async function GET() {
  // Check if we're in development mode without services
  const isDevelopment = process.env.NODE_ENV === 'development';
  const useMockData = process.env.USE_MOCK_SERVICES === 'true';
  
  if (isDevelopment || useMockData) {
    // Return mock data for development
    const mockResults = HEALTH_CHECKS.map(service => ({
      name: service.name,
      status: Math.random() > 0.2 ? 'online' : 'offline',
      responseTime: Math.floor(Math.random() * 100) + 10,
      statusCode: Math.random() > 0.2 ? 200 : 503,
    }));
    
    return NextResponse.json({ 
      services: mockResults, 
      timestamp: new Date().toISOString(),
      mock: true 
    });
  }
  
  // Real service checks
  const results = await Promise.all(
    HEALTH_CHECKS.map(async (service) => {
      const startTime = Date.now();
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000); // Reduced timeout
        
        const response = await fetch(service.url, {
          method: service.method,
          headers: service.headers || {},
          signal: controller.signal,
        });
        
        clearTimeout(timeout);
        const responseTime = Date.now() - startTime;
        
        return {
          name: service.name,
          status: response.ok ? 'online' : 'error',
          responseTime,
          statusCode: response.status,
        };
      } catch (error) {
        return {
          name: service.name,
          status: 'offline',
          responseTime: Date.now() - startTime,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    })
  );

  return NextResponse.json({ 
    services: results, 
    timestamp: new Date().toISOString(),
    mock: false 
  });
}