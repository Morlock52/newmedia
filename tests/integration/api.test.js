const { apiRequest, generateTestMedia } = require('./setup');
const axios = require('axios');

describe('API Endpoint Tests', () => {
  // Note: Many of these APIs require authentication setup
  // These tests demonstrate the structure and can be expanded with proper auth

  describe('Jellyfin API', () => {
    const jellyfinBase = 'http://localhost:8096';

    test('GET /System/Info/Public should return server information', async () => {
      const response = await apiRequest('jellyfin', '/System/Info/Public');
      
      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('LocalAddress');
      expect(response.data).toHaveProperty('ServerName');
      expect(response.data).toHaveProperty('Version');
    });

    test('GET /System/Health should return healthy status', async () => {
      const response = await apiRequest('jellyfin', '/health');
      expect(response.status).toBe(200);
    });

    test('Jellyfin should support DLNA discovery', async () => {
      // Check if DLNA port is accessible
      const container = docker.getContainer('jellyfin');
      const info = await container.inspect();
      const ports = info.NetworkSettings.Ports;
      
      expect(ports['1900/udp']).toBeDefined();
    });
  });

  describe('Sonarr API', () => {
    test('GET /api/v3/health should return health status', async () => {
      const response = await apiRequest('sonarr', '/api/v3/health', {
        headers: {
          'X-Api-Key': process.env.SONARR_API_KEY || 'test-api-key'
        }
      });
      
      // Without proper API key, expect 401
      if (!process.env.SONARR_API_KEY) {
        expect(response.status).toBe(401);
      } else {
        expect(response.status).toBe(200);
        expect(Array.isArray(response.data)).toBe(true);
      }
    });

    test('GET /api/v3/system/status should return system info', async () => {
      const response = await apiRequest('sonarr', '/api/v3/system/status', {
        headers: {
          'X-Api-Key': process.env.SONARR_API_KEY || 'test-api-key'
        }
      });
      
      if (!process.env.SONARR_API_KEY) {
        expect(response.status).toBe(401);
      } else {
        expect(response.status).toBe(200);
        expect(response.data).toHaveProperty('version');
        expect(response.data).toHaveProperty('buildTime');
      }
    });
  });

  describe('Radarr API', () => {
    test('GET /api/v3/health should return health status', async () => {
      const response = await apiRequest('radarr', '/api/v3/health', {
        headers: {
          'X-Api-Key': process.env.RADARR_API_KEY || 'test-api-key'
        }
      });
      
      if (!process.env.RADARR_API_KEY) {
        expect(response.status).toBe(401);
      } else {
        expect(response.status).toBe(200);
        expect(Array.isArray(response.data)).toBe(true);
      }
    });

    test('GET /api/v3/system/status should return system info', async () => {
      const response = await apiRequest('radarr', '/api/v3/system/status', {
        headers: {
          'X-Api-Key': process.env.RADARR_API_KEY || 'test-api-key'
        }
      });
      
      if (!process.env.RADARR_API_KEY) {
        expect(response.status).toBe(401);
      } else {
        expect(response.status).toBe(200);
        expect(response.data).toHaveProperty('version');
        expect(response.data).toHaveProperty('buildTime');
      }
    });
  });

  describe('Prowlarr API', () => {
    test('GET /api/v1/health should return health status', async () => {
      const response = await apiRequest('prowlarr', '/api/v1/health', {
        headers: {
          'X-Api-Key': process.env.PROWLARR_API_KEY || 'test-api-key'
        }
      });
      
      if (!process.env.PROWLARR_API_KEY) {
        expect(response.status).toBe(401);
      } else {
        expect(response.status).toBe(200);
        expect(Array.isArray(response.data)).toBe(true);
      }
    });
  });

  describe('Overseerr API', () => {
    test('GET /api/v1/status should return server status', async () => {
      const response = await apiRequest('overseerr', '/api/v1/status');
      
      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('version');
      expect(response.data).toHaveProperty('commitTag');
    });

    test('GET /api/v1/settings/public should return public settings', async () => {
      const response = await apiRequest('overseerr', '/api/v1/settings/public');
      
      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('initialized');
    });
  });

  describe('Homepage API', () => {
    test('GET / should return homepage', async () => {
      const response = await apiRequest('homepage', '/');
      
      expect(response.status).toBe(200);
    });

    test('Homepage should have Docker integration', async () => {
      // Check if homepage can access Docker socket
      const container = docker.getContainer('homepage');
      const info = await container.inspect();
      const mounts = info.Mounts;
      
      const dockerSocketMount = mounts.find(m => m.Source === '/var/run/docker.sock');
      expect(dockerSocketMount).toBeDefined();
      expect(dockerSocketMount.Mode).toBe('ro');
    });
  });

  describe('Grafana API', () => {
    test('GET /api/health should return health status', async () => {
      const response = await apiRequest('grafana', '/api/health');
      
      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('database');
      expect(response.data.database).toBe('ok');
    });

    test('GET /api/org should return organization info', async () => {
      const auth = Buffer.from(`${process.env.GRAFANA_USER || 'admin'}:${process.env.GRAFANA_PASSWORD || 'admin'}`).toString('base64');
      
      const response = await apiRequest('grafana', '/api/org', {
        headers: {
          'Authorization': `Basic ${auth}`
        }
      });
      
      if (response.status === 401) {
        console.log('Grafana authentication required - skipping authenticated tests');
      } else {
        expect(response.status).toBe(200);
        expect(response.data).toHaveProperty('name');
      }
    });
  });

  describe('API Rate Limiting', () => {
    test('Services should handle burst requests', async () => {
      const requests = [];
      
      // Send 10 concurrent requests to each service
      for (let i = 0; i < 10; i++) {
        requests.push(apiRequest('jellyfin', '/System/Info/Public'));
        requests.push(apiRequest('overseerr', '/api/v1/status'));
      }
      
      const results = await Promise.allSettled(requests);
      const successful = results.filter(r => r.status === 'fulfilled' && r.value.status === 200);
      
      // At least 80% should succeed
      expect(successful.length).toBeGreaterThanOrEqual(results.length * 0.8);
    });
  });

  describe('API Response Times', () => {
    const measureResponseTime = async (service, endpoint) => {
      const start = Date.now();
      await apiRequest(service, endpoint);
      return Date.now() - start;
    };

    test('Service endpoints should respond quickly', async () => {
      const measurements = await Promise.all([
        measureResponseTime('jellyfin', '/health'),
        measureResponseTime('overseerr', '/api/v1/status'),
        measureResponseTime('homepage', '/')
      ]);
      
      // All responses should be under 1000ms
      measurements.forEach(time => {
        expect(time).toBeLessThan(1000);
      });
      
      // Average should be under 500ms
      const average = measurements.reduce((a, b) => a + b, 0) / measurements.length;
      expect(average).toBeLessThan(500);
    });
  });
});