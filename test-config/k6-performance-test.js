import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// Custom metrics
export let errorRate = new Rate('errors');
export let responseTime = new Trend('response_time');
export let requestCount = new Counter('requests');

// Test configuration
export let options = {
  stages: [
    { duration: '30s', target: 5 },   // Ramp up
    { duration: '2m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp to 20 users
    { duration: '2m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests must complete below 5s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate below 10%
  },
  ext: {
    influxdb: {
      enabled: true,
      addr: 'http://localhost:8086',
      db: 'k6',
      tags: {
        environment: 'test',
        test_type: 'load',
      },
    },
  },
};

// Base configuration
const baseUrl = __ENV.BASE_URL || 'http://localhost';
const services = {
  jellyfin: `${baseUrl}:8096`,
  plex: `${baseUrl}:32400`,
  emby: `${baseUrl}:8097`,
  sonarr: `${baseUrl}:8989`,
  radarr: `${baseUrl}:7878`,
  lidarr: `${baseUrl}:8686`,
  prowlarr: `${baseUrl}:9696`,
  bazarr: `${baseUrl}:6767`,
  jellyseerr: `${baseUrl}:5055`,
  overseerr: `${baseUrl}:5056`,
  ombi: `${baseUrl}:3579`,
  qbittorrent: `${baseUrl}:8080`,
  sabnzbd: `${baseUrl}:8081`,
  prometheus: `${baseUrl}:9090`,
  grafana: `${baseUrl}:3000`,
  uptime_kuma: `${baseUrl}:3001`,
  homepage: `${baseUrl}:3003`,
  homarr: `${baseUrl}:7575`,
};

// API keys (from environment)
const apiKeys = {
  sonarr: __ENV.SONARR_API_KEY || '',
  radarr: __ENV.RADARR_API_KEY || '',
  lidarr: __ENV.LIDARR_API_KEY || '',
  prowlarr: __ENV.PROWLARR_API_KEY || '',
  bazarr: __ENV.BAZARR_API_KEY || '',
  jellyseerr: __ENV.JELLYSEERR_API_KEY || '',
  overseerr: __ENV.OVERSEERR_API_KEY || '',
};

// Helper function to make authenticated requests
function makeAuthenticatedRequest(url, service, endpoint = '/') {
  let headers = {
    'Accept': 'application/json',
    'User-Agent': 'k6-performance-test/1.0',
  };
  
  // Add authentication headers based on service
  if (apiKeys[service]) {
    if (service === 'jellyfin') {
      headers['X-Emby-Authorization'] = `MediaBrowser Token="${apiKeys[service]}"`;
    } else if (service === 'plex') {
      headers['X-Plex-Token'] = apiKeys[service];
    } else {
      headers['X-Api-Key'] = apiKeys[service];
    }
  }
  
  const fullUrl = `${url}${endpoint}`;
  const response = http.get(fullUrl, { headers, timeout: '30s' });
  
  requestCount.add(1);
  responseTime.add(response.timings.duration);
  
  return response;
}

// Test scenarios
export default function () {
  // Media Server Tests
  group('Media Servers', function () {
    // Jellyfin health check
    let jellyfinResponse = makeAuthenticatedRequest(services.jellyfin, 'jellyfin', '/health');
    check(jellyfinResponse, {
      'Jellyfin health check status is 200': (r) => r.status === 200,
      'Jellyfin response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
    
    // Plex identity check
    let plexResponse = makeAuthenticatedRequest(services.plex, 'plex', '/identity');
    check(plexResponse, {
      'Plex identity status is 200': (r) => r.status === 200,
      'Plex response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
    
    // Emby health check
    let embyResponse = makeAuthenticatedRequest(services.emby, 'emby', '/health');
    check(embyResponse, {
      'Emby health check accessible': (r) => r.status >= 200 && r.status < 400,
      'Emby response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
  });
  
  // ARR Services Tests
  group('ARR Services', function () {
    // Sonarr system status
    let sonarrResponse = makeAuthenticatedRequest(services.sonarr, 'sonarr', '/api/v3/system/status');
    check(sonarrResponse, {
      'Sonarr API status is 200': (r) => r.status === 200,
      'Sonarr response time < 2000ms': (r) => r.timings.duration < 2000,
      'Sonarr returns JSON': (r) => r.headers['Content-Type'] && r.headers['Content-Type'].includes('application/json'),
    }) || errorRate.add(1);
    
    // Radarr system status
    let radarrResponse = makeAuthenticatedRequest(services.radarr, 'radarr', '/api/v3/system/status');
    check(radarrResponse, {
      'Radarr API status is 200': (r) => r.status === 200,
      'Radarr response time < 2000ms': (r) => r.timings.duration < 2000,
      'Radarr returns JSON': (r) => r.headers['Content-Type'] && r.headers['Content-Type'].includes('application/json'),
    }) || errorRate.add(1);
    
    // Lidarr system status
    let lidarrResponse = makeAuthenticatedRequest(services.lidarr, 'lidarr', '/api/v1/system/status');
    check(lidarrResponse, {
      'Lidarr API status is 200': (r) => r.status === 200,
      'Lidarr response time < 2000ms': (r) => r.timings.duration < 2000,
    }) || errorRate.add(1);
    
    // Prowlarr system status
    let prowlarrResponse = makeAuthenticatedRequest(services.prowlarr, 'prowlarr', '/api/v1/system/status');
    check(prowlarrResponse, {
      'Prowlarr API status is 200': (r) => r.status === 200,
      'Prowlarr response time < 2000ms': (r) => r.timings.duration < 2000,
    }) || errorRate.add(1);
  });
  
  // Request Services Tests
  group('Request Services', function () {
    // Jellyseerr status
    let jellyseerrResponse = makeAuthenticatedRequest(services.jellyseerr, 'jellyseerr', '/api/v1/status');
    check(jellyseerrResponse, {
      'Jellyseerr status accessible': (r) => r.status >= 200 && r.status < 400,
      'Jellyseerr response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
    
    // Overseerr status
    let overseerrResponse = makeAuthenticatedRequest(services.overseerr, 'overseerr', '/api/v1/status');
    check(overseerrResponse, {
      'Overseerr status accessible': (r) => r.status >= 200 && r.status < 400,
      'Overseerr response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
  });
  
  // Download Clients Tests
  group('Download Clients', function () {
    // qBittorrent version check
    let qbtResponse = http.get(`${services.qbittorrent}/api/v2/app/version`, { timeout: '10s' });
    check(qbtResponse, {
      'qBittorrent API accessible': (r) => r.status >= 200 && r.status < 500,
      'qBittorrent response time < 5000ms': (r) => r.timings.duration < 5000,
    }) || errorRate.add(1);
    
    // SABnzbd version check
    let sabResponse = http.get(`${services.sabnzbd}/sabnzbd/api?mode=version&output=json`, { timeout: '10s' });
    check(sabResponse, {
      'SABnzbd API accessible': (r) => r.status >= 200 && r.status < 500,
      'SABnzbd response time < 5000ms': (r) => r.timings.duration < 5000,
    }) || errorRate.add(1);
  });
  
  // Monitoring Services Tests
  group('Monitoring Services', function () {
    // Prometheus health
    let prometheusResponse = http.get(`${services.prometheus}/-/healthy`, { timeout: '10s' });
    check(prometheusResponse, {
      'Prometheus health check is 200': (r) => r.status === 200,
      'Prometheus response time < 2000ms': (r) => r.timings.duration < 2000,
    }) || errorRate.add(1);
    
    // Grafana health
    let grafanaResponse = http.get(`${services.grafana}/api/health`, { timeout: '10s' });
    check(grafanaResponse, {
      'Grafana health check accessible': (r) => r.status >= 200 && r.status < 500,
      'Grafana response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
    
    // Uptime Kuma
    let uptimeResponse = http.get(`${services.uptime_kuma}/`, { timeout: '10s' });
    check(uptimeResponse, {
      'Uptime Kuma accessible': (r) => r.status >= 200 && r.status < 500,
      'Uptime Kuma response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
  });
  
  // Dashboard Tests
  group('Dashboards', function () {
    // Homepage dashboard
    let homepageResponse = http.get(`${services.homepage}/`, { timeout: '10s' });
    check(homepageResponse, {
      'Homepage accessible': (r) => r.status >= 200 && r.status < 500,
      'Homepage response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
    
    // Homarr dashboard
    let homarrResponse = http.get(`${services.homarr}/`, { timeout: '10s' });
    check(homarrResponse, {
      'Homarr accessible': (r) => r.status >= 200 && r.status < 500,
      'Homarr response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
  });
  
  // Random delay between iterations (1-3 seconds)
  sleep(Math.random() * 2 + 1);
}

// Setup function (runs once at the beginning)
export function setup() {
  console.log('🚀 Starting Ultimate Media Server 2025 Performance Tests');
  console.log(`Base URL: ${baseUrl}`);
  console.log(`Virtual Users: ${options.stages[1].target}`);
  console.log(`Test Duration: ~${options.stages.reduce((sum, stage) => sum + parseInt(stage.duration), 0)}s`);
  
  // Warm up requests
  let warmupServices = [
    `${services.prometheus}/-/healthy`,
    `${services.grafana}/api/health`,
    `${services.jellyfin}/health`,
  ];
  
  console.log('Warming up services...');
  warmupServices.forEach(url => {
    http.get(url, { timeout: '10s' });
  });
  
  return { timestamp: new Date().toISOString() };
}

// Teardown function (runs once at the end)
export function teardown(data) {
  console.log(`✅ Performance tests completed at ${new Date().toISOString()}`);
  console.log(`Started at: ${data.timestamp}`);
}

// Custom summary report
export function handleSummary(data) {
  return {
    'test-results/performance-summary.html': htmlReport(data),
    'test-results/performance-summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

// Stress test scenario (can be run separately)
export let stressOptions = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up
    { duration: '2m', target: 50 },   // Stay at 50 users
    { duration: '1m', target: 100 },  // Ramp to 100 users
    { duration: '3m', target: 100 },  // Stay at 100 users
    { duration: '1m', target: 200 },  // Spike to 200 users
    { duration: '2m', target: 200 },  // Stay at 200 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<10000'], // 95% of requests must complete below 10s
    http_req_failed: ['rate<0.2'],      // Error rate must be below 20%
  },
};

// Smoke test scenario (quick validation)
export let smokeOptions = {
  vus: 1,
  duration: '30s',
  thresholds: {
    http_req_duration: ['p(95)<3000'],
    http_req_failed: ['rate<0.05'],
  },
};