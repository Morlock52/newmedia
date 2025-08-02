import http from 'k6/http';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const mediaStreamDuration = new Trend('media_stream_duration');
const apiSearchDuration = new Trend('api_search_duration');

// Stress test configuration - push system to its limits
export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 300 },   // Ramp up to 300 users
    { duration: '10m', target: 500 },  // Ramp up to 500 users
    { duration: '5m', target: 1000 },  // Push to 1000 users
    { duration: '10m', target: 1000 }, // Stay at 1000 users
    { duration: '5m', target: 0 },     // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<3000', 'p(99)<5000'], // More relaxed for stress test
    http_req_failed: ['rate<0.2'],                    // Allow up to 20% failure
    errors: ['rate<0.2'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost';

// Simulated media files for streaming tests
const mediaFiles = [
  '/media/movies/sample-1080p.mp4',
  '/media/movies/sample-720p.mp4',
  '/media/tv/sample-episode.mkv',
  '/media/music/sample-album/track01.mp3',
];

// User scenarios
const scenarios = {
  // Simulate media streaming
  streamMedia: (userId) => {
    const mediaFile = mediaFiles[userId % mediaFiles.length];
    const url = `${BASE_URL}:8096/Items/Download?path=${encodeURIComponent(mediaFile)}`;
    
    const params = {
      headers: {
        'Range': 'bytes=0-1048576', // Request first 1MB
        'User-Agent': `k6-stress-test/user-${userId}`,
      },
      timeout: '30s',
      tags: { scenario: 'stream_media' },
    };
    
    const start = new Date();
    const response = http.get(url, params);
    const duration = new Date() - start;
    
    mediaStreamDuration.add(duration);
    
    return check(response, {
      'streaming status ok': (r) => r.status === 200 || r.status === 206,
      'has content': (r) => r.body && r.body.length > 0,
    });
  },
  
  // Simulate API searches
  searchContent: (userId) => {
    const searchTerms = ['action', 'comedy', 'drama', 'thriller', 'sci-fi'];
    const searchTerm = searchTerms[userId % searchTerms.length];
    
    const services = [
      { name: 'sonarr', port: 8989, endpoint: `/api/v3/series/lookup?term=${searchTerm}` },
      { name: 'radarr', port: 7878, endpoint: `/api/v3/movie/lookup?term=${searchTerm}` },
    ];
    
    const service = services[userId % services.length];
    const url = `${BASE_URL}:${service.port}${service.endpoint}`;
    
    const params = {
      headers: {
        'X-Api-Key': __ENV[`${service.name.toUpperCase()}_API_KEY`] || 'test-key',
      },
      timeout: '10s',
      tags: { scenario: 'search_content', service: service.name },
    };
    
    const start = new Date();
    const response = http.get(url, params);
    const duration = new Date() - start;
    
    apiSearchDuration.add(duration);
    
    return check(response, {
      'search successful': (r) => r.status === 200,
      'has results': (r) => {
        try {
          const data = JSON.parse(r.body);
          return Array.isArray(data) && data.length > 0;
        } catch {
          return false;
        }
      },
    });
  },
  
  // Simulate heavy dashboard usage
  browseDashboard: (userId) => {
    const pages = [
      `${BASE_URL}:3001/`,
      `${BASE_URL}:3001/services`,
      `${BASE_URL}:3001/bookmarks`,
      `${BASE_URL}:3000/d/media-stats/media-statistics`,
    ];
    
    const results = [];
    
    for (const page of pages) {
      const response = http.get(page, {
        timeout: '15s',
        tags: { scenario: 'browse_dashboard' },
      });
      
      results.push(check(response, {
        'page loads': (r) => r.status === 200,
        'page has content': (r) => r.body && r.body.length > 1000,
      }));
    }
    
    return results.every(r => r);
  },
  
  // Simulate concurrent downloads
  downloadContent: (userId) => {
    const url = `${BASE_URL}:8081/api?mode=queue&output=json`;
    
    const params = {
      headers: {
        'X-Api-Key': __ENV.SABNZBD_API_KEY || 'test-key',
      },
      timeout: '20s',
      tags: { scenario: 'download_content' },
    };
    
    const response = http.get(url, params);
    
    return check(response, {
      'download queue accessible': (r) => r.status === 200,
    });
  },
};

// Main test function
export default function () {
  const userId = __VU; // Virtual User ID
  const scenarioNames = Object.keys(scenarios);
  
  // Each user focuses on one scenario type based on their ID
  const scenarioName = scenarioNames[userId % scenarioNames.length];
  const scenario = scenarios[scenarioName];
  
  const success = scenario(userId);
  errorRate.add(!success);
  
  // Add some randomness to prevent thundering herd
  const thinkTime = Math.random() * 2 + 0.5; // 0.5-2.5 seconds
  sleep(thinkTime);
}

// Setup function
export function setup() {
  console.log('Starting stress test...');
  console.log('This test will push the system to its limits!');
  
  // Check system is ready
  const healthChecks = [
    { name: 'Jellyfin', url: `${BASE_URL}:8096/health` },
    { name: 'Homepage', url: `${BASE_URL}:3001/` },
    { name: 'Grafana', url: `${BASE_URL}:3000/api/health` },
  ];
  
  for (const check of healthChecks) {
    const response = http.get(check.url, { timeout: '10s' });
    if (response.status !== 200) {
      console.error(`${check.name} is not ready (status ${response.status})`);
      throw new Error(`Pre-test check failed for ${check.name}`);
    }
  }
  
  return { 
    startTime: new Date().toISOString(),
    maxUsers: 1000
  };
}

// Teardown function
export function teardown(data) {
  console.log(`Stress test completed.`);
  console.log(`Started at: ${data.startTime}`);
  console.log(`Maximum concurrent users: ${data.maxUsers}`);
  console.log('Check Grafana dashboards for detailed metrics.');
}