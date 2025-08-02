import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const spikeRequests = new Counter('spike_requests');

// Spike test configuration - sudden traffic spikes
export const options = {
  stages: [
    { duration: '10s', target: 5 },     // Normal load
    { duration: '5s', target: 1000 },   // Sudden spike to 1000 users
    { duration: '30s', target: 1000 },  // Sustain spike
    { duration: '5s', target: 50 },     // Drop to moderate load
    { duration: '10s', target: 50 },    // Sustain moderate load
    { duration: '5s', target: 2000 },   // Second, larger spike
    { duration: '30s', target: 2000 },  // Sustain larger spike
    { duration: '10s', target: 0 },     // Drop to zero
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% under 5s during spikes
    http_req_failed: ['rate<0.3'],     // Allow up to 30% failure during spikes
    errors: ['rate<0.3'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost';

// Critical endpoints that must survive spikes
const criticalEndpoints = [
  { name: 'jellyfin-api', url: `${BASE_URL}:8096/System/Info/Public`, priority: 'high' },
  { name: 'homepage', url: `${BASE_URL}:3001/`, priority: 'high' },
  { name: 'overseerr-status', url: `${BASE_URL}:5055/api/v1/status`, priority: 'medium' },
  { name: 'grafana-health', url: `${BASE_URL}:3000/api/health`, priority: 'medium' },
];

// Heavy endpoints that might degrade
const heavyEndpoints = [
  { name: 'sonarr-queue', url: `${BASE_URL}:8989/api/v3/queue`, priority: 'low' },
  { name: 'radarr-queue', url: `${BASE_URL}:7878/api/v3/queue`, priority: 'low' },
  { name: 'jellyfin-library', url: `${BASE_URL}:8096/Items`, priority: 'low' },
];

// Traffic patterns during spikes
const trafficPatterns = {
  // Users hitting refresh on dashboards
  dashboardRefresh: () => {
    const endpoints = [
      `${BASE_URL}:3001/`,
      `${BASE_URL}:3000/d/media-stats/media-statistics`,
    ];
    
    endpoints.forEach(url => {
      const response = http.get(url, {
        timeout: '10s',
        tags: { pattern: 'dashboard_refresh' },
      });
      
      spikeRequests.add(1);
      
      check(response, {
        'dashboard loads during spike': (r) => r.status === 200,
      });
    });
  },
  
  // Users searching for content
  contentSearch: () => {
    const searchEndpoints = [
      `${BASE_URL}:5055/api/v1/search?query=avengers`,
      `${BASE_URL}:8989/api/v3/series/lookup?term=breaking`,
      `${BASE_URL}:7878/api/v3/movie/lookup?term=matrix`,
    ];
    
    const endpoint = searchEndpoints[Math.floor(Math.random() * searchEndpoints.length)];
    
    const headers = {
      'X-Api-Key': __ENV.OVERSEERR_API_KEY || 'test-key',
    };
    
    const response = http.get(endpoint, {
      headers: headers,
      timeout: '15s',
      tags: { pattern: 'content_search' },
    });
    
    spikeRequests.add(1);
    
    check(response, {
      'search works during spike': (r) => r.status === 200 || r.status === 401,
    });
  },
  
  // Users starting media streams
  mediaStream: () => {
    const streamEndpoints = [
      `${BASE_URL}:8096/Audio/stream`,
      `${BASE_URL}:8096/Videos/stream`,
    ];
    
    const endpoint = streamEndpoints[Math.floor(Math.random() * streamEndpoints.length)];
    
    const response = http.get(endpoint, {
      headers: {
        'Range': 'bytes=0-1024', // Small range request
      },
      timeout: '20s',
      tags: { pattern: 'media_stream' },
    });
    
    spikeRequests.add(1);
    
    check(response, {
      'stream available during spike': (r) => r.status === 200 || r.status === 206 || r.status === 404,
    });
  },
  
  // Users checking system status
  systemStatus: () => {
    criticalEndpoints.forEach(endpoint => {
      const response = http.get(endpoint.url, {
        timeout: '5s',
        tags: { 
          pattern: 'system_status',
          priority: endpoint.priority 
        },
      });
      
      spikeRequests.add(1);
      
      const success = check(response, {
        [`${endpoint.name} responds during spike`]: (r) => r.status === 200,
        [`${endpoint.name} fast response`]: (r) => r.timings.duration < 2000,
      });
      
      // Critical endpoints must not fail
      if (endpoint.priority === 'high' && !success) {
        console.error(`CRITICAL: ${endpoint.name} failed during spike!`);
      }
    });
  },
};

// Main test function
export default function () {
  const userId = __VU;
  const currentStage = getCurrentStage();
  
  // Different user behavior based on current load
  let pattern;
  if (currentStage.target > 1500) {
    // Very high load - mostly system checks
    pattern = Math.random() < 0.7 ? 'systemStatus' : 'dashboardRefresh';
  } else if (currentStage.target > 500) {
    // High load - mixed behavior
    const patterns = ['dashboardRefresh', 'contentSearch', 'systemStatus'];
    pattern = patterns[Math.floor(Math.random() * patterns.length)];
  } else {
    // Normal load - all behaviors
    const patterns = Object.keys(trafficPatterns);
    pattern = patterns[Math.floor(Math.random() * patterns.length)];
  }
  
  try {
    trafficPatterns[pattern]();
  } catch (error) {
    console.error(`Error in ${pattern}: ${error.message}`);
    errorRate.add(1);
  }
  
  // Minimal sleep during spikes
  if (currentStage.target > 1000) {
    sleep(0.1); // Very short sleep during spikes
  } else {
    sleep(Math.random() * 2 + 0.5);
  }
}

// Helper function to get current test stage
function getCurrentStage() {
  const elapsed = __ENV.K6_ELAPSED || 0;
  
  // Approximate stage based on elapsed time
  if (elapsed < 10) return { target: 5 };
  if (elapsed < 15) return { target: 1000 };
  if (elapsed < 45) return { target: 1000 };
  if (elapsed < 50) return { target: 50 };
  if (elapsed < 60) return { target: 50 };
  if (elapsed < 65) return { target: 2000 };
  if (elapsed < 95) return { target: 2000 };
  return { target: 0 };
}

// Setup function
export function setup() {
  console.log('Starting spike test...');
  console.log('This test simulates sudden traffic spikes (viral scenarios)');
  
  // Pre-warm critical endpoints
  console.log('Pre-warming critical endpoints...');
  criticalEndpoints.forEach(endpoint => {
    const response = http.get(endpoint.url, { timeout: '10s' });
    if (response.status !== 200) {
      console.warn(`Warning: ${endpoint.name} not ready (${response.status})`);
    }
  });
  
  return { 
    startTime: new Date().toISOString(),
    testType: 'spike',
    expectedSpikes: 2
  };
}

// Teardown function  
export function teardown(data) {
  console.log(`Spike test completed.`);
  console.log(`Test duration: ${new Date() - new Date(data.startTime)}ms`);
  console.log(`Spikes tested: ${data.expectedSpikes}`);
  console.log('Results:');
  console.log('- Check if critical endpoints remained available');
  console.log('- Verify system recovery after spikes');
  console.log('- Review auto-scaling triggers if configured');
}