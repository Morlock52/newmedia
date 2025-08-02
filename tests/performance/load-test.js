import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const successRate = new Rate('success');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 50 },   // Ramp up to 50 users
    { duration: '10m', target: 100 }, // Stay at 100 users
    { duration: '5m', target: 50 },   // Ramp down to 50 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% of requests must complete below 1s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate below 10%
    success: ['rate>0.9'],             // Success rate above 90%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost';

// Service endpoints to test
const endpoints = [
  { name: 'jellyfin-home', url: `${BASE_URL}:8096/`, weight: 3 },
  { name: 'jellyfin-health', url: `${BASE_URL}:8096/health`, weight: 5 },
  { name: 'sonarr-health', url: `${BASE_URL}:8989/api/v3/health`, weight: 2 },
  { name: 'radarr-health', url: `${BASE_URL}:7878/api/v3/health`, weight: 2 },
  { name: 'prowlarr-health', url: `${BASE_URL}:9696/api/v1/health`, weight: 1 },
  { name: 'overseerr-status', url: `${BASE_URL}:5055/api/v1/status`, weight: 3 },
  { name: 'homepage', url: `${BASE_URL}:3001/`, weight: 4 },
  { name: 'grafana-health', url: `${BASE_URL}:3000/api/health`, weight: 2 },
];

// Calculate total weight for weighted random selection
const totalWeight = endpoints.reduce((sum, endpoint) => sum + endpoint.weight, 0);

// Helper function to select endpoint based on weight
function selectEndpoint() {
  let random = Math.random() * totalWeight;
  
  for (const endpoint of endpoints) {
    random -= endpoint.weight;
    if (random <= 0) {
      return endpoint;
    }
  }
  
  return endpoints[0]; // Fallback
}

// Helper function to get headers for different services
function getHeaders(endpointName) {
  const headers = {
    'User-Agent': 'k6-load-test/1.0',
  };
  
  // Add API keys for services that require them
  if (endpointName.includes('sonarr') && __ENV.SONARR_API_KEY) {
    headers['X-Api-Key'] = __ENV.SONARR_API_KEY;
  } else if (endpointName.includes('radarr') && __ENV.RADARR_API_KEY) {
    headers['X-Api-Key'] = __ENV.RADARR_API_KEY;
  } else if (endpointName.includes('prowlarr') && __ENV.PROWLARR_API_KEY) {
    headers['X-Api-Key'] = __ENV.PROWLARR_API_KEY;
  }
  
  return headers;
}

// Main test scenario
export default function () {
  const endpoint = selectEndpoint();
  const headers = getHeaders(endpoint.name);
  
  const params = {
    headers: headers,
    timeout: '10s',
    tags: { 
      endpoint: endpoint.name,
      service: endpoint.name.split('-')[0]
    },
  };
  
  // Make the request
  const response = http.get(endpoint.url, params);
  
  // Check response
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 1000ms': (r) => r.timings.duration < 1000,
    'response time < 500ms': (r) => r.timings.duration < 500,
    'response has body': (r) => r.body && r.body.length > 0,
  });
  
  // Update custom metrics
  errorRate.add(!success);
  successRate.add(success);
  
  // Log errors for debugging
  if (!success) {
    console.error(`Failed request to ${endpoint.name}: Status ${response.status}, Duration ${response.timings.duration}ms`);
  }
  
  // Simulate user think time
  sleep(Math.random() * 3 + 1); // Random sleep between 1-4 seconds
}

// Optional: Setup function (runs once)
export function setup() {
  console.log('Starting load test...');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Total endpoints: ${endpoints.length}`);
  
  // Verify at least one endpoint is accessible
  const testEndpoint = endpoints.find(e => e.name === 'homepage');
  const response = http.get(testEndpoint.url, { timeout: '10s' });
  
  if (response.status !== 200) {
    throw new Error(`Setup failed: Homepage not accessible (status ${response.status})`);
  }
  
  return { startTime: new Date().toISOString() };
}

// Optional: Teardown function (runs once)
export function teardown(data) {
  console.log(`Load test completed. Started at: ${data.startTime}`);
}