import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics for long-running monitoring
const errorRate = new Rate('errors');
const memoryLeakIndicator = new Trend('response_size_trend');
const connectionErrors = new Counter('connection_errors');
const timeoutErrors = new Counter('timeout_errors');

// Soak test configuration - sustained load over time
export const options = {
  stages: [
    { duration: '5m', target: 50 },    // Ramp up to baseline
    { duration: '60m', target: 50 },   // Sustain for 1 hour
    { duration: '30m', target: 80 },   // Increase load
    { duration: '60m', target: 80 },   // Sustain increased load
    { duration: '30m', target: 100 },  // Peak load
    { duration: '60m', target: 100 },  // Sustain peak
    { duration: '10m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],    // 95% under 2s
    http_req_failed: ['rate<0.05'],       // Less than 5% failure
    errors: ['rate<0.05'],
    connection_errors: ['count<100'],     // Less than 100 connection errors total
    timeout_errors: ['count<50'],         // Less than 50 timeouts total
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost';

// Realistic user journeys for long-term testing
const userJourneys = {
  // Casual browser - checks homepage and browses
  casualBrowser: () => {
    const journey = [
      { url: `${BASE_URL}:3001/`, name: 'homepage' },
      { url: `${BASE_URL}:8096/System/Info/Public`, name: 'jellyfin-info' },
      { url: `${BASE_URL}:5055/api/v1/status`, name: 'overseerr-status' },
    ];
    
    return executeJourney(journey, 'casual_browser');
  },
  
  // Media consumer - streams content
  mediaConsumer: () => {
    const journey = [
      { url: `${BASE_URL}:8096/Users`, name: 'jellyfin-users' },
      { url: `${BASE_URL}:8096/Items?Recursive=true&IncludeItemTypes=Movie`, name: 'jellyfin-movies' },
      { url: `${BASE_URL}:8096/Items?Recursive=true&IncludeItemTypes=Series`, name: 'jellyfin-series' },
    ];
    
    return executeJourney(journey, 'media_consumer');
  },
  
  // Content manager - manages downloads and library
  contentManager: () => {
    const apiKey = __ENV.SONARR_API_KEY || 'test-key';
    const journey = [
      { 
        url: `${BASE_URL}:8989/api/v3/system/status`, 
        name: 'sonarr-status',
        headers: { 'X-Api-Key': apiKey }
      },
      { 
        url: `${BASE_URL}:7878/api/v3/system/status`, 
        name: 'radarr-status',
        headers: { 'X-Api-Key': apiKey }
      },
      { 
        url: `${BASE_URL}:8989/api/v3/queue`, 
        name: 'sonarr-queue',
        headers: { 'X-Api-Key': apiKey }
      },
    ];
    
    return executeJourney(journey, 'content_manager');
  },
  
  // Administrator - monitors system
  administrator: () => {
    const auth = Buffer.from(`${__ENV.GRAFANA_USER || 'admin'}:${__ENV.GRAFANA_PASSWORD || 'admin'}`).toString('base64');
    const journey = [
      { url: `${BASE_URL}:3000/api/health`, name: 'grafana-health' },
      { 
        url: `${BASE_URL}:3000/api/org`, 
        name: 'grafana-org',
        headers: { 'Authorization': `Basic ${auth}` }
      },
      { url: `${BASE_URL}:9000/api/status`, name: 'portainer-status' },
    ];
    
    return executeJourney(journey, 'administrator');
  },
  
  // Background services simulation
  backgroundServices: () => {
    const journey = [
      { url: `${BASE_URL}:9090/api/v1/query?query=up`, name: 'prometheus-query' },
      { url: `${BASE_URL}:8181/api/v2?cmd=get_activity`, name: 'tautulli-activity' },
    ];
    
    return executeJourney(journey, 'background_services');
  },
};

// Execute a user journey
function executeJourney(journey, journeyType) {
  let journeySuccess = true;
  let totalResponseSize = 0;
  
  for (const step of journey) {
    const params = {
      headers: step.headers || {},
      timeout: '30s',
      tags: { 
        journey: journeyType,
        step: step.name 
      },
    };
    
    const response = http.get(step.url, params);
    
    // Track response size for memory leak detection
    if (response.body) {
      totalResponseSize += response.body.length;
      memoryLeakIndicator.add(response.body.length);
    }
    
    // Check for specific error types
    if (response.status === 0) {
      connectionErrors.add(1);
    } else if (response.timings.duration > 30000) {
      timeoutErrors.add(1);
    }
    
    const stepSuccess = check(response, {
      [`${step.name} status ok`]: (r) => r.status === 200 || r.status === 401,
      [`${step.name} responds in time`]: (r) => r.timings.duration < 10000,
      [`${step.name} has content`]: (r) => r.body && r.body.length > 0,
    });
    
    if (!stepSuccess) {
      journeySuccess = false;
      console.warn(`Journey ${journeyType} failed at step ${step.name}: ${response.status}`);
    }
    
    // Pause between steps
    sleep(1);
  }
  
  return journeySuccess;
}

// Memory usage simulation
function simulateMemoryUsage() {
  // Simulate operations that might cause memory leaks
  const operations = [
    () => {
      // Large library scan simulation
      const response = http.get(`${BASE_URL}:8096/Items?Recursive=true&Limit=1000`, {
        timeout: '30s',
        tags: { operation: 'library_scan' }
      });
      return response.status === 200;
    },
    () => {
      // Large search result simulation
      const response = http.get(`${BASE_URL}:5055/api/v1/search?query=marvel&limit=100`, {
        timeout: '30s',
        tags: { operation: 'large_search' }
      });
      return response.status === 200 || response.status === 401;
    },
  ];
  
  const operation = operations[Math.floor(Math.random() * operations.length)];
  return operation();
}

// Main test function
export default function () {
  const userId = __VU;
  const currentTime = new Date();
  
  // Distribute user types based on realistic ratios
  let journeyType;
  const userTypeRandom = Math.random();
  
  if (userTypeRandom < 0.4) {
    journeyType = 'casualBrowser';
  } else if (userTypeRandom < 0.7) {
    journeyType = 'mediaConsumer';
  } else if (userTypeRandom < 0.85) {
    journeyType = 'contentManager';
  } else if (userTypeRandom < 0.95) {
    journeyType = 'administrator';
  } else {
    journeyType = 'backgroundServices';
  }
  
  let success;
  
  // Execute journey or memory operation
  if (Math.random() < 0.9) {
    success = userJourneys[journeyType]();
  } else {
    success = simulateMemoryUsage();
  }
  
  errorRate.add(!success);
  
  // Realistic think time
  const thinkTime = Math.random() * 10 + 5; // 5-15 seconds
  sleep(thinkTime);
}

// Setup function
export function setup() {
  console.log('Starting soak test...');
  console.log('Duration: ~4 hours');
  console.log('This test will monitor for:');
  console.log('- Memory leaks');
  console.log('- Connection pool exhaustion');
  console.log('- Resource degradation over time');
  console.log('- Long-term stability');
  
  // Initial health check
  const healthEndpoints = [
    `${BASE_URL}:3001/`,
    `${BASE_URL}:8096/health`,
    `${BASE_URL}:3000/api/health`,
  ];
  
  for (const endpoint of healthEndpoints) {
    const response = http.get(endpoint, { timeout: '10s' });
    if (response.status !== 200) {
      console.warn(`Warning: ${endpoint} returned ${response.status}`);
    }
  }
  
  return { 
    startTime: new Date().toISOString(),
    testType: 'soak',
    expectedDuration: '4 hours'
  };
}

// Teardown function
export function teardown(data) {
  const endTime = new Date();
  const duration = (endTime - new Date(data.startTime)) / 1000 / 60; // minutes
  
  console.log(`Soak test completed after ${duration.toFixed(1)} minutes`);
  console.log('Post-test analysis:');
  console.log('1. Check system resource usage trends');
  console.log('2. Verify no memory leaks occurred');
  console.log('3. Check for any connection pool issues');
  console.log('4. Review error patterns over time');
  console.log('5. Validate system remained stable throughout');
}