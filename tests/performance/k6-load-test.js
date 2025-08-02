import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics for media server performance testing
const errorCount = new Counter('errors');
const successRate = new Rate('success_rate');
const apiResponseTime = new Trend('api_response_time');

// Test configuration based on 2025 research findings
export const options = {
  stages: [
    { duration: '2m', target: 5 },   // Ramp up to 5 users
    { duration: '5m', target: 10 },  // Stay at 10 users
    { duration: '3m', target: 20 },  // Ramp up to 20 users
    { duration: '5m', target: 20 },  // Stay at 20 users
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    // Performance thresholds based on 2025 best practices
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete within 2s
    http_req_failed: ['rate<0.05'],    // Error rate must be less than 5%
    success_rate: ['rate>0.95'],       // Success rate must be above 95%
  },
};

// Service endpoints configuration
const services = {
  jellyfin: {
    baseUrl: 'http://localhost:8096',
    endpoints: ['/health', '/System/Info', '/Library/VirtualFolders'],
    requiresAuth: false
  },
  plex: {
    baseUrl: 'http://localhost:32400',
    endpoints: ['/identity', '/'],
    requiresAuth: false
  },
  sonarr: {
    baseUrl: 'http://localhost:8989',
    endpoints: ['/ping', '/api/v3/system/status'],
    requiresAuth: true,
    apiKey: __ENV.SONARR_API_KEY || ''
  },
  radarr: {
    baseUrl: 'http://localhost:7878',
    endpoints: ['/ping', '/api/v3/system/status'],
    requiresAuth: true,
    apiKey: __ENV.RADARR_API_KEY || ''
  },
  prowlarr: {
    baseUrl: 'http://localhost:9696',
    endpoints: ['/ping', '/api/v1/system/status'],
    requiresAuth: true,
    apiKey: __ENV.PROWLARR_API_KEY || ''
  },
  jellyseerr: {
    baseUrl: 'http://localhost:5055',
    endpoints: ['/api/v1/status', '/api/v1/discover/movies'],
    requiresAuth: false
  },
  prometheus: {
    baseUrl: 'http://localhost:9090',
    endpoints: ['/-/healthy', '/api/v1/targets'],
    requiresAuth: false
  },
  grafana: {
    baseUrl: 'http://localhost:3000',
    endpoints: ['/api/health', '/api/datasources'],
    requiresAuth: false
  }
};

// Test scenarios for different service types
export function testMediaServers() {
  const mediaServices = ['jellyfin', 'plex'];
  
  mediaServices.forEach(serviceName => {
    const service = services[serviceName];
    if (!service) return;

    service.endpoints.forEach(endpoint => {
      const url = `${service.baseUrl}${endpoint}`;
      const params = {
        timeout: '10s',
        headers: {}
      };

      // Add API key if required
      if (service.requiresAuth && service.apiKey) {
        params.headers['X-Api-Key'] = service.apiKey;
      }

      const response = http.get(url, params);
      
      // Performance tracking
      apiResponseTime.add(response.timings.duration);
      
      // Success/failure tracking
      const success = check(response, {
        [`${serviceName} ${endpoint} status is 200`]: (r) => r.status === 200,
        [`${serviceName} ${endpoint} response time < 5s`]: (r) => r.timings.duration < 5000,
      });

      successRate.add(success);
      if (!success) {
        errorCount.add(1);
        console.log(`${serviceName} ${endpoint} failed: ${response.status}`);
      }
    });
  });
}

export function testArrServices() {
  const arrServices = ['sonarr', 'radarr', 'prowlarr'];
  
  arrServices.forEach(serviceName => {
    const service = services[serviceName];
    if (!service) return;

    service.endpoints.forEach(endpoint => {
      const url = `${service.baseUrl}${endpoint}`;
      const params = {
        timeout: '10s',
        headers: {}
      };

      // Add API key for authenticated endpoints
      if (service.requiresAuth && service.apiKey) {
        params.headers['X-Api-Key'] = service.apiKey;
      }

      const response = http.get(url, params);
      
      apiResponseTime.add(response.timings.duration);
      
      const success = check(response, {
        [`${serviceName} ${endpoint} status is 200`]: (r) => r.status === 200,
        [`${serviceName} ${endpoint} response time < 3s`]: (r) => r.timings.duration < 3000,
      });

      successRate.add(success);
      if (!success) {
        errorCount.add(1);
      }
    });
  });
}

export function testRequestServices() {
  const requestServices = ['jellyseerr'];
  
  requestServices.forEach(serviceName => {
    const service = services[serviceName];
    if (!service) return;

    service.endpoints.forEach(endpoint => {
      const url = `${service.baseUrl}${endpoint}`;
      const response = http.get(url, { timeout: '10s' });
      
      apiResponseTime.add(response.timings.duration);
      
      const success = check(response, {
        [`${serviceName} ${endpoint} status is 200`]: (r) => r.status === 200,
        [`${serviceName} ${endpoint} response time < 2s`]: (r) => r.timings.duration < 2000,
      });

      successRate.add(success);
      if (!success) {
        errorCount.add(1);
      }
    });
  });
}

export function testMonitoringServices() {
  const monitoringServices = ['prometheus', 'grafana'];
  
  monitoringServices.forEach(serviceName => {
    const service = services[serviceName];
    if (!service) return;

    service.endpoints.forEach(endpoint => {
      const url = `${service.baseUrl}${endpoint}`;
      const response = http.get(url, { timeout: '10s' });
      
      apiResponseTime.add(response.timings.duration);
      
      const success = check(response, {
        [`${serviceName} ${endpoint} status is 200`]: (r) => r.status === 200,
        [`${serviceName} ${endpoint} response time < 2s`]: (r) => r.timings.duration < 2000,
      });

      successRate.add(success);
      if (!success) {
        errorCount.add(1);
      }
    });
  });
}

// Stress test for high concurrent media requests
export function stressTestMediaStreaming() {
  // Simulate multiple users accessing Jellyfin simultaneously
  const jellyfinUrls = [
    'http://localhost:8096/health',
    'http://localhost:8096/System/Info',
    'http://localhost:8096/Library/VirtualFolders'
  ];

  jellyfinUrls.forEach(url => {
    const response = http.get(url, { timeout: '15s' });
    
    const success = check(response, {
      'Jellyfin stress test status is 200': (r) => r.status === 200,
      'Jellyfin stress test response time < 10s': (r) => r.timings.duration < 10000,
    });

    successRate.add(success);
    if (!success) {
      errorCount.add(1);
    }
  });
}

// Main test function
export default function() {
  // Test different service categories with different loads
  switch (__ITER % 4) {
    case 0:
      testMediaServers();
      break;
    case 1:
      testArrServices();
      break;
    case 2:
      testRequestServices();
      break;
    case 3:
      testMonitoringServices();
      break;
  }

  // Add stress testing for high VU counts
  if (__VU > 15) {
    stressTestMediaStreaming();
  }

  // Random sleep between 1-3 seconds to simulate real user behavior
  sleep(Math.random() * 2 + 1);
}

// Setup function for test initialization
export function setup() {
  console.log('🚀 Starting Media Server Performance Test Suite - 2025');
  console.log('Configuration:');
  console.log(`- Target services: ${Object.keys(services).length}`);
  console.log(`- Test duration: ~17 minutes`);
  console.log(`- Max concurrent users: 20`);
  
  // Health check before starting load test
  const healthCheck = http.get('http://localhost:8096/health');
  if (healthCheck.status !== 200) {
    console.error('❌ Jellyfin health check failed. Ensure services are running.');
    return null;
  }
  
  console.log('✅ Pre-test health check passed');
  return { startTime: Date.now() };
}

// Teardown function for test cleanup
export function teardown(data) {
  if (data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`\n📊 Performance test completed in ${duration}s`);
  }
  
  console.log('\n💡 Performance Recommendations:');
  console.log('- Monitor response times for services > 2000ms');
  console.log('- Check error logs for failed requests');
  console.log('- Consider scaling resources for high-load scenarios');
  console.log('- Implement caching for frequently accessed endpoints');
}