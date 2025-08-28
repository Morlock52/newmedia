#!/usr/bin/env node

/**
 * Ultimate Media Server 2025 - Swarm Stress Test Suite
 * Advanced stress testing with Serena coordination
 */

const http = require('http');
const https = require('https');
const { performance } = require('perf_hooks');
const cluster = require('cluster');
const os = require('os');

// Configuration
const CONFIG = {
  target: 'http://localhost:3333',
  phases: [
    { duration: 10, arrivalRate: 10, name: 'Warm-up' },
    { duration: 30, arrivalRate: 50, name: 'Normal Load' },
    { duration: 30, arrivalRate: 100, name: 'High Load' },
    { duration: 20, arrivalRate: 200, name: 'Stress Test' },
    { duration: 10, arrivalRate: 500, name: 'Breaking Point' }
  ],
  endpoints: [
    '/',
    '/health',
    '/api/analytics',
    '/api/downloads',
    '/api/media',
    '/api/recommendations',
    '/api/voice',
    '/api/webxr',
    '/api/auth',
    '/api/player',
    '/api/assistant',
    '/api/theme',
    '/api/watchparty',
    '/api/predictions',
    '/api/monitoring',
    '/api/visualization',
    '/api/social',
    '/api/neural'
  ],
  workers: os.cpus().length
};

// Metrics storage
const metrics = {
  requests: 0,
  successes: 0,
  failures: 0,
  responseTimes: [],
  errors: [],
  statusCodes: {},
  endpointMetrics: {},
  phaseMetrics: {}
};

// Color codes for output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

/**
 * Make HTTP request and measure performance
 */
async function makeRequest(endpoint) {
  return new Promise((resolve) => {
    const startTime = performance.now();
    const url = `${CONFIG.target}${endpoint}`;
    
    const req = http.get(url, (res) => {
      let data = '';
      
      res.on('data', chunk => {
        data += chunk;
      });
      
      res.on('end', () => {
        const endTime = performance.now();
        const responseTime = endTime - startTime;
        
        resolve({
          success: res.statusCode === 200,
          statusCode: res.statusCode,
          responseTime,
          endpoint,
          dataSize: data.length
        });
      });
    });
    
    req.on('error', (err) => {
      const endTime = performance.now();
      resolve({
        success: false,
        statusCode: 0,
        responseTime: endTime - startTime,
        endpoint,
        error: err.message
      });
    });
    
    req.setTimeout(5000, () => {
      req.destroy();
      resolve({
        success: false,
        statusCode: 0,
        responseTime: 5000,
        endpoint,
        error: 'Timeout'
      });
    });
  });
}

/**
 * Worker process for distributed load generation
 */
function runWorker() {
  process.on('message', async (msg) => {
    if (msg.type === 'test') {
      const result = await makeRequest(msg.endpoint);
      process.send({ type: 'result', result });
    }
  });
}

/**
 * Master process orchestration
 */
async function runMaster() {
  console.log(`${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}`);
  console.log(`${colors.bright}🚀 ULTIMATE MEDIA SERVER 2025 - SWARM STRESS TEST${colors.reset}`);
  console.log(`${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}`);
  console.log(`Target: ${CONFIG.target}`);
  console.log(`Workers: ${CONFIG.workers}`);
  console.log(`Endpoints: ${CONFIG.endpoints.length}`);
  console.log(`${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}\n`);

  // Create worker pool
  const workers = [];
  for (let i = 0; i < CONFIG.workers; i++) {
    const worker = cluster.fork();
    worker.on('message', (msg) => {
      if (msg.type === 'result') {
        processResult(msg.result);
      }
    });
    workers.push(worker);
  }

  // Run test phases
  for (const phase of CONFIG.phases) {
    await runPhase(phase, workers);
  }

  // Cleanup
  workers.forEach(w => w.kill());
  
  // Display results
  displayResults();
}

/**
 * Run a single test phase
 */
async function runPhase(phase, workers) {
  console.log(`\n${colors.yellow}► Phase: ${phase.name}${colors.reset}`);
  console.log(`  Duration: ${phase.duration}s | Rate: ${phase.arrivalRate} req/s`);
  
  const phaseMetric = {
    name: phase.name,
    requests: 0,
    successes: 0,
    failures: 0,
    avgResponseTime: 0,
    maxResponseTime: 0,
    minResponseTime: Infinity
  };
  
  const startTime = Date.now();
  const endTime = startTime + (phase.duration * 1000);
  const interval = 1000 / phase.arrivalRate;
  
  let workerIndex = 0;
  
  while (Date.now() < endTime) {
    const endpoint = CONFIG.endpoints[Math.floor(Math.random() * CONFIG.endpoints.length)];
    workers[workerIndex].send({ type: 'test', endpoint });
    workerIndex = (workerIndex + 1) % workers.length;
    
    await sleep(interval);
  }
  
  // Wait for phase to complete
  await sleep(2000);
  
  // Calculate phase metrics
  const phaseResults = metrics.responseTimes.slice(-phase.arrivalRate * phase.duration);
  if (phaseResults.length > 0) {
    phaseMetric.avgResponseTime = phaseResults.reduce((a, b) => a + b, 0) / phaseResults.length;
    phaseMetric.maxResponseTime = Math.max(...phaseResults);
    phaseMetric.minResponseTime = Math.min(...phaseResults);
  }
  
  metrics.phaseMetrics[phase.name] = phaseMetric;
  
  // Display phase summary
  const successRate = metrics.successes / metrics.requests * 100;
  console.log(`  ${colors.green}✓${colors.reset} Requests: ${metrics.requests}`);
  console.log(`  ${colors.green}✓${colors.reset} Success Rate: ${successRate.toFixed(2)}%`);
  console.log(`  ${colors.green}✓${colors.reset} Avg Response: ${phaseMetric.avgResponseTime.toFixed(2)}ms`);
}

/**
 * Process individual request result
 */
function processResult(result) {
  metrics.requests++;
  
  if (result.success) {
    metrics.successes++;
  } else {
    metrics.failures++;
    if (result.error) {
      metrics.errors.push(result.error);
    }
  }
  
  metrics.responseTimes.push(result.responseTime);
  
  // Track status codes
  metrics.statusCodes[result.statusCode] = (metrics.statusCodes[result.statusCode] || 0) + 1;
  
  // Track endpoint metrics
  if (!metrics.endpointMetrics[result.endpoint]) {
    metrics.endpointMetrics[result.endpoint] = {
      requests: 0,
      successes: 0,
      failures: 0,
      totalTime: 0
    };
  }
  
  const em = metrics.endpointMetrics[result.endpoint];
  em.requests++;
  em.totalTime += result.responseTime;
  if (result.success) {
    em.successes++;
  } else {
    em.failures++;
  }
}

/**
 * Display comprehensive test results
 */
function displayResults() {
  console.log(`\n${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}`);
  console.log(`${colors.bright}📊 STRESS TEST RESULTS${colors.reset}`);
  console.log(`${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}\n`);

  // Overall metrics
  const successRate = (metrics.successes / metrics.requests * 100).toFixed(2);
  const avgResponseTime = (metrics.responseTimes.reduce((a, b) => a + b, 0) / metrics.responseTimes.length).toFixed(2);
  const maxResponseTime = Math.max(...metrics.responseTimes).toFixed(2);
  const minResponseTime = Math.min(...metrics.responseTimes).toFixed(2);
  const p95ResponseTime = percentile(metrics.responseTimes, 95).toFixed(2);
  const p99ResponseTime = percentile(metrics.responseTimes, 99).toFixed(2);

  console.log(`${colors.bright}Overall Performance:${colors.reset}`);
  console.log(`  Total Requests: ${metrics.requests}`);
  console.log(`  Successful: ${colors.green}${metrics.successes}${colors.reset}`);
  console.log(`  Failed: ${colors.red}${metrics.failures}${colors.reset}`);
  console.log(`  Success Rate: ${successRate >= 95 ? colors.green : successRate >= 80 ? colors.yellow : colors.red}${successRate}%${colors.reset}`);
  console.log(`  Avg Response Time: ${avgResponseTime}ms`);
  console.log(`  Min Response Time: ${minResponseTime}ms`);
  console.log(`  Max Response Time: ${maxResponseTime}ms`);
  console.log(`  95th Percentile: ${p95ResponseTime}ms`);
  console.log(`  99th Percentile: ${p99ResponseTime}ms`);

  // Phase breakdown
  console.log(`\n${colors.bright}Phase Performance:${colors.reset}`);
  Object.values(metrics.phaseMetrics).forEach(phase => {
    console.log(`  ${phase.name}:`);
    console.log(`    Avg Response: ${phase.avgResponseTime.toFixed(2)}ms`);
    console.log(`    Max Response: ${phase.maxResponseTime.toFixed(2)}ms`);
  });

  // Endpoint breakdown
  console.log(`\n${colors.bright}Endpoint Performance:${colors.reset}`);
  Object.entries(metrics.endpointMetrics)
    .sort((a, b) => b[1].requests - a[1].requests)
    .slice(0, 10)
    .forEach(([endpoint, data]) => {
      const avgTime = (data.totalTime / data.requests).toFixed(2);
      const successRate = ((data.successes / data.requests) * 100).toFixed(2);
      console.log(`  ${endpoint}:`);
      console.log(`    Requests: ${data.requests} | Success: ${successRate}% | Avg: ${avgTime}ms`);
    });

  // Status code distribution
  console.log(`\n${colors.bright}Status Code Distribution:${colors.reset}`);
  Object.entries(metrics.statusCodes)
    .sort((a, b) => b[1] - a[1])
    .forEach(([code, count]) => {
      const percentage = ((count / metrics.requests) * 100).toFixed(2);
      const color = code === '200' ? colors.green : code.startsWith('4') ? colors.yellow : code.startsWith('5') ? colors.red : colors.reset;
      console.log(`  ${color}${code}: ${count} (${percentage}%)${colors.reset}`);
    });

  // Error summary
  if (metrics.errors.length > 0) {
    console.log(`\n${colors.bright}Error Summary:${colors.reset}`);
    const errorCounts = {};
    metrics.errors.forEach(err => {
      errorCounts[err] = (errorCounts[err] || 0) + 1;
    });
    Object.entries(errorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .forEach(([error, count]) => {
        console.log(`  ${colors.red}${error}: ${count}${colors.reset}`);
      });
  }

  // Final verdict
  console.log(`\n${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}`);
  console.log(`${colors.bright}🏁 FINAL VERDICT${colors.reset}`);
  console.log(`${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}`);
  
  if (successRate >= 99 && avgResponseTime < 100) {
    console.log(`${colors.green}✅ EXCELLENT: System performed exceptionally well${colors.reset}`);
  } else if (successRate >= 95 && avgResponseTime < 500) {
    console.log(`${colors.green}✅ GOOD: System handled stress test well${colors.reset}`);
  } else if (successRate >= 80 && avgResponseTime < 1000) {
    console.log(`${colors.yellow}⚠️  ACCEPTABLE: System showed some strain but remained operational${colors.reset}`);
  } else {
    console.log(`${colors.red}❌ NEEDS IMPROVEMENT: System struggled under load${colors.reset}`);
  }

  // Recommendations
  console.log(`\n${colors.bright}Recommendations:${colors.reset}`);
  if (avgResponseTime > 500) {
    console.log(`  • Consider implementing caching strategies`);
    console.log(`  • Optimize database queries`);
  }
  if (successRate < 95) {
    console.log(`  • Increase server resources`);
    console.log(`  • Implement rate limiting`);
  }
  if (maxResponseTime > 5000) {
    console.log(`  • Add timeout handling`);
    console.log(`  • Implement circuit breakers`);
  }

  console.log(`\n${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}`);
  console.log(`${colors.magenta}🚀 Stress Test Complete!${colors.reset}`);
  console.log(`${colors.cyan}═══════════════════════════════════════════════════════${colors.reset}\n`);
}

/**
 * Calculate percentile
 */
function percentile(arr, p) {
  const sorted = arr.slice().sort((a, b) => a - b);
  const index = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[index] || 0;
}

/**
 * Sleep utility
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Main execution
if (cluster.isMaster) {
  runMaster().catch(console.error);
} else {
  runWorker();
}