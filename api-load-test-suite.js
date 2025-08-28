#!/usr/bin/env node
/**
 * API Load Testing Suite for Media Server
 * Comprehensive load testing with performance analysis
 * 
 * Features:
 * - Concurrent request testing
 * - Response time analysis
 * - Throughput measurement
 * - Error rate monitoring
 * - Resource usage tracking
 * - Performance regression detection
 * - Real-time metrics dashboard
 */

const http = require('http');
const https = require('https');
const { URL } = require('url');
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const cluster = require('cluster');
const os = require('os');

class APILoadTester {
  constructor(options = {}) {
    this.options = {
      // Test configuration
      duration: options.duration || 60, // Test duration in seconds
      rampUpTime: options.rampUpTime || 10, // Ramp up time in seconds
      concurrency: options.concurrency || 50, // Concurrent users
      requestRate: options.requestRate || 100, // Requests per second
      
      // Target configuration
      baseUrl: options.baseUrl || 'http://localhost',
      timeout: options.timeout || 10000,
      
      // Reporting
      reportInterval: options.reportInterval || 5, // seconds
      outputDir: options.outputDir || './load-test-results',
      generateCharts: options.generateCharts || true,
      
      // Performance thresholds
      thresholds: {
        avgResponseTime: 1000, // ms
        p95ResponseTime: 2000, // ms
        p99ResponseTime: 5000, // ms
        errorRate: 1, // percentage
        throughput: 50 // requests per second
      },
      
      ...options
    };
    
    this.endpoints = this.initializeEndpoints();
    this.metrics = this.initializeMetrics();
    this.isRunning = false;
    this.workers = [];
  }
  
  initializeEndpoints() {
    return [
      // Media Server APIs
      { name: 'jellyfin-health', url: '/health', method: 'GET', weight: 10 },
      { name: 'jellyfin-api', url: '/System/Info/Public', method: 'GET', weight: 5 },
      { name: 'plex-identity', url: '/identity', method: 'GET', port: 32400, weight: 5 },
      
      // *ARR APIs
      { name: 'sonarr-ping', url: '/ping', method: 'GET', port: 8989, weight: 8 },
      { name: 'sonarr-system-status', url: '/api/v3/system/status', method: 'GET', port: 8989, weight: 3 },
      { name: 'radarr-ping', url: '/ping', method: 'GET', port: 7878, weight: 8 },
      { name: 'radarr-system-status', url: '/api/v3/system/status', method: 'GET', port: 7878, weight: 3 },
      { name: 'prowlarr-ping', url: '/ping', method: 'GET', port: 9696, weight: 6 },
      
      // Download Clients
      { name: 'qbittorrent-version', url: '/api/v2/app/version', method: 'GET', port: 8080, weight: 4 },
      { name: 'transmission-stats', url: '/transmission/rpc', method: 'POST', port: 9091, weight: 3 },
      
      // Request Management
      { name: 'overseerr-status', url: '/api/v1/status', method: 'GET', port: 5055, weight: 2 },
      { name: 'jellyseerr-status', url: '/api/v1/status', method: 'GET', port: 5056, weight: 2 },
      
      // Monitoring
      { name: 'prometheus-metrics', url: '/metrics', method: 'GET', port: 9090, weight: 2 },
      { name: 'grafana-health', url: '/api/health', method: 'GET', port: 3000, weight: 1 },
      
      // Dashboard APIs
      { name: 'api-server-health', url: '/health', method: 'GET', port: 3002, weight: 8 },
      { name: 'api-server-services', url: '/api/services', method: 'GET', port: 3002, weight: 4 }
    ];
  }
  
  initializeMetrics() {
    return {
      startTime: null,
      endTime: null,
      totalRequests: 0,
      totalErrors: 0,
      responseTimes: [],
      throughput: [],
      errorRates: [],
      resourceUsage: [],
      endpointMetrics: {},
      workerMetrics: []
    };
  }
  
  async runLoadTest() {
    console.log('🚀 Starting API Load Test Suite');
    console.log('='.repeat(50));
    console.log(`Duration: ${this.options.duration}s`);
    console.log(`Concurrency: ${this.options.concurrency}`);
    console.log(`Target Rate: ${this.options.requestRate} RPS`);
    console.log(`Endpoints: ${this.endpoints.length}`);
    
    // Prepare output directory
    await this.prepareOutputDirectory();
    
    // Pre-test validation
    console.log('\n📋 Pre-test validation...');
    const validation = await this.validateEndpoints();
    if (!validation.allHealthy) {
      console.error('❌ Some endpoints are not accessible. Continuing with available endpoints.');
    }
    
    // Initialize metrics tracking
    this.startMetricsCollection();
    
    // Run the load test
    console.log('\n⚡ Starting load test...');
    this.metrics.startTime = Date.now();
    
    try {
      if (this.options.concurrency > 50) {
        await this.runClusteredTest();
      } else {
        await this.runThreadedTest();
      }
    } catch (error) {
      console.error('❌ Load test failed:', error);
    } finally {
      this.metrics.endTime = Date.now();
      this.stopMetricsCollection();
    }
    
    // Generate reports
    console.log('\n📊 Generating test reports...');
    await this.generateReports();
    
    // Performance analysis
    const analysis = this.analyzePerformance();
    this.displayResults(analysis);
    
    return analysis;
  }
  
  async validateEndpoints() {
    console.log('  Checking endpoint accessibility...');
    
    const results = await Promise.allSettled(
      this.endpoints.map(endpoint => this.testEndpoint(endpoint, 5000))
    );
    
    let accessible = 0;
    results.forEach((result, index) => {
      const endpoint = this.endpoints[index];
      if (result.status === 'fulfilled' && result.value.success) {
        accessible++;
        console.log(`  ✅ ${endpoint.name}: OK (${result.value.responseTime}ms)`);
      } else {
        console.log(`  ❌ ${endpoint.name}: Failed`);
        endpoint.disabled = true; // Disable for load test
      }
    });
    
    console.log(`  📊 ${accessible}/${this.endpoints.length} endpoints accessible`);
    
    return {
      total: this.endpoints.length,
      accessible,
      allHealthy: accessible === this.endpoints.length
    };
  }
  
  async testEndpoint(endpoint, timeout = 10000) {
    return new Promise((resolve) => {
      const startTime = Date.now();
      const port = endpoint.port || 8096;
      const url = `http://localhost:${port}${endpoint.url}`;
      
      const request = http.get(url, { timeout }, (response) => {
        let data = '';
        response.on('data', chunk => data += chunk);
        response.on('end', () => {
          resolve({
            success: true,
            responseTime: Date.now() - startTime,
            statusCode: response.statusCode,
            contentLength: data.length
          });
        });
      });
      
      request.on('timeout', () => {
        request.destroy();
        resolve({ success: false, error: 'timeout' });
      });
      
      request.on('error', (error) => {
        resolve({ success: false, error: error.message });
      });
    });
  }
  
  async runThreadedTest() {
    const numWorkers = Math.min(this.options.concurrency, os.cpus().length);
    const requestsPerWorker = Math.ceil(this.options.concurrency / numWorkers);
    
    console.log(`  Using ${numWorkers} worker threads`);
    console.log(`  ${requestsPerWorker} concurrent requests per worker`);
    
    const workerPromises = [];
    
    for (let i = 0; i < numWorkers; i++) {
      const workerPromise = new Promise((resolve, reject) => {
        const worker = new Worker(__filename, {
          workerData: {
            workerId: i,
            concurrency: requestsPerWorker,
            duration: this.options.duration,
            rampUpTime: this.options.rampUpTime,
            endpoints: this.endpoints.filter(e => !e.disabled),
            options: this.options
          }
        });
        
        worker.on('message', (data) => {
          if (data.type === 'metrics') {
            this.aggregateWorkerMetrics(data.metrics);
          } else if (data.type === 'progress') {
            this.updateProgress(data.progress);
          }
        });
        
        worker.on('error', reject);
        worker.on('exit', (code) => {
          if (code !== 0) {
            reject(new Error(`Worker stopped with exit code ${code}`));
          } else {
            resolve();
          }
        });
        
        this.workers.push(worker);
      });
      
      workerPromises.push(workerPromise);
    }
    
    await Promise.all(workerPromises);
  }
  
  async runClusteredTest() {
    if (cluster.isMaster) {
      console.log(`  Using cluster mode with ${os.cpus().length} processes`);
      
      const numWorkers = Math.min(this.options.concurrency / 10, os.cpus().length);
      
      for (let i = 0; i < numWorkers; i++) {
        const worker = cluster.fork({
          WORKER_ID: i,
          LOAD_TEST_CONFIG: JSON.stringify(this.options)
        });
        
        worker.on('message', (data) => {
          if (data.type === 'metrics') {
            this.aggregateWorkerMetrics(data.metrics);
          }
        });
      }
      
      // Wait for workers to complete
      await new Promise((resolve) => {
        cluster.on('exit', (worker, code, signal) => {
          if (Object.keys(cluster.workers).length === 0) {
            resolve();
          }
        });
        
        setTimeout(() => {
          for (const id in cluster.workers) {
            cluster.workers[id].kill();
          }
        }, this.options.duration * 1000 + 10000);
      });
    } else {
      // Worker process
      await this.runWorkerProcess();
    }
  }
  
  async runWorkerProcess() {
    const workerId = process.env.WORKER_ID || 0;
    const config = JSON.parse(process.env.LOAD_TEST_CONFIG);
    
    const metrics = {
      workerId,
      requests: 0,
      errors: 0,
      responseTimes: []
    };
    
    const startTime = Date.now();
    const endTime = startTime + (config.duration * 1000);
    const rampUpEnd = startTime + (config.rampUpTime * 1000);
    
    while (Date.now() < endTime) {
      const now = Date.now();
      
      // Calculate current concurrency (ramp up)
      let currentConcurrency = config.concurrency;
      if (now < rampUpEnd) {
        const rampUpProgress = (now - startTime) / (rampUpEnd - startTime);
        currentConcurrency = Math.floor(config.concurrency * rampUpProgress) + 1;
      }
      
      // Make requests
      const requests = [];
      for (let i = 0; i < Math.min(currentConcurrency, 10); i++) {
        const endpoint = this.selectEndpoint();
        requests.push(this.makeRequest(endpoint));
      }
      
      const results = await Promise.allSettled(requests);
      
      // Update metrics
      results.forEach(result => {
        metrics.requests++;
        if (result.status === 'fulfilled' && result.value.success) {
          metrics.responseTimes.push(result.value.responseTime);
        } else {
          metrics.errors++;
        }
      });
      
      // Send metrics to master process
      if (process.send) {
        process.send({
          type: 'metrics',
          metrics: {
            ...metrics,
            timestamp: Date.now()
          }
        });
      }
      
      // Control request rate
      await this.sleep(1000 / config.requestRate);
    }
  }
  
  selectEndpoint() {
    // Weighted random selection
    const availableEndpoints = this.endpoints.filter(e => !e.disabled);
    const totalWeight = availableEndpoints.reduce((sum, e) => sum + (e.weight || 1), 0);
    
    let random = Math.random() * totalWeight;
    
    for (const endpoint of availableEndpoints) {
      random -= (endpoint.weight || 1);
      if (random <= 0) {
        return endpoint;
      }
    }
    
    return availableEndpoints[0];
  }
  
  async makeRequest(endpoint) {
    const startTime = Date.now();
    const port = endpoint.port || 8096;
    const url = `http://localhost:${port}${endpoint.url}`;
    
    return new Promise((resolve) => {
      const request = http.get(url, { timeout: this.options.timeout }, (response) => {
        let data = '';
        response.on('data', chunk => data += chunk);
        response.on('end', () => {
          resolve({
            success: response.statusCode < 400,
            responseTime: Date.now() - startTime,
            statusCode: response.statusCode,
            endpoint: endpoint.name
          });
        });
      });
      
      request.on('timeout', () => {
        request.destroy();
        resolve({
          success: false,
          responseTime: Date.now() - startTime,
          error: 'timeout',
          endpoint: endpoint.name
        });
      });
      
      request.on('error', (error) => {
        resolve({
          success: false,
          responseTime: Date.now() - startTime,
          error: error.message,
          endpoint: endpoint.name
        });
      });
    });
  }
  
  startMetricsCollection() {
    this.metricsInterval = setInterval(() => {
      this.collectSystemMetrics();
    }, this.options.reportInterval * 1000);
  }
  
  stopMetricsCollection() {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }
  }
  
  collectSystemMetrics() {
    const memUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    this.metrics.resourceUsage.push({
      timestamp: Date.now(),
      memory: {
        rss: memUsage.rss,
        heapTotal: memUsage.heapTotal,
        heapUsed: memUsage.heapUsed,
        external: memUsage.external
      },
      cpu: cpuUsage,
      loadAvg: os.loadavg()
    });
  }
  
  aggregateWorkerMetrics(workerMetrics) {
    this.metrics.totalRequests += workerMetrics.requests || 0;
    this.metrics.totalErrors += workerMetrics.errors || 0;
    
    if (workerMetrics.responseTimes) {
      this.metrics.responseTimes.push(...workerMetrics.responseTimes);
    }
    
    this.metrics.workerMetrics.push({
      ...workerMetrics,
      timestamp: Date.now()
    });
  }
  
  updateProgress(progress) {
    // Update real-time progress display
    process.stdout.write(`\r  Progress: ${progress.percentage}% | ` +
                        `RPS: ${progress.currentRPS} | ` +
                        `Errors: ${progress.errorRate}%`);
  }
  
  analyzePerformance() {
    const duration = (this.metrics.endTime - this.metrics.startTime) / 1000;
    const responseTimes = this.metrics.responseTimes.sort((a, b) => a - b);
    
    const analysis = {
      duration,
      totalRequests: this.metrics.totalRequests,
      totalErrors: this.metrics.totalErrors,
      averageRPS: this.metrics.totalRequests / duration,
      errorRate: (this.metrics.totalErrors / this.metrics.totalRequests) * 100,
      
      responseTime: {
        min: Math.min(...responseTimes),
        max: Math.max(...responseTimes),
        mean: responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length,
        median: this.percentile(responseTimes, 50),
        p95: this.percentile(responseTimes, 95),
        p99: this.percentile(responseTimes, 99)
      },
      
      thresholds: {
        passed: 0,
        failed: 0,
        details: {}
      }
    };
    
    // Check performance thresholds
    const thresholds = this.options.thresholds;
    
    const checks = [
      { name: 'avgResponseTime', actual: analysis.responseTime.mean, threshold: thresholds.avgResponseTime, unit: 'ms' },
      { name: 'p95ResponseTime', actual: analysis.responseTime.p95, threshold: thresholds.p95ResponseTime, unit: 'ms' },
      { name: 'p99ResponseTime', actual: analysis.responseTime.p99, threshold: thresholds.p99ResponseTime, unit: 'ms' },
      { name: 'errorRate', actual: analysis.errorRate, threshold: thresholds.errorRate, unit: '%' },
      { name: 'throughput', actual: analysis.averageRPS, threshold: thresholds.throughput, unit: 'RPS' }
    ];
    
    checks.forEach(check => {
      const passed = check.name === 'throughput' 
        ? check.actual >= check.threshold
        : check.actual <= check.threshold;
      
      if (passed) {
        analysis.thresholds.passed++;
      } else {
        analysis.thresholds.failed++;
      }
      
      analysis.thresholds.details[check.name] = {
        passed,
        actual: check.actual,
        threshold: check.threshold,
        unit: check.unit
      };
    });
    
    return analysis;
  }
  
  percentile(arr, p) {
    if (arr.length === 0) return 0;
    const index = Math.ceil((p / 100) * arr.length) - 1;
    return arr[Math.max(0, index)];
  }
  
  async prepareOutputDirectory() {
    const dir = this.options.outputDir;
    
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    // Create subdirectories
    const subdirs = ['charts', 'raw-data', 'reports'];
    subdirs.forEach(subdir => {
      const subdirPath = path.join(dir, subdir);
      if (!fs.existsSync(subdirPath)) {
        fs.mkdirSync(subdirPath);
      }
    });
  }
  
  async generateReports() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Save raw metrics
    const rawDataPath = path.join(this.options.outputDir, 'raw-data', `metrics-${timestamp}.json`);
    fs.writeFileSync(rawDataPath, JSON.stringify(this.metrics, null, 2));
    
    // Generate performance report
    const analysis = this.analyzePerformance();
    const reportPath = path.join(this.options.outputDir, 'reports', `report-${timestamp}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(analysis, null, 2));
    
    // Generate HTML report
    await this.generateHTMLReport(analysis, timestamp);
    
    // Generate charts if requested
    if (this.options.generateCharts) {
      await this.generateCharts(timestamp);
    }
    
    console.log(`  📁 Reports saved to ${this.options.outputDir}`);
  }
  
  async generateHTMLReport(analysis, timestamp) {
    const htmlContent = `<!DOCTYPE html>
<html>
<head>
    <title>Load Test Report - ${timestamp}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .threshold-table { width: 100%; border-collapse: collapse; }
        .threshold-table th, .threshold-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .chart-placeholder { background: #f9f9f9; height: 300px; display: flex; align-items: center; justify-content: center; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Load Test Report</h1>
        <p><strong>Timestamp:</strong> ${new Date(timestamp.replace(/-/g, ':')).toLocaleString()}</p>
        <p><strong>Duration:</strong> ${analysis.duration.toFixed(2)}s</p>
        <p><strong>Concurrency:</strong> ${this.options.concurrency}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <h3>📊 Request Metrics</h3>
            <p><strong>Total Requests:</strong> ${analysis.totalRequests.toLocaleString()}</p>
            <p><strong>Total Errors:</strong> ${analysis.totalErrors.toLocaleString()}</p>
            <p><strong>Error Rate:</strong> ${analysis.errorRate.toFixed(2)}%</p>
            <p><strong>Throughput:</strong> ${analysis.averageRPS.toFixed(2)} RPS</p>
        </div>
        
        <div class="metric-card">
            <h3>⏱️ Response Times</h3>
            <p><strong>Min:</strong> ${analysis.responseTime.min}ms</p>
            <p><strong>Max:</strong> ${analysis.responseTime.max}ms</p>
            <p><strong>Mean:</strong> ${analysis.responseTime.mean.toFixed(2)}ms</p>
            <p><strong>Median:</strong> ${analysis.responseTime.median.toFixed(2)}ms</p>
            <p><strong>95th percentile:</strong> ${analysis.responseTime.p95.toFixed(2)}ms</p>
            <p><strong>99th percentile:</strong> ${analysis.responseTime.p99.toFixed(2)}ms</p>
        </div>
        
        <div class="metric-card">
            <h3>🎯 Performance Thresholds</h3>
            <p><strong>Passed:</strong> <span class="pass">${analysis.thresholds.passed}</span></p>
            <p><strong>Failed:</strong> <span class="fail">${analysis.thresholds.failed}</span></p>
            
            <table class="threshold-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Actual</th>
                        <th>Threshold</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(analysis.thresholds.details).map(([name, detail]) => `
                        <tr>
                            <td>${name}</td>
                            <td>${detail.actual.toFixed(2)} ${detail.unit}</td>
                            <td>${detail.threshold} ${detail.unit}</td>
                            <td class="${detail.passed ? 'pass' : 'fail'}">${detail.passed ? 'PASS' : 'FAIL'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="metric-card">
        <h3>📈 Performance Charts</h3>
        <div class="chart-placeholder">
            Response Time Distribution Chart
            <br>(Charts would be generated here with actual charting library)
        </div>
    </div>
</body>
</html>`;
    
    const htmlPath = path.join(this.options.outputDir, 'reports', `report-${timestamp}.html`);
    fs.writeFileSync(htmlPath, htmlContent);
  }
  
  async generateCharts(timestamp) {
    // Placeholder for chart generation
    // In a real implementation, you'd use a charting library like Chart.js or D3.js
    console.log('  📈 Chart generation would be implemented with a charting library');
  }
  
  displayResults(analysis) {
    console.log('\n' + '='.repeat(60));
    console.log('📋 LOAD TEST RESULTS');
    console.log('='.repeat(60));
    
    console.log(`⏱️  Duration: ${analysis.duration.toFixed(2)}s`);
    console.log(`📊 Total Requests: ${analysis.totalRequests.toLocaleString()}`);
    console.log(`❌ Total Errors: ${analysis.totalErrors.toLocaleString()}`);
    console.log(`🔢 Error Rate: ${analysis.errorRate.toFixed(2)}%`);
    console.log(`⚡ Throughput: ${analysis.averageRPS.toFixed(2)} RPS`);
    
    console.log('\n📈 Response Times:');
    console.log(`  Min: ${analysis.responseTime.min}ms`);
    console.log(`  Max: ${analysis.responseTime.max}ms`);
    console.log(`  Mean: ${analysis.responseTime.mean.toFixed(2)}ms`);
    console.log(`  Median: ${analysis.responseTime.median.toFixed(2)}ms`);
    console.log(`  95th percentile: ${analysis.responseTime.p95.toFixed(2)}ms`);
    console.log(`  99th percentile: ${analysis.responseTime.p99.toFixed(2)}ms`);
    
    console.log('\n🎯 Performance Thresholds:');
    Object.entries(analysis.thresholds.details).forEach(([name, detail]) => {
      const status = detail.passed ? '✅ PASS' : '❌ FAIL';
      console.log(`  ${name}: ${detail.actual.toFixed(2)} ${detail.unit} (threshold: ${detail.threshold} ${detail.unit}) - ${status}`);
    });
    
    const overallStatus = analysis.thresholds.failed === 0 ? '✅ PASS' : '❌ FAIL';
    console.log(`\n🏆 Overall Performance: ${overallStatus} (${analysis.thresholds.passed}/${analysis.thresholds.passed + analysis.thresholds.failed})`);
    
    console.log('='.repeat(60));
  }
  
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Worker thread execution
if (!isMainThread && workerData) {
  (async () => {
    const { workerId, concurrency, duration, rampUpTime, endpoints, options } = workerData;
    
    const metrics = {
      workerId,
      requests: 0,
      errors: 0,
      responseTimes: []
    };
    
    const startTime = Date.now();
    const endTime = startTime + (duration * 1000);
    const rampUpEnd = startTime + (rampUpTime * 1000);
    
    const selectEndpoint = () => {
      const totalWeight = endpoints.reduce((sum, e) => sum + (e.weight || 1), 0);
      let random = Math.random() * totalWeight;
      
      for (const endpoint of endpoints) {
        random -= (endpoint.weight || 1);
        if (random <= 0) {
          return endpoint;
        }
      }
      
      return endpoints[0];
    };
    
    const makeRequest = async (endpoint) => {
      const startTime = Date.now();
      const port = endpoint.port || 8096;
      const url = `http://localhost:${port}${endpoint.url}`;
      
      return new Promise((resolve) => {
        const request = http.get(url, { timeout: options.timeout }, (response) => {
          let data = '';
          response.on('data', chunk => data += chunk);
          response.on('end', () => {
            resolve({
              success: response.statusCode < 400,
              responseTime: Date.now() - startTime,
              statusCode: response.statusCode
            });
          });
        });
        
        request.on('timeout', () => {
          request.destroy();
          resolve({
            success: false,
            responseTime: Date.now() - startTime,
            error: 'timeout'
          });
        });
        
        request.on('error', (error) => {
          resolve({
            success: false,
            responseTime: Date.now() - startTime,
            error: error.message
          });
        });
      });
    };
    
    while (Date.now() < endTime) {
      const now = Date.now();
      
      // Calculate current concurrency (ramp up)
      let currentConcurrency = concurrency;
      if (now < rampUpEnd) {
        const rampUpProgress = (now - startTime) / (rampUpEnd - startTime);
        currentConcurrency = Math.floor(concurrency * rampUpProgress) + 1;
      }
      
      // Make requests
      const requests = [];
      for (let i = 0; i < Math.min(currentConcurrency, 20); i++) {
        const endpoint = selectEndpoint();
        requests.push(makeRequest(endpoint));
      }
      
      const results = await Promise.allSettled(requests);
      
      // Update metrics
      results.forEach(result => {
        metrics.requests++;
        if (result.status === 'fulfilled' && result.value.success) {
          metrics.responseTimes.push(result.value.responseTime);
        } else {
          metrics.errors++;
        }
      });
      
      // Send progress update
      const progress = {
        percentage: Math.round(((now - startTime) / (endTime - startTime)) * 100),
        currentRPS: metrics.requests / ((now - startTime) / 1000),
        errorRate: metrics.requests > 0 ? (metrics.errors / metrics.requests) * 100 : 0
      };
      
      if (parentPort) {
        parentPort.postMessage({
          type: 'progress',
          progress
        });
        
        parentPort.postMessage({
          type: 'metrics',
          metrics
        });
      }
      
      // Control request rate
      await new Promise(resolve => setTimeout(resolve, 1000 / options.requestRate));
    }
  })();
}

// CLI execution
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};
  
  // Parse command line arguments
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    if (key && value) {
      if (['duration', 'concurrency', 'requestRate', 'timeout', 'reportInterval'].includes(key)) {
        options[key] = parseInt(value);
      } else {
        options[key] = value;
      }
    }
  }
  
  const tester = new APILoadTester(options);
  
  tester.runLoadTest()
    .then((results) => {
      const exitCode = results.thresholds.failed === 0 ? 0 : 1;
      process.exit(exitCode);
    })
    .catch((error) => {
      console.error('❌ Load test failed:', error);
      process.exit(2);
    });
}

module.exports = APILoadTester;