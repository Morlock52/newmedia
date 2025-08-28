#!/usr/bin/env node
/**
 * Comprehensive Service Health Testing Suite
 * Tests all 30+ services in the media server container
 * Validates process management, dependencies, and performance
 * 
 * Features:
 * - Service discovery and health checks
 * - s6-overlay process monitoring
 * - Memory/CPU usage analysis
 * - Dependency chain validation
 * - Load testing capabilities
 * - Real-time reporting
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const { exec, spawn } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

// Service configuration with health check endpoints
const SERVICE_CONFIG = {
  // Media Servers - Critical Tier
  'jellyfin': {
    port: 8096,
    healthEndpoint: '/health',
    processName: 'jellyfin',
    tier: 'critical',
    dependencies: [],
    expectedMemoryMB: 512,
    startupTimeMs: 30000
  },
  'plex': {
    port: 32400,
    healthEndpoint: '/identity',
    processName: 'plex',
    tier: 'critical',
    dependencies: [],
    expectedMemoryMB: 1024,
    startupTimeMs: 45000
  },
  'emby': {
    port: 8097,
    healthEndpoint: '/System/Info/Public',
    processName: 'emby',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 512,
    startupTimeMs: 30000
  },

  // *ARR Stack - High Priority Tier
  'sonarr': {
    port: 8989,
    healthEndpoint: '/ping',
    processName: 'sonarr',
    tier: 'high',
    dependencies: ['prowlarr'],
    expectedMemoryMB: 256,
    startupTimeMs: 25000
  },
  'radarr': {
    port: 7878,
    healthEndpoint: '/ping',
    processName: 'radarr',
    tier: 'high',
    dependencies: ['prowlarr'],
    expectedMemoryMB: 256,
    startupTimeMs: 25000
  },
  'lidarr': {
    port: 8686,
    healthEndpoint: '/ping',
    processName: 'lidarr',
    tier: 'medium',
    dependencies: ['prowlarr'],
    expectedMemoryMB: 200,
    startupTimeMs: 20000
  },
  'readarr': {
    port: 8787,
    healthEndpoint: '/ping',
    processName: 'readarr',
    tier: 'medium',
    dependencies: ['prowlarr'],
    expectedMemoryMB: 200,
    startupTimeMs: 20000
  },
  'bazarr': {
    port: 6767,
    healthEndpoint: '/api/system/status',
    processName: 'bazarr',
    tier: 'medium',
    dependencies: ['sonarr', 'radarr'],
    expectedMemoryMB: 150,
    startupTimeMs: 15000
  },
  'prowlarr': {
    port: 9696,
    healthEndpoint: '/ping',
    processName: 'prowlarr',
    tier: 'high',
    dependencies: [],
    expectedMemoryMB: 256,
    startupTimeMs: 20000
  },

  // Download Clients - High Priority Tier
  'qbittorrent': {
    port: 8080,
    healthEndpoint: '/api/v2/app/version',
    processName: 'qbittorrent-nox',
    tier: 'high',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 15000
  },
  'transmission': {
    port: 9091,
    healthEndpoint: '/transmission/rpc',
    processName: 'transmission-daemon',
    tier: 'medium',
    dependencies: [],
    expectedMemoryMB: 64,
    startupTimeMs: 10000
  },
  'sabnzbd': {
    port: 8081,
    healthEndpoint: '/api',
    processName: 'sabnzbd',
    tier: 'medium',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 15000
  },
  'nzbget': {
    port: 6789,
    healthEndpoint: '/jsonrpc',
    processName: 'nzbget',
    tier: 'medium',
    dependencies: [],
    expectedMemoryMB: 64,
    startupTimeMs: 10000
  },

  // Request Management - Medium Priority
  'overseerr': {
    port: 5055,
    healthEndpoint: '/api/v1/status',
    processName: 'overseerr',
    tier: 'medium',
    dependencies: ['plex', 'sonarr', 'radarr'],
    expectedMemoryMB: 256,
    startupTimeMs: 30000
  },
  'jellyseerr': {
    port: 5056,
    healthEndpoint: '/api/v1/status',
    processName: 'jellyseerr',
    tier: 'medium',
    dependencies: ['jellyfin', 'sonarr', 'radarr'],
    expectedMemoryMB: 256,
    startupTimeMs: 30000
  },
  'ombi': {
    port: 3579,
    healthEndpoint: '/api/v1/Status',
    processName: 'ombi',
    tier: 'optional',
    dependencies: ['plex', 'sonarr', 'radarr'],
    expectedMemoryMB: 200,
    startupTimeMs: 25000
  },

  // Dashboards - Medium Priority
  'homarr': {
    port: 7575,
    healthEndpoint: '/api/health',
    processName: 'homarr',
    tier: 'medium',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 20000
  },
  'homepage': {
    port: 3003,
    healthEndpoint: '/api/ping',
    processName: 'homepage',
    tier: 'medium',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 15000
  },
  'tautulli': {
    port: 8181,
    healthEndpoint: '/api/v2',
    processName: 'tautulli',
    tier: 'medium',
    dependencies: ['plex'],
    expectedMemoryMB: 128,
    startupTimeMs: 20000
  },

  // Content Libraries - Optional Tier
  'calibre-web': {
    port: 8083,
    healthEndpoint: '/opds',
    processName: 'calibre-web',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 15000
  },
  'audiobookshelf': {
    port: 13378,
    healthEndpoint: '/healthcheck',
    processName: 'audiobookshelf',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 256,
    startupTimeMs: 20000
  },
  'navidrome': {
    port: 4533,
    healthEndpoint: '/app',
    processName: 'navidrome',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 64,
    startupTimeMs: 10000
  },
  'photoprism': {
    port: 2342,
    healthEndpoint: '/api/v1/status',
    processName: 'photoprism',
    tier: 'optional',
    dependencies: ['mariadb'],
    expectedMemoryMB: 512,
    startupTimeMs: 45000
  },

  // Utilities - Optional Tier
  'vaultwarden': {
    port: 8085,
    healthEndpoint: '/alive',
    processName: 'vaultwarden',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 64,
    startupTimeMs: 10000
  },
  'pihole': {
    port: 8053,
    healthEndpoint: '/admin/api.php',
    processName: 'pihole-FTL',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 15000
  },
  'syncthing': {
    port: 8384,
    healthEndpoint: '/rest/system/status',
    processName: 'syncthing',
    tier: 'optional',
    dependencies: [],
    expectedMemoryMB: 64,
    startupTimeMs: 15000
  },

  // Monitoring - Critical for Health Monitoring
  'prometheus': {
    port: 9090,
    healthEndpoint: '/-/healthy',
    processName: 'prometheus',
    tier: 'critical',
    dependencies: [],
    expectedMemoryMB: 256,
    startupTimeMs: 20000
  },
  'grafana': {
    port: 3000,
    healthEndpoint: '/api/health',
    processName: 'grafana-server',
    tier: 'high',
    dependencies: ['prometheus'],
    expectedMemoryMB: 256,
    startupTimeMs: 25000
  },
  'uptime-kuma': {
    port: 3001,
    healthEndpoint: '/',
    processName: 'uptime-kuma',
    tier: 'high',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 20000
  },

  // Database Services - Critical Infrastructure
  'postgres': {
    port: 5432,
    healthEndpoint: null, // TCP check only
    processName: 'postgres',
    tier: 'critical',
    dependencies: [],
    expectedMemoryMB: 256,
    startupTimeMs: 15000
  },
  'redis': {
    port: 6379,
    healthEndpoint: null, // TCP check only
    processName: 'redis-server',
    tier: 'critical',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 5000
  },
  'mariadb': {
    port: 3306,
    healthEndpoint: null, // TCP check only
    processName: 'mariadb',
    tier: 'critical',
    dependencies: [],
    expectedMemoryMB: 256,
    startupTimeMs: 20000
  },

  // Management - Medium Priority
  'portainer': {
    port: 9000,
    healthEndpoint: '/api/status',
    processName: 'portainer',
    tier: 'medium',
    dependencies: [],
    expectedMemoryMB: 128,
    startupTimeMs: 15000
  }
};

class ServiceHealthTester {
  constructor() {
    this.results = {
      services: {},
      summary: {},
      performance: {},
      dependencies: {},
      timestamp: new Date().toISOString()
    };
    this.isContainer = false;
    this.containerName = null;
  }

  // Main test runner
  async runComprehensiveTest() {
    console.log('🏥 Starting Comprehensive Service Health Testing Suite');
    console.log('='.repeat(60));
    
    // Detect deployment type
    await this.detectDeploymentType();
    
    // Phase 1: Infrastructure checks
    console.log('\n📋 Phase 1: Infrastructure Health Checks');
    await this.testInfrastructure();
    
    // Phase 2: Service discovery
    console.log('\n🔍 Phase 2: Service Discovery & Process Validation');
    await this.discoverServices();
    
    // Phase 3: Health endpoint tests
    console.log('\n🌐 Phase 3: HTTP Health Endpoint Testing');
    await this.testHealthEndpoints();
    
    // Phase 4: Dependency validation
    console.log('\n🔗 Phase 4: Service Dependency Chain Validation');
    await this.validateDependencies();
    
    // Phase 5: Performance analysis
    console.log('\n📊 Phase 5: Performance & Resource Analysis');
    await this.analyzePerformance();
    
    // Phase 6: Load testing
    console.log('\n⚡ Phase 6: Basic Load Testing');
    await this.performLoadTest();
    
    // Phase 7: Generate comprehensive report
    console.log('\n📋 Phase 7: Generating Health Status Report');
    await this.generateReport();
    
    return this.results;
  }

  // Detect if we're testing a container or separate services
  async detectDeploymentType() {
    try {
      // Check for single container deployment
      const { stdout } = await execAsync('docker ps --format "table {{.Names}}" | grep -E "(ultimate-media|media-server)"');
      if (stdout.trim()) {
        this.isContainer = true;
        this.containerName = stdout.trim().split('\n')[0];
        console.log(`✅ Detected single container deployment: ${this.containerName}`);
      } else {
        console.log('✅ Detected multi-container or host deployment');
      }
    } catch (error) {
      console.log('ℹ️  Docker not available or no containers running - testing host services');
    }
  }

  // Test infrastructure components
  async testInfrastructure() {
    const infraTests = {
      's6-overlay': () => this.testS6Overlay(),
      'docker-daemon': () => this.testDockerDaemon(),
      'system-resources': () => this.testSystemResources(),
      'network-connectivity': () => this.testNetworkConnectivity()
    };

    for (const [test, fn] of Object.entries(infraTests)) {
      try {
        console.log(`  Testing ${test}...`);
        const result = await fn();
        this.results.infrastructure = this.results.infrastructure || {};
        this.results.infrastructure[test] = result;
        console.log(`  ✅ ${test}: ${result.status}`);
      } catch (error) {
        console.log(`  ❌ ${test}: ${error.message}`);
        this.results.infrastructure[test] = {
          status: 'failed',
          error: error.message
        };
      }
    }
  }

  // Test s6-overlay process supervisor
  async testS6Overlay() {
    if (!this.isContainer) {
      return { status: 'skipped', reason: 'Not a container deployment' };
    }

    try {
      const { stdout } = await execAsync(`docker exec ${this.containerName} s6-svstat /run/service/*`);
      const services = stdout.split('\n').filter(line => line.trim()).length;
      
      return {
        status: 'healthy',
        managedServices: services,
        details: 's6-overlay managing processes successfully'
      };
    } catch (error) {
      return {
        status: 'warning',
        error: error.message,
        details: 'Could not verify s6-overlay status'
      };
    }
  }

  // Test Docker daemon connectivity
  async testDockerDaemon() {
    try {
      await execAsync('docker version --format "{{.Server.Version}}"');
      return {
        status: 'healthy',
        details: 'Docker daemon accessible'
      };
    } catch (error) {
      return {
        status: 'failed',
        error: 'Docker daemon not accessible'
      };
    }
  }

  // Test system resources
  async testSystemResources() {
    try {
      const command = this.isContainer 
        ? `docker exec ${this.containerName} top -bn1`
        : 'top -bn1';
        
      const { stdout } = await execAsync(command);
      const lines = stdout.split('\n');
      
      // Parse CPU and memory info
      const cpuLine = lines.find(line => line.includes('Cpu(s)'));
      const memLine = lines.find(line => line.includes('KiB Mem'));
      
      return {
        status: 'healthy',
        cpu: cpuLine ? this.parseCpuUsage(cpuLine) : 'unknown',
        memory: memLine ? this.parseMemoryUsage(memLine) : 'unknown'
      };
    } catch (error) {
      return {
        status: 'warning',
        error: error.message
      };
    }
  }

  // Test network connectivity
  async testNetworkConnectivity() {
    const testUrls = [
      'https://google.com',
      'https://github.com',
      'https://docker.io'
    ];
    
    const results = await Promise.all(
      testUrls.map(url => this.testNetworkEndpoint(url))
    );
    
    const successful = results.filter(r => r.status === 'success').length;
    
    return {
      status: successful >= 2 ? 'healthy' : 'warning',
      successful,
      total: testUrls.length,
      details: 'External network connectivity test'
    };
  }

  // Discover running services
  async discoverServices() {
    const discovered = {};
    
    for (const [serviceName, config] of Object.entries(SERVICE_CONFIG)) {
      console.log(`  Discovering ${serviceName}...`);
      
      const serviceResult = {
        configured: true,
        processRunning: false,
        portListening: false,
        processDetails: null,
        resourceUsage: null
      };

      // Test if process is running
      serviceResult.processRunning = await this.isProcessRunning(config.processName);
      
      // Test if port is listening
      serviceResult.portListening = await this.isPortListening(config.port);
      
      // Get process details if running
      if (serviceResult.processRunning) {
        serviceResult.processDetails = await this.getProcessDetails(config.processName);
      }
      
      // Get resource usage
      serviceResult.resourceUsage = await this.getResourceUsage(config.processName);
      
      discovered[serviceName] = serviceResult;
      
      const status = serviceResult.processRunning && serviceResult.portListening ? '✅' : '❌';
      console.log(`  ${status} ${serviceName}: Process=${serviceResult.processRunning}, Port=${serviceResult.portListening}`);
    }
    
    this.results.services = discovered;
  }

  // Test health endpoints
  async testHealthEndpoints() {
    const healthTests = [];
    
    for (const [serviceName, config] of Object.entries(SERVICE_CONFIG)) {
      if (config.healthEndpoint && this.results.services[serviceName]?.portListening) {
        healthTests.push(this.testServiceHealth(serviceName, config));
      }
    }
    
    const results = await Promise.allSettled(healthTests);
    
    results.forEach((result, index) => {
      const serviceName = Object.keys(SERVICE_CONFIG)[index];
      if (result.status === 'fulfilled') {
        this.results.services[serviceName].healthEndpoint = result.value;
      } else {
        this.results.services[serviceName].healthEndpoint = {
          status: 'failed',
          error: result.reason.message
        };
      }
    });
  }

  // Test individual service health
  async testServiceHealth(serviceName, config) {
    const url = `http://localhost:${config.port}${config.healthEndpoint}`;
    const timeout = 5000;
    
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const request = http.get(url, { timeout }, (response) => {
        let data = '';
        
        response.on('data', chunk => data += chunk);
        response.on('end', () => {
          const responseTime = Date.now() - startTime;
          
          resolve({
            status: 'healthy',
            httpStatus: response.statusCode,
            responseTime,
            contentLength: data.length,
            headers: response.headers
          });
        });
      });
      
      request.on('timeout', () => {
        request.destroy();
        reject(new Error(`Timeout after ${timeout}ms`));
      });
      
      request.on('error', (error) => {
        reject(error);
      });
    });
  }

  // Validate service dependencies
  async validateDependencies() {
    const dependencyResults = {};
    
    for (const [serviceName, config] of Object.entries(SERVICE_CONFIG)) {
      if (config.dependencies.length > 0) {
        console.log(`  Checking dependencies for ${serviceName}...`);
        
        const dependencyStatus = {};
        let allDependenciesHealthy = true;
        
        for (const dependency of config.dependencies) {
          const depService = this.results.services[dependency];
          const isHealthy = depService?.processRunning && depService?.portListening;
          
          dependencyStatus[dependency] = {
            required: true,
            healthy: isHealthy,
            status: depService || { error: 'Service not found' }
          };
          
          if (!isHealthy) {
            allDependenciesHealthy = false;
          }
        }
        
        dependencyResults[serviceName] = {
          allDependenciesHealthy,
          dependencies: dependencyStatus
        };
        
        const status = allDependenciesHealthy ? '✅' : '⚠️';
        console.log(`  ${status} ${serviceName}: Dependencies ${allDependenciesHealthy ? 'satisfied' : 'missing'}`);
      }
    }
    
    this.results.dependencies = dependencyResults;
  }

  // Analyze performance and resource usage
  async analyzePerformance() {
    const performanceData = {
      totalServices: Object.keys(SERVICE_CONFIG).length,
      runningServices: 0,
      healthyServices: 0,
      resourceUsage: {
        totalMemoryMB: 0,
        totalCPU: 0
      },
      tierHealth: {}
    };
    
    // Count service statuses by tier
    const tiers = ['critical', 'high', 'medium', 'optional'];
    tiers.forEach(tier => {
      performanceData.tierHealth[tier] = {
        total: 0,
        running: 0,
        healthy: 0
      };
    });
    
    for (const [serviceName, config] of Object.entries(SERVICE_CONFIG)) {
      const serviceResult = this.results.services[serviceName];
      const tier = config.tier;
      
      performanceData.tierHealth[tier].total++;
      
      if (serviceResult?.processRunning) {
        performanceData.runningServices++;
        performanceData.tierHealth[tier].running++;
        
        // Add to resource usage
        if (serviceResult.resourceUsage) {
          performanceData.resourceUsage.totalMemoryMB += serviceResult.resourceUsage.memoryMB || 0;
          performanceData.resourceUsage.totalCPU += serviceResult.resourceUsage.cpuPercent || 0;
        }
      }
      
      if (serviceResult?.healthEndpoint?.status === 'healthy') {
        performanceData.healthyServices++;
        performanceData.tierHealth[tier].healthy++;
      }
    }
    
    // Calculate health percentages
    performanceData.overallHealthPercentage = Math.round((performanceData.healthyServices / performanceData.totalServices) * 100);
    
    for (const tier of tiers) {
      const tierData = performanceData.tierHealth[tier];
      tierData.healthPercentage = tierData.total > 0 
        ? Math.round((tierData.healthy / tierData.total) * 100)
        : 0;
    }
    
    this.results.performance = performanceData;
    
    console.log(`  📊 Overall Health: ${performanceData.overallHealthPercentage}% (${performanceData.healthyServices}/${performanceData.totalServices})`);
    console.log(`  💾 Total Memory Usage: ${performanceData.resourceUsage.totalMemoryMB} MB`);
    console.log(`  🖥️ Total CPU Usage: ${Math.round(performanceData.resourceUsage.totalCPU)}%`);
  }

  // Perform basic load testing
  async performLoadTest() {
    const criticalServices = Object.entries(SERVICE_CONFIG)
      .filter(([_, config]) => config.tier === 'critical')
      .filter(([serviceName, _]) => this.results.services[serviceName]?.healthEndpoint?.status === 'healthy');
    
    console.log(`  🚀 Testing ${criticalServices.length} critical services under load...`);
    
    const loadTestResults = {};
    
    for (const [serviceName, config] of criticalServices) {
      console.log(`    Testing ${serviceName}...`);
      
      const testResult = await this.performServiceLoadTest(serviceName, config);
      loadTestResults[serviceName] = testResult;
      
      const status = testResult.averageResponseTime < 1000 ? '✅' : '⚠️';
      console.log(`    ${status} ${serviceName}: Avg ${Math.round(testResult.averageResponseTime)}ms, Success ${testResult.successRate}%`);
    }
    
    this.results.loadTest = loadTestResults;
  }

  // Load test individual service
  async performServiceLoadTest(serviceName, config) {
    const url = `http://localhost:${config.port}${config.healthEndpoint}`;
    const concurrency = 5;
    const requestsPerWorker = 10;
    const results = [];
    
    const workers = Array(concurrency).fill().map(async () => {
      const workerResults = [];
      
      for (let i = 0; i < requestsPerWorker; i++) {
        const startTime = Date.now();
        try {
          await this.makeHttpRequest(url, 3000); // 3 second timeout
          workerResults.push({
            success: true,
            responseTime: Date.now() - startTime
          });
        } catch (error) {
          workerResults.push({
            success: false,
            responseTime: Date.now() - startTime,
            error: error.message
          });
        }
        
        // Small delay between requests
        await this.sleep(100);
      }
      
      return workerResults;
    });
    
    const allResults = (await Promise.all(workers)).flat();
    const successful = allResults.filter(r => r.success).length;
    const responseTimes = allResults.map(r => r.responseTime);
    
    return {
      totalRequests: allResults.length,
      successfulRequests: successful,
      successRate: Math.round((successful / allResults.length) * 100),
      averageResponseTime: responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length,
      minResponseTime: Math.min(...responseTimes),
      maxResponseTime: Math.max(...responseTimes)
    };
  }

  // Generate comprehensive report
  async generateReport() {
    const report = {
      testSummary: this.generateTestSummary(),
      serviceTierAnalysis: this.generateTierAnalysis(),
      criticalIssues: this.identifyCriticalIssues(),
      recommendations: this.generateRecommendations(),
      performanceMetrics: this.results.performance,
      detailedResults: this.results
    };
    
    // Save report to file
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const reportPath = `/tmp/media-server-health-report-${timestamp}.json`;
    
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    // Display summary
    this.displayReportSummary(report);
    
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
    
    return report;
  }

  // Helper methods
  async isProcessRunning(processName) {
    try {
      const command = this.isContainer 
        ? `docker exec ${this.containerName} pgrep -f "${processName}"`
        : `pgrep -f "${processName}"`;
        
      await execAsync(command);
      return true;
    } catch (error) {
      return false;
    }
  }

  async isPortListening(port) {
    return new Promise((resolve) => {
      const socket = require('net').createConnection(port, 'localhost');
      
      socket.on('connect', () => {
        socket.destroy();
        resolve(true);
      });
      
      socket.on('error', () => {
        resolve(false);
      });
      
      setTimeout(() => {
        socket.destroy();
        resolve(false);
      }, 2000);
    });
  }

  async getProcessDetails(processName) {
    try {
      const command = this.isContainer
        ? `docker exec ${this.containerName} ps aux | grep "${processName}" | grep -v grep`
        : `ps aux | grep "${processName}" | grep -v grep`;
        
      const { stdout } = await execAsync(command);
      return stdout.trim() || null;
    } catch (error) {
      return null;
    }
  }

  async getResourceUsage(processName) {
    try {
      const command = this.isContainer
        ? `docker exec ${this.containerName} ps -C "${processName}" -o %cpu,%mem,pid,command --no-headers`
        : `ps -C "${processName}" -o %cpu,%mem,pid,command --no-headers`;
        
      const { stdout } = await execAsync(command);
      
      if (stdout.trim()) {
        const lines = stdout.trim().split('\n');
        const firstProcess = lines[0].trim().split(/\s+/);
        
        return {
          cpuPercent: parseFloat(firstProcess[0]) || 0,
          memoryPercent: parseFloat(firstProcess[1]) || 0,
          memoryMB: this.calculateMemoryMB(firstProcess[1]),
          pid: firstProcess[2]
        };
      }
    } catch (error) {
      // Process not running
    }
    
    return null;
  }

  calculateMemoryMB(memPercent) {
    // Rough estimation based on system memory
    const systemMemoryGB = 8; // Default assumption
    const systemMemoryMB = systemMemoryGB * 1024;
    return Math.round((parseFloat(memPercent) / 100) * systemMemoryMB);
  }

  parseCpuUsage(cpuLine) {
    const match = cpuLine.match(/(\d+\.\d+)%?\s*us/);
    return match ? parseFloat(match[1]) : 'unknown';
  }

  parseMemoryUsage(memLine) {
    const match = memLine.match(/(\d+)\s*total,\s*(\d+)\s*free,\s*(\d+)\s*used/);
    if (match) {
      const [_, total, free, used] = match;
      const usedPercent = Math.round((parseInt(used) / parseInt(total)) * 100);
      return { total, free, used, usedPercent };
    }
    return 'unknown';
  }

  async testNetworkEndpoint(url) {
    try {
      await this.makeHttpRequest(url, 5000);
      return { status: 'success', url };
    } catch (error) {
      return { status: 'failed', url, error: error.message };
    }
  }

  makeHttpRequest(url, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const request = http.get(url, { timeout }, (response) => {
        response.on('data', () => {}); // Consume data
        response.on('end', () => resolve(response));
      });
      
      request.on('timeout', () => {
        request.destroy();
        reject(new Error('Request timeout'));
      });
      
      request.on('error', reject);
    });
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  generateTestSummary() {
    const performance = this.results.performance;
    const tierHealth = performance.tierHealth;
    
    return {
      timestamp: this.results.timestamp,
      deploymentType: this.isContainer ? 'single-container' : 'multi-container',
      totalServices: performance.totalServices,
      runningServices: performance.runningServices,
      healthyServices: performance.healthyServices,
      overallHealthPercentage: performance.overallHealthPercentage,
      criticalServicesHealth: tierHealth.critical.healthPercentage,
      resourceUsage: performance.resourceUsage
    };
  }

  generateTierAnalysis() {
    const analysis = {};
    
    Object.entries(this.results.performance.tierHealth).forEach(([tier, data]) => {
      analysis[tier] = {
        ...data,
        status: data.healthPercentage >= 90 ? 'excellent' :
                data.healthPercentage >= 75 ? 'good' :
                data.healthPercentage >= 50 ? 'warning' : 'critical'
      };
    });
    
    return analysis;
  }

  identifyCriticalIssues() {
    const issues = [];
    
    // Check critical services
    Object.entries(SERVICE_CONFIG).forEach(([serviceName, config]) => {
      if (config.tier === 'critical') {
        const serviceResult = this.results.services[serviceName];
        
        if (!serviceResult?.processRunning) {
          issues.push({
            severity: 'critical',
            service: serviceName,
            issue: 'Process not running',
            impact: 'Core functionality affected'
          });
        }
        
        if (!serviceResult?.portListening) {
          issues.push({
            severity: 'critical',
            service: serviceName,
            issue: 'Port not listening',
            impact: 'Service not accessible'
          });
        }
      }
    });
    
    // Check dependency issues
    Object.entries(this.results.dependencies || {}).forEach(([serviceName, depResult]) => {
      if (!depResult.allDependenciesHealthy) {
        issues.push({
          severity: 'high',
          service: serviceName,
          issue: 'Dependencies not satisfied',
          impact: 'Service may not function correctly'
        });
      }
    });
    
    return issues;
  }

  generateRecommendations() {
    const recommendations = [];
    const performance = this.results.performance;
    
    // Resource usage recommendations
    if (performance.resourceUsage.totalMemoryMB > 4000) {
      recommendations.push({
        type: 'performance',
        priority: 'medium',
        message: 'High memory usage detected - consider optimizing service configurations',
        action: 'Review memory limits and disable unused services'
      });
    }
    
    // Service tier recommendations
    if (performance.tierHealth.critical.healthPercentage < 100) {
      recommendations.push({
        type: 'reliability',
        priority: 'high',
        message: 'Critical services are not fully healthy',
        action: 'Investigate and fix critical service issues immediately'
      });
    }
    
    // Load test recommendations
    Object.entries(this.results.loadTest || {}).forEach(([serviceName, testResult]) => {
      if (testResult.averageResponseTime > 2000) {
        recommendations.push({
          type: 'performance',
          priority: 'medium',
          message: `${serviceName} has high response times under load`,
          action: 'Consider optimizing service configuration or increasing resources'
        });
      }
    });
    
    return recommendations;
  }

  displayReportSummary(report) {
    console.log('\n' + '='.repeat(60));
    console.log('📋 COMPREHENSIVE HEALTH TEST REPORT');
    console.log('='.repeat(60));
    
    const summary = report.testSummary;
    console.log(`📊 Overall Health: ${summary.overallHealthPercentage}% (${summary.healthyServices}/${summary.totalServices} services)`);
    console.log(`🚀 Deployment Type: ${summary.deploymentType}`);
    console.log(`⚡ Running Services: ${summary.runningServices}/${summary.totalServices}`);
    console.log(`🔴 Critical Services Health: ${summary.criticalServicesHealth}%`);
    
    console.log('\n📈 Resource Usage:');
    console.log(`  💾 Total Memory: ${summary.resourceUsage.totalMemoryMB} MB`);
    console.log(`  🖥️ Total CPU: ${Math.round(summary.resourceUsage.totalCPU)}%`);
    
    console.log('\n🎯 Service Tiers:');
    Object.entries(report.serviceTierAnalysis).forEach(([tier, data]) => {
      const emoji = data.status === 'excellent' ? '🟢' : 
                   data.status === 'good' ? '🟡' : 
                   data.status === 'warning' ? '🟠' : '🔴';
      console.log(`  ${emoji} ${tier.toUpperCase()}: ${data.healthPercentage}% (${data.healthy}/${data.total})`);
    });
    
    if (report.criticalIssues.length > 0) {
      console.log('\n🚨 Critical Issues:');
      report.criticalIssues.forEach(issue => {
        console.log(`  🔴 ${issue.service}: ${issue.issue}`);
      });
    } else {
      console.log('\n✅ No critical issues detected!');
    }
    
    if (report.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      report.recommendations.slice(0, 3).forEach(rec => {
        const emoji = rec.priority === 'high' ? '🔴' : '🟡';
        console.log(`  ${emoji} ${rec.message}`);
      });
    }
    
    console.log('\n' + '='.repeat(60));
  }
}

// CLI execution
if (require.main === module) {
  const tester = new ServiceHealthTester();
  
  tester.runComprehensiveTest()
    .then((results) => {
      const exitCode = results.performance.tierHealth.critical.healthPercentage === 100 ? 0 : 1;
      process.exit(exitCode);
    })
    .catch((error) => {
      console.error('❌ Test suite failed:', error);
      process.exit(2);
    });
}

module.exports = ServiceHealthTester;