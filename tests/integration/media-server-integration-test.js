#!/usr/bin/env node

/**
 * Media Server Integration Test Suite - 2025
 * Comprehensive testing for media server Docker stack
 * Based on research findings from MEDIA_SERVER_INTEGRATION_RESEARCH_2025.md
 */

const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

// Configuration
const CONFIG = {
  services: {
    jellyfin: { url: 'http://localhost:8096', healthPath: '/health', timeout: 30000 },
    plex: { url: 'http://localhost:32400', healthPath: '/identity', timeout: 30000 },
    sonarr: { url: 'http://localhost:8989', healthPath: '/ping', timeout: 15000 },
    radarr: { url: 'http://localhost:7878', healthPath: '/ping', timeout: 15000 },
    prowlarr: { url: 'http://localhost:9696', healthPath: '/ping', timeout: 15000 },
    jellyseerr: { url: 'http://localhost:5055', healthPath: '/api/v1/status', timeout: 15000 },
    qbittorrent: { url: 'http://localhost:8080', healthPath: '/', timeout: 10000 },
    prometheus: { url: 'http://localhost:9090', healthPath: '/-/healthy', timeout: 10000 },
    grafana: { url: 'http://localhost:3000', healthPath: '/api/health', timeout: 10000 },
    portainer: { url: 'http://localhost:9000', healthPath: '/api/status', timeout: 10000 }
  },
  testResults: {
    passed: 0,
    failed: 0,
    errors: []
  }
};

class MediaServerTester {
  constructor() {
    this.results = {
      healthChecks: {},
      apiTests: {},
      integrationTests: {},
      performanceTests: {}
    };
  }

  /**
   * Run all test suites
   */
  async runAllTests() {
    console.log('🚀 Starting Media Server Integration Tests - 2025');
    console.log('=' .repeat(60));

    try {
      await this.testHealthChecks();
      await this.testAPIConnectivity();
      await this.testServiceIntegrations();
      await this.testPerformanceMetrics();
      await this.generateReport();
    } catch (error) {
      console.error('❌ Test suite failed:', error.message);
      process.exit(1);
    }
  }

  /**
   * Test health check endpoints for all services
   */
  async testHealthChecks() {
    console.log('\n🔍 Testing Service Health Checks...');
    
    for (const [serviceName, config] of Object.entries(CONFIG.services)) {
      try {
        const startTime = Date.now();
        const response = await axios.get(`${config.url}${config.healthPath}`, {
          timeout: config.timeout,
          validateStatus: (status) => status < 500
        });
        const responseTime = Date.now() - startTime;

        const isHealthy = response.status >= 200 && response.status < 400;
        
        this.results.healthChecks[serviceName] = {
          status: isHealthy ? 'HEALTHY' : 'UNHEALTHY',
          statusCode: response.status,
          responseTime: responseTime,
          error: null
        };

        console.log(`  ${isHealthy ? '✅' : '❌'} ${serviceName}: ${response.status} (${responseTime}ms)`);
        
        if (isHealthy) CONFIG.testResults.passed++;
        else CONFIG.testResults.failed++;

      } catch (error) {
        this.results.healthChecks[serviceName] = {
          status: 'ERROR',
          statusCode: null,
          responseTime: null,
          error: error.message
        };
        
        console.log(`  ❌ ${serviceName}: ERROR - ${error.message}`);
        CONFIG.testResults.failed++;
        CONFIG.testResults.errors.push(`${serviceName}: ${error.message}`);
      }
    }
  }

  /**
   * Test API connectivity and basic functionality
   */
  async testAPIConnectivity() {
    console.log('\n🔌 Testing API Connectivity...');

    // Test Jellyfin System Info
    await this.testJellyfinAPI();
    
    // Test Sonarr System Status
    await this.testSonarrAPI();
    
    // Test Radarr System Status
    await this.testRadarrAPI();
    
    // Test Prowlarr Indexers
    await this.testProwlarrAPI();
    
    // Test Prometheus Metrics
    await this.testPrometheusAPI();
  }

  async testJellyfinAPI() {
    try {
      console.log('  🧪 Testing Jellyfin API...');
      
      // Test system info endpoint
      const systemInfo = await axios.get('http://localhost:8096/System/Info', {
        timeout: 10000
      });
      
      // Test library info
      const libraries = await axios.get('http://localhost:8096/Library/VirtualFolders', {
        timeout: 10000
      });

      this.results.apiTests.jellyfin = {
        systemInfo: systemInfo.status === 200,
        libraries: libraries.status === 200,
        version: systemInfo.data.Version,
        serverName: systemInfo.data.ServerName
      };

      console.log(`    ✅ System Info: ${systemInfo.status}`);
      console.log(`    ✅ Libraries: ${libraries.status}`);
      console.log(`    📋 Version: ${systemInfo.data.Version}`);
      
      CONFIG.testResults.passed += 2;

    } catch (error) {
      console.log(`    ❌ Jellyfin API Error: ${error.message}`);
      this.results.apiTests.jellyfin = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  async testSonarrAPI() {
    try {
      console.log('  🧪 Testing Sonarr API...');
      
      const systemStatus = await axios.get('http://localhost:8989/api/v3/system/status', {
        timeout: 10000,
        headers: { 'X-Api-Key': process.env.SONARR_API_KEY || '' }
      });

      this.results.apiTests.sonarr = {
        systemStatus: systemStatus.status === 200,
        version: systemStatus.data.version,
        isNetCore: systemStatus.data.isNetCore
      };

      console.log(`    ✅ System Status: ${systemStatus.status}`);
      console.log(`    📋 Version: ${systemStatus.data.version}`);
      
      CONFIG.testResults.passed++;

    } catch (error) {
      console.log(`    ❌ Sonarr API Error: ${error.message}`);
      this.results.apiTests.sonarr = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  async testRadarrAPI() {
    try {
      console.log('  🧪 Testing Radarr API...');
      
      const systemStatus = await axios.get('http://localhost:7878/api/v3/system/status', {
        timeout: 10000,
        headers: { 'X-Api-Key': process.env.RADARR_API_KEY || '' }
      });

      this.results.apiTests.radarr = {
        systemStatus: systemStatus.status === 200,
        version: systemStatus.data.version,
        isNetCore: systemStatus.data.isNetCore
      };

      console.log(`    ✅ System Status: ${systemStatus.status}`);
      console.log(`    📋 Version: ${systemStatus.data.version}`);
      
      CONFIG.testResults.passed++;

    } catch (error) {
      console.log(`    ❌ Radarr API Error: ${error.message}`);
      this.results.apiTests.radarr = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  async testProwlarrAPI() {
    try {
      console.log('  🧪 Testing Prowlarr API...');
      
      const systemStatus = await axios.get('http://localhost:9696/api/v1/system/status', {
        timeout: 10000,
        headers: { 'X-Api-Key': process.env.PROWLARR_API_KEY || '' }
      });

      const indexers = await axios.get('http://localhost:9696/api/v1/indexer', {
        timeout: 10000,
        headers: { 'X-Api-Key': process.env.PROWLARR_API_KEY || '' }
      });

      this.results.apiTests.prowlarr = {
        systemStatus: systemStatus.status === 200,
        indexers: indexers.status === 200,
        indexerCount: indexers.data.length,
        version: systemStatus.data.version
      };

      console.log(`    ✅ System Status: ${systemStatus.status}`);
      console.log(`    ✅ Indexers: ${indexers.data.length} configured`);
      
      CONFIG.testResults.passed += 2;

    } catch (error) {
      console.log(`    ❌ Prowlarr API Error: ${error.message}`);
      this.results.apiTests.prowlarr = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  async testPrometheusAPI() {
    try {
      console.log('  🧪 Testing Prometheus API...');
      
      const targets = await axios.get('http://localhost:9090/api/v1/targets', {
        timeout: 10000
      });

      const metrics = await axios.get('http://localhost:9090/api/v1/query?query=up', {
        timeout: 10000
      });

      this.results.apiTests.prometheus = {
        targets: targets.status === 200,
        metrics: metrics.status === 200,
        activeTargets: targets.data.data.activeTargets.length,
        upServices: metrics.data.data.result.length
      };

      console.log(`    ✅ Targets: ${targets.data.data.activeTargets.length} active`);
      console.log(`    ✅ Metrics: ${metrics.data.data.result.length} services up`);
      
      CONFIG.testResults.passed += 2;

    } catch (error) {
      console.log(`    ❌ Prometheus API Error: ${error.message}`);
      this.results.apiTests.prometheus = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  /**
   * Test service-to-service integrations
   */
  async testServiceIntegrations() {
    console.log('\n🔗 Testing Service Integrations...');

    // Test Prowlarr -> Sonarr/Radarr integration
    await this.testProwlarrIntegration();
    
    // Test Download client integration
    await this.testDownloadClientIntegration();
    
    // Test Media server -> Request service integration
    await this.testRequestServiceIntegration();
  }

  async testProwlarrIntegration() {
    try {
      console.log('  🧪 Testing Prowlarr Integration...');
      
      // Check if Prowlarr can sync to Sonarr/Radarr
      const apps = await axios.get('http://localhost:9696/api/v1/applications', {
        timeout: 10000,
        headers: { 'X-Api-Key': process.env.PROWLARR_API_KEY || '' }
      });

      this.results.integrationTests.prowlarr = {
        applications: apps.data.length,
        sonarrConnected: apps.data.some(app => app.name.toLowerCase().includes('sonarr')),
        radarrConnected: apps.data.some(app => app.name.toLowerCase().includes('radarr'))
      };

      console.log(`    ✅ Connected Applications: ${apps.data.length}`);
      console.log(`    ${this.results.integrationTests.prowlarr.sonarrConnected ? '✅' : '❌'} Sonarr Integration`);
      console.log(`    ${this.results.integrationTests.prowlarr.radarrConnected ? '✅' : '❌'} Radarr Integration`);
      
      CONFIG.testResults.passed++;

    } catch (error) {
      console.log(`    ❌ Prowlarr Integration Error: ${error.message}`);
      this.results.integrationTests.prowlarr = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  async testDownloadClientIntegration() {
    try {
      console.log('  🧪 Testing Download Client Integration...');
      
      // Test qBittorrent accessibility
      const qbtResponse = await axios.get('http://localhost:8080', {
        timeout: 10000,
        validateStatus: () => true
      });

      this.results.integrationTests.downloadClients = {
        qbittorrent: qbtResponse.status < 500
      };

      console.log(`    ${qbtResponse.status < 500 ? '✅' : '❌'} qBittorrent: ${qbtResponse.status}`);
      
      CONFIG.testResults.passed++;

    } catch (error) {
      console.log(`    ❌ Download Client Error: ${error.message}`);
      this.results.integrationTests.downloadClients = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  async testRequestServiceIntegration() {
    try {
      console.log('  🧪 Testing Request Service Integration...');
      
      const jellyseerrStatus = await axios.get('http://localhost:5055/api/v1/status', {
        timeout: 10000
      });

      this.results.integrationTests.requestServices = {
        jellyseerr: jellyseerrStatus.status === 200,
        version: jellyseerrStatus.data.version
      };

      console.log(`    ✅ Jellyseerr: ${jellyseerrStatus.status}`);
      console.log(`    📋 Version: ${jellyseerrStatus.data.version}`);
      
      CONFIG.testResults.passed++;

    } catch (error) {
      console.log(`    ❌ Request Service Error: ${error.message}`);
      this.results.integrationTests.requestServices = { error: error.message };
      CONFIG.testResults.failed++;
    }
  }

  /**
   * Test performance metrics
   */
  async testPerformanceMetrics() {
    console.log('\n⚡ Testing Performance Metrics...');

    const performanceTests = [
      { name: 'Jellyfin Response Time', url: 'http://localhost:8096/health' },
      { name: 'Sonarr Response Time', url: 'http://localhost:8989/ping' },
      { name: 'Radarr Response Time', url: 'http://localhost:7878/ping' },
      { name: 'Prowlarr Response Time', url: 'http://localhost:9696/ping' }
    ];

    for (const test of performanceTests) {
      try {
        const startTime = Date.now();
        await axios.get(test.url, { timeout: 5000 });
        const responseTime = Date.now() - startTime;

        this.results.performanceTests[test.name] = {
          responseTime: responseTime,
          status: responseTime < 2000 ? 'GOOD' : responseTime < 5000 ? 'ACCEPTABLE' : 'SLOW'
        };

        const status = responseTime < 2000 ? '🟢' : responseTime < 5000 ? '🟡' : '🔴';
        console.log(`  ${status} ${test.name}: ${responseTime}ms`);
        
        CONFIG.testResults.passed++;

      } catch (error) {
        this.results.performanceTests[test.name] = {
          responseTime: null,
          status: 'ERROR',
          error: error.message
        };
        console.log(`  ❌ ${test.name}: ${error.message}`);
        CONFIG.testResults.failed++;
      }
    }
  }

  /**
   * Generate comprehensive test report
   */
  async generateReport() {
    console.log('\n📊 Generating Test Report...');

    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalTests: CONFIG.testResults.passed + CONFIG.testResults.failed,
        passed: CONFIG.testResults.passed,
        failed: CONFIG.testResults.failed,
        successRate: `${((CONFIG.testResults.passed / (CONFIG.testResults.passed + CONFIG.testResults.failed)) * 100).toFixed(1)}%`
      },
      results: this.results,
      errors: CONFIG.testResults.errors,
      recommendations: this.generateRecommendations()
    };

    // Save report to file
    const reportPath = path.join(__dirname, `../reports/integration-test-report-${Date.now()}.json`);
    
    try {
      await fs.mkdir(path.dirname(reportPath), { recursive: true });
      await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
      console.log(`📁 Report saved to: ${reportPath}`);
    } catch (error) {
      console.log(`❌ Failed to save report: ${error.message}`);
    }

    // Display summary
    console.log('\n' + '='.repeat(60));
    console.log('📈 TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total Tests: ${report.summary.totalTests}`);
    console.log(`Passed: ${report.summary.passed}`);
    console.log(`Failed: ${report.summary.failed}`);
    console.log(`Success Rate: ${report.summary.successRate}`);

    if (CONFIG.testResults.errors.length > 0) {
      console.log('\n🚨 ERRORS:');
      CONFIG.testResults.errors.forEach(error => console.log(`  - ${error}`));
    }

    console.log('\n💡 RECOMMENDATIONS:');
    report.recommendations.forEach(rec => console.log(`  - ${rec}`));

    // Exit with appropriate code
    process.exit(CONFIG.testResults.failed > 0 ? 1 : 0);
  }

  /**
   * Generate recommendations based on test results
   */
  generateRecommendations() {
    const recommendations = [];

    // Check for failed health checks
    Object.entries(this.results.healthChecks).forEach(([service, result]) => {
      if (result.status !== 'HEALTHY') {
        recommendations.push(`Fix ${service} service - currently ${result.status}`);
      }
      if (result.responseTime > 5000) {
        recommendations.push(`Optimize ${service} performance - response time: ${result.responseTime}ms`);
      }
    });

    // Check for API failures
    Object.entries(this.results.apiTests).forEach(([service, result]) => {
      if (result.error) {
        recommendations.push(`Resolve ${service} API connectivity issues`);
      }
    });

    // Check for integration issues
    if (this.results.integrationTests.prowlarr && !this.results.integrationTests.prowlarr.sonarrConnected) {
      recommendations.push('Configure Prowlarr -> Sonarr integration');
    }
    if (this.results.integrationTests.prowlarr && !this.results.integrationTests.prowlarr.radarrConnected) {
      recommendations.push('Configure Prowlarr -> Radarr integration');
    }

    // Performance recommendations
    Object.entries(this.results.performanceTests).forEach(([test, result]) => {
      if (result.status === 'SLOW') {
        recommendations.push(`Optimize performance for ${test}`);
      }
    });

    if (recommendations.length === 0) {
      recommendations.push('All services are healthy and performing well!');
    }

    return recommendations;
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const tester = new MediaServerTester();
  tester.runAllTests().catch(error => {
    console.error('Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = MediaServerTester;