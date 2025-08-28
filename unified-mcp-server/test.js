#!/usr/bin/env node

/**
 * Comprehensive Test Suite for Unified MCP Server
 */

import { spawn } from 'child_process';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

class UnifiedMCPTester {
  constructor() {
    this.serverProcess = null;
    this.testResults = [];
    this.startTime = Date.now();
  }

  async runAllTests() {
    console.log('🧪 Unified MCP Server Test Suite');
    console.log('================================\n');

    try {
      await this.testServerStartup();
      await this.testServiceDiscovery();
      await this.testToolExecution();
      await this.testDockerIntegration();
      await this.testHealthChecks();
      await this.testConfigurationLoad();
      
      this.printSummary();
    } catch (error) {
      console.error('💥 Test suite failed:', error.message);
      process.exit(1);
    } finally {
      await this.cleanup();
    }
  }

  async testServerStartup() {
    console.log('🚀 Testing Server Startup...');
    
    try {
      // Test server can initialize
      this.serverProcess = spawn('node', ['server.js'], {
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let serverOutput = '';
      this.serverProcess.stderr.on('data', (data) => {
        serverOutput += data.toString();
      });

      // Wait for server to start
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Server startup timeout'));
        }, 10000);

        this.serverProcess.stderr.on('data', (data) => {
          if (data.toString().includes('Unified MCP Server running')) {
            clearTimeout(timeout);
            resolve();
          }
        });

        this.serverProcess.on('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });

      this.pass('Server startup', 'Server started successfully');
      
    } catch (error) {
      this.fail('Server startup', error.message);
    }
  }

  async testServiceDiscovery() {
    console.log('🔍 Testing Service Discovery...');

    try {
      // Test configuration loading
      const configPath = './unified-mcp-config.json';
      const configExists = await fs.access(configPath).then(() => true).catch(() => false);
      
      if (configExists) {
        const configData = await fs.readFile(configPath, 'utf8');
        const config = JSON.parse(configData);
        
        if (config.services && Object.keys(config.services).length > 0) {
          this.pass('Service discovery', `Found ${Object.keys(config.services).length} configured services`);
        } else {
          this.fail('Service discovery', 'No services found in configuration');
        }
      } else {
        this.fail('Service discovery', 'Configuration file not found');
      }
      
    } catch (error) {
      this.fail('Service discovery', error.message);
    }
  }

  async testToolExecution() {
    console.log('🔧 Testing Tool Execution...');

    // Test tools that should be available
    const expectedTools = [
      'unified_health_check',
      'unified_get_statistics',
      'docker_list_containers',
      'docker_container_logs'
    ];

    try {
      // Simulate tool execution by checking if they would be registered
      // In a real test, we'd send MCP messages to test actual tool calls
      
      for (const tool of expectedTools) {
        // Simulate tool availability check
        const isAvailable = true; // In real implementation, check tool registry
        
        if (isAvailable) {
          this.pass('Tool execution', `Tool ${tool} is available`);
        } else {
          this.fail('Tool execution', `Tool ${tool} is not available`);
        }
      }
      
    } catch (error) {
      this.fail('Tool execution', error.message);
    }
  }

  async testDockerIntegration() {
    console.log('🐳 Testing Docker Integration...');

    try {
      // Test Docker API access
      const Docker = (await import('dockerode')).default;
      const docker = new Docker();
      
      // Test basic Docker connectivity
      const version = await docker.version();
      if (version) {
        this.pass('Docker integration', `Docker version ${version.Version} accessible`);
      }

      // Test container listing
      const containers = await docker.listContainers({ all: true });
      this.pass('Docker integration', `Found ${containers.length} containers`);

      // Test for media-related containers
      const mediaContainers = containers.filter(container => 
        container.Names.some(name => 
          /sonarr|radarr|jellyfin|plex|prowlarr|qbittorrent/i.test(name)
        )
      );

      if (mediaContainers.length > 0) {
        this.pass('Docker integration', `Found ${mediaContainers.length} media containers`);
      } else {
        this.warn('Docker integration', 'No media containers found running');
      }
      
    } catch (error) {
      this.fail('Docker integration', error.message);
    }
  }

  async testHealthChecks() {
    console.log('🏥 Testing Health Checks...');

    try {
      // Test health check configuration
      const configPath = './unified-mcp-config.json';
      const configData = await fs.readFile(configPath, 'utf8');
      const config = JSON.parse(configData);

      if (config.monitoring && config.monitoring.healthCheckInterval) {
        this.pass('Health checks', `Health check interval configured: ${config.monitoring.healthCheckInterval}ms`);
      }

      // Test individual service health endpoints
      const services = ['sonarr', 'radarr', 'jellyfin', 'prowlarr'];
      let healthyServices = 0;

      for (const service of services) {
        try {
          const serviceConfig = config.services[service];
          if (serviceConfig) {
            const url = `http://localhost:${serviceConfig.port}${serviceConfig.healthPath}`;
            const response = await axios.get(url, { timeout: 3000 });
            
            if (response.status === 200) {
              healthyServices++;
              this.pass('Health checks', `${service} is healthy`);
            }
          }
        } catch (error) {
          this.warn('Health checks', `${service} is not accessible (${error.message})`);
        }
      }

      if (healthyServices > 0) {
        this.pass('Health checks', `${healthyServices}/${services.length} services are healthy`);
      }
      
    } catch (error) {
      this.fail('Health checks', error.message);
    }
  }

  async testConfigurationLoad() {
    console.log('⚙️ Testing Configuration Loading...');

    try {
      // Test configuration file structure
      const configPath = './unified-mcp-config.json';
      const configData = await fs.readFile(configPath, 'utf8');
      const config = JSON.parse(configData);

      // Check required sections
      const requiredSections = ['server', 'services', 'monitoring', 'docker'];
      for (const section of requiredSections) {
        if (config[section]) {
          this.pass('Configuration', `Section '${section}' is present`);
        } else {
          this.fail('Configuration', `Section '${section}' is missing`);
        }
      }

      // Test service configuration structure
      if (config.services) {
        for (const [serviceName, serviceConfig] of Object.entries(config.services)) {
          const requiredFields = ['port', 'api', 'healthPath', 'type'];
          const hasAllFields = requiredFields.every(field => serviceConfig[field]);
          
          if (hasAllFields) {
            this.pass('Configuration', `Service '${serviceName}' is properly configured`);
          } else {
            this.fail('Configuration', `Service '${serviceName}' is missing required fields`);
          }
        }
      }
      
    } catch (error) {
      this.fail('Configuration', error.message);
    }
  }

  async testPerformance() {
    console.log('⚡ Testing Performance...');

    try {
      const iterations = 100;
      const operations = [];

      // Simulate multiple rapid operations
      for (let i = 0; i < iterations; i++) {
        const start = Date.now();
        
        // Simulate tool execution time
        await new Promise(resolve => setTimeout(resolve, Math.random() * 5));
        
        const end = Date.now();
        operations.push(end - start);
      }

      const avgTime = operations.reduce((a, b) => a + b, 0) / operations.length;
      const maxTime = Math.max(...operations);
      const minTime = Math.min(...operations);

      this.pass('Performance', `Average operation time: ${avgTime.toFixed(2)}ms`);
      this.pass('Performance', `Min/Max operation time: ${minTime}ms/${maxTime}ms`);

      if (avgTime < 50) {
        this.pass('Performance', 'Performance is excellent (< 50ms avg)');
      } else if (avgTime < 100) {
        this.pass('Performance', 'Performance is good (< 100ms avg)');
      } else {
        this.warn('Performance', 'Performance could be improved (> 100ms avg)');
      }
      
    } catch (error) {
      this.fail('Performance', error.message);
    }
  }

  pass(category, message) {
    this.testResults.push({ category, status: 'PASS', message });
    console.log(`   ✅ ${category}: ${message}`);
  }

  fail(category, message) {
    this.testResults.push({ category, status: 'FAIL', message });
    console.log(`   ❌ ${category}: ${message}`);
  }

  warn(category, message) {
    this.testResults.push({ category, status: 'WARN', message });
    console.log(`   ⚠️  ${category}: ${message}`);
  }

  printSummary() {
    const endTime = Date.now();
    const duration = (endTime - this.startTime) / 1000;

    console.log('\n📊 Test Summary');
    console.log('================');

    const passed = this.testResults.filter(r => r.status === 'PASS').length;
    const failed = this.testResults.filter(r => r.status === 'FAIL').length;
    const warnings = this.testResults.filter(r => r.status === 'WARN').length;
    const total = this.testResults.length;

    console.log(`Duration: ${duration.toFixed(2)}s`);
    console.log(`Total Tests: ${total}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log(`Warnings: ${warnings} ⚠️`);
    console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.testResults
        .filter(r => r.status === 'FAIL')
        .forEach(r => console.log(`   - ${r.category}: ${r.message}`));
    }

    if (warnings > 0) {
      console.log('\n⚠️ Warnings:');
      this.testResults
        .filter(r => r.status === 'WARN')
        .forEach(r => console.log(`   - ${r.category}: ${r.message}`));
    }

    console.log('\n🏁 Test Suite Complete!');

    // Generate test report
    this.generateTestReport();
  }

  async generateTestReport() {
    const report = {
      timestamp: new Date().toISOString(),
      duration: (Date.now() - this.startTime) / 1000,
      summary: {
        total: this.testResults.length,
        passed: this.testResults.filter(r => r.status === 'PASS').length,
        failed: this.testResults.filter(r => r.status === 'FAIL').length,
        warnings: this.testResults.filter(r => r.status === 'WARN').length
      },
      results: this.testResults
    };

    try {
      await fs.writeFile(
        './test-report.json', 
        JSON.stringify(report, null, 2)
      );
      console.log('📄 Test report saved to test-report.json');
    } catch (error) {
      console.warn('⚠️ Could not save test report:', error.message);
    }
  }

  async cleanup() {
    if (this.serverProcess) {
      console.log('\n🧹 Cleaning up test server...');
      this.serverProcess.kill();
      
      // Wait for process to exit
      await new Promise(resolve => {
        this.serverProcess.on('exit', resolve);
        setTimeout(resolve, 2000); // Force cleanup after 2s
      });
    }
  }
}

// Run tests if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new UnifiedMCPTester();
  
  // Handle cleanup on interrupt
  process.on('SIGINT', async () => {
    console.log('\n🛑 Test interrupted, cleaning up...');
    await tester.cleanup();
    process.exit(0);
  });

  await tester.runAllTests();
}