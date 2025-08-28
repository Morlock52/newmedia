#!/usr/bin/env node

/**
 * Quick MCP Test Script
 * Starts MCP suite and runs basic tests
 */

const { spawn } = require('child_process');
const axios = require('axios');

class QuickMCPTest {
  constructor() {
    this.mcpProcess = null;
  }

  log(message, type = 'INFO') {
    const timestamp = new Date().toISOString();
    const emoji = type === 'PASS' ? '✅' : type === 'FAIL' ? '❌' : 'ℹ️';
    console.log(`${emoji} [${timestamp}] ${message}`);
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async startMCPSuite() {
    this.log('🚀 Starting MCP Suite...');
    
    this.mcpProcess = spawn('node', ['src/simple-index.js'], {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe']
    });

    this.mcpProcess.stdout.on('data', (data) => {
      console.log(`📡 MCP: ${data.toString().trim()}`);
    });

    this.mcpProcess.stderr.on('data', (data) => {
      console.error(`❌ MCP Error: ${data.toString().trim()}`);
    });

    this.mcpProcess.on('close', (code) => {
      this.log(`MCP process exited with code ${code}`);
    });

    // Wait for startup
    await this.sleep(8000);
  }

  async testEndpoint(url, name) {
    try {
      const response = await axios.get(url, { timeout: 5000 });
      this.log(`${name}: ${response.status} - ${JSON.stringify(response.data).substring(0, 100)}...`, 'PASS');
      return true;
    } catch (error) {
      this.log(`${name}: Failed - ${error.message}`, 'FAIL');
      return false;
    }
  }

  async testToolCall() {
    try {
      const response = await axios.post('http://localhost:3001/call/get_system_info', 
        { arguments: {} }, 
        { 
          timeout: 10000,
          headers: { 'Content-Type': 'application/json' }
        }
      );
      this.log(`Tool Call: ${response.status} - Success`, 'PASS');
      return true;
    } catch (error) {
      this.log(`Tool Call: Failed - ${error.message}`, 'FAIL');
      return false;
    }
  }

  async runTests() {
    await this.startMCPSuite();

    this.log('🧪 Running endpoint tests...');
    
    const results = [];
    results.push(await this.testEndpoint('http://localhost:8090/health', 'Main Dashboard Health'));
    results.push(await this.testEndpoint('http://localhost:3001/health', 'Jellyfin MCP Health'));
    results.push(await this.testEndpoint('http://localhost:3001/info', 'Jellyfin MCP Info'));
    results.push(await this.testEndpoint('http://localhost:3001/tools', 'Jellyfin MCP Tools'));
    results.push(await this.testEndpoint('http://localhost:8090/api/mcp/status', 'MCP Status'));
    results.push(await this.testToolCall());

    const passed = results.filter(r => r).length;
    const total = results.length;

    console.log('\n' + '='.repeat(50));
    this.log(`Test Results: ${passed}/${total} passed (${(passed/total*100).toFixed(1)}%)`);
    
    if (passed === total) {
      this.log('🎉 All tests passed! MCP suite is working correctly!', 'PASS');
    } else {
      this.log('⚠️ Some tests failed. MCP suite has issues.', 'FAIL');
    }

    console.log('\n🔗 Access Points:');
    console.log('  • Main Dashboard: http://localhost:8090');
    console.log('  • Jellyfin MCP: http://localhost:3001');
    console.log('  • MCP Tools: http://localhost:3001/tools');
    console.log('  • Health Check: http://localhost:8090/health');

    // Cleanup
    if (this.mcpProcess) {
      this.log('🛑 Stopping MCP suite...');
      this.mcpProcess.kill('SIGTERM');
    }

    setTimeout(() => {
      process.exit(passed === total ? 0 : 1);
    }, 2000);
  }
}

const tester = new QuickMCPTest();
tester.runTests().catch(error => {
  console.error('💥 Test failed:', error);
  process.exit(1);
});