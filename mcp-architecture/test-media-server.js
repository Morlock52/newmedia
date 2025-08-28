#!/usr/bin/env node

/**
 * Comprehensive Media Server MCP Test Suite
 * Tests all tools, resources, and prompts
 */

const { spawn } = require('child_process');

class MediaServerTester {
  constructor() {
    this.tests = [];
    this.results = { passed: 0, failed: 0, total: 0 };
  }

  async runTest(name, method, params = {}) {
    console.log(`🧪 Testing: ${name}`);
    
    const request = {
      method,
      params,
      jsonrpc: '2.0',
      id: this.tests.length + 1
    };

    try {
      const result = await this.sendRequest(request);
      
      if (result.error) {
        console.log(`❌ ${name}: ${result.error.message}`);
        this.results.failed++;
        return false;
      } else {
        console.log(`✅ ${name}: Success`);
        this.results.passed++;
        return true;
      }
    } catch (error) {
      console.log(`❌ ${name}: ${error.message}`);
      this.results.failed++;
      return false;
    } finally {
      this.results.total++;
    }
  }

  sendRequest(request) {
    return new Promise((resolve, reject) => {
      const server = spawn('./mcp-node-wrapper.sh', ['proper-media-server-mcp.js'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let errorOutput = '';

      server.stdout.on('data', (data) => {
        output += data.toString();
      });

      server.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      server.on('close', (code) => {
        try {
          const lines = output.trim().split('\n');
          const lastLine = lines[lines.length - 1];
          const response = JSON.parse(lastLine);
          resolve(response);
        } catch (error) {
          reject(new Error(`Parse error: ${error.message}`));
        }
      });

      server.on('error', reject);

      // Send request and close stdin
      server.stdin.write(JSON.stringify(request) + '\n');
      server.stdin.end();

      // Timeout after 5 seconds
      setTimeout(() => {
        server.kill();
        reject(new Error('Test timeout'));
      }, 5000);
    });
  }

  async runAllTests() {
    console.log('🚀 Starting Media Server MCP Test Suite\n');

    // Test initialization
    await this.runTest('Server Initialize', 'initialize', {
      protocolVersion: '2025-06-18',
      capabilities: {},
      clientInfo: { name: 'test', version: '1.0' }
    });

    // Test tools list
    await this.runTest('Tools List', 'tools/list');

    // Test each tool
    await this.runTest('Search Media Tool', 'tools/call', {
      name: 'search_media',
      arguments: { query: 'matrix', type: 'movie' }
    });

    await this.runTest('Library Stats Tool', 'tools/call', {
      name: 'get_library_stats',
      arguments: {}
    });

    await this.runTest('Recent Media Tool', 'tools/call', {
      name: 'get_recent_media',
      arguments: { limit: 3 }
    });

    await this.runTest('System Info Tool', 'tools/call', {
      name: 'get_system_info',
      arguments: {}
    });

    // Test resources
    await this.runTest('Resources List', 'resources/list');
    
    await this.runTest('Library Resource', 'resources/read', {
      uri: 'media://library'
    });

    await this.runTest('Stats Resource', 'resources/read', {
      uri: 'media://stats'
    });

    // Test prompts
    await this.runTest('Prompts List', 'prompts/list');

    await this.runTest('Media Search Assistant Prompt', 'prompts/get', {
      name: 'media_search_assistant',
      arguments: { media_type: 'movie', genre: 'sci-fi' }
    });

    await this.runTest('Library Organizer Prompt', 'prompts/get', {
      name: 'library_organizer',
      arguments: { library_size: 'large' }
    });

    // Summary
    console.log('\n📊 Test Results Summary:');
    console.log(`✅ Passed: ${this.results.passed}`);
    console.log(`❌ Failed: ${this.results.failed}`);
    console.log(`📈 Total: ${this.results.total}`);
    console.log(`🎯 Success Rate: ${Math.round((this.results.passed / this.results.total) * 100)}%`);

    if (this.results.failed === 0) {
      console.log('\n🎉 All tests passed! Media server is working as designed.');
    } else {
      console.log('\n⚠️ Some tests failed. Check the error messages above.');
    }
  }
}

// Run tests
const tester = new MediaServerTester();
tester.runAllTests().catch(console.error);