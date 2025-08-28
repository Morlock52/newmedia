#!/usr/bin/env node

/**
 * Test script for the Media Services MCP Server
 * Verifies MCP protocol compliance and functionality
 */

const { spawn } = require('child_process');
const readline = require('readline');

class MCPTester {
  constructor() {
    this.server = null;
    this.testId = 0;
    this.responses = new Map();
  }
  
  log(message) {
    console.log(`[TEST] ${new Date().toISOString()} ${message}`);
  }
  
  error(message) {
    console.error(`[ERROR] ${new Date().toISOString()} ${message}`);
  }
  
  async startServer() {
    this.log('Starting MCP Media Server...');
    
    this.server = spawn('node', ['mcp-media-server.js'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: process.cwd()
    });
    
    this.server.stderr.on('data', (data) => {
      this.log(`Server stderr: ${data.toString()}`);
    });
    
    this.server.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(line => line.trim());
      for (const line of lines) {
        try {
          const response = JSON.parse(line);
          this.log(`Received: ${JSON.stringify(response, null, 2)}`);
          if (response.id) {
            this.responses.set(response.id, response);
          }
        } catch (error) {
          this.log(`Non-JSON response: ${line}`);
        }
      }
    });
    
    this.server.on('exit', (code) => {
      this.log(`Server exited with code: ${code}`);
    });
    
    this.server.on('error', (error) => {
      this.error(`Server error: ${error.message}`);
    });
    
    // Wait a moment for server to start
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  async sendRequest(method, params = {}) {
    const id = ++this.testId;
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };
    
    this.log(`Sending: ${JSON.stringify(request)}`);
    this.server.stdin.write(JSON.stringify(request) + '\n');
    
    // Wait for response
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Timeout waiting for response to ${method}`));
      }, 5000);
      
      const checkResponse = () => {
        if (this.responses.has(id)) {
          clearTimeout(timeout);
          const response = this.responses.get(id);
          this.responses.delete(id);
          resolve(response);
        } else {
          setTimeout(checkResponse, 100);
        }
      };
      
      checkResponse();
    });
  }
  
  async testInitialize() {
    this.log('Testing initialize...');
    try {
      const response = await this.sendRequest('initialize', {
        protocolVersion: '1.0',
        capabilities: {},
        clientInfo: { name: 'test-client', version: '1.0.0' }
      });
      
      if (response.result && response.result.serverInfo) {
        this.log('✅ Initialize test passed');
        return true;
      } else {
        this.error('❌ Initialize test failed: Invalid response structure');
        return false;
      }
    } catch (error) {
      this.error(`❌ Initialize test failed: ${error.message}`);
      return false;
    }
  }
  
  async testToolsList() {
    this.log('Testing tools/list...');
    try {
      const response = await this.sendRequest('tools/list');
      
      if (response.result && Array.isArray(response.result.tools)) {
        this.log(`✅ Tools list test passed: Found ${response.result.tools.length} tools`);
        this.log(`Tools: ${response.result.tools.map(t => t.name).join(', ')}`);
        return true;
      } else {
        this.error('❌ Tools list test failed: Invalid response structure');
        return false;
      }
    } catch (error) {
      this.error(`❌ Tools list test failed: ${error.message}`);
      return false;
    }
  }
  
  async testResourcesList() {
    this.log('Testing resources/list...');
    try {
      const response = await this.sendRequest('resources/list');
      
      if (response.result && Array.isArray(response.result.resources)) {
        this.log(`✅ Resources list test passed: Found ${response.result.resources.length} resources`);
        return true;
      } else {
        this.error('❌ Resources list test failed: Invalid response structure');
        return false;
      }
    } catch (error) {
      this.error(`❌ Resources list test failed: ${error.message}`);
      return false;
    }
  }
  
  async testToolCall() {
    this.log('Testing tools/call...');
    try {
      const response = await this.sendRequest('tools/call', {
        name: 'get_system_status',
        arguments: { detailed: false }
      });
      
      if (response.result && response.result.content) {
        this.log('✅ Tool call test passed');
        return true;
      } else {
        this.error('❌ Tool call test failed: Invalid response structure');
        return false;
      }
    } catch (error) {
      this.error(`❌ Tool call test failed: ${error.message}`);
      return false;
    }
  }
  
  async testResourceRead() {
    this.log('Testing resources/read...');
    try {
      const response = await this.sendRequest('resources/read', {
        uri: 'media://config/services'
      });
      
      if (response.result && response.result.contents) {
        this.log('✅ Resource read test passed');
        return true;
      } else {
        this.error('❌ Resource read test failed: Invalid response structure');
        return false;
      }
    } catch (error) {
      this.error(`❌ Resource read test failed: ${error.message}`);
      return false;
    }
  }
  
  async runAllTests() {
    this.log('=== Starting MCP Media Server Tests ===');
    
    try {
      await this.startServer();
      
      const tests = [
        this.testInitialize(),
        this.testToolsList(),
        this.testResourcesList(),
        this.testToolCall(),
        this.testResourceRead()
      ];
      
      const results = await Promise.all(tests);
      const passed = results.filter(r => r).length;
      const total = results.length;
      
      this.log(`=== Test Results: ${passed}/${total} passed ===`);
      
      if (passed === total) {
        this.log('🎉 All tests passed! MCP server is working correctly.');
        return true;
      } else {
        this.error(`❌ ${total - passed} tests failed`);
        return false;
      }
      
    } catch (error) {
      this.error(`Test suite failed: ${error.message}`);
      return false;
    } finally {
      if (this.server) {
        this.server.kill();
      }
    }
  }
}

// Run tests if executed directly
if (require.main === module) {
  const tester = new MCPTester();
  tester.runAllTests().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}

module.exports = MCPTester;