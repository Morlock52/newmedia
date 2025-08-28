#!/usr/bin/env node

/**
 * Comprehensive MCP Suite Test Script
 * Tests all MCP servers locally without Docker
 */

const axios = require('axios');
const EventSource = require('eventsource');

// Import our simple MCP suite
const path = require('path');
// Check if mcp-architecture directory exists
const fs = require('fs');
const mcpPath = path.join(__dirname, 'mcp-architecture', 'src', 'simple-index');
console.log('🔍 Looking for MCP suite at:', mcpPath);
console.log('🔍 Directory exists:', fs.existsSync(path.dirname(mcpPath)));
console.log('🔍 File exists:', fs.existsSync(mcpPath + '.js'));

// Let's try relative import
const SimpleMediaServerSuite = require('./mcp-architecture/src/simple-index');

class MCPTester {
  constructor() {
    this.suite = null;
    this.testResults = {
      passed: 0,
      failed: 0,
      total: 0,
      details: []
    };
  }

  log(message, type = 'INFO') {
    const timestamp = new Date().toISOString();
    const emoji = type === 'PASS' ? '✅' : type === 'FAIL' ? '❌' : type === 'WARN' ? '⚠️' : 'ℹ️';
    console.log(`${emoji} [${timestamp}] ${message}`);
  }

  async test(name, testFn) {
    this.testResults.total++;
    try {
      await testFn();
      this.testResults.passed++;
      this.testResults.details.push({ name, status: 'PASS', error: null });
      this.log(`${name} - PASSED`, 'PASS');
    } catch (error) {
      this.testResults.failed++;
      this.testResults.details.push({ name, status: 'FAIL', error: error.message });
      this.log(`${name} - FAILED: ${error.message}`, 'FAIL');
    }
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async startMCPSuite() {
    this.log('🚀 Starting MCP Suite locally...');
    this.suite = new SimpleMediaServerSuite();
    
    // Start in background
    setTimeout(() => {
      this.suite.start().catch(error => {
        this.log(`Failed to start MCP suite: ${error.message}`, 'FAIL');
      });
    }, 100);

    // Wait for services to start
    await this.sleep(5000);
    this.log('⏳ Waiting for services to initialize...');
    await this.sleep(5000);
  }

  async testMainDashboard() {
    await this.test('Main Dashboard Health Check', async () => {
      const response = await axios.get('http://localhost:8090/health', {
        timeout: 5000
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.status || response.data.status !== 'healthy') {
        throw new Error(`Expected healthy status, got ${response.data.status}`);
      }
    });
  }

  async testJellyfinMCP() {
    await this.test('Jellyfin MCP Health Check', async () => {
      const response = await axios.get('http://localhost:3001/health', {
        timeout: 5000
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.status || response.data.status !== 'healthy') {
        throw new Error(`Expected healthy status, got ${response.data.status}`);
      }
    });

    await this.test('Jellyfin MCP Info Endpoint', async () => {
      const response = await axios.get('http://localhost:3001/info', {
        timeout: 5000
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.name) {
        throw new Error('Missing server name in info response');
      }
    });

    await this.test('Jellyfin MCP Tools List', async () => {
      const response = await axios.get('http://localhost:3001/tools', {
        timeout: 5000
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.data || !response.data.data.tools) {
        throw new Error('Missing tools in response');
      }
      
      const tools = response.data.data.tools;
      const expectedTools = ['search_media', 'get_library_stats', 'get_recent_media', 'get_system_info'];
      
      for (const toolName of expectedTools) {
        const tool = tools.find(t => t.name === toolName);
        if (!tool) {
          throw new Error(`Missing expected tool: ${toolName}`);
        }
      }
      
      this.log(`📋 Found ${tools.length} tools: ${tools.map(t => t.name).join(', ')}`);
    });
  }

  async testToolCalling() {
    await this.test('Tool Call - Get System Info', async () => {
      const response = await axios.post('http://localhost:3001/call/get_system_info', {
        arguments: {}
      }, {
        timeout: 10000,
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.success) {
        throw new Error(`Tool call failed: ${response.data.error || 'Unknown error'}`);
      }
      
      const content = response.data.data.content;
      if (!content || !Array.isArray(content) || content.length === 0) {
        throw new Error('Missing or invalid content in tool response');
      }
      
      this.log(`🔧 System info tool returned: ${content[0].text.substring(0, 100)}...`);
    });

    await this.test('Tool Call - Search Media', async () => {
      const response = await axios.post('http://localhost:3001/call/search_media', {
        arguments: {
          query: 'test',
          limit: 5
        }
      }, {
        timeout: 10000,
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.success) {
        // This might fail if no Jellyfin server is running, which is expected
        this.log(`⚠️ Search tool failed (expected if no Jellyfin server): ${response.data.error}`, 'WARN');
        return;
      }
      
      this.log(`🔍 Search tool executed successfully`);
    });
  }

  async testSSEStreaming() {
    await this.test('Server-Sent Events Stream', async () => {
      return new Promise((resolve, reject) => {
        const eventSource = new EventSource('http://localhost:3001/events');
        let receivedConnection = false;
        
        const timeout = setTimeout(() => {
          eventSource.close();
          if (!receivedConnection) {
            reject(new Error('Did not receive connection event within 5 seconds'));
          }
        }, 5000);
        
        eventSource.onopen = () => {
          this.log('📡 SSE connection opened');
        };
        
        eventSource.addEventListener('connected', (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.server && data.timestamp && data.clientId) {
              receivedConnection = true;
              this.log(`📨 Received connection event for server: ${data.server}`);
              clearTimeout(timeout);
              eventSource.close();
              resolve();
            } else {
              reject(new Error('Invalid connection event data'));
            }
          } catch (error) {
            reject(new Error(`Failed to parse connection event: ${error.message}`));
          }
        });
        
        eventSource.onerror = (error) => {
          clearTimeout(timeout);
          eventSource.close();
          reject(new Error(`SSE error: ${error.message || 'Connection failed'}`));
        };
      });
    });
  }

  async testChatEndpoint() {
    await this.test('Chat API Endpoint', async () => {
      const response = await axios.post('http://localhost:8090/api/chat', {
        message: 'Hello MCP test'
      }, {
        timeout: 5000,
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.response || !response.data.agent) {
        throw new Error('Missing response or agent in chat response');
      }
      
      this.log(`💬 Chat response: ${response.data.response}`);
    });
  }

  async testMCPStatus() {
    await this.test('MCP Status Endpoint', async () => {
      const response = await axios.get('http://localhost:8090/api/mcp/status', {
        timeout: 5000
      });
      
      if (response.status !== 200) {
        throw new Error(`Expected status 200, got ${response.status}`);
      }
      
      if (!response.data.jellyfin) {
        throw new Error('Missing jellyfin status');
      }
      
      this.log(`📊 MCP Status: ${JSON.stringify(response.data, null, 2)}`);
    });
  }

  async runAllTests() {
    this.log('🧪 Starting comprehensive MCP suite tests...');
    
    try {
      // Start the MCP suite
      await this.startMCPSuite();
      
      // Test main dashboard
      await this.testMainDashboard();
      
      // Test Jellyfin MCP
      await this.testJellyfinMCP();
      
      // Test tool calling
      await this.testToolCalling();
      
      // Test SSE streaming
      await this.testSSEStreaming();
      
      // Test chat endpoint
      await this.testChatEndpoint();
      
      // Test MCP status
      await this.testMCPStatus();
      
    } catch (error) {
      this.log(`💥 Critical test failure: ${error.message}`, 'FAIL');
    }
    
    // Print final results
    this.printResults();
  }

  printResults() {
    console.log('\n' + '='.repeat(60));
    console.log('🏁 TEST RESULTS SUMMARY');
    console.log('='.repeat(60));
    console.log(`📊 Total Tests: ${this.testResults.total}`);
    console.log(`✅ Passed: ${this.testResults.passed}`);
    console.log(`❌ Failed: ${this.testResults.failed}`);
    console.log(`📈 Success Rate: ${((this.testResults.passed / this.testResults.total) * 100).toFixed(1)}%`);
    
    if (this.testResults.failed > 0) {
      console.log('\n❌ FAILED TESTS:');
      this.testResults.details
        .filter(test => test.status === 'FAIL')
        .forEach(test => {
          console.log(`  • ${test.name}: ${test.error}`);
        });
    }
    
    console.log('\n✅ PASSED TESTS:');
    this.testResults.details
      .filter(test => test.status === 'PASS')
      .forEach(test => {
        console.log(`  • ${test.name}`);
      });
    
    console.log('\n' + '='.repeat(60));
    
    if (this.testResults.passed === this.testResults.total) {
      console.log('🎉 ALL TESTS PASSED! MCP suite is working correctly.');
    } else {
      console.log('⚠️  Some tests failed. Check the details above.');
    }
    
    console.log('\n🔗 Access Points:');
    console.log('  • Main Dashboard: http://localhost:8090');
    console.log('  • Jellyfin MCP: http://localhost:3001');
    console.log('  • Health Check: http://localhost:8090/health');
    console.log('  • Tools List: http://localhost:3001/tools');
    console.log('  • SSE Stream: http://localhost:3001/events');
    
    // Exit after displaying results
    setTimeout(() => {
      process.exit(this.testResults.failed > 0 ? 1 : 0);
    }, 2000);
  }
}

// Run the tests
const tester = new MCPTester();
tester.runAllTests().catch(error => {
  console.error('💥 Test runner failed:', error);
  process.exit(1);
});