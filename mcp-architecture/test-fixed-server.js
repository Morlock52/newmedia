#!/usr/bin/env node

/**
 * Test script for the fixed MCP server
 * Simulates Claude Desktop's communication with the server
 */

const { spawn } = require('child_process');
const readline = require('readline');

class MCPServerTester {
  constructor() {
    this.requestId = 0;
    this.pendingRequests = new Map();
  }

  log(message) {
    console.log(`[TESTER] ${new Date().toISOString()} ${message}`);
  }

  sendRequest(method, params = {}) {
    const id = ++this.requestId;
    const request = {
      jsonrpc: '2.0',
      id: id,
      method: method,
      params: params
    };
    
    this.log(`Sending: ${JSON.stringify(request)}`);
    this.serverProcess.stdin.write(JSON.stringify(request) + '\n');
    
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      
      // Timeout after 5 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${id} timed out`));
        }
      }, 5000);
    });
  }

  async runTests() {
    this.log('Starting Fixed MCP Server for testing...');
    
    // Start the server process
    this.serverProcess = spawn('node', ['fixed-mcp-server.js'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, MCP_DEBUG: 'true' }
    });
    
    // Handle server stdout (JSON-RPC responses)
    const rl = readline.createInterface({
      input: this.serverProcess.stdout,
      terminal: false
    });
    
    rl.on('line', (line) => {
      try {
        const response = JSON.parse(line);
        this.log(`Received: ${JSON.stringify(response)}`);
        
        if (response.id && this.pendingRequests.has(response.id)) {
          const { resolve, reject } = this.pendingRequests.get(response.id);
          this.pendingRequests.delete(response.id);
          
          if (response.error) {
            reject(new Error(`Server error: ${response.error.message}`));
          } else {
            resolve(response.result);
          }
        }
      } catch (error) {
        this.log(`Failed to parse server response: ${error.message}`);
      }
    });
    
    // Handle server stderr (debug output)
    this.serverProcess.stderr.on('data', (data) => {
      console.error(`[SERVER-DEBUG] ${data.toString().trim()}`);
    });
    
    // Handle server exit
    this.serverProcess.on('exit', (code) => {
      this.log(`Server exited with code ${code}`);
    });
    
    // Wait a bit for server to start
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      // Test 1: Initialize
      this.log('\n=== Test 1: Initialize ===');
      const initResult = await this.sendRequest('initialize', {
        clientInfo: { name: 'test-client', version: '1.0.0' }
      });
      this.log(`✅ Initialize successful: ${JSON.stringify(initResult, null, 2)}`);
      
      // Test 2: List tools
      this.log('\n=== Test 2: List Tools ===');
      const toolsResult = await this.sendRequest('tools/list');
      this.log(`✅ Tools listed: ${JSON.stringify(toolsResult, null, 2)}`);
      
      // Test 3: Call a tool
      this.log('\n=== Test 3: Call Tool ===');
      const toolResult = await this.sendRequest('tools/call', {
        name: 'test_echo',
        arguments: { message: 'Hello from test!' }
      });
      this.log(`✅ Tool called: ${JSON.stringify(toolResult, null, 2)}`);
      
      // Test 4: List resources
      this.log('\n=== Test 4: List Resources ===');
      const resourcesResult = await this.sendRequest('resources/list');
      this.log(`✅ Resources listed: ${JSON.stringify(resourcesResult, null, 2)}`);
      
      // Test 5: Read a resource
      this.log('\n=== Test 5: Read Resource ===');
      const resourceResult = await this.sendRequest('resources/read', {
        uri: 'test://status'
      });
      this.log(`✅ Resource read: ${JSON.stringify(resourceResult, null, 2)}`);
      
      // Test 6: Invalid method
      this.log('\n=== Test 6: Invalid Method (should error) ===');
      try {
        await this.sendRequest('invalid/method');
        this.log('❌ Should have thrown an error');
      } catch (error) {
        this.log(`✅ Got expected error: ${error.message}`);
      }
      
      // Test 7: Get server status
      this.log('\n=== Test 7: Get Server Status ===');
      const statusResult = await this.sendRequest('tools/call', {
        name: 'get_status',
        arguments: {}
      });
      this.log(`✅ Status retrieved: ${JSON.stringify(statusResult, null, 2)}`);
      
      this.log('\n🎉 All tests passed!');
      
    } catch (error) {
      this.log(`❌ Test failed: ${error.message}`);
    } finally {
      // Clean up
      this.log('\nClosing server...');
      this.serverProcess.kill();
    }
  }
}

// Run the tests
const tester = new MCPServerTester();
tester.runTests().catch(error => {
  console.error('Test runner failed:', error);
  process.exit(1);
});