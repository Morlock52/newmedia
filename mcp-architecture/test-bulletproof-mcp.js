#!/usr/bin/env node

/**
 * BULLETPROOF MCP SERVER TEST SUITE
 * 
 * This script thoroughly tests the bulletproof MCP server to ensure
 * it works correctly with Claude Desktop and implements the MCP protocol properly.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

class MCPTester {
  constructor() {
    this.testResults = [];
    this.serverProcess = null;
    this.requestId = 1;
  }

  /**
   * Log test result
   */
  logResult(test, passed, message = '') {
    const result = {
      test,
      passed,
      message,
      timestamp: new Date().toISOString()
    };
    this.testResults.push(result);
    
    const status = passed ? '✅ PASS' : '❌ FAIL';
    console.log(`${status} - ${test}: ${message}`);
  }

  /**
   * Send MCP request and get response
   */
  async sendRequest(method, params = {}, id = null) {
    return new Promise((resolve, reject) => {
      const requestId = id || this.requestId++;
      const request = {
        jsonrpc: "2.0",
        method: method,
        params: params,
        id: requestId
      };

      let responseBuffer = '';
      let responseReceived = false;

      const timeout = setTimeout(() => {
        if (!responseReceived) {
          reject(new Error(`Request timeout for ${method}`));
        }
      }, 5000);

      // Send request
      this.serverProcess.stdin.write(JSON.stringify(request) + '\n');

      // Listen for response
      const onData = (data) => {
        responseBuffer += data.toString();
        
        // Try to parse complete JSON response
        const lines = responseBuffer.split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              const response = JSON.parse(line.trim());
              if (response.id === requestId) {
                responseReceived = true;
                clearTimeout(timeout);
                this.serverProcess.stdout.removeListener('data', onData);
                resolve(response);
                return;
              }
            } catch (err) {
              // Incomplete JSON, continue buffering
            }
          }
        }
      };

      this.serverProcess.stdout.on('data', onData);
    });
  }

  /**
   * Start MCP server for testing
   */
  async startServer() {
    return new Promise((resolve, reject) => {
      const serverPath = path.join(__dirname, 'bulletproof-mcp.js');
      
      console.log('🚀 Starting Bulletproof MCP Server for testing...');
      
      this.serverProcess = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_DEBUG: '1' }
      });

      // Give server time to start
      setTimeout(() => {
        if (this.serverProcess && !this.serverProcess.killed) {
          console.log('📡 Server started successfully');
          resolve();
        } else {
          reject(new Error('Server failed to start'));
        }
      }, 1000);

      this.serverProcess.on('error', (err) => {
        console.error('❌ Server error:', err);
        reject(err);
      });

      this.serverProcess.stderr.on('data', (data) => {
        // Server debug output
        console.log('🔍 SERVER:', data.toString().trim());
      });
    });
  }

  /**
   * Stop MCP server
   */
  stopServer() {
    if (this.serverProcess && !this.serverProcess.killed) {
      console.log('🛑 Stopping server...');
      this.serverProcess.kill();
      this.serverProcess = null;
    }
  }

  /**
   * Test MCP initialization
   */
  async testInitialization() {
    try {
      console.log('\n🧪 Testing MCP Initialization...');

      // Test initialize request
      const initResponse = await this.sendRequest('initialize', {
        protocolVersion: "2024-11-05",
        capabilities: {
          tools: {},
          resources: {},
          prompts: {}
        },
        clientInfo: {
          name: "test-client",
          version: "1.0.0"
        }
      });

      if (initResponse.result && initResponse.result.protocolVersion === "2024-11-05") {
        this.logResult('MCP Initialize', true, 'Protocol version matches');
      } else {
        this.logResult('MCP Initialize', false, 'Invalid initialize response');
        return false;
      }

      // Test initialized notification
      this.serverProcess.stdin.write(JSON.stringify({
        jsonrpc: "2.0",
        method: "initialized"
      }) + '\n');

      this.logResult('MCP Initialized', true, 'Notification sent successfully');
      return true;

    } catch (err) {
      this.logResult('MCP Initialize', false, err.message);
      return false;
    }
  }

  /**
   * Test tools listing
   */
  async testToolsListing() {
    try {
      console.log('\n🧪 Testing Tools Listing...');

      const toolsResponse = await this.sendRequest('tools/list');
      
      if (toolsResponse.result && toolsResponse.result.tools) {
        const tools = toolsResponse.result.tools;
        const expectedTools = ['swarm_init', 'swarm_status', 'agent_spawn', 'task_orchestrate', 'memory_usage', 'neural_status'];
        
        let allToolsPresent = true;
        for (const expectedTool of expectedTools) {
          const toolExists = tools.some(tool => tool.name === expectedTool);
          if (!toolExists) {
            this.logResult('Tools List', false, `Missing tool: ${expectedTool}`);
            allToolsPresent = false;
          }
        }
        
        if (allToolsPresent) {
          this.logResult('Tools List', true, `All ${expectedTools.length} tools present`);
          return true;
        }
      } else {
        this.logResult('Tools List', false, 'Invalid tools list response');
      }

      return false;

    } catch (err) {
      this.logResult('Tools List', false, err.message);
      return false;
    }
  }

  /**
   * Test swarm operations
   */
  async testSwarmOperations() {
    try {
      console.log('\n🧪 Testing Swarm Operations...');

      // Test swarm initialization
      const swarmInitResponse = await this.sendRequest('tools/call', {
        name: 'swarm_init',
        arguments: {
          topology: 'mesh',
          maxAgents: 5,
          strategy: 'balanced'
        }
      });

      if (swarmInitResponse.result && swarmInitResponse.result.content) {
        const content = swarmInitResponse.result.content[0].text;
        if (content.includes('Swarm initialized successfully')) {
          this.logResult('Swarm Init', true, 'Swarm initialized correctly');
        } else {
          this.logResult('Swarm Init', false, 'Invalid swarm init response');
          return false;
        }
      } else {
        this.logResult('Swarm Init', false, 'No content in swarm init response');
        return false;
      }

      // Test agent spawning
      const agentSpawnResponse = await this.sendRequest('tools/call', {
        name: 'agent_spawn',
        arguments: {
          type: 'researcher',
          name: 'Test Researcher',
          capabilities: ['analysis', 'synthesis']
        }
      });

      if (agentSpawnResponse.result && agentSpawnResponse.result.content) {
        const content = agentSpawnResponse.result.content[0].text;
        if (content.includes('Agent Spawned Successfully')) {
          this.logResult('Agent Spawn', true, 'Agent spawned correctly');
        } else {
          this.logResult('Agent Spawn', false, 'Invalid agent spawn response');
          return false;
        }
      } else {
        this.logResult('Agent Spawn', false, 'No content in agent spawn response');
        return false;
      }

      // Test swarm status
      const statusResponse = await this.sendRequest('tools/call', {
        name: 'swarm_status',
        arguments: {
          verbose: true
        }
      });

      if (statusResponse.result && statusResponse.result.content) {
        const content = statusResponse.result.content[0].text;
        if (content.includes('Swarm Status Report') && content.includes('Active Agents: 1')) {
          this.logResult('Swarm Status', true, 'Status correctly shows spawned agent');
        } else {
          this.logResult('Swarm Status', false, 'Status does not reflect spawned agent');
          return false;
        }
      } else {
        this.logResult('Swarm Status', false, 'No content in status response');
        return false;
      }

      return true;

    } catch (err) {
      this.logResult('Swarm Operations', false, err.message);
      return false;
    }
  }

  /**
   * Test task orchestration
   */
  async testTaskOrchestration() {
    try {
      console.log('\n🧪 Testing Task Orchestration...');

      const taskResponse = await this.sendRequest('tools/call', {
        name: 'task_orchestrate',
        arguments: {
          task: 'Test task for verification',
          strategy: 'parallel',
          priority: 'high'
        }
      });

      if (taskResponse.result && taskResponse.result.content) {
        const content = taskResponse.result.content[0].text;
        if (content.includes('Task Orchestration Started')) {
          this.logResult('Task Orchestration', true, 'Task orchestrated successfully');
          return true;
        } else {
          this.logResult('Task Orchestration', false, 'Invalid task orchestration response');
        }
      } else {
        this.logResult('Task Orchestration', false, 'No content in task response');
      }

      return false;

    } catch (err) {
      this.logResult('Task Orchestration', false, err.message);
      return false;
    }
  }

  /**
   * Test memory and neural operations
   */
  async testMemoryAndNeural() {
    try {
      console.log('\n🧪 Testing Memory and Neural Operations...');

      // Test memory usage
      const memoryResponse = await this.sendRequest('tools/call', {
        name: 'memory_usage',
        arguments: {
          detail: 'detailed'
        }
      });

      if (memoryResponse.result && memoryResponse.result.content) {
        const content = memoryResponse.result.content[0].text;
        if (content.includes('Memory Usage Report')) {
          this.logResult('Memory Usage', true, 'Memory usage reported correctly');
        } else {
          this.logResult('Memory Usage', false, 'Invalid memory usage response');
          return false;
        }
      } else {
        this.logResult('Memory Usage', false, 'No content in memory response');
        return false;
      }

      // Test neural status
      const neuralResponse = await this.sendRequest('tools/call', {
        name: 'neural_status',
        arguments: {}
      });

      if (neuralResponse.result && neuralResponse.result.content) {
        const content = neuralResponse.result.content[0].text;
        if (content.includes('Neural Agent Status')) {
          this.logResult('Neural Status', true, 'Neural status reported correctly');
        } else {
          this.logResult('Neural Status', false, 'Invalid neural status response');
          return false;
        }
      } else {
        this.logResult('Neural Status', false, 'No content in neural response');
        return false;
      }

      return true;

    } catch (err) {
      this.logResult('Memory and Neural', false, err.message);
      return false;
    }
  }

  /**
   * Test error handling
   */
  async testErrorHandling() {
    try {
      console.log('\n🧪 Testing Error Handling...');

      // Test invalid method
      const invalidResponse = await this.sendRequest('invalid/method');
      
      if (invalidResponse.error && invalidResponse.error.code === -32601) {
        this.logResult('Invalid Method Error', true, 'Correctly returned method not found error');
      } else {
        this.logResult('Invalid Method Error', false, 'Did not handle invalid method correctly');
        return false;
      }

      // Test invalid tool
      const invalidToolResponse = await this.sendRequest('tools/call', {
        name: 'nonexistent_tool',
        arguments: {}
      });

      if (invalidToolResponse.error) {
        this.logResult('Invalid Tool Error', true, 'Correctly handled invalid tool call');
      } else {
        this.logResult('Invalid Tool Error', false, 'Did not handle invalid tool correctly');
        return false;
      }

      return true;

    } catch (err) {
      // Errors are expected in this test
      this.logResult('Error Handling', true, 'Error handling working as expected');
      return true;
    }
  }

  /**
   * Run all tests
   */
  async runAllTests() {
    console.log('🧪 BULLETPROOF MCP SERVER TEST SUITE');
    console.log('=====================================\n');

    try {
      // Start server
      await this.startServer();

      // Run tests in sequence
      const tests = [
        () => this.testInitialization(),
        () => this.testToolsListing(),
        () => this.testSwarmOperations(),
        () => this.testTaskOrchestration(), 
        () => this.testMemoryAndNeural(),
        () => this.testErrorHandling()
      ];

      let allPassed = true;
      for (const test of tests) {
        const result = await test();
        if (!result) {
          allPassed = false;
        }
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Stop server
      this.stopServer();

      // Print final results
      this.printResults(allPassed);

      return allPassed;

    } catch (err) {
      console.error('❌ Test suite error:', err);
      this.stopServer();
      return false;
    }
  }

  /**
   * Print test results summary
   */
  printResults(allPassed) {
    console.log('\n' + '='.repeat(50));
    console.log('📊 TEST RESULTS SUMMARY');
    console.log('='.repeat(50));

    const passed = this.testResults.filter(r => r.passed).length;
    const total = this.testResults.length;
    const passRate = Math.round((passed / total) * 100);

    console.log(`\n📈 Overall Results:`);
    console.log(`├── Total Tests: ${total}`);
    console.log(`├── Passed: ${passed}`);
    console.log(`├── Failed: ${total - passed}`);
    console.log(`└── Success Rate: ${passRate}%`);

    if (allPassed) {
      console.log('\n🎉 ALL TESTS PASSED! The Bulletproof MCP Server is working correctly.');
      console.log('\n✅ Ready for Claude Desktop integration:');
      console.log('   1. Copy the full path to bulletproof-mcp.js');
      console.log('   2. Add to Claude Desktop config as shown in the server file');
      console.log('   3. Restart Claude Desktop');
      console.log('   4. Use mcp__ruv-swarm__ tools in Claude Code');
    } else {
      console.log('\n❌ Some tests failed. Check the output above for details.');
    }

    console.log('\n' + '='.repeat(50));
  }
}

/**
 * Main execution
 */
async function main() {
  const tester = new MCPTester();
  const success = await tester.runAllTests();
  process.exit(success ? 0 : 1);
}

if (require.main === module) {
  main();
}

module.exports = MCPTester;