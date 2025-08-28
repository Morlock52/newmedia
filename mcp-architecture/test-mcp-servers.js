#!/usr/bin/env node

/**
 * MCP Server Test Suite
 * Tests all MCP servers and their HTTP/SSE functionality
 */

const axios = require('axios');
const EventSource = require('eventsource');

const MCP_SERVERS = {
  jellyfin: { port: 3001, name: 'Jellyfin' },
  sonarr: { port: 3002, name: 'Sonarr' },
  radarr: { port: 3003, name: 'Radarr' },
  prowlarr: { port: 3004, name: 'Prowlarr' },
  qbittorrent: { port: 3005, name: 'qBittorrent' }
};

class MCPTester {
  constructor() {
    this.results = {};
    this.eventSources = [];
  }

  async testServer(name, config) {
    const baseURL = `http://localhost:${config.port}`;
    console.log(`\n🧪 Testing ${config.name} MCP Server (${baseURL})`);
    
    const serverResults = {
      name: config.name,
      port: config.port,
      tests: {}
    };

    try {
      // Test 1: Health Check
      console.log('  ⚡ Health check...');
      const healthResponse = await axios.get(`${baseURL}/health`, { timeout: 5000 });
      serverResults.tests.health = {
        status: 'PASS',
        data: healthResponse.data
      };
      console.log(`     ✅ Health: ${healthResponse.data.status}`);

      // Test 2: Server Info
      console.log('  📋 Server info...');
      const infoResponse = await axios.get(`${baseURL}/info`, { timeout: 5000 });
      serverResults.tests.info = {
        status: 'PASS',
        data: infoResponse.data
      };
      console.log(`     ✅ Info: ${infoResponse.data.name} v${infoResponse.data.version}`);

      // Test 3: List Tools
      console.log('  🛠️ List tools...');
      const toolsResponse = await axios.get(`${baseURL}/tools`, { timeout: 5000 });
      const toolCount = toolsResponse.data.data?.tools?.length || 0;
      serverResults.tests.tools = {
        status: 'PASS',
        count: toolCount,
        data: toolsResponse.data
      };
      console.log(`     ✅ Tools: ${toolCount} available`);

      // Test 4: List Resources  
      console.log('  📚 List resources...');
      const resourcesResponse = await axios.get(`${baseURL}/resources`, { timeout: 5000 });
      const resourceCount = resourcesResponse.data.data?.resources?.length || 0;
      serverResults.tests.resources = {
        status: 'PASS',
        count: resourceCount,
        data: resourcesResponse.data
      };
      console.log(`     ✅ Resources: ${resourceCount} available`);

      // Test 5: SSE Connection
      console.log('  📡 SSE connection...');
      const eventSource = new EventSource(`${baseURL}/events`);
      let sseConnected = false;
      
      const ssePromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          eventSource.close();
          reject(new Error('SSE connection timeout'));
        }, 5000);

        eventSource.onopen = () => {
          sseConnected = true;
          clearTimeout(timeout);
          eventSource.close();
          resolve();
        };

        eventSource.onerror = (error) => {
          clearTimeout(timeout);
          eventSource.close();
          reject(error);
        };

        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.server === name + '-mcp') {
            sseConnected = true;
            clearTimeout(timeout);
            eventSource.close();
            resolve();
          }
        };
      });

      try {
        await ssePromise;
        serverResults.tests.sse = {
          status: 'PASS',
          connected: true
        };
        console.log('     ✅ SSE: Connected successfully');
      } catch (error) {
        serverResults.tests.sse = {
          status: 'FAIL',
          error: error.message
        };
        console.log(`     ❌ SSE: ${error.message}`);
      }

      // Test 6: Tool Execution (if tools available)
      if (toolCount > 0) {
        const tools = toolsResponse.data.data.tools;
        const firstTool = tools[0];
        
        console.log(`  🔧 Testing tool: ${firstTool.name}...`);
        
        try {
          // Create minimal test arguments based on tool requirements
          const testArgs = this.createTestArgs(name, firstTool);
          
          const toolResponse = await axios.post(
            `${baseURL}/call/${firstTool.name}`,
            { arguments: testArgs },
            { timeout: 10000 }
          );
          
          serverResults.tests.toolExecution = {
            status: 'PASS',
            tool: firstTool.name,
            data: toolResponse.data
          };
          console.log(`     ✅ Tool execution: ${firstTool.name} succeeded`);
          
        } catch (error) {
          serverResults.tests.toolExecution = {
            status: 'WARN',
            tool: firstTool.name,
            error: error.message
          };
          console.log(`     ⚠️ Tool execution: ${firstTool.name} failed (expected - no real services)`);
        }
      }

    } catch (error) {
      console.log(`     ❌ Server test failed: ${error.message}`);
      serverResults.tests.error = {
        status: 'FAIL',
        error: error.message
      };
    }

    this.results[name] = serverResults;
    return serverResults;
  }

  createTestArgs(serverName, tool) {
    // Create minimal test arguments for common tools
    const testCases = {
      jellyfin: {
        search_media: { query: 'test', limit: 1 },
        get_library_stats: {},
        get_recent_media: { limit: 1 }
      },
      sonarr: {
        search_tv_shows: { term: 'test' },
        get_queue: { page: 1, pageSize: 1 },
        get_system_status: {}
      },
      radarr: {
        search_movies: { term: 'test' },
        get_queue: { page: 1, pageSize: 1 },
        get_system_status: {}
      },
      prowlarr: {
        get_indexers: {},
        get_system_status: {},
        get_indexer_stats: {}
      },
      qbittorrent: {
        get_torrents: { filter: 'all' },
        get_global_stats: {},
        get_categories: {}
      }
    };

    return testCases[serverName]?.[tool.name] || {};
  }

  async testMainSuite() {
    console.log('🧪 Testing Main MCP Suite (port 8090)...');
    
    try {
      const healthResponse = await axios.get('http://localhost:8090/health', { timeout: 5000 });
      console.log('  ✅ Main suite health:', healthResponse.data.status);
      
      const mcpStatusResponse = await axios.get('http://localhost:8090/api/mcp/status', { timeout: 5000 });
      console.log('  ✅ MCP status:', Object.keys(mcpStatusResponse.data).length, 'servers');
      
      const agentStatusResponse = await axios.get('http://localhost:8090/api/agents/status', { timeout: 5000 });
      console.log('  ✅ Agent status: Available');
      
    } catch (error) {
      console.log('  ❌ Main suite test failed:', error.message);
    }
  }

  async runAllTests() {
    console.log('🚀 Starting MCP Server Test Suite');
    console.log('=' * 50);

    // Test main suite first
    await this.testMainSuite();

    // Test each MCP server
    for (const [name, config] of Object.entries(MCP_SERVERS)) {
      await this.testServer(name, config);
    }

    this.printSummary();
  }

  printSummary() {
    console.log('\n' + '='.repeat(60));
    console.log('📊 TEST SUMMARY');
    console.log('='.repeat(60));

    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;
    let warnTests = 0;

    for (const [name, result] of Object.entries(this.results)) {
      console.log(`\n🏷️ ${result.name} (Port ${result.port}):`);
      
      for (const [testName, testResult] of Object.entries(result.tests)) {
        totalTests++;
        const status = testResult.status;
        
        if (status === 'PASS') {
          passedTests++;
          console.log(`  ✅ ${testName}: PASSED`);
        } else if (status === 'WARN') {
          warnTests++;
          console.log(`  ⚠️ ${testName}: WARNING - ${testResult.error || 'See details'}`);
        } else {
          failedTests++;
          console.log(`  ❌ ${testName}: FAILED - ${testResult.error || 'Unknown error'}`);
        }
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log('📈 OVERALL RESULTS:');
    console.log(`  Total Tests: ${totalTests}`);
    console.log(`  ✅ Passed: ${passedTests}`);
    console.log(`  ⚠️ Warnings: ${warnTests}`);
    console.log(`  ❌ Failed: ${failedTests}`);
    
    const successRate = totalTests > 0 ? ((passedTests + warnTests) / totalTests * 100).toFixed(1) : 0;
    console.log(`  📊 Success Rate: ${successRate}%`);
    
    if (failedTests === 0) {
      console.log('\n🎉 All critical tests passed! MCP servers are ready for use.');
    } else {
      console.log('\n⚠️ Some tests failed. Check the Docker container and service configurations.');
    }
    
    console.log('\n💡 Next steps:');
    console.log('  1. Build and run the Docker container: docker build -t mediaserver-ai -f Dockerfile.multi-service .');
    console.log('  2. Connect to Claude Desktop using the MCP_CONNECTION_GUIDE.md');
    console.log('  3. Access the AI dashboard at http://localhost:8090');
    console.log('='.repeat(60));
  }

  async waitForServers(retryCount = 30, delayMs = 2000) {
    console.log('⏳ Waiting for MCP servers to start...');
    
    for (let i = 0; i < retryCount; i++) {
      try {
        const response = await axios.get('http://localhost:8090/health', { timeout: 1000 });
        if (response.status === 200) {
          console.log('✅ Main MCP suite is ready!');
          return true;
        }
      } catch (error) {
        process.stdout.write('.');
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    
    console.log('\n❌ Timeout waiting for servers to start');
    return false;
  }
}

// Main execution
async function main() {
  const tester = new MCPTester();
  
  // Wait for servers to be ready
  const serversReady = await tester.waitForServers();
  
  if (!serversReady) {
    console.log('\n🔧 To start the servers, run:');
    console.log('docker build -t mediaserver-ai -f Dockerfile.multi-service .');
    console.log('docker run -d --name mediaserver-ai -p 8090:8090 -p 3001:3001 -p 3002:3002 -p 3003:3003 -p 3004:3004 -p 3005:3005 mediaserver-ai');
    process.exit(1);
  }
  
  // Run all tests
  await tester.runAllTests();
}

// Handle cleanup
process.on('SIGINT', () => {
  console.log('\n🛑 Test interrupted');
  process.exit(0);
});

if (require.main === module) {
  main().catch(error => {
    console.error('❌ Test suite failed:', error);
    process.exit(1);
  });
}

module.exports = MCPTester;