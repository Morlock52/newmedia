#!/usr/bin/env node

/**
 * Test Server-Sent Events streaming
 */

const { spawn } = require('child_process');
const EventSource = require('eventsource');

class SSETest {
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
    this.log('🚀 Starting MCP Suite for SSE test...');
    
    this.mcpProcess = spawn('node', ['src/index.js'], {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe']
    });

    this.mcpProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output.includes('running on port')) {
        console.log(`📡 ${output}`);
      }
    });

    // Wait for startup
    await this.sleep(8000);
  }

  async testSSE() {
    return new Promise((resolve, reject) => {
      this.log('📡 Testing Server-Sent Events...');
      
      const eventSource = new EventSource('http://localhost:3001/events');
      let receivedConnection = false;
      let eventsReceived = 0;
      
      const timeout = setTimeout(() => {
        eventSource.close();
        if (receivedConnection) {
          this.log(`SSE test completed: ${eventsReceived} events received`, 'PASS');
          resolve(true);
        } else {
          this.log('SSE test failed: No connection event received', 'FAIL');
          resolve(false);
        }
      }, 10000);
      
      eventSource.onopen = () => {
        this.log('📺 SSE connection opened');
      };
      
      eventSource.addEventListener('connected', (event) => {
        try {
          const data = JSON.parse(event.data);
          this.log(`📨 Connected event: server=${data.server}, clientId=${data.clientId}`);
          receivedConnection = true;
          eventsReceived++;
        } catch (error) {
          this.log(`Failed to parse connected event: ${error.message}`, 'FAIL');
        }
      });
      
      eventSource.addEventListener('tool_call_start', (event) => {
        try {
          const data = JSON.parse(event.data);
          this.log(`🔧 Tool call started: ${data.tool}`);
          eventsReceived++;
        } catch (error) {
          this.log(`Failed to parse tool_call_start: ${error.message}`, 'FAIL');
        }
      });
      
      eventSource.addEventListener('tool_call_complete', (event) => {
        try {
          const data = JSON.parse(event.data);
          this.log(`✅ Tool call completed: ${data.tool}`);
          eventsReceived++;
        } catch (error) {
          this.log(`Failed to parse tool_call_complete: ${error.message}`, 'FAIL');
        }
      });
      
      eventSource.onerror = (error) => {
        this.log(`SSE error: ${error.message || 'Connection failed'}`, 'FAIL');
        clearTimeout(timeout);
        eventSource.close();
        resolve(false);
      };

      // Trigger a tool call to test event broadcasting
      setTimeout(async () => {
        try {
          const axios = require('axios');
          this.log('🔧 Triggering tool call to test event broadcasting...');
          await axios.post('http://localhost:3001/call/get_system_info', 
            { arguments: {} }, 
            { headers: { 'Content-Type': 'application/json' } }
          );
        } catch (error) {
          this.log(`Tool call failed: ${error.message}`, 'FAIL');
        }
      }, 3000);
    });
  }

  async runTest() {
    await this.startMCPSuite();
    const sseResult = await this.testSSE();

    console.log('\n' + '='.repeat(50));
    this.log(`SSE Test Result: ${sseResult ? 'PASSED' : 'FAILED'}`, sseResult ? 'PASS' : 'FAIL');
    
    if (sseResult) {
      this.log('🎉 SSE streaming is working correctly!', 'PASS');
    } else {
      this.log('❌ SSE streaming has issues', 'FAIL');
    }

    // Cleanup
    if (this.mcpProcess) {
      this.log('🛑 Stopping MCP suite...');
      this.mcpProcess.kill('SIGTERM');
    }

    setTimeout(() => {
      process.exit(sseResult ? 0 : 1);
    }, 2000);
  }
}

const tester = new SSETest();
tester.runTest().catch(error => {
  console.error('💥 SSE test failed:', error);
  process.exit(1);
});