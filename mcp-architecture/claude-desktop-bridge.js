#!/usr/bin/env node

/**
 * Claude Desktop MCP Bridge
 * This bridges our HTTP MCP servers to Claude Desktop's stdio protocol
 */

const readline = require('readline');
const axios = require('axios');

class ClaudeDesktopBridge {
  constructor() {
    this.baseUrl = process.env.MCP_BASE_URL || 'http://localhost:3001';
    this.serverInfo = {
      name: 'media-server-mcp',
      version: '1.0.0',
      protocolVersion: '0.1.0',
      capabilities: {
        tools: true,
        resources: true,
        prompts: false,
        sampling: false
      }
    };
  }

  log(message) {
    if (process.env.DEBUG) {
      process.stderr.write(`[MCP Bridge] ${message}\n`);
    }
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: this.serverInfo.protocolVersion,
            serverInfo: this.serverInfo
          };
          
        case 'tools/list':
          const toolsResponse = await axios.get(`${this.baseUrl}/tools`);
          return toolsResponse.data.data || { tools: [] };
          
        case 'tools/call':
          const { name, arguments: args } = request.params;
          const callResponse = await axios.post(
            `${this.baseUrl}/call/${name}`,
            { arguments: args || {} },
            { 
              headers: { 'Content-Type': 'application/json' },
              timeout: 30000
            }
          );
          
          if (callResponse.data.success) {
            return callResponse.data.data;
          } else {
            throw new Error(callResponse.data.error || 'Tool call failed');
          }
          
        case 'resources/list':
          const resourcesResponse = await axios.get(`${this.baseUrl}/resources`);
          return resourcesResponse.data.data || { resources: [] };
          
        case 'resources/read':
          const { uri } = request.params;
          const readResponse = await axios.get(`${this.baseUrl}/resources/${encodeURIComponent(uri)}`);
          return readResponse.data.data;
          
        case 'completion/complete':
          return { completion: { values: [] } };
          
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      this.log(`Error handling request: ${error.message}`);
      throw error;
    }
  }

  async start() {
    this.log('Starting Claude Desktop MCP Bridge...');
    
    // Test connection to MCP server
    try {
      const health = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
      this.log(`Connected to MCP server: ${health.data.server || 'unknown'}`);
    } catch (error) {
      this.log(`Warning: Cannot connect to MCP server at ${this.baseUrl}`);
      this.log('Make sure to start the MCP suite first: node src/simple-index.js');
    }
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        this.log(`Received request: ${JSON.stringify(request)}`);
        
        const result = await this.handleRequest(request);
        
        const response = {
          jsonrpc: '2.0',
          id: request.id,
          result
        };
        
        console.log(JSON.stringify(response));
        this.log(`Sent response: ${JSON.stringify(response)}`);
      } catch (error) {
        const errorResponse = {
          jsonrpc: '2.0',
          id: JSON.parse(line).id,
          error: {
            code: -32603,
            message: error.message
          }
        };
        console.log(JSON.stringify(errorResponse));
        this.log(`Sent error: ${JSON.stringify(errorResponse)}`);
      }
    });

    rl.on('close', () => {
      this.log('Bridge closing...');
      process.exit(0);
    });
  }
}

// Start the bridge
const bridge = new ClaudeDesktopBridge();
bridge.start();