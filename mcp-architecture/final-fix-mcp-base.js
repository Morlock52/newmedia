#!/usr/bin/env node

/**
 * Final Fix MCP Server Base - Compatible with Claude Desktop 2025
 */

const readline = require('readline');

class FinalFixMCPServer {
  constructor(serverName, tools = [], resources = []) {
    this.serverInfo = {
      name: serverName,
      version: '1.0.0'
    };
    
    this.tools = tools;
    this.resources = resources;
    
    // Protocol version that matches Claude Desktop
    this.protocolVersion = '2025-06-18';
    
    // Debug logging
    this.debug = process.env.MCP_DEBUG === 'true';
  }

  log(message) {
    if (this.debug) {
      console.error(`[${this.serverInfo.name}] ${new Date().toISOString()} ${message}`);
    }
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    try {
      switch (request.method) {
        case 'initialize':
          const initResult = {
            protocolVersion: this.protocolVersion,
            capabilities: {
              tools: this.tools.length > 0 ? {} : undefined,
              resources: this.resources.length > 0 ? {} : undefined
            },
            serverInfo: this.serverInfo
          };
          this.log(`Sending initialize response: ${JSON.stringify(initResult)}`);
          return initResult;
          
        case 'tools/list':
          return { tools: this.tools };
          
        case 'tools/call':
          return await this.handleToolCall(
            request.params.name,
            request.params.arguments || {}
          );
          
        case 'resources/list':
          return { resources: this.resources };
          
        case 'resources/read':
          return await this.handleResourceRead(request.params.uri);
          
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      this.log(`Request error: ${error.message}`);
      throw error;
    }
  }

  // Override in subclasses
  async handleToolCall(name, args) {
    return {
      content: [{
        type: 'text',
        text: `Tool ${name} called with args: ${JSON.stringify(args)}`
      }]
    };
  }

  // Override in subclasses
  async handleResourceRead(uri) {
    return {
      contents: [{
        uri,
        mimeType: 'application/json',
        text: JSON.stringify({ message: `Resource ${uri} read` }, null, 2)
      }]
    };
  }

  start() {
    console.error(`[${this.serverInfo.name}] Starting MCP Server...`);
    
    // Create readline interface
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    // Keep the process alive
    const keepAlive = setInterval(() => {
      // No-op to keep event loop active
    }, 60000);

    // Prevent Node.js from exiting
    process.stdin.on('end', () => {
      console.error(`[${this.serverInfo.name}] stdin ended, keeping process alive...`);
    });

    // Handle incoming messages
    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        this.log(`Received: ${JSON.stringify(request)}`);
        
        const result = await this.handleRequest(request);
        
        const response = {
          jsonrpc: '2.0',
          id: request.id,
          result
        };
        
        const responseStr = JSON.stringify(response);
        process.stdout.write(responseStr + '\n');
        this.log(`Sent: ${responseStr}`);
      } catch (error) {
        let errorRequest;
        try {
          errorRequest = JSON.parse(line);
        } catch (e) {
          console.error(`Failed to parse request: ${line}`);
          return;
        }
        
        const errorResponse = {
          jsonrpc: '2.0',
          id: errorRequest.id,
          error: {
            code: -32603,
            message: error.message
          }
        };
        process.stdout.write(JSON.stringify(errorResponse) + '\n');
        this.log(`Error response: ${JSON.stringify(errorResponse)}`);
      }
    });

    // Handle signals gracefully
    const shutdown = () => {
      console.error(`[${this.serverInfo.name}] Shutting down...`);
      clearInterval(keepAlive);
      process.exit(0);
    };

    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    // Log that we're ready
    console.error(`[${this.serverInfo.name}] Server ready and listening`);
  }
}

module.exports = FinalFixMCPServer;