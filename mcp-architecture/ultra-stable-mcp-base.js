#!/usr/bin/env node

/**
 * Ultra-Stable MCP Server Base
 * Fixes the "server disconnected" issue by properly keeping the process alive
 */

const readline = require('readline');

class UltraStableMCPServer {
  constructor(serverName, tools = [], resources = []) {
    this.serverInfo = {
      name: serverName,
      version: '1.0.0',
      protocolVersion: '1.0',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    this.tools = tools;
    this.resources = resources;
    
    // Keep-alive interval to prevent process from exiting
    this.keepAliveInterval = null;
  }

  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[${this.serverInfo.name}] ${new Date().toISOString()} ${message}\n`);
    }
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: '1.0',
            capabilities: this.serverInfo.capabilities || { tools: {}, resources: {} },
            serverInfo: {
              name: this.serverInfo.name,
              version: this.serverInfo.version
            }
          };
          
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
          
        case 'completion/complete':
          return { completion: { values: [] } };
          
        default:
          throw new Error(`Unknown method: ${request.method}`);
      }
    } catch (error) {
      this.log(`Request error: ${error.message}`);
      throw error;
    }
  }

  // Override this in subclasses
  async handleToolCall(name, args) {
    return {
      content: [{
        type: 'text',
        text: `Tool ${name} called with args: ${JSON.stringify(args)}`
      }]
    };
  }

  // Override this in subclasses
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
    this.log('Starting MCP Server...');
    
    // Create readline interface
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
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
        
        process.stdout.write(JSON.stringify(response) + '\n');
        this.log(`Sent: ${JSON.stringify(response)}`);
      } catch (error) {
        const errorResponse = {
          jsonrpc: '2.0',
          id: JSON.parse(line).id,
          error: {
            code: -32603,
            message: error.message,
            data: error.stack
          }
        };
        process.stdout.write(JSON.stringify(errorResponse) + '\n');
        this.log(`Error: ${JSON.stringify(errorResponse)}`);
      }
    });

    // Handle readline close
    rl.on('close', () => {
      this.log('Readline closed, shutting down...');
      if (this.keepAliveInterval) {
        clearInterval(this.keepAliveInterval);
      }
      process.exit(0);
    });

    // Handle process signals
    process.on('SIGINT', () => {
      this.log('Received SIGINT, shutting down...');
      if (this.keepAliveInterval) {
        clearInterval(this.keepAliveInterval);
      }
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      this.log('Received SIGTERM, shutting down...');
      if (this.keepAliveInterval) {
        clearInterval(this.keepAliveInterval);
      }
      process.exit(0);
    });
    
    // Handle errors gracefully
    process.on('uncaughtException', (error) => {
      this.log(`Uncaught exception: ${error.message}`);
      process.stderr.write(`MCP Error: ${error.message}\n`);
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      this.log(`Unhandled rejection: ${reason}`);
      process.stderr.write(`MCP Unhandled Rejection: ${reason}\n`);
    });

    // CRITICAL: Keep the process alive
    // This prevents Node.js from exiting when there are no active handles
    this.keepAliveInterval = setInterval(() => {
      // Just a no-op to keep the event loop active
      this.log('Keep-alive tick');
    }, 30000); // Every 30 seconds

    // Also keep stdin open to prevent exit
    process.stdin.resume();
    
    this.log('Server started and listening for commands');
  }
}

module.exports = UltraStableMCPServer;