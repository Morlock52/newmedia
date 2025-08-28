#!/usr/bin/env node

/**
 * Fixed MCP Server Implementation
 * Properly handles stdio protocol with JSON-RPC messages
 * 
 * Key fixes:
 * - Only JSON-RPC messages go to stdout
 * - All debug/logging goes to stderr
 * - Process stays running with readline
 * - Proper error handling returns JSON-RPC errors
 * - No console.log() pollution
 */

const readline = require('readline');

class FixedMCPServer {
  constructor() {
    this.serverInfo = {
      name: 'fixed-mcp-server',
      version: '1.0.0',
      protocolVersion: '2024.11',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    this.tools = [
      {
        name: 'test_echo',
        description: 'Echo back the input for testing',
        inputSchema: {
          type: 'object',
          properties: {
            message: { 
              type: 'string', 
              description: 'Message to echo back' 
            }
          },
          required: ['message']
        }
      },
      {
        name: 'get_status',
        description: 'Get server status',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      }
    ];
    
    this.resources = [
      {
        uri: 'test://status',
        name: 'Server Status',
        mimeType: 'application/json',
        description: 'Current server status information'
      }
    ];
  }

  // ALL logging must go to stderr, never stdout
  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[MCP-DEBUG] ${new Date().toISOString()} ${message}\n`);
    }
  }

  async handleToolCall(name, args) {
    this.log(`Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    switch (name) {
      case 'test_echo':
        return {
          content: [{
            type: 'text',
            text: `Echo: ${args.message || 'No message provided'}`
          }]
        };
        
      case 'get_status':
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: 'running',
              uptime: process.uptime(),
              memory: process.memoryUsage(),
              timestamp: new Date().toISOString()
            }, null, 2)
          }]
        };
        
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  async handleResourceRead(uri) {
    this.log(`Resource read: ${uri}`);
    
    switch (uri) {
      case 'test://status':
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify({
              server: this.serverInfo.name,
              version: this.serverInfo.version,
              uptime: process.uptime(),
              pid: process.pid,
              platform: process.platform,
              nodeVersion: process.version
            }, null, 2)
          }]
        };
        
      default:
        throw new Error(`Unknown resource: ${uri}`);
    }
  }

  async handleRequest(request) {
    this.log(`Handling request: ${request.method}`);
    
    switch (request.method) {
      case 'initialize':
        return {
          protocolVersion: this.serverInfo.protocolVersion,
          serverInfo: this.serverInfo
        };
        
      case 'tools/list':
        return { tools: this.tools };
        
      case 'tools/call':
        if (!request.params || !request.params.name) {
          throw new Error('Missing tool name');
        }
        return await this.handleToolCall(
          request.params.name,
          request.params.arguments || {}
        );
        
      case 'resources/list':
        return { resources: this.resources };
        
      case 'resources/read':
        if (!request.params || !request.params.uri) {
          throw new Error('Missing resource URI');
        }
        return await this.handleResourceRead(request.params.uri);
        
      case 'completion/complete':
        return { 
          completion: { 
            values: [],
            hasMore: false
          } 
        };
        
      default:
        throw new Error(`Unknown method: ${request.method}`);
    }
  }

  sendResponse(id, result) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      result: result
    };
    
    // CRITICAL: Use process.stdout.write with newline
    // Never use console.log as it may add extra formatting
    process.stdout.write(JSON.stringify(response) + '\n');
    this.log(`Sent response: ${JSON.stringify(response)}`);
  }

  sendError(id, code, message, data = null) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      error: {
        code: code,
        message: message
      }
    };
    
    if (data) {
      response.error.data = data;
    }
    
    process.stdout.write(JSON.stringify(response) + '\n');
    this.log(`Sent error: ${JSON.stringify(response)}`);
  }

  start() {
    this.log('Starting Fixed MCP Server...');
    
    // Create readline interface to keep process running
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    // Handle incoming JSON-RPC requests
    rl.on('line', async (line) => {
      let request;
      
      try {
        // Parse the incoming request
        request = JSON.parse(line);
        this.log(`Received: ${JSON.stringify(request)}`);
        
        // Validate JSON-RPC request
        if (!request.jsonrpc || request.jsonrpc !== '2.0') {
          this.sendError(request.id || null, -32600, 'Invalid JSON-RPC version');
          return;
        }
        
        if (!request.method) {
          this.sendError(request.id, -32600, 'Missing method');
          return;
        }
        
        // Handle the request
        const result = await this.handleRequest(request);
        
        // Send successful response
        if (request.id !== undefined) {
          this.sendResponse(request.id, result);
        }
        
      } catch (parseError) {
        // JSON parse error
        if (parseError instanceof SyntaxError) {
          this.sendError(null, -32700, 'Parse error', parseError.message);
        } else if (request && request.id !== undefined) {
          // Request processing error
          this.sendError(
            request.id,
            -32603,
            parseError.message || 'Internal error',
            process.env.MCP_DEBUG === 'true' ? parseError.stack : undefined
          );
        }
      }
    });

    // Handle readline close event
    rl.on('close', () => {
      this.log('Readline interface closed, exiting...');
      process.exit(0);
    });
    
    // Handle process errors gracefully
    process.on('uncaughtException', (error) => {
      this.log(`Uncaught exception: ${error.message}`);
      // Don't exit on uncaught exceptions, just log them
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      this.log(`Unhandled rejection at ${promise}: ${reason}`);
      // Don't exit on unhandled rejections, just log them
    });

    // Handle termination signals gracefully
    process.on('SIGINT', () => {
      this.log('Received SIGINT, shutting down gracefully...');
      rl.close();
    });

    process.on('SIGTERM', () => {
      this.log('Received SIGTERM, shutting down gracefully...');
      rl.close();
    });

    // Log that we're ready (to stderr, not stdout!)
    this.log('MCP Server is ready and listening for JSON-RPC requests');
  }
}

// Only start the server if this file is run directly
if (require.main === module) {
  const server = new FixedMCPServer();
  server.start();
}

// Export for testing
module.exports = FixedMCPServer;