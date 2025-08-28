#!/usr/bin/env node

/**
 * Utility to help convert existing MCP servers to the fixed format
 * This provides a template and guidance for conversion
 */

const fs = require('fs');
const path = require('path');

const FIXED_TEMPLATE = `#!/usr/bin/env node

/**
 * Fixed MCP Server - [SERVER_NAME]
 * Converted to properly handle stdio protocol
 */

const readline = require('readline');

class Fixed[SERVER_NAME]MCPServer {
  constructor() {
    this.serverInfo = {
      name: '[server-name-mcp]',
      version: '1.0.0',
      protocolVersion: '2024.11',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    // Define your tools here
    this.tools = [
      // Example tool:
      // {
      //   name: 'example_tool',
      //   description: 'Description of what this tool does',
      //   inputSchema: {
      //     type: 'object',
      //     properties: {
      //       param1: { type: 'string', description: 'Parameter description' }
      //     },
      //     required: ['param1']
      //   }
      // }
    ];
    
    // Define your resources here
    this.resources = [
      // Example resource:
      // {
      //   uri: 'example://resource',
      //   name: 'Resource Name',
      //   mimeType: 'application/json',
      //   description: 'Resource description'
      // }
    ];
  }

  // ALL logging must go to stderr
  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(\`[[SERVER_NAME]-MCP] \${new Date().toISOString()} \${message}\\n\`);
    }
  }

  async handleToolCall(name, args) {
    this.log(\`Tool call: \${name} with args: \${JSON.stringify(args)}\`);
    
    try {
      switch (name) {
        // Add your tool handlers here
        // case 'example_tool':
        //   return {
        //     content: [{
        //       type: 'text',
        //       text: 'Tool response'
        //     }]
        //   };
          
        default:
          throw new Error(\`Unknown tool: \${name}\`);
      }
    } catch (error) {
      this.log(\`Tool call error: \${error.message}\`);
      throw error;
    }
  }

  async handleResourceRead(uri) {
    this.log(\`Resource read: \${uri}\`);
    
    try {
      switch (uri) {
        // Add your resource handlers here
        // case 'example://resource':
        //   return {
        //     contents: [{
        //       uri,
        //       mimeType: 'application/json',
        //       text: JSON.stringify({ data: 'example' }, null, 2)
        //     }]
        //   };
          
        default:
          throw new Error(\`Unknown resource: \${uri}\`);
      }
    } catch (error) {
      this.log(\`Resource read error: \${error.message}\`);
      throw error;
    }
  }

  async handleRequest(request) {
    this.log(\`Handling request: \${request.method}\`);
    
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
        throw new Error(\`Unknown method: \${request.method}\`);
    }
  }

  sendResponse(id, result) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      result: result
    };
    
    // CRITICAL: Only send JSON to stdout
    process.stdout.write(JSON.stringify(response) + '\\n');
    this.log(\`Sent response for request \${id}\`);
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
    
    process.stdout.write(JSON.stringify(response) + '\\n');
    this.log(\`Sent error for request \${id}: \${message}\`);
  }

  start() {
    this.log('Starting Fixed [SERVER_NAME] MCP Server...');
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    rl.on('line', async (line) => {
      let request;
      
      try {
        request = JSON.parse(line);
        this.log(\`Received request: \${JSON.stringify(request)}\`);
        
        if (!request.jsonrpc || request.jsonrpc !== '2.0') {
          this.sendError(request.id || null, -32600, 'Invalid JSON-RPC version');
          return;
        }
        
        if (!request.method) {
          this.sendError(request.id, -32600, 'Missing method');
          return;
        }
        
        const result = await this.handleRequest(request);
        
        if (request.id !== undefined) {
          this.sendResponse(request.id, result);
        }
        
      } catch (parseError) {
        if (parseError instanceof SyntaxError) {
          this.sendError(null, -32700, 'Parse error', parseError.message);
        } else if (request && request.id !== undefined) {
          this.sendError(
            request.id,
            -32603,
            parseError.message || 'Internal error',
            process.env.MCP_DEBUG === 'true' ? parseError.stack : undefined
          );
        }
      }
    });

    rl.on('close', () => {
      this.log('Readline closed, exiting...');
      process.exit(0);
    });
    
    process.on('SIGINT', () => {
      this.log('Received SIGINT');
      rl.close();
    });

    process.on('SIGTERM', () => {
      this.log('Received SIGTERM');
      rl.close();
    });

    this.log('[SERVER_NAME] MCP Server is ready');
  }
}

// Start the server
if (require.main === module) {
  const server = new Fixed[SERVER_NAME]MCPServer();
  server.start();
}

module.exports = Fixed[SERVER_NAME]MCPServer;
`;

function showUsage() {
  console.log(`
MCP Server Conversion Helper

This tool helps convert existing MCP servers to the fixed format that properly
handles the stdio protocol.

Usage:
  node convert-mcp-server.js <server-name>

Example:
  node convert-mcp-server.js MyService

This will create a file called 'fixed-myservice-mcp.js' with the proper structure.

Key Changes Needed:
1. Replace all console.log() with process.stdout.write() for responses
2. Use process.stderr.write() for all debug/log output
3. Ensure each JSON response ends with \\n
4. Add proper JSON-RPC error handling
5. Keep the process running with readline

After generation, you'll need to:
1. Copy your tool definitions from the old server
2. Copy your tool handler logic
3. Copy your resource definitions and handlers
4. Update any API calls or external dependencies
`);
}

function convertServerName(name) {
  // Convert to different formats
  const pascal = name.charAt(0).toUpperCase() + name.slice(1);
  const kebab = name.toLowerCase().replace(/([A-Z])/g, '-$1').toLowerCase();
  const filename = `fixed-${kebab}-mcp.js`;
  
  return { pascal, kebab, filename };
}

function generateTemplate(serverName) {
  const { pascal, kebab, filename } = convertServerName(serverName);
  
  let template = FIXED_TEMPLATE;
  template = template.replace(/\[SERVER_NAME\]/g, pascal);
  template = template.replace(/\[server-name-mcp\]/g, kebab);
  
  return { template, filename };
}

// Main execution
if (process.argv.length < 3) {
  showUsage();
  process.exit(1);
}

const serverName = process.argv[2];
const { template, filename } = generateTemplate(serverName);

// Write the template file
fs.writeFileSync(filename, template);
fs.chmodSync(filename, '755');

console.log(`✅ Created ${filename}`);
console.log(`
Next steps:
1. Open ${filename} in your editor
2. Copy your tool definitions from your original server
3. Copy your tool handler implementations
4. Copy your resource definitions and handlers
5. Test with: ./test-fixed-server.js

Remember:
- Use process.stdout.write() ONLY for JSON-RPC responses
- Use process.stderr.write() for ALL debug output
- Each JSON response must end with \\n
- Test thoroughly before deploying!
`);