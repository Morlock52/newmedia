#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Template for a perfect MCP server
const createMCPServer = (name, displayName, tools) => `#!/usr/bin/env node

const readline = require('readline');

class ${name}MCPServer {
  constructor() {
    this.serverInfo = {
      name: '${displayName}',
      version: '1.0.0',
      capabilities: {
        tools: {},
        resources: {}
      }
    };
    
    this.tools = ${JSON.stringify(tools, null, 6)};
  }

  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(\`[MCP] \${new Date().toISOString()} \${message}\\n\`);
    }
  }

  async handleToolCall(name, args) {
    this.log(\`Tool call: \${name} with args: \${JSON.stringify(args)}\`);
    
    // Demo responses for each tool
    const responses = {
${tools.map(tool => `      '${tool.name}': {
        content: [{
          type: 'text',
          text: 'Demo response for ${tool.name}'
        }]
      }`).join(',\n')}
    };
    
    return responses[name] || {
      content: [{
        type: 'text',
        text: \`Demo result for \${name}\`
      }]
    };
  }

  async handleRequest(request) {
    this.log(\`Handling request: \${request.method}\`);
    
    try {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: '1.0',
            capabilities: this.serverInfo.capabilities,
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
          
        default:
          throw new Error(\`Unknown method: \${request.method}\`);
      }
    } catch (error) {
      this.log(\`Request error: \${error.message}\`);
      throw error;
    }
  }

  start() {
    this.log('Starting ${displayName} MCP Server...');
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        this.log(\`Received: \${JSON.stringify(request)}\`);
        
        const result = await this.handleRequest(request);
        
        const response = {
          jsonrpc: '2.0',
          id: request.id,
          result
        };
        
        process.stdout.write(JSON.stringify(response) + '\\n');
        this.log(\`Sent: \${JSON.stringify(response)}\`);
      } catch (error) {
        const errorResponse = {
          jsonrpc: '2.0',
          id: line ? JSON.parse(line).id : null,
          error: {
            code: -32603,
            message: error.message
          }
        };
        process.stdout.write(JSON.stringify(errorResponse) + '\\n');
        this.log(\`Error: \${JSON.stringify(errorResponse)}\`);
      }
    });

    rl.on('close', () => {
      this.log('Server closing...');
      process.exit(0);
    });
  }
}

const server = new ${name}MCPServer();
server.start();
`;

// Define all servers
const servers = [
  {
    filename: 'perfect-jellyfin-mcp.js',
    className: 'Jellyfin',
    displayName: 'jellyfin-mcp',
    tools: [
      {
        name: 'search_media',
        description: 'Search Jellyfin media library',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] }
      },
      {
        name: 'get_library_stats',
        description: 'Get Jellyfin library statistics',
        inputSchema: { type: 'object', properties: {} }
      }
    ]
  },
  {
    filename: 'perfect-radarr-mcp.js',
    className: 'Radarr',
    displayName: 'radarr-mcp',
    tools: [
      {
        name: 'search_movies',
        description: 'Search for movies',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] }
      },
      {
        name: 'get_movie_list',
        description: 'Get all movies',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_upcoming_movies',
        description: 'Get upcoming movies',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_missing_movies',
        description: 'Get missing movies',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_system_status',
        description: 'Get Radarr status',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_download_queue',
        description: 'Get download queue',
        inputSchema: { type: 'object', properties: {} }
      }
    ]
  },
  {
    filename: 'perfect-prowlarr-mcp.js',
    className: 'Prowlarr',
    displayName: 'prowlarr-mcp',
    tools: [
      {
        name: 'search_indexers',
        description: 'Search indexers',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] }
      },
      {
        name: 'get_indexer_list',
        description: 'Get all indexers',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_indexer_stats',
        description: 'Get indexer statistics',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'test_indexers',
        description: 'Test indexer connections',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_system_status',
        description: 'Get Prowlarr status',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'sync_apps',
        description: 'Sync to connected apps',
        inputSchema: { type: 'object', properties: {} }
      }
    ]
  }
];

// Create all servers
servers.forEach(server => {
  const content = createMCPServer(server.className, server.displayName, server.tools);
  const filePath = path.join(__dirname, server.filename);
  fs.writeFileSync(filePath, content);
  fs.chmodSync(filePath, '755');
  console.log(`✅ Created ${server.filename}`);
});

console.log('\n🎉 All perfect MCP servers created!');