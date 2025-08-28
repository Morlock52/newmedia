#!/usr/bin/env node

/**
 * Test MCP Server with stdio transport for Claude Desktop
 * This is a minimal working MCP server to test stdio connection
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');

// Create a minimal MCP server
const server = new Server({
  name: "test-media-server",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});

// Add a simple test tool
server.setRequestHandler('tools/list', async () => {
  return {
    tools: [
      {
        name: "test_connection",
        description: "Test MCP connection to media server",
        inputSchema: {
          type: "object",
          properties: {
            service: {
              type: "string", 
              description: "Service to test (jellyfin, sonarr, etc.)"
            }
          }
        }
      }
    ]
  };
});

server.setRequestHandler('tools/call', async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "test_connection") {
    const service = args?.service || "jellyfin";
    return {
      content: [
        {
          type: "text",
          text: `Testing connection to ${service}... Connection successful! 🎉`
        }
      ]
    };
  }
  
  throw new Error(`Unknown tool: ${name}`);
});

// Start stdio transport
const transport = new StdioServerTransport();
server.connect(transport);

console.error('MCP server started with stdio transport');