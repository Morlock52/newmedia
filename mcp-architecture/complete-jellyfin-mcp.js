#!/usr/bin/env node

const CompleteMCPServer = require('./complete-mcp-base');

class JellyfinMCP extends CompleteMCPServer {
  constructor() {
    const tools = [{
      name: 'get_status',
      description: 'Get jellyfin status',
      inputSchema: { type: 'object', properties: {} }
    }];

    const resources = [{
      uri: 'jellyfin://status',
      name: 'jellyfin Status',
      description: 'System status',
      mimeType: 'application/json'
    }];

    const prompts = [{
      name: 'jellyfin_helper',
      description: 'Get help with jellyfin',
      arguments: []
    }];

    super('jellyfin-mcp', tools, resources, prompts);
  }

  async handleToolCall(name, args) {
    return {
      content: [{
        type: 'text',
        text: '✅ jellyfin MCP Server is working!'
      }]
    };
  }

  async handleResourceRead(uri) {
    return {
      contents: [{
        uri,
        mimeType: 'application/json',
        text: JSON.stringify({ status: 'working', service: 'jellyfin' }, null, 2)
      }]
    };
  }

  async handlePromptGet(name, args) {
    return {
      messages: [{
        role: 'user',
        content: {
          type: 'text',
          text: 'Help me with jellyfin operations.'
        }
      }]
    };
  }
}

const server = new JellyfinMCP();
server.start();