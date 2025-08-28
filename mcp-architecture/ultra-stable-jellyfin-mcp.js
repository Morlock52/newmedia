#!/usr/bin/env node

const UltraStableMCPServer = require('./ultra-stable-mcp-base');
const http = require('http');
const https = require('https');
const url = require('url');

class JellyfinMCP extends UltraStableMCPServer {
  constructor() {
    const tools = [
      {
        name: 'search_media',
        description: 'Search for media in Jellyfin',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            type: { type: 'string', enum: ['Movie', 'Series', 'Audio', 'All'], description: 'Media type to search' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_libraries',
        description: 'Get all Jellyfin libraries',
        inputSchema: { type: 'object', properties: {} }
      },
      {
        name: 'get_latest_media',
        description: 'Get recently added media',
        inputSchema: {
          type: 'object',
          properties: {
            limit: { type: 'number', description: 'Number of items to return' }
          }
        }
      },
      {
        name: 'get_system_info',
        description: 'Get Jellyfin system information',
        inputSchema: { type: 'object', properties: {} }
      }
    ];

    const resources = [
      { uri: 'jellyfin://libraries', name: 'Media Libraries', mimeType: 'application/json' },
      { uri: 'jellyfin://latest', name: 'Latest Media', mimeType: 'application/json' },
      { uri: 'jellyfin://system', name: 'System Info', mimeType: 'application/json' }
    ];

    super('jellyfin', tools, resources);
    
    this.jellyfinUrl = process.env.JELLYFIN_URL || 'http://localhost:8096';
    this.jellyfinApiKey = process.env.JELLYFIN_API_KEY || '';
  }

  async handleToolCall(name, args) {
    // Demo implementation - would connect to real Jellyfin with API key
    switch (name) {
      case 'search_media':
        return {
          content: [{
            type: 'text',
            text: `🔍 Jellyfin search for "${args.query}":\n\n• The Matrix (1999)\n• The Matrix Reloaded (2003)\n• The Matrix Revolutions (2003)\n\nNote: This is demo data. Configure JELLYFIN_API_KEY for real results.`
          }]
        };
      case 'get_libraries':
        return {
          content: [{
            type: 'text',
            text: '📚 Jellyfin Libraries:\n\n• Movies (1,247 items)\n• TV Shows (89 series)\n• Music (15,623 tracks)\n• Home Videos (432 items)'
          }]
        };
      case 'get_latest_media':
        return {
          content: [{
            type: 'text',
            text: '🆕 Recently Added:\n\n• Oppenheimer (2023)\n• The Last of Us S01E09\n• Succession S04E10\n• Ted Lasso S03E12'
          }]
        };
      case 'get_system_info':
        return {
          content: [{
            type: 'text',
            text: '🖥️ Jellyfin System:\n\nVersion: 10.8.13\nOS: Linux\nArchitecture: x64\nActive Streams: 2\nTranscoding: Enabled'
          }]
        };
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }
}

// Start server
const server = new JellyfinMCP();
server.start();