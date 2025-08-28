#!/usr/bin/env node

const CompleteMCPServer = require('./complete-mcp-base');

class CompleteMediaServerMCP extends CompleteMCPServer {
  constructor() {
    const tools = [
      {
        name: 'search_media',
        description: 'Search for media across all services',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            type: { type: 'string', enum: ['movie', 'tv', 'music', 'all'], description: 'Media type' }
          },
          required: ['query']
        }
      },
      {
        name: 'get_library_stats',
        description: 'Get statistics about media libraries',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      },
      {
        name: 'get_recent_media',
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
        description: 'Get system information for all services',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      }
    ];

    const resources = [
      { 
        uri: 'media://library',
        name: 'Media Library',
        description: 'Complete media library information',
        mimeType: 'application/json'
      },
      {
        uri: 'media://stats',
        name: 'Library Statistics',
        description: 'Statistics about the media library',
        mimeType: 'application/json'
      }
    ];

    const prompts = [
      {
        name: 'media_search_assistant',
        description: 'Get help searching for specific media',
        arguments: [
          {
            name: 'media_type',
            description: 'Type of media to search for',
            required: false
          },
          {
            name: 'genre',
            description: 'Preferred genre',
            required: false
          }
        ]
      },
      {
        name: 'library_organizer',
        description: 'Get suggestions for organizing your media library',
        arguments: [
          {
            name: 'library_size',
            description: 'Size of your current library',
            required: false
          }
        ]
      }
    ];

    super('media-server-mcp', tools, resources, prompts);
  }

  async handleToolCall(name, args) {
    console.error(`[${this.serverInfo.name}] Tool call: ${name} with args: ${JSON.stringify(args)}`);
    
    switch (name) {
      case 'search_media':
        return {
          content: [{
            type: 'text',
            text: `🔍 Search results for "${args.query}":\n\nMovies:\n• The Matrix (1999) - Sci-Fi\n• The Matrix Reloaded (2003) - Sci-Fi\n\nTV Shows:\n• Breaking Bad (2008-2013) - Drama\n• Better Call Saul (2015-2022) - Drama\n\nNote: This is demo data. Connect real services for actual results.`
          }]
        };
        
      case 'get_library_stats':
        return {
          content: [{
            type: 'text',
            text: `📊 Media Library Statistics:\n\nMovies: 1,247 titles\nTV Shows: 89 series (2,341 episodes)\nMusic: 15,623 tracks\nTotal Size: 12.4 TB\n\nActive Streams: 3\nBandwidth: 42 Mbps`
          }]
        };
        
      case 'get_recent_media':
        const limit = args.limit || 5;
        return {
          content: [{
            type: 'text',
            text: `🆕 Recently Added Media (Last ${limit} items):\n\n1. Oppenheimer (2023) - Added 2 hours ago\n2. The Last of Us S01E09 - Added 5 hours ago\n3. Succession S04E10 - Added 1 day ago\n4. Spider-Man: Across the Spider-Verse (2023) - Added 2 days ago\n5. Ted Lasso S03E12 - Added 3 days ago`
          }]
        };
        
      case 'get_system_info':
        return {
          content: [{
            type: 'text',
            text: `🖥️ System Information:\n\nJellyfin: ✅ Running (v10.8.13)\nSonarr: ✅ Running (v4.0.0.731)\nRadarr: ✅ Running (v5.2.6.8376)\nProwlarr: ✅ Running (v1.11.4.4173)\nqBittorrent: ✅ Running (v4.6.2)\n\nCPU Usage: 12%\nMemory: 4.2GB / 16GB\nStorage: 2.8TB free`
          }]
        };
        
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  async handleResourceRead(uri) {
    console.error(`[${this.serverInfo.name}] Resource read: ${uri}`);
    
    switch (uri) {
      case 'media://library':
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify({
              movies: 1247,
              shows: 89,
              episodes: 2341,
              music: 15623,
              lastUpdated: new Date().toISOString()
            }, null, 2)
          }]
        };
        
      case 'media://stats':
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify({
              totalSize: '12.4TB',
              activeStreams: 3,
              bandwidth: '42Mbps',
              topGenres: ['Drama', 'Comedy', 'Action'],
              lastUpdated: new Date().toISOString()
            }, null, 2)
          }]
        };
        
      default:
        throw new Error(`Unknown resource: ${uri}`);
    }
  }

  async handlePromptGet(name, args) {
    console.error(`[${this.serverInfo.name}] Prompt get: ${name} with args: ${JSON.stringify(args)}`);
    
    switch (name) {
      case 'media_search_assistant':
        const mediaType = args.media_type || 'any';
        const genre = args.genre || 'any';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `I'm looking for ${mediaType} content${genre !== 'any' ? ` in the ${genre} genre` : ''}. Can you help me search through my media library and suggest some options? Please consider my viewing history and preferences.`
            }
          }]
        };
        
      case 'library_organizer':
        const librarySize = args.library_size || 'medium';
        return {
          messages: [{
            role: 'user',
            content: {
              type: 'text',
              text: `I have a ${librarySize} media library and I want to organize it better. Can you suggest some strategies for categorizing, naming conventions, and folder structures that would make it easier to browse and maintain?`
            }
          }]
        };
        
      default:
        throw new Error(`Unknown prompt: ${name}`);
    }
  }
}

// Start the server
const server = new CompleteMediaServerMCP();
server.start();