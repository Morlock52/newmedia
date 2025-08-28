#!/usr/bin/env node

/**
 * Fixed MCP Server with proper stdio transport for Claude Desktop
 * Uses the latest MCP SDK with correct request handlers
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { 
  ListToolsRequestSchema, 
  CallToolRequestSchema 
} = require('@modelcontextprotocol/sdk/types.js');
const axios = require('axios');

// Create the MCP server
const server = new Server(
  {
    name: "media-server-suite",
    version: "1.0.0"
  },
  {
    capabilities: {
      tools: {}
    }
  }
);

// Define available tools
const tools = [
  {
    name: "get_jellyfin_stats",
    description: "Get Jellyfin server statistics and library information",
    inputSchema: {
      type: "object",
      properties: {
        include_library: {
          type: "boolean", 
          description: "Include detailed library statistics"
        }
      }
    }
  },
  {
    name: "search_media",
    description: "Search for media across Jellyfin, Sonarr, Radarr",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query"
        },
        media_type: {
          type: "string",
          enum: ["movie", "tv", "music", "book"],
          description: "Type of media to search for"
        }
      },
      required: ["query"]
    }
  },
  {
    name: "get_download_status",
    description: "Get current download status from qBittorrent and SABnzbd", 
    inputSchema: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "manage_service",
    description: "Start, stop, or restart media server services",
    inputSchema: {
      type: "object",
      properties: {
        service: {
          type: "string",
          enum: ["jellyfin", "sonarr", "radarr", "prowlarr", "qbittorrent"],
          description: "Service to manage"
        },
        action: {
          type: "string",
          enum: ["start", "stop", "restart", "status"],
          description: "Action to perform"
        }
      },
      required: ["service", "action"]
    }
  }
];

// Set up tool list handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});

// Set up tool call handler  
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    switch (name) {
      case "get_jellyfin_stats":
        return await handleJellyfinStats(args);
      case "search_media":
        return await handleMediaSearch(args);
      case "get_download_status":
        return await handleDownloadStatus();
      case "manage_service":
        return await handleServiceManagement(args);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    console.error(`Error in tool ${name}:`, error.message);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error.message}`
        }
      ]
    };
  }
});

// Tool implementations
async function handleJellyfinStats(args) {
  const jellyfinUrl = process.env.JELLYFIN_URL || 'http://localhost:8096';
  const apiKey = process.env.JELLYFIN_API_KEY;
  
  if (!apiKey) {
    return {
      content: [
        {
          type: "text",
          text: "❌ Jellyfin API key not configured. Please set JELLYFIN_API_KEY environment variable."
        }
      ]
    };
  }

  try {
    // Test connection and get basic info
    const systemInfo = await axios.get(`${jellyfinUrl}/System/Info`, {
      headers: { 'X-MediaBrowser-Token': apiKey }
    });

    let libraryInfo = "";
    if (args.include_library) {
      const libraries = await axios.get(`${jellyfinUrl}/Library/VirtualFolders`, {
        headers: { 'X-MediaBrowser-Token': apiKey }
      });
      
      libraryInfo = "\n\n📚 Libraries:\n" + 
        libraries.data.map(lib => `- ${lib.Name}: ${lib.Locations.join(', ')}`).join('\n');
    }

    return {
      content: [
        {
          type: "text",
          text: `✅ Jellyfin Server Status:
🏷️ Name: ${systemInfo.data.ServerName}
🔢 Version: ${systemInfo.data.Version}
🌐 URL: ${jellyfinUrl}
⏰ Started: ${new Date(systemInfo.data.StartupWizardCompleted).toLocaleString()}${libraryInfo}`
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text", 
          text: `❌ Failed to connect to Jellyfin at ${jellyfinUrl}: ${error.message}`
        }
      ]
    };
  }
}

async function handleMediaSearch(args) {
  const { query, media_type } = args;
  
  return {
    content: [
      {
        type: "text",
        text: `🔍 Searching for "${query}" ${media_type ? `(${media_type})` : ''}...
        
📺 Would search across:
- Jellyfin: Local library
- Sonarr: TV series management  
- Radarr: Movie management
- Prowlarr: Indexer search

🔧 Note: Full search functionality requires service APIs to be configured.`
      }
    ]
  };
}

async function handleDownloadStatus() {
  const qbUrl = process.env.QBITTORRENT_URL || 'http://localhost:8080';
  const sabnzbdUrl = process.env.SABNZBD_URL || 'http://localhost:8085';
  
  let status = "📥 Download Status:\n\n";
  
  // Try qBittorrent
  try {
    const qbResponse = await axios.get(`${qbUrl}/api/v2/torrents/info`, {
      timeout: 5000
    });
    status += `✅ qBittorrent: ${qbResponse.data.length} torrents\n`;
  } catch (error) {
    status += `❌ qBittorrent: Not accessible (${qbUrl})\n`;
  }
  
  // Try SABnzbd
  try {
    const sabResponse = await axios.get(`${sabnzbdUrl}/api?mode=queue&output=json`, {
      timeout: 5000
    });
    status += `✅ SABnzbd: Connected\n`;
  } catch (error) {
    status += `❌ SABnzbd: Not accessible (${sabnzbdUrl})\n`;
  }
  
  return {
    content: [
      {
        type: "text",
        text: status
      }
    ]
  };
}

async function handleServiceManagement(args) {
  const { service, action } = args;
  
  return {
    content: [
      {
        type: "text",
        text: `🔧 Service Management:
Service: ${service}
Action: ${action}

⚠️ Note: Direct service management requires Docker API or system permissions.
For now, this is a placeholder showing what would be managed.

Typical commands:
- docker restart ${service}
- systemctl ${action} ${service}
- docker-compose restart ${service}`
      }
    ]
  };
}

// Start the server with stdio transport
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('🚀 MCP Media Server Suite started with stdio transport');
}

// Handle shutdown gracefully
process.on('SIGTERM', async () => {
  console.error('🛑 Shutting down...');
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.error('🛑 Shutting down...');
  process.exit(0);
});

main().catch((error) => {
  console.error('Failed to start MCP server:', error);
  process.exit(1);
});