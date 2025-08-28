# 🔌 MCP Suite Connection Guide - HTTP/SSE Streaming

## 🚀 Overview

Your MediaServer MCP Suite provides **HTTP/SSE (Server-Sent Events) streaming** for all MCP servers, making them accessible via standard HTTP APIs with real-time updates. This allows you to connect any client, tool, or application to your media services.

## 📊 Available MCP Servers

### 🎬 Jellyfin MCP Server
- **Port**: `3001`
- **Base URL**: `http://localhost:3001`
- **Service**: Media streaming and library management

### 📺 Sonarr MCP Server  
- **Port**: `3002`
- **Base URL**: `http://localhost:3002`
- **Service**: TV show automation and management

### 🎞️ Radarr MCP Server
- **Port**: `3003` 
- **Base URL**: `http://localhost:3003`
- **Service**: Movie automation and management

### 🔍 Prowlarr MCP Server
- **Port**: `3004`
- **Base URL**: `http://localhost:3004`
- **Service**: Indexer management and search

### 🌊 qBittorrent MCP Server
- **Port**: `3005`
- **Base URL**: `http://localhost:3005`
- **Service**: Torrent client management

## 🔗 HTTP API Endpoints

Each MCP server exposes the following HTTP endpoints:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/info` | GET | Server information and capabilities |
| `/events` | GET | **SSE stream** for real-time updates |
| `/resources` | GET | List available resources |
| `/resources/*` | GET | Read specific resource |
| `/tools` | GET | List available tools |
| `/call/:toolName` | POST | Execute a specific tool |
| `/mcp` | POST | Generic MCP request handler |

### 📡 Real-Time Streaming (SSE)

**Connect to live events:**
```bash
curl -N -H "Accept: text/event-stream" http://localhost:3001/events
```

**JavaScript Example:**
```javascript
const eventSource = new EventSource('http://localhost:3001/events');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};

eventSource.addEventListener('tool_start', function(event) {
  console.log('Tool started:', JSON.parse(event.data));
});
```

## 🛠️ Using the MCP Servers

### 1. Health Check Example
```bash
# Check if Jellyfin MCP is running
curl http://localhost:3001/health

# Response:
{
  "status": "healthy",
  "server": "jellyfin-mcp",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "clients": 2
}
```

### 2. List Available Tools
```bash
# Get all Jellyfin tools
curl http://localhost:3001/tools

# Response:
{
  "success": true,
  "data": {
    "tools": [
      {
        "name": "search_media",
        "description": "Search for movies, TV shows, music",
        "inputSchema": { ... }
      },
      {
        "name": "get_library_stats", 
        "description": "Get media library statistics",
        "inputSchema": { ... }
      }
    ]
  }
}
```

### 3. Execute a Tool
```bash
# Search for movies in Jellyfin
curl -X POST http://localhost:3001/call/search_media \
  -H "Content-Type: application/json" \
  -d '{
    "arguments": {
      "query": "Inception",
      "type": "Movie",
      "limit": 10
    }
  }'

# Response:
{
  "success": true,
  "data": {
    "content": [{
      "type": "text", 
      "text": "Found 3 movies matching 'Inception':\n• Inception (2010) - Christopher Nolan\n..."
    }]
  },
  "timestamp": "2025-01-15T10:35:00.000Z"
}
```

### 4. Get Resources
```bash
# List all resources
curl http://localhost:3002/resources

# Get specific resource (Sonarr series)
curl http://localhost:3002/resources/sonarr://series
```

## 🧠 Connecting to Claude Desktop

### Method 1: Direct HTTP MCP Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jellyfin": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {
        "FETCH_BASE_URL": "http://localhost:3001"
      }
    },
    "sonarr": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {
        "FETCH_BASE_URL": "http://localhost:3002"
      }
    },
    "radarr": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"], 
      "env": {
        "FETCH_BASE_URL": "http://localhost:3003"
      }
    },
    "prowlarr": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {
        "FETCH_BASE_URL": "http://localhost:3004"
      }
    },
    "qbittorrent": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {
        "FETCH_BASE_URL": "http://localhost:3005"
      }
    }
  }
}
```

### Method 2: Single Orchestrator Connection

```json
{
  "mcpServers": {
    "mediaserver-suite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {
        "FETCH_BASE_URL": "http://localhost:8090/api/mcp"
      }
    }
  }
}
```

## 🔌 Connecting Other MCP Clients

### Python MCP Client
```python
import requests
import json

class MCPClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def call_tool(self, tool_name, arguments=None):
        url = f"{self.base_url}/call/{tool_name}"
        payload = {"arguments": arguments or {}}
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def list_tools(self):
        response = requests.get(f"{self.base_url}/tools")
        return response.json()

# Usage
jellyfin = MCPClient("http://localhost:3001")
result = jellyfin.call_tool("search_media", {
    "query": "The Matrix",
    "type": "Movie"
})
print(result)
```

### Node.js MCP Client
```javascript
const axios = require('axios');

class MCPClient {
  constructor(baseURL) {
    this.baseURL = baseURL;
    this.client = axios.create({ baseURL });
  }

  async callTool(toolName, arguments = {}) {
    const response = await this.client.post(`/call/${toolName}`, {
      arguments
    });
    return response.data;
  }

  async listTools() {
    const response = await this.client.get('/tools');
    return response.data;
  }

  // Real-time events
  connectToEvents() {
    const EventSource = require('eventsource');
    const eventSource = new EventSource(`${this.baseURL}/events`);
    
    return eventSource;
  }
}

// Usage
const sonarr = new MCPClient('http://localhost:3002');

// Get TV shows
sonarr.callTool('get_series', { seriesId: 123 })
  .then(result => console.log(result));

// Listen to real-time updates
const events = sonarr.connectToEvents();
events.onmessage = (event) => {
  console.log('Real-time update:', JSON.parse(event.data));
};
```

## 🔄 Real-Time Event Types

Each MCP server streams these event types via SSE:

### Common Events
- `connected` - Client connection established
- `tool_start` - Tool execution started
- `tool_complete` - Tool execution completed
- `tool_error` - Tool execution failed

### Service-Specific Events
- **Jellyfin**: `media_scan`, `playback_start`, `playback_stop`
- **Sonarr**: `episode_downloaded`, `series_added`, `search_started`
- **Radarr**: `movie_downloaded`, `movie_added`, `search_started`
- **Prowlarr**: `indexer_test`, `search_results`, `sync_complete`
- **qBittorrent**: `torrent_added`, `download_complete`, `status_change`

## 📋 Tool Reference Guide

### 🎬 Jellyfin Tools
- `search_media` - Search movies, TV shows, music
- `get_library_stats` - Library statistics
- `get_recent_media` - Recently added content
- `control_playback` - Control media playback
- `get_user_activity` - User activity monitoring

### 📺 Sonarr Tools  
- `search_tv_shows` - Search for TV series
- `get_series` - Get series information
- `monitor_series` - Enable/disable monitoring
- `get_queue` - Download queue status
- `get_calendar` - Upcoming episodes
- `refresh_series` - Refresh series metadata

### 🎞️ Radarr Tools
- `search_movies` - Search for movies
- `get_movie` - Get movie information  
- `add_movie` - Add movie to library
- `monitor_movie` - Enable/disable monitoring
- `get_missing_movies` - Missing movies list
- `search_movie` - Trigger movie search

### 🔍 Prowlarr Tools
- `get_indexers` - List all indexers
- `test_indexer` - Test indexer connection
- `search_indexers` - Search across indexers
- `toggle_indexer` - Enable/disable indexer
- `sync_applications` - Sync with apps
- `get_indexer_stats` - Performance statistics

### 🌊 qBittorrent Tools
- `get_torrents` - List all torrents
- `get_torrent_info` - Detailed torrent info
- `pause_torrents` / `resume_torrents` - Control torrents
- `add_torrent` - Add new torrent
- `delete_torrents` - Remove torrents
- `get_global_stats` - Transfer statistics

## 🛡️ Security & Authentication

### Environment Variables
Set these in your `.env` file:
```bash
# API Keys for each service
JELLYFIN_API_KEY=your-jellyfin-api-key
SONARR_API_KEY=your-sonarr-api-key
RADARR_API_KEY=your-radarr-api-key
PROWLARR_API_KEY=your-prowlarr-api-key

# qBittorrent credentials
QBITTORRENT_USERNAME=admin
QBITTORRENT_PASSWORD=your-secure-password

# OpenAI for AI agents
OPENAI_API_KEY=sk-your-openai-key

# Security
JWT_SECRET=your-jwt-secret
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8090
```

### Rate Limiting
Each server implements rate limiting:
- **100 requests per 15 minutes** per IP
- **WebSocket connections limited** to 10 per IP
- **SSE connections** automatically managed

## 🧪 Testing Your Connection

### Quick Health Check Script
```bash
#!/bin/bash
echo "🔍 Testing MCP Server Connections..."

services=("jellyfin:3001" "sonarr:3002" "radarr:3003" "prowlarr:3004" "qbittorrent:3005")

for service in "${services[@]}"; do
  name=${service%%:*}
  port=${service##*:}
  
  echo -n "Testing $name (port $port): "
  
  if curl -s http://localhost:$port/health > /dev/null; then
    echo "✅ Connected"
  else
    echo "❌ Failed"
  fi
done

echo "🚀 Test complete!"
```

### Real-Time Event Monitor
```bash
#!/bin/bash
echo "📡 Monitoring real-time events from all MCP servers..."

# Monitor all event streams in parallel
curl -N -s http://localhost:3001/events | sed 's/^/[Jellyfin] /' &
curl -N -s http://localhost:3002/events | sed 's/^/[Sonarr] /' &  
curl -N -s http://localhost:3003/events | sed 's/^/[Radarr] /' &
curl -N -s http://localhost:3004/events | sed 's/^/[Prowlarr] /' &
curl -N -s http://localhost:3005/events | sed 's/^/[qBittorrent] /' &

wait
```

## 🎯 Use Cases & Examples

### 1. Media Library Dashboard
Connect to Jellyfin MCP to create custom dashboards:
```javascript
// Get library overview
const stats = await jellyfin.callTool('get_library_stats');
const recent = await jellyfin.callTool('get_recent_media', { limit: 10 });

// Real-time updates for new additions
events.addEventListener('media_scan', (event) => {
  updateDashboard(JSON.parse(event.data));
});
```

### 2. Automated Download Management
Use Sonarr + Radarr + qBittorrent together:
```javascript
// Add new movie via Radarr
await radarr.callTool('add_movie', {
  tmdbId: 123456,
  qualityProfileId: 1,
  rootFolderPath: '/movies'
});

// Monitor download progress via qBittorrent
events.addEventListener('torrent_added', async (event) => {
  const torrent = JSON.parse(event.data);
  console.log(`Download started: ${torrent.title}`);
});
```

### 3. Search Optimization with Prowlarr
```javascript
// Test all indexers
const indexers = await prowlarr.callTool('get_indexers');
for (const indexer of indexers.data) {
  await prowlarr.callTool('test_indexer', { indexerId: indexer.id });
}

// Search across all enabled indexers
const results = await prowlarr.callTool('search_indexers', {
  query: 'The Last of Us',
  categories: [5000] // TV category
});
```

## 📞 Support & Troubleshooting

### Common Issues

**Connection Refused**
- Check if Docker container is running: `docker ps`
- Verify port mapping: `docker port mediaserver-ai`
- Check logs: `docker logs mediaserver-ai`

**API Key Errors**
- Verify API keys in `.env` file
- Check service web interfaces for correct keys
- Restart container after updating keys

**Real-Time Events Not Working**
- Ensure client supports Server-Sent Events
- Check firewall settings for ports 3001-3005
- Verify CORS headers in browser developer tools

### Debug Mode
Enable debug logging:
```bash
# Set environment variable
DEBUG=mediaserver:* npm start

# Or in Docker
docker run -e DEBUG=mediaserver:* mediaserver-ai
```

### Health Check Commands
```bash
# Check all services
curl http://localhost:8090/health

# Check specific MCP server
curl http://localhost:3001/health

# Get detailed status
curl http://localhost:8090/api/mcp/status
```

---

## 🎉 Ready to Connect!

Your MediaServer MCP Suite is now fully HTTP/SSE streamable and ready for integration with:

- ✅ **Claude Desktop** (via fetch server)
- ✅ **Custom applications** (via HTTP API)
- ✅ **Real-time dashboards** (via SSE streams)
- ✅ **Automation scripts** (via REST calls)
- ✅ **Third-party tools** (via MCP protocol)

Start with the health checks above, then dive into the tool reference to explore what each service can do! 🚀