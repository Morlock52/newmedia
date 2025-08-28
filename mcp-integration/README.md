# MCP Integration for Media Server

## Overview

This MCP (Model Context Protocol) integration provides a unified interface for managing and coordinating all media server services through intelligent agents and external APIs.

## Current Status

### ✅ Available MCP Servers
- **claude-flow** - Swarm orchestration and agent coordination
- **ruv-swarm** - Distributed consensus and coordination

### ⚠️ Not Configured
- **Archon** - Would provide task management (currently not installed)
- **Serena** - Would provide voice/audio processing (not configured)

## Architecture

```
┌─────────────────────────────────────────────┐
│           MCP Integration Layer             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐  ┌───────────┐  ┌─────────┐ │
│  │ Claude   │  │   RUV     │  │ Media   │ │
│  │  Flow    │  │  Swarm    │  │ Unified │ │
│  └──────────┘  └───────────┘  └─────────┘ │
│                                             │
├─────────────────────────────────────────────┤
│              Service Layer                  │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐  ┌───────────┐  ┌─────────┐ │
│  │ Jellyfin │  │   Plex    │  │ Sonarr  │ │
│  └──────────┘  └───────────┘  └─────────┘ │
│                                             │
│  ┌──────────┐  ┌───────────┐  ┌─────────┐ │
│  │  Radarr  │  │ Prowlarr  │  │ Lidarr  │ │
│  └──────────┘  └───────────┘  └─────────┘ │
│                                             │
│  ┌──────────┐  ┌───────────┐  ┌─────────┐ │
│  │qBittorrent│ │  SABnzbd  │  │ Bazarr  │ │
│  └──────────┘  └───────────┘  └─────────┘ │
│                                             │
└─────────────────────────────────────────────┘
```

## Features

### MCP Tools
- **media.scan** - Scan media libraries for new content
- **media.transcode** - Transcode media to different formats
- **media.metadata** - Fetch or update media metadata
- **indexer.search** - Search for media across indexers
- **indexer.monitor** - Monitor media for new releases
- **download.add** - Add downloads to clients
- **download.status** - Get download status
- **swarm.coordinate** - Coordinate tasks across services

### MCP Resources
- `media://library` - Unified media library
- `indexer://search` - Unified search across indexers
- `download://queue` - Unified download queue
- `system://status` - System-wide status and health

## Installation

1. **Install dependencies:**
```bash
npm install axios
```

2. **Configure environment variables:**
```bash
cp .env.template .env
# Edit .env with your API keys
```

3. **Start media services:**
```bash
docker-compose up -d
```

4. **Test MCP integration:**
```bash
node mcp-integration/test-mcp-integration.js
```

## Usage

### Basic Example
```javascript
const MCPClient = require('./mcp-integration/MCPClient');

async function main() {
  const client = new MCPClient();
  
  // Initialize connections
  await client.initialize();
  
  // Search for media
  const results = await client.executeTool('indexer.search', {
    query: 'Breaking Bad',
    type: 'series'
  });
  
  // Add to downloads
  await client.executeTool('download.add', {
    url: results[0].downloadUrl,
    category: 'tv'
  });
  
  // Scan libraries
  await client.executeTool('media.scan', {
    library: 'all'
  });
  
  // Cleanup
  await client.cleanup();
}
```

### Advanced Coordination
```javascript
// Coordinate multiple services
await client.executeTool('swarm.coordinate', {
  task: 'Process new episode',
  services: ['sonarr', 'qbittorrent', 'jellyfin', 'bazarr'],
  strategy: 'sequential'
});
```

## Configuration

Edit `mcp-integration/media-server-mcp-config.js` to customize:
- Service endpoints
- API keys
- Tool definitions
- Resource mappings
- Error handling
- Monitoring settings

## Testing

Run the test suite:
```bash
node mcp-integration/test-mcp-integration.js
```

Expected output:
- MCP server connections
- Service availability
- Tool functionality
- Health status

## Troubleshooting

### MCP Servers Not Available
```bash
# Install Claude Flow
npx claude-flow@alpha mcp start

# Install RUV Swarm
npx ruv-swarm@latest mcp start
```

### Services Not Connecting
1. Check Docker containers: `docker ps`
2. Verify API keys in `.env`
3. Test service endpoints manually
4. Check firewall/network settings

### Archon Not Available
Archon requires separate installation:
```bash
# Clone Archon repository
git clone https://github.com/Archon-AI/archon.git ~/archon
cd ~/archon
docker-compose up -d
```

## API Reference

### MCPClient Methods

- `initialize()` - Initialize all connections
- `executeTool(name, params)` - Execute an MCP tool
- `getResource(uri)` - Get resource data
- `healthCheck()` - Check system health
- `cleanup()` - Close all connections

### Events

- `initialized` - All connections ready
- `server:connected` - MCP server connected
- `service:connected` - Service connected
- `cleanup` - Cleanup complete

## Security

- API keys stored in environment variables
- Bearer token authentication
- Circuit breaker for failed connections
- Request timeout protection
- Input validation on all tools

## Performance

- Connection pooling for services
- Exponential backoff for retries
- Parallel service coordination
- Resource caching
- Lazy connection initialization

## Next Steps

1. **Complete Archon Integration** - Primary task management
2. **Add Serena MCP** - Voice/audio processing
3. **Implement Memory Persistence** - Cross-session state
4. **Enhanced Authentication** - OAuth2/JWT support
5. **Monitoring Dashboard** - Real-time metrics
6. **WebSocket Support** - Real-time updates
7. **GraphQL API** - Flexible queries
8. **Plugin System** - Extensible architecture

## Support

- Check logs: `docker logs <container>`
- MCP status: `node mcp-integration/test-mcp-integration.js`
- Service health: `curl http://localhost:<port>/health`
- Debug mode: Set `LOG_LEVEL=debug` in `.env`