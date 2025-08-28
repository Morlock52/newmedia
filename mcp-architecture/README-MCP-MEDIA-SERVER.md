# Media Services MCP Server

A production-ready Model Context Protocol (MCP) server for managing media services including Jellyfin, Sonarr, Radarr, Prowlarr, qBittorrent, Bazarr, and Lidarr.

## Features

🎬 **Complete Media Management**
- Jellyfin media streaming server integration
- Sonarr for TV show automation
- Radarr for movie automation  
- Lidarr for music automation
- Prowlarr for indexer management
- qBittorrent for download management
- Bazarr for subtitle management

🚀 **Production-Ready**
- Full MCP 1.0 protocol compliance
- Stdio transport for Claude Desktop
- Comprehensive error handling
- Performance caching
- Authentication support
- Graceful degradation

🛠️ **Rich Tool Set**
- System status monitoring
- Media search across services
- Library statistics
- Recent activity tracking
- Download management
- Media requests
- Subtitle management
- Indexer management
- Release calendar

## Quick Start

### 1. Test the Server

```bash
# Make executable
chmod +x mcp-media-server.js

# Test syntax
node -c mcp-media-server.js

# Run tests
node test-mcp-media-server.js
```

### 2. Configure Claude Desktop

Add to your Claude Desktop configuration file (`~/.claude/config.json`):

```json
{
  "mcpServers": {
    "media-services": {
      "command": "node",
      "args": ["/path/to/mcp-media-server.js"],
      "env": {
        "MCP_DEBUG": "true",
        "JELLYFIN_URL": "http://localhost:8096",
        "JELLYFIN_API_KEY": "your_jellyfin_api_key_here",
        "SONARR_URL": "http://localhost:8989",
        "SONARR_API_KEY": "your_sonarr_api_key_here",
        "RADARR_URL": "http://localhost:7878",
        "RADARR_API_KEY": "your_radarr_api_key_here",
        "PROWLARR_URL": "http://localhost:9696",
        "PROWLARR_API_KEY": "your_prowlarr_api_key_here",
        "QBITTORRENT_URL": "http://localhost:8080",
        "QBITTORRENT_USER": "admin",
        "QBITTORRENT_PASS": "your_password",
        "BAZARR_URL": "http://localhost:6767",
        "BAZARR_API_KEY": "your_bazarr_api_key_here",
        "LIDARR_URL": "http://localhost:8686",
        "LIDARR_API_KEY": "your_lidarr_api_key_here"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

After updating the configuration, restart Claude Desktop to load the MCP server.

## Available Tools

### System Management
- **`get_system_status`** - Get overall system status for all media services
- **`get_library_stats`** - Get comprehensive library statistics
- **`get_recent_activity`** - Get recent media activity and additions

### Media Discovery
- **`search_media`** - Search for movies, TV shows, music across all services
- **`get_calendar`** - Get upcoming releases calendar

### Download Management
- **`manage_downloads`** - View and manage active downloads
- **`add_media_request`** - Add a new media request (movie/TV show/music)

### Advanced Features
- **`manage_subtitles`** - Search and download subtitles for media
- **`manage_indexers`** - View and manage torrent/usenet indexers

## Available Resources

Access media information directly:
- `media://system/status` - System status information
- `media://library/stats` - Library statistics
- `media://activity/recent` - Recent activity
- `media://downloads/active` - Active downloads
- `media://calendar/upcoming` - Upcoming releases
- `media://indexers/status` - Indexer status
- `media://config/services` - Service configuration

## Environment Variables

Configure your media services using these environment variables:

### Jellyfin
- `JELLYFIN_URL` - Jellyfin server URL (default: http://localhost:8096)
- `JELLYFIN_API_KEY` - Jellyfin API key

### Sonarr
- `SONARR_URL` - Sonarr server URL (default: http://localhost:8989)
- `SONARR_API_KEY` - Sonarr API key

### Radarr
- `RADARR_URL` - Radarr server URL (default: http://localhost:7878)
- `RADARR_API_KEY` - Radarr API key

### Prowlarr
- `PROWLARR_URL` - Prowlarr server URL (default: http://localhost:9696)
- `PROWLARR_API_KEY` - Prowlarr API key

### qBittorrent
- `QBITTORRENT_URL` - qBittorrent web UI URL (default: http://localhost:8080)
- `QBITTORRENT_USER` - qBittorrent username (default: admin)
- `QBITTORRENT_PASS` - qBittorrent password

### Bazarr
- `BAZARR_URL` - Bazarr server URL (default: http://localhost:6767)
- `BAZARR_API_KEY` - Bazarr API key

### Lidarr
- `LIDARR_URL` - Lidarr server URL (default: http://localhost:8686)
- `LIDARR_API_KEY` - Lidarr API key

### Debug
- `MCP_DEBUG` - Enable debug logging (set to "true")

## Example Usage

Once configured in Claude Desktop, you can interact with your media services:

```
Claude: Get the status of all my media services

Claude: Search for "Breaking Bad" across all services

Claude: Show me recent activity in the last 24 hours

Claude: What new movies are being released this week?

Claude: Add "The Matrix" to my movie collection

Claude: Show me active downloads

Claude: Get library statistics for all services
```

## Error Handling

The server includes comprehensive error handling:

- **Connection Errors**: Services that are down show error status
- **Authentication Errors**: Invalid API keys are reported clearly  
- **Timeout Handling**: Requests timeout after 10 seconds
- **Graceful Degradation**: Server continues working even if some services fail
- **Detailed Logging**: Debug mode provides detailed request/response logging

## Performance Features

- **Caching**: Responses are cached for 5 minutes to improve performance
- **Keep-Alive**: Server stays running with proper process management
- **Concurrent Requests**: Multiple API calls handled efficiently
- **Resource Optimization**: Minimal memory footprint

## Security

- **No Hardcoded Secrets**: All credentials via environment variables
- **Request Validation**: Input validation for all tool parameters
- **Error Sanitization**: Sensitive information filtered from error messages
- **Timeout Protection**: Prevents hanging requests

## Testing

Run the comprehensive test suite:

```bash
node test-mcp-media-server.js
```

This tests:
- MCP protocol compliance
- Tool registration
- Resource availability  
- Error handling
- Request/response flow

## Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check syntax
node -c mcp-media-server.js

# Enable debug mode
MCP_DEBUG=true node mcp-media-server.js
```

**Services showing as errors:**
- Verify service URLs are accessible
- Check API keys are correct
- Ensure services are running
- Review firewall settings

**Claude Desktop not connecting:**
- Verify config file path and syntax
- Restart Claude Desktop after config changes
- Check Claude Desktop logs

### Debug Mode

Enable detailed logging:
```bash
export MCP_DEBUG=true
```

This shows:
- All incoming requests
- API calls to media services
- Response processing
- Error details
- Cache hit/miss information

## Architecture

The server is built with:
- **Node.js** native modules (no external dependencies)
- **Stdio transport** for Claude Desktop integration
- **Modular design** for easy maintenance
- **Production-ready** error handling and logging
- **Extensible** for adding new media services

## API Coverage

### Jellyfin
- System information
- Library statistics
- Media search
- Recent additions

### Sonarr/Radarr/Lidarr
- Series/Movie/Artist lookup
- Library management
- Calendar/upcoming releases
- History tracking
- Quality profiles

### Prowlarr
- Indexer management
- Search capabilities
- Statistics

### qBittorrent
- Download monitoring
- Torrent management
- Transfer statistics

### Bazarr
- Subtitle search
- Missing subtitles
- Download management

## Contributing

The server is designed to be easily extensible. To add new services:

1. Add service configuration to the `services` object
2. Implement API methods in the respective handler functions  
3. Add new tools to the `_defineTools()` method
4. Update tests to include new functionality

## License

MIT License - feel free to use and modify for your needs.