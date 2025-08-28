# Unified MCP Server

## 🌟 Overview

The Unified MCP Server provides a single interface to manage all your media services through Claude Desktop.

## 🔧 Available Tools

### Service Management
- `unified_health_check` - Check health of all services
- `unified_restart_all` - Restart all services (requires confirmation)
- `unified_backup_configs` - Backup all configurations
- `unified_sync_libraries` - Synchronize libraries between services
- `unified_get_statistics` - Get comprehensive statistics

### Docker Operations
- `docker_list_containers` - List all containers
- `docker_container_logs` - Get container logs
- `docker_restart_container` - Restart specific container

### Service-Specific Tools
Each discovered service gets its own set of tools:
- `{service}_status` - Get service status
- `{service}_api_call` - Make custom API calls

### Arr Services (Sonarr, Radarr, Lidarr)
- `{service}_add_media` - Add new media
- `{service}_get_media` - Get media library
- `{service}_search` - Search for media

### Jellyfin
- `jellyfin_get_libraries` - Get all libraries
- `jellyfin_get_items` - Get library items
- `jellyfin_scan_library` - Trigger library scan

### Prowlarr
- `prowlarr_get_indexers` - Get all indexers
- `prowlarr_test_indexer` - Test indexer
- `prowlarr_search` - Search across indexers

### Download Clients (qBittorrent, Transmission)
- `{service}_get_torrents` - List torrents
- `{service}_add_torrent` - Add new torrent
- `{service}_control_torrent` - Control torrent

## 🚀 Usage Examples

### Check All Services
```
Use the unified_health_check tool to see the status of all services
```

### Restart a Container
```
Use docker_restart_container with container name "sonarr"
```

### Get Service Statistics
```
Use unified_get_statistics to see comprehensive system stats
```

## 📋 Management Commands

- `./start.sh` - Start the MCP server
- `./stop.sh` - Stop the MCP server  
- `./status.sh` - Check server status
- `npm test` - Run tests
- `npm run dev` - Start in development mode

## 🔧 Configuration

Edit `unified-mcp-config.json` to customize:
- Service endpoints and ports
- Health check intervals
- Docker settings
- Security options
- Logging preferences

Generated on: 2025-08-03T10:59:49.252Z
