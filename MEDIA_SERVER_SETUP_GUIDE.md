# Media Server Setup Guide

## 🚀 Current Status
All core services are running and accessible. Initial configuration is required.

## 📊 Service Access URLs

### Media Servers
- **Jellyfin**: http://localhost:8096
  - Status: ⚠️ Needs initial setup wizard
  - Default login will be created during setup
  
- **Plex**: http://localhost:32400/web
  - Status: Not deployed yet

### Arr Services (Automation)
- **Sonarr** (TV Shows): http://localhost:8989
  - Status: ✅ Running - needs configuration
  
- **Radarr** (Movies): http://localhost:7878
  - Status: ✅ Running - needs configuration
  
- **Prowlarr** (Indexer Manager): http://localhost:9696
  - Status: ✅ Running - needs configuration
  
- **Lidarr** (Music): http://localhost:8686
  - Status: Not deployed yet
  
- **Bazarr** (Subtitles): http://localhost:6767
  - Status: Not deployed yet

### Download Clients
- **qBittorrent**: http://localhost:8080
  - Status: ✅ Running
  - Default login: admin/adminadmin

### Management Tools
- **Portainer**: http://localhost:9443
  - Status: Not deployed yet
  
- **Nginx Proxy Manager**: http://localhost:81
  - Status: Not deployed yet

## 🔧 Initial Setup Steps

### 1. Jellyfin Setup
1. Navigate to http://localhost:8096
2. Complete the setup wizard:
   - Set language
   - Create admin user (suggested: admin/admin123)
   - Configure media libraries (can be done later)
   - Enable remote access
3. Save the API key from Dashboard > API Keys

### 2. Sonarr Setup
1. Navigate to http://localhost:8989
2. Go to Settings > General
3. Copy the API Key
4. Configure:
   - Settings > Media Management > Root Folders: `/media/tv`
   - Settings > Profiles > Quality Profiles (keep defaults)
   - Settings > Download Clients > Add qBittorrent

### 3. Radarr Setup
1. Navigate to http://localhost:7878
2. Go to Settings > General
3. Copy the API Key
4. Configure:
   - Settings > Media Management > Root Folders: `/media/movies`
   - Settings > Profiles > Quality Profiles (keep defaults)
   - Settings > Download Clients > Add qBittorrent

### 4. Prowlarr Setup
1. Navigate to http://localhost:9696
2. Go to Settings > General
3. Copy the API Key
4. Add Indexers:
   - Settings > Indexers > Add (choose public trackers)
5. Sync to Apps:
   - Settings > Apps > Add Sonarr
   - Settings > Apps > Add Radarr

### 5. qBittorrent Configuration
1. Navigate to http://localhost:8080
2. Login with admin/adminadmin
3. Go to Settings > Web UI
4. Enable "Bypass authentication for clients on localhost"
5. Configure download paths:
   - Default Save Path: `/downloads/complete`
   - Temp path: `/downloads/incomplete`

## 📁 Directory Structure
```
/media/
├── tv/         # TV shows for Sonarr
├── movies/     # Movies for Radarr
├── music/      # Music for Lidarr
└── books/      # Books for Readarr

/downloads/
├── complete/   # Completed downloads
├── incomplete/ # In-progress downloads
└── torrents/   # Torrent files
```

## 🔄 Integration Workflow
1. **Prowlarr** searches indexers for content
2. **Sonarr/Radarr** sends download requests to qBittorrent
3. **qBittorrent** downloads to `/downloads/`
4. **Sonarr/Radarr** moves completed files to `/media/`
5. **Jellyfin** scans media folders and serves content

## 🔑 API Keys Storage
After setup, API keys will be stored in:
- Sonarr: `/sonarr-config/config.xml`
- Radarr: `/radarr-config/config.xml`
- Prowlarr: `/prowlarr-config/config.xml`
- Jellyfin: Dashboard > API Keys

## 🚨 Troubleshooting

### Services not accessible
```bash
# Check service status
docker ps

# View logs
docker logs [service-name]

# Restart service
docker restart [service-name]
```

### Permission issues
```bash
# Fix permissions
sudo chown -R $USER:$USER ./media ./downloads
```

### Network issues
```bash
# Check Docker networks
docker network ls

# Inspect network
docker network inspect media-net
```

## 📈 Next Steps
1. Complete initial setup for each service
2. Add indexers in Prowlarr
3. Configure quality profiles
4. Test download workflow
5. Setup media libraries in Jellyfin
6. Deploy monitoring stack

## 🎯 Using Claude Flow for Coordination

The project is being managed using Claude Flow MCP with:
- **Swarm ID**: swarm_1755251976479_9h1eoxsgx
- **Topology**: Hierarchical
- **Active Agents**: 3 (Coordinator, Architect, Media Expert)

To check progress:
```bash
# View current tasks
curl http://localhost:8051/health  # If local Archon was running
```

Currently using claude-flow MCP for task orchestration and memory management.