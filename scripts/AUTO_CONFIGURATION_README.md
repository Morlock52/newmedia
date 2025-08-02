# 🚀 Media Server Auto-Configuration Scripts

This directory contains powerful automation scripts that configure your entire media server stack automatically.

## 📋 Available Scripts

### 1. **auto-configure-all-services.sh** (Complete Setup)
The most comprehensive configuration script that:
- ✅ Detects all running Docker services
- ✅ Configures Prowlarr with free indexers (1337x, TPB, YTS, etc.)
- ✅ Connects all ARR apps to Prowlarr
- ✅ Sets up download clients (qBittorrent, Transmission, SABnzbd)
- ✅ Connects ARR apps to download clients
- ✅ Configures media server libraries
- ✅ Sets up request services (Jellyseerr/Overseerr)
- ✅ Configures monitoring (Prometheus, Grafana, Uptime Kuma)
- ✅ Creates management scripts

**Usage:**
```bash
./scripts/auto-configure-all-services.sh

# Or specify a custom docker-compose file:
./scripts/auto-configure-all-services.sh /path/to/docker-compose.yml
```

### 2. **quick-setup.sh** (Essential Services Only)
A faster script that configures only the essential services:
- ✅ Starts core services (Prowlarr, Sonarr, Radarr, Jellyfin, qBittorrent)
- ✅ Adds 1337x indexer to Prowlarr
- ✅ Configures qBittorrent with optimal settings
- ✅ Takes only ~30 seconds to run

**Usage:**
```bash
./scripts/quick-setup.sh
```

### 3. **check-configuration-status.sh** (Status Monitor)
Shows the current configuration state of all services:
- ✅ Service health status
- ✅ API connectivity checks
- ✅ Integration status
- ✅ Storage usage
- ✅ Configuration summary

**Usage:**
```bash
./scripts/check-configuration-status.sh
```

### 4. **auto-configure-arr-stack.sh** (Original ARR Script)
The original script focused on ARR stack configuration with Recyclarr integration.

## 🎯 Features

### Automatic Service Detection
- Scans for running Docker containers
- Adapts configuration based on available services
- No manual service specification needed

### Smart API Key Management
- Automatically extracts API keys from service configs
- Generates keys if needed
- Handles different key storage methods

### Idempotent Operations
- Safe to run multiple times
- Skips already configured services
- Won't duplicate configurations

### Comprehensive Logging
- All operations logged to `logs/auto-config/`
- Color-coded console output
- Progress indicators

### Error Handling
- Graceful failure handling
- Service availability checks
- Timeout protection

## 📦 What Gets Configured

### Prowlarr
- Free public indexers:
  - 1337x
  - The Pirate Bay
  - RARBG
  - YTS
  - EZTV
  - Nyaa
  - LimeTorrents
  - TorrentGalaxy

### ARR Applications
- Connected to Prowlarr for indexer management
- Connected to download clients
- Proper category configuration
- API key synchronization

### Download Clients
- **qBittorrent**: Admin UI enabled, optimal settings
- **Transmission**: RPC enabled, no authentication
- **SABnzbd**: Category management configured

### Media Servers
- Directory structure created
- Library paths configured
- Initial setup guidance provided

### Monitoring
- Prometheus targets configured
- Grafana data sources added
- Basic dashboards created
- Uptime monitoring ready

### Dashboards
- Homarr auto-detection enabled
- Homepage services configured
- Service widgets with API integration

## 🛠️ Created Management Scripts

After running the auto-configuration, you'll have these additional scripts:

### backup-configs.sh
Backs up all service configurations to timestamped archives.

### health-check.sh
Checks the health status of all services and containers.

### update-all-services.sh
Updates all services to their latest versions.

## 🔧 Prerequisites

1. **Docker & Docker Compose** installed and running
2. **Services started** via docker-compose
3. **Write permissions** in the project directory
4. **curl** command available

## 📝 Post-Configuration Steps

### 1. Media Servers
- **Jellyfin**: Complete setup at http://localhost:8096
- **Plex**: Get claim token from https://www.plex.tv/claim

### 2. Request Services
- **Jellyseerr**: Connect to Jellyfin and ARR apps
- **Overseerr**: Connect to Plex and ARR apps

### 3. Quality Profiles
- Adjust quality settings in Sonarr/Radarr
- Configure preferred release groups
- Set up custom formats

### 4. Storage Paths
- Verify media folder permissions
- Configure root folders in ARR apps
- Set up remote path mappings if needed

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check logs
docker logs <service-name>

# Restart service
docker restart <service-name>
```

### API Key Issues
```bash
# Manually get API key
docker exec <service> cat /config/config.xml | grep ApiKey
```

### Connection Failures
- Ensure services are on the same Docker network
- Check firewall rules
- Verify port availability

## 🔒 Security Notes

- Default passwords are used for initial setup
- **Change all default passwords** after configuration
- Consider using environment variables for sensitive data
- Enable authentication on all services

## 📊 Performance Tips

- Run configuration during low-usage times
- Monitor resource usage during setup
- Consider staging configuration for large setups
- Use SSD storage for config directories

## 🎉 Success Indicators

You'll know configuration is successful when:
- ✅ All services show green health status
- ✅ Prowlarr shows connected applications
- ✅ ARR apps show available indexers
- ✅ Download clients appear in ARR apps
- ✅ Dashboards display service information

## 📚 Additional Resources

- [TRaSH Guides](https://trash-guides.info/) - Quality settings
- [WikiArr](https://wiki.servarr.com/) - Service documentation
- [r/selfhosted](https://reddit.com/r/selfhosted) - Community support

---

**Happy Streaming! 🎬🎵📺**