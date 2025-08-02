# Media Server Automation Guide - August 2025

## 🚀 Quick Start

The fastest way to get your media server running:

```bash
# Make the quick setup script executable
chmod +x scripts/quick-setup-arr-stack.sh

# Run the quick setup
./scripts/quick-setup-arr-stack.sh

# Start the media server stack
./start-media-server.sh
```

This will deploy a complete media server with:
- **Prowlarr** - Indexer management
- **Sonarr** - TV show automation
- **Radarr** - Movie automation
- **Lidarr** - Music automation
- **Bazarr** - Subtitle management
- **qBittorrent** - Download client
- **Jellyfin** - Media streaming
- **Jellyseerr** - Request management
- **Homarr** - Beautiful dashboard

## 📋 2025 Best Practices & Automation

### 1. **Prowlarr Free Indexers (Working in 2025)**

Popular free public indexers that are currently working:

**Torrent Indexers:**
- 1337x - General purpose, very reliable
- The Pirate Bay - Oldest and most comprehensive
- RARBG - High-quality releases
- YTS - Movies with small file sizes
- EZTV - TV show specialized
- Nyaa - Anime, manga, and Asian media
- TorrentGalaxy - Modern tracker with good UI
- LimeTorrents - General purpose
- Torlock - Verified torrents only

**Adding Indexers in Prowlarr:**
1. Go to Settings → Indexers → Add Indexer
2. Search for the indexer name
3. Click on it and test the connection
4. Save if the test passes

### 2. **Recyclarr - Auto-sync TRaSH Guides**

Recyclarr automatically syncs recommended settings from TRaSH Guides:

```bash
# Install Recyclarr
docker run --rm \
    -v ./config/recyclarr:/config \
    -e SONARR_API_KEY=your_sonarr_api_key \
    -e RADARR_API_KEY=your_radarr_api_key \
    ghcr.io/recyclarr/recyclarr:latest sync
```

### 3. **Optimal ARR Settings for 2025**

**Sonarr (TV Shows):**
- Episode Format: `{Series TitleYear} - S{season:00}E{episode:00} - {Episode CleanTitle} [{Quality Full}]`
- Enable: Rename Episodes, Import Extra Files, Delete Empty Folders
- Quality Profile: HD-1080p for most users, Ultra-HD for 4K setups

**Radarr (Movies):**
- Movie Format: `{Movie CleanTitle} ({Release Year}) {imdbid-{ImdbId}} [{Quality Full}]`
- Enable: Rename Movies, Import Extra Files
- Quality Profile: HD-1080p with Bluray-1080p as cutoff

**Lidarr (Music):**
- Track Format: `{Artist CleanName} - {Album Title} - {track:00} - {Track Title}`
- Quality Profile: FLAC for lossless, MP3-320 for standard

### 4. **Download Client Configuration**

**qBittorrent Optimal Settings:**
- Default Save Path: `/downloads/complete/`
- Temp Path: `/downloads/incomplete/`
- Enable: Pre-allocate all files, Append .!qB extension
- Connection: Port 6881, Max connections 200
- BitTorrent: Enable DHT, PEX, LSD
- Queueing: Max active downloads 3, Max active uploads 5

### 5. **Automation Scripts**

We've included several automation scripts:

- **auto-configure-arr-stack.sh** - Full automatic configuration
- **quick-setup-arr-stack.sh** - Quick deployment script
- **configure-services.sh** - Service configuration
- **update-services.sh** - Update all containers
- **backup-configs.sh** - Backup configurations
- **health-check.sh** - Check service health

## 🔧 Advanced Configuration

### Using VPN with Downloads

To route downloads through VPN, uncomment the gluetun service in docker-compose.yml and configure:

```yaml
environment:
  VPN_SERVICE_PROVIDER: nordvpn  # or your provider
  OPENVPN_USER: your_username
  OPENVPN_PASSWORD: your_password
  SERVER_COUNTRIES: Switzerland
```

Then change qBittorrent to use gluetun's network:

```yaml
qbittorrent:
  network_mode: "service:gluetun"
```

### Custom Indexers

For indexers not built into Prowlarr, you can add custom definitions:

1. Go to Settings → Indexers → Add Indexer
2. Select "Generic Newznab" (for Usenet) or "Generic Torznab" (for torrents)
3. Enter the indexer's URL and API key

### Media Organization

Recommended folder structure:

```
/media
├── movies
│   └── Movie Name (Year)
├── tv
│   └── Show Name
│       └── Season 01
├── music
│   └── Artist Name
│       └── Album Name
└── downloads
    ├── complete
    ├── incomplete
    └── torrents
```

## 🛠️ Troubleshooting

### Common Issues

**Services not accessible:**
- Check if containers are running: `docker-compose ps`
- Check logs: `docker-compose logs [service-name]`
- Ensure ports aren't already in use

**Indexers not working:**
- Some indexers may be blocked by ISPs
- Try using a VPN or different DNS servers
- Check if the indexer is still active

**Downloads not importing:**
- Verify download paths match between download client and ARR apps
- Check permissions (PUID/PGID)
- Ensure hardlinks are possible (same filesystem)

## 📚 Additional Resources

- [TRaSH Guides](https://trash-guides.info/) - The bible of ARR configuration
- [WikiArr](https://wiki.servarr.com/) - Official documentation
- [r/selfhosted](https://reddit.com/r/selfhosted) - Community support
- [Awesome-Selfhosted](https://github.com/awesome-selfhosted/awesome-selfhosted) - More self-hosted apps

## 🎯 Quick Commands Reference

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Update all services
docker-compose pull && docker-compose up -d

# Restart a specific service
docker-compose restart [service-name]

# Execute command in container
docker exec -it [container-name] bash
```

## 🔐 Security Best Practices

1. **Change all default passwords** in the .env file
2. **Use HTTPS** with a reverse proxy like Nginx Proxy Manager
3. **Enable authentication** on all services
4. **Regular backups** of configuration directories
5. **Keep containers updated** with Watchtower or manual updates
6. **Use VPN** for download clients if needed

## 🎬 Enjoy Your Media Server!

Your automated media server is now ready. Add your favorite shows and movies to Sonarr/Radarr, and they'll automatically download and organize themselves. Access everything through Jellyfin on any device!

For requests, share Jellyseerr with family/friends so they can request content without accessing the admin panels.