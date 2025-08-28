# Ultimate Media Server 2025 - Complete Automation Setup Guide

## 🚀 Quick Start

```bash
# One-command setup
./scripts/quick-start-automation.sh

# Or manually
docker-compose -f docker-compose-automation.yml up -d
```

## 📋 Services Overview

### Core Automation Stack
- **Prowlarr** (Port 9696) - Indexer management for all *arr services
- **Sonarr** (Port 8989) - TV show automation
- **Radarr** (Port 7878) - Movie automation  
- **Lidarr** (Port 8686) - Music automation
- **Readarr** (Port 8787) - Book/audiobook automation
- **Bazarr** (Port 6767) - Subtitle automation
- **Overseerr** (Port 5055) - User request management

### Download Clients
- **qBittorrent** (Port 8080) - Torrent downloads via VPN
- **SABnzbd** (Port 8081) - Usenet downloads

## 🔄 Automation Workflow

```
User Request (Overseerr) 
    ↓
*arr Service (Sonarr/Radarr/etc.) 
    ↓
Prowlarr (Find Release) 
    ↓
Download Client (qBittorrent/SABnzbd) 
    ↓
Media Library (Jellyfin) 
    ↓
Bazarr (Download Subtitles)
```

## ⚙️ Configuration Steps

### 1. Start Services
```bash
# Using the automation-focused compose file
docker-compose -f docker-compose-automation.yml up -d

# Or use the quick start script
./scripts/quick-start-automation.sh
```

### 2. Configure Prowlarr (First!)
1. Access http://localhost:9696
2. Add indexers:
   - **Public Trackers**: 1337x, RARBG, The Pirate Bay
   - **Private Trackers**: Add your private tracker credentials
   - **Usenet**: Add your usenet indexers
3. Add applications:
   - Sonarr: http://sonarr:8989
   - Radarr: http://radarr:7878
   - Lidarr: http://lidarr:8686
   - Readarr: http://readarr:8787
4. Test indexers to ensure they're working

### 3. Configure Download Clients

#### qBittorrent
1. Access http://localhost:8080
2. Default login: admin/adminadmin
3. Configure categories:
   - movies → `/downloads/torrents/movies`
   - tv → `/downloads/torrents/tv`
   - music → `/downloads/torrents/music`
   - books → `/downloads/torrents/books`

#### SABnzbd
1. Access http://localhost:8081
2. Add your usenet provider
3. Configure categories:
   - movies → `/downloads/usenet/movies`
   - tv → `/downloads/usenet/tv`
   - music → `/downloads/usenet/music`
   - books → `/downloads/usenet/books`

### 4. Configure *arr Services

#### Sonarr (TV Shows)
1. Access http://localhost:8989
2. **Settings > Download Clients**:
   - Add qBittorrent: Host: `qbittorrent`, Port: `8080`
   - Add SABnzbd: Host: `sabnzbd`, Port: `8080`
3. **Settings > Media Management**:
   - Root Folder: `/media/tv`
   - Enable: Rename Episodes, Replace Illegal Characters
4. **Settings > Profiles**: Adjust quality profiles as needed
5. **Settings > Indexers**: Should auto-populate from Prowlarr

#### Radarr (Movies)
1. Access http://localhost:7878
2. **Settings > Download Clients**: Same as Sonarr
3. **Settings > Media Management**:
   - Root Folder: `/media/movies`
   - Movie Naming: `{Movie Title} ({Release Year})`
4. **Settings > Profiles**: Adjust quality profiles
5. **Settings > Indexers**: Should auto-populate from Prowlarr

#### Lidarr (Music)
1. Access http://localhost:8686
2. **Settings > Download Clients**: Same as above
3. **Settings > Media Management**:
   - Root Folder: `/media/music`
   - Enable: Rename Tracks
4. **Settings > Profiles**: Set preferred formats (FLAC, MP3, etc.)

#### Readarr (Books)
1. Access http://localhost:8787
2. **Settings > Download Clients**: Same as above
3. **Settings > Media Management**:
   - Root Folders: 
     - `/media/books` (for ebooks)
     - `/media/audiobooks` (for audiobooks)
4. **Settings > Profiles**: Set preferred formats (EPUB, PDF, M4B, etc.)

### 5. Configure Bazarr (Subtitles)
1. Access http://localhost:6767
2. **Settings > Sonarr**: 
   - Enable, Host: `sonarr`, Port: `8989`
   - Add Sonarr API key
3. **Settings > Radarr**: 
   - Enable, Host: `radarr`, Port: `7878`
   - Add Radarr API key
4. **Settings > Subtitles**: 
   - Add providers (OpenSubtitles, Subscene, etc.)
   - Set preferred languages

### 6. Configure Overseerr (Request Management)
1. Access http://localhost:5055
2. **Setup Wizard**:
   - Connect to Jellyfin: `http://jellyfin:8096`
   - Add Jellyfin API key
3. **Settings > Services**:
   - Add Radarr: `http://radarr:7878`
   - Add Sonarr: `http://sonarr:8989`
   - Add respective API keys
4. **Settings > Users**: Configure permissions

## 🛠️ Automated Configuration

Run the automation configuration script after services are started:

```bash
# Wait for services to fully start (2-3 minutes)
./scripts/configure-automation-apis.sh
```

This script will:
- Extract API keys from all services
- Configure download clients in Prowlarr
- Add applications to Prowlarr
- Configure download clients in each *arr service

## 📁 Directory Structure

```
media-data/
├── downloads/
│   ├── torrents/
│   │   ├── movies/         # Completed movie torrents
│   │   ├── tv/             # Completed TV torrents
│   │   ├── music/          # Completed music torrents
│   │   └── books/          # Completed book torrents
│   ├── usenet/
│   │   ├── movies/         # Completed movie usenet
│   │   ├── tv/             # Completed TV usenet
│   │   ├── music/          # Completed music usenet
│   │   └── books/          # Completed book usenet
│   ├── complete/           # General completed downloads
│   └── incomplete/         # Active downloads
├── movies/                 # Final organized movies
├── tv/                     # Final organized TV shows
├── music/                  # Final organized music
├── books/                  # Final organized books
├── audiobooks/             # Final organized audiobooks
├── podcasts/               # Podcast storage
└── comics/                 # Comic storage
```

## 🔧 Quality Profiles

### Sonarr (TV Shows)
- **Web-DL 1080p**: Best quality for most shows
- **HDTV 720p**: Good quality, smaller files
- **Any**: For hard-to-find content

### Radarr (Movies)
- **UHD 2160p**: 4K movies (if you have space/bandwidth)
- **HD 1080p**: Standard high definition
- **HD 720p**: Good quality, smaller files

### Lidarr (Music)
- **Lossless**: FLAC preferred
- **High Quality Lossy**: 320kbps MP3/AAC
- **Standard**: 192-256kbps for mobile

### Readarr (Books)
- **eBook**: EPUB > PDF > MOBI
- **Audiobook**: M4B > MP3

## 🚨 Troubleshooting

### Services Can't Communicate
**Problem**: Services showing connection errors
**Solution**: 
- Check that all containers are running: `docker-compose ps`
- Verify API keys match between services
- Ensure container names are correct in connections

### Downloads Not Starting
**Problem**: Requests sent but no downloads begin
**Solution**:
- Check VPN connection for qBittorrent
- Verify indexers are working in Prowlarr
- Check download client connections in *arr services

### Files Not Moving to Media Folders
**Problem**: Downloads complete but don't import
**Solution**:
- Check path mappings in *arr services
- Verify permissions: `chmod -R 755 media-data/`
- Check logs: `docker-compose logs sonarr`

### No Search Results
**Problem**: Can't find releases for requested content
**Solution**:
- Add more indexers in Prowlarr
- Check indexer status and rate limits
- Verify release naming matches expectations

## 📊 Monitoring

### Check Service Status
```bash
# View all container status
docker-compose ps

# Check specific service logs
docker-compose logs -f sonarr
docker-compose logs -f prowlarr

# Monitor all logs
docker-compose logs -f
```

### Health Checks
- All services include health checks
- Failed health checks indicate service issues
- Restart problematic services: `docker-compose restart service_name`

## 🔒 Security Considerations

### VPN Configuration
- qBittorrent routes through Gluetun VPN
- Configure your VPN provider in `.env`:
  ```
  VPN_PROVIDER=mullvad
  VPN_PRIVATE_KEY=your_key_here
  VPN_ADDRESSES=10.x.x.x/32
  ```

### API Security
- API keys are auto-generated for security
- Change default passwords immediately
- Consider enabling authentication in services for external access

### Network Security
- Services communicate on internal Docker networks
- Only necessary ports exposed to host
- Use reverse proxy (Traefik) for external HTTPS access

## 🎯 Performance Optimization

### Download Optimization
- **qBittorrent**: Limit active torrents (5-10)
- **SABnzbd**: Adjust connections based on provider limits
- **Parallel Downloads**: Limit to 3-5 simultaneous downloads

### Storage Optimization
- **Hardlinks**: Enable in *arr services to save space
- **Recycle Bin**: Enable for safety, set cleanup schedule
- **Monitoring**: Use Grafana dashboards for storage tracking

## 🚀 Advanced Features

### Custom Scripts
- **Post-processing**: Add custom scripts for format conversion
- **Notifications**: Set up Discord/Telegram notifications
- **Backup**: Automated config backups to cloud storage

### Quality Upgrades
- **Automatic Upgrades**: Replace lower quality with higher
- **Cutoff**: Set minimum acceptable quality
- **Preferred Words**: Boost/lower releases with specific terms

## 🎉 Testing Your Setup

1. **Request Content**: Go to Overseerr, request a popular movie/show
2. **Monitor Progress**: Watch logs for download start
3. **Check Downloads**: Verify download appears in qBittorrent/SABnzbd
4. **Import Verification**: Confirm file moves to media folder
5. **Subtitle Check**: Verify Bazarr downloads subtitles
6. **Media Server**: Confirm content appears in Jellyfin

## 📞 Support Resources

- **Logs Location**: `./config/[service]/logs/`
- **Documentation**: Each service has built-in help
- **Community**: Reddit r/selfhosted, Discord servers
- **Updates**: Use `docker-compose pull` to update images

---

**Congratulations!** You now have a fully automated media server that can:
- Automatically download requested movies, TV shows, music, and books
- Organize everything with proper naming
- Download subtitles automatically
- Provide a beautiful interface for users to request content
- Monitor everything through comprehensive dashboards

Enjoy your automated media experience! 🍿🎬🎵📚