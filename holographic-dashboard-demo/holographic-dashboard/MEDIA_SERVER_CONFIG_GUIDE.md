# Complete Media Server Stack Configuration Guide - 2025 Edition

## 🎯 Overview

This guide provides the essential environment variables and configuration steps for setting up a complete media server stack in 2025. Based on current best practices and the latest Docker images.

## 📋 Quick Setup Checklist

1. Copy `.env.example` to `.env`
2. Update system configuration (PUID, PGID, TZ, paths)
3. Configure network and volume paths
4. Choose your media server (Jellyfin recommended)
5. Set up *arr stack with Prowlarr
6. Configure download clients
7. Set up request management (Overseerr/Jellyseerr)
8. Configure notifications
9. Set up monitoring and dashboards
10. Implement security and backups

## 🚀 Recommended Stack for 2025

### Core Media Server
- **Jellyfin** (Primary recommendation - fully open-source, excellent hardware transcoding)
- Plex (Alternative - $249.99 lifetime, has remote restrictions)
- Emby (Balanced commercial/open-source option)

### Management Stack (*arr Suite)
- **Prowlarr** - Indexer manager (replaces Jackett)
- **Sonarr** - TV show management
- **Radarr** - Movie management
- **Lidarr** - Music management
- **Readarr** - Book/audiobook management
- **Bazarr** - Subtitle management

### Download Clients
- **qBittorrent** (Preferred - better performance than Transmission)
- Transmission (Alternative)
- SABnzbd (Usenet)

### Request & Discovery
- **Jellyseerr** (For Jellyfin/Emby)
- Overseerr (For Plex)

### Monitoring & Management
- **Tautulli** (Media server analytics)
- **Homepage/Homarr** (Dashboard)
- **Portainer** (Docker management)

## 📁 Directory Structure Best Practices

```
/opt/docker/               # Docker configurations
├── config/               # Application configs
│   ├── jellyfin/
│   ├── sonarr/
│   ├── radarr/
│   └── ...
└── compose.yml

/mnt/media/               # Media storage
├── movies/              # Organized movies
├── tv/                  # Organized TV shows
├── music/               # Music library
├── books/               # Book collection
└── downloads/           # Download staging
    ├── complete/        # Finished downloads
    ├── incomplete/      # In-progress
    └── watch/           # Watch folder
```

## 🔧 Service-Specific Configuration

### Jellyfin (Recommended)

```yaml
services:
  jellyfin:
    image: lscr.io/linuxserver/jellyfin:latest
    container_name: jellyfin
    environment:
      - PUID=${PUID}
      - PGID=${PGID}
      - TZ=${TZ}
      - JELLYFIN_PublishedServerUrl=${JELLYFIN_PUBLISHED_SERVER_URL}
    volumes:
      - ${JELLYFIN_CONFIG_DIR}:/config
      - ${TV_SHOWS_PATH}:/data/tvshows
      - ${MOVIES_PATH}:/data/movies
      - ${MUSIC_PATH}:/data/music
    ports:
      - "8096:8096"
      - "8920:8920" # Optional HTTPS
      - "7359:7359/udp" # Optional discovery
      - "1900:1900/udp" # Optional DLNA
    restart: unless-stopped
```

**Key Features:**
- Free and open-source
- Superior hardware transcoding support
- Extensive plugin ecosystem
- No artificial limitations

### Prowlarr (Indexer Manager)

```yaml
services:
  prowlarr:
    image: lscr.io/linuxserver/prowlarr:latest
    container_name: prowlarr
    environment:
      - PUID=${PUID}
      - PGID=${PGID}
      - TZ=${TZ}
    volumes:
      - ${PROWLARR_CONFIG}:/config
    ports:
      - "${PROWLARR_PORT}:9696"
    restart: unless-stopped
```

**Setup Steps:**
1. Access WebUI at `http://localhost:9696`
2. Add indexers (public and private trackers)
3. Configure *arr app connections
4. Copy API key from Settings > Security

### qBittorrent (Preferred Download Client)

```yaml
services:
  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=${PUID}
      - PGID=${PGID}
      - TZ=${TZ}
      - WEBUI_PORT=${QBITTORRENT_WEBUI_PORT}
    volumes:
      - ${QBITTORRENT_CONFIG}:/config
      - ${DOWNLOADS_ROOT}:/downloads
    ports:
      - "${QBITTORRENT_PORT}:8080"
      - "6881:6881"
      - "6881:6881/udp"
    restart: unless-stopped
```

**Important Notes:**
- For versions ≥4.6.1: Check logs for temporary password on first run
- Login with `admin` and the temporary password
- **IMMEDIATELY** set a permanent password in WebUI
- For older versions: Default is `admin`/`adminadmin`

### *arr Stack Configuration

All *arr applications follow similar patterns:

```yaml
services:
  sonarr:
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=${PUID}
      - PGID=${PGID}
      - TZ=${TZ}
    volumes:
      - ${SONARR_CONFIG}:/config
      - ${TV_SHOWS_PATH}:/tv
      - ${DOWNLOADS_ROOT}:/downloads
    ports:
      - "${SONARR_PORT}:8989"
    restart: unless-stopped
```

**Critical Path Configuration:**
- Root folders: `/tv`, `/movies`, `/music`, `/books`
- Download client path: `/downloads`
- **Must use consistent paths** across all containers

### Jellyseerr (Request Management)

```yaml
services:
  jellyseerr:
    image: fallenbagel/jellyseerr:latest
    container_name: jellyseerr
    environment:
      - LOG_LEVEL=${JELLYSEERR_LOG_LEVEL}
      - TZ=${TZ}
      - API_KEY=${JELLYSEERR_API_KEY} # 2025 feature!
    volumes:
      - ${JELLYSEERR_CONFIG}:/app/config
    ports:
      - "${JELLYSEERR_PORT}:5055"
    restart: unless-stopped
```

**New in 2025:** Jellyseerr supports `API_KEY` environment variable for automated setups!

## 🔑 API Key Management

### Where to Find API Keys

1. **Prowlarr**: Settings > Security > API Key
2. **Sonarr/Radarr/Lidarr**: Settings > Security > API Key  
3. **Jellyseerr**: Settings > General > API Key
4. **Overseerr**: Settings > General > API Key
5. **Tautulli**: Settings > Web Interface > API Key

### Service URLs for Integration

When configuring connections between services:
- **Prowlarr**: `http://prowlarr:9696`
- **Sonarr**: `http://sonarr:8989`
- **Radarr**: `http://radarr:7878`
- **qBittorrent**: `http://qbittorrent:8080`
- **Jellyfin**: `http://jellyfin:8096`

## 📢 Notification Configuration

### Discord Webhooks

1. Create webhook in Discord channel settings
2. Copy webhook URL: `https://discord.com/api/webhooks/ID/TOKEN`
3. Set environment variables:
   ```bash
   DISCORD_WEBHOOK_ID=your-webhook-id
   DISCORD_WEBHOOK_TOKEN=your-webhook-token
   ```

### Telegram Bot Setup

1. Message `@BotFather` on Telegram
2. Send `/newbot` and follow instructions
3. Get your bot token and chat ID
4. Set environment variables:
   ```bash
   TELEGRAM_BOT_TOKEN=your-bot-token
   TELEGRAM_CHAT_ID=your-chat-id
   ```

### Watchtower Auto-Updates

```yaml
services:
  watchtower:
    image: containrrr/watchtower:latest
    container_name: watchtower
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - WATCHTOWER_SCHEDULE=${WATCHTOWER_SCHEDULE}
      - WATCHTOWER_CLEANUP=${WATCHTOWER_CLEANUP}
      - WATCHTOWER_NOTIFICATIONS=discord
      - WATCHTOWER_NOTIFICATION_URL=discord://${DISCORD_WEBHOOK_TOKEN}@${DISCORD_WEBHOOK_ID}
    restart: unless-stopped
```

## 🛡️ Security Best Practices

### File Permissions
```bash
# Set proper ownership
sudo chown -R $USER:$USER /opt/docker
sudo chown -R $USER:$USER /mnt/media

# Set PUID/PGID to match your user
id $USER  # Get your UID and GID
```

### Network Security
- Use reverse proxy (Traefik/Nginx Proxy Manager)
- Implement SSL certificates
- Consider VPN for download clients
- Use Authelia for SSO/2FA

### Backup Strategy
```bash
# Backup configurations
tar -czf backup-$(date +%Y%m%d).tar.gz /opt/docker/config

# Schedule with cron
0 2 * * * tar -czf /mnt/backup/mediaserver-$(date +\%Y\%m\%d).tar.gz /opt/docker/config
```

## 🔧 Hardware Acceleration

### Intel Quick Sync (recommended)
```yaml
services:
  jellyfin:
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
      - /dev/dri/card0:/dev/dri/card0
```

### NVIDIA GPU
```yaml
services:
  jellyfin:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
```

## 📊 Monitoring Setup

### Prometheus + Grafana
```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "${PROMETHEUS_PORT}:9090"
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "${GRAFANA_PORT}:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
```

### Exportarr (Metrics for *arr apps)
```yaml
services:
  exportarr:
    image: ghcr.io/onedr0p/exportarr:latest
    command: ["exportarr", "sonarr"]
    environment:
      - URL=http://sonarr:8989
      - APIKEY=${SONARR_API_KEY}
```

## 🚀 Quick Start Commands

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit configuration (update PUID, PGID, paths, etc.)
nano .env

# 3. Create directories
mkdir -p /opt/docker/config/{jellyfin,sonarr,radarr,prowlarr,qbittorrent}
mkdir -p /mnt/media/{movies,tv,music,downloads/{complete,incomplete,watch}}

# 4. Start core services
docker-compose up -d jellyfin prowlarr

# 5. Start *arr stack
docker-compose up -d sonarr radarr lidarr

# 6. Start download client
docker-compose up -d qbittorrent

# 7. Start request management
docker-compose up -d jellyseerr

# 8. Start monitoring
docker-compose up -d tautulli homepage
```

## 🔄 Migration from Plex

If migrating from Plex to Jellyfin:

1. Export Plex libraries and metadata
2. Copy media files to new structure
3. Import libraries in Jellyfin
4. Reconfigure *arr apps to point to Jellyfin
5. Switch Overseerr to Jellyseerr

## ⚠️ Common Issues & Solutions

### Permission Errors
```bash
# Fix ownership
sudo chown -R $PUID:$PGID /opt/docker /mnt/media

# Check docker user mapping
docker exec -it jellyfin id
```

### Network Connectivity
```bash
# Test container communication
docker exec -it sonarr ping prowlarr
docker exec -it radarr curl http://qbittorrent:8080
```

### Path Mapping Issues
- Ensure download and media paths are consistent across all containers
- Use `/data` structure: `/data/media/movies`, `/data/downloads/complete`
- Avoid nested Docker volumes

### qBittorrent Login Issues
```bash
# Check logs for temporary password
docker logs qbittorrent

# Reset qBittorrent config if needed
docker exec -it qbittorrent rm /config/qBittorrent/qBittorrent.conf
docker restart qbittorrent
```

## 📚 Additional Resources

- [Trash Guides](https://trash-guides.info/) - Quality profiles and naming
- [TRaSH-Guides Docker Guide](https://trash-guides.info/Hardlinks/How-to-setup-for/Docker/) - Path setup
- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [ServarrWiki](https://wiki.servarr.com/) - *arr applications
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🆕 What's New in 2025

1. **Jellyfin** continues to lead as the best open-source option
2. **Prowlarr** has fully replaced Jackett for indexer management
3. **Jellyseerr** now supports API_KEY environment variable
4. **qBittorrent** generates secure temporary passwords by default
5. **Enhanced security** focus with better container isolation
6. **Improved hardware acceleration** support across all platforms
7. **Better monitoring** integration with Prometheus/Grafana
8. **Streamlined configuration** with fewer manual steps

---

**Pro Tip**: Start with Jellyfin + Prowlarr + Sonarr + Radarr + qBittorrent for a minimal but complete setup, then expand with additional services as needed.