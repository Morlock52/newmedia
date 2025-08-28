# Ultimate Media Server 2025 - Simplified Deployment Guide

## 🚀 Quick Start

This simplified deployment strategy focuses on **reliability** and **ease of use**. All services use well-maintained, public Docker images that are guaranteed to work.

### Prerequisites

- Docker Desktop installed and running
- At least 10GB free disk space
- Basic understanding of Docker

### 1-Minute Deployment

```bash
# Make scripts executable
chmod +x deploy-simplified-2025.sh troubleshoot-deployment.sh

# Run the deployment
./deploy-simplified-2025.sh
```

## 📋 What's Included

### Core Services (Always Deployed)
- **Jellyfin** - Your personal Netflix
- **Sonarr** - TV show automation
- **Radarr** - Movie automation
- **Prowlarr** - Indexer management
- **qBittorrent** - Torrent downloads
- **Overseerr** - Request management

### Additional Services
- **Lidarr** - Music management
- **Readarr** - Book management
- **Bazarr** - Subtitle automation
- **SABnzbd** - Usenet downloads
- **Tautulli** - Media analytics
- **Homepage** - Beautiful dashboard
- **Portainer** - Docker management
- **Uptime Kuma** - Service monitoring

## 🎯 Phased Deployment Approach

The deployment script guides you through 6 phases:

### Phase 1: Core Infrastructure
- Redis (caching)
- PostgreSQL (database)

### Phase 2: Media Servers
- Jellyfin (primary media server)

### Phase 3: Media Management
- Prowlarr (indexers)
- Sonarr (TV)
- Radarr (movies)
- Lidarr (music)
- Readarr (books)
- Bazarr (subtitles)

### Phase 4: Download Clients
- qBittorrent (torrents)
- SABnzbd (usenet)

### Phase 5: Request & Analytics
- Overseerr (requests)
- Tautulli (analytics)

### Phase 6: Management Tools
- Homepage (dashboard)
- Portainer (Docker UI)
- Uptime Kuma (monitoring)

## 🔧 Configuration

### Environment Variables

The deployment creates a `.env` file with sensible defaults:

```env
# User Configuration
PUID=1000           # Your user ID
PGID=1000           # Your group ID
TZ=America/New_York # Your timezone

# Paths
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads
CONFIG_PATH=./config

# Database
DB_PASSWORD=mediaserver2025
```

### Directory Structure

```
newmedia/
├── media/
│   ├── movies/
│   ├── tv/
│   ├── music/
│   ├── books/
│   └── audiobooks/
├── downloads/
│   ├── complete/
│   ├── incomplete/
│   └── torrents/
└── config/
    └── [service configs]
```

## 🚦 Initial Setup

### 1. Configure Prowlarr
1. Open http://localhost:9696
2. Add your indexers (torrent/usenet sites)
3. Test each indexer

### 2. Connect *arr Apps
1. In each *arr app (Sonarr, Radarr, etc.)
2. Settings → Indexers → Add → Prowlarr
3. Use the Prowlarr API key

### 3. Setup Download Clients
1. In each *arr app
2. Settings → Download Clients
3. Add qBittorrent or SABnzbd

### 4. Configure Jellyfin
1. Open http://localhost:8096
2. Run the setup wizard
3. Add media libraries pointing to `/media/*`

### 5. Setup Overseerr
1. Open http://localhost:5055
2. Connect to Jellyfin
3. Connect to Sonarr/Radarr

## 🔍 Troubleshooting

### Run the Troubleshooting Script
```bash
./troubleshoot-deployment.sh
```

This script will:
- Check Docker status
- Find failed containers
- Detect port conflicts
- Generate diagnostic report

### Common Issues

#### "Image not found" or "denied" errors
- The simplified deployment removes all proprietary/private images
- Only uses public, well-maintained images

#### Port conflicts
- Check what's using the port: `lsof -i :PORT`
- Change port in docker-compose file or stop conflicting service

#### Container won't start
- Check logs: `docker logs [container-name]`
- Ensure paths exist and have correct permissions

#### On macOS with Apple Silicon
- The deployment script automatically handles platform compatibility
- Some services may take longer to start on first run

## 📊 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Jellyfin | http://localhost:8096 | Media streaming |
| Sonarr | http://localhost:8989 | TV management |
| Radarr | http://localhost:7878 | Movie management |
| Prowlarr | http://localhost:9696 | Indexer management |
| qBittorrent | http://localhost:8080 | Torrent client |
| Overseerr | http://localhost:5055 | Request management |
| Homepage | http://localhost:3000 | Main dashboard |
| Portainer | http://localhost:9000 | Docker management |

## 🛠️ Maintenance

### Update All Services
```bash
docker compose -f docker-compose-simplified-2025.yml pull
docker compose -f docker-compose-simplified-2025.yml up -d
```

### Backup Configuration
```bash
tar -czf backup-$(date +%Y%m%d).tar.gz config/
```

### Clean Up
```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Full cleanup (careful!)
docker system prune -a
```

## 🎯 Best Practices

1. **Start Small**: Deploy core services first, add others as needed
2. **Monitor Resources**: Use `docker stats` to watch resource usage
3. **Regular Backups**: Backup your config directory weekly
4. **Update Carefully**: Test updates on one service before updating all
5. **Use VPN**: Consider adding VPN service for download clients

## 🆘 Getting Help

1. Check container logs: `docker logs [container-name]`
2. Run troubleshooting script: `./troubleshoot-deployment.sh`
3. Review diagnostic report: `deployment-diagnostic.txt`
4. Check service documentation on DockerHub

## 🚀 Advanced Options

### Enable VPN
Uncomment VPN settings in `.env`:
```env
VPN_PROVIDER=nordvpn
VPN_USER=your_username
VPN_PASS=your_password
```

Then deploy with VPN profile:
```bash
docker compose -f docker-compose-simplified-2025.yml --profile with-vpn up -d
```

### Enable Reverse Proxy
Deploy with Traefik:
```bash
docker compose -f docker-compose-simplified-2025.yml --profile with-proxy up -d
```

## 📈 Performance Tips

1. **Use SSD**: Store config and downloads on SSD for better performance
2. **Separate Drives**: Keep media on different drive than system
3. **RAM**: Allocate at least 4GB to Docker
4. **CPU**: Modern quad-core recommended for transcoding

## 🎉 Success!

Your media server should now be running! Access the dashboard to start configuring your services.

Remember: This simplified deployment prioritizes **stability** over features. Once running smoothly, you can add more advanced services as needed.