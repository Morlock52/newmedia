# Media Server Stack 2025 - Production Optimized

A complete, production-ready media server stack with automated media management, secure access, and comprehensive monitoring.

## Features

### Core Services
- **Jellyfin** - Primary media server with hardware transcoding support
- **Navidrome** - Music streaming server
- **AudioBookshelf** - Audiobook and podcast server
- **Calibre-Web** - E-book server and reader
- **Kavita** - Manga and comics server
- **Immich** - Self-hosted photo management with AI features

### Media Management (*arr Stack)
- **Prowlarr** - Indexer manager for all *arr apps
- **Sonarr** - TV show management and automation
- **Radarr** - Movie management and automation
- **Lidarr** - Music management and automation
- **Readarr** - Book management and automation
- **Bazarr** - Subtitle management for movies and TV shows

### Download Clients
- **qBittorrent** - Torrent client (VPN-protected)
- **SABnzbd** - Usenet downloader
- **Gluetun** - VPN container for secure downloading

### Infrastructure
- **Traefik** - Reverse proxy with automatic SSL via Cloudflare
- **Authelia** - Single sign-on and 2FA authentication
- **PostgreSQL** - Database backend
- **Redis** - Cache and session storage

### Management & Monitoring
- **Homepage** - Beautiful dashboard for all services
- **Portainer** - Docker container management
- **Overseerr** - Media request management
- **Tautulli** - Media server analytics
- **Prometheus** - Metrics collection
- **Grafana** - Metrics visualization
- **Loki** - Log aggregation
- **Duplicati** - Automated backups

## Architecture

### Network Segmentation
The stack uses multiple isolated networks for security:
- `public` - Internet-facing services (Traefik only)
- `frontend` - Web UI services
- `backend` - Internal services and APIs
- `downloads` - VPN-isolated download network
- `monitoring` - Monitoring services

### Security Features
- SSL/TLS encryption for all services
- Single sign-on with Authelia
- 2FA support
- VPN protection for downloads
- Network isolation
- Security headers
- Rate limiting
- No root containers

### Performance Optimizations
- Hardware transcoding support (Intel QSV, NVIDIA, AMD)
- Redis caching
- RAM-based transcoding
- Resource limits and reservations
- Health checks for all services

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- A domain name with Cloudflare DNS
- Linux server with at least:
  - 8GB RAM (16GB recommended)
  - 4 CPU cores
  - SSD for OS and configs
  - Large storage for media

### Installation

1. **Clone and setup:**
```bash
git clone <repository>
cd newmedia
chmod +x setup-media-stack.sh
./setup-media-stack.sh
```

2. **Configure environment:**
```bash
cp .env.production .env
nano .env  # Edit with your values
```

3. **Required configuration:**
- `DOMAIN` - Your domain name
- `CLOUDFLARE_EMAIL` - Your Cloudflare email
- `CLOUDFLARE_API_TOKEN` - Cloudflare API token with DNS edit permissions
- Storage paths (`MEDIA_PATH`, `DOWNLOADS_PATH`)
- VPN credentials (if using)

4. **Start the stack:**
```bash
./dc up -d
```

5. **Check logs:**
```bash
./dc logs -f
```

## Service URLs

After deployment, services will be available at:

- **Homepage Dashboard**: https://home.yourdomain.com
- **Media Servers**:
  - Jellyfin: https://jellyfin.yourdomain.com
  - Music: https://music.yourdomain.com
  - Audiobooks: https://audiobooks.yourdomain.com
  - Books: https://books.yourdomain.com
  - Comics: https://comics.yourdomain.com
  - Photos: https://photos.yourdomain.com
- **Media Management**:
  - Sonarr: https://sonarr.yourdomain.com
  - Radarr: https://radarr.yourdomain.com
  - Prowlarr: https://prowlarr.yourdomain.com
  - Overseerr: https://requests.yourdomain.com
- **Administration**:
  - Traefik: https://traefik.yourdomain.com
  - Portainer: https://portainer.yourdomain.com
  - Grafana: https://grafana.yourdomain.com

## Initial Configuration

### 1. Authelia Setup
- Access https://auth.yourdomain.com
- Register admin account
- Configure 2FA
- Edit `config/authelia/users_database.yml` for additional users

### 2. Media Library Setup
Create proper folder structure:
```bash
media/
├── movies/
├── tv/
├── music/
├── audiobooks/
├── books/
├── podcasts/
├── comics/
└── photos/
```

### 3. Prowlarr Configuration
1. Access Prowlarr
2. Add indexers
3. Configure download clients
4. Sync with other *arr apps

### 4. Download Client Setup
- **qBittorrent**: Accessible via VPN at port 8081
- **SABnzbd**: Configure news servers

### 5. Media Server Configuration
- **Jellyfin**: Run setup wizard, enable hardware transcoding
- **Overseerr**: Connect to Jellyfin and *arr apps

## Maintenance

### Backups
Automated backups run daily at 2 AM. Configure in Duplicati:
- Source: `/source/config` and `/source/data`
- Destination: Your backup location
- Encryption: Enable with strong password

### Updates
```bash
# Check for updates (monitor mode)
./dc logs watchtower

# Manual update specific service
./dc pull jellyfin
./dc up -d jellyfin

# Update all services
./dc pull
./dc up -d
```

### Monitoring
- **Grafana dashboards**: https://grafana.yourdomain.com
- **Service health**: Homepage dashboard
- **Container logs**: `./dc logs -f [service-name]`

## Troubleshooting

### Service won't start
```bash
# Check logs
./dc logs [service-name]

# Check service health
./dc ps

# Restart service
./dc restart [service-name]
```

### SSL certificate issues
```bash
# Check Traefik logs
./dc logs traefik

# Verify Cloudflare API token
# Check DNS records in Cloudflare
```

### Permission issues
```bash
# Fix ownership
sudo chown -R $USER:$USER config/ data/

# Fix permissions
find config/ -type d -exec chmod 755 {} \;
find config/ -type f -exec chmod 644 {} \;
```

### VPN connection issues
```bash
# Check Gluetun logs
./dc logs gluetun

# Test VPN connection
./dc exec gluetun sh -c "curl ifconfig.me"
```

## Advanced Configuration

### Hardware Transcoding

**Intel QuickSync:**
```yaml
devices:
  - /dev/dri:/dev/dri
```

**NVIDIA GPU:**
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

### Custom Domain Configuration
Edit `config/traefik/dynamic/custom-routes.yml` for additional domains or services.

### Resource Limits
Adjust in docker-compose file:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

## Security Considerations

1. **Use strong passwords** for all services
2. **Enable 2FA** in Authelia
3. **Regular backups** of configuration and data
4. **Monitor logs** for suspicious activity
5. **Keep services updated**
6. **Use VPN** for all download traffic
7. **Restrict network access** with firewall rules

## Support & Documentation

- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [Servarr Wiki](https://wiki.servarr.com/)
- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [Authelia Documentation](https://www.authelia.com/docs/)

## License

This configuration is provided as-is for personal use. Please respect the licenses of individual applications.