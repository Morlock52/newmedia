# Ultimate Media Server Stack - Complete Guide

## 🚀 Overview

This is the **Ultimate Media Server Stack** that supports **ALL media types** with production-ready configuration. This setup includes everything you need for a complete, self-hosted media ecosystem.

## 📦 What's Included

### Media Servers
- **Jellyfin** - Primary media server for movies, TV shows, and general media
- **Navidrome** - Dedicated music streaming server with Subsonic API support
- **AudioBookshelf** - Complete audiobook and podcast server
- **Calibre-Web** - E-book server and reader
- **Kavita** - Manga and comics reader
- **Immich** - Modern photo management (Google Photos alternative)

### Media Management (*arr Stack)
- **Sonarr** - TV show management
- **Radarr** - Movie management
- **Lidarr** - Music management
- **Readarr** - Book management
- **Bazarr** - Subtitle management
- **Prowlarr** - Indexer management

### Download Clients
- **qBittorrent** - Torrent client (through VPN)
- **SABnzbd** - Usenet client
- **Gluetun** - VPN container for secure downloading

### Request & Discovery
- **Overseerr** - Media request management with user-friendly interface

### Automation & Processing
- **FileFlows** - Automated media processing and conversion
- **Podgrab** - Podcast management and downloading

### Infrastructure
- **Traefik** - Reverse proxy with SSL support
- **Authelia** - Authentication and 2FA
- **Homepage** - Beautiful dashboard
- **Portainer** - Container management

### Monitoring & Analytics
- **Tautulli** - Media server analytics
- **Prometheus** - Metrics collection
- **Grafana** - Beautiful dashboards

### Utilities
- **Duplicati** - Automated backups
- **FileBrowser** - Web-based file management
- **Watchtower** - Automated container updates

## 🏗️ Architecture

```
Internet
    ↓
[Traefik Reverse Proxy] ← SSL Certificates
    ↓
[Authelia] ← Authentication Layer
    ↓
[Services] ← Protected by Auth
    ↓
[Media Storage] ← Shared across services
```

## 📋 Prerequisites

1. **Operating System**: Linux (Ubuntu/Debian recommended) or macOS
2. **Docker**: Version 20.10 or higher
3. **Docker Compose**: Version 2.0 or higher
4. **Storage**: Minimum 100GB free space (more for media)
5. **RAM**: Minimum 8GB (16GB recommended)
6. **VPN Account**: For secure torrenting (NordVPN, PIA, etc.)

## 🚀 Quick Start

### 1. Clone or Download Files

```bash
cd /Users/morlock/fun/newmedia
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

**Required Settings:**
- `DOMAIN` - Your domain or use 'localhost'
- `VPN_PROVIDER` - Your VPN provider
- `VPN_USERNAME` - VPN username
- `VPN_PASSWORD` - VPN password

### 3. Run Setup Script

```bash
# Make executable
chmod +x setup-ultimate.sh

# Run setup
./setup-ultimate.sh
```

### 4. Access Services

After setup completes, access your services:

| Service | URL | Purpose |
|---------|-----|---------|
| **Homepage** | http://localhost:3000 | Main dashboard |
| **Jellyfin** | http://localhost:8096 | Media streaming |
| **Navidrome** | http://localhost:4533 | Music streaming |
| **AudioBookshelf** | http://localhost:13378 | Audiobooks |
| **Immich** | http://localhost:2283 | Photos |
| **Overseerr** | http://localhost:5055 | Media requests |

## 📁 Directory Structure

```
/Users/morlock/fun/newmedia/
├── config/              # Service configurations
│   ├── jellyfin/
│   ├── sonarr/
│   ├── radarr/
│   └── ...
├── data/               # Media storage
│   ├── media/
│   │   ├── movies/
│   │   ├── tv/
│   │   ├── music/
│   │   ├── audiobooks/
│   │   ├── books/
│   │   ├── comics/
│   │   ├── photos/
│   │   └── podcasts/
│   └── downloads/
├── cache/              # Temporary files
├── transcodes/         # Jellyfin transcoding
└── backups/            # Automated backups
```

## 🔧 Initial Configuration

### 1. Jellyfin Setup
1. Access http://localhost:8096
2. Follow setup wizard
3. Add media libraries pointing to `/media/*` folders
4. Configure hardware acceleration if available

### 2. *arr Stack Setup
1. Access each service (Sonarr, Radarr, etc.)
2. Configure download client (qBittorrent)
3. Add indexers from Prowlarr
4. Set up quality profiles
5. Add media folders

### 3. Overseerr Setup
1. Access http://localhost:5055
2. Sign in with Plex/Jellyfin account
3. Configure Sonarr/Radarr integration
4. Set up user permissions

### 4. AudioBookshelf Setup
1. Access http://localhost:13378
2. Create admin account
3. Add library pointing to `/audiobooks`
4. Install mobile apps

### 5. Immich Setup
1. Access http://localhost:2283
2. Create admin account
3. Install mobile app
4. Configure auto-backup

## 🔒 Security Configuration

### Enable HTTPS with Traefik

1. **Set up domain**:
   ```bash
   # Edit .env file
   DOMAIN=yourdomain.com
   CLOUDFLARE_EMAIL=your@email.com
   CLOUDFLARE_API_TOKEN=your_api_token
   ```

2. **Configure DNS**:
   - Point *.yourdomain.com to your server IP
   - Enable Cloudflare proxy (orange cloud)

3. **Restart Traefik**:
   ```bash
   docker compose -f docker-compose-ultimate.yml restart traefik
   ```

### Configure Authelia

1. **Change default password**:
   ```bash
   docker exec -it authelia authelia hash-password
   # Copy the hash and update users_database.yml
   ```

2. **Enable 2FA**:
   - Access https://auth.yourdomain.com
   - Configure TOTP in user settings

## 📱 Mobile Apps

### iOS Apps
- **Jellyfin**: Official Jellyfin app
- **Finamp**: Music streaming (Jellyfin)
- **play:Sub**: Music streaming (Navidrome)
- **AudioBookshelf**: Official app
- **Immich**: Official app
- **Paperback**: Manga reader (Kavita)

### Android Apps
- **Jellyfin**: Official app
- **Findroid**: Alternative Jellyfin client
- **DSub**: Music streaming (Navidrome)
- **AudioBookshelf**: Official app
- **Immich**: Official app
- **Tachiyomi**: Manga reader (Kavita)

## 🎯 Best Practices

### 1. Storage Organization
- Keep downloads separate from media library
- Use atomic moves (same filesystem)
- Enable hardlinks in *arr settings

### 2. Backup Strategy
- Configure Duplicati for automated backups
- Backup config folders regularly
- Store backups offsite (cloud/NAS)

### 3. Resource Management
- Monitor with Grafana dashboards
- Set container resource limits
- Use FileFlows for media optimization

### 4. Security
- Always use VPN for torrenting
- Enable 2FA on all services
- Regular security updates
- Use strong passwords

## 🔧 Advanced Configuration

### Hardware Acceleration (Jellyfin)

**Intel GPU (QuickSync)**:
```yaml
devices:
  - /dev/dri:/dev/dri
group_add:
  - "989"  # render group
```

**NVIDIA GPU**:
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

### Custom Domain Setup

1. **Update .env**:
   ```
   DOMAIN=media.yourdomain.com
   ```

2. **Configure services**:
   - jellyfin.media.yourdomain.com
   - music.media.yourdomain.com
   - books.media.yourdomain.com
   - etc.

### Performance Tuning

1. **Database optimization**:
   ```bash
   # Optimize Jellyfin database
   docker exec jellyfin sqlite3 /config/data/jellyfin.db "VACUUM;"
   ```

2. **Transcoding optimization**:
   - Use RAM disk for transcoding
   - Enable hardware acceleration
   - Optimize encoding settings

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**:
   ```bash
   # Check logs
   docker compose -f docker-compose-ultimate.yml logs [service-name]
   
   # Restart service
   docker compose -f docker-compose-ultimate.yml restart [service-name]
   ```

2. **Permission issues**:
   ```bash
   # Fix permissions
   sudo chown -R 1000:1000 config data
   ```

3. **VPN not connecting**:
   - Verify credentials in .env
   - Check VPN provider status
   - Try different server location

4. **Can't access services**:
   - Check firewall rules
   - Verify port forwarding
   - Check service health

### Health Checks

```bash
# Check all services
docker compose -f docker-compose-ultimate.yml ps

# Check service health
docker compose -f docker-compose-ultimate.yml exec [service] healthcheck

# View real-time logs
docker compose -f docker-compose-ultimate.yml logs -f
```

## 📊 Monitoring

### Grafana Dashboards

1. Access http://localhost:3001
2. Login with admin/admin
3. Import dashboards:
   - Docker monitoring
   - Node exporter
   - Media server stats

### Prometheus Metrics

- Service uptime
- Resource usage
- Request rates
- Error tracking

## 🔄 Maintenance

### Regular Updates

```bash
# Update all containers
docker compose -f docker-compose-ultimate.yml pull
docker compose -f docker-compose-ultimate.yml up -d

# Or use Watchtower for auto-updates
```

### Backup Schedule

1. **Daily**: Configuration files
2. **Weekly**: Database backups
3. **Monthly**: Full system backup

### Cleanup

```bash
# Remove unused images
docker image prune -a

# Clean up volumes
docker volume prune

# Clear cache
rm -rf cache/* transcodes/*
```

## 🚀 Scaling

### Adding More Storage

1. Mount additional drives
2. Update docker-compose volumes
3. Configure mergerfs for pooling

### High Availability

1. Use external database (PostgreSQL)
2. Configure Redis clustering
3. Load balance with multiple instances

## 📚 Additional Resources

- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [TRaSH Guides](https://trash-guides.info/)
- [Servarr Wiki](https://wiki.servarr.com/)
- [Awesome Selfhosted](https://github.com/awesome-selfhosted/awesome-selfhosted)

## 🤝 Support

- Check logs: `docker compose logs -f [service]`
- Service documentation (linked above)
- Community forums and Discord servers

## 🎉 Conclusion

You now have a complete media server supporting:
- Movies & TV Shows
- Music & Audiobooks  
- Books & Comics
- Photos & Videos
- Podcasts
- And more!

All with automated downloading, processing, and a beautiful interface. Enjoy your ultimate media server!