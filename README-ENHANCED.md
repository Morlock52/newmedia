# Enhanced Media Server Stack - Production Ready

A comprehensive, production-ready media server solution with all media types supported, advanced monitoring, security hardening, and performance optimizations.

## 🚀 Features

### Complete Media Support
- **Video**: Jellyfin, Plex, Emby
- **Music**: Navidrome, Airsonic-Advanced  
- **Photos**: Immich, PhotoPrism
- **Books**: Calibre-Web, Kavita
- **Audiobooks**: Audiobookshelf
- **Comics/Manga**: Kavita, Mylar3

### Media Management (*arr Stack)
- **Movies**: Radarr
- **TV Shows**: Sonarr
- **Music**: Lidarr
- **Books**: Readarr
- **Subtitles**: Bazarr
- **Indexers**: Prowlarr

### Infrastructure
- **Reverse Proxy**: Traefik v3 with automatic SSL
- **Authentication**: Authelia with 2FA support
- **VPN**: Gluetun (supports 30+ providers)
- **Monitoring**: Prometheus + Grafana
- **Dashboards**: Homepage, Homarr
- **Container Management**: Portainer
- **Backup**: Duplicati

### Security Features
- SSL/TLS encryption for all services
- Authentication middleware
- Fail2ban integration
- AppArmor profiles
- Network isolation
- Audit logging
- Security monitoring

### Performance Optimizations
- Hardware acceleration support
- Optimized network buffers
- SSD-aware configurations
- Memory management
- CPU governor optimization
- Database performance tuning

## 📋 Prerequisites

- Docker 20.10+ and Docker Compose v2
- 4GB+ RAM (8GB recommended)
- 50GB+ free disk space
- Linux OS (Ubuntu 20.04+ recommended)
- Domain name with Cloudflare DNS

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd newmedia

# Run the enhanced setup wizard
sudo ./setup-enhanced.sh
```

### 2. Configure Environment

The setup wizard will guide you through:
- Domain configuration
- Cloudflare API setup
- VPN credentials (optional)
- SMTP settings (optional)
- Password generation

### 3. Deploy Services

```bash
# Start all services
docker-compose -f docker-compose-production.yml up -d

# Check service health
./scripts/health-check-all.sh
```

### 4. Access Services

After deployment, access your services at:
- Dashboard: `https://home.yourdomain.com`
- Jellyfin: `https://jellyfin.yourdomain.com`
- Requests: `https://requests.yourdomain.com`
- And more...

## 🔧 Configuration

### Service Ports

| Service | Internal Port | External URL |
|---------|--------------|--------------|
| Homepage | 3000 | home.domain.com |
| Jellyfin | 8096 | jellyfin.domain.com |
| Plex | 32400 | plex.domain.com |
| Sonarr | 8989 | sonarr.domain.com |
| Radarr | 7878 | radarr.domain.com |
| Prowlarr | 9696 | prowlarr.domain.com |
| Overseerr | 5055 | requests.domain.com |
| qBittorrent | 8080 | qbittorrent.domain.com |
| And more... | | |

### Directory Structure

```
newmedia/
├── config/           # Service configurations
├── data/            # Media and downloads
│   ├── media/       # Organized media files
│   │   ├── movies/
│   │   ├── tv/
│   │   ├── music/
│   │   ├── books/
│   │   ├── audiobooks/
│   │   ├── photos/
│   │   └── comics/
│   └── downloads/   # Download directories
├── cache/           # Temporary files
├── logs/            # Application logs
├── backups/         # Backup storage
├── secrets/         # Sensitive credentials
└── scripts/         # Utility scripts
```

### Hardware Acceleration

The stack supports hardware acceleration for:
- Intel Quick Sync (QSV)
- NVIDIA GPU (with nvidia-docker)
- AMD GPU (VAAPI)

Configure in `.env`:
```bash
RENDER_GROUP_ID=989  # For Intel GPU
```

## 🛡️ Security

### Authentication Flow
1. All services protected by Authelia
2. 2FA support via TOTP
3. SSO for seamless access
4. Group-based permissions

### Network Security
- Isolated Docker networks
- VPN for download clients
- Firewall rules (UFW)
- Fail2ban for brute force protection

### SSL/TLS
- Automatic certificates via Cloudflare
- Strong cipher suites
- HSTS enabled
- TLS 1.2+ only

## 📊 Monitoring

### Prometheus Metrics
- Container resource usage
- Service availability
- Network statistics
- Custom application metrics

### Grafana Dashboards
- System overview
- Media server statistics
- Download performance
- Error tracking

### Health Checks
All services include health checks with:
- Automatic restarts on failure
- Status monitoring
- Alert notifications

## 🔄 Maintenance

### Backup Strategy

```bash
# Manual backup
./scripts/backup-configs.sh

# Automated backups via Duplicati
# Access at: https://backup.yourdomain.com
```

### Updates

```bash
# Update all containers
./scripts/update-containers.sh

# Update specific service
docker-compose -f docker-compose-production.yml pull [service]
docker-compose -f docker-compose-production.yml up -d [service]
```

### Performance Tuning

```bash
# Apply system optimizations
sudo ./scripts/performance-tuning.sh

# Monitor performance
sudo /usr/local/bin/media-server-perfmon.sh
```

### Security Hardening

```bash
# Apply security hardening
sudo ./scripts/security-hardening.sh

# Check security status
sudo /usr/local/bin/security-monitor.sh
```

## 🐛 Troubleshooting

### Check Logs

```bash
# All services
docker-compose -f docker-compose-production.yml logs

# Specific service
docker-compose -f docker-compose-production.yml logs -f [service]

# Traefik access logs
tail -f logs/traefik/access.log
```

### Common Issues

1. **SSL Certificate Issues**
   - Verify Cloudflare API token
   - Check domain DNS settings
   - Review Traefik logs

2. **Service Can't Connect**
   - Check network isolation
   - Verify internal DNS
   - Review firewall rules

3. **Performance Issues**
   - Run performance monitor
   - Check resource usage
   - Review container limits

4. **Authentication Problems**
   - Check Authelia logs
   - Verify user database
   - Test Redis connection

## 📚 Service Configuration Guides

### Initial Setup Order
1. Configure Prowlarr indexers
2. Add Prowlarr to Sonarr/Radarr
3. Configure download clients
4. Set up Jellyfin libraries
5. Connect Overseerr to Jellyfin
6. Configure Tautulli analytics

### Prowlarr Setup
1. Access `https://prowlarr.yourdomain.com`
2. Add indexers (Usenet and torrent)
3. Configure download clients
4. Set up app connections

### Jellyfin Configuration
1. Access `https://jellyfin.yourdomain.com`
2. Create admin account
3. Add media libraries
4. Configure transcoding
5. Enable hardware acceleration

### Overseerr Integration
1. Access `https://requests.yourdomain.com`
2. Connect to Jellyfin
3. Configure Sonarr/Radarr
4. Set up user permissions
5. Configure notifications

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- LinuxServer.io for maintained Docker images
- Traefik team for the excellent reverse proxy
- All open-source media server projects

---

**Need Help?** Check the [logs](#check-logs) first, then open an issue with:
- Docker and OS versions
- Error messages
- Steps to reproduce

Happy streaming! 🎬🎵📚📷