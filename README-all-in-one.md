# Single Container Media Server 2025 🚀

The ultimate all-in-one media server solution combining Jellyfin, *arr stack, download clients, and Caddy reverse proxy in a single Docker container using s6-overlay for process supervision.

## 🎯 Key Features

- **Single Container Architecture**: All services in one container for simplified deployment
- **S6-Overlay Process Supervision**: Industry-standard init system for reliable multi-process management
- **Caddy Reverse Proxy**: Automatic HTTPS, HTTP/3 support, zero-config SSL
- **Complete Media Stack**:
  - Jellyfin for media streaming
  - Radarr for movie management
  - Sonarr for TV show management
  - Prowlarr for indexer management
  - qBittorrent for downloads
- **Hardware Acceleration**: Intel QuickSync and NVIDIA GPU support
- **Performance Optimized**: Static binaries, efficient IPC, optimized for streaming

## 📊 Research Highlights

Based on August 2025 trends:
- S6-overlay is the preferred init system for multi-process containers
- Caddy outperforms traditional reverse proxies for media streaming
- Single container solutions reduce complexity while maintaining flexibility
- Hardware acceleration is essential for 4K/HDR content

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+ and Docker Compose 1.29+
- 4GB+ RAM (8GB recommended)
- 50GB+ storage for applications
- Linux, macOS, or Windows with WSL2

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mediaserver-2025
cd mediaserver-2025
```

2. **Run the setup script**
```bash
chmod +x setup-all-in-one.sh
./setup-all-in-one.sh
```

3. **Access your media server**
- Main URL: `https://your-domain.com`
- Services append their paths: `/radarr/`, `/sonarr/`, etc.

### Manual Installation

1. **Create directory structure**
```bash
mkdir -p config/{caddy,jellyfin,radarr,sonarr,prowlarr,qbittorrent}
mkdir -p media/{movies,tv,music,photos}
mkdir -p downloads/{complete,incomplete}
mkdir -p transcodes
```

2. **Create `.env` file**
```bash
cp .env.example .env
nano .env
```

3. **Build and start**
```bash
docker-compose -f docker-compose.all-in-one.yml up -d
```

## 🏗️ Architecture

### Process Hierarchy
```
s6-overlay (PID 1)
├── caddy (reverse proxy)
├── jellyfin (media server)
├── radarr (movies)
├── sonarr (tv shows)
├── prowlarr (indexers)
└── qbittorrent (downloads)
```

### Service Dependencies
- Base services: Caddy, Jellyfin, Prowlarr, qBittorrent
- Dependent services: Radarr/Sonarr (depend on Prowlarr)

### Port Mappings
- 80: HTTP (redirects to HTTPS)
- 443: HTTPS
- 443/udp: HTTP/3 (QUIC)

## 🔧 Configuration

### Environment Variables
```bash
# User configuration
PUID=1000
PGID=1000
TZ=America/New_York

# Paths
MEDIA_PATH=/path/to/media
DOWNLOADS_PATH=/path/to/downloads
TRANSCODES_PATH=/path/to/fast/storage

# Domain
DOMAIN=https://media.example.com

# Resources
MEMORY_LIMIT=8G
CPU_LIMIT=4.0
```

### Caddy Configuration
The included Caddyfile provides:
- Automatic HTTPS with Let's Encrypt
- HTTP/3 support
- Security headers
- Gzip compression
- Reverse proxy for all services

### Hardware Acceleration

**Intel QuickSync:**
```yaml
devices:
  - /dev/dri:/dev/dri
environment:
  - LIBVA_DRIVER_NAME=iHD
```

**NVIDIA GPU:**
```yaml
devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
runtime: nvidia
```

## 📈 Performance Optimization

### 1. **Storage**
- Use SSD for config and transcoding
- Mount options: `noatime,nodiratime`
- Separate volumes for different content types

### 2. **Network**
- Enable HTTP/3 in Caddy
- Use Unix sockets for internal communication
- Optimize kernel parameters

### 3. **Transcoding**
- Hardware acceleration is crucial
- Pre-transcode popular content
- Use optimized encoding settings

## 🔒 Security

### Built-in Security Features
- All services behind reverse proxy
- Automatic HTTPS encryption
- Internal services on localhost only
- Non-root user execution
- Security headers enabled

### Additional Hardening
1. Enable Caddy authentication
2. Use VPN for download clients
3. Regular backups
4. Monitor logs
5. Keep services updated

## 🛠️ Maintenance

### Viewing Logs
```bash
# All services
docker logs -f mediaserver-aio

# Specific service logs
docker exec mediaserver-aio tail -f /config/jellyfin/log/log*.log
```

### Backup
```bash
# Stop container
docker-compose -f docker-compose.all-in-one.yml down

# Backup config
tar -czf backup-$(date +%Y%m%d).tar.gz config/

# Start container
docker-compose -f docker-compose.all-in-one.yml up -d
```

### Updates
```bash
# Pull latest changes
git pull

# Rebuild container
docker-compose -f docker-compose.all-in-one.yml build --no-cache

# Restart services
docker-compose -f docker-compose.all-in-one.yml up -d
```

## 📊 Monitoring

### Service Status
```bash
docker exec mediaserver-aio s6-svstat /var/run/s6/services/*
```

### Resource Usage
```bash
docker stats mediaserver-aio
```

### Health Check
```bash
curl -f https://localhost/health || echo "Unhealthy"
```

## 🚨 Troubleshooting

### Service Won't Start
1. Check logs: `docker logs mediaserver-aio`
2. Verify permissions: `ls -la config/`
3. Check port conflicts: `netstat -tulpn | grep -E '80|443'`

### Performance Issues
1. Enable hardware acceleration
2. Check disk I/O: `iotop`
3. Monitor CPU/RAM: `docker stats`
4. Review transcoding settings

### SSL Certificate Issues
1. Ensure ports 80/443 are accessible
2. Check domain DNS
3. Review Caddy logs

## 🔄 Migration Guide

### From Multi-Container Setup
1. Export databases from existing containers
2. Copy configuration files
3. Update connection strings to localhost
4. Import data into single container

### From Other Media Servers
1. Use Jellyfin's import tools
2. Migrate *arr databases
3. Re-download metadata
4. Verify file paths

## 🌟 Advanced Features

### Custom Integrations
- Webhook support for automation
- API access for all services
- Custom scripts via s6-overlay
- Plugin support in Jellyfin

### Scaling Options
- Horizontal scaling with multiple instances
- CDN integration for streaming
- Remote storage mounting
- Distributed transcoding

## 📚 Resources

- [S6-Overlay Documentation](https://github.com/just-containers/s6-overlay)
- [Caddy Documentation](https://caddyserver.com/docs/)
- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [TRaSH Guides](https://trash-guides.info/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🎉 Acknowledgments

- LinuxServer.io for container best practices
- The r/selfhosted community
- S6-overlay maintainers
- All the individual project maintainers

---

**Built with ❤️ for the self-hosting community**

*Last Updated: August 2025*