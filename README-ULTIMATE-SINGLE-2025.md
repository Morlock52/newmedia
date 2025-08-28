# Ultimate Media Server 2025 - Single Container Solution

🎯 **Complete media server stack in ONE container using s6-overlay**

## 🚀 What's Included

All services running in a single optimized container:

| Service | Purpose | Port | Access |
|---------|---------|------|--------|
| **Caddy** | Reverse Proxy | 80 | Main dashboard entry point |
| **Jellyfin** | Media Server | 8096 | Stream movies, TV, music |
| **Sonarr** | TV Management | 8989 | Automatic TV show downloads |
| **Radarr** | Movie Management | 7878 | Automatic movie downloads |
| **Lidarr** | Music Management | 8686 | Automatic music downloads |
| **Prowlarr** | Indexer Manager | 9696 | Manage torrent/usenet indexers |
| **Bazarr** | Subtitle Manager | 6767 | Download subtitles |
| **qBittorrent** | Torrent Client | 8080 | Download torrents |
| **SABnzbd** | Usenet Client | 8085 | Download from usenet |
| **Transmission** | Alt Torrent Client | 9091 | Alternative torrent client |
| **Dashboard** | Web Interface | 3001 | Modern React dashboard |
| **API Server** | Backend API | 3002 | Service management API |
| **Uptime Kuma** | Monitoring | 3001 | Service health monitoring |

## ⚡ Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- 10GB+ free disk space
- 4GB+ RAM recommended

### 2. Deploy
```bash
# Clone or download the files
git clone <your-repo> media-server
cd media-server

# Make deployment script executable
chmod +x deploy-ultimate-2025.sh

# Deploy everything
./deploy-ultimate-2025.sh
```

### 3. Access
Once deployed, access your services:
- 🌐 **Main Dashboard**: http://localhost
- 📺 **Jellyfin**: http://localhost:8096
- All other services are listed in the dashboard

## 📁 File Structure

```
media-server/
├── Dockerfile.ultimate-2025      # Main container build file
├── docker-compose.ultimate-2025.yml  # Compose configuration
├── Caddyfile                     # Reverse proxy config
├── deploy-ultimate-2025.sh       # Deployment script
├── health-check.sh              # Health monitoring
├── s6-services/                 # Service definitions
│   ├── caddy/
│   ├── jellyfin/
│   ├── sonarr/
│   └── ...
├── init-scripts/               # Initialization scripts
└── README-ULTIMATE-SINGLE-2025.md  # This file
```

## 🔧 Configuration

### Environment Variables (.env)
The deployment creates a `.env` file with these key settings:

```bash
# System
TZ=America/New_York
PUID=1000
PGID=1000

# Ports (customize if needed)
DASHBOARD_PORT=80
JELLYFIN_PORT=8096
SONARR_PORT=8989
# ... etc

# Paths (customize if needed)
CONFIG_PATH=./config
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads

# Security
SECURE_MODE=true
API_KEY=your-secure-api-key
```

### Directory Structure
```
./
├── config/          # Application configurations
│   ├── jellyfin/
│   ├── sonarr/
│   └── ...
├── downloads/       # Download staging area
│   ├── complete/
│   ├── incomplete/
│   └── torrents/
├── media/          # Your media library
│   ├── movies/
│   ├── tv/
│   └── music/
├── logs/           # Application logs
└── backups/        # Configuration backups
```

## 📊 Architecture

### s6-overlay Process Management
- All services managed by s6-overlay v3
- Proper process supervision and logging
- Graceful shutdown handling
- Service dependency management

### Reverse Proxy (Caddy)
- Single entry point for all services
- Automatic HTTPS (optional)
- Load balancing and health checks
- Security headers and rate limiting

### Multi-stage Docker Build
- Optimized layer caching
- Separate build stages for dashboard and API
- Minimal final image size
- Production-ready configuration

## 🛠 Management Commands

### Deployment Script Options
```bash
./deploy-ultimate-2025.sh           # Full deployment
./deploy-ultimate-2025.sh --stop    # Stop all services  
./deploy-ultimate-2025.sh --logs    # View container logs
./deploy-ultimate-2025.sh --health  # Run health check
```

### Docker Commands
```bash
# View logs
docker logs ultimate-media-server-2025

# Execute commands in container
docker exec -it ultimate-media-server-2025 bash

# Health check
docker exec ultimate-media-server-2025 /usr/local/bin/health-check

# Restart container
docker restart ultimate-media-server-2025
```

### Service Management
Inside the container, s6 manages services:
```bash
# List services
s6-rc-status

# Restart a service
s6-svc -r /run/service/jellyfin

# View service logs
s6-tail /run/service/jellyfin/log
```

## 🔒 Security Features

### Container Security
- Non-root user execution
- Security capabilities dropped
- Read-only root filesystem (where possible)
- No new privileges allowed

### Network Security
- Internal service communication
- Configurable external port exposure
- Rate limiting on public endpoints
- Security headers via Caddy

### Data Protection
- Configuration backups
- Persistent volume mounts
- Secure API key authentication
- Optional HTTPS/TLS encryption

## 📈 Monitoring & Health

### Health Checks
- Comprehensive health check script
- Process and port monitoring
- Resource usage alerts
- Service dependency verification

### Uptime Kuma Integration
- Web-based monitoring dashboard
- Service status tracking
- Alert notifications
- Historical uptime data

### Logging
- Centralized log collection
- Structured JSON logging
- Log rotation and retention
- Debug and error tracking

## 🚀 Performance Optimizations

### Resource Management
- CPU and memory limits
- Disk I/O optimization
- Network performance tuning
- Caching strategies

### Hardware Acceleration
- GPU transcoding support (Intel/AMD/NVIDIA)
- Hardware-accelerated encoding
- Optimized media processing
- Parallel download handling

### Caching & Storage
- tmpfs for temporary files
- Optimized storage layout
- Download client optimization
- Media library organization

## 🔄 Maintenance

### Updates
```bash
# Update container
./deploy-ultimate-2025.sh --stop
docker pull ultimate-media-server:2025
./deploy-ultimate-2025.sh

# Backup configurations
docker exec ultimate-media-server-2025 tar -czf /backups/config-$(date +%Y%m%d).tar.gz /config
```

### Troubleshooting
```bash
# Check service status
docker exec ultimate-media-server-2025 /usr/local/bin/health-check

# View service logs
docker logs ultimate-media-server-2025

# Access container shell
docker exec -it ultimate-media-server-2025 bash

# Restart specific service
docker exec ultimate-media-server-2025 s6-svc -r /run/service/sonarr
```

## 🎯 Initial Setup Workflow

1. **Deploy**: Run `./deploy-ultimate-2025.sh`
2. **Access Dashboard**: Go to http://localhost
3. **Configure Jellyfin**: Add media libraries
4. **Setup Prowlarr**: Add indexers for content discovery
5. **Configure *arr Apps**: Set download clients and root folders
6. **Setup Download Clients**: Configure qBittorrent/SABnzbd
7. **Add Content**: Start adding movies/TV shows
8. **Monitor**: Check Uptime Kuma for service health

## 📝 Default Credentials

| Service | Username | Password | Notes |
|---------|----------|----------|--------|
| qBittorrent | admin | adminadmin | **Change immediately** |
| Other services | N/A | Configure on first access | Follow setup wizards |

## 🔗 Service Integration

The services are pre-configured to work together:

- **Prowlarr** → **Sonarr/Radarr/Lidarr** (indexer sync)
- **Sonarr/Radarr/Lidarr** → **qBittorrent/SABnzbd** (downloads)
- **Download Clients** → **Media Library** (completed downloads)
- **Jellyfin** → **Media Library** (streaming)
- **Bazarr** → **Sonarr/Radarr** (subtitle management)

## 🆘 Support

### Common Issues
- **Port conflicts**: Modify ports in `.env` file
- **Permission errors**: Check PUID/PGID in `.env`
- **Storage issues**: Ensure sufficient disk space
- **Network problems**: Verify Docker networking

### Getting Help
1. Check the logs: `docker logs ultimate-media-server-2025`
2. Run health check: `./deploy-ultimate-2025.sh --health`
3. Review service-specific logs in the dashboard
4. Check GitHub issues for similar problems

## 📈 Scaling

For larger deployments:
- Increase resource limits in compose file
- Add external databases (PostgreSQL)
- Implement external storage (NFS/SMB)
- Use external reverse proxy (Traefik/NGINX)
- Add monitoring stack (Prometheus/Grafana)

---

**Ultimate Media Server 2025** - Everything you need for media management in one container! 🎬🎵📺