# Ultimate Media Server - Single Container Solution

## 🚀 DevOps-Automated All-in-One Media Server

A comprehensive, production-ready media server that runs everything in a single container with full DevOps automation including monitoring, backup, security, and AI integration.

### ✨ Features

#### 🎬 Media Services
- **Jellyfin** - Open-source media server with hardware acceleration
- **Sonarr** - TV show management and automation
- **Radarr** - Movie management and automation  
- **Lidarr** - Music management and automation
- **Prowlarr** - Indexer management for all *arr services
- **qBittorrent** - Download client with web interface

#### 🔧 DevOps Automation
- **Caddy** - Automatic HTTPS reverse proxy
- **Prometheus** - Metrics collection and monitoring
- **Grafana** - Visualization and alerting dashboards
- **Redis** - Caching and session storage
- **Ultimate Dashboard** - Real-time service monitoring with WebSocket updates

#### 🛡️ Security & Operations
- **Automated Backups** - Scheduled configuration backups with retention
- **Health Monitoring** - Comprehensive health checks and alerting
- **Log Management** - Centralized logging with rotation
- **Security Scanning** - Built-in security tools (fail2ban, rkhunter)
- **Resource Monitoring** - CPU, memory, and disk usage tracking
- **Auto-Updates** - Automatic container and service updates

#### 🤖 AI Integration
- **MCP Suite** - AI-powered media management assistant
- **Smart Monitoring** - Intelligent anomaly detection
- **Automated Troubleshooting** - AI-driven issue resolution

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ultimate Media Server                    │
├─────────────────────────────────────────────────────────────┤
│  Caddy Reverse Proxy (Port 80)                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Jellyfin   │ │   Sonarr    │ │   Radarr    │          │
│  │   :8096     │ │   :8989     │ │   :7878     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Lidarr    │ │  Prowlarr   │ │qBittorrent  │          │
│  │   :8686     │ │   :9696     │ │   :8080     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Dashboard   │ │  Grafana    │ │ Prometheus  │          │
│  │   :3000     │ │   :3001     │ │   :9090     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Redis     │ │ MCP Suite   │ │ Automation  │          │
│  │   :6379     │ │   :8090     │ │  Services   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 Quick Start

#### Prerequisites
- Docker 20.10+ and Docker Compose v2+
- 4GB+ RAM recommended
- 10GB+ free disk space
- Linux/macOS/Windows with WSL2

#### 1. One-Command Installation
```bash
# Download and run the deployment script
curl -fsSL https://raw.githubusercontent.com/your-repo/ultimate-media-server/main/deploy-ultimate-single.sh -o deploy-ultimate-single.sh
chmod +x deploy-ultimate-single.sh
./deploy-ultimate-single.sh install
```

#### 2. Manual Installation
```bash
# Clone repository
git clone https://github.com/your-repo/ultimate-media-server.git
cd ultimate-media-server

# Run installation
./deploy-ultimate-single.sh install
```

#### 3. Custom Configuration
```bash
# Generate configuration interactively
./deploy-ultimate-single.sh install

# Or copy and edit environment file
cp .env.ultimate.example .env.ultimate
# Edit .env.ultimate with your settings
./deploy-ultimate-single.sh deploy
```

### 🎯 Usage

#### Service Access URLs
After deployment, access services at:

- **🏠 Main Dashboard**: http://localhost/
- **🎬 Jellyfin**: http://localhost/jellyfin
- **📺 Sonarr**: http://localhost/sonarr  
- **🎥 Radarr**: http://localhost/radarr
- **🎵 Lidarr**: http://localhost/lidarr
- **🔍 Prowlarr**: http://localhost/prowlarr
- **⬇️ qBittorrent**: http://localhost/qbittorrent
- **📊 Grafana**: http://localhost/grafana
- **🤖 AI Assistant**: http://localhost/mcp

#### Management Commands
```bash
# Check service status
./deploy-ultimate-single.sh status

# View logs
./deploy-ultimate-single.sh logs
./deploy-ultimate-single.sh logs jellyfin

# Create backup
./deploy-ultimate-single.sh backup

# Update services
./deploy-ultimate-single.sh update

# Restart services
./deploy-ultimate-single.sh restart

# Stop all services
./deploy-ultimate-single.sh stop
```

### ⚙️ Configuration

#### Environment Variables
Key configuration options in `.env.ultimate`:

```bash
# System
TZ=America/New_York
DOMAIN=localhost
WEB_PORT=80

# Resources
MEMORY_LIMIT=4G
CPU_LIMIT=2.0

# Features
HARDWARE_ACCELERATION=true
VPN_ENABLED=false
AUTO_UPDATE=true

# Backup
BACKUP_RETENTION_DAYS=7
BACKUP_SCHEDULE="0 */6 * * *"

# Security
SECURITY_SCAN_ENABLED=true
FAIL2BAN_ENABLED=true
```

#### Directory Structure
```
ultimate-media-server/
├── ultimate-config/          # Service configurations
│   ├── caddy/                # Reverse proxy config
│   ├── jellyfin/             # Media server config
│   ├── sonarr/               # TV management config
│   ├── radarr/               # Movie management config
│   ├── lidarr/               # Music management config
│   ├── prowlarr/             # Indexer config
│   ├── qbittorrent/          # Download client config
│   ├── prometheus/           # Monitoring config
│   ├── grafana/              # Dashboard config
│   └── backup/               # Backup storage
├── media/                    # Media files
│   ├── movies/               # Movie library
│   ├── tv/                   # TV show library
│   ├── music/                # Music library
│   └── books/                # Book library
├── downloads/                # Download staging
│   ├── complete/             # Completed downloads
│   └── incomplete/           # In-progress downloads
└── backups/                  # Configuration backups
```

### 🔧 Advanced Features

#### Hardware Acceleration
Enable GPU acceleration for transcoding:

```bash
# Intel Quick Sync (default enabled)
docker run --device=/dev/dri:/dev/dri ...

# NVIDIA GPU support
# Uncomment GPU sections in docker-compose.ultimate-single.yml
```

#### VPN Integration
For secure downloading:

```bash
# Edit .env.ultimate
VPN_ENABLED=true
VPN_PROVIDER=nordvpn
OPENVPN_USER=your_username
OPENVPN_PASSWORD=your_password

# Deploy with VPN profile
docker-compose --profile vpn up -d
```

#### External Database
For improved performance with large libraries:

```bash
# Deploy with external PostgreSQL
docker-compose --profile external-db up -d
```

#### SSL/HTTPS Support
Automatic HTTPS with Let's Encrypt:

```bash
# Deploy with SSL profile
docker-compose --profile ssl up -d

# Configure domain in .env.ultimate
DOMAIN=yourdomain.com
TRAEFIK_ENABLED=true
TRAEFIK_TLS=true
```

### 📊 Monitoring & Alerting

#### Built-in Dashboards
- **System Overview**: CPU, memory, disk usage
- **Service Health**: All service status and response times  
- **Download Statistics**: Torrent/Usenet activity
- **Media Library Stats**: Movies, TV shows, music counts
- **Network Traffic**: Bandwidth usage patterns

#### Health Checks
Comprehensive health monitoring includes:
- Service availability checks
- Resource usage monitoring  
- Disk space alerts
- Performance degradation detection
- Automated issue notifications

#### Custom Alerts
Configure email/webhook notifications:

```bash
# Edit .env.ultimate
SMTP_SERVER=smtp.gmail.com
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_TO=alerts@yourdomain.com
```

### 🛡️ Security Features

#### Built-in Security
- **Fail2ban**: Automatic IP blocking for failed logins
- **Security scanning**: Regular vulnerability checks
- **Access controls**: Role-based authentication
- **Network isolation**: Containerized service isolation
- **Encrypted storage**: Configuration encryption at rest

#### Security Best Practices
- Change default passwords immediately
- Enable 2FA where available
- Regular security updates
- Monitor access logs
- Use VPN for external access

### 🔄 Backup & Recovery

#### Automated Backups
- **Schedule**: Every 6 hours by default
- **Retention**: 7 days (configurable)
- **Content**: All service configurations
- **Compression**: Gzipped for space efficiency

#### Manual Backup/Restore
```bash
# Create immediate backup
./deploy-ultimate-single.sh backup

# Restore from backup file
./deploy-ultimate-single.sh restore backup_20250803_120000.tar.gz

# List available backups
ls -la backups/
```

#### Disaster Recovery
1. Stop services: `./deploy-ultimate-single.sh stop`
2. Restore configuration: `./deploy-ultimate-single.sh restore <backup_file>`
3. Rebuild if needed: `./deploy-ultimate-single.sh build`
4. Deploy: `./deploy-ultimate-single.sh deploy`

### 🔍 Troubleshooting

#### Common Issues

**Services not starting:**
```bash
# Check container logs
./deploy-ultimate-single.sh logs

# Check system resources
docker stats ultimate-media-server

# Check health status
./deploy-ultimate-single.sh status
```

**Port conflicts:**
```bash
# Edit .env.ultimate to change ports
WEB_PORT=8080
JELLYFIN_PORT=8097
# etc.
```

**Permission issues:**
```bash
# Fix ownership
sudo chown -R $(id -u):$(id -g) ultimate-config/ media/ downloads/
```

**Out of disk space:**
```bash
# Clean Docker system
docker system prune -af

# Check directory sizes
du -sh ultimate-config/ media/ downloads/ backups/
```

#### Log Files
- **Deployment logs**: `ultimate-deploy.log`
- **Service logs**: `./deploy-ultimate-single.sh logs [service]`
- **System logs**: `/var/log/mediaserver/monitoring.log`

#### Health Check Details
```bash
# Detailed health check
docker exec ultimate-media-server /usr/local/bin/healthcheck

# Service-specific checks
curl http://localhost/health
curl http://localhost:8096/health
curl http://localhost:8989/ping
```

### 🚀 Performance Optimization

#### Resource Tuning
Adjust based on your hardware:

```bash
# High-performance setup (8GB+ RAM)
MEMORY_LIMIT=8G
CPU_LIMIT=4.0
JELLYFIN_CACHE_SIZE=1024

# Low-resource setup (2GB RAM)
MEMORY_LIMIT=2G
CPU_LIMIT=1.0
JELLYFIN_CACHE_SIZE=128
```

#### Storage Optimization
- Use SSD for configuration and cache
- Separate media storage can be HDD
- Enable compression for backups
- Regular cleanup of old downloads

#### Network Optimization
- Use wired connection for best performance
- Configure QoS for media streaming
- Monitor bandwidth usage in Grafana

### 🤖 AI Assistant Features

The integrated MCP Suite provides:
- **Smart media organization**: Automatic metadata enhancement
- **Download optimization**: Intelligent quality selection
- **Issue troubleshooting**: Automated problem detection and resolution
- **Performance insights**: AI-driven optimization recommendations
- **Predictive maintenance**: Proactive issue prevention

Access the AI Assistant at: http://localhost/mcp

### 📚 Additional Resources

#### Documentation
- [Advanced Configuration Guide](docs/ADVANCED_CONFIG.md)
- [Security Hardening Guide](docs/SECURITY.md)
- [Performance Tuning Guide](docs/PERFORMANCE.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

#### Community
- [GitHub Issues](https://github.com/your-repo/ultimate-media-server/issues)
- [Discord Server](https://discord.gg/your-invite)
- [Reddit Community](https://reddit.com/r/UltimateMediaServer)

#### Development
- [Contributing Guidelines](CONTRIBUTING.md)
- [Development Setup](docs/DEVELOPMENT.md)
- [API Documentation](docs/API.md)

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

Built with and inspired by:
- [Jellyfin Media Server](https://jellyfin.org/)
- [Sonarr](https://sonarr.tv/), [Radarr](https://radarr.video/), [Lidarr](https://lidarr.audio/), [Prowlarr](https://prowlarr.com/)
- [qBittorrent](https://www.qbittorrent.org/)
- [Caddy Web Server](https://caddyserver.com/)
- [Prometheus](https://prometheus.io/) & [Grafana](https://grafana.com/)
- [Docker](https://docker.com/) & [s6-overlay](https://github.com/just-containers/s6-overlay)

---

## 🎯 Quick Commands Reference

```bash
# Complete installation
./deploy-ultimate-single.sh install

# Check status
./deploy-ultimate-single.sh status

# View all logs
./deploy-ultimate-single.sh logs

# Create backup
./deploy-ultimate-single.sh backup

# Update services
./deploy-ultimate-single.sh update

# Get help
./deploy-ultimate-single.sh help
```

**🏠 Main Dashboard**: http://localhost/  
**📊 Monitoring**: http://localhost/grafana  
**🤖 AI Assistant**: http://localhost/mcp  

Happy media serving! 🎬🍿