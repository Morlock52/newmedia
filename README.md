# 🎬 Ultimate Media Server Stack - 30 Services Complete Solution

A comprehensive, production-ready media server solution featuring 30 integrated services for media streaming, management, monitoring, and automation. Built with Docker Compose for easy deployment and management.

## 🚀 Project Vision

This stack provides everything needed for a complete home media server setup:
- **Media Streaming**: Multiple server options (Jellyfin, Plex, Emby)
- **Content Management**: Full *ARR suite for automated downloads
- **Monitoring & Analytics**: Comprehensive system health tracking
- **Security & VPN**: Built-in VPN protection and ad blocking
- **Content Libraries**: Books, audiobooks, music, photos, documents
- **User Management**: Request systems and dashboard interfaces
- **Self-Hosting**: Git server, cloud storage, password management

## ⚡ Quick Start

### Prerequisites
- Docker & Docker Compose installed
- 8GB+ RAM recommended
- 100GB+ storage space
- Linux/macOS/Windows with WSL2

### 1. Clone and Setup
```bash
git clone <repository-url>
cd newmedia
cp .env.example .env
# Edit .env with your configuration
```

### 2. Launch Core Services
```bash
# Start essential services first
docker-compose up -d jellyfin sonarr radarr prowlarr qbittorrent

# Verify core services are running
docker-compose ps
```

### 3. Access Services
- **Main Dashboard**: http://localhost:7575 (Homarr)
- **Jellyfin Media**: http://localhost:8096
- **Download Manager**: http://localhost:8080 (qBittorrent)

### 4. Full Stack Deployment
```bash
# Deploy all 30 services
docker-compose up -d

# Monitor deployment
docker-compose logs -f
```

## 📊 Feature Matrix

| Category | Service | Status | Port | Description |
|----------|---------|--------|------|-------------|
| **Media Servers** |
| 🎬 | Jellyfin | ✅ Core | 8096 | Free media streaming (primary) |
| 🎭 | Plex | ✅ Core | 32400 | Premium media streaming |
| 📺 | Emby | ⚙️ Optional | 8097 | Alternative media server |
| **Content Management** |
| 📺 | Sonarr | ✅ Core | 8989 | TV show management |
| 🎬 | Radarr | ✅ Core | 7878 | Movie management |
| 🎵 | Lidarr | ⚙️ Optional | 8686 | Music management |
| 📚 | Readarr | ⚙️ Optional | 8787 | Book management |
| 💬 | Bazarr | ⚙️ Optional | 6767 | Subtitle management |
| 🔍 | Prowlarr | ✅ Core | 9696 | Indexer management |
| **Download Clients** |
| 🔽 | qBittorrent | ✅ Core | 8080 | Primary torrent client |
| 📥 | Transmission | ⚙️ Alternative | 9091 | Alternative torrent client |
| 📰 | SABnzbd | ⚙️ Optional | 8081 | Usenet downloader |
| 📊 | NZBGet | ⚙️ Alternative | 6789 | Alternative usenet client |
| **Security & VPN** |
| 🛡️ | Gluetun | ✅ Core | 8888 | VPN container |
| 🚫 | Pi-hole | ⚙️ Optional | 8053 | Network ad blocker |
| 🛡️ | AdGuard Home | ⚙️ Alternative | 8054 | Alternative ad blocker |
| **Request Management** |
| 🎯 | Jellyseerr | ⚙️ Optional | 5055 | Media requests (Jellyfin) |
| 📋 | Overseerr | ⚙️ Optional | 5056 | Media requests (Plex) |
| 📝 | Ombi | ⚙️ Alternative | 3579 | Alternative request system |
| **Monitoring & Analytics** |
| 📊 | Prometheus | ✅ Core | 9090 | Metrics collection |
| 📈 | Grafana | ✅ Core | 3000 | Metrics visualization |
| 📋 | Loki | ⚙️ Optional | 3100 | Log aggregation |
| 🔄 | Promtail | ⚙️ Optional | - | Log shipping |
| ⏰ | Uptime Kuma | ✅ Core | 3001 | Service monitoring |
| 💾 | Scrutiny | ⚙️ Optional | 8082 | HDD health monitoring |
| 👁️ | Glances | ⚙️ Optional | 61208 | System monitoring |
| 📊 | Netdata | ⚙️ Optional | 19999 | Real-time performance |
| **Management Tools** |
| 🐳 | Portainer | ✅ Core | 9000 | Docker management |
| ⛵ | Yacht | ⚙️ Alternative | 8001 | Alternative Docker UI |
| 🌐 | Nginx Proxy Manager | ✅ Core | 81 | Reverse proxy management |
| 🔄 | Watchtower | ⚙️ Optional | - | Auto-update containers |
| 📧 | Diun | ⚙️ Optional | - | Update notifications |
| **Content Libraries** |
| 📚 | Calibre-Web | ⚙️ Optional | 8083 | E-book library |
| 🎧 | Audiobookshelf | ⚙️ Optional | 13378 | Audiobook & podcast server |
| 🎵 | Navidrome | ⚙️ Optional | 4533 | Music streaming |
| 🎶 | Airsonic Advanced | ⚙️ Alternative | 4040 | Alternative music server |
| 📸 | PhotoPrism | ⚙️ Optional | 2342 | Photo management |
| 🖼️ | Immich | ⚙️ Alternative | 2283 | Google Photos alternative |
| 📄 | Paperless-ngx | ⚙️ Optional | 8010 | Document management |
| 📖 | Komga | ⚙️ Optional | 8090 | Comics/manga server |
| **Utilities & Self-Hosting** |
| ☁️ | Nextcloud | ⚙️ Optional | 8084 | Personal cloud storage |
| 🔐 | Vaultwarden | ⚙️ Optional | 8085 | Password manager |
| 🔄 | Syncthing | ⚙️ Optional | 8384 | File synchronization |
| 📁 | FileBrowser | ⚙️ Optional | 8086 | Web file manager |
| 💻 | Code Server | ⚙️ Optional | 8443 | VS Code in browser |
| 🦊 | Gitea | ⚙️ Optional | 3002 | Git server |
| **Dashboards** |
| 🏠 | Homarr | ✅ Core | 7575 | Main dashboard |
| 📊 | Homepage | ⚙️ Alternative | 3003 | Alternative dashboard |
| 🎨 | Dashy | ⚙️ Optional | 4000 | Feature-rich dashboard |
| **Databases** |
| 🐘 | PostgreSQL | ✅ Core | 5432 | Primary database |
| 🛢️ | MariaDB | ✅ Core | 3306 | MySQL-compatible database |
| 🔴 | Redis | ✅ Core | 6379 | In-memory cache |
| **Error Recovery** |
| 🔧 | Error Recovery System | ✅ Core | 3010 | Automated service recovery |

**Status Legend:**
- ✅ **Core**: Essential services, always enabled
- ⚙️ **Optional**: Additional features, enable as needed
- 📧 **Alternative**: Different options for same functionality

## 🏗️ Architecture Overview

### Network Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Networks                         │
├─────────────────────────────────────────────────────────────┤
│  media-net (172.20.0.0/16)  │  vpn-net  │  monitoring-net  │
│  ┌─── Core Services ───┐    │           │                  │
│  │ Jellyfin, *ARR Apps │    │  Gluetun  │   Prometheus     │
│  │ qBittorrent, etc.   │    │    VPN    │   Grafana        │
│  └─────────────────────┘    │           │   Monitoring     │
└─────────────────────────────────────────────────────────────┘
```

### Service Dependencies
```
Internet → Gluetun VPN → Download Clients → *ARR Apps → Media Servers
           ↓
       Pi-hole/AdGuard → Network Security
           ↓
       Nginx Proxy Manager → Reverse Proxy
           ↓
       Dashboards (Homarr/Homepage) → User Interface
```

### Storage Structure
```
/config/            # Service configurations
├── jellyfin/       # Jellyfin settings
├── sonarr/         # Sonarr database & settings
├── radarr/         # Radarr database & settings
└── ...

/data/              # Media and downloads
├── media/          # Organized media files
│   ├── movies/     # Movies (Radarr managed)
│   ├── tv/         # TV Shows (Sonarr managed)
│   ├── music/      # Music (Lidarr managed)
│   └── books/      # Books (Readarr managed)
└── downloads/      # Download staging area
    ├── complete/   # Finished downloads
    └── incomplete/ # In-progress downloads
```

## 🛠️ Setup Instructions

### Environment Configuration
1. Copy `.env.example` to `.env`
2. Configure essential variables:
```bash
# Core Settings
TZ=America/New_York
PUID=1000
PGID=1000

# VPN Configuration (Required for Gluetun)
VPN_PROVIDER=nordvpn
OPENVPN_USER=your_username
OPENVPN_PASSWORD=your_password

# Database Passwords
POSTGRES_PASSWORD=secure_password
MYSQL_ROOT_PASSWORD=secure_password

# Service Passwords
GRAFANA_PASSWORD=admin_password
NEXTCLOUD_PASSWORD=admin_password
```

### Initial Service Setup
1. **Start Core Services First**:
```bash
docker-compose up -d postgres mariadb redis
docker-compose up -d jellyfin sonarr radarr prowlarr qbittorrent
```

2. **Configure Prowlarr Indexers**:
   - Access Prowlarr at http://localhost:9696
   - Add indexers for torrent/usenet sources
   - Configure API connections to Sonarr/Radarr

3. **Setup Download Client**:
   - Access qBittorrent at http://localhost:8080
   - Default: admin/adminadmin (change immediately!)
   - Configure download paths: `/downloads`

4. **Configure *ARR Apps**:
   - Link Sonarr/Radarr to Prowlarr
   - Add qBittorrent as download client
   - Set media paths: `/data/media/tv`, `/data/media/movies`

### Advanced Configuration
- **VPN Setup**: Configure Gluetun with your VPN provider
- **SSL/HTTPS**: Use Nginx Proxy Manager for SSL certificates
- **Monitoring**: Setup Grafana dashboards for system metrics
- **Backups**: Configure regular database and config backups

## 📚 Documentation Structure

### Core Documentation
- [📋 Service Catalog](docs/services/README.md) - Detailed service descriptions
- [🔧 Configuration Guide](docs/configuration/README.md) - Setup and configuration
- [🚀 Deployment Guide](docs/deployment/README.md) - Installation and deployment
- [📊 Monitoring Setup](docs/monitoring/README.md) - Observability and alerting

### Feature Documentation
- [🎬 Media Servers](docs/features/media-servers.md) - Jellyfin, Plex, Emby setup
- [⬇️ Download Management](docs/features/download-management.md) - *ARR apps and clients
- [🛡️ Security & VPN](docs/features/security-vpn.md) - VPN and ad blocking
- [📊 Monitoring & Analytics](docs/features/monitoring.md) - Metrics and logging
- [🏠 Dashboards](docs/features/dashboards.md) - Web interfaces and management

### Operations
- [🔧 Troubleshooting](docs/operations/troubleshooting.md) - Common issues and solutions
- [📈 Performance Tuning](docs/operations/performance.md) - Optimization guide
- [🔒 Security Guide](docs/operations/security.md) - Security best practices
- [💾 Backup & Recovery](docs/operations/backup-recovery.md) - Data protection

## 🎯 Use Cases

### Home Media Server
- Stream movies, TV shows, music to all devices
- Automatic content discovery and downloading
- Family-friendly request system
- Mobile apps for remote access

### Content Creator Workflow
- Automated content ingestion
- Multiple format support
- Remote editing with Code Server
- Git repository for project management

### Self-Hosted Infrastructure
- Replace Google Photos with PhotoPrism/Immich
- Personal cloud storage with Nextcloud
- Password management with Vaultwarden
- Document management with Paperless-ngx

### Developer Environment
- Full monitoring stack with Prometheus/Grafana
- Container management with Portainer
- Private Git server with Gitea
- Remote development with Code Server

## 🤝 Contributing

We welcome contributions! Please see:
- [Contributing Guidelines](CONTRIBUTING.md)
- [Development Setup](docs/development/README.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 **Documentation**: Comprehensive guides in `/docs`
- 🐛 **Issues**: Report bugs and feature requests on GitHub
- 💬 **Community**: Join our Discord server for support
- 📧 **Contact**: Email support for enterprise inquiries

## 🏆 Acknowledgments

- Built with love for the self-hosting community
- Powered by Docker and open-source software
- Inspired by r/selfhosted and r/homelab communities

---

**🚀 Ready to get started?** Follow the [Quick Start](#-quick-start) guide above!

**📚 Need more details?** Check out our comprehensive [documentation](docs/README.md).

**🎯 Looking for specific features?** See the [Feature Matrix](#-feature-matrix) above.