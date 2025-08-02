# 🚀 Ultimate Media Server 2025

<div align="center">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Jellyfin-00A4DC?style=for-the-badge&logo=jellyfin&logoColor=white" alt="Jellyfin">
  <img src="https://img.shields.io/badge/Caddy-22B638?style=for-the-badge&logo=caddy&logoColor=white" alt="Caddy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-blue?style=for-the-badge" alt="Platform">
</div>

<div align="center">
  <h3>🎬 The Most Advanced Self-Hosted Media Server Solution for 2025</h3>
  <p>Single-container architecture • Hardware acceleration • Auto-configuration • Beautiful UI</p>
</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🖼️ Screenshots](#️-screenshots)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [🛠️ Installation](#️-installation)
- [🎯 Architecture](#-architecture)
- [⚙️ Configuration](#️-configuration)
- [🔧 Advanced Setup](#-advanced-setup)
- [📊 Performance](#-performance)
- [🔒 Security](#-security)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Features
- **Single Container Architecture** - All services in one Docker container with s6-overlay
- **Hardware Acceleration** - Intel QuickSync, NVIDIA, AMD GPU support
- **Auto-Configuration** - Zero-config setup with intelligent defaults
- **Beautiful Dashboards** - Multiple UI options included
- **Cross-Platform** - Works on Linux, macOS, Windows (WSL2)

</td>
<td width="50%">

### 🚀 Advanced Features
- **S6-Overlay Process Manager** - Industry-standard init system
- **Caddy Reverse Proxy** - Automatic HTTPS, HTTP/3 support
- **TRaSH Guides Standards** - Optimal media organization
- **4K/HDR Support** - Hardware transcoding for smooth playback
- **Mobile Ready** - Responsive UI for all devices

</td>
</tr>
</table>

---

## 🖼️ Screenshots

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="docs/images/homepage-dashboard.png" width="400" alt="Homepage Dashboard">
        <br><b>Homepage Dashboard</b>
      </td>
      <td align="center">
        <img src="docs/images/jellyfin-library.png" width="400" alt="Jellyfin Library">
        <br><b>Jellyfin Media Library</b>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="docs/images/sonarr-calendar.png" width="400" alt="Sonarr Calendar">
        <br><b>Sonarr TV Calendar</b>
      </td>
      <td align="center">
        <img src="docs/images/env-manager.png" width="400" alt="Environment Manager">
        <br><b>Environment Manager UI</b>
      </td>
    </tr>
  </table>
</div>

---

## 🚀 Quick Start

### One-Line Installation

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/install.sh | bash
```

**Windows (PowerShell as Administrator):**
```powershell
iwr -useb https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/install.ps1 | iex
```

**Docker Run (Single Container):**
```bash
docker run -d \
  --name mediaserver \
  -p 80:80 -p 443:443 \
  -v /path/to/config:/config \
  -v /path/to/media:/data \
  -e PUID=$(id -u) -e PGID=$(id -g) \
  --restart unless-stopped \
  ghcr.io/yourusername/mediaserver-aio:latest
```

---

## 📦 What's Included

<div align="center">

| Service | Description | Port | Purpose |
|---------|-------------|------|---------|
| 🎬 **Jellyfin** | Media Streaming Server | 8096 | Your personal Netflix |
| 📺 **Sonarr** | TV Show Automation | 8989 | Automatic TV downloads |
| 🎥 **Radarr** | Movie Automation | 7878 | Automatic movie downloads |
| 🎵 **Lidarr** | Music Management | 8686 | Music library automation |
| 📚 **Readarr** | Book Management | 8787 | eBook library automation |
| 🎧 **AudioBookshelf** | Audiobook Server | 13378 | Audiobooks & podcasts |
| 🔍 **Prowlarr** | Indexer Manager | 9696 | Torrent/Usenet indexers |
| 📥 **qBittorrent** | Download Client | 8080 | Torrent downloads |
| 💬 **Bazarr** | Subtitle Manager | 6767 | Automatic subtitles |
| 🏠 **Homepage** | Dashboard | 3000 | Beautiful dashboard |
| 🛡️ **Caddy** | Reverse Proxy | 80/443 | HTTPS & routing |
| 📊 **Uptime Kuma** | Monitoring | 3011 | Service health checks |

</div>

### Additional Services (Full Installation)

<details>
<summary>Click to expand full service list (30+ services)</summary>

| Service | Port | Description |
|---------|------|-------------|
| Plex | 32400 | Alternative media server |
| Emby | 8920 | Another media server option |
| Overseerr | 5055 | Media request management |
| Tautulli | 8181 | Media analytics |
| Navidrome | 4533 | Music streaming server |
| Kavita | 5000 | Manga/comic server |
| Calibre-Web | 8083 | eBook server |
| Transmission | 9091 | Alternative torrent client |
| SABnzbd | 8081 | Usenet downloader |
| NZBGet | 6789 | Alternative Usenet client |
| Jackett | 9117 | Alternative indexer proxy |
| FlareSolverr | 8191 | Cloudflare bypass |
| Autobrr | 7474 | IRC automation |
| Cross-seed | 2468 | Cross-seeding automation |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Metrics visualization |
| Portainer | 9443 | Docker management |
| Watchtower | - | Auto-updates containers |
| Duplicati | 8200 | Backup solution |
| Nginx Proxy Manager | 81 | Alternative reverse proxy |
| Authelia | 9091 | Authentication portal |
| Redis | 6379 | Cache server |
| PostgreSQL | 5432 | Database server |

</details>

---

## 🛠️ Installation

### Prerequisites

<table>
<tr>
<td width="33%">

**🐧 Linux**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Fedora/RHEL
sudo dnf install docker docker-compose

# Arch
sudo pacman -S docker docker-compose
```

</td>
<td width="33%">

**🍎 macOS**
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from:
# https://docker.com/products/docker-desktop
```

</td>
<td width="34%">

**🪟 Windows**
```powershell
# Enable WSL2
wsl --install

# Install Docker Desktop
# Download from:
# https://docker.com/products/docker-desktop
```

</td>
</tr>
</table>

### Method 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server

# Run the installation wizard
./install.sh
```

The installation wizard will:
- ✅ Check system requirements
- ✅ Create necessary directories
- ✅ Generate secure passwords
- ✅ Configure all services
- ✅ Start the media server
- ✅ Open the dashboard

### Method 2: Single Container Deployment

<details>
<summary>Click for single container deployment</summary>

```bash
# Build the all-in-one image
docker build -t mediaserver-aio -f Dockerfile.multi-service .

# Run with full configuration
docker run -d \
  --name mediaserver \
  -p 80:80 \
  -p 443:443 \
  -p 8096:8096 \
  -p 8989:8989 \
  -p 7878:7878 \
  -p 9696:9696 \
  -p 8080:8080 \
  -v $(pwd)/config:/config \
  -v /path/to/media:/data/media \
  -v /path/to/downloads:/data/downloads \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  -e TZ=America/New_York \
  --restart unless-stopped \
  mediaserver-aio

# Or use the provided script
./deploy-single-container.sh
```

</details>

### Method 3: Docker Compose (Multi-Container)

<details>
<summary>Click for docker-compose deployment</summary>

1. **Create directory structure:**
```bash
mkdir -p ~/mediaserver/{config,media,downloads}
cd ~/mediaserver

# Create subdirectories
mkdir -p config/{caddy,jellyfin,sonarr,radarr,prowlarr,qbittorrent}
mkdir -p media/{movies,tv,music,books,photos}
mkdir -p downloads/{complete,incomplete,torrents,watch}
```

2. **Download configuration files:**
```bash
# Download docker-compose.yml
curl -O https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/docker-compose.yml

# Download .env template
curl -O https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/.env.example
mv .env.example .env
```

3. **Configure environment:**
```bash
# Edit .env file
nano .env

# Set these required variables:
PUID=1000              # Your user ID (run: id -u)
PGID=1000              # Your group ID (run: id -g)
TZ=America/New_York    # Your timezone
DOMAIN=media.local     # Your domain (or use localhost)
```

4. **Start the services:**
```bash
# Pull images and start
docker-compose up -d

# Watch logs
docker-compose logs -f
```

5. **Access the dashboard:**
Open http://localhost:3001 in your browser

</details>

---

## 🎯 Architecture

### System Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "Media Server Container"
            Caddy[Caddy Reverse Proxy<br/>:80/:443]
            
            subgraph "Media Services"
                Jellyfin[Jellyfin :8096]
                AudioBookshelf[AudioBookshelf :13378]
                Navidrome[Navidrome :4533]
            end
            
            subgraph "Download Automation"
                Sonarr[Sonarr :8989]
                Radarr[Radarr :7878]
                Lidarr[Lidarr :8686]
                Readarr[Readarr :8787]
                Prowlarr[Prowlarr :9696]
                Bazarr[Bazarr :6767]
            end
            
            subgraph "Download Clients"
                qBittorrent[qBittorrent :8080]
                SABnzbd[SABnzbd :8081]
            end
            
            subgraph "Management"
                Homepage[Homepage :3000]
                UptimeKuma[Uptime Kuma :3011]
            end
        end
        
        subgraph "Storage"
            Config[/config]
            Media[/media]
            Downloads[/downloads]
        end
    end
    
    Internet((Internet)) --> Caddy
    Caddy --> Jellyfin
    Caddy --> Sonarr
    Caddy --> Radarr
    Prowlarr --> Sonarr
    Prowlarr --> Radarr
    Sonarr --> qBittorrent
    Radarr --> qBittorrent
    qBittorrent --> Downloads
    Downloads --> Media
    Media --> Jellyfin
```

### Directory Structure

```
mediaserver/
├── config/                 # Service configurations
│   ├── caddy/             # Reverse proxy config
│   ├── jellyfin/          # Media server config
│   ├── sonarr/            # TV automation
│   ├── radarr/            # Movie automation
│   └── ...                # Other service configs
├── media/                 # Media library
│   ├── movies/            # Movie files
│   ├── tv/                # TV show files
│   ├── music/             # Music files
│   └── ...                # Other media types
├── downloads/             # Download directory
│   ├── complete/          # Completed downloads
│   ├── incomplete/        # In-progress downloads
│   └── watch/             # Watch folder
├── docker-compose.yml     # Service definitions
├── .env                   # Environment variables
└── Caddyfile             # Caddy configuration
```

---

## ⚙️ Configuration

### Essential Configuration

<details>
<summary>📝 Environment Variables (.env)</summary>

```bash
# User/Group IDs
PUID=1000
PGID=1000

# Timezone
TZ=America/New_York

# Paths
CONFIG_PATH=./config
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads

# Domain Configuration
DOMAIN=media.yourdomain.com
EMAIL=your-email@example.com

# Service Ports (change if needed)
JELLYFIN_PORT=8096
SONARR_PORT=8989
RADARR_PORT=7878
PROWLARR_PORT=9696
QBITTORRENT_PORT=8080

# Resource Limits
MEMORY_LIMIT=8G
CPU_LIMIT=4.0

# Feature Flags
ENABLE_HARDWARE_ACCELERATION=true
ENABLE_HTTPS=true
ENABLE_AUTO_UPDATES=false
```

</details>

<details>
<summary>🔐 Caddy Configuration (Caddyfile)</summary>

```caddyfile
{
    email {$EMAIL}
    # Uncomment for local HTTPS
    # local_certs
}

# Main domain
{$DOMAIN} {
    # Homepage dashboard
    handle / {
        reverse_proxy homepage:3000
    }
    
    # Jellyfin
    handle /jellyfin* {
        reverse_proxy jellyfin:8096
    }
    
    # Sonarr
    handle /sonarr* {
        reverse_proxy sonarr:8989 {
            header_up X-Real-IP {remote_host}
        }
    }
    
    # Radarr
    handle /radarr* {
        reverse_proxy radarr:7878 {
            header_up X-Real-IP {remote_host}
        }
    }
    
    # Global headers
    header {
        # Security headers
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "SAMEORIGIN"
        Referrer-Policy "strict-origin-when-cross-origin"
        X-XSS-Protection "1; mode=block"
        # Remove server header
        -Server
    }
    
    # Enable compression
    encode gzip
    
    # Logging
    log {
        output file /logs/access.log
        format json
    }
}
```

</details>

<details>
<summary>🎬 Media Organization (TRaSH Guides)</summary>

```
media/
├── movies/
│   ├── Movie Name (Year)/
│   │   ├── Movie Name (Year).mkv
│   │   ├── Movie Name (Year).srt
│   │   └── poster.jpg
│   └── ...
├── tv/
│   ├── Show Name/
│   │   ├── Season 01/
│   │   │   ├── Show Name - S01E01 - Episode Title.mkv
│   │   │   ├── Show Name - S01E02 - Episode Title.mkv
│   │   │   └── ...
│   │   └── Season 02/
│   └── ...
├── music/
│   ├── Artist Name/
│   │   ├── Album Name/
│   │   │   ├── 01 - Track Name.flac
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

</details>

---

## 🔧 Advanced Setup

### Hardware Acceleration

<details>
<summary>🎮 Intel QuickSync</summary>

```yaml
# Add to docker-compose.yml service
devices:
  - /dev/dri:/dev/dri
environment:
  - LIBVA_DRIVER_NAME=iHD
  - JELLYFIN_FFmpeg__hwaccel_args=-hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -hwaccel_output_format vaapi
```

</details>

<details>
<summary>🎮 NVIDIA GPU</summary>

```yaml
# Add to docker-compose.yml service
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
  - JELLYFIN_FFmpeg__hwaccel_args=-hwaccel cuda -hwaccel_output_format cuda
```

</details>

<details>
<summary>🎮 AMD GPU</summary>

```yaml
# Add to docker-compose.yml service
devices:
  - /dev/dri:/dev/dri
  - /dev/kfd:/dev/kfd
environment:
  - LIBVA_DRIVER_NAME=radeonsi
  - JELLYFIN_FFmpeg__hwaccel_args=-hwaccel vaapi -hwaccel_device /dev/dri/renderD128
```

</details>

### Network Configuration

<details>
<summary>🌐 VPN Integration (Gluetun)</summary>

```yaml
# Add Gluetun service for VPN
gluetun:
  image: qmcgaw/gluetun
  cap_add:
    - NET_ADMIN
  environment:
    - VPN_SERVICE_PROVIDER=mullvad
    - VPN_TYPE=wireguard
    - WIREGUARD_PRIVATE_KEY=${VPN_PRIVATE_KEY}
    - WIREGUARD_ADDRESSES=${VPN_ADDRESSES}
  ports:
    - 8080:8080  # qBittorrent
    
# Update qBittorrent to use VPN
qbittorrent:
  network_mode: service:gluetun
  depends_on:
    - gluetun
```

</details>

<details>
<summary>🔒 SSL/TLS Configuration</summary>

```caddyfile
# Force HTTPS with custom certificates
{$DOMAIN} {
    tls /certificates/cert.pem /certificates/key.pem
    
    # Or use DNS challenge for wildcard certs
    tls {
        dns cloudflare {$CLOUDFLARE_API_TOKEN}
    }
}
```

</details>

### Storage Optimization

<details>
<summary>💾 Cache Configuration</summary>

```yaml
# Add Redis for caching
redis:
  image: redis:alpine
  volumes:
    - ./config/redis:/data
  command: redis-server --save 60 1 --loglevel warning

# Configure services to use Redis
environment:
  - REDIS_HOST=redis
  - REDIS_PORT=6379
```

</details>

<details>
<summary>🚀 SSD Cache for Transcoding</summary>

```yaml
# Mount fast storage for transcoding
volumes:
  - /mnt/ssd/transcodes:/transcodes
environment:
  - JELLYFIN_CACHE_DIR=/transcodes
```

</details>

---

## 📊 Performance

### Optimization Tips

<table>
<tr>
<td width="50%">

**Hardware Requirements**
- **Minimum**: 2 CPU cores, 4GB RAM
- **Recommended**: 4+ CPU cores, 8GB+ RAM
- **4K Streaming**: GPU with hardware encoding
- **Storage**: SSD for OS/apps, HDD for media

</td>
<td width="50%">

**Performance Tuning**
- Enable hardware acceleration
- Use SSD for transcoding cache
- Optimize database queries
- Enable HTTP/3 in Caddy
- Use CDN for remote access

</td>
</tr>
</table>

### Benchmarks

| Scenario | CPU Usage | RAM Usage | Transcode Speed |
|----------|-----------|-----------|-----------------|
| Direct Play | 1-5% | 500MB | N/A |
| 1080p → 720p (Software) | 80-100% | 2GB | 1.2x |
| 1080p → 720p (Hardware) | 10-20% | 1GB | 4-8x |
| 4K HDR → 1080p SDR (Hardware) | 20-30% | 2GB | 2-4x |

---

## 🔒 Security

### Security Features

- ✅ **HTTPS by default** with automatic certificates
- ✅ **Reverse proxy** hides internal services
- ✅ **Non-root execution** for all services
- ✅ **Network isolation** between services
- ✅ **Regular security updates** via Watchtower
- ✅ **Fail2ban integration** for brute force protection

### Hardening Guide

<details>
<summary>🛡️ Security Best Practices</summary>

1. **Enable Authentication**
```caddyfile
# Add to Caddyfile
basicauth /sonarr* {
    user $2a$14$HASH_HERE
}
```

2. **Use Strong Passwords**
```bash
# Generate secure passwords
openssl rand -base64 32
```

3. **Enable 2FA** where supported:
- Jellyfin: Settings → Users → Enable 2FA
- Overseerr: Settings → Users → 2FA

4. **Regular Backups**
```bash
# Automated backup script
./scripts/backup.sh
```

5. **Monitor Logs**
```bash
# Check for suspicious activity
docker logs -f caddy | grep -E "401|403|404"
```

</details>

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary>❌ Service Won't Start</summary>

```bash
# Check logs
docker-compose logs service-name

# Common fixes:
# 1. Fix permissions
sudo chown -R $USER:$USER ./config

# 2. Check ports
sudo netstat -tulpn | grep :8096

# 3. Restart service
docker-compose restart service-name
```

</details>

<details>
<summary>❌ Can't Access Web UI</summary>

```bash
# 1. Check if service is running
docker ps | grep jellyfin

# 2. Test local access
curl http://localhost:8096

# 3. Check firewall
sudo ufw allow 8096/tcp

# 4. Verify Caddy config
docker logs caddy
```

</details>

<details>
<summary>❌ Slow Performance</summary>

```bash
# 1. Check resource usage
docker stats

# 2. Enable hardware acceleration
# See Hardware Acceleration section

# 3. Optimize database
docker exec jellyfin sqlite3 /config/data/jellyfin.db "VACUUM;"

# 4. Clear cache
rm -rf ./config/jellyfin/cache/*
```

</details>

### Health Checks

```bash
# Run comprehensive health check
./scripts/health-check.sh

# Manual checks
curl http://localhost:8096/health  # Jellyfin
curl http://localhost:8989/ping     # Sonarr
curl http://localhost:7878/ping     # Radarr
```

### Detailed Troubleshooting Guide

See our comprehensive [Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md) for:
- Service-specific issues
- Performance optimization
- Network problems
- Storage issues
- Docker troubleshooting
- Log analysis
- Recovery procedures

---

## 🤝 Contributing

We love contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### How to Contribute

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎯 Open a Pull Request

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/ultimate-media-server.git

# Install development dependencies
npm install

# Run tests
npm test

# Build documentation
npm run docs:build
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <h3>🌟 Star this repository if you find it helpful!</h3>
  <p>
    <a href="https://github.com/yourusername/ultimate-media-server/stargazers">
      <img src="https://img.shields.io/github/stars/yourusername/ultimate-media-server?style=social" alt="Stars">
    </a>
    <a href="https://github.com/yourusername/ultimate-media-server/network/members">
      <img src="https://img.shields.io/github/forks/yourusername/ultimate-media-server?style=social" alt="Forks">
    </a>
  </p>
  <p>
    <a href="https://discord.gg/mediaserver">Join our Discord</a> •
    <a href="https://reddit.com/r/selfhosted">Reddit Community</a> •
    <a href="https://docs.mediaserver.app">Documentation</a>
  </p>
  <p>Made with ❤️ by the self-hosting community</p>
</div>