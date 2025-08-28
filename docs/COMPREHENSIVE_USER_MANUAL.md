# 📚 Ultimate Media Server 2025 - Complete User Manual

<div align="center">
  <img src="https://img.shields.io/badge/Version-2025.1-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-green?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker" alt="Docker">
</div>

---

## 🎯 Table of Contents

1. [Getting Started](#-getting-started)
2. [Installation Guide](#-installation-guide)
3. [Service Configuration](#-service-configuration)
4. [Dashboard Guide](#-dashboard-guide)
5. [Media Management](#-media-management)
6. [Download Automation](#-download-automation)
7. [Monitoring & Analytics](#-monitoring--analytics)
8. [Advanced Features](#-advanced-features)
9. [Troubleshooting](#-troubleshooting)
10. [API Reference](#-api-reference)

---

## 🚀 Getting Started

### What is Ultimate Media Server 2025?

The Ultimate Media Server 2025 is a comprehensive, self-hosted media solution that combines 50+ services into a unified ecosystem. It provides everything you need for:

- **Media Streaming** (Jellyfin, Plex, Emby)
- **Content Automation** (*ARR stack: Sonarr, Radarr, Lidarr, etc.)
- **Download Management** (qBittorrent, Transmission, SABnzbd)
- **System Monitoring** (Grafana, Prometheus, Uptime Kuma)
- **User Management** (Request systems, dashboards)

### Key Features

✨ **One-Click Deployment** - Complete setup in minutes
🎬 **Multiple Media Servers** - Choose from Jellyfin, Plex, or Emby
📱 **Beautiful Dashboards** - Responsive, real-time monitoring
🤖 **AI Integration** - Built-in AI assistant for media management
🔒 **Enterprise Security** - SSL/TLS, authentication, monitoring
📊 **Advanced Analytics** - Performance metrics and insights

---

## 🛠️ Installation Guide

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores (Intel/AMD x64)
- RAM: 4GB
- Storage: 50GB free space
- OS: Linux, macOS, Windows 10/11

**Recommended Requirements:**
- CPU: 4+ cores with hardware transcoding support
- RAM: 8GB+ 
- Storage: 1TB+ (SSD recommended for transcoding)
- GPU: Intel QuickSync, NVIDIA, or AMD for hardware acceleration

### Pre-Installation Setup

#### 1. Install Docker & Docker Compose

**Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from: https://docs.docker.com/desktop/mac/install/
```

**Windows:**
```powershell
# Enable WSL2
wsl --install

# Install Docker Desktop
# Download from: https://docs.docker.com/desktop/windows/install/
```

#### 2. Prepare Directory Structure

```bash
# Create main directory
mkdir -p ~/ultimate-media-server
cd ~/ultimate-media-server

# Create subdirectories
mkdir -p {config,media,downloads,backups}
mkdir -p config/{jellyfin,sonarr,radarr,prowlarr,qbittorrent}
mkdir -p media/{movies,tv,music,books,photos}
mkdir -p downloads/{complete,incomplete,torrents,watch}

# Set permissions
sudo chown -R $USER:$USER ~/ultimate-media-server
chmod -R 755 ~/ultimate-media-server
```

### Installation Methods

#### Method 1: Automated Installation (Recommended)

```bash
# Download and run installer
curl -fsSL https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/install.sh | bash

# Or step-by-step:
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server
chmod +x install.sh
./install.sh
```

**What the installer does:**
1. ✅ Checks system requirements
2. ✅ Downloads configuration files
3. ✅ Creates directory structure
4. ✅ Generates secure passwords
5. ✅ Configures environment variables
6. ✅ Starts all services
7. ✅ Opens dashboard

#### Method 2: Manual Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Configure `.env` file:**
```bash
# User/Group IDs (run: id -u && id -g)
PUID=1000
PGID=1000

# Timezone (run: timedatectl list-timezones)
TZ=America/New_York

# Domain (use localhost for local access)
DOMAIN=localhost
EMAIL=your-email@example.com

# Paths (absolute paths recommended)
CONFIG_PATH=./config
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads

# Database passwords (generate with: openssl rand -base64 32)
POSTGRES_PASSWORD=your-secure-password
MYSQL_ROOT_PASSWORD=your-mysql-password

# Service-specific settings
JELLYFIN_ENABLE_HARDWARE_ACCEL=true
SONARR_BRANCH=main
RADARR_BRANCH=master
```

#### Method 3: Single Container Deployment

```bash
# Build all-in-one container
docker build -t mediaserver-aio -f Dockerfile.multi-service .

# Run single container
docker run -d \
  --name ultimate-media-server \
  -p 80:80 -p 443:443 \
  -p 8096:8096 -p 8989:8989 -p 7878:7878 \
  -v $(pwd)/config:/config \
  -v $(pwd)/media:/media \
  -v $(pwd)/downloads:/downloads \
  -e PUID=$(id -u) -e PGID=$(id -g) \
  -e TZ=America/New_York \
  --restart unless-stopped \
  mediaserver-aio
```

### First-Time Setup

#### 1. Start Services

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

#### 2. Access Dashboard

Open your browser and navigate to:
- **Main Dashboard**: http://localhost:3001
- **Alternative Dashboard**: http://localhost:7575
- **Monitoring**: http://localhost:3000

#### 3. Initial Configuration Wizard

The dashboard will guide you through:

1. **Service Health Check** - Verify all services are running
2. **Media Library Setup** - Configure media directories
3. **Download Client Configuration** - Set up qBittorrent/Transmission
4. **Indexer Setup** - Configure Prowlarr indexers
5. **User Account Creation** - Set up admin accounts

---

## ⚙️ Service Configuration

### Media Servers

#### Jellyfin Configuration

**Access**: http://localhost:8096

**Initial Setup:**
1. Create admin account
2. Add media libraries:
   - Movies: `/media/movies`
   - TV Shows: `/media/tv`
   - Music: `/media/music`
3. Enable hardware transcoding:
   - Admin → Dashboard → Playback
   - Enable hardware acceleration
   - Select appropriate method (Intel QSV, NVIDIA NVENC, AMD AMF)

**Recommended Settings:**
```yaml
# Hardware Acceleration
Transcoding:
  Hardware acceleration: Intel QuickSync (H.264)
  Hardware encoding: Enabled
  Allow encoding in HEVC format: Yes
  Enable VPP Tone mapping: Yes
  
# Library Settings
Movies:
  Path: /media/movies
  Content type: Movies
  Language: English
  Country: United States
  
TV Shows:
  Path: /media/tv
  Content type: TV Shows
  Season folder pattern: Season {season:00}
  Episode naming: {Series} - S{season:00}E{episode:00} - {Episode}
```

#### Plex Configuration

**Access**: http://localhost:32400/web

**Setup Process:**
1. Sign in with Plex account
2. Claim server with token (if needed)
3. Add libraries with same paths as Jellyfin
4. Configure transcoder settings

#### Emby Configuration

**Access**: http://localhost:8097

Similar setup process to Jellyfin with libraries and transcoding configuration.

### Content Management (*ARR Services)

#### Sonarr (TV Shows)

**Access**: http://localhost:8989

**Configuration Steps:**
1. **Settings → Media Management**
   ```
   Rename episodes: Yes
   Standard Episode Format: {Series Title} - S{season:00}E{episode:00} - {Episode Title} {Quality Full}
   Daily Episode Format: {Series Title} - {Air-Date} - {Episode Title} {Quality Full}
   Season Folder Format: Season {season:00}
   ```

2. **Settings → Profiles**
   - Create quality profiles (1080p, 4K, etc.)
   - Set size limits and quality upgrades

3. **Settings → Download Clients**
   - Add qBittorrent: http://qbittorrent:8080
   - Username: admin
   - Password: (check qBittorrent logs for temporary password)

4. **Settings → Indexers**
   - Add Prowlarr: http://prowlarr:9696
   - API Key: (from Prowlarr settings)

#### Radarr (Movies)

**Access**: http://localhost:7878

Similar configuration to Sonarr but for movies:
- Movie naming format
- Quality profiles for different resolutions
- Same download client and indexer setup

#### Lidarr (Music)

**Access**: http://localhost:8686

**Music-specific settings:**
- Artist folder format: `{Artist Name}`
- Album folder format: `{Artist Name} - {Album Title} ({Release Year})`
- Track file naming: `{track:00} - {Track Title}`

#### Prowlarr (Indexer Manager)

**Access**: http://localhost:9696

**Setup Process:**
1. **Settings → Apps**
   - Add Sonarr: http://sonarr:8989
   - Add Radarr: http://radarr:7878
   - Add Lidarr: http://lidarr:8686
   - Add API keys from each service

2. **Indexers**
   - Add public trackers (1337x, RARBG, etc.)
   - Configure private trackers if available
   - Test each indexer

3. **Download Clients**
   - Add qBittorrent for torrent management

### Download Clients

#### qBittorrent

**Access**: http://localhost:8080

**Initial Setup:**
1. Default login: admin / (check docker logs for password)
2. Change password: Tools → Options → Web UI
3. Configure directories:
   - Default save path: `/downloads/complete`
   - Incomplete torrents: `/downloads/incomplete`
   - Watch folder: `/downloads/watch`

**Recommended Settings:**
```
Downloads:
- Keep incomplete torrents in: /downloads/incomplete
- Copy .torrent files to: /downloads/torrents
- Copy .torrent files for finished downloads to: /downloads/torrents/finished

BitTorrent:
- Maximum active downloads: 5
- Maximum active uploads: 5
- Maximum active torrents: 10

Speed:
- Upload rate limit: 1000 KiB/s (adjust based on internet)
- Download rate limit: 0 (unlimited)

Connection:
- Listening port: 6881
- Enable UPnP/NAT-PMP: Yes
```

#### Transmission (Alternative)

**Access**: http://localhost:9091

Pre-configured through VPN if using Gluetun.

#### SABnzbd (Usenet)

**Access**: http://localhost:8081

**Setup Process:**
1. Run setup wizard
2. Add Usenet provider credentials
3. Configure download directories
4. Set up categories for different content types

---

## 🎛️ Dashboard Guide

### Main Dashboard Features

#### Homepage Dashboard (Port 3001)

**Navigation:**
- **Service Overview**: Real-time status of all services
- **Quick Actions**: Common tasks and shortcuts
- **System Metrics**: CPU, memory, disk usage
- **Recent Activity**: Latest downloads and additions

**Key Features:**
- ✅ Real-time service monitoring
- ✅ Health check automation
- ✅ Integrated search across all services
- ✅ Mobile-responsive design
- ✅ Dark/light theme toggle

#### Homarr Dashboard (Port 7575)

**Features:**
- Customizable widget layout
- Service status indicators
- Integrated search bars
- Weather widgets
- Calendar integration

**Customization:**
1. Click "Edit Mode" toggle
2. Drag and drop widgets
3. Configure service connections
4. Set up custom backgrounds

### AI Assistant Integration

**Access**: Click the robot icon in the dashboard

**Capabilities:**
- Service status queries
- Download management
- Media library statistics
- System performance insights
- Troubleshooting assistance

**Example Commands:**
```
"What's the status of all services?"
"Show me recent downloads"
"Check Jellyfin performance"
"Help me troubleshoot Sonarr"
"What movies were added this week?"
```

### Real-time Monitoring

#### WebSocket Integration

The dashboard uses WebSocket connections for real-time updates:
- Service status changes
- Download progress
- System metrics
- Error notifications

#### Health Check System

Automated health checks every 5 minutes:
- HTTP endpoint testing
- Service responsiveness
- Resource usage monitoring
- Error log analysis

---

## 🎬 Media Management

### Library Organization

#### Directory Structure (TRaSH Guides Standard)

```
media/
├── movies/
│   ├── Movie Name (Year)/
│   │   ├── Movie Name (Year).mkv
│   │   ├── Movie Name (Year).srt
│   │   └── poster.jpg
│   └── Another Movie (2024)/
├── tv/
│   ├── Show Name/
│   │   ├── Season 01/
│   │   │   ├── Show Name - S01E01 - Episode Title.mkv
│   │   │   └── Show Name - S01E02 - Episode Title.mkv
│   │   └── Season 02/
│   └── Another Show/
├── music/
│   ├── Artist Name/
│   │   ├── Album Name (Year)/
│   │   │   ├── 01 - Track Name.flac
│   │   │   └── 02 - Track Name.flac
│   │   └── Another Album (Year)/
│   └── Another Artist/
└── books/
    ├── Author Name/
    │   ├── Book Title (Year)/
    │   │   └── Book Title.epub
    │   └── Another Book (Year)/
    └── Another Author/
```

### Content Acquisition Workflow

#### 1. Request Submission
- Users submit requests via Overseerr/Jellyseerr
- Requests are automatically approved or queued for approval
- Email notifications sent to users

#### 2. Automated Search
- *ARR services search configured indexers
- Quality profiles determine acceptable releases
- Releases are scored and ranked automatically

#### 3. Download Management
- Best release is sent to download client
- Progress monitored in real-time
- Downloads moved to appropriate directories upon completion

#### 4. Post-Processing
- Files renamed according to configured standards
- Metadata and artwork downloaded
- Media libraries updated automatically
- Users notified of new content

### Quality Management

#### Quality Profiles Setup

**4K Profile (Radarr/Sonarr):**
```yaml
Name: 4K-UHD
Qualities:
  - WEBDL-2160p (score: 100)
  - BluRay-2160p (score: 95)
  - HDTV-2160p (score: 90)
  
Size Limits:
  Minimum: 15GB
  Maximum: 100GB
  
Upgrades:
  Enabled: Yes
  Until Score: 100
```

**1080p Profile:**
```yaml
Name: HD-1080p
Qualities:
  - WEBDL-1080p (score: 100)
  - BluRay-1080p (score: 95)
  - HDTV-1080p (score: 90)
  
Size Limits:
  Minimum: 2GB
  Maximum: 15GB
```

#### Custom Formats (TRaSH Guides)

**Movie Custom Formats:**
- DV (Dolby Vision)
- HDR
- IMAX Enhanced
- Streaming Services (Netflix, Amazon, etc.)
- Audio formats (Atmos, DTS-X)
- Release groups (quality indicators)

**TV Show Custom Formats:**
- Streaming optimized
- Scene/P2P groups
- Audio quality indicators
- HDR/DV specifications

---

## 📥 Download Automation

### Indexer Configuration

#### Public Indexers

**Recommended Public Trackers:**
- 1337x
- RARBG (if available in your region)
- The Pirate Bay
- EZTV (TV shows)
- YTS (movies)

**Setup in Prowlarr:**
1. Indexers → Add Indexer
2. Select tracker type
3. Configure connection settings
4. Test indexer connectivity
5. Sync to *ARR applications

#### Private Trackers

**Benefits:**
- Higher quality releases
- Faster download speeds
- Better retention
- Exclusive content

**Popular Private Trackers:**
- PassThePopcorn (movies)
- BroadcasTheNet (TV)
- What.CD alternatives (music)
- RED (music)

### Download Optimization

#### qBittorrent Settings

**Connection Optimization:**
```yaml
Global maximum connections: 200
Maximum connections per torrent: 100
Maximum uploads per torrent: 4
Maximum active downloads: 5
Maximum active uploads: 5

# Port forwarding
Listening port: 6881 (forward this port on your router)
Enable UPnP/NAT-PMP port forwarding: Yes
```

**Speed Settings:**
```yaml
# Adjust based on your internet connection
Upload rate limit: 1000 KiB/s (1 MB/s)
Download rate limit: 0 (unlimited)
Alternative rate limits schedule: Enable during peak hours
```

#### VPN Integration (Gluetun)

**Supported VPN Providers:**
- NordVPN
- Surfshark
- ProtonVPN
- Mullvad
- Private Internet Access

**Configuration:**
```yaml
# In .env file
VPN_PROVIDER=nordvpn
VPN_TYPE=openvpn
OPENVPN_USER=your_username
OPENVPN_PASSWORD=your_password
SERVER_COUNTRIES=Switzerland,Netherlands
```

**Benefits:**
- Anonymous downloading
- Bypass ISP throttling
- Access geo-restricted content
- Kill-switch protection

### Download Categories

#### Sonarr Categories
```yaml
TV-HD: /downloads/complete/tv-hd
TV-UHD: /downloads/complete/tv-uhd
TV-SD: /downloads/complete/tv-sd
```

#### Radarr Categories
```yaml
Movies-HD: /downloads/complete/movies-hd
Movies-UHD: /downloads/complete/movies-uhd
Movies-SD: /downloads/complete/movies-sd
```

---

## 📊 Monitoring & Analytics

### Prometheus Metrics

#### System Metrics Collected
- CPU usage and load averages
- Memory consumption and availability
- Disk space and I/O statistics
- Network traffic and bandwidth
- Container resource usage
- Service response times

#### Media Server Specific Metrics
- Jellyfin/Plex active streams
- Transcoding sessions and performance
- Library scan times and frequency
- User activity and content consumption
- API request rates and errors

### Grafana Dashboards

#### Pre-configured Dashboards

**1. System Overview Dashboard**
- Server hardware utilization
- Docker container status
- Network and storage performance
- Service availability timeline

**2. Media Server Dashboard**
- Active streaming sessions
- Transcoding performance metrics
- User activity heatmaps
- Content library growth over time

**3. Download Performance Dashboard**
- Download speeds and completion rates
- Indexer success rates
- Queue management statistics
- Storage utilization trends

#### Custom Dashboard Creation

**Step-by-step:**
1. Access Grafana: http://localhost:3000
2. Login: admin / admin (change on first login)
3. Click "+" → Dashboard
4. Add Panel → Choose visualization type
5. Configure data source (Prometheus)
6. Write PromQL queries for desired metrics

**Example Queries:**
```promql
# CPU Usage
100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory Usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Active Jellyfin Sessions
jellyfin_active_sessions

# Download Speed
rate(qbittorrent_downloaded_bytes_total[5m])
```

### Uptime Kuma Monitoring

**Access**: http://localhost:3001

**Monitored Services:**
- All media servers (HTTP/HTTPS checks)
- *ARR applications (API endpoints)  
- Download clients (WebUI availability)
- Database services (TCP connections)
- External dependencies (DNS, internet connectivity)

**Alert Configuration:**
- Email notifications
- Discord/Slack webhooks
- Push notifications to mobile devices
- Status page generation

### Log Management (Loki)

#### Centralized Logging

**Log Sources:**
- Docker container logs
- System logs (syslog)
- Application-specific logs
- Nginx access logs
- Error logs from all services

**Log Analysis:**
```logql
# Search for errors across all services
{job="docker"} |= "error" | json | level="error"

# Jellyfin transcoding logs  
{container_name="jellyfin"} |= "transcode"

# Download completion events
{container_name="qbittorrent"} |= "finished"
```

---

## 🔧 Advanced Features

### Hardware Acceleration

#### Intel QuickSync (Recommended)

**Requirements:**
- 8th generation Intel CPU or newer
- Integrated graphics enabled in BIOS
- iGPU not disabled by discrete GPU

**Docker Configuration:**
```yaml
jellyfin:
  devices:
    - /dev/dri:/dev/dri
  environment:
    - LIBVA_DRIVER_NAME=iHD
    - JELLYFIN_FFmpeg__hwaccel_args=-hwaccel vaapi -hwaccel_device /dev/dri/renderD128
```

**Performance Benefits:**
- 10-20x faster transcoding
- 60-80% less CPU usage
- Multiple simultaneous 4K streams
- Reduced power consumption

#### NVIDIA GPU Acceleration

**Requirements:**
- NVIDIA GPU with NVENC support
- nvidia-docker2 installed on host
- Latest NVIDIA drivers

**Setup:**
```bash
# Install nvidia-docker2
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Docker Configuration:**
```yaml
jellyfin:
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all  
    - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
    - JELLYFIN_FFmpeg__hwaccel_args=-hwaccel cuda -hwaccel_output_format cuda
```

### Storage Optimization

#### Cache Configuration

**SSD Cache for Transcoding:**
```yaml
# Mount fast storage for transcoding cache
jellyfin:
  volumes:
    - /mnt/ssd/jellyfin-cache:/cache/transcodes
  environment:
    - JELLYFIN_CACHE_DIR=/cache/transcodes
```

**Benefits:**
- Faster transcoding start times
- Reduced wear on primary storage
- Better concurrent streaming performance
- Temporary file isolation

#### Storage Monitoring

**Disk Usage Alerts:**
```yaml
# Prometheus alert rule
- alert: HighDiskUsage
  expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Disk usage is above 90%"
```

### Network Configuration

#### Reverse Proxy (Caddy)

**Automatic HTTPS:**
```caddyfile
your-domain.com {
    reverse_proxy jellyfin:8096
    encode gzip
    
    # Security headers
    header {
        Strict-Transport-Security "max-age=31536000;"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
    }
}

# Subdomains for services
sonarr.your-domain.com {
    reverse_proxy sonarr:8989
    basicauth {
        user $2a$14$HASHED_PASSWORD
    }
}
```

#### Custom Domains

**Local DNS (Pi-hole):**
```
# /etc/pihole/custom.list
192.168.1.100 media.local
192.168.1.100 jellyfin.local
192.168.1.100 sonarr.local
192.168.1.100 radarr.local
```

### Backup & Recovery

#### Automated Backups

**Configuration Backup Script:**
```bash
#!/bin/bash
# backup-config.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
tar -czf "$BACKUP_DIR/config-backup.tar.gz" config/
tar -czf "$BACKUP_DIR/docker-compose.tar.gz" docker-compose.yml .env

# Backup databases
docker exec postgres pg_dumpall -U postgres > "$BACKUP_DIR/postgres-backup.sql"
docker exec mariadb mysqldump --all-databases -u root -p$MYSQL_ROOT_PASSWORD > "$BACKUP_DIR/mysql-backup.sql"

# Keep only last 7 days
find /backups -type d -mtime +7 -exec rm -rf {} \;
```

**Restore Process:**
```bash
#!/bin/bash
# restore-config.sh

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/restore"

# Extract backup
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Stop services
docker compose down

# Restore configurations
cp -r "$RESTORE_DIR/config"/* config/
cp "$RESTORE_DIR/docker-compose.yml" .
cp "$RESTORE_DIR/.env" .

# Start services
docker compose up -d
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Services Won't Start

**Symptoms:**
- Container exits immediately
- "Port already in use" errors
- Permission denied errors

**Solutions:**

1. **Check port conflicts:**
```bash
# Find what's using a port
sudo netstat -tulpn | grep :8096
sudo lsof -i :8096

# Kill conflicting process
sudo kill -9 PID
```

2. **Fix permissions:**
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/ultimate-media-server

# Fix permissions
chmod -R 755 ~/ultimate-media-server/config
chmod -R 755 ~/ultimate-media-server/media
chmod -R 755 ~/ultimate-media-server/downloads
```

3. **Check logs:**
```bash
# View specific service logs
docker compose logs jellyfin
docker compose logs sonarr

# Follow logs in real-time
docker compose logs -f --tail=50
```

#### Can't Access Web Interfaces

**Symptoms:**
- Connection refused errors
- Blank pages
- 502 Bad Gateway errors

**Solutions:**

1. **Verify container status:**
```bash
docker compose ps
# All services should show "Up"
```

2. **Check internal networking:**
```bash
# Test internal connectivity  
docker exec -it jellyfin ping sonarr
docker exec -it sonarr ping prowlarr
```

3. **Verify firewall settings:**
```bash
# Ubuntu/Debian
sudo ufw status
sudo ufw allow 8096/tcp

# CentOS/RHEL
sudo firewall-cmd --list-ports
sudo firewall-cmd --add-port=8096/tcp --permanent
sudo firewall-cmd --reload
```

#### Slow Performance

**Symptoms:**
- Long loading times
- Buffering during playback
- High CPU/memory usage

**Solutions:**

1. **Enable hardware acceleration:**
```yaml
# In docker-compose.yml
jellyfin:
  devices:
    - /dev/dri:/dev/dri  # Intel QuickSync
  # OR
  runtime: nvidia        # NVIDIA GPU
```

2. **Optimize transcode settings:**
```bash
# Jellyfin transcoding settings
Admin Dashboard → Playback:
- Hardware acceleration: Intel QuickSync
- Hardware encoding: Enabled
- Throttle transcodes: Enabled
```

3. **Check system resources:**
```bash
# Monitor system resources
htop
docker stats

# Check disk I/O
iotop
```

#### Download Issues  

**Symptoms:**
- Downloads stuck in queue
- "No indexers available" errors
- Slow download speeds

**Solutions:**

1. **Check indexer connectivity:**
```bash
# Test indexer from Prowlarr
Prowlarr → Indexers → Test All
```

2. **Verify download client settings:**
```yaml
# qBittorrent configuration
Downloads:
  Default save path: /downloads/complete
  Keep incomplete torrents in: /downloads/incomplete
  
Connection:
  Listening port: 6881 (ensure this is forwarded)
  Enable UPnP/NAT-PMP: Yes
```

3. **Check VPN connectivity (if using Gluetun):**
```bash
# Check VPN container logs
docker compose logs gluetun

# Test external IP
docker exec gluetun curl ifconfig.me
```

### Database Issues

#### Corrupt Database Recovery

**Jellyfin Database:**
```bash
# Stop Jellyfin
docker compose stop jellyfin

# Backup current database
cp config/jellyfin/data/jellyfin.db config/jellyfin/data/jellyfin.db.backup

# Rebuild database
docker compose run --rm jellyfin sqlite3 /config/data/jellyfin.db "VACUUM;"

# Start Jellyfin
docker compose start jellyfin
```

**Sonarr/Radarr Database:**
```bash
# Stop service
docker compose stop sonarr

# Check database integrity
docker compose run --rm sonarr sqlite3 /config/sonarr.db "PRAGMA integrity_check;"

# Vacuum database
docker compose run --rm sonarr sqlite3 /config/sonarr.db "VACUUM;"

# Start service
docker compose start sonarr
```

### Performance Tuning

#### Optimizing Docker Performance

**Docker Daemon Configuration (`/etc/docker/daemon.json`):**
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "dns": ["1.1.1.1", "8.8.8.8"]
}
```

**Container Resource Limits:**
```yaml
# In docker-compose.yml
services:
  jellyfin:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

#### System-Level Optimizations

**Network Tuning:**
```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 65536 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

**Disk I/O Optimization:**
```bash
# Set I/O scheduler for SSDs
echo noop > /sys/block/sda/queue/scheduler

# Increase dirty page timeout for HDDs
echo vm.dirty_expire_centisecs = 12000 >> /etc/sysctl.conf
```

---

## 🔌 API Reference

### Dashboard API Endpoints

#### Service Management

**Get All Services Status:**
```http
GET /api/services
```

**Response:**
```json
{
  "success": true,
  "services": {
    "jellyfin": {
      "name": "Jellyfin",
      "url": "http://localhost:8096",
      "type": "media-server",
      "health": {
        "status": "healthy",
        "last_check": "2025-01-15T10:30:00Z",
        "response_time": 150
      }
    }
  }
}
```

**Health Check Single Service:**
```http
POST /api/services/{service}/health
```

**Restart Service:**
```http
POST /api/services/{service}/restart
```

#### System Information

**Get System Stats:**
```http
GET /api/system
```

**Response:**
```json
{
  "success": true,
  "system": {
    "cpu_usage": 25.4,
    "memory_usage": 68.2,
    "disk_usage": 45.1,
    "uptime": 86400,
    "services": {
      "total": 15,
      "healthy": 13,
      "unhealthy": 1,
      "unknown": 1
    }
  }
}
```

#### Download Management

**Get Download Queue:**
```http
GET /api/downloads
```

**Add Download:**
```http
POST /api/downloads
Content-Type: application/json

{
  "url": "magnet:?xt=...",
  "category": "movies",
  "priority": "high"
}
```

### Service-Specific APIs

#### Jellyfin API

**Authentication:**
```http
POST http://localhost:8096/Users/authenticatebyname
Content-Type: application/json

{
  "Username": "admin",
  "Pw": "password"
}
```

**Get Library Items:**
```http
GET http://localhost:8096/Items?api_key=YOUR_API_KEY
```

#### Sonarr API

**Base URL:** `http://localhost:8989/api/v3`

**Get Series:**
```http
GET /api/v3/series?apikey=YOUR_API_KEY
```

**Add Series:**
```http
POST /api/v3/series?apikey=YOUR_API_KEY
Content-Type: application/json

{
  "title": "Series Name",
  "tvdbId": 12345,
  "qualityProfile": 1,
  "languageProfile": 1,
  "path": "/media/tv/Series Name",
  "monitored": true
}
```

#### Radarr API

**Base URL:** `http://localhost:7878/api/v3`

**Get Movies:**
```http
GET /api/v3/movie?apikey=YOUR_API_KEY
```

**Add Movie:**
```http
POST /api/v3/movie?apikey=YOUR_API_KEY
Content-Type: application/json

{
  "title": "Movie Title",
  "tmdbId": 12345,
  "qualityProfile": 1,
  "path": "/media/movies/Movie Title (2024)",
  "monitored": true
}
```

### Webhook Integration

#### Discord Notifications

**Setup Webhook:**
```yaml
# In Sonarr/Radarr settings
Connection Type: Webhook
Name: Discord Notifications
URL: https://discord.com/api/webhooks/YOUR_WEBHOOK_URL

# Triggers
On Grab: Yes
On Import: Yes
On Upgrade: Yes
On Health Issue: Yes
```

**Custom Webhook Payload:**
```json
{
  "content": null,
  "embeds": [
    {
      "title": "{{EventType}} - {{Series.Title}}",
      "description": "{{EpisodeFile.RelativePath}}",
      "color": 3066993,
      "timestamp": "{{UtcTime}}",
      "thumbnail": {
        "url": "{{Series.Images.0.Url}}"
      }
    }
  ]
}
```

### Custom Scripts

#### Post-Processing Script

**Location:** `/config/scripts/post-process.sh`

```bash
#!/bin/bash
# Post-processing script for completed downloads

CATEGORY="$1"
FILEPATH="$2"
FILENAME="$3"

case "$CATEGORY" in
  "movies")
    # Run Radarr import
    curl -X POST "http://radarr:7878/api/v3/command" \
      -H "X-Api-Key: YOUR_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"name": "DownloadedMoviesScan"}'
    ;;
  "tv")
    # Run Sonarr import
    curl -X POST "http://sonarr:8989/api/v3/command" \
      -H "X-Api-Key: YOUR_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"name": "DownloadedEpisodesScan"}'
    ;;
esac

# Update Jellyfin library
curl -X POST "http://jellyfin:8096/Library/Refresh" \
  -H "X-Emby-Token: YOUR_API_KEY"

# Send notification
curl -X POST "YOUR_DISCORD_WEBHOOK" \
  -H "Content-Type: application/json" \
  -d "{\"content\": \"Download completed: $FILENAME\"}"
```

---

## 📱 Mobile Access & Apps

### Recommended Mobile Apps

#### Media Streaming Apps

**Jellyfin:**
- **Android:** Jellyfin for Android
- **iOS:** Jellyfin Mobile
- **Features:** Native transcoding, offline sync, cast support

**Plex:**
- **Android/iOS:** Plex
- **Features:** Premium features, offline sync, live TV

#### Media Management Apps

**Sonarr/Radarr:**
- **Android:** NZB360, LunaSea
- **iOS:** LunaSea, Remotely for Sonarr
- **Features:** Queue management, search, statistics

**qBittorrent:**
- **Android:** qBittorrent Controller
- **iOS:** iTransmission 4
- **Features:** Download control, RSS feeds, remote management

### Remote Access Setup

#### Tailscale VPN (Recommended)

**Setup:**
```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Connect to network
sudo tailscale up

# Get machine IP
tailscale ip -4
```

**Benefits:**
- Secure end-to-end encryption
- No port forwarding required
- Works behind NAT/firewalls
- Easy device management

#### Cloudflare Tunnel

**Setup:**
```bash
# Install cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create mediaserver

# Configure tunnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: mediaserver
credentials-file: ~/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: jellyfin.yourdomain.com
    service: http://localhost:8096
  - hostname: sonarr.yourdomain.com
    service: http://localhost:8989
  - service: http_status:404
EOF

# Run tunnel
cloudflared tunnel run mediaserver
```

---

## 🔐 Security & Privacy

### Security Best Practices

#### Authentication & Authorization

**Enable 2FA where supported:**
- Jellyfin: Admin → Users → Enable Two-Factor Authentication
- Overseerr: Settings → Users → Two-Factor Authentication
- Grafana: Profile → Two-Factor Auth

**Strong Password Policy:**
```bash
# Generate secure passwords
openssl rand -base64 32

# Use password managers
# Recommended: Bitwarden, 1Password, KeePass
```

#### Network Security

**Firewall Configuration:**
```bash
# Ubuntu UFW
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific services
sudo ufw allow 8096/tcp comment 'Jellyfin'
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow from 192.168.1.0/24 to any port 8989 comment 'Sonarr LAN only'
```

**Reverse Proxy Security:**
```caddyfile
# Caddy security headers
header {
  # HSTS
  Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
  
  # Content Security Policy
  Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'"
  
  # Other security headers
  X-Content-Type-Options "nosniff"
  X-Frame-Options "SAMEORIGIN"
  X-XSS-Protection "1; mode=block"
  Referrer-Policy "strict-origin-when-cross-origin"
}
```

#### Container Security

**Non-root User Configuration:**
```yaml
# In docker-compose.yml
services:
  jellyfin:
    user: "${PUID}:${PGID}"
    environment:
      - PUID=${PUID}
      - PGID=${PGID}
```

**Read-only Root Filesystem:**
```yaml
services:
  jellyfin:
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
```

### Privacy Considerations

#### Data Collection Opt-out

**Jellyfin Privacy Settings:**
```yaml
# In Jellyfin Admin Dashboard
Settings → General:
  - Enable metrics collection: Disabled
  - Send anonymous usage statistics: Disabled
  
Settings → DLNA:
  - Enable DLNA server: Disabled (if not needed)
```

**Docker Privacy:**
```bash
# Disable Docker analytics
mkdir -p ~/.docker
echo '{"auths":{},"HttpHeaders":{"User-Agent":"Docker-Client/19.03.0 (linux)"}}' > ~/.docker/config.json
```

#### Log Management

**Log Rotation Configuration:**
```yaml
# In docker-compose.yml
services:
  jellyfin:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Sensitive Data Filtering:**
```bash
# Filter sensitive data from logs
docker logs jellyfin 2>&1 | grep -v -E "(password|token|key)"
```

---

## 🎓 Advanced Tutorials

### Custom Development

#### Creating Custom Widgets

**Dashboard Widget Example:**
```javascript
// custom-widget.js
class CustomWidget {
    constructor(containerId, config) {
        this.container = document.getElementById(containerId);
        this.config = config;
        this.init();
    }
    
    async init() {
        await this.fetchData();
        this.render();
        this.startUpdates();
    }
    
    async fetchData() {
        try {
            const response = await fetch('/api/custom-data');
            this.data = await response.json();
        } catch (error) {
            console.error('Failed to fetch data:', error);
        }
    }
    
    render() {
        this.container.innerHTML = `
            <div class="custom-widget">
                <h3>${this.config.title}</h3>
                <div class="widget-content">
                    ${this.data.map(item => `
                        <div class="widget-item">${item.name}: ${item.value}</div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    startUpdates() {
        setInterval(() => {
            this.fetchData().then(() => this.render());
        }, this.config.updateInterval || 30000);
    }
}

// Usage
const widget = new CustomWidget('widget-container', {
    title: 'Custom Metrics',
    updateInterval: 15000
});
```

#### Plugin Development

**Jellyfin Plugin Structure:**
```csharp
// Plugin.cs
using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Model.Plugins;
using MediaBrowser.Model.Serialization;

namespace MyCustomPlugin
{
    public class Plugin : BasePlugin<PluginConfiguration>
    {
        public override string Name => "My Custom Plugin";
        public override Guid Id => Guid.Parse("12345678-1234-1234-1234-123456789012");
        
        public Plugin(IApplicationPaths applicationPaths, IXmlSerializer xmlSerializer) 
            : base(applicationPaths, xmlSerializer)
        {
        }
    }
}
```

### Integration Examples

#### Home Assistant Integration

**Configuration.yaml:**
```yaml
# Home Assistant integration
rest:
  - resource: http://192.168.1.100:3001/api/system
    scan_interval: 30
    sensor:
      - name: "Media Server CPU"
        value_template: "{{ value_json.system.cpu_usage }}"
        unit_of_measurement: "%"
      - name: "Media Server Memory"
        value_template: "{{ value_json.system.memory_usage }}"
        unit_of_measurement: "%"

automation:
  - alias: "Media Server Alert"
    trigger:
      platform: numeric_state
      entity_id: sensor.media_server_cpu
      above: 90
    action:
      service: notify.mobile_app
      data:
        message: "Media server CPU usage is high: {{ states('sensor.media_server_cpu') }}%"
```

#### Node-RED Integration

**Flow Example:**
```json
[
    {
        "id": "1",
        "type": "http request",
        "name": "Get Media Server Status",
        "method": "GET",
        "url": "http://localhost:3001/api/services",
        "x": 200,
        "y": 100
    },
    {
        "id": "2",
        "type": "function",
        "name": "Process Status",
        "func": "const services = msg.payload.services;\nconst unhealthy = Object.values(services).filter(s => s.health.status !== 'healthy');\n\nif (unhealthy.length > 0) {\n    msg.payload = `${unhealthy.length} services are unhealthy: ${unhealthy.map(s => s.name).join(', ')}`;\n    return msg;\n}\n\nreturn null;",
        "x": 400,
        "y": 100
    },
    {
        "id": "3",
        "type": "telegram sender",
        "name": "Send Alert",
        "x": 600,
        "y": 100
    }
]
```

---

## 📚 Additional Resources

### Official Documentation Links

- **Docker Compose**: https://docs.docker.com/compose/
- **Jellyfin**: https://jellyfin.org/docs/
- **Sonarr**: https://wiki.servarr.com/sonarr
- **Radarr**: https://wiki.servarr.com/radarr
- **Prowlarr**: https://wiki.servarr.com/prowlarr
- **qBittorrent**: https://github.com/qbittorrent/qBittorrent/wiki
- **Grafana**: https://grafana.com/docs/grafana/latest/
- **Prometheus**: https://prometheus.io/docs/

### Community Resources

**Forums & Communities:**
- **Reddit**: r/selfhosted, r/jellyfin, r/sonarr
- **Discord**: Jellyfin Discord, Sonarr/Radarr Discord
- **GitHub**: Project repositories and issue trackers

**YouTube Channels:**
- TRaSH Guides
- SpaceInvader One
- Ibracorp
- TechnoTim

### TRaSH Guides Integration

**Quality Profiles**: https://trash-guides.info/
**Custom Formats**: Pre-configured for optimal quality
**Naming Schemes**: Standardized file organization

---

## 📄 Changelog & Updates

### Version 2025.1.0 (Current)

**New Features:**
- ✅ Real-time dashboard with WebSocket support
- ✅ AI assistant integration
- ✅ Enhanced mobile responsiveness
- ✅ Comprehensive monitoring suite
- ✅ Hardware acceleration optimization
- ✅ VPN integration with Gluetun
- ✅ Automated backup system

**Improvements:**
- 🔧 Faster container startup times
- 🔧 Reduced memory footprint
- 🔧 Better error handling and recovery
- 🔧 Enhanced security configurations
- 🔧 Improved documentation

**Bug Fixes:**
- 🐛 Fixed port conflicts in multi-container setup
- 🐛 Resolved permissions issues on macOS
- 🐛 Corrected health check timeouts
- 🐛 Fixed database initialization race conditions

### Upcoming Features (2025.2.0)

**Planned Additions:**
- 🔮 Kubernetes deployment support
- 🔮 Multi-user dashboard customization
- 🔮 Enhanced AI recommendations
- 🔮 Automated content curation
- 🔮 Advanced analytics and reporting
- 🔮 Mobile app for system management

---

## 💡 Tips & Best Practices

### Performance Optimization Tips

1. **Use SSD for frequently accessed data**
   - OS and Docker on SSD
   - Transcoding cache on NVMe if available
   - Media can remain on HDD

2. **Optimize network settings**
   - Use wired connections for servers
   - Configure QoS for streaming traffic
   - Monitor bandwidth usage

3. **Resource allocation**
   - Allocate sufficient RAM for transcoding
   - Reserve CPU cores for critical services
   - Monitor and adjust container limits

### Maintenance Schedule

**Daily (Automated):**
- Health checks for all services
- Log rotation and cleanup
- Backup verification

**Weekly:**
- Update container images
- Review system performance
- Check disk space usage

**Monthly:**
- Security updates for host system
- Review and cleanup old backups
- Performance optimization review

### Content Organization Tips

1. **Follow TRaSH Guides standards**
2. **Use consistent naming conventions**
3. **Organize by content type and quality**
4. **Implement proper folder structures**
5. **Regular library maintenance**

---

*This manual is continuously updated. For the latest version and community contributions, visit our [GitHub repository](https://github.com/yourusername/ultimate-media-server).*

---

<div align="center">
  <p><strong>Ultimate Media Server 2025 - Complete User Manual</strong></p>
  <p>Made with ❤️ by the self-hosting community</p>
  <p><a href="#-table-of-contents">Back to Top</a></p>
</div>