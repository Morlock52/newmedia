# Ultimate Media Server 2025 - Complete Installation Guide

![Media Server Banner](./images/banner-placeholder.png)

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Installation Instructions](#installation-instructions)
   - [Windows Installation](#windows-installation)
   - [macOS Installation](#macos-installation)
   - [Linux Installation](#linux-installation)
4. [Configuration Options](#configuration-options)
5. [Post-Installation Setup](#post-installation-setup)
6. [Service Access](#service-access)
7. [Updating](#updating)
8. [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| **CPU** | 4 cores / 8 threads |
| **RAM** | 8 GB |
| **Storage** | 100 GB SSD (OS + Apps) |
| **Network** | 100 Mbps connection |
| **OS** | Windows 10/11, macOS 10.15+, Ubuntu 20.04+ |

### Recommended Requirements

| Component | Recommended Specification |
|-----------|--------------------------|
| **CPU** | 8+ cores / 16+ threads |
| **RAM** | 16-32 GB |
| **Storage** | 500 GB SSD (OS + Apps) + Multi-TB HDD (Media) |
| **Network** | 1 Gbps connection |
| **GPU** | NVIDIA GPU for hardware transcoding |

### Storage Planning

```
📁 Storage Layout Recommendation
├── 🖥️ OS Drive (SSD)
│   ├── Operating System: 50 GB
│   ├── Docker Images: 30 GB
│   └── App Configs: 20 GB
│
└── 💾 Media Drive (HDD/NAS)
    ├── Movies: 2-4 TB
    ├── TV Shows: 2-4 TB
    ├── Music: 500 GB
    ├── Books/Audiobooks: 100 GB
    └── Downloads: 500 GB
```

---

## Pre-Installation Checklist

### ✅ Required Software

- [ ] **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux)
- [ ] **Docker Compose** v2.0+ 
- [ ] **Git** (for cloning repository)
- [ ] **Text Editor** (VS Code, Notepad++, etc.)

### ✅ Network Requirements

- [ ] **Ports Available**: Ensure ports 80, 443, 8096, 7878, 8989, etc. are not in use
- [ ] **Static IP** (recommended) or **DHCP Reservation**
- [ ] **Router Access** for port forwarding (optional, for remote access)
- [ ] **Domain Name** (optional, for SSL certificates)

### ✅ Accounts & API Keys

- [ ] **VPN Account** (NordVPN, Mullvad, etc.) - Optional but recommended
- [ ] **Plex Account** (if using Plex) - Free or Premium
- [ ] **TMDB API Key** (for metadata) - Free
- [ ] **Cloudflare Account** (for tunnels) - Optional

### ✅ System Preparation

- [ ] **Update System**: Ensure OS is up to date
- [ ] **Firewall Rules**: Configure if necessary
- [ ] **Antivirus Exclusions**: Add Docker directories
- [ ] **Backup**: Create system restore point

---

## Installation Instructions

### Windows Installation

#### Step 1: Install Docker Desktop

1. **Download Docker Desktop**
   ```powershell
   # Open PowerShell as Administrator
   winget install Docker.DockerDesktop
   ```
   
   Or download from: https://www.docker.com/products/docker-desktop/

2. **Configure Docker Desktop**
   - Enable WSL 2 backend
   - Allocate resources:
     - CPUs: 4+
     - Memory: 8GB+
     - Disk: 100GB+

3. **Restart Computer**

#### Step 2: Clone Repository

```powershell
# Open PowerShell
cd C:\
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server
```

#### Step 3: Configure Environment

```powershell
# Copy environment template
Copy-Item .env.example .env

# Edit configuration
notepad .env
```

#### Step 4: Deploy Services

```powershell
# Start all services
docker-compose up -d

# Or use the deployment script
.\deploy-ultimate-2025.ps1
```

#### Step 5: Verify Installation

```powershell
# Check running containers
docker ps

# View logs
docker-compose logs -f
```

![Windows Installation](./images/windows-install-placeholder.png)

---

### macOS Installation

#### Step 1: Install Prerequisites

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker Desktop
brew install --cask docker

# Install Git
brew install git
```

#### Step 2: Configure Docker Desktop

1. **Launch Docker Desktop**
2. **Go to Preferences → Resources**
   - CPUs: 4+
   - Memory: 8GB+
   - Disk: 100GB+
3. **Apply & Restart**

#### Step 3: Clone and Configure

```bash
# Clone repository
cd ~/
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server

# Copy and edit environment
cp .env.example .env
nano .env
```

#### Step 4: Deploy Services

```bash
# Make script executable
chmod +x deploy-ultimate-2025.sh

# Run deployment
./deploy-ultimate-2025.sh
```

#### Step 5: Access Services

```bash
# Open main dashboard
open http://localhost:3001

# Check service status
docker ps
```

![macOS Installation](./images/macos-install-placeholder.png)

---

### Linux Installation

#### Step 1: Install Docker

**Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y curl git apt-transport-https ca-certificates software-properties-common

# Add Docker repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**RHEL/CentOS/Fedora:**
```bash
# Install Docker
sudo dnf install -y docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

#### Step 2: Clone Repository

```bash
# Create directory
mkdir -p ~/media-server
cd ~/media-server

# Clone repository
git clone https://github.com/yourusername/ultimate-media-server.git .
```

#### Step 3: Configure Environment

```bash
# Copy environment file
cp .env.example .env

# Edit configuration
nano .env
```

#### Step 4: Set Permissions

```bash
# Get user/group IDs
id

# Update .env with your PUID and PGID
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env
```

#### Step 5: Deploy Services

```bash
# Make script executable
chmod +x deploy-ultimate-2025.sh

# Run deployment
./deploy-ultimate-2025.sh

# Or use docker-compose directly
docker compose up -d
```

#### Step 6: Configure Firewall (if enabled)

```bash
# UFW (Ubuntu)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8096/tcp
sudo ufw allow 7878/tcp
sudo ufw allow 8989/tcp
# ... add other required ports

# Firewalld (RHEL/CentOS)
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
# ... add other required ports
sudo firewall-cmd --reload
```

![Linux Installation](./images/linux-install-placeholder.png)

---

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `TZ` | Timezone | `America/New_York` | Yes |
| `PUID` | User ID | `1000` | Yes |
| `PGID` | Group ID | `1000` | Yes |
| `MEDIA_PATH` | Media storage path | `./media-data` | Yes |
| `DOWNLOADS_PATH` | Downloads path | `./downloads` | Yes |
| `VPN_PROVIDER` | VPN service provider | `none` | No |
| `VPN_USER` | VPN username | - | If VPN enabled |
| `VPN_PASSWORD` | VPN password | - | If VPN enabled |
| `DOMAIN` | Your domain name | `localhost` | No |

### Volume Mappings

```yaml
# Media Storage
- ${MEDIA_PATH}/movies:/movies
- ${MEDIA_PATH}/tv:/tv
- ${MEDIA_PATH}/music:/music
- ${MEDIA_PATH}/books:/books

# Application Data
- ./config/service-name:/config

# Downloads
- ${DOWNLOADS_PATH}:/downloads
```

### Network Configuration

```yaml
networks:
  media-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## Post-Installation Setup

### 1. Initial Service Configuration

#### Jellyfin Setup
1. Navigate to http://localhost:8096
2. Follow setup wizard
3. Create admin account
4. Add media libraries
5. Configure transcoding settings

#### Sonarr/Radarr Setup
1. Access Sonarr: http://localhost:8989
2. Access Radarr: http://localhost:7878
3. Configure:
   - Authentication
   - Download clients
   - Indexers (via Prowlarr)
   - Media folders

#### Prowlarr Setup
1. Navigate to http://localhost:9696
2. Add indexers
3. Configure app connections:
   - Sonarr
   - Radarr
   - Lidarr
   - Readarr

### 2. Security Configuration

```bash
# Generate secure passwords
openssl rand -base64 32

# Update .env file with secure passwords
nano .env
```

### 3. SSL/TLS Setup (Optional)

```bash
# Using Nginx Proxy Manager
# Access at http://localhost:81
# Default login: admin@example.com / changeme

# Add SSL certificates for each service
```

### 4. Backup Configuration

```bash
# Create backup script
cat > backup-media-server.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configurations
docker run --rm -v media-server_config:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/config_$DATE.tar.gz -C /data .

# Backup database
docker exec postgres pg_dumpall -U postgres > $BACKUP_DIR/postgres_$DATE.sql
EOF

chmod +x backup-media-server.sh
```

---

## Service Access

### Main Dashboards

| Service | URL | Default Port | Description |
|---------|-----|--------------|-------------|
| **Homepage** | http://localhost:3001 | 3001 | Main dashboard |
| **Homarr** | http://localhost:7575 | 7575 | Alternative dashboard |
| **Portainer** | https://localhost:9443 | 9443 | Container management |

### Media Services

| Service | URL | Default Port | Purpose |
|---------|-----|--------------|---------|
| **Jellyfin** | http://localhost:8096 | 8096 | Media streaming |
| **Plex** | http://localhost:32400/web | 32400 | Premium media streaming |
| **Emby** | http://localhost:8097 | 8097 | Alternative streaming |

### Media Management

| Service | URL | Default Port | Purpose |
|---------|-----|--------------|---------|
| **Sonarr** | http://localhost:8989 | 8989 | TV show management |
| **Radarr** | http://localhost:7878 | 7878 | Movie management |
| **Lidarr** | http://localhost:8686 | 8686 | Music management |
| **Readarr** | http://localhost:8787 | 8787 | Book management |
| **Bazarr** | http://localhost:6767 | 6767 | Subtitle management |
| **Prowlarr** | http://localhost:9696 | 9696 | Indexer management |

### Request Services

| Service | URL | Default Port | Purpose |
|---------|-----|--------------|---------|
| **Overseerr** | http://localhost:5056 | 5056 | Media requests (Plex) |
| **Jellyseerr** | http://localhost:5055 | 5055 | Media requests (Jellyfin) |
| **Ombi** | http://localhost:3579 | 3579 | Universal requests |

### Download Clients

| Service | URL | Default Port | Purpose |
|---------|-----|--------------|---------|
| **qBittorrent** | http://localhost:8080 | 8080 | Torrent client |
| **SABnzbd** | http://localhost:8081 | 8081 | Usenet client |

### Monitoring

| Service | URL | Default Port | Purpose |
|---------|-----|--------------|---------|
| **Grafana** | http://localhost:3000 | 3000 | Metrics visualization |
| **Prometheus** | http://localhost:9090 | 9090 | Metrics collection |
| **Uptime Kuma** | http://localhost:3011 | 3011 | Service monitoring |

---

## Updating

### Automatic Updates (Watchtower)

Watchtower automatically updates containers daily. To configure:

```yaml
# In docker-compose.yml
watchtower:
  environment:
    - WATCHTOWER_CLEANUP=true
    - WATCHTOWER_POLL_INTERVAL=86400  # 24 hours
    - WATCHTOWER_INCLUDE_RESTARTING=true
```

### Manual Updates

```bash
# Update all services
docker-compose pull
docker-compose up -d

# Update specific service
docker-compose pull jellyfin
docker-compose up -d jellyfin

# View update logs
docker-compose logs -f watchtower
```

### Backup Before Updating

```bash
# Create full backup
./backup-media-server.sh

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d
```

---

## Uninstallation

### Complete Removal

```bash
# Stop all services
docker-compose down

# Remove containers and networks
docker-compose down --rmi all -v --remove-orphans

# Remove data (CAUTION: This deletes all media and configurations)
rm -rf ./config ./media-data ./downloads

# Remove Docker images
docker system prune -a --volumes
```

### Partial Removal

```bash
# Remove specific service
docker-compose stop servicename
docker-compose rm servicename

# Remove unused resources
docker system prune
```

---

## Next Steps

1. **Read the [Quick Start Guide](./QUICKSTART_GUIDE.md)** for rapid deployment
2. **Check the [Troubleshooting Guide](./TROUBLESHOOTING_GUIDE.md)** if you encounter issues
3. **Configure services** according to your needs
4. **Set up automated backups** for your configuration
5. **Join our community** for support and updates

---

## Support

- **Documentation**: [https://docs.mediaserver.com](https://docs.mediaserver.com)
- **GitHub Issues**: [https://github.com/yourusername/ultimate-media-server/issues](https://github.com/yourusername/ultimate-media-server/issues)
- **Discord Community**: [https://discord.gg/mediaserver](https://discord.gg/mediaserver)
- **Reddit**: [r/selfhosted](https://reddit.com/r/selfhosted)

---

*Last updated: November 2024*