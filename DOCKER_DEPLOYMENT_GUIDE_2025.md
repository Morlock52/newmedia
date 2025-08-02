# 🚀 Ultimate Media Server 2025 - Docker Deployment & Maintenance Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites) 
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Security Hardening](#security-hardening)
- [Monitoring Setup](#monitoring-setup)
- [Update Management](#update-management)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Migration Guide](#migration-guide)

---

## Quick Start

**Get your media server running in 5 minutes!**

### 🎯 One-Line Deploy (For the Impatient)

```bash
curl -sSL https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/quick-deploy.sh | bash
```

### 🔧 Manual Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server

# 2. Copy environment template
cp .env.template .env

# 3. Deploy with default settings
./deploy-ultimate-2025.sh

# 4. Open the dashboard
open http://localhost:3001
```

**That's it!** Your media server is now running. Continue reading for customization and optimization.

---

## Prerequisites

### System Requirements

#### Minimum Specifications
- **CPU**: 2 cores (x86_64 or ARM64)
- **RAM**: 4GB
- **Storage**: 50GB for applications + media storage
- **OS**: Linux, macOS, or Windows with WSL2
- **Network**: Stable internet connection

#### Recommended Specifications
- **CPU**: 4+ cores with hardware transcoding support
- **RAM**: 8GB+ (16GB for heavy usage)
- **Storage**: SSD for applications, HDD for media
- **GPU**: Intel QuickSync, NVIDIA, or AMD for transcoding
- **Network**: Gigabit ethernet

### Software Requirements

#### 1. Install Docker

**Linux (Ubuntu/Debian):**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
```

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop
```

**Windows:**
1. Enable WSL2: `wsl --install`
2. Download Docker Desktop: https://www.docker.com/products/docker-desktop
3. Enable WSL2 integration in Docker settings

#### 2. Install Docker Compose

```bash
# Check if already installed with Docker
docker compose version

# If not installed, get it separately
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Additional Tools

```bash
# Install helpful utilities
sudo apt install -y git curl wget htop ncdu
```

---

## Configuration

### Step 1: Environment Setup

#### Create Configuration File

```bash
# Copy the template
cp .env.template .env

# Edit with your favorite editor
nano .env
```

#### Essential Settings to Configure

```bash
# 1. Basic Settings
TZ=America/New_York          # Your timezone
PUID=1000                    # Your user ID (run: id -u)
PGID=1000                    # Your group ID (run: id -g)

# 2. Storage Paths (IMPORTANT!)
MEDIA_ROOT=/path/to/media    # Where your media files will be stored
DOWNLOADS_ROOT=/path/to/downloads  # Download location
CONFIG_ROOT=./config         # Application configs (keep default)

# 3. Domain Configuration (for remote access)
DOMAIN=example.com           # Your domain
CLOUDFLARE_EMAIL=you@email.com
CLOUDFLARE_API_KEY=your-key-here

# 4. Security (CHANGE THESE!)
ADMIN_PASSWORD=change-me-please
DB_PASSWORD=secure-database-password
VPN_PASSWORD=your-vpn-password
```

### Step 2: Directory Structure

The deployment script creates this structure automatically:

```
ultimate-media-server/
├── config/              # Application configurations
│   ├── jellyfin/       # Media server config
│   ├── sonarr/         # TV automation
│   ├── radarr/         # Movie automation
│   └── ...             # Other services
├── data/               # Media and downloads
│   ├── media/          
│   │   ├── movies/     # Your movie library
│   │   ├── tv/         # TV shows
│   │   ├── music/      # Music collection
│   │   └── books/      # E-books
│   └── downloads/      # Temporary downloads
├── logs/               # Application logs
├── backups/            # Backup storage
└── secrets/            # Sensitive data (excluded from git)
```

### Step 3: Network Configuration

#### Port Mappings

| Service | Port | Purpose |
|---------|------|---------|
| Jellyfin | 8096 | Media streaming |
| Sonarr | 8989 | TV show management |
| Radarr | 7878 | Movie management |
| Prowlarr | 9696 | Indexer management |
| qBittorrent | 8080 | Download client |
| Overseerr | 5055 | Request management |
| Homepage | 3001 | Main dashboard |
| Traefik | 80/443 | Reverse proxy |

#### Firewall Rules

```bash
# Allow required ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8096/tcp  # Jellyfin
# Add other ports as needed
```

---

## Deployment

### Method 1: Automated Deployment (Recommended)

```bash
# Run the deployment script
./deploy-ultimate-2025.sh

# For specific profiles
./manage-services.sh wizard  # Interactive setup
./manage-services.sh preset standard  # Standard setup
```

### Method 2: Manual Deployment

```bash
# 1. Deploy core infrastructure
docker compose --profile core up -d

# 2. Deploy media services
docker compose --profile media up -d

# 3. Deploy automation
docker compose --profile automation up -d

# 4. Check status
docker compose ps
```

### Method 3: Selective Deployment

```bash
# Enable only what you need
./manage-services.sh enable media
./manage-services.sh enable downloads
./manage-services.sh enable requests

# Disable unwanted services
./manage-services.sh disable monitoring
```

### Post-Deployment Setup

1. **Access the Dashboard**
   ```
   http://localhost:3001
   ```

2. **Configure Prowlarr**
   - Add indexers (torrent/usenet sites)
   - Test connections

3. **Connect Services**
   ```bash
   # Run the integration script
   ./scripts/api-integration.sh
   ```

4. **Set Up Media Libraries**
   - In Jellyfin: Settings → Libraries → Add
   - Point to your media folders

---

## Security Hardening

### 1. Enable Authentication

```yaml
# In docker-compose.yml, add Authelia
authelia:
  image: authelia/authelia:latest
  volumes:
    - ./config/authelia:/config
  environment:
    - TZ=${TZ}
```

### 2. Configure Reverse Proxy

```bash
# Enable HTTPS with Traefik
# Edit traefik configuration
nano config/traefik/traefik.yml
```

```yaml
# traefik.yml
entryPoints:
  websecure:
    address: ":443"
certificatesResolvers:
  cloudflare:
    acme:
      email: ${CLOUDFLARE_EMAIL}
      storage: /letsencrypt/acme.json
      dnsChallenge:
        provider: cloudflare
```

### 3. VPN for Downloads

```bash
# Configure VPN in .env
USE_VPN=true
VPN_SERVICE=mullvad  # or nordvpn, expressvpn
VPN_USERNAME=your-username
VPN_PASSWORD=your-password
```

### 4. Security Checklist

- [ ] Change all default passwords
- [ ] Enable 2FA on Authelia
- [ ] Configure firewall rules
- [ ] Set up fail2ban
- [ ] Enable HTTPS only
- [ ] Regular security updates
- [ ] Monitor access logs

---

## Monitoring Setup

### 1. Enable Monitoring Stack

```bash
./manage-services.sh enable monitoring
```

### 2. Configure Alerts

```yaml
# config/prometheus/alert.rules.yml
groups:
  - name: media-server
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        annotations:
          summary: "Service {{ $labels.job }} is down"
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Uptime Kuma**: http://localhost:3011

### 4. Set Up Notifications

```bash
# Configure in .env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN=your-bot-token
SMTP_SERVER=smtp.gmail.com
```

---

## Update Management

### Safe Update Procedure

```bash
# 1. Backup current state
./scripts/backup.sh

# 2. Pull latest images
docker compose pull

# 3. Update one service at a time
docker compose up -d jellyfin
# Test, then continue...

# 4. Update all services
docker compose up -d

# 5. Verify everything works
./manage-services.sh status
```

### Automatic Updates (Watchtower)

```yaml
# Add to docker-compose.yml
watchtower:
  image: containrrr/watchtower
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  environment:
    - WATCHTOWER_SCHEDULE=0 0 4 * * *
    - WATCHTOWER_CLEANUP=true
    - WATCHTOWER_NOTIFICATIONS=email
```

### Rollback Procedure

```bash
# If update fails
docker compose down
docker compose up -d --force-recreate
```

---

## Backup & Recovery

### Automated Backup Setup

```bash
# 1. Configure backup paths in .env
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # 2 AM daily
BACKUP_RETENTION_DAYS=30
BACKUP_LOCATION=/path/to/backup/storage

# 2. Enable backup service
./manage-services.sh enable backup
```

### Manual Backup

```bash
# Backup script
./scripts/backup.sh

# What gets backed up:
# - All service configurations
# - Database dumps
# - Docker volumes
# - Environment files
```

### Recovery Procedure

```bash
# 1. Stop all services
docker compose down

# 2. Restore from backup
./scripts/restore.sh /path/to/backup.tar.gz

# 3. Restart services
docker compose up -d

# 4. Verify data integrity
docker compose ps
```

### Backup Best Practices

1. **3-2-1 Rule**: 3 copies, 2 different media, 1 offsite
2. **Test restores regularly**
3. **Encrypt sensitive backups**
4. **Monitor backup jobs**

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start

```bash
# Check logs
docker logs jellyfin
docker compose logs -f sonarr

# Common fixes:
# - Check permissions: sudo chown -R $USER:$USER ./config
# - Verify ports: sudo netstat -tulpn | grep 8096
# - Check disk space: df -h
```

#### 2. Permission Errors

```bash
# Fix ownership
sudo chown -R 1000:1000 ./config
sudo chown -R 1000:1000 ./data

# Fix permissions
find ./config -type d -exec chmod 755 {} \;
find ./config -type f -exec chmod 644 {} \;
```

#### 3. Database Connection Issues

```bash
# Reset database
docker compose down postgres
docker volume rm mediaserver_postgres_data
docker compose up -d postgres
```

#### 4. Slow Performance

```bash
# Check resource usage
docker stats

# Limit container resources
# In docker-compose.yml:
services:
  jellyfin:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

#### 5. Network Issues

```bash
# Test connectivity
docker exec jellyfin ping google.com

# Reset network
docker network prune
docker compose down
docker compose up -d
```

### Debug Mode

```bash
# Enable debug logging
DEBUG_MODE=true
ENABLE_VERBOSE_LOGGING=true

# View real-time logs
docker compose logs -f --tail=100
```

---

## Performance Tuning

### 1. Hardware Acceleration

```bash
# Intel QuickSync
JELLYFIN_HARDWARE_ACCELERATION=intel

# NVIDIA GPU
JELLYFIN_HARDWARE_ACCELERATION=nvidia
# Add to docker-compose.yml:
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

### 2. Cache Optimization

```bash
# Enable Redis caching
./manage-services.sh enable core

# Configure in services
ENABLE_CACHE=true
CACHE_TTL=3600
```

### 3. Storage Optimization

```bash
# Use separate disks
MEDIA_PATH=/mnt/media    # Large HDD
CONFIG_PATH=/mnt/ssd     # Fast SSD
TRANSCODE_PATH=/tmp      # RAM disk
```

### 4. Network Optimization

```yaml
# In docker-compose.yml
networks:
  media_network:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000  # Jumbo frames
```

### 5. Resource Limits

```yaml
# Prevent runaway containers
services:
  jellyfin:
    mem_limit: 4g
    cpus: 2
    restart: unless-stopped
```

---

## Migration Guide

### From Existing Setup

#### 1. Export Data from Old System

```bash
# Jellyfin
cp -r /old/jellyfin/config ./config/jellyfin/

# Sonarr/Radarr
# Export settings via UI: System → Backup

# Plex Migration
./scripts/plex-to-jellyfin.sh
```

#### 2. Database Migration

```bash
# PostgreSQL
pg_dump old_db > backup.sql
psql new_db < backup.sql

# SQLite (for *arr apps)
cp /old/sonarr/sonarr.db ./config/sonarr/
```

#### 3. Media Library Migration

```bash
# Preserve folder structure
rsync -avP /old/media/ /new/media/

# Update paths in applications
# Sonarr: Settings → Media Management
# Update root folders to new paths
```

### Platform-Specific Migration

#### From Unraid
1. Export Docker templates
2. Convert to docker-compose format
3. Copy appdata folders

#### From TrueNAS
1. Create dataset for Docker
2. Mount NFS shares for media
3. Import jail configurations

#### From Synology
1. Export Docker configurations
2. Copy shared folders
3. Update permissions

---

## Advanced Topics

### Multi-Server Setup

```yaml
# docker-compose.override.yml
services:
  jellyfin:
    labels:
      - "traefik.http.services.jellyfin.loadbalancer.server.port=8096"
      - "traefik.http.routers.jellyfin.rule=Host(`media.example.com`)"
```

### GPU Transcoding

```bash
# NVIDIA Setup
sudo apt install nvidia-docker2
sudo systemctl restart docker

# Verify GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Custom Scripts

```bash
# Add to scripts/ directory
- pre-start.sh    # Run before services start
- post-update.sh  # Run after updates
- health-check.sh # Custom health checks
```

---

## Support and Resources

### Getting Help

1. **Documentation**: Check service-specific wikis
   - [Jellyfin Docs](https://jellyfin.org/docs/)
   - [Servarr Wiki](https://wiki.servarr.com/)
   
2. **Community Support**:
   - Discord: [Join Server](https://discord.gg/mediaserver)
   - Reddit: r/selfhosted, r/jellyfin
   - Forums: [Community Forums](https://forum.example.com)

3. **Logs and Diagnostics**:
   ```bash
   # Generate support bundle
   ./scripts/support-bundle.sh
   ```

### Useful Commands

```bash
# View all containers
docker ps -a

# Container shell access
docker exec -it jellyfin bash

# View logs
docker logs --tail 50 jellyfin

# Restart service
docker compose restart jellyfin

# Update single service
docker compose pull jellyfin && docker compose up -d jellyfin

# Clean up
docker system prune -a
```

### Contributing

Found a bug or want to contribute?
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

---

## Conclusion

Congratulations! You now have a fully functional media server. Remember:

- **Start small**: Enable only what you need
- **Learn gradually**: Master one service before adding more
- **Stay secure**: Regular updates and backups
- **Have fun**: Enjoy your personal Netflix!

For the latest updates and features, visit:
- GitHub: https://github.com/yourusername/ultimate-media-server
- Documentation: https://docs.mediaserver.com

**Happy Streaming! 🎬🎵📚**