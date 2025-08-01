# 🎬 Production Media Server Stack 2025

**Complete, Security-Hardened Media Server with Latest Best Practices**

Created: July 27, 2025  
Version: 2025.1  
Security Standards: 2025 Compliant  

## 📋 Overview

This is a **production-ready, security-hardened media server stack** that implements the latest 2025 Docker security best practices and provides comprehensive media management across all formats.

### 🎯 What's Included

- **25+ Services** in a single, coordinated stack
- **Complete Security Layer** with Docker socket proxy, VPN, SSL/TLS
- **Full Media Coverage** - Movies, TV, Music, Audiobooks, Photos, E-books
- **Advanced Monitoring** - Prometheus, Grafana, Alertmanager
- **Automated Backups** - Daily encrypted backups with retention
- **Professional Dashboard** - Homepage with integrated management

## 🔒 Security Features (2025 Standards)

### ✅ **Critical Security Implementations**

- **Docker Socket Proxy**: Secure Docker API access using Tecnativa proxy
- **Network Segmentation**: 6 isolated networks (frontend/backend/download/monitoring/database/socket)
- **VPN Protection**: All download traffic routed through VPN (Gluetun)
- **SSL/TLS Encryption**: Automatic Let's Encrypt certificates for all services
- **Security Headers**: HTTPS redirects, XSS protection, frame denial
- **Rate Limiting**: API abuse protection on all public services
- **Resource Limits**: Prevents resource exhaustion attacks
- **Health Monitoring**: Real-time service health checks and alerts
- **Firewall Integration**: Automatic UFW/FirewallD configuration
- **Secret Management**: Secure password generation and storage

### 🛡️ **Security Hardening Applied**

- Removed deprecated Docker Compose version field
- Fixed privileged container usage (cAdvisor now uses specific capabilities)
- Replaced all `:latest` tags with specific versions for reproducibility
- Added comprehensive resource limits to prevent DoS
- Implemented proper health check timeouts and intervals
- Enhanced password generation (32-character, full character set)
- Added security headers to all web services
- Configured proper startup dependencies with health conditions

## 🎬 Media Applications

### **Core Media Server**
- **Jellyfin 10.9.8** - Open-source media server with hardware transcoding
- **Homepage 0.9.2** - Modern dashboard with service integration

### **Automation Suite (Arr Applications)**
- **Sonarr 4.0.8** - TV show management and automation
- **Radarr 5.8.3** - Movie management and automation  
- **Lidarr 2.4.3** - Music management and automation
- **Readarr 0.3.29** - E-book management and automation
- **Bazarr 1.4.3** - Subtitle management and automation
- **Prowlarr 1.21.2** - Indexer management and integration

### **Download Clients**
- **qBittorrent 4.6.5** - Torrent client (VPN protected)
- **SABnzbd 4.3.2** - Usenet client (VPN protected)

### **New Media Applications (Added for Complete Coverage)**
- **AudioBookshelf 2.12.1** - Audiobook and podcast server
- **Navidrome 0.55.0** - Music streaming server (Subsonic API)
- **Immich v1.134.0** - Modern photo management with AI
- **Calibre-Web 0.6.23** - E-book reading interface

### **Request & Discovery**
- **Overseerr 1.33.2** - Multi-platform request management

## 📊 Monitoring & Management

### **Observability Stack**
- **Prometheus v2.53.1** - Metrics collection (15-second intervals)
- **Grafana 11.1.3** - Visualization dashboards with security headers
- **Alertmanager v0.27.0** - Alert routing and notifications
- **Node Exporter v1.8.2** - System metrics collection
- **cAdvisor v0.49.1** - Container resource monitoring

### **Management Tools**
- **Portainer 2.20.3** - Container management interface
- **Traefik v3.1.4** - Reverse proxy with automatic SSL
- **Duplicati 2.0.6.3** - Automated backup solution

## 🚀 Quick Start

### **Prerequisites**
- Linux server with 50GB+ available space
- Docker and Docker Compose installed
- Domain name (for SSL certificates)
- VPN subscription (recommended)

### **One-Command Deployment**

```bash
# Clone or download the files, then:
chmod +x setup-production-2025.sh
./setup-production-2025.sh
```

The setup script will:
1. ✅ Check all prerequisites
2. ✅ Generate secure passwords
3. ✅ Configure environment variables
4. ✅ Set up monitoring configurations  
5. ✅ Configure firewall
6. ✅ Deploy all services in proper order
7. ✅ Perform health checks
8. ✅ Generate access summary

## 🔗 Service URLs

After deployment, access your services at:

### **Primary Access**
- 🏠 **Homepage Dashboard**: `https://yourdomain.com`

### **Media Services**  
- 🎬 **Jellyfin**: `https://jellyfin.yourdomain.com`
- 📚 **AudioBookshelf**: `https://audiobooks.yourdomain.com`
- 🎵 **Navidrome**: `https://music.yourdomain.com`
- 📸 **Immich Photos**: `https://photos.yourdomain.com`
- 📖 **Calibre-Web**: `https://books.yourdomain.com`

### **Management**
- 🎭 **Radarr**: `https://radarr.yourdomain.com`
- 📺 **Sonarr**: `https://sonarr.yourdomain.com`
- 🎵 **Lidarr**: `https://lidarr.yourdomain.com`
- 📚 **Readarr**: `https://readarr.yourdomain.com`
- 💬 **Bazarr**: `https://bazarr.yourdomain.com`
- 🔍 **Prowlarr**: `https://prowlarr.yourdomain.com`
- 📋 **Overseerr**: `https://requests.yourdomain.com`

### **Monitoring & Admin**
- 📊 **Grafana**: `https://grafana.yourdomain.com`
- 🎯 **Prometheus**: `https://prometheus.yourdomain.com`
- 🚨 **Alertmanager**: `https://alertmanager.yourdomain.com`
- 🐳 **Portainer**: `https://portainer.yourdomain.com`
- 🔒 **Traefik**: `https://traefik.yourdomain.com`
- 💾 **Backup**: `https://backup.yourdomain.com`

## 📁 Directory Structure

```
newmedia/
├── docker-compose-2025-fixed.yml    # Production-ready compose file
├── .env.example                     # Environment template
├── setup-production-2025.sh         # Automated deployment script
├── config/                          # Configuration files
│   ├── prometheus/                  # Monitoring configs
│   ├── grafana/                     # Dashboard configs  
│   └── alertmanager/               # Alert configs
└── DEPLOYMENT_SUMMARY_2025.md      # Generated after deployment
```

## 🛠️ Management Commands

```bash
# Check all services
docker compose -f docker-compose-2025-fixed.yml ps

# View logs for specific service
docker compose -f docker-compose-2025-fixed.yml logs jellyfin

# Restart specific service
docker compose -f docker-compose-2025-fixed.yml restart jellyfin

# Stop all services
docker compose -f docker-compose-2025-fixed.yml stop

# Update all services to latest versions
docker compose -f docker-compose-2025-fixed.yml pull
docker compose -f docker-compose-2025-fixed.yml up -d

# Backup media data
docker run --rm -v media_data:/data -v $(pwd):/backup alpine tar czf /backup/media_backup_$(date +%Y%m%d).tar.gz /data
```

## 🔐 Security Considerations

### **Password Management**
- All passwords are auto-generated with 32-character complexity
- Stored in `.generated_passwords.txt` (delete after secure storage)
- Traefik uses bcrypt hashed authentication

### **Network Security**
- 6 isolated Docker networks with minimal cross-communication
- Download clients isolated behind VPN
- Internal networks have no internet access
- Firewall configured for minimal attack surface

### **Container Security**
- All containers run with security-hardened configurations
- Dropped all capabilities, only adding necessary ones
- AppArmor profiles applied where available
- No privileged containers (cAdvisor uses specific capabilities)

### **Data Protection**
- Named volumes prevent host filesystem exposure
- Read-only mounts where appropriate
- Automated encrypted backups with 30-day retention
- Database integrity checking enabled

## 📊 Monitoring & Alerts

### **System Monitoring**
- CPU, Memory, Disk usage tracking
- Container health and resource monitoring
- Network traffic analysis
- Service availability monitoring

### **Alert Thresholds**
- CPU usage > 80% for 5 minutes
- Memory usage > 90% for 5 minutes  
- Disk space < 10%
- Container down for 2 minutes
- SSL certificate expiry < 30 days

### **Grafana Dashboards**
- Node Exporter Full Dashboard (ID: 1860)
- Container monitoring dashboards
- Media server specific metrics
- Custom application dashboards

## 💾 Backup Strategy

### **Automated Backups**
- **Schedule**: Daily at 2 AM
- **Retention**: 30 days
- **Encryption**: AES-256
- **Coverage**: All configuration and data volumes
- **Verification**: Integrity checks on restore

### **Manual Backup Commands**
```bash
# Backup all volumes
./backup-all-volumes.sh

# Backup specific application
docker run --rm -v jellyfin_config:/data -v $(pwd):/backup alpine tar czf /backup/jellyfin_backup.tar.gz /data

# Restore from backup
docker run --rm -v jellyfin_config:/data -v $(pwd):/backup alpine tar xzf /backup/jellyfin_backup.tar.gz -C /
```

## 🔧 Configuration

### **Environment Variables**
Copy `.env.example` to `.env` and configure:

```bash
# Domain configuration
DOMAIN=yourdomain.com
ACME_EMAIL=admin@yourdomain.com

# VPN configuration (required for secure downloading)
VPN_PROVIDER=nordvpn
VPN_USER=your_username
VPN_PASSWORD=your_password

# Generated automatically by setup script
IMMICH_DB_PASSWORD=auto_generated
GRAFANA_PASSWORD=auto_generated
TRAEFIK_AUTH=auto_generated_hash
```

### **Media Library Setup**
1. **Jellyfin**: Point libraries to `/media/movies`, `/media/tv`, `/media/music`
2. **Arr Apps**: Configure download client connections and quality profiles
3. **Indexers**: Add indexers in Prowlarr, sync to arr applications
4. **Requests**: Connect Overseerr to Jellyfin, Sonarr, and Radarr

## 🚨 Troubleshooting

### **Common Issues**

**Services won't start:**
```bash
# Check logs
docker compose -f docker-compose-2025-fixed.yml logs

# Check health status
docker compose -f docker-compose-2025-fixed.yml ps
```

**SSL certificate issues:**
```bash
# Check Traefik logs
docker logs traefik

# Verify domain DNS
dig yourdomain.com
```

**VPN connection problems:**
```bash
# Check Gluetun logs
docker logs gluetun

# Test VPN connection
docker exec gluetun curl https://ipinfo.io
```

**Permission issues:**
```bash
# Fix ownership
sudo chown -R 1000:1000 ./config ./media-data
```

### **Performance Optimization**

**High resource usage:**
- Reduce quality profiles in arr applications
- Enable hardware transcoding in Jellyfin
- Increase monitoring intervals
- Optimize database settings

**Slow web interfaces:**
- Check reverse proxy configuration
- Verify SSL certificate status
- Monitor network latency
- Review resource allocation

## 📱 Mobile Apps

### **Recommended Mobile Applications**

- **Jellyfin**: Official apps for iOS/Android
- **AudioBookshelf**: Official apps for iOS/Android with sync
- **Navidrome**: DSub (Android), play:Sub (iOS), Ultrasonic
- **Immich**: Official apps with automatic photo backup
- **Overseerr**: Progressive web app works great on mobile

## 🔄 Updates & Maintenance

### **Monthly Updates**
```bash
# Update all container images
docker compose -f docker-compose-2025-fixed.yml pull

# Restart with new images
docker compose -f docker-compose-2025-fixed.yml up -d

# Clean up old images
docker image prune -a
```

### **Security Updates**
- Monitor security advisories for used applications
- Update base images monthly
- Review and update VPN configurations
- Rotate passwords quarterly
- Review access logs monthly

## 📄 License & Support

This configuration is provided as-is for educational and personal use. 

### **Getting Help**
- Check service-specific documentation
- Review container logs for errors
- Consult application GitHub repositories
- Use community forums and Discord servers

---

**🎬 Your complete, secure, production-ready media server awaits!**

*Production Media Server Stack 2025 - Security First, Media Complete*