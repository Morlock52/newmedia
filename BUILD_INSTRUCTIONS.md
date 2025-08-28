# Ultimate Media Server - Build Instructions

## 🚀 Single Container Build & Deployment

This document provides step-by-step instructions for building and deploying the Ultimate Media Server single container solution.

## 📁 Files Created

### Core Files
1. **`Dockerfile.ultimate-single`** - Main container definition with all services
2. **`docker-compose.ultimate-single.yml`** - Production-ready compose configuration  
3. **`deploy-ultimate-single.sh`** - Automated deployment and management script
4. **`README-ULTIMATE-SINGLE.md`** - Comprehensive documentation

### Key Features Included
- **All Media Services**: Jellyfin, Sonarr, Radarr, Lidarr, Prowlarr, qBittorrent
- **DevOps Automation**: Monitoring, backup, security, health checks
- **AI Integration**: MCP Suite for intelligent media management
- **Performance Optimization**: Multi-stage builds, caching, resource management
- **Security**: Built-in fail2ban, security scanning, encrypted storage

## 🔨 Build Process

### 1. Quick Build & Deploy
```bash
# Make deployment script executable
chmod +x deploy-ultimate-single.sh

# Complete automated installation
./deploy-ultimate-single.sh install
```

### 2. Manual Build Steps
```bash
# Build container image
docker build -f Dockerfile.ultimate-single -t ultimate-media-server:latest .

# Deploy with docker-compose
docker-compose -f docker-compose.ultimate-single.yml up -d
```

### 3. Custom Configuration Build
```bash
# Generate environment file first
./deploy-ultimate-single.sh generate-env

# Edit configuration
nano .env.ultimate

# Build and deploy
./deploy-ultimate-single.sh build
./deploy-ultimate-single.sh deploy
```

## 🏗️ Container Architecture

### Multi-Stage Build Process
```dockerfile
Stage 1: Base Builder - Common dependencies and tools
Stage 2: S6-Overlay - Process management system
Stage 3: Caddy - Reverse proxy server
Stage 4: *arr Apps - Media management applications
Stage 5: Monitoring - Prometheus and Grafana
Stage 6: Dashboard - Node.js based dashboard with AI integration
Stage 7: Final - Assembled optimized container
```

### Service Management with s6-overlay
- **Process Supervision**: Automatic service restart on failure
- **Dependency Management**: Proper service startup ordering
- **Signal Handling**: Graceful shutdown and restart
- **Health Monitoring**: Built-in health check system

## 🔧 Configuration Options

### Environment Variables
```bash
# System Settings
PUID=1000                    # User ID for file permissions
PGID=1000                    # Group ID for file permissions  
TZ=UTC                       # Timezone
DOMAIN=localhost             # Domain name for reverse proxy

# Port Configuration
WEB_PORT=80                  # Main web interface port
JELLYFIN_PORT=8096          # Jellyfin media server
SONARR_PORT=8989            # TV show management
RADARR_PORT=7878            # Movie management
LIDARR_PORT=8686            # Music management
PROWLARR_PORT=9696          # Indexer management
QBITTORRENT_PORT=8080       # Download client

# Performance Tuning
MEMORY_LIMIT=4G             # Container memory limit
CPU_LIMIT=2.0               # Container CPU limit
JELLYFIN_CACHE_SIZE=256     # Jellyfin cache size (MB)

# Feature Toggles  
HARDWARE_ACCELERATION=true   # Enable GPU acceleration
VPN_ENABLED=false           # Enable VPN integration
SECURITY_SCAN_ENABLED=true  # Enable security scanning
AUTO_UPDATE=true            # Enable automatic updates

# Backup Configuration
BACKUP_RETENTION_DAYS=7     # Backup retention period
BACKUP_SCHEDULE="0 */6 * * *" # Backup schedule (cron format)
```

### Volume Mappings
```yaml
# Configuration persistence
./ultimate-config:/config

# Media storage
./media:/data/media

# Downloads
./downloads:/data/downloads

# Backups
./backups:/data/backup

# Hardware acceleration (optional)
/dev/dri:/dev/dri
```

## 🚀 Deployment Scenarios

### 1. Development Environment
```bash
# Quick development setup
./deploy-ultimate-single.sh install

# Access services
echo "Dashboard: http://localhost/"
echo "Jellyfin: http://localhost:8096"
```

### 2. Production Deployment
```bash
# Set production environment variables
export DOMAIN=yourmediaserver.com
export GRAFANA_PASSWORD=$(openssl rand -base64 16)
export POSTGRES_PASSWORD=$(openssl rand -base64 16)

# Deploy with SSL and monitoring
docker-compose --profile ssl --profile external-db up -d
```

### 3. High-Performance Setup
```bash
# Configure for high performance
cat > .env.ultimate <<EOF
MEMORY_LIMIT=8G
CPU_LIMIT=4.0
HARDWARE_ACCELERATION=true
JELLYFIN_CACHE_SIZE=1024
EOF

# Deploy with performance optimization
./deploy-ultimate-single.sh deploy
```

## 📊 Monitoring & Management

### Built-in Monitoring
- **Prometheus**: Metrics collection on port 9090
- **Grafana**: Visualization dashboard on port 3001
- **Health Checks**: Comprehensive service monitoring
- **Log Aggregation**: Centralized logging system

### Management Commands
```bash
# Service status
./deploy-ultimate-single.sh status

# View logs
./deploy-ultimate-single.sh logs [service_name]

# Create backup
./deploy-ultimate-single.sh backup

# Update services
./deploy-ultimate-single.sh update

# Health check
docker exec ultimate-media-server /usr/local/bin/healthcheck
```

## 🛡️ Security Features

### Built-in Security
- **Fail2ban**: Automatic IP blocking
- **Security Scanning**: Regular vulnerability checks  
- **Encrypted Storage**: Configuration encryption
- **Network Isolation**: Container-based isolation
- **Access Controls**: Role-based authentication

### Security Configuration
```bash
# Enable all security features
SECURITY_SCAN_ENABLED=true
FAIL2BAN_ENABLED=true

# Configure email alerts
SMTP_SERVER=smtp.gmail.com
EMAIL_TO=security@yourdomain.com
```

## 🔄 Backup & Recovery

### Automated Backups
- **Schedule**: Every 6 hours by default
- **Content**: All service configurations
- **Retention**: 7 days (configurable)
- **Compression**: Gzipped for efficiency

### Backup Commands
```bash
# Manual backup
./deploy-ultimate-single.sh backup

# Restore from backup
./deploy-ultimate-single.sh restore backup_file.tar.gz

# List backups
ls -la backups/
```

## 🤖 AI Integration

### MCP Suite Features
- **Smart Media Organization**: Automatic metadata enhancement
- **Download Optimization**: Intelligent quality selection
- **Issue Resolution**: Automated troubleshooting
- **Performance Insights**: AI-driven recommendations

### AI Configuration
```bash
# Enable AI features
MCP_SUITE_ENABLED=true
AI_ASSISTANT_ENABLED=true

# Access AI dashboard
echo "AI Assistant: http://localhost/mcp"
```

## 🔍 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check Docker version
docker --version
docker-compose --version

# Clean build cache
docker builder prune -af

# Rebuild from scratch
docker build --no-cache -f Dockerfile.ultimate-single -t ultimate-media-server:latest .
```

#### Service Issues
```bash
# Check service logs
./deploy-ultimate-single.sh logs

# Check container status
docker ps -a

# Check health status
./deploy-ultimate-single.sh status
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $(id -u):$(id -g) ultimate-config/ media/ downloads/

# Check user/group settings
echo "PUID=$(id -u)" >> .env.ultimate
echo "PGID=$(id -g)" >> .env.ultimate
```

## 🚀 Performance Optimization

### Resource Tuning
```bash
# High-performance configuration
MEMORY_LIMIT=8G              # 8GB RAM limit
CPU_LIMIT=4.0                # 4 CPU cores
JELLYFIN_CACHE_SIZE=1024     # 1GB cache

# Low-resource configuration  
MEMORY_LIMIT=2G              # 2GB RAM limit
CPU_LIMIT=1.0                # 1 CPU core
JELLYFIN_CACHE_SIZE=128      # 128MB cache
```

### Hardware Acceleration
```bash
# Intel Quick Sync
docker run --device=/dev/dri:/dev/dri ...

# NVIDIA GPU
# Uncomment GPU sections in docker-compose file
runtime: nvidia
```

## 📚 Additional Resources

### Documentation Files
- **README-ULTIMATE-SINGLE.md** - Complete user guide
- **docker-compose.ultimate-single.yml** - Production configuration
- **deploy-ultimate-single.sh** - Management script
- **.env.ultimate.example** - Configuration template

### Support Resources
- Health check endpoint: `http://localhost/health`
- Metrics endpoint: `http://localhost/metrics`
- API documentation: `http://localhost/api/docs`

### Build Artifacts
After successful build, you'll have:
- Container image: `ultimate-media-server:latest`
- Configuration directory: `./ultimate-config/`
- Media directories: `./media/`, `./downloads/`
- Backup storage: `./backups/`

## ✅ Validation Checklist

After deployment, verify:
- [ ] All services accessible via main dashboard (http://localhost/)
- [ ] Jellyfin media server running (http://localhost:8096)
- [ ] *arr applications configured (Sonarr, Radarr, etc.)
- [ ] Download client operational (qBittorrent)
- [ ] Monitoring active (Grafana dashboard)
- [ ] Health checks passing
- [ ] Backup system operational
- [ ] AI assistant accessible

## 🎯 Next Steps

1. **Configure Media Libraries**: Set up your movie, TV, and music libraries in Jellyfin
2. **Setup Indexers**: Configure Prowlarr with your preferred indexers
3. **Download Client**: Configure qBittorrent with download categories
4. **Monitoring**: Set up Grafana dashboards and alerts
5. **Security**: Change default passwords and enable 2FA
6. **Backup**: Test backup and restore procedures

## 🏁 Success Indicators

Your Ultimate Media Server is successfully deployed when:
- Dashboard shows all services as "healthy" 
- You can stream media through Jellyfin
- *arr applications can search and download content
- Monitoring dashboards display system metrics
- Automated backups are creating successfully
- All service URLs are accessible through the reverse proxy

---

**🎬 Happy Media Serving!**

The Ultimate Media Server single container provides enterprise-grade media server capabilities with full DevOps automation in one easy-to-deploy package. Enjoy your automated, monitored, and AI-enhanced media experience!