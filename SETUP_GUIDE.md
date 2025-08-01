# 🎬 NewMedia Setup Guide

Welcome to your comprehensive media server stack! This guide will help you get everything running on your system.

## 📋 Current Status

✅ **Project Structure**: Complete and organized  
✅ **Configuration Files**: Ready with proper settings  
✅ **Secrets**: Generated and configured  
✅ **Docker Compose**: Full stack definition ready  
⚠️ **Docker**: Not currently installed  

## 🚀 Quick Start Options

### Option 1: Docker Desktop Installation (Recommended)

1. **Download Docker Desktop for Mac**:
   ```bash
   # Visit: https://www.docker.com/products/docker-desktop/
   # Or install via Homebrew (if available):
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install --cask docker
   ```

2. **Start Docker Desktop** and ensure it's running

3. **Deploy the stack**:
   ```bash
   cd /Users/morlock/fun/newmedia/media-server-stack
   ./scripts/deploy.sh
   ```

### Option 2: Homebrew + Docker Installation

```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker
brew install docker docker-compose

# Start Docker service (may require Docker Desktop)
open -a Docker

# Deploy the stack
cd /Users/morlock/fun/newmedia/media-server-stack
./scripts/deploy.sh
```

### Option 3: Direct Container Runtime

If you prefer alternatives to Docker Desktop:

```bash
# Install Lima + Docker
brew install lima docker
limactl start template://docker
export DOCKER_HOST=$(limactl list docker --format 'unix://{{.Dir}}/sock/docker.sock')

# Or use Colima
brew install colima docker docker-compose
colima start

# Deploy the stack
cd /Users/morlock/fun/newmedia/media-server-stack
./scripts/deploy.sh
```

## 🎯 What You Get

Once running, you'll have access to:

### 🎬 Media Services
- **Jellyfin**: Your personal Netflix-like media server
- **Sonarr**: Automatic TV show management
- **Radarr**: Automatic movie management
- **Lidarr**: Music collection management
- **Readarr**: Book and audiobook management

### 🔍 Content Discovery
- **Prowlarr**: Indexer management for finding content
- **Overseerr**: User-friendly request management
- **Bazarr**: Automatic subtitle downloads

### 📊 Management & Monitoring
- **Homarr**: Beautiful dashboard for all services
- **Traefik**: Reverse proxy with SSL
- **Grafana**: Monitoring and analytics
- **Prometheus**: Metrics collection

### 🔒 Security & Privacy
- **Gluetun**: VPN integration with kill-switch
- **Authelia**: Two-factor authentication
- **Cloudflare Tunnel**: Secure external access

## 🌐 Access Your Services

After deployment, access services at:

- **Main Dashboard**: http://localhost (Homarr)
- **Jellyfin**: http://localhost:8096
- **Web UI Manager**: http://localhost:3000
- **Sonarr**: http://sonarr.localhost
- **Radarr**: http://radarr.localhost
- **Prowlarr**: http://prowlarr.localhost
- **Overseerr**: http://overseerr.localhost

## ⚙️ Pre-configured Features

### VPN Configuration
- **Provider**: Private Internet Access (PIA)
- **Type**: WireGuard
- **Region**: US East
- **Port Forwarding**: Enabled
- **Kill Switch**: Active

### Storage Structure
```
/Users/morlock/fun/newmedia/
├── data/
│   ├── media/          # Your organized media library
│   │   ├── movies/
│   │   ├── tv/
│   │   ├── music/
│   │   └── online-videos/
│   ├── torrents/       # Download staging area
│   └── usenet/         # Usenet downloads
├── config/             # Service configurations
└── secrets/            # Encrypted credentials
```

### Security Features
- Non-root container execution
- Network isolation between services
- Automatic SSL certificate management
- VPN kill-switch protection
- Docker secrets for sensitive data

## 🔧 Management Commands

### Stack Management
```bash
# Deploy complete stack
./scripts/deploy.sh

# Check service health
./scripts/health-check.sh

# View logs
docker-compose logs [service-name]

# Restart services
docker-compose restart [service-name]

# Stop everything
docker-compose down
```

### Backup & Recovery
```bash
# Create backup
./scripts/backup.sh backup

# List backups
./scripts/backup.sh list

# Restore from backup
./scripts/backup.sh restore [date]
```

### Security Management
```bash
# Setup security
./scripts/setup-security.sh setup

# Security audit
./scripts/setup-security.sh audit

# Verify settings
./scripts/setup-security.sh verify
```

## 🎨 Web UI Features

The built-in Web UI at `http://localhost:3000` provides:

### Setup Tab
- Environment configuration wizard
- Real-time validation
- Auto-detection of system settings
- VPN provider selection
- Storage path configuration

### Management Tab
- Stack control (start/stop/restart)
- Service status monitoring
- Updates management
- Individual service control
- Resource monitoring

### Monitoring Tab
- System statistics
- Service health checks
- Performance metrics
- Service links

### Logs Tab
- Real-time log viewing
- Service-specific logs
- Download logs
- Search and filter

## 🔍 Troubleshooting

### Common Issues

**Docker not running**:
```bash
# Check Docker status
docker --version
docker info

# Start Docker Desktop
open -a Docker
```

**Services won't start**:
```bash
# Check logs
docker-compose logs [service-name]

# Restart problematic service
docker-compose restart [service-name]

# Full restart
docker-compose down && docker-compose up -d
```

**VPN connection issues**:
```bash
# Check VPN status
docker exec gluetun cat /tmp/gluetun/ip

# View VPN logs
docker logs gluetun --tail 100

# Restart VPN
docker restart gluetun
```

**Permission issues**:
```bash
# Fix permissions
sudo chown -R $USER:$USER /Users/morlock/fun/newmedia/
chmod -R 755 /Users/morlock/fun/newmedia/data
chmod -R 755 /Users/morlock/fun/newmedia/config
```

## 📱 Mobile Access

Your services are configured for secure external access via Cloudflare Tunnel:

- Set up your domain in Cloudflare
- Configure DNS to point to your tunnel
- Access from anywhere securely

## 🔐 Security Notes

- All secrets are pre-generated and stored securely
- VPN kill-switch prevents IP leaks
- Services run as unprivileged users
- Network segmentation isolates services
- SSL certificates auto-renew

## 📚 Additional Resources

- **Project Documentation**: See `README.md` in media-server-stack/
- **Security Guide**: `SECURITY.md`
- **Implementation Plan**: `PLAN.md`
- **Claude Assistant**: `.claude/CLAUDE.md`

## 🤝 Next Steps

1. **Install Docker** using one of the methods above
2. **Run deployment**: `./scripts/deploy.sh`
3. **Access Web UI**: http://localhost:3000
4. **Configure services** through the dashboard
5. **Start adding media** to your library

Your media server stack is ready to transform your entertainment experience! 🎉
