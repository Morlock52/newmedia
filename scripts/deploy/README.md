# Media Server Deployment Scripts

Clean, modular bash scripts for deploying and managing your media server on macOS.

## 🚀 Quick Start

For new users, run the one-command deployment:

```bash
./scripts/deploy/quick-start.sh
```

This will:
1. Set up directory structure and configuration
2. Deploy core services (Jellyfin, qBittorrent, Homepage)  
3. Run a health check to verify everything is working

## 📂 Script Overview

### Core Scripts

- **`setup.sh`** - Initial setup and configuration
  - Creates directory structure
  - Generates environment configuration
  - Sets up service configurations
  - Creates macOS-specific overrides

- **`deploy.sh`** - Service deployment and orchestration
  - Supports deployment modes: `core`, `media`, `full`
  - Handles image pulling and container creation
  - Post-deployment configuration
  - Service health verification

- **`service-control.sh`** - Service management utility
  - Start/stop/restart services
  - View logs and status
  - Update services
  - Execute commands in containers

### Maintenance Scripts

- **`health-check.sh`** - System and service health monitoring
  - Checks Docker status
  - Monitors system resources (CPU, memory, disk)
  - Verifies service endpoints
  - Analyzes logs for errors
  - Generates health reports

- **`backup.sh`** - Automated backup system
  - Backs up configurations and databases
  - Supports compressed archives
  - Automatic cleanup of old backups
  - Metadata generation

- **`restore.sh`** - Restore from backups
  - Interactive backup selection
  - Full configuration restore
  - Database restoration
  - Permission management

## 🎯 Usage Examples

### Basic Operations

```bash
# Initial setup (run once)
./scripts/deploy/setup.sh

# Deploy all services
./scripts/deploy/deploy.sh

# Deploy only core services
./scripts/deploy/deploy.sh core

# Check system health
./scripts/deploy/health-check.sh

# View service status
./scripts/deploy/service-control.sh status

# View logs for a specific service
./scripts/deploy/service-control.sh logs jellyfin

# Stop all services
./scripts/deploy/service-control.sh stop

# Update all services
./scripts/deploy/service-control.sh update
```

### Advanced Operations

```bash
# Force recreate containers
./scripts/deploy/deploy.sh full --force

# Deploy without pulling images
./scripts/deploy/deploy.sh --no-pull

# Execute command in container
./scripts/deploy/service-control.sh exec qbittorrent sh

# Reset a service
./scripts/deploy/service-control.sh reset sonarr

# Create backup
./scripts/deploy/backup.sh

# Restore from backup
./scripts/deploy/restore.sh
```

## 🏗️ Directory Structure

After running setup, your directory structure will be:

```
newmedia/
├── config/           # Service configurations
│   ├── jellyfin/
│   ├── radarr/
│   ├── sonarr/
│   └── ...
├── data/            # Media and downloads
│   ├── media/
│   │   ├── movies/
│   │   ├── tv/
│   │   └── music/
│   └── downloads/
├── logs/            # Application logs
├── backups/         # Backup archives
└── scripts/
    └── deploy/      # Deployment scripts
```

## 🔧 Configuration

### Environment Variables

The `.env` file contains:
- System settings (timezone, user/group IDs)
- Directory paths
- Service ports
- API keys (auto-populated after first run)

### Service Groups

- **Core**: Jellyfin, qBittorrent, Homepage
- **Media**: Radarr, Sonarr, Prowlarr, Bazarr
- **Utility**: Overseerr, Tautulli, Portainer

## 🍎 macOS Optimizations

The scripts include macOS-specific optimizations:
- Hardware acceleration disabled (not supported in Docker)
- Optimized volume mounts with `delegated` flag
- macOS-compatible permission handling
- Resource usage monitoring adapted for macOS

## 🔍 Health Monitoring

The health check monitors:
- Docker daemon status
- Container health
- Service endpoint availability
- System resources (CPU, memory, disk)
- Recent errors in logs

Run with JSON output:
```bash
./scripts/deploy/health-check.sh --json
```

## 💾 Backup & Restore

### Creating Backups

```bash
# Manual backup
./scripts/deploy/backup.sh

# The script will:
# - Optionally stop services for consistency
# - Backup configurations
# - Backup databases
# - Create compressed archive
# - Clean up old backups (keeps last 7 by default)
```

### Restoring from Backup

```bash
# Interactive restore
./scripts/deploy/restore.sh

# Restore specific backup
./scripts/deploy/restore.sh 20240731_120000
```

## 🚨 Troubleshooting

### Common Issues

1. **Docker not running**
   ```bash
   # Start Docker Desktop on macOS
   open -a Docker
   ```

2. **Port conflicts**
   ```bash
   # Check what's using a port
   lsof -i :8096
   ```

3. **Permission issues**
   ```bash
   # Fix permissions
   ./scripts/deploy/setup.sh
   ```

4. **Service not responding**
   ```bash
   # Check service logs
   ./scripts/deploy/service-control.sh logs [service]
   
   # Restart service
   ./scripts/deploy/service-control.sh restart [service]
   ```

## 🛡️ Security Recommendations

1. Change default passwords immediately after deployment
2. Use strong, unique passwords for each service
3. Keep services updated regularly
4. Enable authentication on all public-facing services
5. Use VPN for remote access
6. Regular backups are essential

## 📚 Additional Resources

- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [Servarr Wiki](https://wiki.servarr.com/)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Contributing

Feel free to submit issues or pull requests to improve these scripts!