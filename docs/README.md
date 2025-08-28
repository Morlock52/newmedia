# 📚 Ultimate Media Server Stack - Documentation Hub

Welcome to the comprehensive documentation for the Ultimate Media Server Stack. This documentation is organized to help you get started quickly and dive deep into specific areas as needed.

## 🗂️ Documentation Structure

### 🚀 Getting Started

- [Quick Start Guide](../README.md#-quick-start) - Get up and running in minutes
- [Prerequisites](deployment/prerequisites.md) - System requirements and setup
- [Environment Setup](configuration/environment.md) - Configuration

### 📋 Services & Features

- [📋 Service Catalog](services/README.md) - Complete list of all 30 services
- [🎬 Media Servers](features/media-servers.md) - Jellyfin, Plex, Emby
- [⬇️ Download Management](features/download-management.md) - *ARR suite and clients
- [🛡️ Security & VPN](features/security-vpn.md) - VPN and network security
- [📊 Monitoring](features/monitoring.md) - System monitoring and analytics
- [🏠 Dashboards](features/dashboards.md) - Web interfaces and UIs
- [📚 Content Libraries](features/content-libraries.md) - Books, music, photos
- [🔧 Utilities](features/utilities.md) - Self-hosting tools

### 🔧 Configuration & Setup

- [🌍 Environment Configuration](configuration/environment.md) - .env setup
- [🐳 Docker Configuration](configuration/docker.md) - Container settings
- [🌐 Network Configuration](configuration/networking.md) - Network setup
- [💾 Storage Configuration](configuration/storage.md) - Volume management
- [🔐 Security Configuration](configuration/security.md) - Security settings

### 🚀 Deployment

- [📦 Docker Compose Deployment](deployment/docker-compose.md) - Standard deployment
- [🏗️ Single Container Deployment](deployment/single-container.md) - All-in-one container
- [☁️ Cloud Deployment](deployment/cloud.md) - VPS and cloud platforms
- [🏠 Local Deployment](deployment/local.md) - Home server setup
- [📊 Production Deployment](deployment/production.md) - Production considerations

### 📊 Monitoring & Observability

- [📈 Prometheus Setup](monitoring/prometheus.md) - Metrics collection
- [📊 Grafana Dashboards](monitoring/grafana.md) - Visualization setup
- [📋 Log Management](monitoring/logging.md) - Loki and log aggregation
- [⏰ Uptime Monitoring](monitoring/uptime.md) - Service health checks
- [🚨 Alerting](monitoring/alerting.md) - Alert configuration

### 🛠️ Operations

- [🔧 Troubleshooting](operations/troubleshooting.md) - Common issues and solutions
- [📈 Performance Tuning](operations/performance.md) - Optimization guide
- [💾 Backup & Recovery](operations/backup-recovery.md) - Data protection
- [🔄 Updates & Maintenance](operations/maintenance.md) - Keeping services updated
- [🔒 Security Operations](operations/security.md) - Security best practices

### 👨‍💻 Development

- [🏗️ Development Setup](development/README.md) - Local development environment
- [🧪 Testing](development/testing.md) - Testing procedures
- [🤝 Contributing](development/contributing.md) - How to contribute
- [📚 API Documentation](development/api.md) - Service APIs

## 📊 Service Reference

### Core Services (Always Enabled)

| Service | Port | Documentation | Configuration |
|---------|------|---------------|---------------|
| Jellyfin | 8096 | [Guide](features/media-servers.md#jellyfin) | [Config](services/jellyfin.md) |
| Plex | 32400 | [Guide](features/media-servers.md#plex) | [Config](services/plex.md) |
| Sonarr | 8989 | [Guide](features/download-management.md#sonarr) | [Config](services/sonarr.md) |
| Radarr | 7878 | [Guide](features/download-management.md#radarr) | [Config](services/radarr.md) |
| Prowlarr | 9696 | [Guide](features/download-management.md#prowlarr) | [Config](services/prowlarr.md) |
| qBittorrent | 8080 | [Guide](features/download-management.md#qbittorrent) | [Config](services/qbittorrent.md) |
| Gluetun | 8888 | [Guide](features/security-vpn.md#gluetun) | [Config](services/gluetun.md) |
| Prometheus | 9090 | [Guide](features/monitoring.md#prometheus) | [Config](services/prometheus.md) |
| Grafana | 3000 | [Guide](features/monitoring.md#grafana) | [Config](services/grafana.md) |
| Uptime Kuma | 3001 | [Guide](features/monitoring.md#uptime-kuma) | [Config](services/uptime-kuma.md) |
| Portainer | 9000 | [Guide](features/dashboards.md#portainer) | [Config](services/portainer.md) |
| Nginx Proxy Manager | 81 | [Guide](features/dashboards.md#nginx-proxy-manager) | [Config](services/nginx-proxy-manager.md) |
| Homarr | 7575 | [Guide](features/dashboards.md#homarr) | [Config](services/homarr.md) |
| PostgreSQL | 5432 | [Guide](services/postgresql.md) | [Config](services/postgresql.md) |
| MariaDB | 3306 | [Guide](services/mariadb.md) | [Config](services/mariadb.md) |
| Redis | 6379 | [Guide](services/redis.md) | [Config](services/redis.md) |

### Optional Services

| Service | Port | Documentation | Use Case |
|---------|------|---------------|----------|
| Emby | 8097 | [Guide](features/media-servers.md#emby) | Alternative media server |
| Lidarr | 8686 | [Guide](features/download-management.md#lidarr) | Music management |
| Readarr | 8787 | [Guide](features/download-management.md#readarr) | Book management |
| Bazarr | 6767 | [Guide](features/download-management.md#bazarr) | Subtitle management |
| Jellyseerr | 5055 | [Guide](features/download-management.md#jellyseerr) | Media requests |
| Overseerr | 5056 | [Guide](features/download-management.md#overseerr) | Media requests |
| PhotoPrism | 2342 | [Guide](features/content-libraries.md#photoprism) | Photo management |
| Nextcloud | 8084 | [Guide](features/utilities.md#nextcloud) | Personal cloud |
| Vaultwarden | 8085 | [Guide](features/utilities.md#vaultwarden) | Password manager |

## 🎯 Quick Navigation

### I want to

- **🚀 Get started quickly**: [Quick Start Guide](../README.md#-quick-start)
- **🎬 Stream media**: [Media Servers Guide](features/media-servers.md)
- **⬇️ Download content automatically**: [Download Management](features/download-management.md)
- **🛡️ Secure my setup**: [Security & VPN Guide](features/security-vpn.md)
- **📊 Monitor my system**: [Monitoring Setup](features/monitoring.md)
- **🏠 Set up a dashboard**: [Dashboard Guide](features/dashboards.md)
- **🔧 Troubleshoot issues**: [Troubleshooting Guide](operations/troubleshooting.md)
- **📈 Optimize performance**: [Performance Tuning](operations/performance.md)

### I'm looking for

- **📋 Complete service list**: [Service Catalog](services/README.md)
- **🔧 Configuration examples**: [Configuration Guide](configuration/README.md)
- **🚀 Deployment options**: [Deployment Guide](deployment/README.md)
- **📚 API documentation**: [API Guide](development/api.md)
- **🤝 How to contribute**: [Contributing Guide](development/contributing.md)

## 📱 Mobile-Friendly Access

All documentation is optimized for mobile viewing. You can access and configure your media server from any device.

## 🔄 Keep Documentation Updated

This documentation is continuously updated. For the latest version:

- Check the [GitHub repository](https://github.com/yourusername/media-server)
- Follow the [changelog](CHANGELOG.md)
- Join our [community discussions](https://github.com/yourusername/media-server/discussions)

## 🆘 Get Help

If you can't find what you're looking for:

1. Check the [Troubleshooting Guide](operations/troubleshooting.md)
2. Search the [GitHub Issues](https://github.com/yourusername/media-server/issues)
3. Join our [Discord Community](https://discord.gg/mediaserver)
4. Create a [new issue](https://github.com/yourusername/media-server/issues/new)

---

**Happy self-hosting!** 🏠
