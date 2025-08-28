# Media Server Infrastructure Stack

A comprehensive monitoring, security, and networking infrastructure for your ultimate media server. This infrastructure provides enterprise-grade monitoring, 2FA authentication, VPN protection, GPU transcoding, and automated recovery capabilities.

## 🏗️ Infrastructure Components

### Core Infrastructure
- **Traefik**: Reverse proxy with automatic HTTPS and load balancing
- **Authelia**: 2FA authentication with TOTP and optional DUO push
- **Gluetun**: VPN container for secure downloads with kill-switch
- **Redis**: Session storage and caching
- **PostgreSQL**: Primary database for services

### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards and analytics
- **Loki**: Log aggregation and search
- **Promtail**: Log shipping agent
- **Alertmanager**: Alert routing and notifications
- **Uptime Kuma**: Service availability monitoring
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics
- **Blackbox Exporter**: External endpoint monitoring

### Media Processing
- **FileFlows**: GPU-accelerated media processing
- **Tdarr**: Alternative transcoding service
- **Hardware acceleration**: NVIDIA GPU and Intel Quick Sync support

### Security & Recovery
- **Fail2ban**: Intrusion prevention
- **Webhook Manager**: Automated response system
- **Critical Alert Handler**: Automated recovery actions
- **Duplicati**: Automated backups
- **SSL/TLS**: Automatic certificate management

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Clone or navigate to your media server directory
cd /path/to/your/media-server
```

### 2. Configuration
```bash
# Copy and edit the environment template
cp .env.template .env
nano .env  # Edit with your configuration
```

### 3. Deploy Infrastructure
```bash
# Make startup script executable
chmod +x start-infrastructure.sh

# Deploy the complete infrastructure
./start-infrastructure.sh start
```

### 4. Access Services
After deployment, access your services at:
- **Traefik Dashboard**: `http://traefik.yourdomain.com:8080`
- **Grafana**: `http://grafana.yourdomain.com` (admin/your-password)
- **Prometheus**: `http://prometheus.yourdomain.com`
- **Uptime Kuma**: `http://uptime.yourdomain.com`
- **Authelia**: `http://auth.yourdomain.com`

## 📊 Monitoring Features

### Dashboards
- **Media Server Overview**: System resources, service status, performance
- **Infrastructure Monitoring**: Network, security, VPN status
- **Application Metrics**: *arr services, download clients, media servers
- **Security Dashboard**: Authentication logs, failed attempts, VPN status

### Alerting
- **Critical Alerts**: VPN disconnection, service failures, resource exhaustion
- **Performance Alerts**: High CPU/memory usage, slow response times
- **Security Alerts**: Authentication failures, suspicious activity
- **Storage Alerts**: Low disk space, backup failures

### Log Analysis
- **Centralized Logging**: All container and system logs in one place
- **Structured Search**: Find specific events across all services
- **Real-time Monitoring**: Live log streaming and filtering
- **Error Tracking**: Automatic error detection and alerting

## 🔐 Security Features

### Authentication
- **2FA Required**: TOTP (Google Authenticator, Authy) for admin access
- **Role-Based Access**: Admin, power user, regular user, guest levels
- **Session Management**: Secure session handling with Redis
- **IP Allowlisting**: Restrict admin access to specific networks

### Network Security
- **Network Segmentation**: Isolated networks for different service tiers
- **VPN Protection**: All downloads routed through VPN with kill-switch
- **SSL Termination**: Automatic HTTPS with Let's Encrypt certificates
- **Rate Limiting**: Protection against brute force and DDoS attacks
- **Security Headers**: HSTS, CSP, and other security headers

### Intrusion Prevention
- **Fail2ban**: Automatic IP blocking for failed authentication attempts
- **Log Monitoring**: Real-time analysis of security events
- **Webhook Security**: HMAC signature verification for webhooks
- **Container Isolation**: Secure container networking and permissions

## 🤖 Automation Features

### Automated Recovery
- **Service Restart**: Automatic restart of failed services
- **VPN Reconnection**: Automatic VPN reconnection with download client protection
- **Disk Cleanup**: Automatic cleanup when disk space is low
- **Container Health**: Proactive container health monitoring

### Media Processing
- **Post-Download Processing**: Automatic optimization and metadata extraction
- **GPU Acceleration**: Hardware-accelerated transcoding and processing
- **Library Updates**: Automatic media library refreshing
- **Quality Control**: Automated file validation and optimization

### Webhook Integrations
- **Sonarr/Radarr**: Download completion notifications and processing
- **Jellyseerr**: Request notifications and automatic approval
- **System Events**: Critical alert handling and automated responses
- **Custom Actions**: Extensible webhook system for custom automation

## 🛠️ Configuration Guide

### Environment Variables

#### Core Settings
```bash
# Domain and SSL
DOMAIN=yourdomain.com
ACME_EMAIL=admin@yourdomain.com

# Authentication Secrets (generate secure random strings)
AUTHELIA_JWT_SECRET=your-super-secure-jwt-secret-here-minimum-32-chars
AUTHELIA_SESSION_SECRET=your-super-secure-session-secret-here-minimum-32-chars
AUTHELIA_STORAGE_ENCRYPTION_KEY=your-super-secure-storage-encryption-key-here-minimum-32-chars

# Database Passwords
POSTGRES_PASSWORD=secure-postgres-password
MYSQL_ROOT_PASSWORD=secure-mysql-root-password
```

#### VPN Configuration
```bash
VPN_PROVIDER=nordvpn  # or surfshark, expressvpn, etc.
VPN_TYPE=openvpn
OPENVPN_USER=your-vpn-username
OPENVPN_PASSWORD=your-vpn-password
VPN_COUNTRY=Switzerland
```

#### Monitoring & Alerts
```bash
# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=secure-grafana-password

# Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ADMIN_EMAIL=admin@yourdomain.com
ALERT_EMAIL_FROM=alerts@yourdomain.com
```

#### API Keys
```bash
SONARR_API_KEY=your-sonarr-api-key
RADARR_API_KEY=your-radarr-api-key
PROWLARR_API_KEY=your-prowlarr-api-key
JELLYFIN_API_KEY=your-jellyfin-api-key
```

### User Management

Edit `authelia/users_database.yml` to configure users:

```yaml
users:
  admin:
    displayname: "Administrator"
    password: "$argon2id$v=19$m=65536,t=3,p=4$YourHashedPassword"
    email: admin@example.com
    groups:
      - admins
      - power_users
      - users
```

Generate password hashes:
```bash
# Using authelia container
docker run --rm authelia/authelia:latest authelia hash-password 'your-password'

# Or use online Argon2 generators with these settings:
# Memory: 64 MB, Iterations: 3, Parallelism: 4, Salt length: 16
```

### Network Configuration

The infrastructure creates isolated networks:
- **security-net** (172.35.0.0/16): Auth and proxy services
- **monitoring-net** (172.33.0.0/16): Monitoring stack
- **vpn-net** (172.32.0.0/16): VPN and protected downloads
- **media-net**: Connects to main media services (external)

## 📈 Performance Tuning

### Resource Allocation
```bash
# Set memory limits for services
PROMETHEUS_MEMORY_LIMIT=2g
GRAFANA_MEMORY_LIMIT=512m
JELLYFIN_MEMORY_LIMIT=4g
```

### Storage Optimization
```bash
# Prometheus retention
PROMETHEUS_RETENTION_DAYS=15

# Loki retention
LOKI_RETENTION_DAYS=7

# Backup retention
BACKUP_RETENTION_DAYS=30
```

### GPU Configuration
```bash
# Enable GPU acceleration
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
```

## 🔧 Management Commands

### Infrastructure Management
```bash
# Deploy infrastructure
./start-infrastructure.sh start

# Stop all services
./start-infrastructure.sh stop

# Restart services
./start-infrastructure.sh restart

# Check service status
./start-infrastructure.sh status

# View logs
./start-infrastructure.sh logs [service-name]

# Update all services
./start-infrastructure.sh update
```

### Individual Service Management
```bash
# Docker Compose commands
docker-compose -f docker-compose.infrastructure.yml ps
docker-compose -f docker-compose.infrastructure.yml logs -f grafana
docker-compose -f docker-compose.infrastructure.yml restart prometheus
```

### Health Monitoring
```bash
# Run infrastructure health check
./scripts/infrastructure-health-check.sh

# Check VPN status
curl http://gluetun:8000/v1/openvpn/status

# Prometheus targets
curl http://prometheus:9090/api/v1/targets
```

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check disk space
df -h

# Check logs
docker-compose -f docker-compose.infrastructure.yml logs service-name
```

#### Authentication Issues
```bash
# Reset Authelia database
docker-compose -f docker-compose.infrastructure.yml stop authelia
docker volume rm media-server-infrastructure_authelia-data
docker-compose -f docker-compose.infrastructure.yml up -d authelia
```

#### VPN Connection Problems
```bash
# Check VPN logs
docker-compose -f docker-compose.infrastructure.yml logs gluetun

# Test VPN connection
docker exec gluetun curl ifconfig.me

# Restart VPN
docker-compose -f docker-compose.infrastructure.yml restart gluetun
```

#### Monitoring Data Loss
```bash
# Backup Grafana dashboards
docker exec grafana curl -s "http://admin:password@localhost:3000/api/search?type=dash-db" | jq

# Restore Prometheus data
docker-compose -f docker-compose.infrastructure.yml stop prometheus
# Restore data volume
docker-compose -f docker-compose.infrastructure.yml up -d prometheus
```

### Log Locations
- Infrastructure logs: `/var/log/health-checks/`
- Webhook logs: `/var/log/webhooks/`
- Container logs: `docker-compose logs`
- System logs: Via Loki at `http://grafana.yourdomain.com`

## 🔗 Integration with Main Stack

### Media Services
The infrastructure integrates with your main media server stack via:
- **Shared networks**: Media services connect to monitoring and security networks
- **Service discovery**: Automatic discovery of containers with proper labels
- **Health monitoring**: All media services monitored automatically
- **Authentication**: All services protected by Authelia 2FA

### Required Labels
Add these labels to your main stack services:
```yaml
services:
  jellyfin:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.jellyfin.rule=Host(`jellyfin.${DOMAIN}`)"
      - "traefik.http.routers.jellyfin.middlewares=authelia@docker"
      - "prometheus.scrape=true"
      - "prometheus.port=8096"
      - "logging=promtail"
```

## 📚 Additional Resources

- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [Authelia Documentation](https://www.authelia.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Gluetun Documentation](https://github.com/qdm12/gluetun)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review service logs for specific errors
3. Consult the relevant service documentation
4. Check infrastructure health with monitoring tools

Remember to keep your configuration files secure and regularly backup your setup!