# Unified Media Server Architecture - Implementation Guide

## Quick Start Guide

This guide provides step-by-step instructions to implement the unified media server architecture with enable/disable functionality.

## Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM (32GB+ recommended for full stack)
- 500GB+ storage for media and applications
- Domain name with Cloudflare (optional but recommended)
- VPN subscription (for secure downloading)

## Implementation Steps

### 1. Initial Setup

```bash
# Create project directory
sudo mkdir -p /opt/media-server
cd /opt/media-server

# Set permissions
sudo chown -R $USER:$USER /opt/media-server

# Create directory structure
mkdir -p {config,data,media,downloads,backups,scripts}
mkdir -p config/{traefik,authelia,homepage,grafana}
mkdir -p media/{movies,tv,music,books,audiobooks,podcasts,comics,photos}
mkdir -p downloads/{complete,incomplete,torrents,usenet}
```

### 2. Deploy Core Files

Create the main docker-compose.yml file with the unified architecture (use the docker-compose.unified.yml from the main document).

### 3. Environment Configuration

```bash
# Create .env file
cat > .env << 'EOF'
# Basic Configuration
TZ=America/New_York
PUID=1000
PGID=1000

# Domain Configuration
DOMAIN=yourdomain.com
ACME_EMAIL=your-email@example.com

# Security - Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
AUTHELIA_JWT_SECRET=$(openssl rand -base64 32)
AUTHELIA_SESSION_SECRET=$(openssl rand -base64 32)
AUTHELIA_STORAGE_ENCRYPTION_KEY=$(openssl rand -base64 32)

# Paths
MEDIA_PATH=/opt/media-server/media
DOWNLOADS_PATH=/opt/media-server/downloads
CONFIG_PATH=/opt/media-server/config
DATA_PATH=/opt/media-server/data

# Service Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Hardware Acceleration
JELLYFIN_HARDWARE_ACCELERATION=none  # Change to intel/nvidia/amd if available
EOF

# Secure the .env file
chmod 600 .env
```

### 4. Service Management Scripts

Create the unified management script:

```bash
# Download or create the unified-media-manager.sh script
# Make it executable
chmod +x unified-media-manager.sh

# Create alias for easy access
echo 'alias media-server="/opt/media-server/unified-media-manager.sh"' >> ~/.bashrc
source ~/.bashrc
```

### 5. Initial Deployment

```bash
# Start core services only
./unified-media-manager.sh enable core

# Check status
./unified-media-manager.sh status

# View logs
./unified-media-manager.sh logs traefik
```

### 6. Configure Authentication (Authelia)

```yaml
# config/authelia/configuration.yml
---
theme: dark
jwt_secret: ${AUTHELIA_JWT_SECRET}
default_redirection_url: https://home.${DOMAIN}

server:
  host: 0.0.0.0
  port: 9091

log:
  level: info

totp:
  issuer: Media Server

authentication_backend:
  file:
    path: /config/users_database.yml
    password:
      algorithm: argon2id
      iterations: 1
      salt_length: 16
      parallelism: 8
      memory: 64

access_control:
  default_policy: deny
  rules:
    - domain: "*.${DOMAIN}"
      policy: two_factor

session:
  name: authelia_session
  secret: ${AUTHELIA_SESSION_SECRET}
  expiration: 3600
  inactivity: 300
  domain: ${DOMAIN}

regulation:
  max_retries: 3
  find_time: 120
  ban_time: 300

storage:
  encryption_key: ${AUTHELIA_STORAGE_ENCRYPTION_KEY}
  postgres:
    host: postgres
    port: 5432
    database: authelia
    username: authelia
    password: ${POSTGRES_PASSWORD}

notifier:
  filesystem:
    filename: /config/notification.txt
```

Create users database:
```yaml
# config/authelia/users_database.yml
users:
  admin:
    displayname: "Admin User"
    password: "$argon2id$v=19$m=65536,t=3,p=4$..."  # Generate with docker run authelia/authelia:latest authelia hash-password
    email: admin@example.com
    groups:
      - admins
      - users
```

### 7. Deploy Media Services

```bash
# Enable media streaming
./unified-media-manager.sh enable media

# Enable content automation
./unified-media-manager.sh enable automation

# Enable downloads
./unified-media-manager.sh enable downloads

# Or use a preset
./unified-media-manager.sh preset movies_tv
```

### 8. Configure Services

#### Jellyfin Initial Setup
1. Access: https://jellyfin.yourdomain.com
2. Complete wizard
3. Add media libraries pointing to /media subdirectories
4. Enable hardware acceleration if available

#### Prowlarr Configuration
1. Access: https://prowlarr.yourdomain.com
2. Add indexers
3. Configure apps (Sonarr, Radarr, etc.)
4. Test connectivity

#### Sonarr/Radarr Setup
1. Add Prowlarr as indexer proxy
2. Configure download clients (qBittorrent)
3. Set up root folders
4. Import existing media

### 9. Enable Additional Features

```bash
# Photo management
./unified-media-manager.sh enable photos

# Music streaming
./unified-media-manager.sh enable music

# Books and audiobooks
./unified-media-manager.sh enable books

# Monitoring
./unified-media-manager.sh enable monitoring

# Management dashboards
./unified-media-manager.sh enable management
```

### 10. Configure Homepage Dashboard

```yaml
# config/homepage/services.yaml
---
- Media:
    - Jellyfin:
        href: https://jellyfin.${DOMAIN}
        icon: jellyfin.png
        description: Media streaming
        widget:
          type: jellyfin
          url: http://jellyfin:8096
          key: ${JELLYFIN_API_KEY}

    - Overseerr:
        href: https://requests.${DOMAIN}
        icon: overseerr.png
        description: Request management
        widget:
          type: overseerr
          url: http://overseerr:5055
          key: ${OVERSEERR_API_KEY}

- Automation:
    - Sonarr:
        href: https://sonarr.${DOMAIN}
        icon: sonarr.png
        description: TV management
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: ${SONARR_API_KEY}

    - Radarr:
        href: https://radarr.${DOMAIN}
        icon: radarr.png
        description: Movie management
        widget:
          type: radarr
          url: http://radarr:7878
          key: ${RADARR_API_KEY}
```

### 11. Setup Monitoring

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'docker'
    static_configs:
      - targets: ['docker-exporter:9417']
```

### 12. Backup Configuration

```bash
# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/media-server/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
cp -r /opt/media-server/config "$BACKUP_DIR/"
cp /opt/media-server/.env "$BACKUP_DIR/"
cp /opt/media-server/docker-compose.yml "$BACKUP_DIR/"

# Backup Docker volumes
docker run --rm -v media-server_config:/source -v "$BACKUP_DIR":/backup alpine tar -czf /backup/volumes.tar.gz -C /source .

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x scripts/backup.sh

# Schedule daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/media-server/scripts/backup.sh") | crontab -
```

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check logs
   ./unified-media-manager.sh logs service-name
   
   # Verify permissions
   sudo chown -R 1000:1000 config/ data/
   ```

2. **Network issues**
   ```bash
   # Recreate networks
   docker network prune
   docker-compose up -d
   ```

3. **Authentication problems**
   ```bash
   # Check Authelia logs
   docker logs authelia
   
   # Verify configuration
   docker exec authelia authelia validate-config
   ```

### Performance Tuning

1. **Enable Hardware Acceleration**
   ```yaml
   # For Intel Quick Sync
   devices:
     - /dev/dri:/dev/dri
   
   # For NVIDIA
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Optimize Database**
   ```sql
   -- Connect to PostgreSQL
   docker exec -it postgres psql -U mediauser -d mediaserver
   
   -- Optimize tables
   VACUUM ANALYZE;
   ```

3. **Configure Caching**
   ```bash
   # Increase Redis memory
   docker exec -it redis redis-cli
   CONFIG SET maxmemory 2gb
   CONFIG SET maxmemory-policy allkeys-lru
   ```

## Security Hardening

1. **Firewall Rules**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

2. **Fail2ban Integration**
   ```bash
   # Install fail2ban
   sudo apt install fail2ban
   
   # Configure for Docker
   cat > /etc/fail2ban/jail.d/docker-authelia.conf << EOF
   [authelia]
   enabled = true
   filter = authelia
   logpath = /opt/media-server/config/authelia/authelia.log
   maxretry = 3
   bantime = 3600
   findtime = 600
   EOF
   ```

3. **Regular Updates**
   ```bash
   # Update all images
   ./unified-media-manager.sh update
   
   # Automated updates with Watchtower
   docker run -d \
     --name watchtower \
     -v /var/run/docker.sock:/var/run/docker.sock \
     containrrr/watchtower \
     --cleanup --interval 86400
   ```

## Maintenance Schedule

### Daily
- Check service health: `./unified-media-manager.sh health`
- Review logs for errors
- Monitor disk space

### Weekly
- Update container images
- Review security logs
- Clean up old downloads

### Monthly
- Full system backup
- Security audit
- Performance review
- Update documentation

## Advanced Configurations

### Multi-Server Setup
For large deployments, services can be distributed across multiple servers:

1. **Media Server**: Jellyfin, Plex, Navidrome
2. **Download Server**: VPN, qBittorrent, Arr stack
3. **Database Server**: PostgreSQL, Redis
4. **Monitoring Server**: Prometheus, Grafana

### CDN Integration
```yaml
# Cloudflare CDN caching rules
Page Rules:
  - URL: *.yourdomain.com/images/*
    Cache Level: Cache Everything
    Edge Cache TTL: 1 month
  
  - URL: *.yourdomain.com/api/*
    Cache Level: Bypass
```

### High Availability
For production environments:
- Use Docker Swarm or Kubernetes
- Implement database replication
- Set up load balancing
- Configure automatic failover

## Migration from Existing Setup

1. **Inventory Current Services**
   ```bash
   docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
   ```

2. **Backup Existing Data**
   ```bash
   # Backup all volumes
   for volume in $(docker volume ls -q); do
     docker run --rm -v $volume:/source -v /backup:/backup alpine \
       tar -czf /backup/$volume.tar.gz -C /source .
   done
   ```

3. **Migrate Configurations**
   ```bash
   # Copy existing configs
   cp -r /old/config/* /opt/media-server/config/
   ```

4. **Import Media Libraries**
   - Point new services to existing media directories
   - Rescan libraries
   - Verify metadata

## Conclusion

This implementation provides:
- Modular service deployment
- Easy enable/disable functionality
- Comprehensive monitoring
- Strong security
- Scalable architecture

The system can grow with your needs, from a basic media server to a full-featured media ecosystem with advanced features like AI/ML processing, photo management, and more.