# Media Stack Implementation Guide
## Based on Architecture Analysis

### Quick Start Implementation

#### Phase 1: Core Infrastructure (Day 1)
```bash
# 1. Create directory structure
mkdir -p /opt/mediaserver/{config,data,secrets,backups}
mkdir -p /opt/mediaserver/config/{traefik,authelia,postgres,redis}
mkdir -p /opt/mediaserver/data/{media,downloads,transcode}

# 2. Set permissions
sudo chown -R $USER:docker /opt/mediaserver
chmod 750 /opt/mediaserver/secrets

# 3. Generate secrets
openssl rand -base64 32 > /opt/mediaserver/secrets/postgres_password
openssl rand -base64 32 > /opt/mediaserver/secrets/redis_password
openssl rand -base64 64 > /opt/mediaserver/secrets/jwt_secret

# 4. Create .env file
cat > /opt/mediaserver/.env <<EOF
# User/Group IDs
PUID=1000
PGID=1000

# Timezone
TZ=America/New_York

# Domain
DOMAIN=media.example.com
CF_EMAIL=admin@example.com
CF_API_TOKEN=your-cloudflare-api-token

# Paths
CONFIG_PATH=/opt/mediaserver/config
DATA_PATH=/opt/mediaserver/data
MEDIA_PATH=/mnt/storage/media
DOWNLOAD_PATH=/mnt/storage/downloads

# Database
POSTGRES_PASSWORD=$(cat /opt/mediaserver/secrets/postgres_password)
REDIS_PASSWORD=$(cat /opt/mediaserver/secrets/redis_password)

# VPN (if using)
VPN_PROVIDER=mullvad
VPN_TYPE=wireguard
WIREGUARD_PRIVATE_KEY=your-wireguard-private-key
EOF
```

#### Phase 2: Deploy Core Services
```yaml
# docker-compose.core.yml
version: '3.9'

x-security: &security
  security_opt:
    - no-new-privileges:true
  restart: unless-stopped

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

services:
  traefik:
    <<: *security
    image: traefik:v3.0
    container_name: traefik
    command:
      - --api.dashboard=true
      - --providers.docker=true
      - --providers.docker.exposedbydefault=false
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.cloudflare.acme.dnschallenge=true
      - --certificatesresolvers.cloudflare.acme.dnschallenge.provider=cloudflare
      - --certificatesresolvers.cloudflare.acme.email=${CF_EMAIL}
      - --certificatesresolvers.cloudflare.acme.storage=/letsencrypt/acme.json
    environment:
      CF_API_EMAIL: ${CF_EMAIL}
      CF_DNS_API_TOKEN: ${CF_API_TOKEN}
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ${CONFIG_PATH}/traefik:/letsencrypt
    networks:
      - frontend
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.dashboard.rule=Host(`traefik.${DOMAIN}`)"
      - "traefik.http.routers.dashboard.entrypoints=websecure"
      - "traefik.http.routers.dashboard.tls.certresolver=cloudflare"
      - "traefik.http.routers.dashboard.service=api@internal"
      - "traefik.http.routers.dashboard.middlewares=auth"
      - "traefik.http.middlewares.auth.basicauth.users=admin:$$2y$$10$$..."

  postgres:
    <<: *security
    image: postgres:16-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: mediaserver
      POSTGRES_USER: mediaserver
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - ${DATA_PATH}/postgres:/var/lib/postgresql/data
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mediaserver"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    <<: *security
    image: redis:7-alpine
    container_name: redis
    command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes
    volumes:
      - ${DATA_PATH}/redis:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
```

#### Phase 3: Deploy Media Services
```yaml
# docker-compose.media.yml
version: '3.9'

services:
  jellyfin:
    <<: *security
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    environment:
      - PUID=${PUID}
      - PGID=${PGID}
      - TZ=${TZ}
      - JELLYFIN_PublishedServerUrl=https://jellyfin.${DOMAIN}
    volumes:
      - ${CONFIG_PATH}/jellyfin:/config
      - ${MEDIA_PATH}:/media:ro
      - ${DATA_PATH}/transcode:/transcode
    devices:
      - /dev/dri:/dev/dri  # Intel GPU
    networks:
      - frontend
      - backend
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.jellyfin.rule=Host(`jellyfin.${DOMAIN}`)"
      - "traefik.http.routers.jellyfin.entrypoints=websecure"
      - "traefik.http.routers.jellyfin.tls.certresolver=cloudflare"
      - "traefik.http.services.jellyfin.loadbalancer.server.port=8096"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
```

### Security Hardening Checklist

#### 1. Container Security
- [ ] Enable read-only root filesystem where possible
- [ ] Drop all capabilities and add only required ones
- [ ] Use non-root user inside containers
- [ ] Enable AppArmor/SELinux profiles
- [ ] Implement resource limits

#### 2. Network Security
- [ ] Use internal networks for backend services
- [ ] Implement firewall rules
- [ ] Enable fail2ban for brute force protection
- [ ] Use VPN for remote access
- [ ] Implement rate limiting in Traefik

#### 3. Data Security
- [ ] Encrypt volumes using LUKS
- [ ] Implement automated backups
- [ ] Use Docker secrets for sensitive data
- [ ] Enable audit logging
- [ ] Implement key rotation

### Monitoring Setup

#### Deploy Monitoring Stack
```yaml
# docker-compose.monitoring.yml
version: '3.9'

services:
  prometheus:
    <<: *security
    image: prom/prometheus:latest
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    volumes:
      - ${CONFIG_PATH}/prometheus:/etc/prometheus
      - ${DATA_PATH}/prometheus:/prometheus
    networks:
      - backend
      - monitoring
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.prometheus.rule=Host(`prometheus.${DOMAIN}`)"
      - "traefik.http.routers.prometheus.entrypoints=websecure"
      - "traefik.http.routers.prometheus.tls.certresolver=cloudflare"
      - "traefik.http.routers.prometheus.middlewares=auth"

  grafana:
    <<: *security
    image: grafana/grafana:latest
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - ${DATA_PATH}/grafana:/var/lib/grafana
      - ${CONFIG_PATH}/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - frontend
      - monitoring
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=Host(`grafana.${DOMAIN}`)"
      - "traefik.http.routers.grafana.entrypoints=websecure"
      - "traefik.http.routers.grafana.tls.certresolver=cloudflare"
    depends_on:
      - prometheus
```

### Backup Strategy

#### Automated Backup Script
```bash
#!/bin/bash
# /opt/mediaserver/scripts/backup.sh

BACKUP_DIR="/mnt/backups/mediaserver"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p ${BACKUP_DIR}/${DATE}

# Backup configurations
echo "Backing up configurations..."
tar -czf ${BACKUP_DIR}/${DATE}/configs.tar.gz -C /opt/mediaserver config

# Backup databases
echo "Backing up PostgreSQL..."
docker exec postgres pg_dumpall -U mediaserver | gzip > ${BACKUP_DIR}/${DATE}/postgres.sql.gz

echo "Backing up Redis..."
docker exec redis redis-cli --rdb /backup/dump.rdb BGSAVE
docker cp redis:/backup/dump.rdb ${BACKUP_DIR}/${DATE}/redis.rdb

# Backup Docker volumes
echo "Backing up Docker volumes..."
for volume in $(docker volume ls -q | grep mediaserver); do
    docker run --rm -v ${volume}:/source -v ${BACKUP_DIR}/${DATE}:/backup alpine tar -czf /backup/${volume}.tar.gz -C /source .
done

# Clean old backups
find ${BACKUP_DIR} -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \;

echo "Backup completed: ${BACKUP_DIR}/${DATE}"
```

### Performance Tuning

#### System Optimization
```bash
# /etc/sysctl.d/99-mediaserver.conf
# Network optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# File system
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Memory
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
```

#### Docker Daemon Configuration
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
```

### Troubleshooting Guide

#### Common Issues and Solutions

1. **SSL Certificate Issues**
   ```bash
   # Check Traefik logs
   docker logs traefik | grep -i error
   
   # Verify DNS resolution
   dig +short jellyfin.${DOMAIN}
   
   # Test certificate
   curl -vI https://jellyfin.${DOMAIN}
   ```

2. **Permission Problems**
   ```bash
   # Fix ownership
   sudo chown -R ${PUID}:${PGID} /opt/mediaserver/config/*
   
   # Check container user
   docker exec jellyfin id
   ```

3. **Performance Issues**
   ```bash
   # Check resource usage
   docker stats --no-stream
   
   # Monitor disk I/O
   iostat -x 1
   
   # Check network bandwidth
   iftop -i docker0
   ```

### Migration from Existing Setup

#### Step-by-Step Migration
1. **Backup existing data**
2. **Export configurations**
3. **Stop old services**
4. **Deploy new stack**
5. **Import data**
6. **Verify functionality**
7. **Update DNS**
8. **Monitor for 24 hours**

### Maintenance Schedule

#### Daily
- Monitor service health
- Check backup completion
- Review error logs

#### Weekly
- Update container images
- Run security scans
- Review resource usage

#### Monthly
- Rotate secrets
- Update documentation
- Performance analysis
- Security audit

#### Quarterly
- Major version updates
- Disaster recovery test
- Architecture review

---
*Implementation Guide v1.0*
*Last Updated: 2025-08-02*