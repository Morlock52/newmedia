# Media Server Deployment Guide

## Prerequisites

### Hardware Requirements
- **Minimum**: 8 cores, 32GB RAM, 10TB storage, GTX 1660
- **Recommended**: 16 cores, 64GB RAM, 50TB storage, RTX 4070
- **Enterprise**: 32 cores, 128GB RAM, 100TB+ storage, 2x RTX 4090

### Software Requirements
- Docker Engine 24.0+
- Docker Compose v2.20+
- Ubuntu 22.04 LTS or similar
- NVIDIA drivers (for GPU acceleration)
- Git

## Initial Setup

### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# Install Docker
curl -fsSL https://get.docker.com | sudo bash

# Add user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (if using NVIDIA GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Directory Structure

```bash
# Create directory structure
sudo mkdir -p /opt/mediaserver/{config,data,backups,cache,logs}
sudo mkdir -p /opt/mediaserver/config/{traefik,authelia,jellyfin,navidrome,immich,grafana,prometheus}
sudo mkdir -p /opt/mediaserver/data/{media,torrents,elasticsearch,postgres,redis}
sudo mkdir -p /opt/mediaserver/data/media/{movies,tv,music,audiobooks,podcasts,books,comics,photos}

# Set permissions
sudo chown -R $USER:$USER /opt/mediaserver
chmod -R 755 /opt/mediaserver
```

### 3. Network Configuration

```bash
# Create Docker networks
docker network create --driver=bridge --subnet=10.10.0.0/24 proxy_network
docker network create --driver=bridge --subnet=10.10.1.0/24 media_network
docker network create --driver=bridge --subnet=10.10.2.0/24 admin_network
docker network create --driver=bridge --subnet=10.10.3.0/24 data_network

# Configure firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 51820/udp  # WireGuard
sudo ufw enable
```

## Configuration

### 1. Environment Setup

```bash
cd /opt/mediaserver

# Copy environment template
cp architecture/.env.example .env

# Generate secrets
echo "AUTHELIA_JWT_SECRET=$(openssl rand -hex 32)" >> .env
echo "AUTHELIA_SESSION_SECRET=$(openssl rand -hex 32)" >> .env
echo "AUTHELIA_STORAGE_ENCRYPTION_KEY=$(openssl rand -hex 32)" >> .env
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)" >> .env
echo "REDIS_PASSWORD=$(openssl rand -base64 32)" >> .env
echo "IMMICH_DB_PASSWORD=$(openssl rand -base64 32)" >> .env
echo "MEILI_MASTER_KEY=$(openssl rand -hex 32)" >> .env

# Edit .env file with your domain and other settings
nano .env
```

### 2. SSL Certificates

```bash
# For Let's Encrypt (automatic)
# Certificates will be generated automatically by Traefik

# For custom certificates
mkdir -p config/traefik/certs
cp /path/to/fullchain.pem config/traefik/certs/
cp /path/to/privkey.pem config/traefik/certs/
```

### 3. Service Configuration

```bash
# Copy configuration files
cp architecture/configs/traefik.yml config/traefik/
cp architecture/configs/traefik-dynamic.yml config/traefik/dynamic/
cp architecture/configs/authelia-configuration.yml config/authelia/
cp architecture/configs/prometheus.yml config/prometheus/
cp architecture/configs/prometheus-alerts.yml config/prometheus/rules/

# Create Authelia users database
cat > config/authelia/users_database.yml << EOF
users:
  admin:
    displayname: "Admin User"
    password: "\$argon2id\$v=19\$m=65536,t=3,p=4\$..."  # Generate with authelia hash-password
    email: admin@example.com
    groups:
      - admins
      - users
EOF
```

## Deployment

### 1. Start Core Services

```bash
# Start infrastructure services first
docker-compose -f architecture/docker-compose-enhanced.yml up -d \
    traefik authelia postgres redis elasticsearch

# Wait for services to be ready
sleep 30

# Start media services
docker-compose -f architecture/docker-compose-enhanced.yml up -d \
    jellyfin navidrome audiobookshelf immich-server kavita

# Start content management
docker-compose -f architecture/docker-compose-enhanced.yml up -d \
    sonarr radarr lidarr readarr prowlarr

# Start monitoring
docker-compose -f architecture/docker-compose-enhanced.yml up -d \
    prometheus grafana loki
```

### 2. Initial Service Configuration

#### Jellyfin Setup
1. Access https://jellyfin.media.example.com
2. Complete wizard
3. Add media libraries pointing to /media/*
4. Configure hardware acceleration
5. Set up user accounts

#### Immich Setup
1. Access https://photos.media.example.com
2. Create admin account
3. Configure machine learning models
4. Set up mobile app sync

#### Grafana Setup
1. Access https://grafana.media.example.com
2. Import dashboards from configs/grafana-dashboards.json
3. Configure alert notifications

### 3. Performance Tuning

```bash
# System tuning
sudo sysctl -w vm.max_map_count=262144
sudo sysctl -w fs.file-max=65536
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
echo "fs.file-max=65536" | sudo tee -a /etc/sysctl.conf

# Docker daemon configuration
sudo cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker
```

## Maintenance

### Daily Tasks
- Monitor Grafana dashboards
- Check backup status
- Review security alerts

### Weekly Tasks
- Update container images
- Clean up old logs
- Verify backup integrity

### Monthly Tasks
- Security audit
- Performance review
- Storage capacity planning

## Backup and Recovery

### Automated Backups

```bash
# Configure Duplicati
# Access https://backup.media.example.com
# Set up backup jobs for:
# - /opt/mediaserver/config (daily)
# - /opt/mediaserver/data/media/metadata (daily)
# - Database dumps (hourly)
```

### Manual Backup

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/opt/mediaserver/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configurations
tar -czf $BACKUP_DIR/configs.tar.gz /opt/mediaserver/config

# Backup databases
docker exec postgres pg_dumpall -U mediaserver > $BACKUP_DIR/postgres.sql
docker exec immich-postgres pg_dump -U immich immich > $BACKUP_DIR/immich.sql

# Backup Docker volumes
docker run --rm -v mediaserver_config_data:/data -v $BACKUP_DIR:/backup \
    alpine tar -czf /backup/volumes.tar.gz /data
```

### Disaster Recovery

```bash
# Restore from backup
tar -xzf /path/to/backup/configs.tar.gz -C /
docker exec -i postgres psql -U mediaserver < /path/to/backup/postgres.sql
docker-compose -f architecture/docker-compose-enhanced.yml up -d
```

## Security Hardening

### 1. Firewall Rules

```bash
# Restrict access to admin services
sudo iptables -A INPUT -s 10.10.2.0/24 -p tcp --dport 9000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9000 -j DROP
```

### 2. Fail2ban Configuration

```bash
# Install fail2ban
sudo apt install fail2ban

# Configure for Authelia
cat > /etc/fail2ban/jail.local << EOF
[authelia]
enabled = true
port = http,https
filter = authelia
logpath = /opt/mediaserver/logs/authelia.log
maxretry = 5
bantime = 900
findtime = 300
EOF
```

### 3. Regular Updates

```bash
# Update script
#!/bin/bash
docker-compose -f architecture/docker-compose-enhanced.yml pull
docker-compose -f architecture/docker-compose-enhanced.yml up -d
docker image prune -af
```

## Troubleshooting

### Common Issues

1. **Service not accessible**
   - Check Traefik logs: `docker logs traefik`
   - Verify DNS resolution
   - Check firewall rules

2. **Authentication issues**
   - Check Authelia logs: `docker logs authelia`
   - Verify LDAP connectivity
   - Check session Redis

3. **Performance problems**
   - Monitor with: `docker stats`
   - Check disk I/O: `iostat -x 1`
   - Review Grafana metrics

### Debug Commands

```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Enter container
docker exec -it [container_name] /bin/bash

# Test connectivity
docker exec -it traefik wget -O- http://jellyfin:8096/health

# Check DNS
docker exec -it traefik nslookup jellyfin
```

## Scaling

### Horizontal Scaling

```bash
# Scale specific service
docker-compose -f architecture/docker-compose-enhanced.yml up -d --scale jellyfin=3

# Add load balancer configuration
# Update traefik-dynamic.yml with multiple backend servers
```

### Adding Storage

```bash
# Add new mount point
sudo mkdir -p /mnt/storage2/media
sudo mount /dev/sdb1 /mnt/storage2/media

# Update docker-compose.yml volumes
volumes:
  - /mnt/storage2/media:/media2:ro
```

## Monitoring and Alerts

### Prometheus Queries

```promql
# Service availability
up{job=~"jellyfin|navidrome|immich|audiobookshelf|kavita"}

# Active streams
sum(jellyfin_active_streams) by (type)

# Storage usage
100 - (node_filesystem_avail_bytes / node_filesystem_size_bytes * 100)

# GPU utilization
nvidia_gpu_utilization_percentage
```

### Alert Configuration

Configure alerts in Grafana for:
- Service downtime
- High resource usage
- Storage capacity
- Backup failures
- Security events