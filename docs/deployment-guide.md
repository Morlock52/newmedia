# 🚀 NEXUS Platform Deployment Guide

Complete deployment and setup guide for the NEXUS Media Server Platform, covering all components from basic media services to advanced AI/ML, AR/VR, and Web3 features.

## Table of Contents

1. [Overview & Prerequisites](#overview--prerequisites)
2. [Quick Start Deployment](#quick-start-deployment)
3. [Component-Specific Deployments](#component-specific-deployments)
4. [Production Deployment](#production-deployment)
5. [Platform Configuration](#platform-configuration)
6. [Security Hardening](#security-hardening)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Health Checks](#monitoring--health-checks)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance & Updates](#maintenance--updates)

---

## Overview & Prerequisites

### System Requirements

#### Minimum Requirements (Core Media Stack)
```bash
# Hardware
CPU: Dual-core 2.5GHz (Intel/AMD)
RAM: 8GB
Storage: 50GB free space
Network: Broadband internet connection

# Software
OS: Ubuntu 20.04+, macOS 10.15+, Windows 10+
Docker: 20.10+
Docker Compose: 2.0+
```

#### Recommended Requirements (Full Platform)
```bash
# Hardware
CPU: 8-core 3.0GHz with AVX support
RAM: 32GB (64GB for heavy AI/ML workloads)
GPU: NVIDIA RTX 4070+ with 12GB VRAM (for AI/ML)
Storage: 500GB SSD + 2TB HDD
Network: Gigabit ethernet

# Software
OS: Ubuntu 22.04 LTS (recommended)
Docker: 24.0+
Docker Compose: 2.20+
NVIDIA Docker Runtime (for GPU acceleration)
```

#### Optional Requirements
```bash
# For Advanced Features
VPN Account: Mullvad, NordVPN, or PIA (for torrents)
Cloudflare Account: For SSL certificates and CDN
Domain Name: For public access
NVIDIA GPU: RTX 4070+ for AI/ML acceleration
WebXR Browser: Chrome 90+, Edge 90+, Safari 15.4+
Crypto Wallet: MetaMask or similar for Web3 features
```

### Pre-Installation Checklist

```bash
# 1. Verify Docker installation
docker --version
docker compose version

# 2. Check available ports (none should be occupied)
netstat -tuln | grep -E ':(3000|3001|5055|6767|7878|8080|8081|8082|8096|8181|8686|8989|9000|9090|9696)'

# 3. Verify disk space
df -h

# 4. Check system resources
free -h
cat /proc/cpuinfo | grep -c ^processor

# 5. Verify GPU (if available)
nvidia-smi  # For NVIDIA GPUs
```

---

## Quick Start Deployment

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/nexus-media-platform.git
cd nexus-media-platform

# Make scripts executable
chmod +x *.sh
chmod +x scripts/*.sh

# Verify repository structure
ls -la
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

**Essential `.env` Configuration:**
```env
# === CORE SETTINGS ===
TZ=America/New_York
PUID=1000
PGID=1000
DOMAIN=localhost

# === PATHS ===
MEDIA_PATH=./media-data
DOWNLOADS_PATH=./media-data/downloads
CONFIG_PATH=./config

# === VPN (for torrents) ===
VPN_PROVIDER=mullvad
VPN_PRIVATE_KEY=your_wireguard_private_key
VPN_ADDRESSES=10.x.x.x/32

# === OPTIONAL: SSL/CLOUDFLARE ===
CLOUDFLARE_EMAIL=your-email@example.com
CLOUDFLARE_API_KEY=your_cloudflare_api_key

# === AI/ML SETTINGS ===
ENABLE_AI_ML=true
ENABLE_GPU=true
TF_BACKEND=tensorflow-gpu

# === AR/VR SETTINGS ===
ENABLE_WEBXR=true
XR_FEATURES=hand-tracking,passthrough

# === WEB3 SETTINGS ===
ENABLE_WEB3=false
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/your_key
POLYGON_RPC_URL=https://polygon-rpc.com

# === SECURITY ===
ENABLE_QUANTUM_SECURITY=true
QUANTUM_HYBRID_MODE=true
```

### 3. Basic Deployment

```bash
# Deploy core media services only
./deploy.sh

# Or deploy everything (may take 15-30 minutes)
./deploy-ultimate-2025.sh

# Monitor deployment progress
docker compose logs -f
```

### 4. Verify Deployment

```bash
# Check service health
./health-check-ultimate.sh

# Access primary services
echo "Jellyfin: http://localhost:8096"
echo "Dashboard: http://localhost:3001"
echo "Portainer: http://localhost:9000"
```

---

## Component-Specific Deployments

### Core Media Stack Only

```bash
# Deploy traditional media services
./deploy-media.sh

# Includes:
# - Jellyfin (Media Server)
# - Sonarr/Radarr/Lidarr (Content Management)
# - Prowlarr/Bazarr (Indexers/Subtitles)
# - qBittorrent/SABnzbd (Downloads)
# - Overseerr (Requests)
# - Grafana/Prometheus (Monitoring)

# Verify core services
curl -s http://localhost:8096/health
curl -s http://localhost:3001
```

### AI/ML Nexus System

```bash
# Deploy AI/ML services
cd ai-ml-nexus
./deploy.sh

# Or using main deployment with AI enabled
export ENABLE_AI_ML=true
./deploy.sh

# Verify AI/ML services
curl -s http://localhost:8080/api/health
curl -s http://localhost:8081/api/health  # Recommendations
curl -s http://localhost:8082/api/health  # Content Analysis
```

**AI/ML GPU Setup:**
```bash
# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### AR/VR WebXR Platform

```bash
# Deploy AR/VR services
cd ar-vr-media
./deploy.sh

# Or enable in main deployment
export ENABLE_WEBXR=true
./deploy.sh

# Generate SSL certificate for WebXR (required)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Start with HTTPS
node server.js --https --cert cert.pem --key key.pem

# Access AR/VR platform
echo "WebXR Platform: https://localhost:8080"
echo "Note: WebXR requires HTTPS and compatible browser"
```

### Web3 Blockchain Integration

```bash
# Deploy Web3 services
cd web3-blockchain-integration
npm install
./deploy-web3.sh

# Configure blockchain networks
cp .env.web3.example .env.web3
# Edit with your RPC URLs and private keys

# Deploy smart contracts
npm run deploy:polygon
npm run deploy:ethereum

# Verify Web3 services
curl -s http://localhost:3001/api/web3/health
```

### Quantum Security Layer

```bash
# Deploy quantum-resistant security
cd quantum-security
docker compose up -d

# Test quantum TLS
./scripts/test-quantum-tls.sh

# Verify quantum security status
curl -s http://localhost:8080/api/security/quantum/status
```

### Voice AI System

```bash
# Deploy voice processing
cd voice-ai-system
npm install
npm start

# Test voice recognition
curl -X POST http://localhost:8083/api/voice/test \
  -H "Content-Type: application/json" \
  -d '{"text": "test voice synthesis"}'
```

---

## Production Deployment

### 1. Production Environment Setup

```bash
# Create production environment file
cp .env.example .env.production

# Configure for production
sed -i 's/localhost/your-domain.com/g' .env.production
sed -i 's/DEBUG=true/DEBUG=false/g' .env.production
sed -i 's/ENABLE_DEVELOPMENT_MODE=true/ENABLE_DEVELOPMENT_MODE=false/g' .env.production
```

**Production `.env` Configuration:**
```env
# === PRODUCTION SETTINGS ===
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# === DOMAIN & SSL ===
DOMAIN=your-domain.com
ENABLE_SSL=true
CLOUDFLARE_EMAIL=your-email@example.com
CLOUDFLARE_API_KEY=your_api_key
CLOUDFLARE_ZONE_ID=your_zone_id

# === SECURITY ===
ENABLE_AUTHENTICATION=true
ENABLE_2FA=true
ENABLE_QUANTUM_SECURITY=true
SESSION_SECRET=your_secure_session_secret_32_chars

# === DATABASE ===
POSTGRES_PASSWORD=secure_postgres_password
REDIS_PASSWORD=secure_redis_password

# === BACKUP ===
ENABLE_AUTOMATED_BACKUP=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30

# === MONITORING ===
ENABLE_MONITORING=true
ENABLE_ALERTING=true
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

### 2. SSL/TLS Configuration

```bash
# Using Cloudflare (Recommended)
./scripts/setup-cloudflare-ssl.sh

# Or using Let's Encrypt
./scripts/setup-letsencrypt.sh

# Verify SSL configuration
curl -I https://your-domain.com
```

### 3. Reverse Proxy Setup (Traefik)

```yaml
# config/traefik/traefik.yml
api:
  dashboard: true
  insecure: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entrypoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

certificatesResolvers:
  cloudflare:
    acme:
      dnschallenge:
        provider: cloudflare
      email: your-email@example.com
      storage: /letsencrypt/acme.json

providers:
  docker:
    exposedByDefault: false
  file:
    filename: /config/dynamic.yml
```

### 4. Database Configuration

```bash
# Setup PostgreSQL for production
docker run -d \
  --name postgres-prod \
  -e POSTGRES_DB=nexus \
  -e POSTGRES_USER=nexus \
  -e POSTGRES_PASSWORD=secure_password \
  -v postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15-alpine

# Setup Redis for caching
docker run -d \
  --name redis-prod \
  -e REDIS_PASSWORD=secure_password \
  -v redis_data:/data \
  -p 6379:6379 \
  redis:7-alpine redis-server --requirepass secure_password
```

### 5. Production Deployment

```bash
# Deploy to production
./deploy-production-2025.sh

# Or with specific environment
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify production deployment
./health-check-production.sh
```

---

## Platform Configuration

### 1. Jellyfin Media Server

```bash
# Access Jellyfin setup
open http://localhost:8096

# Initial configuration steps:
# 1. Select language and create admin user
# 2. Add media libraries:
#    - Movies: /media/movies
#    - TV Shows: /media/tv
#    - Music: /media/music
# 3. Configure hardware transcoding (if available)
# 4. Enable DLNA and remote access
```

**Hardware Transcoding Setup:**
```bash
# For Intel Quick Sync
docker compose exec jellyfin ls -la /dev/dri/

# For NVIDIA NVENC
docker compose exec jellyfin nvidia-smi
```

### 2. Content Management (*arr Suite)

**Prowlarr (Indexer Management):**
```bash
# Access Prowlarr
open http://localhost:9696

# Configuration steps:
# 1. Add indexers (public and private)
# 2. Configure API keys for other *arr apps
# 3. Test indexer connections
# 4. Set up automatic sync
```

**Sonarr/Radarr Configuration:**
```bash
# Common settings for both
curl -X POST "http://localhost:8989/api/v3/rootfolder" \
  -H "X-Api-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/media/tv",
    "accessible": true,
    "freeSpace": 1000000000000
  }'

# Quality profiles
curl -X POST "http://localhost:8989/api/v3/qualityprofile" \
  -H "X-Api-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "4K Ultra HD",
    "upgradeAllowed": true,
    "cutoff": 19,
    "items": [...]
  }'
```

### 3. Download Clients

**qBittorrent Configuration:**
```bash
# Access qBittorrent
open http://localhost:8080

# Default credentials: admin/adminadmin
# IMPORTANT: Change password immediately!

# Configure download paths:
# - Default Save Path: /downloads/complete
# - Incomplete Save Path: /downloads/incomplete
# - Category paths:
#   - movies: /downloads/complete/movies
#   - tv: /downloads/complete/tv
#   - music: /downloads/complete/music
```

**SABnzbd Configuration:**
```bash
# Access SABnzbd
open http://localhost:8081

# Add Usenet providers
# Configure categories and paths
# Set up post-processing scripts
```

### 4. AI/ML System Configuration

```bash
# Configure AI/ML models
curl -X POST "http://localhost:8080/api/models/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "recommendation_model": "collaborative-filtering-v2",
    "content_analysis_model": "yolo-v8",
    "voice_model": "whisper-large-v3",
    "compression_model": "neural-autoencoder-v2"
  }'

# Set GPU memory allocation
curl -X POST "http://localhost:8080/api/gpu/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_limit": "8GB",
    "enable_mixed_precision": true,
    "concurrent_sessions": 4
  }'
```

### 5. AR/VR Platform Configuration

```javascript
// WebXR configuration
const xrConfig = {
  handTracking: {
    enabled: true,
    precision: 'high',
    gestures: ['pinch', 'point', 'grab', 'peace', 'thumbs_up']
  },
  spatialVideo: {
    formats: ['side-by-side', 'over-under', 'mv-hevc'],
    maxResolution: '4K',
    frameRate: 60
  },
  mixedReality: {
    passthrough: true,
    planeDetection: true,
    anchoring: true,
    occlusion: true
  }
};
```

### 6. Web3 Configuration

```bash
# Deploy smart contracts
cd web3-blockchain-integration/smart-contracts
npm run deploy --network polygon

# Configure IPFS node
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["PUT", "POST", "GET"]'

# Set up MetaMask connection
# Add custom RPC networks
# Import contract ABIs
```

---

## Security Hardening

### 1. Container Security

```bash
# Run security scan
docker scout cves

# Update base images
docker compose pull
docker system prune -af

# Enable security scanning
docker compose -f docker-compose.yml -f docker-compose.security.yml up -d
```

### 2. Network Security

```yaml
# docker-compose.security.yml
networks:
  frontend:
    driver: bridge
    internal: false
  backend:
    driver: bridge
    internal: true
  ai-ml:
    driver: bridge
    internal: true
  downloads:
    driver: bridge
    internal: false  # Needs internet for VPN
```

### 3. Authentication & Authorization

```bash
# Setup Authelia
cp config/authelia/configuration.yml.example config/authelia/configuration.yml

# Generate secrets
openssl rand -hex 32 > secrets/authelia_jwt_secret.txt
openssl rand -hex 32 > secrets/authelia_session_secret.txt
openssl rand -hex 32 > secrets/authelia_encryption_key.txt

# Configure users database
htpasswd -B -C 12 config/authelia/users_database.yml admin
```

### 4. SSL/TLS Hardening

```yaml
# config/traefik/tls.yml
tls:
  options:
    default:
      sslProtocols:
        - "TLSv1.2"
        - "TLSv1.3"
      cipherSuites:
        - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
        - "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"
        - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
      curvePreferences:
        - "CurveP521"
        - "CurveP384"
      minVersion: "VersionTLS12"
```

### 5. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8096/tcp  # Jellyfin (if direct access needed)

# Docker iptables rules
sudo iptables -I DOCKER-USER -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -I DOCKER-USER -p tcp --dport 8080 -j DROP
```

---

## Performance Optimization

### 1. System-Level Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf
echo "vm.max_map_count = 262144" >> /etc/sysctl.conf
sysctl -p

# Docker optimization
echo '{"log-driver": "json-file", "log-opts": {"max-size": "10m", "max-file": "3"}}' > /etc/docker/daemon.json
systemctl restart docker
```

### 2. Database Optimization

```sql
-- PostgreSQL optimization
-- config/postgres/postgresql.conf
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 32MB
min_wal_size = 2GB
max_wal_size = 8GB
```

### 3. AI/ML Performance Optimization

```bash
# GPU memory optimization
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_MEMORY_LIMIT=8192  # 8GB

# Model quantization
curl -X POST "http://localhost:8080/api/models/quantize" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "recommendation-engine",
    "quantization": "int8",
    "optimize_for": "latency"
  }'

# Batch processing optimization
curl -X POST "http://localhost:8080/api/config/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 32,
    "max_concurrent_batches": 4,
    "queue_timeout": 5000
  }'
```

### 4. Storage Optimization

```bash
# SSD optimization
echo 'deadline' > /sys/block/sda/queue/scheduler
echo '1' > /sys/block/sda/queue/iosched/fifo_batch

# Media storage optimization
# Use different storage tiers:
# - SSD: OS, databases, active downloads
# - HDD: Media libraries, completed downloads
# - Network: Backup storage

# Docker volume optimization
docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=2g \
  nexus-temp-processing
```

---

## Monitoring & Health Checks

### 1. Grafana Dashboard Setup

```bash
# Import pre-built dashboards
curl -X POST "http://admin:admin@localhost:3000/api/dashboards/db" \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/nexus-overview.json

# Access Grafana
open http://localhost:3000
# Login: admin/admin (change on first login)
```

### 2. Prometheus Metrics

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'nexus-core'
    static_configs:
      - targets: ['jellyfin:8096', 'sonarr:8989', 'radarr:7878']
  
  - job_name: 'nexus-ai-ml'
    static_configs:
      - targets: ['ai-orchestrator:8080', 'recommendation-engine:8081']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
  
  - job_name: 'gpu-exporter'
    static_configs:
      - targets: ['gpu-exporter:9101']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 3. Health Check Scripts

```bash
#!/bin/bash
# health-check-ultimate.sh

echo "=== NEXUS Platform Health Check ==="

# Core services
services=(
  "jellyfin:8096"
  "sonarr:8989"
  "radarr:7878"
  "prowlarr:9696"
  "overseerr:5055"
  "homepage:3001"
)

# AI/ML services
if [ "$ENABLE_AI_ML" = "true" ]; then
  services+=(
    "ai-orchestrator:8080"
    "recommendation-engine:8081"
    "content-analysis:8082"
  )
fi

# Check each service
for service in "${services[@]}"; do
  IFS=':' read -r name port <<< "$service"
  if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
    echo "✓ $name is healthy"
  else
    echo "✗ $name is not responding"
  fi
done

# Check GPU (if available)
if command -v nvidia-smi &> /dev/null; then
  echo "=== GPU Status ==="
  nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
fi

# Check disk space
echo "=== Disk Usage ==="
df -h | grep -E "(/$|/var/lib/docker|media-data)"

# Check memory usage
echo "=== Memory Usage ==="
free -h

echo "=== Health Check Complete ==="
```

### 4. Alerting Configuration

```yaml
# config/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@nexus-platform.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'

receivers:
- name: 'default'
  email_configs:
  - to: 'admin@nexus-platform.com'
    subject: 'NEXUS Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Services Not Starting

```bash
# Check Docker daemon
sudo systemctl status docker
sudo systemctl start docker

# Check container logs
docker compose logs -f [service_name]

# Check for port conflicts
sudo netstat -tuln | grep :8096

# Restart specific service
docker compose restart jellyfin
```

#### 2. Permission Issues

```bash
# Fix file permissions
sudo chown -R 1000:1000 ./config ./media-data

# Check mount points
docker compose exec jellyfin ls -la /media

# Fix SELinux context (if applicable)
sudo setsebool -P container_manage_cgroup on
```

#### 3. VPN Connection Issues

```bash
# Check VPN status
docker compose logs gluetun

# Test VPN connection
docker compose exec gluetun curl ifconfig.me

# Restart VPN container
docker compose restart gluetun

# Check firewall rules
iptables -L DOCKER-USER
```

#### 4. AI/ML GPU Issues

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker compose exec ai-orchestrator nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi
```

#### 5. WebXR Not Working

```bash
# Check HTTPS certificate
openssl s_client -connect localhost:8080 -servername localhost

# Verify WebXR browser support
# Chrome: chrome://flags/#webxr-incubations
# Enable "WebXR Incubations"

# Check device permissions
# Camera, microphone, and motion sensors
```

#### 6. Database Connection Issues

```bash
# Check PostgreSQL
docker compose exec postgres psql -U nexus -d nexus -c "SELECT version();"

# Check Redis
docker compose exec redis redis-cli ping

# Reset databases
docker compose down
docker volume rm nexus_postgres_data nexus_redis_data
docker compose up -d
```

### Performance Issues

#### High CPU Usage
```bash
# Identify resource-heavy containers
docker stats

# Limit container resources
# Add to docker-compose.yml:
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

#### High Memory Usage
```bash
# Check memory usage by container
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Slow Media Streaming
```bash
# Check transcoding
docker compose logs jellyfin | grep transcode

# Verify hardware acceleration
docker compose exec jellyfin ls -la /dev/dri/

# Check network bandwidth
iperf3 -c target_server
```

---

## Maintenance & Updates

### 1. Regular Updates

```bash
#!/bin/bash
# update-nexus-platform.sh

echo "=== NEXUS Platform Update ==="

# Backup configurations
./scripts/backup-config.sh

# Pull latest images
docker compose pull

# Update containers with minimal downtime
docker compose up -d --remove-orphans

# Clean up old images
docker image prune -f

# Verify health after update
sleep 30
./health-check-ultimate.sh

echo "=== Update Complete ==="
```

### 2. Backup Strategy

```bash
#!/bin/bash
# backup-system.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/nexus-$BACKUP_DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup configurations
tar -czf "$BACKUP_DIR/config.tar.gz" ./config/

# Backup databases
docker compose exec postgres pg_dump -U nexus nexus > "$BACKUP_DIR/postgres.sql"
docker compose exec redis redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# Backup Docker compose files
cp docker-compose*.yml "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/"

# Cleanup old backups (keep 30 days)
find /backup -name "nexus-*" -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
```

### 3. Disaster Recovery

```bash
#!/bin/bash
# disaster-recovery.sh

BACKUP_PATH=$1
if [ -z "$BACKUP_PATH" ]; then
  echo "Usage: $0 /path/to/backup"
  exit 1
fi

# Stop all services
docker compose down

# Restore configurations
tar -xzf "$BACKUP_PATH/config.tar.gz" ./

# Restore databases
docker compose up -d postgres redis
sleep 10
docker compose exec postgres psql -U nexus -d nexus < "$BACKUP_PATH/postgres.sql"
docker compose exec redis redis-cli --pipe < "$BACKUP_PATH/redis.rdb"

# Restore environment
cp "$BACKUP_PATH/.env" ./
cp "$BACKUP_PATH/docker-compose"*.yml ./

# Start all services
docker compose up -d

echo "Disaster recovery completed"
```

### 4. Log Management

```bash
# Configure log rotation
cat > /etc/logrotate.d/nexus-platform << EOF
/var/lib/docker/containers/*/*-json.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0644 root root
    postrotate
        docker kill -s USR1 \$(docker ps -q) 2>/dev/null || true
    endscript
}
EOF

# Clean old logs
docker system prune -f --volumes

# Export logs for analysis
docker compose logs --since="24h" > nexus-logs-$(date +%Y%m%d).log
```

### 5. Security Updates

```bash
#!/bin/bash
# security-update.sh

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker
curl -fsSL https://get.docker.com | sh

# Scan for vulnerabilities
docker scout cves

# Update security configurations
./scripts/security-hardening.sh

# Restart services
docker compose restart

echo "Security update completed"
```

---

## Advanced Deployment Scenarios

### 1. Multi-Node Deployment

```yaml
# docker-swarm.yml
version: '3.8'

services:
  jellyfin:
    image: jellyfin/jellyfin:latest
    deploy:
      replicas: 2
      placement:
        constraints: [node.labels.type == media]
      resources:
        limits:
          cpus: '4'
          memory: 8G
    networks:
      - media-cluster

  ai-orchestrator:
    image: nexus/ai-orchestrator:latest
    deploy:
      replicas: 3
      placement:
        constraints: [node.labels.gpu == true]
      resources:
        reservations:
          generic_resources:
            - discrete_resource_spec:
                kind: 'NVIDIA-GPU'
                value: 1
    networks:
      - ai-cluster

networks:
  media-cluster:
    driver: overlay
  ai-cluster:
    driver: overlay
```

### 2. Kubernetes Deployment

```yaml
# k8s/nexus-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-jellyfin
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nexus-jellyfin
  template:
    metadata:
      labels:
        app: nexus-jellyfin
    spec:
      containers:
      - name: jellyfin
        image: jellyfin/jellyfin:latest
        ports:
        - containerPort: 8096
        volumeMounts:
        - name: media-storage
          mountPath: /media
        - name: config-storage
          mountPath: /config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: media-storage
        persistentVolumeClaim:
          claimName: nexus-media-pvc
      - name: config-storage
        persistentVolumeClaim:
          claimName: nexus-config-pvc
```

### 3. Cloud Deployment (AWS/Azure/GCP)

```bash
# AWS deployment with Terraform
terraform init
terraform plan -var="instance_type=c5.4xlarge" -var="gpu_instance_type=p3.2xlarge"
terraform apply

# Configure auto-scaling
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name nexus-asg \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3
```

---

This comprehensive deployment guide covers all aspects of setting up and maintaining the NEXUS Media Server Platform. For additional support or specific deployment scenarios, please refer to the community documentation or open an issue on the project repository.

**Last Updated**: January 2025  
**Guide Version**: 2.1  
**Platform Compatibility**: NEXUS 2025.1+