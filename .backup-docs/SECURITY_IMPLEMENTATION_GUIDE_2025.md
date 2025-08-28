# Security Implementation Guide: 5 Critical Improvements

This guide provides step-by-step instructions to implement the 5 critical security improvements identified in the security analysis.

## Prerequisites

```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Create backup of current configuration
cp docker-compose.yml docker-compose.yml.backup
cp .env .env.backup
```

## Improvement 1: Implement Reverse Proxy with Authentication

### Step 1.1: Create Authelia Configuration

```bash
# Create Authelia directory structure
mkdir -p config/authelia
```

Create `config/authelia/configuration.yml`:
```yaml
# Authelia configuration for media server
theme: dark
server:
  host: 0.0.0.0
  port: 9091

log:
  level: info

totp:
  issuer: MediaServer
  period: 30
  skew: 1

authentication_backend:
  file:
    path: /config/users_database.yml
    password:
      algorithm: argon2id
      iterations: 3
      memory: 65536
      parallelism: 4
      key_length: 32
      salt_length: 16

access_control:
  default_policy: deny
  rules:
    # Public access to Overseerr for requests
    - domain: request.media.local
      policy: bypass
    
    # Two-factor for admin services
    - domain: 
        - sonarr.media.local
        - radarr.media.local
        - qbittorrent.media.local
      policy: two_factor
    
    # One-factor for media consumption
    - domain: jellyfin.media.local
      policy: one_factor

session:
  name: authelia_session
  secret: insecure_session_secret # Change this!
  expiration: 1h
  inactivity: 15m
  domain: media.local

regulation:
  max_retries: 3
  find_time: 2m
  ban_time: 5m

storage:
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt
```

Create `config/authelia/users_database.yml`:
```yaml
users:
  admin:
    displayname: "Admin User"
    password: "$argon2id$v=19$m=65536,t=3,p=4$BpLnfgDsc2WD8F2q$o/vzA4myCqZZ36bUGsDY//8mKUYNZZaR0t4MFFSs+iM"
    email: admin@media.local
    groups:
      - admins
      - users
  
  user:
    displayname: "Media User"
    password: "$argon2id$v=19$m=65536,t=3,p=4$BpLnfgDsc2WD8F2q$o/vzA4myCqZZ36bUGsDY//8mKUYNZZaR0t4MFFSs+iM"
    email: user@media.local
    groups:
      - users
```

### Step 1.2: Create Traefik Configuration

Create `config/traefik/traefik.yml`:
```yaml
global:
  checkNewVersion: false
  sendAnonymousUsage: false

api:
  dashboard: true
  debug: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: proxy
  file:
    directory: /etc/traefik/dynamic
    watch: true

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@media.local
      storage: /letsencrypt/acme.json
      httpChallenge:
        entryPoint: web
```

## Improvement 2: Secure API Keys with Environment Variables

### Step 2.1: Generate Secure API Keys

```bash
#!/bin/bash
# generate-api-keys.sh

# Generate secure random API keys
generate_key() {
    openssl rand -hex 32
}

# Create .env.secure file
cat > .env.secure << EOF
# Secure API Keys (Generated: $(date))
SONARR_API_KEY=$(generate_key)
RADARR_API_KEY=$(generate_key)
PROWLARR_API_KEY=$(generate_key)
LIDARR_API_KEY=$(generate_key)
READARR_API_KEY=$(generate_key)
BAZARR_API_KEY=$(generate_key)
JELLYFIN_API_KEY=$(generate_key)
OVERSEERR_API_KEY=$(generate_key)
TAUTULLI_API_KEY=$(generate_key)

# Authentication Secrets
AUTHELIA_JWT_SECRET=$(generate_key)
AUTHELIA_SESSION_SECRET=$(generate_key)
AUTHELIA_STORAGE_ENCRYPTION_KEY=$(generate_key)

# Database Passwords
POSTGRES_PASSWORD=$(generate_key)
REDIS_PASSWORD=$(generate_key)
MYSQL_ROOT_PASSWORD=$(generate_key)
MYSQL_PASSWORD=$(generate_key)

# Admin Passwords
GRAFANA_ADMIN_PASSWORD=$(generate_key)
PORTAINER_ADMIN_PASSWORD=$(generate_key)
PHOTOPRISM_ADMIN_PASSWORD=$(generate_key)
EOF

# Set proper permissions
chmod 600 .env.secure
echo "Secure API keys generated in .env.secure"
```

### Step 2.2: Update Docker Compose for Environment Variables

Create `docker-compose.secure.yml`:
```yaml
version: '3.9'

services:
  sonarr:
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
      - SONARR__API_KEY=${SONARR_API_KEY}
    volumes:
      - ./config/sonarr:/config
      - ${MEDIA_ROOT:-./data/media}/tv:/tv
      - ${DOWNLOADS_ROOT:-./data/downloads}:/downloads
    networks:
      - media_backend
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.sonarr.entrypoints=websecure"
      - "traefik.http.routers.sonarr.rule=Host(`sonarr.${DOMAIN}`)"
      - "traefik.http.routers.sonarr.middlewares=authelia@docker"
    restart: unless-stopped
```

## Improvement 3: Network Segmentation

### Step 3.1: Create Segmented Networks

```yaml
# Add to docker-compose.secure.yml
networks:
  proxy:
    name: proxy
    driver: bridge
  
  media_frontend:
    name: media_frontend
    driver: bridge
  
  media_backend:
    name: media_backend
    driver: bridge
    internal: true
  
  media_download:
    name: media_download
    driver: bridge
    internal: true
  
  monitoring:
    name: monitoring
    driver: bridge
    internal: true
```

### Step 3.2: Assign Services to Networks

```yaml
services:
  # Frontend services (user-facing)
  traefik:
    networks:
      - proxy
      - media_frontend
  
  authelia:
    networks:
      - proxy
      - media_backend
  
  jellyfin:
    networks:
      - media_frontend
      - media_backend
  
  overseerr:
    networks:
      - media_frontend
      - media_backend
  
  # Backend services (internal only)
  sonarr:
    networks:
      - media_backend
      - media_download
  
  radarr:
    networks:
      - media_backend
      - media_download
  
  # Download services (isolated)
  qbittorrent:
    networks:
      - media_download
  
  # Monitoring (separate network)
  prometheus:
    networks:
      - monitoring
```

## Improvement 4: Container Hardening

### Step 4.1: Create Security Policy Template

Create `docker-compose.security.yml`:
```yaml
version: '3.9'

# Security defaults for all services
x-security-common: &security-common
  security_opt:
    - no-new-privileges:true
    - seccomp:unconfined
    - apparmor:docker-default
  cap_drop:
    - ALL
  read_only: true
  restart: unless-stopped
  healthcheck:
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 40s

# Media service security settings
x-media-service: &media-service
  <<: *security-common
  cap_add:
    - CHOWN
    - SETUID
    - SETGID
    - DAC_OVERRIDE
  tmpfs:
    - /tmp:noexec,nosuid,size=100M
    - /var/run:noexec,nosuid,size=10M

services:
  jellyfin:
    <<: *media-service
    image: jellyfin/jellyfin:10.8.13
    read_only: false # Required for transcoding
    tmpfs:
      - /tmp:noexec,nosuid,size=1G
      - /var/run:noexec,nosuid,size=10M
      - /cache:noexec,nosuid,size=2G
    
  sonarr:
    <<: *media-service
    image: lscr.io/linuxserver/sonarr:4.0.0
    volumes:
      - type: bind
        source: ./config/sonarr
        target: /config
        read_only: false
      - type: bind
        source: ${MEDIA_ROOT}/tv
        target: /tv
        read_only: false
      - type: bind
        source: ${DOWNLOADS_ROOT}
        target: /downloads
        read_only: true
```

### Step 4.2: Implement Resource Limits

```yaml
services:
  jellyfin:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 1G
    
  qbittorrent:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## Improvement 5: Security Monitoring

### Step 5.1: Deploy Security Monitoring Stack

Create `docker-compose.monitoring.yml`:
```yaml
version: '3.9'

services:
  # Runtime security monitoring
  falco:
    image: falcosecurity/falco:latest
    container_name: falco
    privileged: true
    volumes:
      - /var/run/docker.sock:/host/var/run/docker.sock
      - /dev:/host/dev
      - /proc:/host/proc:ro
      - /boot:/host/boot:ro
      - /lib/modules:/host/lib/modules:ro
      - /usr:/host/usr:ro
      - ./config/falco:/etc/falco
    networks:
      - monitoring
    restart: unless-stopped

  # Intrusion prevention
  crowdsec:
    image: crowdsecurity/crowdsec:latest
    container_name: crowdsec
    environment:
      - COLLECTIONS=crowdsecurity/linux crowdsecurity/traefik crowdsecurity/http-cve
    volumes:
      - ./config/crowdsec:/etc/crowdsec
      - ./logs:/logs:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - monitoring
    restart: unless-stopped

  # Vulnerability scanning
  trivy:
    image: aquasec/trivy:latest
    container_name: trivy-server
    command: server --listen 0.0.0.0:8080
    volumes:
      - trivy-cache:/root/.cache
    networks:
      - monitoring
    restart: unless-stopped

  # Log aggregation
  loki:
    image: grafana/loki:latest
    container_name: loki
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./config/loki:/etc/loki
      - loki-data:/loki
    networks:
      - monitoring
    restart: unless-stopped

  # Log shipping
  promtail:
    image: grafana/promtail:latest
    container_name: promtail
    command: -config.file=/etc/promtail/config.yml
    volumes:
      - ./config/promtail:/etc/promtail
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    networks:
      - monitoring
    restart: unless-stopped

volumes:
  trivy-cache:
  loki-data:
```

### Step 5.2: Create Monitoring Dashboard

Create `config/grafana/dashboards/security.json`:
```json
{
  "dashboard": {
    "title": "Security Monitoring",
    "panels": [
      {
        "title": "Failed Login Attempts",
        "targets": [
          {
            "expr": "rate(authelia_authentication_attempts_total{success=\"false\"}[5m])"
          }
        ]
      },
      {
        "title": "Container Security Events",
        "targets": [
          {
            "expr": "falco_events_total"
          }
        ]
      },
      {
        "title": "Blocked IPs",
        "targets": [
          {
            "expr": "crowdsec_blocked_ips_total"
          }
        ]
      }
    ]
  }
}
```

## Deployment Script

Create `deploy-secure.sh`:
```bash
#!/bin/bash
set -euo pipefail

echo "=== Media Server Security Deployment ==="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Generate secure keys if not exists
if [ ! -f .env.secure ]; then
    echo "Generating secure API keys..."
    bash generate-api-keys.sh
fi

# Create required directories
echo "Creating directory structure..."
mkdir -p config/{authelia,traefik/dynamic,falco,crowdsec,loki,promtail,grafana/dashboards}
mkdir -p data/{media/{movies,tv,music},downloads}
mkdir -p logs
mkdir -p letsencrypt

# Set permissions
echo "Setting permissions..."
chmod 700 config/authelia
chmod 600 .env.secure

# Deploy monitoring first
echo "Deploying security monitoring..."
docker-compose -f docker-compose.monitoring.yml up -d

# Deploy main stack with security
echo "Deploying secure media stack..."
docker-compose -f docker-compose.secure.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 30

# Run initial security scan
echo "Running security scan..."
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v trivy-cache:/root/.cache/ \
  aquasec/trivy image \
  --severity HIGH,CRITICAL \
  --format table \
  $(docker-compose -f docker-compose.secure.yml config | grep 'image:' | awk '{print $2}' | sort -u)

echo "=== Deployment Complete ==="
echo "Access your media server at: https://media.local"
echo "Default credentials: admin / password (change immediately!)"
echo ""
echo "Security monitoring available at:"
echo "- Grafana: https://grafana.media.local"
echo "- Falco: Check logs with: docker logs falco"
echo "- CrowdSec: https://app.crowdsec.net"
```

## Post-Deployment Security Checklist

- [ ] Change all default passwords
- [ ] Enable 2FA in Authelia for admin accounts  
- [ ] Configure SSL certificates (Let's Encrypt)
- [ ] Review and adjust firewall rules
- [ ] Set up automated backups
- [ ] Configure log retention policies
- [ ] Schedule regular vulnerability scans
- [ ] Document emergency procedures
- [ ] Test disaster recovery plan
- [ ] Monitor security dashboards

## Maintenance Commands

```bash
# Check security status
docker-compose -f docker-compose.monitoring.yml logs falco | grep "Warning\|Error"

# Update all images
docker-compose pull && docker-compose up -d

# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz config/ .env.secure

# Security audit
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --severity HIGH,CRITICAL \
  $(docker ps --format "{{.Image}}" | sort -u)
```

This implementation guide provides practical, step-by-step instructions to secure your media server following 2025 best practices.