# 🛡️ Security Remediation Guide - NEXUS Media Server 2025

**Priority**: CRITICAL  
**Timeline**: Implement within 48-72 hours  
**Risk Level**: HIGH (Current deployment vulnerable)

---

## 🚨 CRITICAL SECURITY FIXES (Implement Immediately)

### 1. **Docker Socket Protection** [SEVERITY: CRITICAL]

**Current Risk**: Direct Docker socket mounting gives containers root access to host

#### Fix Implementation:
```yaml
# docker-compose.security.yml
services:
  docker-socket-proxy:
    image: tecnativa/docker-socket-proxy:latest
    container_name: docker-socket-proxy
    restart: unless-stopped
    environment:
      - CONTAINERS=1
      - NETWORKS=1
      - SERVICES=0
      - TASKS=0
      - POST=0
      - DELETE=0
      - EXEC=0
      - BUILD=0
      - COMMIT=0
      - CONFIGS=0
      - DISTRIBUTION=0
      - IMAGES=0
      - INFO=0
      - NODES=0
      - PLUGINS=0
      - SECRETS=0
      - SESSIONS=0
      - SWARM=0
      - SYSTEM=0
      - VOLUMES=0
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - socket_proxy
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    cap_drop:
      - ALL

  # Update services that need Docker access
  traefik:
    # Remove: - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      - DOCKER_HOST=tcp://docker-socket-proxy:2375
    depends_on:
      - docker-socket-proxy
    networks:
      - socket_proxy
      - frontend
```

### 2. **Secrets Management** [SEVERITY: HIGH]

**Current Risk**: Hardcoded passwords and API keys in configuration

#### Fix Implementation:
```bash
# generate-secrets.sh
#!/bin/bash
set -euo pipefail

# Create secrets directory
mkdir -p ./secrets
chmod 700 ./secrets

# Generate strong passwords
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Generate all required secrets
echo "$(generate_password)" > ./secrets/postgres_password
echo "$(generate_password)" > ./secrets/redis_password
echo "$(generate_password)" > ./secrets/jellyfin_api_key
echo "$(generate_password)" > ./secrets/traefik_dashboard_password

# Generate authentication for Traefik
htpasswd -nb admin "$(cat ./secrets/traefik_dashboard_password)" > ./secrets/traefik_users

# Set proper permissions
chmod 600 ./secrets/*
```

#### Docker Compose Integration:
```yaml
# docker-compose.security.yml
secrets:
  postgres_password:
    file: ./secrets/postgres_password
  redis_password:
    file: ./secrets/redis_password
  jellyfin_api_key:
    file: ./secrets/jellyfin_api_key
  traefik_users:
    file: ./secrets/traefik_users

services:
  postgres:
    secrets:
      - postgres_password
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
```

### 3. **Network Segmentation** [SEVERITY: HIGH]

**Current Risk**: Insufficient network isolation allows lateral movement

#### Fix Implementation:
```yaml
# docker-compose.security.yml
networks:
  frontend:
    driver: bridge
    internal: false
  backend:
    driver: bridge
    internal: true
  database:
    driver: bridge
    internal: true
  download:
    driver: bridge
    internal: true
  monitoring:
    driver: bridge
    internal: true
  socket_proxy:
    driver: bridge
    internal: true

services:
  # Frontend services (exposed to internet)
  traefik:
    networks:
      - frontend
      - socket_proxy
  
  jellyfin:
    networks:
      - frontend
      - backend
  
  # Backend services (internal only)
  sonarr:
    networks:
      - backend
      - download
  
  # Database services (most restricted)
  postgres:
    networks:
      - database
```

### 4. **Container Security Hardening** [SEVERITY: HIGH]

**Apply to ALL containers**:
```yaml
services:
  service_name:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETUID
      - SETGID
    read_only: true  # Where possible
    tmpfs:
      - /tmp
      - /var/run
    ulimits:
      nproc: 65535
      nofile:
        soft: 4096
        hard: 8192
```

### 5. **Authentication Layer - Authelia** [SEVERITY: CRITICAL]

#### Quick Deployment:
```yaml
# docker-compose.security.yml
services:
  authelia:
    image: authelia/authelia:latest
    container_name: authelia
    secrets:
      - authelia_jwt_secret
      - authelia_session_secret
      - authelia_storage_encryption_key
    environment:
      - TZ=${TZ:-America/New_York}
    volumes:
      - ./config/authelia:/config
    networks:
      - frontend
      - backend
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.authelia.rule=Host(`auth.${DOMAIN}`)"
      - "traefik.http.routers.authelia.entrypoints=websecure"
      - "traefik.http.routers.authelia.tls.certresolver=letsencrypt"
      - "traefik.http.middlewares.authelia.forwardauth.address=http://authelia:9091/api/verify?rd=https://auth.${DOMAIN}"
      - "traefik.http.middlewares.authelia.forwardauth.trustForwardHeader=true"
    restart: unless-stopped
```

#### Minimal Authelia Configuration:
```yaml
# config/authelia/configuration.yml
server:
  host: 0.0.0.0
  port: 9091

log:
  level: info

theme: dark

jwt_secret: ${AUTHELIA_JWT_SECRET}
default_redirection_url: https://${DOMAIN}

totp:
  issuer: authelia.com
  period: 30
  skew: 1

authentication_backend:
  file:
    path: /config/users_database.yml
    password:
      algorithm: argon2id
      iterations: 1
      key_length: 32
      salt_length: 16
      memory: 512
      parallelism: 8

access_control:
  default_policy: deny
  rules:
    - domain: ${DOMAIN}
      policy: one_factor
    - domain: "*.${DOMAIN}"
      policy: two_factor

session:
  name: authelia_session
  secret: ${AUTHELIA_SESSION_SECRET}
  expiration: 1h
  inactivity: 5m
  domain: ${DOMAIN}

regulation:
  max_retries: 3
  find_time: 2m
  ban_time: 5m

storage:
  encryption_key: ${AUTHELIA_STORAGE_ENCRYPTION_KEY}
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt
```

---

## 🔐 IMMEDIATE SECURITY CHECKLIST

### Within 24 Hours:
- [ ] Implement Docker socket proxy
- [ ] Generate and apply all secrets
- [ ] Enable firewall (UFW/iptables)
- [ ] Change all default passwords
- [ ] Disable unnecessary ports

### Within 48 Hours:
- [ ] Deploy Authelia authentication
- [ ] Configure fail2ban
- [ ] Enable audit logging
- [ ] Set up automated backups
- [ ] Test disaster recovery

### Within 72 Hours:
- [ ] Complete network segmentation
- [ ] Apply container hardening
- [ ] Configure monitoring alerts
- [ ] Document security procedures
- [ ] Perform penetration testing

---

## 🚀 QUICK DEPLOYMENT SCRIPT

```bash
#!/bin/bash
# secure-deploy.sh

set -euo pipefail

echo "🛡️ Securing NEXUS Media Server..."

# Generate secrets
./generate-secrets.sh

# Apply security compose
docker-compose -f docker-compose.yml -f docker-compose.security.yml up -d

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Install fail2ban
sudo apt-get update && sudo apt-get install -y fail2ban

# Configure fail2ban for Docker
cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[docker-auth]
enabled = true
filter = docker-auth
logpath = /var/lib/docker/containers/*/*-json.log
port = http,https
EOF

echo "✅ Security measures applied!"
echo "⚠️  Remember to:"
echo "   - Update DNS records"
echo "   - Configure Authelia users"
echo "   - Test all services"
echo "   - Monitor logs"
```

---

## 📊 SECURITY VALIDATION

After implementing fixes, validate:

```bash
# Check exposed ports
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "0.0.0.0|:::"

# Verify network isolation
docker network ls
docker inspect <network_name>

# Check for secrets in environment
docker inspect <container> | grep -i password

# Test authentication
curl -I https://jellyfin.${DOMAIN}  # Should redirect to auth

# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image <image_name>
```

---

**⚠️ CRITICAL**: These security measures are the MINIMUM required for production. Additional hardening recommended for public-facing deployments.