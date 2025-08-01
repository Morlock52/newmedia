# Security Remediation Guide - Media Server Project

## 🚨 CRITICAL: Immediate Actions Required

### Step 1: Remove Hardcoded API Keys

#### 1.1 Update docker-compose.yml

Replace hardcoded API keys with environment variable references:

**BEFORE (INSECURE):**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "-H", "X-Api-Key: 79eecf2b23f34760b91cfcbf97189dd0", "http://localhost:8989/api/v3/system/status"]
```

**AFTER (SECURE):**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "-H", "X-Api-Key: ${SONARR_API_KEY}", "http://localhost:8989/api/v3/system/status"]
```

#### 1.2 Update homepage-config/services.yaml

**BEFORE (INSECURE):**
```yaml
- Sonarr:
    widget:
      type: sonarr
      url: http://sonarr:8989
      key: "79eecf2b23f34760b91cfcbf97189dd0"
```

**AFTER (SECURE):**
```yaml
- Sonarr:
    widget:
      type: sonarr
      url: http://sonarr:8989
      key: "{{HOMEPAGE_VAR_SONARR_API_KEY}}"
```

### Step 2: Regenerate All API Keys

1. **Access each service's web interface:**
   - Sonarr: http://localhost:8989/settings/general
   - Radarr: http://localhost:7878/settings/general
   - Prowlarr: http://localhost:9696/settings/general

2. **Regenerate API keys in each service:**
   - Navigate to Settings → General → Security
   - Click "Regenerate" next to API Key
   - Copy the new key

3. **Update .env file with new keys:**
```bash
# API Keys (regenerated and secure)
SONARR_API_KEY=your_new_sonarr_key_here
RADARR_API_KEY=your_new_radarr_key_here
PROWLARR_API_KEY=your_new_prowlarr_key_here
OVERSEERR_API_KEY=your_new_overseerr_key_here

# Homepage Variables
HOMEPAGE_VAR_SONARR_API_KEY=your_new_sonarr_key_here
HOMEPAGE_VAR_RADARR_API_KEY=your_new_radarr_key_here
HOMEPAGE_VAR_PROWLARR_API_KEY=your_new_prowlarr_key_here
HOMEPAGE_VAR_OVERSEERR_API_KEY=your_new_overseerr_key_here
```

### Step 3: Clean Git History

If these files were committed to git:

```bash
# Check if secrets are in git history
git log -p | grep -E "79eecf2b23f34760b91cfcbf97189dd0|1c0fe63736a04e6394dacb3aa1160b1c|5a35dd23f90c4d2bb69caa1eb0e1c534"

# If found, use BFG Repo-Cleaner to remove them
# Install BFG first: brew install bfg (on macOS)
bfg --delete-files docker-compose.yml
bfg --delete-files services.yaml
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# Or use git-filter-branch (more complex but built-in)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch docker-compose.yml' \
  --prune-empty --tag-name-filter cat -- --all
```

## 🔐 Implementing Proper Authentication

### Step 4: Add Authentication Middleware

#### 4.1 Create users.htpasswd file

```bash
# Install htpasswd if not available
sudo apt-get install apache2-utils  # Debian/Ubuntu
# or
brew install httpd  # macOS

# Generate password hash
htpasswd -nb admin your_secure_password > ./config/traefik/users.htpasswd

# Add additional users
htpasswd -b ./config/traefik/users.htpasswd user2 password2
```

#### 4.2 Update docker-compose.yml with auth labels

Add authentication to sensitive services:

```yaml
sonarr:
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.sonarr.rule=Host(`sonarr.${DOMAIN}`)"
    - "traefik.http.routers.sonarr.entrypoints=websecure"
    - "traefik.http.routers.sonarr.tls.certresolver=cloudflare"
    - "traefik.http.services.sonarr.loadbalancer.server.port=8989"
    - "traefik.http.routers.sonarr.middlewares=auth,security-headers,rate-limit"
```

### Step 5: Implement Authelia (Advanced)

#### 5.1 Create Authelia configuration

```yaml
# ./config/authelia/configuration.yml
server:
  host: 0.0.0.0
  port: 9091

log:
  level: info

theme: dark

jwt_secret: ${AUTHELIA_JWT_SECRET}
default_redirection_url: https://home.${DOMAIN}

totp:
  issuer: ${DOMAIN}
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
      memory: 65536
      parallelism: 8

access_control:
  default_policy: deny
  rules:
    # Public access
    - domain: requests.${DOMAIN}
      policy: bypass
    
    # Admin only
    - domain: 
        - sonarr.${DOMAIN}
        - radarr.${DOMAIN}
        - prowlarr.${DOMAIN}
      policy: two_factor
      subject: "group:admin"
    
    # Users
    - domain: 
        - jellyfin.${DOMAIN}
        - home.${DOMAIN}
      policy: one_factor

session:
  name: authelia_session
  secret: ${AUTHELIA_SESSION_SECRET}
  expiration: 1h
  inactivity: 5m
  remember_me_duration: 1M
  domain: ${DOMAIN}

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

#### 5.2 Add Authelia to docker-compose.yml

```yaml
authelia:
  image: authelia/authelia:latest
  container_name: authelia
  environment:
    - TZ=${TZ}
    - AUTHELIA_JWT_SECRET=${AUTHELIA_JWT_SECRET}
    - AUTHELIA_SESSION_SECRET=${AUTHELIA_SESSION_SECRET}
    - AUTHELIA_STORAGE_ENCRYPTION_KEY=${AUTHELIA_STORAGE_ENCRYPTION_KEY}
  volumes:
    - ./config/authelia:/config
  networks:
    - media_network
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.authelia.rule=Host(`auth.${DOMAIN}`)"
    - "traefik.http.routers.authelia.entrypoints=websecure"
    - "traefik.http.routers.authelia.tls.certresolver=cloudflare"
    - "traefik.http.middlewares.authelia.forwardauth.address=http://authelia:9091/api/verify?rd=https://auth.${DOMAIN}"
    - "traefik.http.middlewares.authelia.forwardauth.trustForwardHeader=true"
    - "traefik.http.middlewares.authelia.forwardauth.authResponseHeaders=Remote-User,Remote-Groups,Remote-Name,Remote-Email"
  restart: unless-stopped
```

## 🔒 Network Security Improvements

### Step 6: Implement Network Segmentation

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  
  backend:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.1.0/24
  
  vpn:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.2.0/24

services:
  traefik:
    networks:
      - frontend
      - backend
  
  jellyfin:
    networks:
      - frontend
      - backend
  
  sonarr:
    networks:
      - backend
  
  qbittorrent:
    networks:
      - vpn
```

## 🛡️ Container Security Hardening

### Step 7: Apply Security Options

Update each service with security best practices:

```yaml
sonarr:
  image: lscr.io/linuxserver/sonarr:latest
  container_name: sonarr
  security_opt:
    - no-new-privileges:true
    - apparmor:docker-default
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - DAC_OVERRIDE
    - FOWNER
    - SETGID
    - SETUID
  read_only: true
  tmpfs:
    - /tmp
    - /app/tmp
  environment:
    - PUID=${PUID:-1000}
    - PGID=${PGID:-1000}
    - TZ=${TZ}
  volumes:
    - ./config/sonarr:/config
    - ./data:/data:ro
    - ./data/downloads:/data/downloads:rw
```

## 📊 Monitoring and Alerting

### Step 8: Implement Security Monitoring

#### 8.1 Add Prometheus monitoring

```yaml
prometheus:
  image: prom/prometheus:latest
  container_name: prometheus
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.path=/prometheus'
    - '--web.enable-lifecycle'
  volumes:
    - ./config/prometheus:/etc/prometheus
    - prometheus_data:/prometheus
  networks:
    - backend
  restart: unless-stopped

grafana:
  image: grafana/grafana:latest
  container_name: grafana
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    - GF_USERS_ALLOW_SIGN_UP=false
  volumes:
    - ./config/grafana:/etc/grafana/provisioning
    - grafana_data:/var/lib/grafana
  networks:
    - frontend
    - backend
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.grafana.rule=Host(`grafana.${DOMAIN}`)"
    - "traefik.http.routers.grafana.entrypoints=websecure"
    - "traefik.http.routers.grafana.tls.certresolver=cloudflare"
    - "traefik.http.routers.grafana.middlewares=authelia,security-headers"
```

## 🔑 Secrets Management

### Step 9: Implement Docker Secrets (Swarm mode)

```bash
# Initialize Docker Swarm
docker swarm init

# Create secrets
echo "your_sonarr_api_key" | docker secret create sonarr_api_key -
echo "your_radarr_api_key" | docker secret create radarr_api_key -
echo "your_prowlarr_api_key" | docker secret create prowlarr_api_key -

# Update docker-compose.yml to use secrets
version: "3.9"

secrets:
  sonarr_api_key:
    external: true
  radarr_api_key:
    external: true
  prowlarr_api_key:
    external: true

services:
  sonarr:
    secrets:
      - sonarr_api_key
    environment:
      - API_KEY_FILE=/run/secrets/sonarr_api_key
```

## 📋 Security Checklist

- [ ] Remove all hardcoded API keys from configuration files
- [ ] Regenerate all API keys in services
- [ ] Update .env file with new keys
- [ ] Clean git history of exposed secrets
- [ ] Implement authentication middleware
- [ ] Set up network segmentation
- [ ] Apply container security hardening
- [ ] Configure monitoring and alerting
- [ ] Test all services after changes
- [ ] Document new security procedures

## 🚀 Testing After Implementation

```bash
# Test service connectivity
docker-compose up -d
docker-compose ps

# Test authentication
curl -I https://sonarr.yourdomain.com  # Should redirect to auth

# Test API key usage
curl -H "X-Api-Key: $SONARR_API_KEY" http://localhost:8989/api/v3/system/status

# Monitor logs
docker-compose logs -f traefik
docker-compose logs -f authelia
```

## 📚 Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Traefik Security Documentation](https://doc.traefik.io/traefik/https/overview/)
- [Authelia Documentation](https://www.authelia.com/docs/)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---
*This remediation guide provides step-by-step instructions to address the critical security vulnerabilities identified in the security audit. Follow these steps in order and test thoroughly after each change.*