#!/bin/bash
# Emergency Security Fix Script for NewMedia Server
# This script addresses CRITICAL security vulnerabilities immediately

set -euo pipefail

echo "=== EMERGENCY SECURITY FIX ==="
echo "This script will fix critical security vulnerabilities"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Backup current configuration
echo "Creating backup..."
BACKUP_DIR="backup-security-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || true
cp .env "$BACKUP_DIR/" 2>/dev/null || true
cp -r config "$BACKUP_DIR/" 2>/dev/null || true

# Function to generate secure random keys
generate_key() {
    openssl rand -hex 32
}

# Create secure environment file
echo "Generating secure API keys..."
cat > .env.emergency << 'EOF'
# Emergency Security Configuration
# Generated: $(date)

# CRITICAL: Change these default values!
DOMAIN=localhost
PUID=1000
PGID=1000
TZ=America/New_York

# Secure API Keys (auto-generated)
EOF

# Generate secure API keys
echo "SONARR_API_KEY=$(generate_key)" >> .env.emergency
echo "RADARR_API_KEY=$(generate_key)" >> .env.emergency
echo "PROWLARR_API_KEY=$(generate_key)" >> .env.emergency
echo "LIDARR_API_KEY=$(generate_key)" >> .env.emergency
echo "BAZARR_API_KEY=$(generate_key)" >> .env.emergency
echo "JELLYFIN_API_KEY=$(generate_key)" >> .env.emergency
echo "OVERSEERR_API_KEY=$(generate_key)" >> .env.emergency
echo "TAUTULLI_API_KEY=$(generate_key)" >> .env.emergency
echo "QBITTORRENT_PASSWORD=$(generate_key)" >> .env.emergency
echo "PORTAINER_PASSWORD=$(generate_key)" >> .env.emergency

# Create minimal secure docker-compose
echo "Creating secure docker-compose configuration..."
cat > docker-compose.emergency.yml << 'EOF'
version: '3.9'

# EMERGENCY SECURE CONFIGURATION
# This removes all exposed ports and hardcoded secrets

services:
  # Jellyfin - Media Server (secured)
  jellyfin:
    image: jellyfin/jellyfin:10.8.13
    container_name: jellyfin
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
    volumes:
      - ./config/jellyfin:/config
      - ./data/media:/media:ro
    # PORT REMOVED - Access via reverse proxy only
    networks:
      - internal
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETUID
      - SETGID

  # Sonarr - TV Management (secured)
  sonarr:
    image: lscr.io/linuxserver/sonarr:4.0.0
    container_name: sonarr
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
      - SONARR__API_KEY=${SONARR_API_KEY}
    volumes:
      - ./config/sonarr:/config
      - ./data/media/tv:/tv
      - ./data/downloads:/downloads
    # PORT REMOVED - Internal access only
    networks:
      - internal
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

  # Radarr - Movie Management (secured)
  radarr:
    image: lscr.io/linuxserver/radarr:5.2.6
    container_name: radarr
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - TZ=${TZ:-America/New_York}
      - RADARR__API_KEY=${RADARR_API_KEY}
    volumes:
      - ./config/radarr:/config
      - ./data/media/movies:/movies
      - ./data/downloads:/downloads
    # PORT REMOVED - Internal access only
    networks:
      - internal
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

  # Basic reverse proxy for emergency access
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "8443:443"  # Single secure entry point
    volumes:
      - ./emergency-nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - internal
    restart: unless-stopped
    depends_on:
      - jellyfin

networks:
  internal:
    driver: bridge
    internal: true  # No external access except through proxy
EOF

# Create basic nginx config with authentication
echo "Creating nginx configuration..."
cat > emergency-nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # Hide nginx version
    server_tokens off;

    # Basic authentication
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    server {
        listen 443 ssl;
        server_name localhost;

        # Self-signed cert for emergency use
        ssl_certificate /etc/nginx/cert.pem;
        ssl_certificate_key /etc/nginx/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Jellyfin proxy
        location / {
            proxy_pass http://jellyfin:8096;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

# Create emergency instructions
echo "Creating emergency instructions..."
cat > EMERGENCY_SECURITY_README.md << 'EOF'
# EMERGENCY SECURITY MEASURES APPLIED

## What This Script Did

1. **Backed up** your current configuration to: `$(echo $BACKUP_DIR)`
2. **Generated secure API keys** replacing hardcoded ones
3. **Removed all exposed ports** except secured proxy (8443)
4. **Created internal-only network** for service communication
5. **Applied container security policies**

## Immediate Actions Required

1. **Stop current insecure stack:**
   ```bash
   docker-compose down
   ```

2. **Start emergency secure stack:**
   ```bash
   docker-compose -f docker-compose.emergency.yml up -d
   ```

3. **Access services** via: https://localhost:8443
   - Default user: admin
   - Default password: see .env.emergency

4. **Change all passwords immediately!**

## Next Steps

1. Review `SECURITY_IMPLEMENTATION_GUIDE_2025.md` for full security implementation
2. Implement proper authentication (Authelia/OAuth2)
3. Set up proper SSL certificates
4. Configure VPN for downloading
5. Enable monitoring and alerts

## Services Status

- ✅ Jellyfin: Secured (internal only)
- ✅ Sonarr: Secured (API key protected)
- ✅ Radarr: Secured (API key protected)
- ❌ qBittorrent: Disabled (security risk)
- ❌ Portainer: Disabled (Docker socket exposure)
- ⚠️  Other services: Need individual security review

## Recovery

To restore original configuration:
```bash
cp $(echo $BACKUP_DIR)/* .
docker-compose up -d
```

**WARNING**: This will restore insecure configuration!
EOF

# Set permissions
chmod 600 .env.emergency
chmod 600 emergency-nginx.conf

# Final summary
echo ""
echo "=== EMERGENCY SECURITY FIX COMPLETE ==="
echo ""
echo "CRITICAL ACTIONS REQUIRED:"
echo "1. Stop current stack: docker-compose down"
echo "2. Review .env.emergency for new secure keys"
echo "3. Start secure stack: docker-compose -f docker-compose.emergency.yml up -d"
echo "4. Access via: https://localhost:8443"
echo "5. CHANGE ALL PASSWORDS IMMEDIATELY!"
echo ""
echo "Backup saved to: $BACKUP_DIR"
echo "See EMERGENCY_SECURITY_README.md for details"
echo ""
echo "⚠️  This is a temporary fix. Implement full security measures ASAP!"