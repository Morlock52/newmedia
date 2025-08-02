#!/bin/bash

# Complete Authentication and Routing Fix
# Fixes 404 errors for auth.localhost

set -e

echo "🔧 Fixing Authentication Routing Issues..."

# 1. Add localhost entries to /etc/hosts
echo "📝 Adding localhost entries to /etc/hosts..."
sudo bash -c 'cat >> /etc/hosts << EOF
# Media Server Auth Routes
127.0.0.1    auth.localhost
127.0.0.1    traefik.localhost
127.0.0.1    sonarr.localhost
127.0.0.1    radarr.localhost
127.0.0.1    prowlarr.localhost
127.0.0.1    overseerr.localhost
127.0.0.1    jellyfin.localhost
127.0.0.1    homarr.localhost
EOF'

echo "✅ Localhost entries added to /etc/hosts"

# 2. Stop any running services
echo "🛑 Stopping existing services..."
docker-compose down || true

# 3. Create minimal working docker-compose with proper networking
echo "📋 Creating working docker-compose configuration..."
cat > docker-compose-auth-test.yml << 'EOF'
version: "3.8"

services:
  # Reverse Proxy
  traefik:
    image: traefik:v3.1.4
    container_name: traefik-auth
    command:
      - --api=true
      - --api.dashboard=true
      - --providers.docker=true
      - --providers.docker.exposedbydefault=false
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --entrypoints.traefik.address=:8080
      - --log.level=DEBUG
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - traefik_network
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.traefik-api.rule=Host(`traefik.localhost`)"
      - "traefik.http.routers.traefik-api.service=api@internal"

  # Authentication Service
  authelia:
    image: authelia/authelia:4.38
    container_name: authelia-auth
    volumes:
      - ../config/authelia:/config
    ports:
      - "9091:9091"
    networks:
      - traefik_network
    environment:
      - AUTHELIA_JWT_SECRET=your_jwt_secret_here_change_this_in_production
      - AUTHELIA_SESSION_SECRET=your_session_secret_here_change_this_in_production
      - AUTHELIA_STORAGE_ENCRYPTION_KEY=your_encryption_key_here_change_this_in_production
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.authelia.rule=Host(`auth.localhost`)"
      - "traefik.http.routers.authelia.entrypoints=web"
      - "traefik.http.services.authelia.loadbalancer.server.port=9091"
      - "traefik.http.middlewares.authelia.forwardauth.address=http://authelia:9091/api/verify?rd=http://auth.localhost"
      - "traefik.http.middlewares.authelia.forwardauth.trustForwardHeader=true"
      - "traefik.http.middlewares.authelia.forwardauth.authResponseHeaders=Remote-User,Remote-Groups,Remote-Name,Remote-Email"
    restart: unless-stopped

  # Test service with auth
  whoami:
    image: traefik/whoami
    container_name: whoami-test
    networks:
      - traefik_network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.whoami.rule=Host(`whoami.localhost`)"
      - "traefik.http.routers.whoami.entrypoints=web"
      - "traefik.http.routers.whoami.middlewares=authelia"

networks:
  traefik_network:
    driver: bridge

EOF

# 4. Create simplified Authelia config
echo "⚙️  Creating simplified Authelia configuration..."
mkdir -p ../config/authelia

cat > ../config/authelia/configuration.yml << 'EOF'
server:
  host: 0.0.0.0
  port: 9091

log:
  level: debug

theme: dark

jwt_secret: your_jwt_secret_here_change_this_in_production

default_redirection_url: http://auth.localhost

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
      salt_length: 16
      parallelism: 8
      memory: 64

access_control:
  default_policy: deny
  rules:
    - domain: auth.localhost
      policy: bypass
    - domain: traefik.localhost
      policy: bypass
    - domain: "*.localhost"
      policy: one_factor

session:
  name: authelia_session
  secret: your_session_secret_here_change_this_in_production
  expiration: 3600
  inactivity: 300
  domain: localhost

regulation:
  max_retries: 3
  find_time: 120
  ban_time: 300

storage:
  encryption_key: your_encryption_key_here_change_this_in_production
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt
EOF

# 5. Create user database with working credentials
cat > ../config/authelia/users_database.yml << 'EOF'
users:
  admin:
    displayname: "Administrator"
    # Password: admin123
    password: "$argon2id$v=19$m=65536,t=3,p=4$BpLnQJMGaB7vOUlxvTZSCg$QDZPX3SZPWLhp/YK3paDBgvyE8WKxJnV8lgJbUfLxdI"
    email: admin@localhost.com
    groups:
      - admins
EOF

echo "✅ Configuration files created"

# 6. Start the test environment
echo "🚀 Starting authentication test environment..."
docker-compose -f docker-compose-auth-test.yml up -d

# 7. Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# 8. Test connectivity
echo "🧪 Testing connectivity..."
echo "Testing Traefik dashboard..."
curl -s -o /dev/null -w "%{http_code}" http://traefik.localhost:8080 || echo "Traefik direct test"

echo "Testing Authelia service..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:9091 || echo "Authelia direct test"

echo ""
echo "🎉 Authentication test environment ready!"
echo ""
echo "📋 Test URLs:"
echo "   Traefik Dashboard: http://traefik.localhost"
echo "   Authelia Login: http://auth.localhost"
echo "   Test Service: http://whoami.localhost (requires auth)"
echo ""
echo "🔐 Login Credentials:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "🔍 Debug commands if issues persist:"
echo "   docker logs traefik-auth"
echo "   docker logs authelia-auth"
echo "   curl -v http://auth.localhost"
echo ""