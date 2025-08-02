#!/bin/bash

# Fix Port 80 Conflict and Test Authentication
# Uses alternative ports to avoid conflicts

set -e

echo "🔧 Fixing Port 80 Conflict..."

# 1. Stop the test environment that failed
echo "🛑 Stopping failed test environment..."
docker-compose -f docker-compose-auth-test.yml down 2>/dev/null || true

# 2. Find what's using port 80
echo "🔍 Checking what's using port 80..."
lsof -i :80 || echo "No processes found on port 80"

# 3. Create alternative port configuration
echo "📋 Creating alternative port configuration..."
cat > docker-compose-auth-test-alt.yml << 'EOF'
version: "3.8"

services:
  # Reverse Proxy on alternative ports
  traefik:
    image: traefik:v3.1.4
    container_name: traefik-auth-alt
    command:
      - --api=true
      - --api.dashboard=true
      - --providers.docker=true
      - --providers.docker.exposedbydefault=false
      - --entrypoints.web.address=:8000
      - --entrypoints.websecure.address=:8443
      - --entrypoints.traefik.address=:8080
      - --log.level=DEBUG
    ports:
      - "8000:8000"  # Alternative to port 80
      - "8443:8443"  # Alternative to port 443
      - "8081:8080"  # Alternative dashboard port
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - traefik_network
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.traefik-api.rule=Host(`traefik.localhost`)"
      - "traefik.http.routers.traefik-api.service=api@internal"
      - "traefik.http.services.traefik-api.loadbalancer.server.port=8080"

  # Authentication Service
  authelia:
    image: authelia/authelia:4.38
    container_name: authelia-auth-alt
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
      - "traefik.http.middlewares.authelia.forwardauth.address=http://authelia:9091/api/verify?rd=http://auth.localhost:8000"
      - "traefik.http.middlewares.authelia.forwardauth.trustForwardHeader=true"
      - "traefik.http.middlewares.authelia.forwardauth.authResponseHeaders=Remote-User,Remote-Groups,Remote-Name,Remote-Email"
    restart: unless-stopped

  # Test service with auth
  whoami:
    image: traefik/whoami
    container_name: whoami-test-alt
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

# 4. Start the alternative port environment
echo "🚀 Starting authentication test environment on alternative ports..."
docker-compose -f docker-compose-auth-test-alt.yml up -d

# 5. Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 15

# 6. Test connectivity
echo "🧪 Testing connectivity..."
echo "Testing Traefik dashboard on port 8081..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8081 || echo "Traefik dashboard test failed"

echo "Testing Authelia direct on port 9091..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:9091 || echo "Authelia direct test failed"

echo "Testing auth.localhost through Traefik on port 8000..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://auth.localhost:8000 || echo "Auth routing test failed"

# 7. Show logs for debugging
echo ""
echo "📋 Container Status:"
docker-compose -f docker-compose-auth-test-alt.yml ps

echo ""
echo "🎉 Authentication test environment ready on alternative ports!"
echo ""
echo "📋 Test URLs (using alternative ports):"
echo "   Traefik Dashboard: http://localhost:8081"
echo "   Authelia Direct: http://localhost:9091"
echo "   Authelia via Traefik: http://auth.localhost:8000"
echo "   Protected Test Service: http://whoami.localhost:8000"
echo ""
echo "🔐 Login Credentials:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "🔍 Debug commands if needed:"
echo "   docker logs traefik-auth-alt"
echo "   docker logs authelia-auth-alt"
echo "   curl -v http://auth.localhost:8000"
echo ""
echo "⚠️  Note: Using port 8000 instead of 80 due to port conflict"
echo "   In production, you'd need to stop the service using port 80"
echo ""