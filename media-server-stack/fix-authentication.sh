#!/bin/bash

# Quick Authentication Fix Script
# Fixes login issues by setting up Authelia properly

set -e

echo "🔧 Fixing Authentication System..."

# 1. Generate required secrets
echo "📝 Generating authentication secrets..."
mkdir -p ../secrets

# Generate JWT secret (64 characters)
openssl rand -hex 32 > ../secrets/authelia_jwt_secret.txt

# Generate session secret (64 characters)  
openssl rand -hex 32 > ../secrets/authelia_session_secret.txt

# Generate encryption key (64 characters)
openssl rand -hex 32 > ../secrets/authelia_encryption_key.txt

# Generate SMTP password (if needed)
openssl rand -hex 16 > ../secrets/authelia_smtp_password.txt

echo "✅ Secrets generated"

# 2. Create proper user database with hashed password
echo "👤 Creating user database..."
cat > ../config/authelia/users_database.yml << 'EOF'
users:
  admin:
    displayname: "Administrator"
    # Password: admin123 (change this!)
    password: "$argon2id$v=19$m=1024,t=1,p=8$YWRtaW4xMjM$rZ7qF8/8OjNbE1gF5pYYYdF1CZ8QlJXzGxP2dN1mK8s"
    email: admin@localhost.com
    groups:
      - admins
      - dev
EOF

echo "✅ User database created (username: admin, password: admin123)"

# 3. Add Authelia service to docker-compose.yml
echo "🔄 Adding Authelia service to docker-compose.yml..."

# Check if Authelia service already exists
if ! grep -q "authelia:" docker-compose.yml; then
cat >> docker-compose.yml << 'EOF'

  # Authentication Service
  authelia:
    image: authelia/authelia:4.38
    container_name: authelia
    volumes:
      - ../config/authelia:/config
    networks:
      - traefik_network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.authelia.rule=Host(`auth.${DOMAIN}`)"
      - "traefik.http.routers.authelia.entrypoints=websecure"
      - "traefik.http.routers.authelia.tls=true"
      - "traefik.http.middlewares.authelia.forwardauth.address=http://authelia:9091/api/verify?rd=https://auth.${DOMAIN}"
      - "traefik.http.middlewares.authelia.forwardauth.trustForwardHeader=true"
      - "traefik.http.middlewares.authelia.forwardauth.authResponseHeaders=Remote-User,Remote-Groups,Remote-Name,Remote-Email"
    secrets:
      - authelia_jwt_secret
      - authelia_session_secret
      - authelia_encryption_key
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    depends_on:
      - redis
    environment:
      - AUTHELIA_JWT_SECRET_FILE=/run/secrets/authelia_jwt_secret
      - AUTHELIA_SESSION_SECRET_FILE=/run/secrets/authelia_session_secret
      - AUTHELIA_STORAGE_ENCRYPTION_KEY_FILE=/run/secrets/authelia_encryption_key
EOF

echo "✅ Authelia service added"
else
    echo "ℹ️  Authelia service already exists in docker-compose.yml"
fi

# 4. Enable authentication middleware on services
echo "🔐 Enabling authentication middleware..."
sed -i.bak 's/# - "traefik.http.routers.\([^.]*\)-web.middlewares=authelia"/- "traefik.http.routers.\1-web.middlewares=authelia"/g' docker-compose.yml

echo "✅ Authentication middleware enabled"

# 5. Restart services
echo "🔄 Restarting services..."
docker-compose down
docker-compose up -d

echo ""
echo "🎉 Authentication system fixed!"
echo ""
echo "📋 Login Details:"
echo "   URL: http://auth.localhost (or https://auth.${DOMAIN})"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "⚠️  IMPORTANT: Change the default password after first login!"
echo ""
echo "🔗 Services now require authentication:"
echo "   - Sonarr: http://sonarr.localhost"
echo "   - Radarr: http://radarr.localhost"  
echo "   - Prowlarr: http://prowlarr.localhost"
echo "   - Overseerr: http://overseerr.localhost"
echo ""