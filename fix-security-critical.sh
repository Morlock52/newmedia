#!/usr/bin/env bash
set -euo pipefail

# Media Server Security Fix Script
# This script addresses critical security vulnerabilities

echo "🔒 Media Server Security Remediation Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function for logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root!"
   exit 1
fi

# Step 1: Create secure environment file
log "Creating secure environment file..."
cat > .env.secure << 'EOF'
# API Keys - REGENERATE ALL OF THESE!
SONARR_API_KEY=
RADARR_API_KEY=
PROWLARR_API_KEY=
LIDARR_API_KEY=
READARR_API_KEY=
BAZARR_API_KEY=
OVERSEERR_API_KEY=
TAUTULLI_API_KEY=
JELLYFIN_API_KEY=

# Database Passwords
POSTGRES_PASSWORD=
MYSQL_ROOT_PASSWORD=
REDIS_PASSWORD=

# Authentication
AUTHELIA_JWT_SECRET=
AUTHELIA_SESSION_SECRET=
AUTHELIA_STORAGE_ENCRYPTION_KEY=

# Other Secrets
TRAEFIK_DASHBOARD_AUTH=
GRAFANA_ADMIN_PASSWORD=
EOF

chmod 600 .env.secure
log "✅ Created .env.secure with proper permissions"

# Step 2: Backup current docker-compose.yml
log "Backing up current docker-compose.yml..."
cp docker-compose.yml docker-compose.yml.insecure.backup
log "✅ Backup created: docker-compose.yml.insecure.backup"

# Step 3: Remove hardcoded API keys from docker-compose.yml
log "Removing hardcoded API keys from docker-compose.yml..."
sed -i.bak \
    -e 's/X-Api-Key: 79eecf2b23f34760b91cfcbf97189dd0/X-Api-Key: ${SONARR_API_KEY}/g' \
    -e 's/X-Api-Key: 1c0fe63736a04e6394dacb3aa1160b1c/X-Api-Key: ${RADARR_API_KEY}/g' \
    -e 's/X-Api-Key: 5a35dd23f90c4d2bb69caa1eb0e1c534/X-Api-Key: ${PROWLARR_API_KEY}/g' \
    docker-compose.yml
log "✅ Removed hardcoded API keys from docker-compose.yml"

# Step 4: Update homepage services.yaml
if [ -f "homepage-config/services.yaml" ]; then
    log "Updating homepage services configuration..."
    cp homepage-config/services.yaml homepage-config/services.yaml.backup
    sed -i \
        -e 's/key: 79eecf2b23f34760b91cfcbf97189dd0/key: {{HOMEPAGE_VAR_SONARR_API_KEY}}/g' \
        -e 's/key: 1c0fe63736a04e6394dacb3aa1160b1c/key: {{HOMEPAGE_VAR_RADARR_API_KEY}}/g' \
        -e 's/key: 5a35dd23f90c4d2bb69caa1eb0e1c534/key: {{HOMEPAGE_VAR_PROWLARR_API_KEY}}/g' \
        homepage-config/services.yaml
    log "✅ Updated homepage configuration"
fi

# Step 5: Create docker-compose override for env_file
log "Creating docker-compose.override.yml for environment variables..."
cat > docker-compose.override.yml << 'EOF'
# Security Override - Use environment variables
version: '3.8'

services:
  sonarr:
    env_file:
      - .env.secure
    environment:
      - SONARR__ApiKey=${SONARR_API_KEY}

  radarr:
    env_file:
      - .env.secure
    environment:
      - RADARR__ApiKey=${RADARR_API_KEY}

  prowlarr:
    env_file:
      - .env.secure
    environment:
      - PROWLARR__ApiKey=${PROWLARR_API_KEY}

  lidarr:
    env_file:
      - .env.secure
    environment:
      - LIDARR__ApiKey=${LIDARR_API_KEY}

  bazarr:
    env_file:
      - .env.secure
    environment:
      - BAZARR__ApiKey=${BAZARR_API_KEY}

  overseerr:
    env_file:
      - .env.secure

  tautulli:
    env_file:
      - .env.secure

  homepage:
    env_file:
      - .env.secure
    environment:
      - HOMEPAGE_VAR_SONARR_API_KEY=${SONARR_API_KEY}
      - HOMEPAGE_VAR_RADARR_API_KEY=${RADARR_API_KEY}
      - HOMEPAGE_VAR_PROWLARR_API_KEY=${PROWLARR_API_KEY}
      - HOMEPAGE_VAR_LIDARR_API_KEY=${LIDARR_API_KEY}
      - HOMEPAGE_VAR_OVERSEERR_API_KEY=${OVERSEERR_API_KEY}
EOF
log "✅ Created docker-compose.override.yml"

# Step 6: Generate secure random passwords
log "Generating secure random passwords..."
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Update .env.secure with generated passwords
sed -i \
    -e "s/POSTGRES_PASSWORD=/POSTGRES_PASSWORD=$(generate_password)/g" \
    -e "s/MYSQL_ROOT_PASSWORD=/MYSQL_ROOT_PASSWORD=$(generate_password)/g" \
    -e "s/REDIS_PASSWORD=/REDIS_PASSWORD=$(generate_password)/g" \
    -e "s/AUTHELIA_JWT_SECRET=/AUTHELIA_JWT_SECRET=$(generate_password)/g" \
    -e "s/AUTHELIA_SESSION_SECRET=/AUTHELIA_SESSION_SECRET=$(generate_password)/g" \
    -e "s/AUTHELIA_STORAGE_ENCRYPTION_KEY=/AUTHELIA_STORAGE_ENCRYPTION_KEY=$(generate_password)/g" \
    -e "s/GRAFANA_ADMIN_PASSWORD=/GRAFANA_ADMIN_PASSWORD=$(generate_password)/g" \
    .env.secure
log "✅ Generated secure passwords"

# Step 7: Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    log "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Environment files
.env
.env.*
!.env.example

# Sensitive data
*.key
*.pem
*.crt
secrets/
**/api_keys.json

# Backup files
*.backup
*.bak

# Logs
*.log
logs/

# Database files
*.db
*.sqlite

# Cache
cache/
__pycache__/
*.pyc
EOF
    log "✅ Created .gitignore"
else
    log "Adding security entries to existing .gitignore..."
    echo -e "\n# Security files\n.env.secure\n*.backup" >> .gitignore
fi

# Step 8: Instructions for manual steps
echo ""
echo "=========================================="
echo "🚨 CRITICAL MANUAL STEPS REQUIRED 🚨"
echo "=========================================="
echo ""
echo "1. REGENERATE ALL API KEYS:"
echo "   - Start services: docker-compose up -d"
echo "   - For each service, go to Settings → General → API Key → Regenerate"
echo "   - Copy the new API key to .env.secure"
echo ""
echo "2. SERVICES TO UPDATE:"
echo "   - Sonarr: http://localhost:8989"
echo "   - Radarr: http://localhost:7878"
echo "   - Prowlarr: http://localhost:9696"
echo "   - Lidarr: http://localhost:8686"
echo "   - Bazarr: http://localhost:6767"
echo "   - Overseerr: http://localhost:5055"
echo "   - Tautulli: http://localhost:8181"
echo ""
echo "3. RESTART SERVICES AFTER UPDATING .env.secure:"
echo "   docker-compose down"
echo "   docker-compose up -d"
echo ""
echo "4. VERIFY SERVICES ARE WORKING:"
echo "   docker-compose ps"
echo "   docker-compose logs -f"
echo ""
echo "5. IF USING GIT, CLEAN HISTORY:"
warning "   If you committed the exposed keys to git, you MUST clean the history!"
echo "   See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository"
echo ""
echo "=========================================="
log "Security remediation script complete!"
log "Remember: This is just the first step. Review ACTIONABLE_IMPROVEMENT_PLAN_2025.md for full security hardening."