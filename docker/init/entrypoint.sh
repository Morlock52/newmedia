#!/bin/bash
# Media Stack Initialization Script - Production 2025
# ====================================================
# Handles permissions, directories, and service readiness

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Environment variables with defaults
PUID=${PUID:-1000}
PGID=${PGID:-1000}
CONFIG_ROOT=${CONFIG_ROOT:-/opt/media-stack/config}
MEDIA_ROOT=${MEDIA_ROOT:-/mnt/media}
DOWNLOADS_ROOT=${DOWNLOADS_ROOT:-/mnt/downloads}
BACKUP_LOCATION=${BACKUP_LOCATION:-/mnt/backups}

log "Starting Media Stack initialization..."
log "PUID: $PUID, PGID: $PGID"

# Create media user if it doesn't exist
if ! id -u mediauser >/dev/null 2>&1; then
    log "Creating mediauser with UID:$PUID and GID:$PGID"
    groupadd -g $PGID mediauser 2>/dev/null || true
    useradd -u $PUID -g $PGID -m -s /bin/bash mediauser 2>/dev/null || true
else
    log "User mediauser already exists"
fi

# Directory structure
DIRECTORIES=(
    "$CONFIG_ROOT"
    "$CONFIG_ROOT/jellyfin"
    "$CONFIG_ROOT/sonarr"
    "$CONFIG_ROOT/radarr"
    "$CONFIG_ROOT/prowlarr"
    "$CONFIG_ROOT/lidarr"
    "$CONFIG_ROOT/readarr"
    "$CONFIG_ROOT/bazarr"
    "$CONFIG_ROOT/overseerr"
    "$CONFIG_ROOT/tautulli"
    "$CONFIG_ROOT/qbittorrent"
    "$CONFIG_ROOT/sabnzbd"
    "$CONFIG_ROOT/gluetun"
    "$CONFIG_ROOT/duplicati"
    "$CONFIG_ROOT/portainer"
    "$CONFIG_ROOT/authelia"
    "$CONFIG_ROOT/traefik"
    "$CONFIG_ROOT/postgres"
    "$CONFIG_ROOT/redis"
    "$CONFIG_ROOT/grafana"
    "$CONFIG_ROOT/prometheus"
    "$MEDIA_ROOT"
    "$MEDIA_ROOT/movies"
    "$MEDIA_ROOT/tv"
    "$MEDIA_ROOT/music"
    "$MEDIA_ROOT/audiobooks"
    "$MEDIA_ROOT/books"
    "$MEDIA_ROOT/podcasts"
    "$DOWNLOADS_ROOT"
    "$DOWNLOADS_ROOT/movies"
    "$DOWNLOADS_ROOT/tv"
    "$DOWNLOADS_ROOT/music"
    "$DOWNLOADS_ROOT/audiobooks"
    "$DOWNLOADS_ROOT/incomplete"
    "$BACKUP_LOCATION"
    "/logs"
    "/logs/jellyfin"
    "/logs/traefik"
    "/logs/authelia"
    "/logs/api"
    "/transcodes"
    "/cache"
)

# Create directories
log "Creating directory structure..."
for dir in "${DIRECTORIES[@]}"; do
    if [[ ! -d "$dir" ]]; then
        log "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Set ownership and permissions
log "Setting ownership and permissions..."
for dir in "${DIRECTORIES[@]}"; do
    chown -R $PUID:$PGID "$dir" 2>/dev/null || warn "Failed to set ownership for $dir"
    chmod -R 755 "$dir" 2>/dev/null || warn "Failed to set permissions for $dir"
done

# Special permissions for security-sensitive directories
log "Setting special permissions for security directories..."
if [[ -d "$CONFIG_ROOT/authelia" ]]; then
    chmod 700 "$CONFIG_ROOT/authelia"
    chown -R $PUID:$PGID "$CONFIG_ROOT/authelia"
fi

if [[ -d "$CONFIG_ROOT/traefik" ]]; then
    chmod 700 "$CONFIG_ROOT/traefik"
    chown -R $PUID:$PGID "$CONFIG_ROOT/traefik"
fi

# Set transcoding directory permissions for hardware acceleration
if [[ -d "/transcodes" ]]; then
    chmod 777 "/transcodes"
    log "Set transcoding directory permissions for hardware acceleration"
fi

# Wait for database services
wait_for_service() {
    local service=$1
    local port=$2
    local timeout=${3:-60}
    
    log "Waiting for $service on port $port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$service" "$port" 2>/dev/null; then
            success "$service is ready!"
            return 0
        fi
        sleep 1
    done
    
    error "$service is not ready after ${timeout}s"
    return 1
}

# Wait for critical services if they're defined
if [[ -n "${DB_HOST:-}" ]] && [[ -n "${DB_PORT:-}" ]]; then
    wait_for_service "${DB_HOST}" "${DB_PORT}" 60
fi

if [[ -n "${REDIS_HOST:-}" ]] && [[ -n "${REDIS_PORT:-}" ]]; then
    wait_for_service "${REDIS_HOST}" "${REDIS_PORT}" 30
fi

# Generate random passwords if not set
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Create secrets file if it doesn't exist
SECRETS_FILE="$CONFIG_ROOT/.secrets.env"
if [[ ! -f "$SECRETS_FILE" ]]; then
    log "Generating secrets file..."
    cat > "$SECRETS_FILE" << EOF
# Auto-generated secrets - $(date)
# DO NOT COMMIT THIS FILE TO VERSION CONTROL
DB_PASSWORD=$(generate_password)
REDIS_PASSWORD=$(generate_password)
AUTHELIA_JWT_SECRET=$(generate_password)$(generate_password)
AUTHELIA_SESSION_SECRET=$(generate_password)$(generate_password)
AUTHELIA_ENCRYPTION_KEY=$(generate_password)
EOF
    chmod 600 "$SECRETS_FILE"
    chown $PUID:$PGID "$SECRETS_FILE"
    success "Generated secrets file at $SECRETS_FILE"
    warn "Please update your .env file with these generated secrets!"
fi

# Check and create initial Authelia users file
AUTHELIA_USERS="$CONFIG_ROOT/authelia/users_database.yml"
if [[ ! -f "$AUTHELIA_USERS" ]]; then
    log "Creating initial Authelia users database..."
    mkdir -p "$(dirname "$AUTHELIA_USERS")"
    cat > "$AUTHELIA_USERS" << 'EOF'
users:
  admin:
    displayname: "Administrator"
    password: "$argon2id$v=19$m=65536,t=3,p=4$CHANGEME_HASH_PASSWORD"
    email: admin@morloksmaze.com
    groups:
      - admins
      - users
EOF
    chown $PUID:$PGID "$AUTHELIA_USERS"
    chmod 600 "$AUTHELIA_USERS"
    warn "Created default Authelia users file. Please generate password hash and update!"
fi

# Health check function
health_check() {
    local checks_passed=0
    local total_checks=5
    
    log "Running system health checks..."
    
    # Check disk space
    if [[ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -lt 90 ]]; then
        success "Disk space check passed"
        ((checks_passed++))
    else
        error "Disk space critical (>90% used)"
    fi
    
    # Check memory
    if [[ $(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d. -f1) -lt 85 ]]; then
        success "Memory usage check passed"
        ((checks_passed++))
    else
        warn "Memory usage high (>85%)"
    fi
    
    # Check directory permissions
    if [[ -w "$CONFIG_ROOT" ]] && [[ -w "$MEDIA_ROOT" ]] && [[ -w "$DOWNLOADS_ROOT" ]]; then
        success "Directory permissions check passed"
        ((checks_passed++))
    else
        error "Directory permissions check failed"
    fi
    
    # Check user/group
    if id -u mediauser >/dev/null 2>&1; then
        success "User check passed"
        ((checks_passed++))
    else
        error "User check failed"
    fi
    
    # Check secrets file
    if [[ -f "$SECRETS_FILE" ]] && [[ -r "$SECRETS_FILE" ]]; then
        success "Secrets file check passed"
        ((checks_passed++))
    else
        error "Secrets file check failed"
    fi
    
    log "Health check completed: $checks_passed/$total_checks checks passed"
    
    if [[ $checks_passed -eq $total_checks ]]; then
        success "All health checks passed!"
        return 0
    else
        error "Some health checks failed!"
        return 1
    fi
}

# Run health check
health_check

# Create readiness indicator
touch /tmp/init-complete
success "Initialization completed successfully!"

# If running as init container, exit successfully
if [[ "${1:-}" == "--init-only" ]]; then
    log "Init container mode - exiting after initialization"
    exit 0
fi

# Otherwise, start the specified command or default shell
if [[ $# -gt 0 ]]; then
    log "Starting command: $*"
    exec su-exec mediauser "$@"
else
    log "No command specified, starting shell"
    exec su-exec mediauser /bin/bash
fi