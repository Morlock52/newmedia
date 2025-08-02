#!/bin/bash
# Automated secret rotation script for media server

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(dirname "$SCRIPT_DIR")"
SECRETS_DIR="$SECURITY_DIR/secrets"
BACKUP_DIR="$SECURITY_DIR/secrets-backup"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create necessary directories
mkdir -p "$SECRETS_DIR" "$BACKUP_DIR"

# Function to generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 "$length" | tr -d '\n'
}

# Function to generate API key
generate_api_key() {
    local prefix=${1:-"ms"}
    echo "${prefix}_$(openssl rand -hex 16)"
}

# Function to backup current secret
backup_secret() {
    local secret_name=$1
    local backup_file="$BACKUP_DIR/${secret_name}-$(date +%Y%m%d-%H%M%S)"
    
    if docker secret inspect "$secret_name" >/dev/null 2>&1; then
        log_info "Backing up secret: $secret_name"
        # Note: Docker doesn't allow direct export of secret values
        # This creates a reference backup
        docker secret inspect "$secret_name" > "${backup_file}.json"
        log_success "Secret backup created: ${backup_file}.json"
    fi
}

# Function to rotate a secret
rotate_secret() {
    local secret_name=$1
    local secret_value=$2
    local service_name=${3:-""}
    
    log_info "Rotating secret: $secret_name"
    
    # Backup existing secret
    backup_secret "$secret_name"
    
    # Create new secret with timestamp suffix
    local new_secret_name="${secret_name}_new_$(date +%s)"
    echo -n "$secret_value" | docker secret create "$new_secret_name" -
    
    if [[ -n "$service_name" ]]; then
        # Update service to use new secret
        log_info "Updating service $service_name with new secret..."
        
        if docker service update \
            --secret-rm "$secret_name" \
            --secret-add "source=$new_secret_name,target=$secret_name" \
            "$service_name" >/dev/null 2>&1; then
            log_success "Service $service_name updated with new secret"
            
            # Wait for service to stabilize
            sleep 10
            
            # Remove old secret
            docker secret rm "$secret_name" >/dev/null 2>&1 || true
            
            # Rename new secret to original name
            # Note: Docker doesn't support secret rename, so we create another one
            echo -n "$secret_value" | docker secret create "$secret_name" -
            docker secret rm "$new_secret_name" >/dev/null 2>&1 || true
            
            log_success "Secret rotation completed for $secret_name"
        else
            log_error "Failed to update service $service_name"
            return 1
        fi
    else
        # For secrets not attached to services
        docker secret rm "$secret_name" >/dev/null 2>&1 || true
        echo -n "$secret_value" | docker secret create "$secret_name" -
        docker secret rm "$new_secret_name" >/dev/null 2>&1 || true
        log_success "Secret $secret_name rotated"
    fi
}

# Function to rotate database passwords
rotate_database_passwords() {
    log_info "Rotating database passwords..."
    
    # PostgreSQL passwords
    local pg_root_pass=$(generate_password 32)
    local pg_user_pass=$(generate_password 32)
    
    rotate_secret "postgres_root_password" "$pg_root_pass"
    rotate_secret "postgres_user_password" "$pg_user_pass"
    
    # Update PostgreSQL user password in database
    docker exec postgres psql -U postgres -c "ALTER USER media_user PASSWORD '$pg_user_pass';" || true
    
    # Redis password
    local redis_pass=$(generate_password 32)
    rotate_secret "redis_password" "$redis_pass"
    
    log_success "Database passwords rotated"
}

# Function to rotate API keys
rotate_api_keys() {
    log_info "Rotating API keys..."
    
    # Media server API keys
    rotate_secret "jellyfin_api_key" "$(generate_api_key 'jf')"
    rotate_secret "plex_token" "$(generate_api_key 'plex')"
    
    # Arr suite API keys
    rotate_secret "sonarr_api_key" "$(generate_api_key 'sonarr')"
    rotate_secret "radarr_api_key" "$(generate_api_key 'radarr')"
    rotate_secret "lidarr_api_key" "$(generate_api_key 'lidarr')"
    rotate_secret "prowlarr_api_key" "$(generate_api_key 'prowlarr')"
    rotate_secret "bazarr_api_key" "$(generate_api_key 'bazarr')"
    
    log_success "API keys rotated"
}

# Function to rotate authentication secrets
rotate_auth_secrets() {
    log_info "Rotating authentication secrets..."
    
    # Authelia secrets
    local jwt_secret=$(generate_password 64)
    local session_secret=$(generate_password 64)
    local encryption_key=$(generate_password 32)
    
    rotate_secret "authelia_jwt_secret" "$jwt_secret"
    rotate_secret "authelia_session_secret" "$session_secret"
    rotate_secret "authelia_storage_encryption_key" "$encryption_key"
    
    log_success "Authentication secrets rotated"
}

# Function to generate SSL certificates
generate_ssl_certificates() {
    log_info "Generating new SSL certificates..."
    
    local ssl_dir="$SECRETS_DIR/ssl"
    mkdir -p "$ssl_dir"
    
    # Generate self-signed certificate for internal use
    openssl req -x509 -newkey rsa:4096 -keyout "$ssl_dir/key.pem" -out "$ssl_dir/cert.pem" \
        -days 365 -nodes -subj "/C=US/ST=State/L=City/O=MediaServer/CN=*.local" 2>/dev/null
    
    log_success "SSL certificates generated"
}

# Function to verify service health after rotation
verify_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=0
    
    log_info "Verifying health of service: $service"
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker ps | grep -q "$service"; then
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "none")
            
            if [[ "$health" == "healthy" ]] || [[ "$health" == "none" ]]; then
                log_success "Service $service is healthy"
                return 0
            fi
        fi
        
        ((attempt++))
        sleep 2
    done
    
    log_error "Service $service failed health check"
    return 1
}

# Function to create initial secrets
create_initial_secrets() {
    log_info "Creating initial secrets..."
    
    # Database passwords
    echo -n "$(generate_password 32)" | docker secret create postgres_root_password - 2>/dev/null || true
    echo -n "$(generate_password 32)" | docker secret create postgres_user_password - 2>/dev/null || true
    echo -n "$(generate_password 32)" | docker secret create redis_password - 2>/dev/null || true
    
    # API keys
    echo -n "$(generate_api_key 'jf')" | docker secret create jellyfin_api_key - 2>/dev/null || true
    echo -n "$(generate_api_key 'plex')" | docker secret create plex_token - 2>/dev/null || true
    echo -n "$(generate_api_key 'sonarr')" | docker secret create sonarr_api_key - 2>/dev/null || true
    echo -n "$(generate_api_key 'radarr')" | docker secret create radarr_api_key - 2>/dev/null || true
    echo -n "$(generate_api_key 'lidarr')" | docker secret create lidarr_api_key - 2>/dev/null || true
    echo -n "$(generate_api_key 'prowlarr')" | docker secret create prowlarr_api_key - 2>/dev/null || true
    echo -n "$(generate_api_key 'bazarr')" | docker secret create bazarr_api_key - 2>/dev/null || true
    
    # Authentication secrets
    echo -n "$(generate_password 64)" | docker secret create authelia_jwt_secret - 2>/dev/null || true
    echo -n "$(generate_password 64)" | docker secret create authelia_session_secret - 2>/dev/null || true
    echo -n "$(generate_password 32)" | docker secret create authelia_storage_encryption_key - 2>/dev/null || true
    
    # VPN key (placeholder - should be replaced with actual key)
    echo -n "VPN_PRIVATE_KEY_PLACEHOLDER" | docker secret create vpn_private_key - 2>/dev/null || true
    
    log_success "Initial secrets created"
}

# Function to display secret status
display_secret_status() {
    log_info "Current secret status:"
    echo ""
    
    docker secret ls --format "table {{.Name}}\t{{.CreatedAt}}\t{{.UpdatedAt}}"
    echo ""
}

# Main execution
main() {
    local action=${1:-"status"}
    
    case "$action" in
        init)
            log_info "Initializing secrets..."
            create_initial_secrets
            generate_ssl_certificates
            display_secret_status
            ;;
            
        rotate-all)
            log_info "Rotating all secrets..."
            rotate_database_passwords
            rotate_api_keys
            rotate_auth_secrets
            display_secret_status
            
            # Verify critical services
            for service in postgres redis authelia jellyfin; do
                verify_service_health "$service" || log_warning "Service $service may need attention"
            done
            ;;
            
        rotate-db)
            rotate_database_passwords
            verify_service_health "postgres"
            verify_service_health "redis"
            ;;
            
        rotate-api)
            rotate_api_keys
            ;;
            
        rotate-auth)
            rotate_auth_secrets
            verify_service_health "authelia"
            ;;
            
        status)
            display_secret_status
            ;;
            
        backup)
            log_info "Backing up all secrets..."
            for secret in $(docker secret ls --format "{{.Name}}"); do
                backup_secret "$secret"
            done
            log_success "All secrets backed up to $BACKUP_DIR"
            ;;
            
        *)
            echo "Usage: $0 {init|rotate-all|rotate-db|rotate-api|rotate-auth|status|backup}"
            echo ""
            echo "  init        - Create initial secrets"
            echo "  rotate-all  - Rotate all secrets"
            echo "  rotate-db   - Rotate database passwords only"
            echo "  rotate-api  - Rotate API keys only"
            echo "  rotate-auth - Rotate authentication secrets only"
            echo "  status      - Display current secret status"
            echo "  backup      - Backup all secrets"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"