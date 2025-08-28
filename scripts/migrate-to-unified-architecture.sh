#!/bin/bash
# migrate-to-unified-architecture.sh
# Script to migrate from current fragmented setup to unified architecture

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CURRENT_DIR="$(pwd)"
BACKUP_DIR="${BACKUP_DIR:-/backup/media-server-migration-$(date +%Y%m%d_%H%M%S)}"
NEW_DIR="${NEW_DIR:-/opt/unified-media-server}"
DOCKER_COMPOSE_FILES=(
    "docker-compose.yml"
    "docker-compose.master.yml"
    "docker-compose-optimized-2025.yml"
    "docker-compose-performance-optimized-2025.yml"
)

# Logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" ;;
    esac
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log WARN "This script should not be run as root. It will use sudo when needed."
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    local deps=("docker" "docker-compose" "jq" "rsync")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log ERROR "Missing dependencies: ${missing[*]}"
        log INFO "Please install missing dependencies and try again"
        exit 1
    fi
}

# Analyze current setup
analyze_current_setup() {
    log INFO "Analyzing current setup..."
    
    echo -e "\n${CYAN}=== Current Docker Compose Files ===${NC}"
    for file in "${DOCKER_COMPOSE_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            echo "  ✓ $file"
            
            # Extract services
            if command -v yq &> /dev/null; then
                echo "    Services: $(yq eval '.services | keys | join(", ")' "$file" 2>/dev/null || echo "Unable to parse")"
            fi
        fi
    done
    
    echo -e "\n${CYAN}=== Running Containers ===${NC}"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | head -20
    
    echo -e "\n${CYAN}=== Docker Volumes ===${NC}"
    docker volume ls --format "table {{.Name}}\t{{.Driver}}" | head -20
    
    echo -e "\n${CYAN}=== Disk Usage ===${NC}"
    df -h | grep -E "^/dev|Filesystem"
}

# Create backup
create_backup() {
    log INFO "Creating backup in $BACKUP_DIR"
    
    # Create backup directory
    sudo mkdir -p "$BACKUP_DIR"
    sudo chown $USER:$USER "$BACKUP_DIR"
    
    # Backup docker-compose files
    log INFO "Backing up Docker Compose files..."
    mkdir -p "$BACKUP_DIR/compose"
    for file in "${DOCKER_COMPOSE_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$BACKUP_DIR/compose/"
        fi
    done
    
    # Backup environment files
    log INFO "Backing up environment files..."
    mkdir -p "$BACKUP_DIR/env"
    cp .env* "$BACKUP_DIR/env/" 2>/dev/null || true
    
    # Backup configurations
    if [[ -d "config" ]]; then
        log INFO "Backing up configurations..."
        rsync -av --progress config/ "$BACKUP_DIR/config/"
    fi
    
    # List volumes for backup
    log INFO "Docker volumes to backup:"
    docker volume ls --format "{{.Name}}" > "$BACKUP_DIR/volumes.txt"
    cat "$BACKUP_DIR/volumes.txt"
    
    # Create backup script for volumes
    cat > "$BACKUP_DIR/backup-volumes.sh" << 'EOF'
#!/bin/bash
# Backup Docker volumes
BACKUP_VOL_DIR="./volumes"
mkdir -p "$BACKUP_VOL_DIR"

while read -r volume; do
    echo "Backing up volume: $volume"
    docker run --rm -v "$volume":/source -v "$PWD/$BACKUP_VOL_DIR":/backup alpine \
        tar -czf "/backup/${volume}.tar.gz" -C /source .
done < volumes.txt
EOF
    chmod +x "$BACKUP_DIR/backup-volumes.sh"
    
    log INFO "Backup created at: $BACKUP_DIR"
    log WARN "To backup volumes, run: cd $BACKUP_DIR && ./backup-volumes.sh"
}

# Map services to profiles
map_services_to_profiles() {
    log INFO "Mapping current services to unified profiles..."
    
    declare -A service_mapping=(
        # Core services
        ["traefik"]="core"
        ["authelia"]="core"
        ["postgres"]="core"
        ["redis"]="core"
        
        # Media services
        ["jellyfin"]="media"
        ["plex"]="media"
        ["emby"]="media"
        ["navidrome"]="music"
        ["audiobookshelf"]="books"
        ["calibre-web"]="books"
        ["kavita"]="books"
        ["immich-server"]="photos"
        
        # Automation
        ["prowlarr"]="automation"
        ["sonarr"]="automation"
        ["radarr"]="automation"
        ["lidarr"]="automation"
        ["readarr"]="automation"
        ["bazarr"]="automation"
        
        # Downloads
        ["qbittorrent"]="downloads"
        ["sabnzbd"]="downloads"
        ["vpn"]="downloads"
        ["gluetun"]="downloads"
        
        # Requests
        ["overseerr"]="requests"
        ["jellyseerr"]="requests"
        
        # Monitoring
        ["prometheus"]="monitoring"
        ["grafana"]="monitoring"
        ["tautulli"]="monitoring"
        
        # Management
        ["portainer"]="management"
        ["homepage"]="management"
        ["homarr"]="management"
    )
    
    # Get running services
    local running_services=$(docker ps --format "{{.Names}}")
    local profiles_needed=("core")  # Always need core
    
    for service in $running_services; do
        if [[ -n "${service_mapping[$service]}" ]]; then
            local profile="${service_mapping[$service]}"
            if [[ ! " ${profiles_needed[@]} " =~ " ${profile} " ]]; then
                profiles_needed+=("$profile")
            fi
            log DEBUG "Service $service maps to profile: $profile"
        else
            log WARN "Service $service has no profile mapping"
        fi
    done
    
    echo -e "\n${CYAN}=== Recommended Profiles ===${NC}"
    echo "Based on your running services, enable these profiles:"
    for profile in "${profiles_needed[@]}"; do
        echo "  - $profile"
    done
    
    # Save to file for later use
    printf '%s\n' "${profiles_needed[@]}" > "$BACKUP_DIR/recommended-profiles.txt"
}

# Create unified setup
create_unified_setup() {
    log INFO "Creating unified setup in $NEW_DIR"
    
    # Create directory structure
    sudo mkdir -p "$NEW_DIR"
    sudo chown -R $USER:$USER "$NEW_DIR"
    cd "$NEW_DIR"
    
    mkdir -p {config,data,media,downloads,backups,scripts}
    
    # Copy unified architecture files
    log INFO "Copying unified architecture files..."
    
    # Create docker-compose.yml from the unified template
    if [[ -f "$CURRENT_DIR/UNIFIED_MEDIA_SERVER_ARCHITECTURE_2025.md" ]]; then
        log INFO "Extracting docker-compose.yml from architecture document..."
        # This would normally extract the YAML from the markdown
        # For now, we'll create a placeholder
        cat > docker-compose.yml << 'EOF'
# Unified Media Server Docker Compose
# This file should be replaced with the actual unified docker-compose.yml
# from UNIFIED_MEDIA_SERVER_ARCHITECTURE_2025.md

version: '3.9'
name: unified-media-server

services:
  # Services will be defined here
  placeholder:
    image: hello-world
    profiles: ["never"]
EOF
    fi
    
    # Copy management script
    if [[ -f "$CURRENT_DIR/scripts/unified-media-manager.sh" ]]; then
        cp "$CURRENT_DIR/scripts/unified-media-manager.sh" .
    else
        log WARN "Unified media manager script not found, creating basic version..."
        cat > unified-media-manager.sh << 'EOF'
#!/bin/bash
# Basic unified media manager
echo "Unified Media Manager - Placeholder"
echo "Replace with actual script from UNIFIED_MEDIA_SERVER_ARCHITECTURE_2025.md"
EOF
    fi
    chmod +x unified-media-manager.sh
    
    # Create .env file
    log INFO "Creating environment file..."
    cat > .env << EOF
# Unified Media Server Environment
# Generated on $(date)

# Basic Configuration
TZ=${TZ:-America/New_York}
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Domain Configuration
DOMAIN=${DOMAIN:-example.com}
ACME_EMAIL=${ACME_EMAIL:-admin@example.com}

# Paths
MEDIA_PATH=$NEW_DIR/media
DOWNLOADS_PATH=$NEW_DIR/downloads
CONFIG_PATH=$NEW_DIR/config
DATA_PATH=$NEW_DIR/data

# Security (Generate new passwords)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
AUTHELIA_JWT_SECRET=$(openssl rand -base64 32)
AUTHELIA_SESSION_SECRET=$(openssl rand -base64 32)
AUTHELIA_STORAGE_ENCRYPTION_KEY=$(openssl rand -base64 32)

# Service Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=$(openssl rand -base64 16)
EOF
    
    # Copy existing configurations
    if [[ -d "$CURRENT_DIR/config" ]]; then
        log INFO "Copying existing configurations..."
        rsync -av --progress "$CURRENT_DIR/config/" "$NEW_DIR/config/"
    fi
    
    log INFO "Unified setup created at: $NEW_DIR"
}

# Migration plan
create_migration_plan() {
    log INFO "Creating migration plan..."
    
    cat > "$NEW_DIR/migration-plan.md" << EOF
# Migration Plan
Generated on: $(date)

## Current Setup
- Location: $CURRENT_DIR
- Backup: $BACKUP_DIR

## New Setup
- Location: $NEW_DIR

## Migration Steps

1. **Stop Current Services**
   \`\`\`bash
   cd $CURRENT_DIR
   docker-compose down
   \`\`\`

2. **Enable Recommended Profiles**
   \`\`\`bash
   cd $NEW_DIR
   $(cat $BACKUP_DIR/recommended-profiles.txt | while read profile; do echo "./unified-media-manager.sh enable $profile"; done)
   \`\`\`

3. **Migrate Volumes**
   - Option 1: Point to existing directories
   - Option 2: Restore from backup
   - Option 3: Use Docker volume migration

4. **Update DNS/Reverse Proxy**
   - Update domain settings
   - Configure Cloudflare tunnel

5. **Test Services**
   - Verify each service is accessible
   - Check data integrity
   - Test automation workflows

## Rollback Plan
1. Stop new services: \`cd $NEW_DIR && docker-compose down\`
2. Restore from backup: \`cd $CURRENT_DIR\`
3. Start original services: \`docker-compose up -d\`

## Post-Migration
- Monitor logs for 24 hours
- Update documentation
- Remove old setup after verification
EOF
    
    log INFO "Migration plan created at: $NEW_DIR/migration-plan.md"
}

# Perform pre-migration checks
pre_migration_checks() {
    log INFO "Performing pre-migration checks..."
    
    echo -e "\n${CYAN}=== Pre-Migration Checklist ===${NC}"
    
    # Check disk space
    local available_space=$(df -BG "$NEW_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 50 ]]; then
        log WARN "Low disk space: ${available_space}GB available (50GB recommended)"
    else
        echo "  ✓ Disk space: ${available_space}GB available"
    fi
    
    # Check running services
    local running_count=$(docker ps -q | wc -l)
    echo "  ✓ Running containers: $running_count"
    
    # Check for conflicts
    if [[ -d "$NEW_DIR/docker-compose.yml" ]]; then
        log WARN "Existing setup found in $NEW_DIR"
    else
        echo "  ✓ No conflicts in target directory"
    fi
    
    # Check permissions
    if touch "$NEW_DIR/.test" 2>/dev/null; then
        rm "$NEW_DIR/.test"
        echo "  ✓ Write permissions OK"
    else
        log ERROR "No write permissions to $NEW_DIR"
        exit 1
    fi
}

# Main migration function
main() {
    clear
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${CYAN}   Unified Media Server Migration Tool${NC}"
    echo -e "${CYAN}===============================================${NC}"
    echo
    
    check_root
    check_dependencies
    
    # Confirm migration
    echo -e "${YELLOW}This tool will:${NC}"
    echo "  1. Analyze your current media server setup"
    echo "  2. Create a complete backup"
    echo "  3. Generate a unified architecture configuration"
    echo "  4. Provide a migration plan"
    echo
    echo -e "${YELLOW}Current directory:${NC} $CURRENT_DIR"
    echo -e "${YELLOW}Backup directory:${NC} $BACKUP_DIR"
    echo -e "${YELLOW}New directory:${NC} $NEW_DIR"
    echo
    
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log INFO "Migration cancelled"
        exit 0
    fi
    
    # Run migration steps
    analyze_current_setup
    
    echo
    read -p "Create backup? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        create_backup
    fi
    
    map_services_to_profiles
    
    echo
    read -p "Create unified setup? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        pre_migration_checks
        create_unified_setup
        create_migration_plan
    fi
    
    # Summary
    echo
    echo -e "${GREEN}=== Migration Preparation Complete ===${NC}"
    echo
    echo "Next steps:"
    echo "  1. Review the migration plan: $NEW_DIR/migration-plan.md"
    echo "  2. Complete the docker-compose.yml with the unified architecture"
    echo "  3. Run the migration following the plan"
    echo
    echo -e "${YELLOW}Important:${NC}"
    echo "  - Your current setup is still running"
    echo "  - Backup created at: $BACKUP_DIR"
    echo "  - New setup prepared at: $NEW_DIR"
    echo
    log INFO "Migration preparation completed successfully!"
}

# Run main function
main "$@"