#!/bin/bash

# Media Server - Backup Script
# Automated backup for configuration and databases
# Version: 1.0.0

set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
readonly BACKUP_ROOT="${PROJECT_ROOT}/backups"
readonly TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
readonly BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
fi

# Backup configuration
readonly CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config}"
readonly DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data}"
readonly MAX_BACKUPS="${MAX_BACKUPS:-7}"
readonly COMPRESS_BACKUP="${COMPRESS_BACKUP:-true}"

# Services to backup
readonly SERVICES=(
    "jellyfin"
    "radarr"
    "sonarr"
    "prowlarr"
    "qbittorrent"
    "bazarr"
    "overseerr"
    "tautulli"
    "homepage"
    "portainer"
)

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}==== $1 ====${NC}"
}

# Show backup banner
show_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    ____             __                
   / __ )____ ______/ /____  ______    
  / __  / __ `/ ___/ //_/ / / / __ \   
 / /_/ / /_/ / /__/ ,< / /_/ / /_/ /   
/_____/\__,_/\___/_/|_|\__,_/ .___/    
                            /_/         
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Media Server Backup System${NC}"
    echo -e "${CYAN}Backup ID: ${TIMESTAMP}${NC}\n"
}

# Create backup directory
create_backup_dir() {
    log_header "Initializing Backup"
    
    if [ ! -d "$BACKUP_ROOT" ]; then
        mkdir -p "$BACKUP_ROOT"
        log_info "Created backup root directory"
    fi
    
    mkdir -p "$BACKUP_DIR"
    log_info "Created backup directory: $BACKUP_DIR"
    
    # Create subdirectories
    mkdir -p "$BACKUP_DIR"/{config,databases,metadata}
}

# Check disk space
check_disk_space() {
    log_header "Checking Disk Space"
    
    local available_space=$(df -BG "$BACKUP_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    local config_size=$(du -BG -s "$CONFIG_PATH" 2>/dev/null | awk '{print $1}' | sed 's/G//')
    
    log_info "Available space: ${available_space}GB"
    log_info "Estimated backup size: ${config_size}GB"
    
    if [ "$available_space" -lt "$config_size" ]; then
        log_error "Insufficient disk space for backup"
        return 1
    fi
    
    return 0
}

# Stop services before backup
stop_services() {
    log_header "Stopping Services"
    
    read -p "Stop services for consistent backup? (recommended) [Y/n]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_info "Stopping services..."
        docker-compose stop
        sleep 5
        return 0
    else
        log_warn "Services will remain running - backup may be inconsistent"
        return 1
    fi
}

# Backup service configurations
backup_configs() {
    log_header "Backing Up Configurations"
    
    for service in "${SERVICES[@]}"; do
        local service_config="$CONFIG_PATH/$service"
        
        if [ -d "$service_config" ]; then
            log_info "Backing up $service configuration..."
            
            # Create service backup directory
            mkdir -p "$BACKUP_DIR/config/$service"
            
            # Use rsync for efficient copying
            if command -v rsync &> /dev/null; then
                rsync -av --exclude='*.log' --exclude='*.tmp' \
                    "$service_config/" "$BACKUP_DIR/config/$service/"
            else
                cp -R "$service_config/"* "$BACKUP_DIR/config/$service/" 2>/dev/null || true
            fi
            
            # Get directory size
            local size=$(du -sh "$BACKUP_DIR/config/$service" | awk '{print $1}')
            log_info "  └─ Backed up $service ($size)"
        else
            log_warn "  └─ $service config not found, skipping"
        fi
    done
}

# Backup databases
backup_databases() {
    log_header "Backing Up Databases"
    
    # Backup SQLite databases
    local db_files=(
        "sonarr/sonarr.db"
        "radarr/radarr.db"
        "prowlarr/prowlarr.db"
        "bazarr/bazarr.db"
        "overseerr/db/db.sqlite3"
        "tautulli/tautulli.db"
    )
    
    for db_path in "${db_files[@]}"; do
        local full_path="$CONFIG_PATH/$db_path"
        
        if [ -f "$full_path" ]; then
            local service=$(echo "$db_path" | cut -d'/' -f1)
            log_info "Backing up $service database..."
            
            mkdir -p "$BACKUP_DIR/databases/$service"
            
            # Use SQLite backup if available
            if command -v sqlite3 &> /dev/null; then
                sqlite3 "$full_path" ".backup '$BACKUP_DIR/databases/$service/$(basename "$db_path")'"
            else
                cp "$full_path" "$BACKUP_DIR/databases/$service/"
            fi
            
            log_info "  └─ Database backed up"
        fi
    done
}

# Backup Docker volumes
backup_docker_volumes() {
    log_header "Backing Up Docker Volumes"
    
    # Get list of volumes used by our services
    local volumes=$(docker-compose config --volumes 2>/dev/null || echo "")
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            if docker volume inspect "$volume" &> /dev/null; then
                log_info "Backing up volume: $volume"
                
                # Export volume to tar
                docker run --rm \
                    -v "$volume":/source:ro \
                    -v "$BACKUP_DIR/volumes":/backup \
                    alpine tar -czf "/backup/${volume}.tar.gz" -C /source .
                
                log_info "  └─ Volume exported"
            fi
        done
    fi
}

# Create backup metadata
create_metadata() {
    log_header "Creating Backup Metadata"
    
    local metadata_file="$BACKUP_DIR/metadata/backup_info.json"
    
    cat > "$metadata_file" << EOF
{
  "backup_id": "$TIMESTAMP",
  "backup_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "backup_type": "full",
  "services": [
$(for service in "${SERVICES[@]}"; do
    echo "    \"$service\","
done | sed '$ s/,$//')
  ],
  "docker_compose_version": "$(docker-compose version --short 2>/dev/null || echo "unknown")",
  "host_info": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "hostname": "$(hostname)"
  },
  "backup_size": "pending"
}
EOF
    
    # Save docker-compose configuration
    docker-compose config > "$BACKUP_DIR/metadata/docker-compose-config.yml" 2>/dev/null || true
    
    # Save environment variables (sanitized)
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        grep -v -E '(PASSWORD|KEY|SECRET|TOKEN)' "${PROJECT_ROOT}/.env" > "$BACKUP_DIR/metadata/env.sample" || true
    fi
    
    log_info "Metadata created"
}

# Compress backup
compress_backup() {
    if [ "$COMPRESS_BACKUP" = "true" ]; then
        log_header "Compressing Backup"
        
        local archive_name="media-server-backup-${TIMESTAMP}.tar.gz"
        local archive_path="${BACKUP_ROOT}/${archive_name}"
        
        log_info "Creating archive: $archive_name"
        
        # Create compressed archive
        cd "$BACKUP_ROOT"
        tar -czf "$archive_name" "$TIMESTAMP/"
        
        # Verify archive
        if [ -f "$archive_path" ]; then
            local size=$(du -sh "$archive_path" | awk '{print $1}')
            log_info "Archive created: $size"
            
            # Update metadata with final size
            if command -v jq &> /dev/null; then
                local metadata_file="$BACKUP_DIR/metadata/backup_info.json"
                jq --arg size "$size" '.backup_size = $size' "$metadata_file" > "${metadata_file}.tmp"
                mv "${metadata_file}.tmp" "$metadata_file"
            fi
            
            # Remove uncompressed backup
            rm -rf "$BACKUP_DIR"
            log_info "Removed uncompressed files"
        else
            log_error "Failed to create archive"
            return 1
        fi
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log_header "Cleaning Up Old Backups"
    
    # Count existing backups
    local backup_count=$(find "$BACKUP_ROOT" -name "media-server-backup-*.tar.gz" -o -type d -name "[0-9]*_[0-9]*" | wc -l)
    
    if [ "$backup_count" -gt "$MAX_BACKUPS" ]; then
        log_info "Found $backup_count backups, keeping latest $MAX_BACKUPS"
        
        # Remove oldest backups
        if [ "$COMPRESS_BACKUP" = "true" ]; then
            find "$BACKUP_ROOT" -name "media-server-backup-*.tar.gz" | sort | head -n -"$MAX_BACKUPS" | xargs rm -f
        else
            find "$BACKUP_ROOT" -type d -name "[0-9]*_[0-9]*" | sort | head -n -"$MAX_BACKUPS" | xargs rm -rf
        fi
        
        log_info "Old backups removed"
    else
        log_info "Backup count ($backup_count) within limit ($MAX_BACKUPS)"
    fi
}

# Restart services
restart_services() {
    if [ "${SERVICES_STOPPED:-false}" = "true" ]; then
        log_header "Restarting Services"
        log_info "Starting services..."
        docker-compose up -d
        log_info "Services restarted"
    fi
}

# Show backup summary
show_summary() {
    log_header "Backup Summary"
    
    echo -e "${GREEN}✅ Backup completed successfully!${NC}\n"
    
    if [ "$COMPRESS_BACKUP" = "true" ]; then
        local archive_name="media-server-backup-${TIMESTAMP}.tar.gz"
        local archive_path="${BACKUP_ROOT}/${archive_name}"
        
        if [ -f "$archive_path" ]; then
            local size=$(du -sh "$archive_path" | awk '{print $1}')
            echo -e "${CYAN}Backup Details:${NC}"
            echo "  Location: $archive_path"
            echo "  Size: $size"
            echo "  Type: Compressed archive"
        fi
    else
        echo -e "${CYAN}Backup Details:${NC}"
        echo "  Location: $BACKUP_DIR"
        echo "  Type: Uncompressed directory"
    fi
    
    echo -e "\n${CYAN}Restore Command:${NC}"
    echo "  ./scripts/deploy/restore.sh $TIMESTAMP"
    
    echo -e "\n${CYAN}Backup Contents:${NC}"
    echo "  ✓ Service configurations"
    echo "  ✓ Application databases"
    echo "  ✓ Docker compose settings"
    echo "  ✓ Backup metadata"
}

# Main backup flow
main() {
    show_banner
    
    # Check prerequisites
    check_disk_space || exit 1
    
    # Create backup directory
    create_backup_dir
    
    # Stop services if requested
    if stop_services; then
        SERVICES_STOPPED=true
    fi
    
    # Perform backup
    backup_configs
    backup_databases
    backup_docker_volumes
    create_metadata
    
    # Compress if enabled
    compress_backup
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Restart services if stopped
    restart_services
    
    # Show summary
    show_summary
}

# Run main function
main "$@"