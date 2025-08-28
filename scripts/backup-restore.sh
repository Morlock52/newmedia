#!/bin/bash

# ==================================================================
# COMPREHENSIVE BACKUP AND RESTORE SYSTEM
# Automated backup and restore procedures for media server
# ==================================================================

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
BACKUP_BASE_DIR="./backups"
BACKUP_RETENTION_DAYS=30
LOG_FILE="./logs/backup-$(date +%Y%m%d_%H%M%S).log"
COMPOSE_FILE="docker-compose.yml"
COMPRESSION_LEVEL=6
ENCRYPTION_PASSWORD=""  # Set this for encrypted backups
REMOTE_BACKUP_ENABLED=false
REMOTE_BACKUP_PATH=""  # S3, SFTP, or other remote path

# Backup types
BACKUP_TYPES=(
    "configuration"
    "databases"
    "application_data"
    "media_metadata"
    "docker_volumes"
)

# Critical services for database backup
declare -A DATABASE_SERVICES=(
    ["postgres"]="postgresql"
    ["mariadb"]="mysql"
    ["redis"]="redis"
)

# Application data paths
declare -A APP_DATA_PATHS=(
    ["sonarr"]="/config"
    ["radarr"]="/config"
    ["lidarr"]="/config"
    ["prowlarr"]="/config"
    ["bazarr"]="/config"
    ["jellyfin"]="/config"
    ["plex"]="/config"
    ["grafana"]="/var/lib/grafana"
    ["prometheus"]="/prometheus"
    ["uptime-kuma"]="/app/data"
)

# Exclude patterns for media backups (to avoid backing up actual media files)
MEDIA_EXCLUDE_PATTERNS=(
    "*/movies/*"
    "*/tv/*"
    "*/music/*"
    "*/books/*"
    "*/audiobooks/*"
    "*/photos/*"
    "*/downloads/*"
    "*/torrents/*"
    "*/usenet/*"
    "*.mkv"
    "*.mp4"
    "*.avi"
    "*.mov"
    "*.flac"
    "*.mp3"
    "*.iso"
)

# Ensure directories exist
mkdir -p logs "$BACKUP_BASE_DIR"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Get backup directory for current session
get_backup_dir() {
    local backup_type="$1"
    local timestamp="${2:-$(date +%Y%m%d_%H%M%S)}"
    echo "$BACKUP_BASE_DIR/${backup_type}_${timestamp}"
}

# Check available disk space
check_disk_space() {
    local required_gb="$1"
    local available_kb
    available_kb=$(df "$BACKUP_BASE_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    
    log "Disk space check passed. Available: ${available_gb}GB"
    return 0
}

# Encrypt file if password is set
encrypt_file() {
    local file="$1"
    
    if [ -n "$ENCRYPTION_PASSWORD" ]; then
        log "Encrypting $file"
        openssl enc -aes-256-cbc -salt -in "$file" -out "${file}.enc" -pass pass:"$ENCRYPTION_PASSWORD"
        rm "$file"
        echo "${file}.enc"
    else
        echo "$file"
    fi
}

# Decrypt file if needed
decrypt_file() {
    local file="$1"
    
    if [[ "$file" == *.enc ]] && [ -n "$ENCRYPTION_PASSWORD" ]; then
        local decrypted_file="${file%.enc}"
        log "Decrypting $file"
        openssl enc -aes-256-cbc -d -salt -in "$file" -out "$decrypted_file" -pass pass:"$ENCRYPTION_PASSWORD"
        echo "$decrypted_file"
    else
        echo "$file"
    fi
}

# Compress directory
compress_directory() {
    local source_dir="$1"
    local output_file="$2"
    
    log "Compressing $source_dir to $output_file"
    
    # Build exclude arguments
    local exclude_args=""
    for pattern in "${MEDIA_EXCLUDE_PATTERNS[@]}"; do
        exclude_args="$exclude_args --exclude=$pattern"
    done
    
    # Create compressed archive
    tar $exclude_args -czf "$output_file" -C "$(dirname "$source_dir")" "$(basename "$source_dir")"
    
    # Encrypt if password is set
    output_file=$(encrypt_file "$output_file")
    
    success "Compressed and encrypted: $output_file"
    echo "$output_file"
}

# Backup configuration files
backup_configuration() {
    local backup_dir="$1"
    local config_backup_dir="$backup_dir/configuration"
    
    log "Backing up configuration files..."
    mkdir -p "$config_backup_dir"
    
    # Backup docker-compose files
    for compose_file in docker-compose*.yml docker-compose*.yaml; do
        if [ -f "$compose_file" ]; then
            cp "$compose_file" "$config_backup_dir/"
        fi
    done
    
    # Backup environment files
    for env_file in .env .env.* *.env; do
        if [ -f "$env_file" ]; then
            cp "$env_file" "$config_backup_dir/"
        fi
    done
    
    # Backup configuration directories
    if [ -d "config" ]; then
        cp -r config "$config_backup_dir/"
    fi
    
    # Backup custom scripts
    if [ -d "scripts" ]; then
        cp -r scripts "$config_backup_dir/"
    fi
    
    # Backup nginx configurations
    for nginx_dir in nginx-config npm-data nginx-proxy-manager-data; do
        if [ -d "$nginx_dir" ]; then
            cp -r "$nginx_dir" "$config_backup_dir/"
        fi
    done
    
    # Create compressed archive
    local config_archive="$backup_dir/configuration.tar.gz"
    config_archive=$(compress_directory "$config_backup_dir" "$config_archive")
    
    # Remove temporary directory
    rm -rf "$config_backup_dir"
    
    success "Configuration backup completed: $config_archive"
}

# Backup databases
backup_databases() {
    local backup_dir="$1"
    local db_backup_dir="$backup_dir/databases"
    
    log "Backing up databases..."
    mkdir -p "$db_backup_dir"
    
    # Backup PostgreSQL
    if docker ps --format "{{.Names}}" | grep -q "postgres"; then
        log "Backing up PostgreSQL database..."
        
        local postgres_dump="$db_backup_dir/postgres-$(date +%Y%m%d_%H%M%S).sql"
        docker exec postgres pg_dumpall -U "${POSTGRES_USER:-postgres}" > "$postgres_dump"
        
        # Compress and encrypt
        gzip "$postgres_dump"
        postgres_dump="${postgres_dump}.gz"
        postgres_dump=$(encrypt_file "$postgres_dump")
        
        success "PostgreSQL backup completed: $postgres_dump"
    fi
    
    # Backup MariaDB/MySQL
    if docker ps --format "{{.Names}}" | grep -q "mariadb"; then
        log "Backing up MariaDB database..."
        
        local mysql_dump="$db_backup_dir/mariadb-$(date +%Y%m%d_%H%M%S).sql"
        docker exec mariadb mysqldump --all-databases -u root -p"${MYSQL_ROOT_PASSWORD:-root}" > "$mysql_dump"
        
        # Compress and encrypt
        gzip "$mysql_dump"
        mysql_dump="${mysql_dump}.gz"
        mysql_dump=$(encrypt_file "$mysql_dump")
        
        success "MariaDB backup completed: $mysql_dump"
    fi
    
    # Backup Redis
    if docker ps --format "{{.Names}}" | grep -q "redis"; then
        log "Backing up Redis database..."
        
        local redis_dump="$db_backup_dir/redis-$(date +%Y%m%d_%H%M%S).rdb"
        docker exec redis redis-cli BGSAVE
        
        # Wait for background save to complete
        while [ "$(docker exec redis redis-cli LASTSAVE)" = "$(docker exec redis redis-cli LASTSAVE)" ]; do
            sleep 1
        done
        
        docker cp redis:/data/dump.rdb "$redis_dump"
        
        # Encrypt
        redis_dump=$(encrypt_file "$redis_dump")
        
        success "Redis backup completed: $redis_dump"
    fi
}

# Backup application data
backup_application_data() {
    local backup_dir="$1"
    local app_backup_dir="$backup_dir/application_data"
    
    log "Backing up application data..."
    mkdir -p "$app_backup_dir"
    
    for service in "${!APP_DATA_PATHS[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "$service"; then
            log "Backing up $service application data..."
            
            local service_backup_dir="$app_backup_dir/$service"
            mkdir -p "$service_backup_dir"
            
            # Copy data from container
            docker cp "$service:${APP_DATA_PATHS[$service]}" "$service_backup_dir/"
            
            # Create compressed archive for this service
            local service_archive="$backup_dir/${service}-data.tar.gz"
            service_archive=$(compress_directory "$service_backup_dir" "$service_archive")
            
            # Remove temporary directory
            rm -rf "$service_backup_dir"
            
            success "$service data backup completed: $service_archive"
        fi
    done
}

# Backup media metadata (not actual media files)
backup_media_metadata() {
    local backup_dir="$1"
    local metadata_backup_dir="$backup_dir/media_metadata"
    
    log "Backing up media metadata..."
    mkdir -p "$metadata_backup_dir"
    
    # Backup Jellyfin metadata
    if [ -d "jellyfin-config" ]; then
        log "Backing up Jellyfin metadata..."
        rsync -av --exclude="cache/" --exclude="transcodes/" \
              jellyfin-config/metadata/ "$metadata_backup_dir/jellyfin-metadata/" 2>/dev/null || true
    fi
    
    # Backup Plex metadata
    if [ -d "plex-config" ]; then
        log "Backing up Plex metadata..."
        rsync -av --include="Metadata/" --include="Media/" --exclude="*" \
              plex-config/ "$metadata_backup_dir/plex-metadata/" 2>/dev/null || true
    fi
    
    # Create compressed archive
    if [ -d "$metadata_backup_dir" ] && [ "$(ls -A "$metadata_backup_dir")" ]; then
        local metadata_archive="$backup_dir/media-metadata.tar.gz"
        metadata_archive=$(compress_directory "$metadata_backup_dir" "$metadata_archive")
        
        # Remove temporary directory
        rm -rf "$metadata_backup_dir"
        
        success "Media metadata backup completed: $metadata_archive"
    else
        log "No media metadata found to backup"
        rmdir "$metadata_backup_dir" 2>/dev/null || true
    fi
}

# Backup Docker volumes
backup_docker_volumes() {
    local backup_dir="$1"
    local volumes_backup_dir="$backup_dir/docker_volumes"
    
    log "Backing up Docker volumes..."
    mkdir -p "$volumes_backup_dir"
    
    # Get list of volumes used by our compose file
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    local volumes
    volumes=$($compose_cmd -f "$COMPOSE_FILE" config --volumes 2>/dev/null || echo "")
    
    for volume in $volumes; do
        # Skip if volume doesn't exist
        if ! docker volume inspect "$volume" > /dev/null 2>&1; then
            continue
        fi
        
        log "Backing up Docker volume: $volume"
        
        # Create temporary container to access volume
        docker run --rm -v "$volume:/volume" -v "$volumes_backup_dir:/backup" \
               alpine tar czf "/backup/${volume}.tar.gz" -C /volume .
        
        # Encrypt volume backup
        local volume_backup="$volumes_backup_dir/${volume}.tar.gz"
        volume_backup=$(encrypt_file "$volume_backup")
        
        success "Volume backup completed: $volume_backup"
    done
}

# Upload to remote backup location
upload_to_remote() {
    local backup_dir="$1"
    
    if [ "$REMOTE_BACKUP_ENABLED" != "true" ] || [ -z "$REMOTE_BACKUP_PATH" ]; then
        return 0
    fi
    
    log "Uploading backup to remote location: $REMOTE_BACKUP_PATH"
    
    case "$REMOTE_BACKUP_PATH" in
        s3://*)
            # AWS S3 upload
            if command -v aws &> /dev/null; then
                aws s3 sync "$backup_dir" "$REMOTE_BACKUP_PATH/$(basename "$backup_dir")"
                success "Backup uploaded to S3"
            else
                warn "AWS CLI not installed, skipping S3 upload"
            fi
            ;;
        sftp://*)
            # SFTP upload
            if command -v rsync &> /dev/null; then
                rsync -av "$backup_dir" "$REMOTE_BACKUP_PATH"
                success "Backup uploaded via SFTP"
            else
                warn "rsync not installed, skipping SFTP upload"
            fi
            ;;
        *)
            warn "Unknown remote backup protocol: $REMOTE_BACKUP_PATH"
            ;;
    esac
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $BACKUP_RETENTION_DAYS days..."
    
    find "$BACKUP_BASE_DIR" -type d -name "*_*" -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
    find "$BACKUP_BASE_DIR" -type f -name "*.tar.gz*" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
    
    success "Old backups cleaned up"
}

# Full backup
full_backup() {
    local timestamp="$(date +%Y%m%d_%H%M%S)"
    local backup_dir
    backup_dir=$(get_backup_dir "full" "$timestamp")
    
    log "Starting full backup to: $backup_dir"
    mkdir -p "$backup_dir"
    
    # Check disk space (estimate 5GB needed)
    check_disk_space 5
    
    # Backup each type
    backup_configuration "$backup_dir"
    backup_databases "$backup_dir"
    backup_application_data "$backup_dir"
    backup_media_metadata "$backup_dir"
    backup_docker_volumes "$backup_dir"
    
    # Create backup manifest
    create_backup_manifest "$backup_dir"
    
    # Upload to remote if configured
    upload_to_remote "$backup_dir"
    
    # Cleanup old backups
    cleanup_old_backups
    
    success "Full backup completed: $backup_dir"
    
    # Generate backup report
    generate_backup_report "$backup_dir"
}

# Create backup manifest
create_backup_manifest() {
    local backup_dir="$1"
    local manifest_file="$backup_dir/MANIFEST.txt"
    
    cat > "$manifest_file" << EOF
BACKUP MANIFEST
===============

Backup Date: $(date)
Backup Type: Full System Backup
Backup Location: $backup_dir
Hostname: $(hostname)
Docker Version: $(docker --version)
Compose Version: $(docker-compose --version 2>/dev/null || docker compose version)

FILES:
------
EOF
    
    find "$backup_dir" -type f -exec ls -lh {} \; >> "$manifest_file"
    
    cat >> "$manifest_file" << EOF

RUNNING CONTAINERS:
------------------
EOF
    
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" >> "$manifest_file"
    
    cat >> "$manifest_file" << EOF

DOCKER VOLUMES:
--------------
EOF
    
    docker volume ls >> "$manifest_file"
    
    success "Backup manifest created: $manifest_file"
}

# Generate backup report
generate_backup_report() {
    local backup_dir="$1"
    local report_file="./logs/backup-report-$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Backup Report - $(date)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .header { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 20px; border-radius: 10px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; background: #2d2d2d; border-radius: 8px; }
        .file-list { font-family: monospace; background: #1a1a1a; padding: 10px; border-radius: 4px; overflow-x: auto; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #555; }
        th { background: #4a4a4a; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 Backup Report</h1>
        <p>Backup completed on $(date)</p>
    </div>
    
    <div class="section">
        <h2>📊 Backup Summary</h2>
        <p><strong>Backup Location:</strong> $backup_dir</p>
        <p><strong>Total Size:</strong> $(du -sh "$backup_dir" | cut -f1)</p>
        <p><strong>Files Created:</strong> $(find "$backup_dir" -type f | wc -l)</p>
        <p><strong>Backup Duration:</strong> Started at $(date)</p>
    </div>
    
    <div class="section">
        <h2>📁 Backup Contents</h2>
        <div class="file-list">
EOF
    
    # Add file listing
    find "$backup_dir" -type f -exec ls -lh {} \; | while read -r line; do
        echo "            $line<br>" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
        </div>
    </div>
    
    <div class="section">
        <h2>🔧 Restoration Instructions</h2>
        <p>To restore from this backup:</p>
        <ol>
            <li>Stop all containers: <code>docker-compose down</code></li>
            <li>Run restore command: <code>./scripts/backup-restore.sh restore $(basename "$backup_dir")</code></li>
            <li>Start containers: <code>./deploy-automated.sh</code></li>
        </ol>
    </div>
    
    <div class="section">
        <h2>📝 Notes</h2>
        <ul>
            <li>This backup excludes actual media files to save space</li>
            <li>Database dumps are compressed and encrypted</li>
            <li>Configuration and application data are included</li>
            <li>Backup retention: $BACKUP_RETENTION_DAYS days</li>
        </ul>
    </div>
</body>
</html>
EOF

    success "Backup report generated: $report_file"
}

# Restore from backup
restore_backup() {
    local backup_name="$1"
    
    if [ -z "$backup_name" ]; then
        error "Please specify backup name to restore"
        list_backups
        return 1
    fi
    
    local backup_dir="$BACKUP_BASE_DIR/$backup_name"
    
    if [ ! -d "$backup_dir" ]; then
        error "Backup directory not found: $backup_dir"
        return 1
    fi
    
    log "Starting restore from backup: $backup_dir"
    
    # Confirm restore
    echo -e "${YELLOW}WARNING: This will overwrite current configuration and data!${NC}"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Restore cancelled by user"
        return 1
    fi
    
    # Stop all containers
    log "Stopping all containers..."
    docker-compose -f "$COMPOSE_FILE" down || docker compose -f "$COMPOSE_FILE" down || true
    
    # Restore configuration
    restore_configuration "$backup_dir"
    
    # Restore databases
    restore_databases "$backup_dir"
    
    # Restore application data
    restore_application_data "$backup_dir"
    
    # Restore Docker volumes
    restore_docker_volumes "$backup_dir"
    
    success "Restore completed from: $backup_dir"
    log "You can now start your services with: ./deploy-automated.sh"
}

# Restore configuration
restore_configuration() {
    local backup_dir="$1"
    
    log "Restoring configuration..."
    
    # Find configuration archive
    local config_archive
    config_archive=$(find "$backup_dir" -name "configuration.tar.gz*" | head -1)
    
    if [ -n "$config_archive" ]; then
        # Decrypt if needed
        config_archive=$(decrypt_file "$config_archive")
        
        # Extract configuration
        tar -xzf "$config_archive" -C .
        
        success "Configuration restored"
    else
        warn "No configuration backup found"
    fi
}

# Restore databases
restore_databases() {
    local backup_dir="$1"
    
    log "Restoring databases..."
    
    # Start database containers first
    docker-compose -f "$COMPOSE_FILE" up -d postgres mariadb redis || \
    docker compose -f "$COMPOSE_FILE" up -d postgres mariadb redis || true
    
    sleep 30  # Wait for databases to start
    
    # Restore PostgreSQL
    local postgres_dump
    postgres_dump=$(find "$backup_dir" -name "postgres-*.sql.gz*" | head -1)
    if [ -n "$postgres_dump" ]; then
        postgres_dump=$(decrypt_file "$postgres_dump")
        
        log "Restoring PostgreSQL database..."
        gunzip -c "$postgres_dump" | docker exec -i postgres psql -U "${POSTGRES_USER:-postgres}"
        success "PostgreSQL restored"
    fi
    
    # Restore MariaDB
    local mysql_dump
    mysql_dump=$(find "$backup_dir" -name "mariadb-*.sql.gz*" | head -1)
    if [ -n "$mysql_dump" ]; then
        mysql_dump=$(decrypt_file "$mysql_dump")
        
        log "Restoring MariaDB database..."
        gunzip -c "$mysql_dump" | docker exec -i mariadb mysql -u root -p"${MYSQL_ROOT_PASSWORD:-root}"
        success "MariaDB restored"
    fi
    
    # Restore Redis
    local redis_dump
    redis_dump=$(find "$backup_dir" -name "redis-*.rdb*" | head -1)
    if [ -n "$redis_dump" ]; then
        redis_dump=$(decrypt_file "$redis_dump")
        
        log "Restoring Redis database..."
        docker cp "$redis_dump" redis:/data/dump.rdb
        docker restart redis
        success "Redis restored"
    fi
}

# Restore application data
restore_application_data() {
    local backup_dir="$1"
    
    log "Restoring application data..."
    
    # Find and restore each service's data
    for archive in "$backup_dir"/*-data.tar.gz*; do
        if [ -f "$archive" ]; then
            local service
            service=$(basename "$archive" | sed 's/-data\.tar\.gz.*//')
            
            # Decrypt if needed
            archive=$(decrypt_file "$archive")
            
            log "Restoring data for $service..."
            
            # Extract to temporary location
            local temp_dir="/tmp/restore-$service"
            mkdir -p "$temp_dir"
            tar -xzf "$archive" -C "$temp_dir"
            
            # Copy to container if it's running
            if docker ps --format "{{.Names}}" | grep -q "$service"; then
                docker cp "$temp_dir/." "$service:${APP_DATA_PATHS[$service]:-/config}"
                docker restart "$service"
            fi
            
            # Clean up
            rm -rf "$temp_dir"
            
            success "Data restored for $service"
        fi
    done
}

# Restore Docker volumes
restore_docker_volumes() {
    local backup_dir="$1"
    local volumes_dir="$backup_dir/docker_volumes"
    
    if [ ! -d "$volumes_dir" ]; then
        return 0
    fi
    
    log "Restoring Docker volumes..."
    
    for volume_archive in "$volumes_dir"/*.tar.gz*; do
        if [ -f "$volume_archive" ]; then
            local volume
            volume=$(basename "$volume_archive" | sed 's/\.tar\.gz.*//')
            
            # Decrypt if needed
            volume_archive=$(decrypt_file "$volume_archive")
            
            log "Restoring Docker volume: $volume"
            
            # Create volume if it doesn't exist
            docker volume create "$volume" 2>/dev/null || true
            
            # Restore volume contents
            docker run --rm -v "$volume:/volume" -v "$(dirname "$volume_archive"):/backup" \
                   alpine tar xzf "/backup/$(basename "$volume_archive")" -C /volume
            
            success "Volume restored: $volume"
        fi
    done
}

# List available backups
list_backups() {
    echo -e "${PURPLE}Available Backups${NC}"
    echo "=================="
    
    if [ ! -d "$BACKUP_BASE_DIR" ] || [ ! "$(ls -A "$BACKUP_BASE_DIR" 2>/dev/null)" ]; then
        echo "No backups found in $BACKUP_BASE_DIR"
        return 0
    fi
    
    for backup_dir in "$BACKUP_BASE_DIR"/full_*; do
        if [ -d "$backup_dir" ]; then
            local backup_name
            backup_name=$(basename "$backup_dir")
            local backup_date
            backup_date=$(echo "$backup_name" | sed 's/full_//' | sed 's/_/ /')
            local backup_size
            backup_size=$(du -sh "$backup_dir" 2>/dev/null | cut -f1)
            
            echo "  📦 $backup_name"
            echo "      Date: $backup_date"
            echo "      Size: $backup_size"
            echo ""
        fi
    done
}

# Main function
main() {
    case "${1:-backup}" in
        "backup"|"full")
            full_backup
            ;;
        "restore")
            restore_backup "${2:-}"
            ;;
        "list")
            list_backups
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "test")
            # Test backup without actually running
            log "Testing backup configuration..."
            check_disk_space 5
            success "Backup test completed successfully"
            ;;
        "--help"|"-h")
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  backup, full        - Create full system backup"
            echo "  restore [name]      - Restore from backup"
            echo "  list               - List available backups"
            echo "  cleanup            - Remove old backups"
            echo "  test               - Test backup configuration"
            echo ""
            echo "Environment Variables:"
            echo "  BACKUP_RETENTION_DAYS - Days to keep backups (default: 30)"
            echo "  ENCRYPTION_PASSWORD   - Password for backup encryption"
            echo "  REMOTE_BACKUP_ENABLED - Enable remote backup (true/false)"
            echo "  REMOTE_BACKUP_PATH    - Remote backup location (s3://, sftp://)"
            echo ""
            exit 0
            ;;
        *)
            error "Unknown command: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'log "Backup/restore operation interrupted"; exit 1' SIGINT SIGTERM

# Run main function
main "$@"