#!/bin/bash

# HoloMedia Hub Backup Script
# Version: 1.0.0
# Description: Comprehensive backup solution with rotation and cloud storage support

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Backup configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="holomedia_backup_${TIMESTAMP}"
BACKUP_PATH="$BACKUP_ROOT/$BACKUP_NAME"
LOG_FILE="$PROJECT_ROOT/logs/backup.log"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Backup components
BACKUP_DATABASE=${BACKUP_DATABASE:-true}
BACKUP_FILES=${BACKUP_FILES:-true}
BACKUP_REDIS=${BACKUP_REDIS:-true}
BACKUP_CONFIG=${BACKUP_CONFIG:-true}
BACKUP_LOGS=${BACKUP_LOGS:-false}

# Storage destinations
USE_LOCAL=${USE_LOCAL:-true}
USE_S3=${USE_S3:-false}
USE_GCS=${USE_GCS:-false}
USE_AZURE=${USE_AZURE:-false}

# Progress tracking
TOTAL_STEPS=0
CURRENT_STEP=0

# Calculate total steps
calculate_steps() {
    [ "$BACKUP_DATABASE" = "true" ] && ((TOTAL_STEPS++))
    [ "$BACKUP_FILES" = "true" ] && ((TOTAL_STEPS++))
    [ "$BACKUP_REDIS" = "true" ] && ((TOTAL_STEPS++))
    [ "$BACKUP_CONFIG" = "true" ] && ((TOTAL_STEPS++))
    [ "$BACKUP_LOGS" = "true" ] && ((TOTAL_STEPS++))
    ((TOTAL_STEPS++)) # Compression step
    [ "$USE_LOCAL" = "true" ] && ((TOTAL_STEPS++))
    [ "$USE_S3" = "true" ] && ((TOTAL_STEPS++))
    [ "$USE_GCS" = "true" ] && ((TOTAL_STEPS++))
    [ "$USE_AZURE" = "true" ] && ((TOTAL_STEPS++))
    ((TOTAL_STEPS++)) # Cleanup step
}

# ASCII Art Banner
show_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ____             __                
   / __ )____ ______/ /____  ________ 
  / __  / __ `/ ___/ //_/ / / / __ \
 / /_/ / /_/ / /__/ ,< / /_/ / /_/ /
/_____/\__,_/\___/_/|_|\__,_/ .___/ 
                            /_/      
EOF
    echo -e "${NC}"
}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Progress bar
progress_bar() {
    ((CURRENT_STEP++))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %d%%" "$percentage"
}

# Error handler
error_exit() {
    echo -e "\n${RED}Error: $1${NC}" >&2
    log "ERROR: $1"
    
    # Cleanup partial backup
    if [ -d "$BACKUP_PATH" ]; then
        rm -rf "$BACKUP_PATH"
    fi
    
    exit 1
}

# Success message
success() {
    echo -e "\n${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "\n${YELLOW}⚠ $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "\n${BLUE}ℹ $1${NC}"
    log "INFO: $1"
}

# Create backup directory structure
create_backup_structure() {
    mkdir -p "$BACKUP_PATH"/{database,files,redis,config,logs}
    mkdir -p "$BACKUP_ROOT"
    mkdir -p "$(dirname "$LOG_FILE")"
}

# Backup PostgreSQL database
backup_database() {
    if [ "$BACKUP_DATABASE" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Backing up PostgreSQL database...${NC}"
    
    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"
    DB_NAME="${DB_NAME:-holomedia}"
    DB_USER="${DB_USER:-holomedia}"
    
    if [ -z "$DB_PASSWORD" ]; then
        error_exit "Database password not set in environment"
    fi
    
    # Check if pg_dump is available
    if ! command -v pg_dump &> /dev/null; then
        warning "pg_dump not found. Trying Docker..."
        
        # Try using Docker container
        if docker ps --format "{{.Names}}" | grep -q "holomedia-postgres"; then
            docker exec holomedia-postgres pg_dump \
                -U "$DB_USER" \
                -d "$DB_NAME" \
                --no-owner \
                --clean \
                --if-exists \
                > "$BACKUP_PATH/database/postgres_dump.sql"
            
            # Also dump globals (roles, etc)
            docker exec holomedia-postgres pg_dumpall \
                -U "$DB_USER" \
                --globals-only \
                > "$BACKUP_PATH/database/postgres_globals.sql"
        else
            error_exit "PostgreSQL backup failed: pg_dump not available"
        fi
    else
        # Use native pg_dump
        PGPASSWORD="$DB_PASSWORD" pg_dump \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            --no-owner \
            --clean \
            --if-exists \
            > "$BACKUP_PATH/database/postgres_dump.sql"
        
        # Also dump globals
        PGPASSWORD="$DB_PASSWORD" pg_dumpall \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            --globals-only \
            > "$BACKUP_PATH/database/postgres_globals.sql"
    fi
    
    # Compress database dump
    gzip "$BACKUP_PATH/database/postgres_dump.sql"
    gzip "$BACKUP_PATH/database/postgres_globals.sql"
    
    # Get database size
    DB_SIZE=$(du -sh "$BACKUP_PATH/database" | cut -f1)
    success "Database backed up ($DB_SIZE)"
}

# Backup uploaded files
backup_files() {
    if [ "$BACKUP_FILES" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Backing up uploaded files...${NC}"
    
    UPLOAD_DIR="${UPLOAD_DIR:-$PROJECT_ROOT/uploads}"
    
    if [ -d "$UPLOAD_DIR" ]; then
        # Create tar archive of uploads
        tar -czf "$BACKUP_PATH/files/uploads.tar.gz" \
            -C "$(dirname "$UPLOAD_DIR")" \
            "$(basename "$UPLOAD_DIR")" \
            --exclude="*.tmp" \
            --exclude="thumbs.db" \
            --exclude=".DS_Store"
        
        FILES_SIZE=$(du -sh "$BACKUP_PATH/files/uploads.tar.gz" | cut -f1)
        success "Files backed up ($FILES_SIZE)"
    else
        warning "Upload directory not found: $UPLOAD_DIR"
    fi
}

# Backup Redis data
backup_redis() {
    if [ "$BACKUP_REDIS" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Backing up Redis data...${NC}"
    
    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"
    
    if command -v redis-cli &> /dev/null; then
        # Trigger Redis BGSAVE
        if [ -n "$REDIS_PASSWORD" ]; then
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGSAVE
        else
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE
        fi
        
        # Wait for background save to complete
        sleep 2
        
        # Copy Redis dump file
        if docker ps --format "{{.Names}}" | grep -q "holomedia-redis"; then
            docker cp holomedia-redis:/data/dump.rdb "$BACKUP_PATH/redis/dump.rdb"
            success "Redis data backed up"
        else
            warning "Redis container not found, skipping Redis backup"
        fi
    else
        warning "redis-cli not found, trying Docker method..."
        
        if docker ps --format "{{.Names}}" | grep -q "holomedia-redis"; then
            docker exec holomedia-redis redis-cli BGSAVE
            sleep 2
            docker cp holomedia-redis:/data/dump.rdb "$BACKUP_PATH/redis/dump.rdb"
            success "Redis data backed up via Docker"
        else
            warning "Redis backup skipped: redis-cli not available"
        fi
    fi
}

# Backup configuration files
backup_config() {
    if [ "$BACKUP_CONFIG" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Backing up configuration files...${NC}"
    
    # List of config files to backup
    CONFIG_FILES=(
        ".env"
        "docker-compose.yml"
        "docker/docker-compose.full.yml"
        "config/*.json"
        "config/*.yml"
        "nginx/*.conf"
        "ssl/*.pem"
    )
    
    # Create config backup
    for pattern in "${CONFIG_FILES[@]}"; do
        if ls $PROJECT_ROOT/$pattern 1> /dev/null 2>&1; then
            # Create directory structure
            dir=$(dirname "$pattern")
            mkdir -p "$BACKUP_PATH/config/$dir"
            
            # Copy files
            cp -r $PROJECT_ROOT/$pattern "$BACKUP_PATH/config/$dir/" 2>/dev/null || true
        fi
    done
    
    # Remove sensitive data from .env backup
    if [ -f "$BACKUP_PATH/config/.env" ]; then
        sed -i.bak 's/\(PASSWORD=\).*/\1[REDACTED]/' "$BACKUP_PATH/config/.env"
        sed -i.bak 's/\(SECRET=\).*/\1[REDACTED]/' "$BACKUP_PATH/config/.env"
        sed -i.bak 's/\(API_KEY=\).*/\1[REDACTED]/' "$BACKUP_PATH/config/.env"
        rm "$BACKUP_PATH/config/.env.bak"
    fi
    
    success "Configuration files backed up"
}

# Backup logs (optional)
backup_logs() {
    if [ "$BACKUP_LOGS" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Backing up application logs...${NC}"
    
    LOG_DIR="$PROJECT_ROOT/logs"
    
    if [ -d "$LOG_DIR" ]; then
        # Backup only recent logs (last 7 days)
        find "$LOG_DIR" -name "*.log" -mtime -7 -exec cp {} "$BACKUP_PATH/logs/" \;
        
        # Compress logs
        if [ "$(ls -A "$BACKUP_PATH/logs/")" ]; then
            tar -czf "$BACKUP_PATH/logs/logs.tar.gz" -C "$BACKUP_PATH/logs" . --remove-files
            success "Logs backed up"
        else
            info "No recent logs to backup"
        fi
    else
        warning "Log directory not found"
    fi
}

# Create compressed archive
create_archive() {
    progress_bar
    echo -e "\n${CYAN}Creating backup archive...${NC}"
    
    cd "$BACKUP_ROOT"
    
    # Create main archive
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    
    # Create checksum
    if command -v sha256sum &> /dev/null; then
        sha256sum "${BACKUP_NAME}.tar.gz" > "${BACKUP_NAME}.tar.gz.sha256"
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "${BACKUP_NAME}.tar.gz" > "${BACKUP_NAME}.tar.gz.sha256"
    fi
    
    # Get final size
    ARCHIVE_SIZE=$(du -sh "${BACKUP_NAME}.tar.gz" | cut -f1)
    
    # Remove uncompressed backup
    rm -rf "$BACKUP_NAME"
    
    success "Archive created: ${BACKUP_NAME}.tar.gz ($ARCHIVE_SIZE)"
}

# Upload to S3
upload_to_s3() {
    if [ "$USE_S3" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Uploading to AWS S3...${NC}"
    
    if [ -z "$AWS_S3_BUCKET" ]; then
        warning "S3 bucket not configured, skipping S3 upload"
        return
    fi
    
    if command -v aws &> /dev/null; then
        aws s3 cp "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz" \
            "s3://${AWS_S3_BUCKET}/backups/${BACKUP_NAME}.tar.gz" \
            --storage-class "${AWS_STORAGE_CLASS:-STANDARD_IA}"
        
        aws s3 cp "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz.sha256" \
            "s3://${AWS_S3_BUCKET}/backups/${BACKUP_NAME}.tar.gz.sha256"
        
        success "Uploaded to S3: s3://${AWS_S3_BUCKET}/backups/"
    else
        warning "AWS CLI not installed, skipping S3 upload"
    fi
}

# Upload to Google Cloud Storage
upload_to_gcs() {
    if [ "$USE_GCS" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Uploading to Google Cloud Storage...${NC}"
    
    if [ -z "$GCS_BUCKET" ]; then
        warning "GCS bucket not configured, skipping GCS upload"
        return
    fi
    
    if command -v gsutil &> /dev/null; then
        gsutil cp "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz" \
            "gs://${GCS_BUCKET}/backups/${BACKUP_NAME}.tar.gz"
        
        gsutil cp "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz.sha256" \
            "gs://${GCS_BUCKET}/backups/${BACKUP_NAME}.tar.gz.sha256"
        
        success "Uploaded to GCS: gs://${GCS_BUCKET}/backups/"
    else
        warning "gsutil not installed, skipping GCS upload"
    fi
}

# Upload to Azure Blob Storage
upload_to_azure() {
    if [ "$USE_AZURE" != "true" ]; then
        return
    fi
    
    progress_bar
    echo -e "\n${CYAN}Uploading to Azure Blob Storage...${NC}"
    
    if [ -z "$AZURE_CONTAINER" ]; then
        warning "Azure container not configured, skipping Azure upload"
        return
    fi
    
    if command -v az &> /dev/null; then
        az storage blob upload \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$AZURE_CONTAINER" \
            --name "backups/${BACKUP_NAME}.tar.gz" \
            --file "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz"
        
        az storage blob upload \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$AZURE_CONTAINER" \
            --name "backups/${BACKUP_NAME}.tar.gz.sha256" \
            --file "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz.sha256"
        
        success "Uploaded to Azure: ${AZURE_CONTAINER}/backups/"
    else
        warning "Azure CLI not installed, skipping Azure upload"
    fi
}

# Clean up old backups
cleanup_old_backups() {
    progress_bar
    echo -e "\n${CYAN}Cleaning up old backups...${NC}"
    
    # Local cleanup
    if [ "$USE_LOCAL" = "true" ]; then
        find "$BACKUP_ROOT" -name "holomedia_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete
        find "$BACKUP_ROOT" -name "holomedia_backup_*.tar.gz.sha256" -mtime +$RETENTION_DAYS -delete
        
        LOCAL_COUNT=$(find "$BACKUP_ROOT" -name "holomedia_backup_*.tar.gz" | wc -l)
        info "Local backups retained: $LOCAL_COUNT"
    fi
    
    # S3 cleanup
    if [ "$USE_S3" = "true" ] && [ -n "$AWS_S3_BUCKET" ] && command -v aws &> /dev/null; then
        aws s3 ls "s3://${AWS_S3_BUCKET}/backups/" | while read -r line; do
            createDate=$(echo "$line" | awk '{print $1" "$2}')
            createDate=$(date -d "$createDate" +%s)
            olderThan=$(date -d "$RETENTION_DAYS days ago" +%s)
            
            if [[ $createDate -lt $olderThan ]]; then
                fileName=$(echo "$line" | awk '{print $4}')
                if [[ $fileName == holomedia_backup_* ]]; then
                    aws s3 rm "s3://${AWS_S3_BUCKET}/backups/$fileName"
                fi
            fi
        done
    fi
    
    success "Old backups cleaned up"
}

# Restore function
restore_backup() {
    local backup_file=$1
    
    if [ -z "$backup_file" ]; then
        echo -e "${RED}Error: No backup file specified${NC}"
        echo "Usage: $0 --restore <backup_file>"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    echo -e "${YELLOW}⚠️  WARNING: This will restore from backup and overwrite current data!${NC}"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Restore cancelled"
        exit 0
    fi
    
    info "Starting restore from: $backup_file"
    
    # Create restore directory
    RESTORE_DIR="/tmp/holomedia_restore_$$"
    mkdir -p "$RESTORE_DIR"
    
    # Extract backup
    tar -xzf "$backup_file" -C "$RESTORE_DIR"
    
    # Find the backup directory
    BACKUP_DIR=$(find "$RESTORE_DIR" -maxdepth 1 -name "holomedia_backup_*" -type d | head -n1)
    
    if [ -z "$BACKUP_DIR" ]; then
        error_exit "Invalid backup file format"
    fi
    
    # Restore database
    if [ -f "$BACKUP_DIR/database/postgres_dump.sql.gz" ]; then
        info "Restoring database..."
        gunzip -c "$BACKUP_DIR/database/postgres_dump.sql.gz" | \
            PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
        success "Database restored"
    fi
    
    # Restore files
    if [ -f "$BACKUP_DIR/files/uploads.tar.gz" ]; then
        info "Restoring uploaded files..."
        tar -xzf "$BACKUP_DIR/files/uploads.tar.gz" -C "$PROJECT_ROOT"
        success "Files restored"
    fi
    
    # Restore Redis
    if [ -f "$BACKUP_DIR/redis/dump.rdb" ]; then
        info "Restoring Redis data..."
        if docker ps --format "{{.Names}}" | grep -q "holomedia-redis"; then
            docker cp "$BACKUP_DIR/redis/dump.rdb" holomedia-redis:/data/dump.rdb
            docker restart holomedia-redis
            success "Redis data restored"
        else
            warning "Redis container not running, skipping Redis restore"
        fi
    fi
    
    # Clean up
    rm -rf "$RESTORE_DIR"
    
    success "Restore completed successfully!"
}

# Backup status
backup_status() {
    echo -e "${CYAN}Backup Status Report${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    # Local backups
    echo -e "${WHITE}Local Backups:${NC}"
    if [ -d "$BACKUP_ROOT" ]; then
        local count=$(find "$BACKUP_ROOT" -name "holomedia_backup_*.tar.gz" | wc -l)
        local size=$(du -sh "$BACKUP_ROOT" 2>/dev/null | cut -f1)
        echo "  Location: $BACKUP_ROOT"
        echo "  Count: $count backups"
        echo "  Total Size: $size"
        
        echo -e "\n  Recent backups:"
        find "$BACKUP_ROOT" -name "holomedia_backup_*.tar.gz" -printf "  - %f (%s bytes) - %TY-%Tm-%Td %TH:%TM\n" | sort -r | head -5
    else
        echo "  No local backups found"
    fi
    
    # S3 backups
    if [ "$USE_S3" = "true" ] && [ -n "$AWS_S3_BUCKET" ] && command -v aws &> /dev/null; then
        echo -e "\n${WHITE}S3 Backups:${NC}"
        echo "  Bucket: s3://$AWS_S3_BUCKET/backups/"
        local s3_count=$(aws s3 ls "s3://${AWS_S3_BUCKET}/backups/" | grep -c "holomedia_backup_" || echo "0")
        echo "  Count: $s3_count backups"
    fi
    
    # Next scheduled backup
    if command -v systemctl &> /dev/null && systemctl is-active --quiet holomedia-backup.timer; then
        echo -e "\n${WHITE}Scheduled Backups:${NC}"
        systemctl status holomedia-backup.timer --no-pager | grep "Trigger:"
    fi
    
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Show help
show_help() {
    echo "HoloMedia Hub Backup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --restore FILE  Restore from backup file"
    echo "  --status        Show backup status"
    echo "  --dry-run       Show what would be backed up"
    echo "  --full          Force full backup of all components"
    echo "  --database      Backup only database"
    echo "  --files         Backup only files"
    echo ""
    echo "Environment Variables:"
    echo "  BACKUP_DIR              Custom backup directory"
    echo "  BACKUP_RETENTION_DAYS   Days to keep backups (default: 30)"
    echo "  USE_S3                  Upload to S3 (true/false)"
    echo "  USE_GCS                 Upload to Google Cloud Storage (true/false)"
    echo "  USE_AZURE               Upload to Azure Blob Storage (true/false)"
}

# Main backup flow
main() {
    # Parse command line arguments
    case "${1:-}" in
        --help)
            show_help
            exit 0
            ;;
        --restore)
            restore_backup "$2"
            exit 0
            ;;
        --status)
            backup_status
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --full)
            BACKUP_DATABASE=true
            BACKUP_FILES=true
            BACKUP_REDIS=true
            BACKUP_CONFIG=true
            BACKUP_LOGS=true
            ;;
        --database)
            BACKUP_DATABASE=true
            BACKUP_FILES=false
            BACKUP_REDIS=false
            BACKUP_CONFIG=false
            BACKUP_LOGS=false
            ;;
        --files)
            BACKUP_DATABASE=false
            BACKUP_FILES=true
            BACKUP_REDIS=false
            BACKUP_CONFIG=false
            BACKUP_LOGS=false
            ;;
    esac
    
    show_banner
    
    # Initialize
    log "Starting backup process"
    echo -e "${CYAN}Starting HoloMedia Hub backup...${NC}"
    echo -e "Timestamp: $(date)\n"
    
    # Calculate total steps
    calculate_steps
    
    # Create backup structure
    create_backup_structure
    
    # Perform backups
    backup_database
    backup_files
    backup_redis
    backup_config
    backup_logs
    
    # Create archive
    create_archive
    
    # Upload to cloud storage
    upload_to_s3
    upload_to_gcs
    upload_to_azure
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Summary
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Backup completed successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${CYAN}Backup Summary:${NC}"
    echo "• Backup Name: ${BACKUP_NAME}.tar.gz"
    echo "• Size: $ARCHIVE_SIZE"
    echo "• Location: $BACKUP_ROOT"
    
    if [ "$USE_S3" = "true" ] && [ -n "$AWS_S3_BUCKET" ]; then
        echo "• S3: s3://${AWS_S3_BUCKET}/backups/"
    fi
    
    if [ "$USE_GCS" = "true" ] && [ -n "$GCS_BUCKET" ]; then
        echo "• GCS: gs://${GCS_BUCKET}/backups/"
    fi
    
    if [ "$USE_AZURE" = "true" ] && [ -n "$AZURE_CONTAINER" ]; then
        echo "• Azure: ${AZURE_CONTAINER}/backups/"
    fi
    
    echo -e "\n${CYAN}To restore from this backup:${NC}"
    echo "  $0 --restore $BACKUP_ROOT/${BACKUP_NAME}.tar.gz"
    
    log "Backup completed: ${BACKUP_NAME}.tar.gz ($ARCHIVE_SIZE)"
}

# Run main function
main "$@"