#!/bin/bash

# Media Server - Restore Script
# Restore from backup archives
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
readonly RESTORE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
fi

# Restore configuration
readonly CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config}"
readonly DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data}"

# Backup ID from command line
BACKUP_ID="${1:-}"

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

# Show restore banner
show_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    ____           __                
   / __ \___  ____/ /_____  ________ 
  / /_/ / _ \/ __  __/ __ \/ ___/ _ \
 / _, _/  __(__  ) /_/ /_/ / /  /  __/
/_/ |_|\___/____/\__/\____/_/   \___/ 
                                      
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Media Server Restore System${NC}\n"
}

# List available backups
list_backups() {
    log_header "Available Backups"
    
    # Find compressed backups
    local compressed_backups=$(find "$BACKUP_ROOT" -name "media-server-backup-*.tar.gz" 2>/dev/null | sort -r)
    
    # Find uncompressed backups
    local uncompressed_backups=$(find "$BACKUP_ROOT" -type d -name "[0-9]*_[0-9]*" 2>/dev/null | sort -r)
    
    if [ -z "$compressed_backups" ] && [ -z "$uncompressed_backups" ]; then
        log_error "No backups found in $BACKUP_ROOT"
        return 1
    fi
    
    echo "Select a backup to restore:"
    echo
    
    local index=1
    declare -a backup_list
    
    # List compressed backups
    if [ -n "$compressed_backups" ]; then
        echo -e "${CYAN}Compressed Backups:${NC}"
        for backup in $compressed_backups; do
            local backup_name=$(basename "$backup")
            local backup_size=$(du -sh "$backup" | awk '{print $1}')
            local backup_date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$backup" 2>/dev/null || stat -c "%y" "$backup" 2>/dev/null | cut -d' ' -f1-2)
            
            echo "  $index) $backup_name ($backup_size) - $backup_date"
            backup_list[$index]="$backup"
            ((index++))
        done
    fi
    
    # List uncompressed backups
    if [ -n "$uncompressed_backups" ]; then
        echo -e "\n${CYAN}Uncompressed Backups:${NC}"
        for backup in $uncompressed_backups; do
            local backup_name=$(basename "$backup")
            local backup_size=$(du -sh "$backup" | awk '{print $1}')
            
            echo "  $index) $backup_name ($backup_size)"
            backup_list[$index]="$backup"
            ((index++))
        done
    fi
    
    echo
    read -p "Enter backup number (1-$((index-1))): " selection
    
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -lt "$index" ]; then
        BACKUP_PATH="${backup_list[$selection]}"
        BACKUP_ID=$(basename "$BACKUP_PATH" | sed 's/media-server-backup-//' | sed 's/.tar.gz//')
        log_info "Selected: $(basename "$BACKUP_PATH")"
        return 0
    else
        log_error "Invalid selection"
        return 1
    fi
}

# Verify backup
verify_backup() {
    log_header "Verifying Backup"
    
    if [ -z "$BACKUP_ID" ]; then
        list_backups || exit 1
    else
        # Find backup by ID
        if [ -f "${BACKUP_ROOT}/media-server-backup-${BACKUP_ID}.tar.gz" ]; then
            BACKUP_PATH="${BACKUP_ROOT}/media-server-backup-${BACKUP_ID}.tar.gz"
        elif [ -d "${BACKUP_ROOT}/${BACKUP_ID}" ]; then
            BACKUP_PATH="${BACKUP_ROOT}/${BACKUP_ID}"
        else
            log_error "Backup not found: $BACKUP_ID"
            list_backups || exit 1
        fi
    fi
    
    log_info "Backup path: $BACKUP_PATH"
    
    # Extract backup if compressed
    if [[ "$BACKUP_PATH" == *.tar.gz ]]; then
        log_info "Extracting compressed backup..."
        
        TEMP_DIR="${BACKUP_ROOT}/restore_temp_${RESTORE_TIMESTAMP}"
        mkdir -p "$TEMP_DIR"
        
        tar -xzf "$BACKUP_PATH" -C "$TEMP_DIR"
        
        # Find extracted directory
        EXTRACTED_DIR=$(find "$TEMP_DIR" -type d -name "[0-9]*_[0-9]*" | head -1)
        if [ -z "$EXTRACTED_DIR" ]; then
            log_error "Failed to extract backup"
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        
        BACKUP_DIR="$EXTRACTED_DIR"
    else
        BACKUP_DIR="$BACKUP_PATH"
    fi
    
    # Verify backup structure
    if [ ! -d "$BACKUP_DIR/config" ]; then
        log_error "Invalid backup structure - missing config directory"
        exit 1
    fi
    
    log_info "Backup verified successfully"
}

# Create restore point
create_restore_point() {
    log_header "Creating Restore Point"
    
    log_info "Creating backup of current configuration..."
    
    # Run backup script with special tag
    "${SCRIPT_DIR}/backup.sh" || log_warn "Failed to create restore point"
}

# Stop services
stop_services() {
    log_header "Stopping Services"
    
    log_info "Stopping all services..."
    docker-compose down
    sleep 5
}

# Restore configurations
restore_configs() {
    log_header "Restoring Configurations"
    
    # Backup current configs
    if [ -d "$CONFIG_PATH" ]; then
        log_info "Backing up current configuration..."
        mv "$CONFIG_PATH" "${CONFIG_PATH}.before_restore_${RESTORE_TIMESTAMP}"
    fi
    
    # Create config directory
    mkdir -p "$CONFIG_PATH"
    
    # Restore each service configuration
    for service_dir in "$BACKUP_DIR/config"/*; do
        if [ -d "$service_dir" ]; then
            local service=$(basename "$service_dir")
            log_info "Restoring $service configuration..."
            
            mkdir -p "$CONFIG_PATH/$service"
            
            # Use rsync if available
            if command -v rsync &> /dev/null; then
                rsync -av "$service_dir/" "$CONFIG_PATH/$service/"
            else
                cp -R "$service_dir/"* "$CONFIG_PATH/$service/" 2>/dev/null || true
            fi
            
            log_info "  └─ $service restored"
        fi
    done
}

# Restore databases
restore_databases() {
    log_header "Restoring Databases"
    
    if [ -d "$BACKUP_DIR/databases" ]; then
        for db_dir in "$BACKUP_DIR/databases"/*; do
            if [ -d "$db_dir" ]; then
                local service=$(basename "$db_dir")
                log_info "Restoring $service database..."
                
                # Find database files
                for db_file in "$db_dir"/*.db "$db_dir"/*.sqlite3; do
                    if [ -f "$db_file" ]; then
                        local db_name=$(basename "$db_file")
                        local target_path=""
                        
                        # Determine target path based on service
                        case "$service" in
                            "overseerr")
                                target_path="$CONFIG_PATH/$service/db/$db_name"
                                mkdir -p "$CONFIG_PATH/$service/db"
                                ;;
                            *)
                                target_path="$CONFIG_PATH/$service/$db_name"
                                ;;
                        esac
                        
                        cp "$db_file" "$target_path"
                        log_info "  └─ $db_name restored"
                    fi
                done
            fi
        done
    else
        log_warn "No database backups found"
    fi
}

# Restore Docker volumes
restore_docker_volumes() {
    log_header "Restoring Docker Volumes"
    
    if [ -d "$BACKUP_DIR/volumes" ]; then
        for volume_archive in "$BACKUP_DIR/volumes"/*.tar.gz; do
            if [ -f "$volume_archive" ]; then
                local volume_name=$(basename "$volume_archive" .tar.gz)
                log_info "Restoring volume: $volume_name"
                
                # Create volume if it doesn't exist
                docker volume create "$volume_name" 2>/dev/null || true
                
                # Restore volume data
                docker run --rm \
                    -v "$volume_name":/target \
                    -v "$volume_archive":/backup.tar.gz:ro \
                    alpine sh -c "cd /target && tar -xzf /backup.tar.gz"
                
                log_info "  └─ Volume restored"
            fi
        done
    fi
}

# Restore environment file
restore_environment() {
    log_header "Restoring Environment Configuration"
    
    if [ -f "$BACKUP_DIR/metadata/env.sample" ]; then
        log_warn "Environment sample found - manual configuration required"
        
        read -p "View environment sample? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cat "$BACKUP_DIR/metadata/env.sample"
            echo
            log_info "Update your .env file with any missing configurations"
        fi
    fi
}

# Set permissions
set_permissions() {
    log_header "Setting Permissions"
    
    # Get current user and group
    local current_user=$(id -u)
    local current_group=$(id -g)
    
    # Set ownership
    log_info "Setting ownership to $current_user:$current_group"
    chown -R "$current_user:$current_group" "$CONFIG_PATH" 2>/dev/null || true
    
    # Set permissions
    find "$CONFIG_PATH" -type d -exec chmod 755 {} \; 2>/dev/null || true
    find "$CONFIG_PATH" -type f -exec chmod 644 {} \; 2>/dev/null || true
    
    log_info "Permissions set"
}

# Start services
start_services() {
    log_header "Starting Services"
    
    log_info "Starting restored services..."
    docker-compose up -d
    
    # Wait for services to start
    sleep 10
    
    # Check service status
    docker-compose ps
}

# Cleanup temporary files
cleanup() {
    if [ -n "${TEMP_DIR:-}" ] && [ -d "$TEMP_DIR" ]; then
        log_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
}

# Show restore summary
show_summary() {
    log_header "Restore Summary"
    
    echo -e "${GREEN}✅ Restore completed successfully!${NC}\n"
    
    echo -e "${CYAN}Restored Components:${NC}"
    echo "  ✓ Service configurations"
    echo "  ✓ Application databases"
    echo "  ✓ Docker volumes"
    
    echo -e "\n${CYAN}Post-Restore Steps:${NC}"
    echo "1. Verify services are running: docker-compose ps"
    echo "2. Check service health: ./scripts/deploy/health-check.sh"
    echo "3. Update any API keys or passwords in .env"
    echo "4. Test service connectivity"
    
    echo -e "\n${CYAN}Rollback Option:${NC}"
    echo "  Previous config saved at: ${CONFIG_PATH}.before_restore_${RESTORE_TIMESTAMP}"
}

# Main restore flow
main() {
    show_banner
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Verify backup
    verify_backup
    
    # Confirm restore
    echo -e "\n${YELLOW}⚠️  WARNING: This will replace current configuration!${NC}"
    read -p "Continue with restore? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    # Create restore point
    create_restore_point
    
    # Stop services
    stop_services
    
    # Perform restore
    restore_configs
    restore_databases
    restore_docker_volumes
    restore_environment
    
    # Set permissions
    set_permissions
    
    # Start services
    start_services
    
    # Show summary
    show_summary
}

# Run main function
main "$@"