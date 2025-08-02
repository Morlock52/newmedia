#!/bin/bash

# Media Server Backup Automation Script
# Performs scheduled backups of media server configuration and databases
# Version: 2025.1

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="/backup/media-server"
TEMP_DIR="/tmp/media-backup-$$"
LOG_DIR="${BACKUP_ROOT}/logs"
LOG_FILE="${LOG_DIR}/backup-$(date +%Y%m%d_%H%M%S).log"

# Backup retention (days)
DAILY_RETENTION=7
WEEKLY_RETENTION=30
MONTHLY_RETENTION=180

# Services to backup
declare -A SERVICES=(
    ["jellyfin"]="/config/jellyfin"
    ["sonarr"]="/config/sonarr"
    ["radarr"]="/config/radarr"
    ["lidarr"]="/config/lidarr"
    ["readarr"]="/config/readarr"
    ["prowlarr"]="/config/prowlarr"
    ["bazarr"]="/config/bazarr"
    ["qbittorrent"]="/config/qbittorrent"
    ["fileflows"]="/config/fileflows"
    ["overseerr"]="/config/overseerr"
    ["tautulli"]="/config/tautulli"
)

# Docker containers
declare -A CONTAINERS=(
    ["jellyfin"]="jellyfin"
    ["sonarr"]="sonarr"
    ["radarr"]="radarr"
    ["lidarr"]="lidarr"
    ["readarr"]="readarr"
    ["prowlarr"]="prowlarr"
    ["bazarr"]="bazarr"
    ["qbittorrent"]="qbittorrent"
    ["fileflows"]="fileflows"
    ["overseerr"]="overseerr"
    ["tautulli"]="tautulli"
)

# Create directories
mkdir -p "${BACKUP_ROOT}"/{daily,weekly,monthly,logs,temp}
mkdir -p "${TEMP_DIR}"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    log "INFO" "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}"
}

# Trap cleanup on exit
trap cleanup EXIT

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local required_tools=("docker" "tar" "gzip" "rsync" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "$tool is required but not installed"
        fi
    done
    
    # Check disk space
    local available_space=$(df -BG "${BACKUP_ROOT}" | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( available_space < 10 )); then
        log "WARN" "Low disk space: ${available_space}GB available"
    fi
}

# Stop container
stop_container() {
    local container=$1
    
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        log "INFO" "Stopping container: ${container}"
        docker stop "${container}" || log "WARN" "Failed to stop ${container}"
        return 0
    else
        log "WARN" "Container ${container} is not running"
        return 1
    fi
}

# Start container
start_container() {
    local container=$1
    
    log "INFO" "Starting container: ${container}"
    docker start "${container}" || log "WARN" "Failed to start ${container}"
}

# Backup database
backup_database() {
    local service=$1
    local backup_dir=$2
    
    case $service in
        "jellyfin")
            # Jellyfin uses SQLite databases
            local db_files=(
                "jellyfin.db"
                "library.db"
                "users.db"
            )
            
            for db in "${db_files[@]}"; do
                if [[ -f "${SERVICES[$service]}/data/${db}" ]]; then
                    log "INFO" "Backing up ${service} database: ${db}"
                    sqlite3 "${SERVICES[$service]}/data/${db}" ".backup '${backup_dir}/${db}'"
                fi
            done
            ;;
            
        "sonarr"|"radarr"|"lidarr"|"readarr"|"prowlarr")
            # *arr apps use SQLite
            local db_file="${service}.db"
            
            if [[ -f "${SERVICES[$service]}/${db_file}" ]]; then
                log "INFO" "Backing up ${service} database"
                sqlite3 "${SERVICES[$service]}/${db_file}" ".backup '${backup_dir}/${db_file}'"
            fi
            ;;
            
        "bazarr")
            # Bazarr database
            if [[ -f "${SERVICES[$service]}/db/bazarr.db" ]]; then
                log "INFO" "Backing up Bazarr database"
                sqlite3 "${SERVICES[$service]}/db/bazarr.db" ".backup '${backup_dir}/bazarr.db'"
            fi
            ;;
    esac
}

# Backup service configuration
backup_service() {
    local service=$1
    local backup_type=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="${service}_${timestamp}"
    local service_backup_dir="${TEMP_DIR}/${backup_name}"
    
    log "INFO" "Backing up ${service}..."
    
    # Create service backup directory
    mkdir -p "${service_backup_dir}"
    
    # Check if container exists
    local container="${CONTAINERS[$service]}"
    local container_was_running=false
    
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        container_was_running=true
        
        # Stop container for consistent backup (optional for some services)
        if [[ "$service" =~ ^(jellyfin|sonarr|radarr|bazarr)$ ]]; then
            stop_container "${container}"
            sleep 2
        fi
    fi
    
    # Backup configuration files
    if [[ -d "${SERVICES[$service]}" ]]; then
        log "INFO" "Copying configuration files for ${service}"
        
        # Use rsync for efficient copying
        rsync -av --exclude='*.log' \
                  --exclude='*.tmp' \
                  --exclude='MediaCover/' \
                  --exclude='logs/' \
                  --exclude='Updates/' \
                  --exclude='Backups/' \
                  "${SERVICES[$service]}/" "${service_backup_dir}/config/"
    else
        log "WARN" "Configuration directory not found for ${service}: ${SERVICES[$service]}"
    fi
    
    # Backup databases
    mkdir -p "${service_backup_dir}/database"
    backup_database "$service" "${service_backup_dir}/database"
    
    # Export Docker container settings
    if docker inspect "${container}" &> /dev/null; then
        log "INFO" "Exporting Docker configuration for ${container}"
        docker inspect "${container}" > "${service_backup_dir}/docker-config.json"
        
        # Export environment variables
        docker exec "${container}" env > "${service_backup_dir}/environment.txt" 2>/dev/null || true
    fi
    
    # Service-specific exports
    case $service in
        "jellyfin")
            # Export library settings
            if $container_was_running; then
                log "INFO" "Exporting Jellyfin library configuration"
                # Note: This would require Jellyfin API access
            fi
            ;;
            
        "sonarr"|"radarr"|"lidarr"|"readarr")
            # Export profiles and settings via API
            if $container_was_running && [[ -n "${API_KEYS[$service]:-}" ]]; then
                log "INFO" "Exporting ${service} settings via API"
                
                # Export quality profiles
                curl -s "http://localhost:${PORTS[$service]}/api/v3/qualityprofile" \
                     -H "X-Api-Key: ${API_KEYS[$service]}" \
                     > "${service_backup_dir}/quality-profiles.json" 2>/dev/null || true
                
                # Export indexers
                curl -s "http://localhost:${PORTS[$service]}/api/v3/indexer" \
                     -H "X-Api-Key: ${API_KEYS[$service]}" \
                     > "${service_backup_dir}/indexers.json" 2>/dev/null || true
            fi
            ;;
    esac
    
    # Restart container if it was running
    if $container_was_running; then
        start_container "${container}"
    fi
    
    # Create tarball
    local archive_name="${backup_name}.tar.gz"
    log "INFO" "Creating archive: ${archive_name}"
    
    cd "${TEMP_DIR}"
    tar -czf "${archive_name}" "${backup_name}/"
    
    # Move to appropriate backup directory
    local dest_dir="${BACKUP_ROOT}/${backup_type}"
    mv "${archive_name}" "${dest_dir}/"
    
    # Calculate checksum
    cd "${dest_dir}"
    sha256sum "${archive_name}" > "${archive_name}.sha256"
    
    # Cleanup temp files
    rm -rf "${service_backup_dir}"
    
    log "INFO" "Backup completed for ${service}: ${dest_dir}/${archive_name}"
}

# Backup all services
backup_all_services() {
    local backup_type=$1
    
    log "INFO" "Starting ${backup_type} backup of all services"
    
    for service in "${!SERVICES[@]}"; do
        backup_service "$service" "$backup_type" || log "ERROR" "Failed to backup ${service}"
    done
}

# Backup media library snapshots
backup_media_snapshots() {
    local snapshot_dir="${BACKUP_ROOT}/snapshots/$(date +%Y%m%d)"
    mkdir -p "${snapshot_dir}"
    
    log "INFO" "Creating media library snapshots"
    
    # Create file lists for each media directory
    local media_dirs=(
        "/media/movies"
        "/media/tv"
        "/media/music"
        "/media/books"
    )
    
    for dir in "${media_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local dirname=$(basename "$dir")
            log "INFO" "Creating snapshot for ${dirname}"
            
            # Create detailed file list with checksums
            find "$dir" -type f -exec sha256sum {} + | \
                sort -k 2 > "${snapshot_dir}/${dirname}-files.txt" 2>/dev/null || true
            
            # Create directory structure snapshot
            find "$dir" -type d | sort > "${snapshot_dir}/${dirname}-directories.txt"
            
            # Create statistics
            {
                echo "Snapshot Date: $(date)"
                echo "Directory: $dir"
                echo "Total Files: $(find "$dir" -type f | wc -l)"
                echo "Total Size: $(du -sh "$dir" | cut -f1)"
                echo "File Types:"
                find "$dir" -type f -name '*.*' | \
                    sed 's/.*\.//' | sort | uniq -c | sort -rn | head -20
            } > "${snapshot_dir}/${dirname}-stats.txt"
        fi
    done
    
    # Compress snapshots
    cd "${BACKUP_ROOT}/snapshots"
    tar -czf "snapshot-$(date +%Y%m%d).tar.gz" "$(date +%Y%m%d)/"
    rm -rf "$(date +%Y%m%d)"
}

# Verify backup integrity
verify_backup() {
    local backup_file=$1
    local checksum_file="${backup_file}.sha256"
    
    if [[ -f "$checksum_file" ]]; then
        log "INFO" "Verifying backup: $(basename "$backup_file")"
        
        if sha256sum -c "$checksum_file" &> /dev/null; then
            log "INFO" "Backup verification passed"
            return 0
        else
            log "ERROR" "Backup verification failed for $(basename "$backup_file")"
            return 1
        fi
    else
        log "WARN" "No checksum file found for $(basename "$backup_file")"
        return 1
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up old backups..."
    
    # Clean daily backups
    find "${BACKUP_ROOT}/daily" -name "*.tar.gz" -mtime +${DAILY_RETENTION} -delete
    find "${BACKUP_ROOT}/daily" -name "*.sha256" -mtime +${DAILY_RETENTION} -delete
    
    # Clean weekly backups
    find "${BACKUP_ROOT}/weekly" -name "*.tar.gz" -mtime +${WEEKLY_RETENTION} -delete
    find "${BACKUP_ROOT}/weekly" -name "*.sha256" -mtime +${WEEKLY_RETENTION} -delete
    
    # Clean monthly backups
    find "${BACKUP_ROOT}/monthly" -name "*.tar.gz" -mtime +${MONTHLY_RETENTION} -delete
    find "${BACKUP_ROOT}/monthly" -name "*.sha256" -mtime +${MONTHLY_RETENTION} -delete
    
    # Clean old logs
    find "${LOG_DIR}" -name "*.log" -mtime +30 -delete
    
    # Clean old snapshots
    find "${BACKUP_ROOT}/snapshots" -name "*.tar.gz" -mtime +90 -delete
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Discord webhook (if configured)
    if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
        local color=$([[ "$status" == "success" ]] && echo "3066993" || echo "15158332")
        
        curl -s -H "Content-Type: application/json" -X POST \
            -d "{\"embeds\":[{\"title\":\"Media Server Backup\",\"description\":\"${message}\",\"color\":${color}}]}" \
            "${DISCORD_WEBHOOK}"
    fi
    
    # Email notification (if configured)
    if [[ -n "${EMAIL_TO:-}" ]]; then
        echo "$message" | mail -s "Media Server Backup - ${status}" "${EMAIL_TO}"
    fi
}

# Generate backup report
generate_report() {
    local report_file="${BACKUP_ROOT}/backup-report.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Backup Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .success { color: green; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>Media Server Backup Report</h1>
    <p>Generated: $(date)</p>
    
    <h2>Backup Summary</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Count</th>
            <th>Total Size</th>
            <th>Oldest</th>
            <th>Newest</th>
        </tr>
EOF
    
    for type in daily weekly monthly; do
        local count=$(find "${BACKUP_ROOT}/${type}" -name "*.tar.gz" | wc -l)
        local size=$(du -sh "${BACKUP_ROOT}/${type}" 2>/dev/null | cut -f1)
        local oldest=$(find "${BACKUP_ROOT}/${type}" -name "*.tar.gz" -printf '%T+ %p\n' | sort | head -1 | cut -d' ' -f1)
        local newest=$(find "${BACKUP_ROOT}/${type}" -name "*.tar.gz" -printf '%T+ %p\n' | sort -r | head -1 | cut -d' ' -f1)
        
        echo "<tr><td>${type^}</td><td>$count</td><td>$size</td><td>$oldest</td><td>$newest</td></tr>" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
    </table>
    
    <h2>Recent Backups</h2>
    <table>
        <tr>
            <th>Service</th>
            <th>Type</th>
            <th>Date</th>
            <th>Size</th>
            <th>Status</th>
        </tr>
EOF
    
    # List recent backups
    find "${BACKUP_ROOT}" -name "*.tar.gz" -mtime -7 -printf '%T+ %p\n' | sort -r | while read -r line; do
        local date=$(echo "$line" | cut -d' ' -f1)
        local file=$(echo "$line" | cut -d' ' -f2)
        local basename=$(basename "$file")
        local service=$(echo "$basename" | cut -d'_' -f1)
        local type=$(basename $(dirname "$file"))
        local size=$(du -h "$file" | cut -f1)
        
        if verify_backup "$file"; then
            local status='<span class="success">✓ Verified</span>'
        else
            local status='<span class="error">✗ Failed</span>'
        fi
        
        echo "<tr><td>$service</td><td>$type</td><td>$date</td><td>$size</td><td>$status</td></tr>" >> "$report_file"
    done
    
    echo "</table></body></html>" >> "$report_file"
    
    log "INFO" "Backup report generated: $report_file"
}

# Restore function
restore_service() {
    local service=$1
    local backup_file=$2
    
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    # Verify backup
    if ! verify_backup "$backup_file"; then
        error_exit "Backup verification failed"
    fi
    
    log "INFO" "Restoring $service from $backup_file"
    
    # Create restore directory
    local restore_dir="${TEMP_DIR}/restore"
    mkdir -p "$restore_dir"
    
    # Extract backup
    tar -xzf "$backup_file" -C "$restore_dir"
    
    # Stop service
    stop_container "${CONTAINERS[$service]}"
    
    # Backup current configuration
    local current_backup="${BACKUP_ROOT}/temp/pre-restore-${service}-$(date +%Y%m%d_%H%M%S).tar.gz"
    backup_service "$service" "temp"
    
    # Restore configuration
    rsync -av --delete "${restore_dir}"/*/config/ "${SERVICES[$service]}/"
    
    # Restore database
    # Implementation depends on service
    
    # Start service
    start_container "${CONTAINERS[$service]}"
    
    log "INFO" "Restore completed for $service"
}

# Main backup function
perform_backup() {
    local backup_type=${1:-daily}
    
    log "INFO" "Starting $backup_type backup"
    
    # Check prerequisites
    check_prerequisites
    
    # Perform backups
    backup_all_services "$backup_type"
    
    # Create media snapshots (only on weekly/monthly)
    if [[ "$backup_type" != "daily" ]]; then
        backup_media_snapshots
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Generate report
    generate_report
    
    # Send notification
    send_notification "success" "$backup_type backup completed successfully"
    
    log "INFO" "Backup completed successfully"
}

# Show help
show_help() {
    cat << EOF
Media Server Backup Automation Script

Usage: $0 [OPTIONS] [COMMAND]

COMMANDS:
    backup [TYPE]       Perform backup (daily|weekly|monthly)
    restore SERVICE FILE Restore service from backup
    verify FILE         Verify backup integrity
    cleanup             Clean old backups
    report              Generate backup report

OPTIONS:
    -h, --help          Show this help message
    -c, --config FILE   Use custom config file
    -s, --service NAME  Backup specific service only

EXAMPLES:
    $0 backup daily     Perform daily backup
    $0 backup weekly    Perform weekly backup
    $0 restore sonarr /backup/sonarr_20250101.tar.gz
    $0 verify /backup/radarr_20250101.tar.gz

BACKUP SCHEDULE (add to crontab):
    0 2 * * * $0 backup daily
    0 3 * * 0 $0 backup weekly
    0 4 1 * * $0 backup monthly

EOF
}

# Parse arguments
case "${1:-backup}" in
    backup)
        perform_backup "${2:-daily}"
        ;;
    restore)
        if [[ -n "$2" && -n "$3" ]]; then
            restore_service "$2" "$3"
        else
            echo "Error: Service and backup file required"
            show_help
            exit 1
        fi
        ;;
    verify)
        if [[ -n "$2" ]]; then
            verify_backup "$2"
        else
            echo "Error: Backup file required"
            exit 1
        fi
        ;;
    cleanup)
        cleanup_old_backups
        ;;
    report)
        generate_report
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac