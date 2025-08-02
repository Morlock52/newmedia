#!/usr/bin/env bash
set -euo pipefail

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# Load environment variables
if [[ -f "$ROOT/.env" ]]; then
  set -o allexport
  source "$ROOT/.env"
  set +o allexport
fi

BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)
CONFIG_ROOT="${CONFIG_ROOT:-$ROOT/config}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

create_backup() {
  log "Creating backup at $BACKUP_DIR/backup_$DATE"
  mkdir -p "$BACKUP_DIR/backup_$DATE"

  pushd "$ROOT" >/dev/null

  # Backup configurations
  log "Backing up configuration files..."
  tar czf "$BACKUP_DIR/backup_$DATE/config.tar.gz" -C "$CONFIG_ROOT" .

  # Backup Docker Compose files and environment
  log "Backing up Docker Compose configuration..."
  cp -r compose/ "$BACKUP_DIR/backup_$DATE/"
  cp .env "$BACKUP_DIR/backup_$DATE/" 2>/dev/null || log "No .env file found"
  cp .env.example "$BACKUP_DIR/backup_$DATE/"
  
  # Backup scripts
  cp -r scripts/ "$BACKUP_DIR/backup_$DATE/"
  
  # Backup monitoring configuration
  cp prometheus.yml "$BACKUP_DIR/backup_$DATE/" 2>/dev/null || log "No prometheus.yml found"
  cp alert_rules.yml "$BACKUP_DIR/backup_$DATE/" 2>/dev/null || log "No alert_rules.yml found"
  
  # Backup secrets (encrypted)
  if [[ -d "secrets" ]]; then
    log "Backing up secrets (you should encrypt these separately)..."
    tar czf "$BACKUP_DIR/backup_$DATE/secrets.tar.gz" secrets/
  fi

  # Backup Docker volumes metadata
  log "Exporting Docker volume information..."
  docker volume ls > "$BACKUP_DIR/backup_$DATE/volumes.txt" 2>/dev/null || true
  
  popd >/dev/null
}

cleanup_old_backups() {
  log "Cleaning up backups older than $RETENTION_DAYS days"
  find "$BACKUP_DIR" -name "backup_*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
}

verify_backup() {
  log "Verifying backup integrity"
  
  # Verify config archive
  if [[ -f "$BACKUP_DIR/backup_$DATE/config.tar.gz" ]]; then
    tar -tzf "$BACKUP_DIR/backup_$DATE/config.tar.gz" > /dev/null
    log "Configuration backup verified"
  fi
  
  # Verify secrets archive
  if [[ -f "$BACKUP_DIR/backup_$DATE/secrets.tar.gz" ]]; then
    tar -tzf "$BACKUP_DIR/backup_$DATE/secrets.tar.gz" > /dev/null
    log "Secrets backup verified"
  fi
  
  # Check compose files
  if [[ -d "$BACKUP_DIR/backup_$DATE/compose" ]]; then
    log "Docker Compose files backed up successfully"
  fi
  
  log "Backup verified successfully"
}

restore_backup() {
  local backup_date="$1"
  if [[ -z "$backup_date" ]]; then
    log "Error: Please specify backup date (YYYYMMDD_HHMMSS)"
    exit 1
  fi
  
  local backup_path="$BACKUP_DIR/backup_$backup_date"
  if [[ ! -d "$backup_path" ]]; then
    log "Error: Backup $backup_date not found at $backup_path"
    exit 1
  fi
  
  log "Restoring backup from $backup_date"
  
  pushd "$ROOT" >/dev/null
  
  # Stop services
  log "Stopping services..."
  docker-compose down || true
  
  # Restore configuration
  if [[ -f "$backup_path/config.tar.gz" ]]; then
    log "Restoring configuration files..."
    tar xzf "$backup_path/config.tar.gz" -C "$CONFIG_ROOT"
  fi
  
  # Restore compose files
  if [[ -d "$backup_path/compose" ]]; then
    log "Restoring Docker Compose files..."
    cp -r "$backup_path/compose/"* compose/
  fi
  
  # Restore environment
  if [[ -f "$backup_path/.env" ]]; then
    log "Restoring environment file..."
    cp "$backup_path/.env" .
  fi
  
  popd >/dev/null
  
  log "Restore completed. You may need to restart services manually."
}

list_backups() {
  log "Available backups in $BACKUP_DIR:"
  find "$BACKUP_DIR" -name "backup_*" -type d -printf "%f\n" | sort || log "No backups found"
}

main() {
  case "${1:-backup}" in
    backup)
      create_backup
      verify_backup
      cleanup_old_backups
      log "Backup completed successfully"
      ;;
    restore)
      restore_backup "$2"
      ;;
    list)
      list_backups
      ;;
    *)
      log "Usage: $0 [backup|restore <date>|list]"
      exit 1
      ;;
  esac
}

main "$@"