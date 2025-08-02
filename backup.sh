#!/bin/bash
# Backup script for Ultimate Media Server 2025

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="media-server-backup-${TIMESTAMP}"

echo "🔄 Starting backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Stop containers for consistent backup (optional)
# docker-compose stop

# Backup configurations
echo "📦 Backing up configurations..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/configs.tar.gz" ./config/

# Backup environment
cp .env "${BACKUP_DIR}/${BACKUP_NAME}/"
cp docker-compose.yml "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup database dumps (if applicable)
if docker ps | grep -q postgres; then
    echo "🗄️ Backing up PostgreSQL databases..."
    docker exec postgres pg_dumpall -U postgres > "${BACKUP_DIR}/${BACKUP_NAME}/postgres_dump.sql"
fi

# Create backup manifest
cat > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.txt" << MANIFEST
Backup created: $(date)
Hostname: $(hostname)
Docker version: $(docker --version)
Compose version: $(docker-compose --version)
MANIFEST

# Compress full backup
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
rm -rf "${BACKUP_NAME}/"

# Clean old backups (keep last 7)
ls -t *.tar.gz | tail -n +8 | xargs -r rm

echo "✅ Backup complete: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
