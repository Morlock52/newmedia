# 🔥 Disaster Recovery Plan - NEXUS Media Server 2025

**RPO (Recovery Point Objective)**: 4 hours  
**RTO (Recovery Time Objective)**: 2 hours  
**Backup Strategy**: 3-2-1 Rule Implementation  
**Last Updated**: August 1, 2025

---

## 🎯 DISASTER RECOVERY STRATEGY

### Critical Data Classification
1. **TIER 1 - Critical** (Backup every 4 hours)
   - Service configurations (./config/*)
   - Database dumps (PostgreSQL, SQLite)
   - SSL certificates
   - API keys and secrets

2. **TIER 2 - Important** (Backup daily)
   - Media metadata
   - User preferences
   - Watch history
   - Playlists

3. **TIER 3 - Replaceable** (Backup weekly)
   - Media files (can be re-downloaded)
   - Temporary transcodes
   - Log files

---

## 🛡️ AUTOMATED BACKUP SYSTEM

### 1. **Local Backup Script**
```bash
#!/bin/bash
# backup-media-server.sh

set -euo pipefail

# Configuration
BACKUP_ROOT="/backup/media-server"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

# Notification function
notify() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    # Add webhook notification here
}

# Create backup directory
mkdir -p "${BACKUP_DIR}"

notify "Starting backup process..."

# Stop services for consistency
docker-compose stop

# Backup configurations
notify "Backing up configurations..."
tar -czf "${BACKUP_DIR}/configs.tar.gz" \
    ./config \
    ./secrets \
    .env \
    docker-compose*.yml

# Backup databases
notify "Backing up databases..."

# PostgreSQL
docker run --rm \
    -v postgres_data:/data \
    -v "${BACKUP_DIR}":/backup \
    postgres:15 \
    pg_dumpall -U postgres > "${BACKUP_DIR}/postgres_backup.sql"

# SQLite databases (Jellyfin, Arr apps)
for db in $(find ./config -name "*.db" -o -name "*.sqlite*"); do
    cp "$db" "${BACKUP_DIR}/"
done

# Backup Docker volumes
notify "Backing up Docker volumes..."
for volume in $(docker volume ls -q | grep media); do
    docker run --rm \
        -v ${volume}:/data \
        -v "${BACKUP_DIR}":/backup \
        alpine \
        tar -czf "/backup/${volume}.tar.gz" /data
done

# Start services
docker-compose up -d

# Create backup manifest
cat > "${BACKUP_DIR}/manifest.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "version": "$(docker-compose version)",
    "services": $(docker ps --format json),
    "volumes": $(docker volume ls --format json),
    "backup_size": "$(du -sh ${BACKUP_DIR} | cut -f1)"
}
EOF

# Cleanup old backups
find "${BACKUP_ROOT}" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} +

notify "Backup completed successfully!"
```

### 2. **Offsite Backup - Rclone to S3**
```bash
#!/bin/bash
# offsite-backup.sh

set -euo pipefail

# Configure rclone for S3
cat > ~/.config/rclone/rclone.conf <<EOF
[backup-s3]
type = s3
provider = AWS
access_key_id = ${AWS_ACCESS_KEY}
secret_access_key = ${AWS_SECRET_KEY}
region = us-east-1
bucket_name = media-server-backups
EOF

# Sync to S3 with encryption
rclone sync \
    --encrypt \
    --compress \
    --progress \
    --transfers 4 \
    /backup/media-server/ backup-s3:media-server-backups/

# Verify backup integrity
rclone check \
    --download \
    /backup/media-server/ backup-s3:media-server-backups/
```

### 3. **Automated Backup Schedule**
```bash
# /etc/cron.d/media-server-backup
# Tier 1 - Every 4 hours
0 */4 * * * root /opt/scripts/backup-media-server.sh tier1
0 */4 * * * root /opt/scripts/offsite-backup.sh tier1

# Tier 2 - Daily at 2 AM
0 2 * * * root /opt/scripts/backup-media-server.sh tier2
30 2 * * * root /opt/scripts/offsite-backup.sh tier2

# Tier 3 - Weekly on Sunday
0 3 * * 0 root /opt/scripts/backup-media-server.sh tier3
0 4 * * 0 root /opt/scripts/offsite-backup.sh tier3

# Backup verification - Daily
0 6 * * * root /opt/scripts/verify-backups.sh
```

---

## 🚨 DISASTER RECOVERY PROCEDURES

### SCENARIO 1: Service Failure
**RTO: 15 minutes**

```bash
#!/bin/bash
# quick-recovery.sh

# Check service health
docker-compose ps

# Restart failed services
docker-compose restart <service_name>

# If persistent failure, recreate container
docker-compose stop <service_name>
docker-compose rm -f <service_name>
docker-compose up -d <service_name>

# Verify service health
docker-compose logs --tail=50 <service_name>
```

### SCENARIO 2: Data Corruption
**RTO: 30 minutes**

```bash
#!/bin/bash
# recover-from-corruption.sh

SERVICE=$1
BACKUP_DATE=$2

# Stop affected service
docker-compose stop ${SERVICE}

# Restore from backup
tar -xzf /backup/media-server/${BACKUP_DATE}/configs.tar.gz

# Restore database
if [ "$SERVICE" == "postgres" ]; then
    docker-compose up -d postgres
    docker exec -i postgres psql -U postgres < /backup/media-server/${BACKUP_DATE}/postgres_backup.sql
fi

# Restart service
docker-compose up -d ${SERVICE}

# Verify integrity
docker-compose exec ${SERVICE} health-check
```

### SCENARIO 3: Complete System Failure
**RTO: 2 hours**

```bash
#!/bin/bash
# full-disaster-recovery.sh

# On new system
# 1. Install Docker
curl -fsSL https://get.docker.com | bash

# 2. Restore from offsite backup
rclone copy backup-s3:media-server-backups/latest/ /restore/

# 3. Extract configurations
cd /opt/media-server
tar -xzf /restore/configs.tar.gz

# 4. Restore Docker volumes
for volume in /restore/*.tar.gz; do
    volume_name=$(basename $volume .tar.gz)
    docker volume create $volume_name
    docker run --rm -v $volume_name:/data -v /restore:/backup alpine \
        tar -xzf /backup/$(basename $volume) -C /
done

# 5. Start services
docker-compose up -d

# 6. Restore databases
docker exec -i postgres psql -U postgres < /restore/postgres_backup.sql

# 7. Verify all services
docker-compose ps
for service in $(docker-compose ps --services); do
    docker-compose exec $service health-check || echo "$service FAILED"
done
```

---

## 📋 RECOVERY VALIDATION CHECKLIST

### Post-Recovery Verification
- [ ] All containers running (`docker-compose ps`)
- [ ] Web interfaces accessible
- [ ] Media library visible in Jellyfin
- [ ] Arr apps connected to indexers
- [ ] Download clients functional
- [ ] Database queries working
- [ ] User authentication working
- [ ] SSL certificates valid
- [ ] Monitoring stack operational

### Data Integrity Checks
```bash
#!/bin/bash
# verify-recovery.sh

# Check service connectivity
curl -f https://jellyfin.${DOMAIN}/health || echo "Jellyfin FAILED"
curl -f https://sonarr.${DOMAIN}/api/v3/health || echo "Sonarr FAILED"
curl -f https://radarr.${DOMAIN}/api/v3/health || echo "Radarr FAILED"

# Verify database integrity
docker exec postgres pg_isready || echo "PostgreSQL FAILED"

# Check media accessibility
docker exec jellyfin ls -la /media || echo "Media mount FAILED"

# Verify backup freshness
latest_backup=$(ls -t /backup/media-server/ | head -1)
backup_age=$(( ($(date +%s) - $(stat -c %Y /backup/media-server/$latest_backup)) / 3600 ))
if [ $backup_age -gt 4 ]; then
    echo "WARNING: Latest backup is $backup_age hours old!"
fi
```

---

## 🔄 BACKUP TESTING PROCEDURE

### Monthly DR Drill
```bash
#!/bin/bash
# dr-drill.sh

# 1. Create test environment
docker-compose -f docker-compose.test.yml up -d

# 2. Restore latest backup
./restore-to-test.sh

# 3. Run validation suite
./verify-recovery.sh

# 4. Document results
cat > /backup/dr-test-$(date +%Y%m%d).log <<EOF
DR Test Date: $(date)
Backup Used: $(ls -t /backup/media-server/ | head -1)
Recovery Time: ${RECOVERY_TIME}
Services Tested: $(docker-compose ps --services)
Issues Found: ${ISSUES}
EOF

# 5. Cleanup test environment
docker-compose -f docker-compose.test.yml down -v
```

---

## 📱 EMERGENCY CONTACTS & PROCEDURES

### Incident Response Team
```yaml
contacts:
  primary:
    name: "System Administrator"
    phone: "+1-XXX-XXX-XXXX"
    email: "admin@domain.com"
  backup:
    name: "DevOps Lead"
    phone: "+1-XXX-XXX-XXXX"
    email: "devops@domain.com"
  vendor:
    name: "Cloud Provider Support"
    phone: "+1-800-XXX-XXXX"
    ticket: "https://support.provider.com"
```

### Emergency Procedures
1. **Power Failure**: UPS provides 30 minutes - initiate graceful shutdown
2. **Network Outage**: Failover to backup ISP (if configured)
3. **Security Breach**: Immediately isolate system, preserve logs
4. **Data Loss**: Stop all writes, begin recovery procedure

---

## 📊 BACKUP METRICS & MONITORING

### Grafana Alert Rules
```yaml
groups:
  - name: backup_alerts
    rules:
      - alert: BackupFailed
        expr: time() - backup_last_success_timestamp > 14400
        annotations:
          summary: "Backup hasn't completed in 4 hours"
          
      - alert: BackupStorageFull
        expr: backup_storage_available_bytes < 10737418240
        annotations:
          summary: "Less than 10GB backup storage remaining"
          
      - alert: BackupValidationFailed
        expr: backup_validation_errors > 0
        annotations:
          summary: "Backup validation detected errors"
```

---

## 🎯 RECOVERY TIME OBJECTIVES

| Scenario | Target RTO | Actual RTO | Data Loss |
|----------|------------|------------|-----------|
| Service Restart | 5 min | 2-3 min | None |
| Container Rebuild | 15 min | 10 min | None |
| Database Recovery | 30 min | 20-25 min | <4 hours |
| Full System Recovery | 2 hours | 1.5-2 hours | <4 hours |
| Disaster Recovery | 4 hours | 3-4 hours | <24 hours |

---

**⚠️ CRITICAL**: Test your disaster recovery plan monthly. An untested backup is not a backup!