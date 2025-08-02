# 👨‍💼 NEXUS Platform Administrator Guide

Comprehensive administration guide for managing, maintaining, and optimizing the NEXUS Media Server Platform.

## Table of Contents

1. [Administrator Overview](#administrator-overview)
2. [Initial System Setup](#initial-system-setup)
3. [User Management](#user-management)
4. [Service Management](#service-management)
5. [Performance Monitoring](#performance-monitoring)
6. [Security Administration](#security-administration)
7. [Backup & Recovery](#backup--recovery)
8. [Updates & Maintenance](#updates--maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Configuration](#advanced-configuration)

---

## Administrator Overview

### Administrative Responsibilities

**Core Responsibilities:**
- System deployment and configuration
- User account and permission management
- Service health monitoring and maintenance
- Security policy implementation and auditing
- Performance optimization and scaling
- Backup and disaster recovery planning
- Software updates and patch management

**Required Skills:**
- Docker and container orchestration
- Linux system administration
- Network configuration and security
- Database administration (PostgreSQL, Redis)
- Monitoring and observability tools
- Basic understanding of AI/ML concepts
- Blockchain and Web3 fundamentals (optional)

### Administrative Access

**Admin Dashboard Access:**
- **URL**: http://localhost:3001/admin
- **Default Credentials**: admin/admin (change immediately)
- **Multi-Factor Authentication**: Enabled by default

**Command Line Access:**
```bash
# SSH into the main server
ssh admin@your-nexus-server

# Access Docker containers
docker exec -it nexus-jellyfin bash
docker exec -it nexus-ai-orchestrator bash

# Kubernetes access (if using K8s)
kubectl get pods -n nexus-platform
kubectl logs -f deployment/nexus-core
```

---

## Initial System Setup

### Post-Deployment Configuration

#### 1. Security Hardening

```bash
# Change default passwords
./scripts/change-default-passwords.sh

# Generate secure API keys
./scripts/generate-api-keys.sh

# Configure SSL certificates
./scripts/setup-ssl.sh --domain your-domain.com --email admin@your-domain.com

# Enable firewall
ufw enable
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
```

#### 2. Database Initialization

```bash
# Initialize PostgreSQL
docker exec -it nexus-postgres psql -U nexus -d nexus

-- Create admin user
INSERT INTO nexus_core.users (username, email, password_hash, role) 
VALUES ('admin', 'admin@nexus-platform.com', 'secure_hash', 'administrator');

-- Create default quality profiles
INSERT INTO nexus_core.quality_profiles (name, settings) 
VALUES ('4K Ultra HD', '{"resolution": "4K", "bitrate": "25000"}'),
       ('1080p High', '{"resolution": "1080p", "bitrate": "8000"}'),
       ('720p Standard', '{"resolution": "720p", "bitrate": "4000"}');
```

#### 3. Service Configuration

```yaml
# config/admin/service-config.yml
services:
  jellyfin:
    admin_user: "admin"
    library_paths:
      movies: "/media/movies"
      tv: "/media/tv"
      music: "/media/music"
    transcoding:
      hardware_acceleration: true
      gpu_device: "/dev/dri/renderD128"
    
  ai_orchestrator:
    gpu_enabled: true
    model_cache_size: "8GB"
    concurrent_requests: 10
    
  web3_integration:
    enabled: false  # Enable only if needed
    ethereum_rpc: "https://mainnet.infura.io/v3/YOUR_KEY"
    ipfs_node: "local"  # or "infura"

  monitoring:
    retention_days: 30
    metrics_interval: "15s"
    log_level: "info"
```

### Initial Content Setup

#### 1. Media Library Configuration

```bash
# Create directory structure
mkdir -p /media/{movies,tv,music,audiobooks}
chown -R 1000:1000 /media

# Configure Jellyfin libraries
curl -X POST "http://localhost:8096/api/Library/VirtualFolders" \
  -H "Authorization: MediaBrowser Token=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "Name": "Movies",
    "CollectionType": "movies",
    "Paths": ["/media/movies"],
    "LibraryOptions": {
      "EnablePhotos": false,
      "EnableRealtimeMonitor": true,
      "EnableChapterImageExtraction": false
    }
  }'
```

#### 2. Download Client Setup

```bash
# Configure qBittorrent
curl -X POST "http://localhost:8080/api/v2/app/setPreferences" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'json={"save_path":"/downloads/complete","temp_path":"/downloads/incomplete","autorun_enabled":false}'

# Configure Sonarr quality profiles
curl -X POST "http://localhost:8989/api/v3/qualityProfile" \
  -H "X-Api-Key: YOUR_SONARR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "4K Ultra HD",
    "upgradeAllowed": true,
    "cutoff": 19,
    "items": [...]
  }'
```

---

## User Management

### User Administration Interface

**Web Interface:**
1. Access Admin Dashboard → User Management
2. View all users, roles, and permissions
3. Create, modify, or disable user accounts
4. Manage group memberships and access levels

**Command Line Interface:**
```bash
# Create new user
./scripts/create-user.sh --username johndoe --email john@example.com --role user

# List all users
./scripts/list-users.sh

# Modify user permissions
./scripts/modify-user.sh --username johndoe --role power-user

# Disable user account
./scripts/disable-user.sh --username johndoe
```

### Role-Based Access Control

#### Default Roles

| Role | Permissions | Description |
|------|------------|-------------|
| **Administrator** | Full system access | Complete platform management |
| **Power User** | Advanced features, no system config | AI/ML, AR/VR, Web3 access |
| **Standard User** | Basic media access | Standard streaming and requests |
| **Family User** | Age-restricted content | Parental controls applied |
| **Guest** | Limited temporary access | Read-only, limited time |

#### Custom Role Creation

```sql
-- Create custom role
INSERT INTO nexus_core.roles (name, description, permissions) VALUES (
  'content_moderator',
  'Can manage content and user reports',
  '{"can_delete_content": true, "can_ban_users": true, "can_access_reports": true}'
);

-- Assign role to user
UPDATE nexus_core.users 
SET role_id = (SELECT id FROM nexus_core.roles WHERE name = 'content_moderator')
WHERE username = 'moderator_user';
```

### Bulk User Management

```bash
# Import users from CSV
./scripts/bulk-import-users.sh --file users.csv --format csv

# Export user data
./scripts/export-users.sh --format json --output user_export.json

# Reset passwords for multiple users
./scripts/bulk-password-reset.sh --file user_list.txt
```

---

## Service Management

### Service Health Monitoring

#### Web Dashboard

**System Overview:**
- Access: http://localhost:3001/admin/services
- Real-time service status indicators
- Resource usage graphs (CPU, memory, disk)
- Alert notifications and incident reports

**Service Control Panel:**
```
🟢 Jellyfin Media Server      [Running] [Restart] [Logs] [Config]
🟢 Sonarr TV Management      [Running] [Restart] [Logs] [Config]  
🟢 Radarr Movie Management   [Running] [Restart] [Logs] [Config]
🟢 AI/ML Orchestrator        [Running] [Restart] [Logs] [Config]
🟡 Web3 Integration          [Disabled] [Enable] [Config]
🔴 Voice AI System           [Error] [Restart] [Debug] [Logs]
```

#### Command Line Management

```bash
# Check service status
./scripts/service-status.sh

# Restart specific service
docker compose restart jellyfin

# View service logs
docker compose logs -f --tail=100 ai-orchestrator

# Service resource usage
docker stats nexus-jellyfin nexus-sonarr nexus-radarr

# Full system health check
./health-check-ultimate.sh
```

### Service Configuration Management

#### Configuration Backup

```bash
# Backup all service configurations
./scripts/backup-configs.sh --output /backup/configs-$(date +%Y%m%d)

# Backup specific service
./scripts/backup-service-config.sh --service jellyfin --output /backup/jellyfin-config.tar.gz

# Automated daily backup
cat > /etc/cron.d/nexus-backup << EOF
0 2 * * * root /opt/nexus/scripts/backup-configs.sh --output /backup/daily-configs-$(date +\%Y\%m\%d)
EOF
```

#### Configuration Validation

```bash
# Validate all configurations
./scripts/validate-configs.sh

# Test specific service configuration
./scripts/test-service-config.sh --service ai-orchestrator

# Configuration diff before applying changes
./scripts/config-diff.sh --service jellyfin --compare-with /backup/jellyfin-config-backup
```

### AI/ML Service Administration

#### Model Management

```bash
# List available AI models
curl http://localhost:8080/api/admin/models

# Update AI models
./scripts/update-ai-models.sh --model recommendation-engine --version v2.1

# GPU resource allocation
./scripts/manage-gpu-allocation.sh --service content-analysis --gpu-memory 4GB

# Performance tuning
./scripts/optimize-ai-performance.sh --batch-size 32 --concurrent-requests 8
```

#### Training Data Management

```bash
# Export training data
./scripts/export-training-data.sh --service recommendation-engine --format json

# Import curated training data
./scripts/import-training-data.sh --file training_data.json --validate

# Clear training data (privacy compliance)
./scripts/clear-training-data.sh --user-id specific_user --confirm
```

---

## Performance Monitoring

### System Metrics Dashboard

**Grafana Dashboard Access:**
- **URL**: http://localhost:3000/d/nexus-overview
- **Login**: admin/admin (change default password)

**Key Metrics to Monitor:**

```
System Performance:
├── CPU Usage: < 80% average
├── Memory Usage: < 85% of available
├── Disk I/O: < 80% utilization  
├── Network Throughput: Monitor for bottlenecks
└── GPU Utilization: 60-80% optimal for AI workloads

Service Performance:
├── Jellyfin: Transcoding queue, active sessions
├── AI/ML: Model inference time, queue depth
├── Database: Query performance, connection pool
├── Web3: Blockchain sync status, IPFS health
└── Storage: Available space, I/O performance
```

### Performance Optimization

#### Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Analyze query performance
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;

-- Rebuild indexes
REINDEX DATABASE nexus;

-- Update table statistics
ANALYZE;
```

#### Container Resource Limits

```yaml
# docker-compose.override.yml
services:
  jellyfin:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
  
  ai-orchestrator:
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 16G
        reservations:
          cpus: '4.0'
          memory: 8G
          generic_resources:
            - discrete_resource_spec:
                kind: 'NVIDIA-GPU'
                value: 1
```

#### Storage Optimization

```bash
# Monitor disk usage
df -h
du -sh /var/lib/docker/volumes/*

# Clean up Docker resources  
docker system prune -af --volumes

# Optimize media storage
./scripts/optimize-media-storage.sh --compress-old --remove-duplicates

# Configure storage tiers
./scripts/setup-storage-tiers.sh --ssd-path /fast-storage --hdd-path /bulk-storage
```

### Alert Configuration

```yaml
# config/alertmanager/alerts.yml
groups:
- name: nexus.critical
  rules:
  - alert: ServiceDown
    expr: up == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.job }} is down"
      
  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      
  - alert: DiskSpaceLow
    expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} < 0.1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Disk space low on {{ $labels.instance }}"

  - alert: AIModelSlowResponse
    expr: ai_model_response_time_seconds > 5
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "AI model response time degrading"
```

---

## Security Administration

### Security Monitoring

#### Security Dashboard

**Access**: http://localhost:3001/admin/security

**Key Security Metrics:**
```
Authentication:
├── Failed login attempts: Monitor for brute force
├── Unusual login locations: Geographic anomalies  
├── Session management: Active sessions, timeouts
└── API key usage: Monitor for abuse

Network Security:
├── Firewall status: Ensure proper rules
├── SSL certificate health: Expiration monitoring
├── VPN connectivity: For secure downloads
└── Intrusion detection: Failed access attempts

Data Security:
├── Encryption status: All data encrypted at rest
├── Backup integrity: Regular backup verification
├── Access logs: User and admin activities
└── Quantum security: Post-quantum crypto status
```

#### Security Hardening Checklist

```bash
# Daily security checks
./scripts/security-check.sh --full

# SSL certificate monitoring
./scripts/check-ssl-certs.sh --warn-days 30

# Update security patches
./scripts/security-updates.sh --auto-approve

# Audit user access
./scripts/audit-user-access.sh --suspicious-activity

# Check for compromised credentials
./scripts/check-compromised-passwords.sh --hash-check
```

### Access Control Management

#### Two-Factor Authentication

```bash
# Enable 2FA for all admin accounts
./scripts/enforce-2fa.sh --role administrator

# Generate backup codes
./scripts/generate-backup-codes.sh --user admin

# Reset 2FA for user (emergency)
./scripts/reset-2fa.sh --user username --admin-override
```

#### API Security

```bash
# Rotate API keys
./scripts/rotate-api-keys.sh --service all --notify-users

# Monitor API abuse
./scripts/monitor-api-usage.sh --threshold 1000/hour --alert

# Rate limiting configuration
curl -X POST http://localhost:3001/admin/api/rate-limits \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "endpoint": "/api/ai/recommendations",
    "limit": 100,
    "window": 3600,
    "burst": 10
  }'
```

### Quantum Security Management

```bash
# Check quantum security status
curl http://localhost:8080/api/security/quantum/status

# Update post-quantum algorithms
./scripts/update-quantum-crypto.sh --algorithm ML-KEM-1024

# Performance impact monitoring
./scripts/monitor-quantum-performance.sh --metrics latency,throughput

# Hybrid mode configuration
./scripts/configure-quantum-hybrid.sh --enable-classical-fallback
```

---

## Backup & Recovery

### Automated Backup System

#### Backup Configuration

```yaml
# config/backup/backup-config.yml
backup:
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  compression: true
  encryption: true
  
  targets:
    - name: "configurations"
      path: "./config"
      type: "incremental"
      
    - name: "databases"
      path: "postgres://"
      type: "full"
      pre_script: "./scripts/db-backup-prep.sh"
      
    - name: "media_metadata"
      path: "./media-data"
      type: "incremental"
      exclude: ["*.mkv", "*.mp4", "*.avi"]
      
    - name: "ai_models"
      path: "./ai-ml-nexus/models"
      type: "weekly"
      
  destinations:
    - name: "local"
      path: "/backup/nexus"
      
    - name: "s3"
      bucket: "nexus-backups"
      region: "us-west-2"
      encryption: "AES256"
      
    - name: "offsite"
      type: "rsync"
      host: "backup.example.com"
      path: "/backups/nexus"
```

#### Backup Execution

```bash
# Manual backup execution
./scripts/backup.sh --full --verify

# Incremental backup
./scripts/backup.sh --incremental --quiet

# Backup specific component
./scripts/backup.sh --component database --destination s3

# Verify backup integrity
./scripts/verify-backup.sh --backup-id 20240115-023000

# List available backups
./scripts/list-backups.sh --destination all --show-details
```

### Disaster Recovery

#### Recovery Procedures

```bash
# Full system recovery
./scripts/disaster-recovery.sh --backup-date 20240115 --confirm

# Selective recovery
./scripts/restore.sh --component database --backup-id 20240115-023000

# Configuration-only recovery
./scripts/restore-configs.sh --backup /backup/configs-20240115.tar.gz

# Test recovery procedures (dry run)
./scripts/test-recovery.sh --backup-date 20240115 --dry-run
```

#### Recovery Planning

**Recovery Time Objectives (RTO):**
- Critical services: 15 minutes
- Full system: 2 hours
- Complete rebuild: 4 hours

**Recovery Point Objectives (RPO):**
- Configuration: 24 hours
- Database: 1 hour
- User data: 4 hours
- Media metadata: 24 hours

### Backup Monitoring

```bash
# Backup status dashboard
./scripts/backup-status.sh --dashboard

# Failed backup notifications
./scripts/check-backup-failures.sh --notify-admin

# Storage usage monitoring
./scripts/monitor-backup-storage.sh --warn-threshold 80%

# Backup restoration testing
./scripts/test-backups.sh --sample-restore --weekly
```

---

## Updates & Maintenance

### Update Management

#### Automated Updates

```bash
# Configure automatic updates
cat > /etc/cron.d/nexus-updates << EOF
# Security updates (daily)
0 3 * * * root /opt/nexus/scripts/security-updates.sh --auto

# Service updates (weekly, Sunday)
0 4 * * 0 root /opt/nexus/scripts/update-services.sh --stable-only

# AI model updates (monthly)
0 5 1 * * root /opt/nexus/scripts/update-ai-models.sh --production
EOF
```

#### Manual Updates

```bash
# Check for available updates
./scripts/check-updates.sh --all-services

# Update specific service
./scripts/update-service.sh --service jellyfin --version latest

# Update AI/ML models
./scripts/update-ai-models.sh --model recommendation-engine --backup-current

# System-wide update
./scripts/system-update.sh --backup-first --test-after

# Rollback if needed
./scripts/rollback.sh --service jellyfin --to-version previous
```

### Maintenance Windows

#### Scheduled Maintenance

```bash
# Schedule maintenance window
./scripts/schedule-maintenance.sh \
  --start "2024-01-15 02:00" \
  --duration 120 \
  --services "jellyfin,sonarr,radarr" \
  --notify-users

# Maintenance mode activation
./scripts/maintenance-mode.sh --enable --message "System maintenance in progress"

# Post-maintenance verification
./scripts/post-maintenance-check.sh --full-test
```

#### Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash
# weekly-maintenance.sh

echo "Starting weekly maintenance..."

# Clean up logs
find /var/log -name "*.log" -mtime +7 -delete

# Docker cleanup
docker system prune -f
docker volume prune -f

# Database maintenance
docker exec nexus-postgres pg_stat_statements_reset
docker exec nexus-postgres vacuumdb -U nexus -d nexus --analyze

# Clear temporary files
rm -rf /tmp/nexus-*
find ./transcodes -name "*.ts" -mtime +1 -delete

# Update AI model cache
./scripts/refresh-ai-cache.sh

# Optimize storage
./scripts/optimize-storage.sh --defragment

# Generate maintenance report
./scripts/maintenance-report.sh --output /var/log/maintenance-$(date +%Y%m%d).log

echo "Weekly maintenance completed."
```

### Health Checks & Monitoring

```bash
# Comprehensive health check
./scripts/health-check-comprehensive.sh

# Performance baseline testing
./scripts/benchmark-system.sh --save-baseline

# Service availability monitoring
./scripts/monitor-availability.sh --sla-check

# Generate system report
./scripts/system-report.sh --detailed --output /tmp/system-report.html
```

---

## Troubleshooting

### Common Issues & Solutions

#### Service Startup Failures

**Issue**: Services fail to start after reboot
```bash
# Check service dependencies
docker compose config --services
docker compose ps --services --filter "status=exited"

# Check logs for startup errors
docker compose logs --tail=50 jellyfin

# Verify configuration files
./scripts/validate-configs.sh --fix-permissions

# Restart with dependency order
docker compose up -d --remove-orphans
```

#### Database Connection Issues

**Issue**: Services can't connect to database
```bash
# Check database status
docker exec nexus-postgres pg_isready -U nexus

# Verify connection parameters
docker exec nexus-postgres psql -U nexus -d nexus -c "SELECT version();"

# Check connection limits
docker exec nexus-postgres psql -U nexus -d nexus -c "SELECT count(*) FROM pg_stat_activity;"

# Reset connections if needed
docker restart nexus-postgres
sleep 10
docker compose restart
```

#### Performance Issues

**Issue**: System running slowly
```bash
# Check resource usage
docker stats --no-stream
htop

# Identify bottlenecks
./scripts/performance-analysis.sh --detailed

# Optimize based on findings
./scripts/auto-optimize.sh --cpu --memory --disk

# Monitor improvement
./scripts/benchmark-system.sh --compare-baseline
```

#### AI/ML Service Issues

**Issue**: AI services not responding
```bash
# Check GPU availability
nvidia-smi
docker exec nexus-ai-orchestrator nvidia-smi

# Verify model loading
curl http://localhost:8080/api/admin/models/status

# Clear model cache if corrupted
./scripts/clear-ai-cache.sh --models --restart-services

# Reload models
./scripts/reload-ai-models.sh --force
```

### Diagnostic Tools

#### System Diagnostics

```bash
# Comprehensive diagnostic script
./scripts/diagnose-system.sh --output /tmp/diagnostic-report.txt

# Network connectivity testing
./scripts/test-network.sh --all-services --external

# Storage health check
./scripts/check-storage-health.sh --smart-check

# Service communication testing
./scripts/test-service-communication.sh --matrix
```

#### Log Analysis

```bash
# Automated log analysis
./scripts/analyze-logs.sh --service all --timeframe 24h --errors-only

# Search for specific errors
./scripts/search-logs.sh --pattern "ERROR" --service jellyfin --last 1h

# Generate error summary
./scripts/error-summary.sh --output /tmp/error-report.html
```

### Emergency Procedures

#### Service Recovery

```bash
# Emergency service restart
./scripts/emergency-restart.sh --service jellyfin --force

# Full platform restart (last resort)
./scripts/emergency-platform-restart.sh --backup-first

# Rollback to previous working state
./scripts/emergency-rollback.sh --to-backup 20240115-023000
```

#### Data Recovery

```bash
# Recover corrupted database
./scripts/recover-database.sh --from-backup --verify-integrity

# Restore from disaster
./scripts/disaster-recovery.sh --emergency-mode --skip-confirmation

# Partial data restoration
./scripts/restore-partial.sh --component user-data --merge-mode
```

---

## Advanced Configuration

### Custom Integrations

#### Third-Party Service Integration

```yaml
# config/integrations/external-services.yml
integrations:
  plex:
    enabled: false
    api_url: "http://plex-server:32400"
    token: "your-plex-token"
    sync_watched: true
    
  emby:
    enabled: false
    api_url: "http://emby-server:8096"
    api_key: "your-emby-api-key"
    
  discord:
    enabled: true
    webhook_url: "https://discord.com/api/webhooks/..."
    notifications:
      - "new_content"
      - "system_alerts"
      - "maintenance_windows"
      
  telegram:
    enabled: false
    bot_token: "your-bot-token"
    chat_id: "your-chat-id"
```

#### Custom Scripts

```bash
# Custom notification script
cat > /opt/nexus/scripts/custom-notify.sh << 'EOF'
#!/bin/bash
# Send notifications to multiple channels

MESSAGE="$1"
SEVERITY="$2"

# Discord notification
if [ "$ENABLE_DISCORD" = "true" ]; then
  curl -X POST "$DISCORD_WEBHOOK" \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"[$SEVERITY] $MESSAGE\"}"
fi

# Email notification
if [ "$ENABLE_EMAIL" = "true" ]; then
  echo "$MESSAGE" | mail -s "NEXUS Alert: $SEVERITY" "$ADMIN_EMAIL"
fi

# Slack notification
if [ "$ENABLE_SLACK" = "true" ]; then
  curl -X POST "$SLACK_WEBHOOK" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"[$SEVERITY] $MESSAGE\"}"
fi
EOF

chmod +x /opt/nexus/scripts/custom-notify.sh
```

### Multi-Site Configuration

#### Distributed Deployment

```yaml
# docker-compose.distributed.yml
version: '3.8'

services:
  jellyfin-primary:
    image: jellyfin/jellyfin:latest
    deploy:
      placement:
        constraints: [node.labels.site == primary]
    volumes:
      - media-primary:/media
      
  jellyfin-secondary:
    image: jellyfin/jellyfin:latest
    deploy:
      placement:
        constraints: [node.labels.site == secondary]
    volumes:
      - media-secondary:/media
      
  ai-orchestrator:
    image: nexus/ai-orchestrator:latest
    deploy:
      placement:
        constraints: [node.labels.gpu == true]
      replicas: 2
      
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    configs:
      - source: nginx-multi-site
        target: /etc/nginx/nginx.conf
```

### Custom Dashboard Configuration

```javascript
// config/dashboard/custom-config.js
const customDashboardConfig = {
  theme: {
    primary_color: "#6366f1",
    secondary_color: "#8b5cf6",
    background: "dark",
    layout: "grid"
  },
  
  widgets: [
    {
      type: "service_status",
      position: { x: 0, y: 0, w: 6, h: 3 },
      config: {
        services: ["jellyfin", "sonarr", "radarr", "ai-orchestrator"]
      }
    },
    {
      type: "system_metrics",
      position: { x: 6, y: 0, w: 6, h: 3 },
      config: {
        metrics: ["cpu", "memory", "disk", "network"]
      }
    },
    {
      type: "recent_activity",
      position: { x: 0, y: 3, w: 12, h: 4 },
      config: {
        limit: 10,
        show_user_activity: true
      }
    }
  ],
  
  notifications: {
    position: "top-right",
    auto_dismiss: 5000,
    types: ["error", "warning", "info", "success"]
  }
};
```

---

This comprehensive administrator guide provides detailed instructions for managing all aspects of the NEXUS Media Server Platform. Regular review and updates of administrative procedures ensure optimal system performance and security.

**Key Administrative Principles:**
- **Proactive Monitoring**: Prevent issues before they occur
- **Security First**: Always prioritize security in decisions
- **Documentation**: Keep detailed records of all changes
- **Testing**: Test all changes in non-production first
- **Backup Strategy**: Maintain robust backup and recovery procedures
- **User Communication**: Keep users informed of maintenance and changes

**Emergency Contacts:**
- System Administrator: admin@nexus-platform.com
- Technical Support: support@nexus-platform.com  
- Security Issues: security@nexus-platform.com

**Last Updated**: January 2025  
**Guide Version**: 2.1  
**Platform Compatibility**: NEXUS 2025.1+