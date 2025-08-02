# Docker Container Auto-Update System

An intelligent and safe auto-update system for Docker containers based on 2025 best practices. This system prioritizes safety, reliability, and control over automatic updates.

## 🚀 Features

- **Diun Integration**: Real-time notifications for available Docker image updates
- **Renovate Bot**: Automated pull requests for docker-compose.yml updates
- **Manual Update Control**: Safe update script with backup and rollback capabilities
- **Update Dashboard**: Web UI for monitoring and managing updates
- **Backup System**: Automatic backups before updates with configurable retention
- **Health Checks**: Validates service health after updates
- **Rollback Capability**: Automatic rollback on update failure

## 📋 Components

### 1. Diun (Docker Image Update Notifier)
- Monitors all running containers for available updates
- Sends notifications via email, webhooks, Discord, Slack, etc.
- Configurable update checking schedule
- Filters for critical services and auto-update candidates

### 2. Renovate Bot
- Creates automated pull requests for docker-compose.yml updates
- Groups related updates (media servers, databases, etc.)
- Configurable auto-merge for safe updates
- Security vulnerability alerts

### 3. Update Strategy Script
- Manual control over update process
- Pre-update backups
- Health check validation
- Automatic rollback on failure
- Dry-run mode for testing

### 4. Update Dashboard
- Web-based monitoring interface
- Real-time update status
- One-click updates for safe services
- Update history tracking

## 🛠️ Setup

### 1. Configure Environment Variables

Create a `.env` file with:

```bash
# GitHub Configuration (for Renovate)
GITHUB_TOKEN=your_github_token
GITHUB_REPOSITORY=user/repo

# Notification Configuration (for Diun)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=your_email@gmail.com
SMTP_TO=recipient@gmail.com

# Optional Webhook URL (Discord/Slack)
WEBHOOK_URL=https://discord.com/api/webhooks/...

# Domain Configuration
DOMAIN=yourdomain.com

# Timezone
TZ=UTC
```

### 2. Deploy the Update System

```bash
# Start the update monitoring system
docker-compose -f updates/docker-compose.updates.yml up -d

# Check service status
docker-compose -f updates/docker-compose.updates.yml ps
```

### 3. Configure Renovate

1. Fork or create your repository on GitHub
2. Update `GITHUB_REPOSITORY` in `.env`
3. Update repository name in `updates/renovate.json`
4. Renovate will create PRs for available updates

### 4. Access the Dashboard

Visit `https://updates.yourdomain.com` (or configure your preferred URL)

## 📖 Usage

### Check for Updates

```bash
# Check all services for updates
./updates/update-strategy.sh --check

# View update status in logs
docker logs diun
```

### Update Services

```bash
# Update all services (with backups and health checks)
./updates/update-strategy.sh --update

# Update specific service
./updates/update-strategy.sh --update-service jellyfin

# Dry run (simulate without changes)
./updates/update-strategy.sh --update --dry-run
```

### Backup Management

```bash
# Manual backup of specific service
./updates/update-strategy.sh --backup sonarr

# View backups
ls -la updates/backups/

# Restore from backup (if needed)
docker run --rm -v sonarr_data:/target -v ./backups:/backup alpine \
  tar xzf /backup/sonarr_20240115_143022_data.tar.gz -C /target
```

## 🔧 Configuration

### Diun Configuration

Edit `updates/diun-config.yml` to:
- Adjust check frequency
- Configure notification channels
- Set up filters for specific images
- Define critical services

### Update Strategy

Environment variables for `update-strategy.sh`:
- `BACKUP_DIR`: Backup location (default: ./backups)
- `MAX_BACKUPS`: Backups to keep per service (default: 10)
- `HEALTH_CHECK_TIMEOUT`: Health check timeout (default: 300s)
- `ROLLBACK_ON_FAILURE`: Auto-rollback on failure (default: true)

### Renovate Configuration

Edit `updates/renovate.json` to:
- Configure auto-merge rules
- Group related updates
- Set update schedules
- Define package rules

## 🚨 Best Practices

1. **Never Auto-Update Critical Services**: Services like Traefik, Authelia, and databases require manual review
2. **Test Updates First**: Use dry-run mode or test in staging environment
3. **Schedule Updates**: Perform updates during maintenance windows
4. **Monitor After Updates**: Check logs and service health after updates
5. **Keep Backups**: Ensure backup retention covers your recovery needs
6. **Review Changelogs**: Always check for breaking changes before updating

## 🛡️ Security Considerations

- Renovate requires GitHub token with repo permissions
- SMTP credentials should use app-specific passwords
- Webhook URLs should be kept private
- Dashboard should be protected with authentication (via Traefik/Authelia)
- Regular security updates are critical but require testing

## 📊 Monitoring

### Diun Metrics
- Access Diun metrics at `http://diun:8080/metrics`
- Monitor with Prometheus/Grafana

### Update Logs
```bash
# View Diun logs
docker logs -f diun

# View update script logs
tail -f updates/update.log

# View Renovate logs
docker logs renovate
```

## 🔄 Troubleshooting

### Common Issues

1. **Diun not detecting updates**
   - Check Docker socket permissions
   - Verify registry access
   - Check container labels

2. **Renovate not creating PRs**
   - Verify GitHub token permissions
   - Check repository configuration
   - Review Renovate logs

3. **Update failures**
   - Check service logs
   - Verify disk space
   - Review health check configuration

### Manual Rollback

If automatic rollback fails:
```bash
# Stop failed container
docker-compose stop service_name

# Restore from backup
docker run --rm -v service_data:/target -v ./backups:/backup alpine \
  tar xzf /backup/service_backup.tar.gz -C /target

# Start previous version
docker-compose up -d service_name
```

## 📚 Additional Resources

- [Diun Documentation](https://crazymax.dev/diun/)
- [Renovate Documentation](https://docs.renovatebot.com/)
- [Docker Compose Best Practices](https://docs.docker.com/compose/compose-file/compose-file-v3/)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is provided as-is for educational and personal use.