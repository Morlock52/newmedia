# Ultimate Media Server 2025 - Troubleshooting Guide

This guide helps you resolve common issues with your media server deployment.

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Service-Specific Problems](#service-specific-problems)
4. [Performance Issues](#performance-issues)
5. [Network Problems](#network-problems)
6. [Storage Issues](#storage-issues)
7. [Docker Problems](#docker-problems)
8. [Log Analysis](#log-analysis)
9. [Recovery Procedures](#recovery-procedures)
10. [Getting Help](#getting-help)

---

## Quick Diagnostics

### 🔍 System Health Check Script

```bash
#!/bin/bash
# Save as health-check.sh and run

echo "🔍 Media Server Health Check"
echo "=========================="

# Check Docker
echo -n "Docker Status: "
if docker version > /dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Running"
fi

# Check containers
echo -e "\n📦 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}"

# Check disk space
echo -e "\n💾 Disk Space:"
df -h | grep -E "^/|Filesystem"

# Check memory
echo -e "\n🧠 Memory Usage:"
free -h

# Check ports
echo -e "\n🔌 Port Status:"
for port in 8096 7878 8989 9696 8080 3001; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Port $port: ✅ In use"
    else
        echo "Port $port: ❌ Free"
    fi
done
```

### 🚀 Quick Fix Commands

```bash
# Restart all services
docker-compose restart

# Rebuild a specific service
docker-compose up -d --force-recreate jellyfin

# Clear Docker issues
docker system prune -a --volumes

# Reset permissions
sudo chown -R $USER:$USER ./config ./media-data
```

---

## Common Issues

### 🔴 Services Won't Start

**Symptoms:**
- Container exits immediately
- Status shows "Restarting"
- Services unreachable

**Solutions:**

1. **Check logs first:**
   ```bash
   docker-compose logs servicename
   ```

2. **Port conflicts:**
   ```bash
   # Find what's using a port
   sudo lsof -i :8096
   
   # Kill process using port
   sudo kill -9 $(sudo lsof -t -i:8096)
   ```

3. **Permission issues:**
   ```bash
   # Fix ownership
   sudo chown -R $USER:$USER ./config
   
   # Fix permissions
   find ./config -type d -exec chmod 755 {} \;
   find ./config -type f -exec chmod 644 {} \;
   ```

4. **Resource limits:**
   ```bash
   # Check Docker resources
   docker system df
   
   # Increase Docker Desktop memory (GUI)
   # Or edit daemon.json for Linux
   ```

### 🔴 Cannot Access Web Interfaces

**Symptoms:**
- "Connection refused"
- "Site can't be reached"
- Timeouts

**Solutions:**

1. **Verify container is running:**
   ```bash
   docker ps | grep jellyfin
   ```

2. **Check correct URL:**
   ```bash
   # Local access
   http://localhost:8096
   
   # Network access
   http://YOUR-SERVER-IP:8096
   ```

3. **Firewall issues:**
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8096/tcp
   
   # CentOS/RHEL
   sudo firewall-cmd --add-port=8096/tcp --permanent
   sudo firewall-cmd --reload
   ```

4. **Browser issues:**
   - Clear cache/cookies
   - Try incognito mode
   - Try different browser

### 🔴 Authentication Problems

**Symptoms:**
- Can't login
- Forgot password
- Account locked

**Solutions:**

1. **Reset Jellyfin admin:**
   ```bash
   docker exec -it jellyfin /bin/bash
   # Inside container:
   /usr/lib/jellyfin/bin/jellyfin --datadir /config --resetadminpassword
   ```

2. **Reset Sonarr/Radarr:**
   ```bash
   # Stop service
   docker-compose stop sonarr
   
   # Edit config
   nano ./config/sonarr/config.xml
   # Set <AuthenticationMethod>None</AuthenticationMethod>
   
   # Restart
   docker-compose start sonarr
   ```

3. **Reset qBittorrent:**
   ```bash
   # Default: admin/adminadmin
   # To reset:
   docker exec -it qbittorrent /bin/bash
   rm /config/qBittorrent/qBittorrent.conf
   # Restart container
   ```

---

## Service-Specific Problems

### 📺 Jellyfin Issues

**Playback Problems:**
```bash
# Enable hardware acceleration
docker exec -it jellyfin /bin/bash
ls -la /dev/dri  # Check GPU access

# Fix permissions
docker exec -it jellyfin /bin/bash
chmod 666 /dev/dri/*
```

**Library Not Scanning:**
```bash
# Check permissions
docker exec -it jellyfin ls -la /media

# Force rescan
curl -X POST "http://localhost:8096/Library/Refresh" \
  -H "X-Emby-Token: YOUR_API_KEY"
```

**Subtitle Issues:**
```bash
# Install additional fonts
docker exec -it jellyfin apt update
docker exec -it jellyfin apt install fonts-noto
```

### 🎬 Sonarr/Radarr Issues

**Indexer Problems:**
```yaml
# Check Prowlarr connection
http://localhost:9696/settings/apps

# Test indexer
http://localhost:8989/settings/indexers
```

**Download Client Issues:**
```bash
# Verify qBittorrent API
curl http://localhost:8080/api/v2/app/version

# Check paths match
docker exec -it sonarr ls -la /downloads
docker exec -it qbittorrent ls -la /downloads
```

**Import Failures:**
- Check file permissions
- Verify path mappings
- Enable debug logging
- Check disk space

### 📥 Download Client Issues

**qBittorrent Won't Download:**
```bash
# Check VPN connection (if using)
docker exec -it gluetun curl ifconfig.me

# Check port forwarding
docker logs gluetun | grep "port forwarding"

# Reset qBittorrent
docker-compose restart qbittorrent
```

**Slow Downloads:**
```bash
# Check connection limits
# qBittorrent → Settings → Connection
# Global maximum connections: 200
# Per torrent connections: 100

# Check bandwidth limits
# Settings → Speed → Alternative rate limits
```

---

## Performance Issues

### 🐌 Slow Performance

**Diagnose:**
```bash
# Check CPU usage
docker stats

# Check disk I/O
iostat -x 1

# Check network
iftop
```

**Optimize:**

1. **Database optimization:**
   ```bash
   # Jellyfin
   docker exec -it jellyfin sqlite3 /config/data/library.db "VACUUM;"
   
   # Sonarr/Radarr
   docker exec -it sonarr sqlite3 /config/sonarr.db "VACUUM;"
   ```

2. **Limit resource usage:**
   ```yaml
   # docker-compose.yml
   services:
     jellyfin:
       deploy:
         resources:
           limits:
             cpus: '4'
             memory: 4G
   ```

3. **Enable caching:**
   ```bash
   # Redis for caching
   docker run -d --name redis redis:alpine
   ```

### 🔥 High CPU Usage

**Identify culprit:**
```bash
docker stats --no-stream
```

**Common causes:**
- Transcoding (Jellyfin)
- Library scanning
- Download unpacking
- Thumbnail generation

**Solutions:**
```bash
# Limit transcoding threads
# Jellyfin → Dashboard → Playback
# Transcoding thread count: 4

# Schedule scans for off-hours
# Sonarr → Settings → Media Management
# Rescan interval: Manual
```

### 💾 Memory Leaks

**Monitor:**
```bash
# Watch memory over time
docker stats servicename

# Check for OOM kills
dmesg | grep -i "killed process"
```

**Fix:**
```bash
# Restart service periodically
crontab -e
# Add: 0 4 * * * docker restart servicename

# Limit memory
docker update --memory="2g" servicename
```

---

## Network Problems

### 🌐 Remote Access Issues

**Can't access from outside network:**

1. **Port forwarding:**
   ```bash
   # Router settings needed:
   # External Port → Internal IP:Port
   # 8096 → 192.168.1.100:8096
   ```

2. **Dynamic DNS:**
   ```bash
   # Use service like DuckDNS
   docker run -d \
     --name=duckdns \
     -e TOKEN=your-token \
     -e SUBDOMAINS=yourdomain \
     ghcr.io/linuxserver/duckdns
   ```

3. **Reverse proxy:**
   ```nginx
   # Nginx Proxy Manager
   # http://localhost:81
   # Add proxy host for each service
   ```

### 🔒 VPN Issues

**Services unreachable with VPN:**
```bash
# Check VPN status
docker logs gluetun

# Test connectivity
docker exec -it gluetun curl ifconfig.me

# Bypass VPN for local access
# Add to docker-compose.yml:
networks:
  - media-net
  - vpn-bypass
```

**Port forwarding with VPN:**
```yaml
# gluetun environment
VPN_PORT_FORWARDING: "on"
VPN_PORT_FORWARDING_PROVIDER: "provider_name"
```

---

## Storage Issues

### 💽 Out of Space

**Quick cleanup:**
```bash
# Find large files
find ./media-data -type f -size +1G -exec ls -lh {} \;

# Clean Docker
docker system prune -a --volumes

# Remove old logs
find ./config -name "*.log" -mtime +30 -delete

# Clear thumbnails
rm -rf ./config/jellyfin/metadata
```

**Prevent future issues:**
```bash
# Monitor disk usage
df -h

# Set up alerts
# Uptime Kuma → Add Monitor → Type: Docker Host
```

### 🔄 Permissions Errors

**Fix ownership:**
```bash
# Find your UID/GID
id

# Update .env
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env

# Fix existing files
docker-compose down
sudo chown -R $(id -u):$(id -g) ./config ./media-data
docker-compose up -d
```

**Prevent issues:**
```yaml
# docker-compose.yml
environment:
  - PUID=${PUID}
  - PGID=${PGID}
  - UMASK=002
```

---

## Docker Problems

### 🐳 Docker Daemon Issues

**Docker won't start:**
```bash
# Linux
sudo systemctl status docker
sudo systemctl restart docker
sudo journalctl -u docker.service

# Check disk space for Docker
df -h /var/lib/docker
```

**Clean up Docker:**
```bash
# Remove unused data
docker system prune -a

# Remove specific types
docker image prune
docker container prune
docker volume prune
docker network prune

# Nuclear option (removes everything)
docker system prune -a --volumes
```

### 📦 Container Crashes

**Debug crash loops:**
```bash
# View logs
docker logs --tail 50 -f containername

# Run interactively
docker run -it --rm \
  -v ./config/jellyfin:/config \
  jellyfin/jellyfin:latest \
  /bin/bash

# Check exit codes
docker inspect containername | grep -i exit
```

**Common fixes:**
- Delete corrupted config files
- Recreate container
- Check volume permissions
- Verify image compatibility

---

## Log Analysis

### 📊 Important Log Locations

**Container logs:**
```bash
# Real-time logs
docker-compose logs -f servicename

# Last 100 lines
docker logs --tail 100 servicename

# Save to file
docker logs servicename > service.log 2>&1
```

**Application logs:**
```bash
# Jellyfin
./config/jellyfin/log/

# Sonarr/Radarr
./config/sonarr/logs/
./config/radarr/logs/

# qBittorrent
./config/qbittorrent/qBittorrent/logs/
```

### 🔍 Log Analysis Tools

```bash
# Search for errors
grep -i error ./config/*/logs/*.txt

# Monitor in real-time
tail -f ./config/jellyfin/log/*.log

# Count occurrences
grep -c "failed" ./config/sonarr/logs/sonarr.txt

# Find recent issues
find ./config -name "*.log" -mtime -1 -exec grep -l ERROR {} \;
```

### 📈 Monitoring Setup

```yaml
# Add Loki for log aggregation
loki:
  image: grafana/loki:latest
  ports:
    - "3100:3100"
  volumes:
    - ./loki-config:/etc/loki
    - ./loki-data:/loki

# Add Promtail for log shipping
promtail:
  image: grafana/promtail:latest
  volumes:
    - ./config:/var/log
    - ./promtail-config:/etc/promtail
```

---

## Recovery Procedures

### 💾 Backup Recovery

**Before making changes:**
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configs
docker-compose down
cp -r ./config "$BACKUP_DIR/"
docker-compose up -d

echo "Backup saved to $BACKUP_DIR"
```

**Restore from backup:**
```bash
# Stop services
docker-compose down

# Restore configs
cp -r ./backups/20240301_120000/config/* ./config/

# Start services
docker-compose up -d
```

### 🔧 Service Recovery

**Corrupted database:**
```bash
# Jellyfin
docker-compose stop jellyfin
mv ./config/jellyfin/data/library.db ./config/jellyfin/data/library.db.corrupt
docker-compose start jellyfin
# Will recreate database - rescan libraries

# Sonarr/Radarr
docker-compose stop sonarr
cd ./config/sonarr
mv sonarr.db sonarr.db.backup
mv sonarr.db-shm sonarr.db-shm.backup
mv sonarr.db-wal sonarr.db-wal.backup
docker-compose start sonarr
```

**Complete reset:**
```bash
# WARNING: Loses all settings
docker-compose down
docker-compose rm servicename
rm -rf ./config/servicename
docker-compose up -d servicename
```

---

## Getting Help

### 📝 Information to Gather

When asking for help, provide:

1. **System info:**
   ```bash
   uname -a
   docker version
   docker-compose version
   ```

2. **Error logs:**
   ```bash
   docker logs --tail 100 servicename
   ```

3. **Configuration:**
   ```bash
   # Sanitized .env
   cat .env | grep -v PASSWORD
   ```

4. **Resource usage:**
   ```bash
   docker stats --no-stream
   df -h
   free -h
   ```

### 🆘 Support Channels

**Community Support:**
- GitHub Issues: [Create detailed issue](https://github.com/yourusername/ultimate-media-server/issues)
- Discord: [Join server](https://discord.gg/mediaserver)
- Reddit: [r/selfhosted](https://reddit.com/r/selfhosted)
- Forums: [LinuxServer.io](https://discourse.linuxserver.io/)

**Documentation:**
- Service wikis
- Docker Hub pages
- Official docs

**Debug Mode:**
```yaml
# Enable debug logging
environment:
  - LOG_LEVEL=debug
  - DEBUG=true
```

---

## Preventive Maintenance

### 📅 Regular Tasks

**Daily:**
- Monitor disk space
- Check service health
- Review error logs

**Weekly:**
- Update containers
- Clean old logs
- Verify backups

**Monthly:**
- Full system backup
- Performance review
- Security updates

### 🛡️ Best Practices

1. **Always backup before major changes**
2. **Test updates on single service first**
3. **Monitor resource usage trends**
4. **Document your configuration**
5. **Keep logs for troubleshooting**
6. **Use health checks in compose**
7. **Set resource limits**
8. **Regular security updates**

---

*Remember: Most issues have simple solutions. Stay calm and check the logs!* 🔍