# Media Server Troubleshooting Guide - 2025

## 🔧 Quick Diagnostic Commands

```bash
# Check all container status
docker compose -f docker-compose-2025-enhanced.yml ps

# Check container logs
docker compose -f docker-compose-2025-enhanced.yml logs [service_name]

# Check system resources
docker stats

# Check network connectivity
docker exec jellyfin curl -f http://sonarr:8989/ping

# Check VPN status
docker exec gluetun curl -s https://ipinfo.io/json
```

## 🎬 Jellyfin Issues

### **Hardware Transcoding Not Working**

**Symptoms:**
- CPU transcoding instead of GPU
- High CPU usage during streaming
- "Hardware acceleration not available" errors

**Intel GPU Troubleshooting:**
```bash
# Check if Intel GPU is available
ls -la /dev/dri/
# Should show: renderD128, card0

# Check render group membership
groups $USER
# Should include: render

# Check container has access
docker exec jellyfin ls -la /dev/dri/

# Add user to render group if missing
sudo usermod -a -G render $USER
# Then restart containers
```

**NVIDIA GPU Troubleshooting:**
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container access
docker exec jellyfin nvidia-smi

# Verify environment variables
docker exec jellyfin env | grep NVIDIA
```

**Fix Steps:**
1. Stop containers: `docker compose down`
2. Update .env with correct RENDER_GROUP_ID
3. Restart containers: `docker compose up -d`
4. Check Jellyfin Dashboard → Playbook → Transcoding
5. Enable hardware acceleration and test

### **Jellyfin Database Corruption**

**Symptoms:**
- Jellyfin won't start
- Database errors in logs
- Missing media libraries

**Recovery Steps:**
```bash
# Stop Jellyfin
docker stop jellyfin

# Backup current database
cp config/jellyfin/data/jellyfin.db config/jellyfin/data/jellyfin.db.backup

# Try to repair database
docker run --rm -v $(pwd)/config/jellyfin:/config \
  alpine:latest sh -c "
    apk add sqlite &&
    sqlite3 /config/data/jellyfin.db '.dump' > /tmp/dump.sql &&
    rm /config/data/jellyfin.db &&
    sqlite3 /config/data/jellyfin.db < /tmp/dump.sql
  "

# Start Jellyfin
docker start jellyfin
```

### **Jellyfin Performance Issues**

**Memory Issues:**
```bash
# Check Jellyfin memory usage
docker stats jellyfin

# Increase memory limit in compose file
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G

# Clear Jellyfin cache
rm -rf cache/jellyfin/*
docker restart jellyfin
```

## 🔽 Download Issues

### **qBittorrent Behind VPN Problems**

**Symptoms:**
- qBittorrent not accessible
- Downloads not starting
- Connection timeouts

**Troubleshooting:**
```bash
# Check VPN container status
docker logs gluetun

# Test VPN connection
docker exec gluetun curl -s https://ipinfo.io/json

# Check if qBittorrent is accessible through VPN
docker exec gluetun curl -f http://localhost:8080

# Test external connectivity
docker exec gluetun curl -f https://google.com
```

**Common Fixes:**

**1. VPN Authentication Issues:**
```bash
# Check VPN credentials in .env
grep VPN_ .env

# Update credentials and restart
docker restart gluetun
```

**2. Firewall Blocking Local Network:**
```bash
# Add local network to VPN firewall rules
# Update gluetun environment in compose file:
environment:
  - FIREWALL_OUTBOUND_SUBNETS=172.20.0.0/16,192.168.0.0/16,10.0.0.0/8
```

**3. Port Forwarding Issues:**
```bash
# Check if VPN provider supports port forwarding
# Update gluetun configuration for port forwarding
environment:
  - VPN_PORT_FORWARDING=on
  - VPN_PORT_FORWARDING_PROVIDER=pia  # for PIA
```

### **Slow Download Speeds**

**Check Points:**
```bash
# 1. Test raw VPN speed
docker exec gluetun curl -w "@curl-format.txt" -s -o /dev/null https://speed.cloudflare.com/__down?bytes=100000000

# 2. Check qBittorrent settings
# Open qBittorrent WebUI → Tools → Options
# Set upload/download limits to 80% of your connection
```

**Optimization Settings:**
```bash
# qBittorrent Advanced Settings:
disk.io_type = posix_aio
max_concurrent_http_announces = 50
max_connec_per_torrent = 100
max_uploads_per_torrent = 15
```

## 🔍 *arr Suite Issues

### **Prowlarr Indexer Problems**

**Symptoms:**
- Indexers showing red/failed status
- No search results
- Rate limiting errors

**Troubleshooting:**
```bash
# Check Prowlarr logs
docker logs prowlarr | grep -i error

# Test indexer manually
docker exec prowlarr curl -f "http://indexer-url.com/api"

# Check if VPN is blocking access
docker exec gluetun curl -f "http://indexer-url.com"
```

**Common Solutions:**

**1. Rate Limiting:**
- Reduce search frequency in Prowlarr settings
- Add delays between requests
- Use fewer indexers simultaneously

**2. VPN Blocking:**
```bash
# Some indexers block VPN IPs
# Create bypass for specific indexers by running Prowlarr outside VPN
# Or whitelist specific domains in VPN configuration
```

### **Sonarr/Radarr Not Finding Releases**

**Diagnosis Steps:**
```bash
# 1. Check Prowlarr indexer status
# Prowlarr → Indexers → Test all

# 2. Check app integration
# Prowlarr → Apps → Test Sonarr/Radarr connection

# 3. Manual search test
# Sonarr → Series → Manual search for episode
```

**Common Issues:**

**1. Wrong Category Mapping:**
```bash
# Check Prowlarr → Settings → Apps
# Sonarr Categories: TV, TV/WEB-DL, TV/HD
# Radarr Categories: Movies, Movies/HD, Movies/UHD
```

**2. Quality Profile Issues:**
```bash
# Check Sonarr/Radarr → Settings → Profiles
# Ensure allowed qualities match available releases
```

### **Import Issues (Hardlink Problems)**

**Symptoms:**
- Files copied instead of moved
- Doubled storage usage
- Slow import process

**Fix Steps:**
```bash
# 1. Check directory structure
ls -la data/
# Should have: downloads/, media/ in same filesystem

# 2. Test hardlink capability
touch data/downloads/test
ln data/downloads/test data/media/test
ls -i data/*/test  # Should show same inode number
rm data/*/test

# 3. Update *arr settings
# Download Client → qBittorrent → Remove completed: No
# Media Management → Use Hardlinks: Yes
```

## 🌐 Network & SSL Issues

### **SSL Certificate Problems**

**Symptoms:**
- "Certificate not valid" errors
- Traefik showing certificate errors
- Websites not accessible via HTTPS

**Troubleshooting:**
```bash
# Check Traefik logs
docker logs traefik | grep -i "acme\|cert\|error"

# Check certificate status
docker exec traefik ls -la /acme.json

# Test DNS resolution
nslookup jellyfin.yourdomain.com
dig @1.1.1.1 jellyfin.yourdomain.com

# Check Cloudflare API
curl -X GET "https://api.cloudflare.com/client/v4/zones" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type:application/json"
```

**Common Fixes:**

**1. DNS Propagation Issues:**
```bash
# Wait for DNS propagation (up to 24 hours)
# Check propagation: https://dnschecker.org/

# Force certificate renewal
docker exec traefik rm /acme.json
docker restart traefik
```

**2. Cloudflare API Token Issues:**
```bash
# Verify token permissions:
# Zone:DNS:Edit, Zone:Zone:Read for all zones

# Test token
curl -X GET "https://api.cloudflare.com/client/v4/user/tokens/verify" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

### **Cannot Access Services Remotely**

**Diagnosis:**
```bash
# 1. Check if ports are open
sudo ufw status
nmap -p 80,443 your-server-ip

# 2. Check Cloudflare proxy status
# Should be orange cloud (proxied)

# 3. Test local access
curl -k https://jellyfin.yourdomain.com
```

**Solutions:**

**1. Firewall Issues:**
```bash
# Allow HTTP/HTTPS through firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
```

**2. Router/ISP Issues:**
```bash
# Check if ISP blocks ports 80/443
# Try alternative ports in Traefik configuration
```

## 💾 Storage & Performance Issues

### **Disk Space Problems**

**Monitoring:**
```bash
# Check disk usage
df -h
du -sh data/*

# Check Docker space usage
docker system df

# Check container logs size
sudo du -sh /var/lib/docker/containers/*/
```

**Cleanup:**
```bash
# Clean up Docker
docker system prune -a

# Clean up downloads
find data/downloads -name "*.part" -delete
find data/downloads -name "*.!qB" -delete

# Clean up old logs
docker exec jellyfin find /config/log -name "*.log.*" -mtime +7 -delete
```

### **High I/O Usage**

**Diagnosis:**
```bash
# Check I/O usage
iotop
sudo iotop -o

# Check for high I/O containers
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.BlockIO}}"
```

**Optimizations:**
```bash
# 1. Move transcodes to RAM disk
sudo mkdir /mnt/transcode-ram
sudo mount -t tmpfs -o size=4G tmpfs /mnt/transcode-ram

# Update docker-compose.yml
volumes:
  - /mnt/transcode-ram:/transcodes

# 2. Optimize qBittorrent
# Set memory usage: 512MB-1GB
# Enable disk cache: Yes
```

## 🔄 Container Issues

### **Container Won't Start**

**Diagnosis:**
```bash
# Check container status
docker ps -a

# Check container logs
docker logs container_name

# Check for port conflicts
sudo netstat -tulpn | grep :8096
```

**Common Solutions:**

**1. Port Already in Use:**
```bash
# Find process using port
sudo lsof -i :8096

# Kill process or change port in compose file
```

**2. Permission Issues:**
```bash
# Fix ownership
sudo chown -R $USER:$USER config/ data/

# Check PUID/PGID in .env
id $(whoami)
```

**3. Resource Exhaustion:**
```bash
# Check available resources
free -h
df -h

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### **Container Keeps Restarting**

**Investigation:**
```bash
# Check restart policy
docker inspect container_name | grep -i restart

# Check exit codes
docker ps -a

# Monitor restart events
docker events --filter container=container_name
```

**Solutions:**

**1. Configuration Errors:**
```bash
# Validate docker-compose syntax
docker compose -f docker-compose-2025-enhanced.yml config

# Check environment variables
docker exec container_name env
```

**2. Health Check Failures:**
```bash
# Disable health checks temporarily
# Comment out healthcheck section in compose file
# Investigate why health check is failing
```

## 🛠️ Maintenance Commands

### **Daily Maintenance**
```bash
#!/bin/bash
# daily-maintenance.sh

# Check container health
docker compose ps

# Check disk space
df -h | grep -E "/$|/data"

# Clean up old logs
find ./config/*/logs -name "*.log.*" -mtime +7 -delete

# Update container stats
docker stats --no-stream > /tmp/docker-stats.log
```

### **Weekly Maintenance**
```bash
#!/bin/bash
# weekly-maintenance.sh

# Update all containers
docker compose pull
docker compose up -d

# Clean up Docker
docker system prune -f

# Backup configurations
tar -czf "backup-$(date +%Y%m%d).tar.gz" config/ .env

# Check SSL certificates
openssl s_client -connect jellyfin.yourdomain.com:443 \
  -servername jellyfin.yourdomain.com < /dev/null 2>/dev/null | \
  openssl x509 -noout -dates
```

### **Emergency Recovery**
```bash
#!/bin/bash
# emergency-recovery.sh

echo "🚨 Emergency Recovery Mode"

# Stop all services
docker compose down

# Check system resources
echo "System Resources:"
free -h
df -h

# Check for corruption
echo "Checking for issues..."
docker system df
docker system events --since 1h

# Restore from backup
echo "Available backups:"
ls -la backup-*.tar.gz

read -p "Enter backup file to restore: " backup_file
if [ -f "$backup_file" ]; then
    tar -xzf "$backup_file"
    echo "Backup restored"
fi

# Start services
docker compose up -d

echo "Recovery complete. Check logs for any issues."
```

## 📞 Getting Help

### **Log Collection for Support**
```bash
#!/bin/bash
# collect-logs.sh

mkdir -p support-logs
cd support-logs

# System information
uname -a > system-info.txt
docker version >> system-info.txt
docker compose version >> system-info.txt

# Container status
docker ps -a > container-status.txt

# Container logs (last 100 lines)
for container in jellyfin sonarr radarr prowlarr qbittorrent gluetun traefik; do
    docker logs --tail 100 $container > ${container}-logs.txt 2>&1
done

# System logs
journalctl -u docker --since "1 hour ago" > docker-system.log

# Create archive
cd ..
tar -czf "support-logs-$(date +%Y%m%d_%H%M%S).tar.gz" support-logs/
rm -rf support-logs/

echo "Support logs collected in support-logs-*.tar.gz"
```

### **Community Resources**
- **Jellyfin**: https://forum.jellyfin.org/
- **Servarr (Sonarr/Radarr)**: https://wiki.servarr.com/
- **TRaSH Guides**: https://trash-guides.info/
- **Docker**: https://forums.docker.com/
- **LinuxServer.io**: https://discourse.linuxserver.io/

### **Before Posting for Help**
1. Collect logs using the script above
2. Describe what you were trying to do
3. Include error messages
4. Mention your setup (OS, Docker version, etc.)
5. Include relevant configuration (remove sensitive data)

---

**Last Updated**: 2025-07-27  
**Guide Version**: 2025.1  
**Covers**: Docker Compose setup with Traefik v3