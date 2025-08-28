# Ultimate Media Server - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- 8GB+ RAM recommended
- 100GB+ storage space

### Step 1: Clone & Setup Environment
```bash
cd /Users/morlock/fun/newmedia
cp .env.fixed .env
# Edit .env with your settings
```

### Step 2: Run the Fix Script
```bash
chmod +x scripts/fix-all-services.sh
./scripts/fix-all-services.sh
```

### Step 3: Deploy Services
```bash
# Use the fixed Docker Compose file
docker-compose -f docker-compose-fixed.yml up -d
```

### Step 4: Monitor Deployment
```bash
# Check service status
docker-compose -f docker-compose-fixed.yml ps

# View logs
docker-compose -f docker-compose-fixed.yml logs -f
```

## 📊 Service URLs

### Media Servers
- **Jellyfin**: http://localhost:8096
- **Plex**: http://localhost:32400/web

### Content Management
- **Sonarr**: http://localhost:8989
- **Radarr**: http://localhost:7878
- **Lidarr**: http://localhost:8686
- **Prowlarr**: http://localhost:9696
- **Bazarr**: http://localhost:6767

### Download Clients
- **qBittorrent**: http://localhost:8080 (admin/adminadmin)
- **Transmission**: http://localhost:9091
- **SABnzbd**: http://localhost:8081

### Monitoring
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Uptime Kuma**: http://localhost:3001

### Management
- **Portainer**: https://localhost:9443
- **Nginx Proxy Manager**: http://localhost:81
- **Homarr**: http://localhost:7575

### Custom Services
- **Media Dashboard**: http://localhost:3030
- **API Server**: http://localhost:3002
- **Health Monitor**: http://localhost:3010/status

## 🔧 Configuration Steps

### 1. Configure Prowlarr
1. Access Prowlarr at http://localhost:9696
2. Go to Settings → General → Get API Key
3. Add indexers (Settings → Indexers)
4. Configure apps (Settings → Apps):
   - Add Sonarr: http://sonarr:8989
   - Add Radarr: http://radarr:7878

### 2. Configure Sonarr/Radarr
1. Access each service
2. Go to Settings → General → Get API Key
3. Configure download clients:
   - Settings → Download Clients → Add qBittorrent
   - Host: qbittorrent, Port: 8080
   - Username: admin, Password: adminadmin
4. Set media paths:
   - Sonarr: /media/tv
   - Radarr: /media/movies

### 3. Configure Jellyfin
1. Access http://localhost:8096
2. Complete setup wizard
3. Add media libraries:
   - Movies: /media/movies
   - TV Shows: /media/tv
   - Music: /media/music

### 4. Configure qBittorrent
1. Access http://localhost:8080
2. Login: admin/adminadmin
3. Settings → Downloads:
   - Default Save Path: /downloads/complete
   - Temp Path: /downloads/incomplete

## 🐛 Troubleshooting

### Services Not Starting
```bash
# Check logs
docker-compose -f docker-compose-fixed.yml logs [service-name]

# Restart specific service
docker-compose -f docker-compose-fixed.yml restart [service-name]

# Rebuild and restart
docker-compose -f docker-compose-fixed.yml up -d --build [service-name]
```

### Network Issues
```bash
# Recreate networks
./scripts/fix-all-services.sh

# Check network connectivity
docker network inspect media-net
```

### Permission Issues
```bash
# Fix permissions
sudo chown -R 1000:1000 ./config ./media ./downloads
chmod -R 755 ./config ./media ./downloads
```

### Database Connection Issues
```bash
# Check database status
docker-compose -f docker-compose-fixed.yml logs postgres
docker-compose -f docker-compose-fixed.yml logs redis

# Restart databases
docker-compose -f docker-compose-fixed.yml restart postgres redis
```

## 📈 Health Monitoring

### Check All Services
```bash
curl http://localhost:3010/status
```

### Individual Health Checks
```bash
# Jellyfin
curl http://localhost:8096/health

# Sonarr
curl http://localhost:8989/ping

# Radarr
curl http://localhost:7878/ping

# API Server
curl http://localhost:3002/health
```

## 🔐 Security Notes

1. **Change Default Passwords**:
   - Update all passwords in `.env`
   - Change qBittorrent password after first login
   - Set strong Grafana admin password

2. **Configure Firewall**:
   - Only expose necessary ports
   - Use reverse proxy for external access
   - Enable SSL/TLS for production

3. **Regular Updates**:
   ```bash
   docker-compose -f docker-compose-fixed.yml pull
   docker-compose -f docker-compose-fixed.yml up -d
   ```

## 📚 Additional Resources

- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [Sonarr Wiki](https://wiki.servarr.com/sonarr)
- [Radarr Wiki](https://wiki.servarr.com/radarr)
- [Docker Compose Reference](https://docs.docker.com/compose/)

## 🎉 Success!

Your media server should now be fully operational. Start by:
1. Adding indexers in Prowlarr
2. Configuring download clients
3. Adding media to Sonarr/Radarr
4. Enjoying your content in Jellyfin!

For support, check the logs or refer to the troubleshooting section above.