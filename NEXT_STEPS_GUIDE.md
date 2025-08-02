# 🚀 NEXT STEPS CONFIGURATION GUIDE

## 🎯 IMMEDIATE SETUP TASKS

### 1. 🎬 Configure Jellyfin Media Server
**URL:** http://localhost:8096

**Setup Steps:**
1. Open Jellyfin and complete the initial setup wizard
2. Create admin account
3. Add media libraries:
   - Movies: `/media/movies`
   - TV Shows: `/media/tv`
   - Music: `/media/music`
4. Enable hardware acceleration (Intel VAAPI)
5. Set up remote access (optional)

### 2. 📥 Configure Download Clients

#### qBittorrent (http://localhost:8080)
1. Default login: `admin` / `adminadmin`
2. Change default password in Tools > Options > Web UI
3. Set download directory to `/downloads`
4. Configure categories for different media types

#### SABnzbd (http://localhost:8081)
1. Complete initial setup wizard
2. Add Usenet provider(s)
3. Set download directory to `/downloads`
4. Configure categories and scripts

### 3. 🔍 Configure Prowlarr Indexers
**URL:** http://localhost:9696

**Setup Steps:**
1. Add indexers (torrent sites, Usenet providers)
2. Configure API keys for private trackers
3. Test indexer connections
4. Set up sync with *arr applications

### 4. 🤖 Configure Automation Apps

#### Sonarr (TV Shows) - http://localhost:8989
1. Add download client (qBittorrent/SABnzbd)
2. Add indexers from Prowlarr
3. Configure media management:
   - Root folder: `/media/tv`
   - File naming conventions
   - Quality profiles
4. Add TV series to monitor

#### Radarr (Movies) - http://localhost:7878
1. Add download client (qBittorrent/SABnzbd)
2. Add indexers from Prowlarr
3. Configure media management:
   - Root folder: `/media/movies`
   - File naming conventions
   - Quality profiles
4. Add movies to monitor

#### Lidarr (Music) - http://localhost:8686
1. Add download client
2. Configure indexers
3. Set root folder: `/media/music`
4. Configure quality and metadata profiles

#### Bazarr (Subtitles) - http://localhost:6767
1. Connect to Sonarr and Radarr
2. Add subtitle providers
3. Configure languages and quality
4. Set up automatic subtitle downloads

### 5. 🎯 Configure Overseerr
**URL:** http://localhost:5055

**Setup Steps:**
1. Connect to Jellyfin server
2. Connect to Sonarr and Radarr
3. Configure user permissions
4. Set up notification services
5. Customize request workflows

---

## 🔗 SERVICE INTEGRATION

### API Key Exchange
Most services need to communicate via API keys:

1. **Get API Keys from each service:**
   - Sonarr: Settings > General > API Key
   - Radarr: Settings > General > API Key
   - Prowlarr: Settings > General > API Key
   - Jellyfin: Dashboard > API Keys

2. **Add API Keys to connecting services:**
   - Prowlarr → Add to Sonarr/Radarr
   - Overseerr → Add Jellyfin/Sonarr/Radarr APIs
   - Bazarr → Add Sonarr/Radarr APIs

### Recommended Connection Order:
1. Configure download clients first
2. Set up Prowlarr with indexers
3. Connect Prowlarr to Sonarr/Radarr
4. Configure Jellyfin libraries
5. Connect Overseerr to all services
6. Set up Bazarr last

---

## 📊 MONITORING SETUP

### Grafana Dashboard
**URL:** http://localhost:3000  
**Login:** admin/changeme

**Setup Tasks:**
1. Change default password
2. Add Prometheus as data source (http://prometheus:9090)
3. Import media server dashboards
4. Set up alerting rules
5. Configure notification channels

### Tautulli Analytics
**URL:** http://localhost:8181

**Setup Tasks:**
1. Connect to Jellyfin server
2. Configure user tracking
3. Set up notification agents
4. Create custom dashboards

---

## 🏠 HOMEPAGE DASHBOARD

**URL:** http://localhost:3001

**Configuration:**
1. Navigate to http://localhost:3001
2. Configure widgets for each service
3. Add bookmarks and weather widgets
4. Customize layout and themes
5. Set up service health monitoring

---

## 🔐 SECURITY HARDENING

### Optional Security Enhancements:
1. **Set up Authelia** (currently available but not configured)
2. **Configure SSL certificates** with Let's Encrypt
3. **Set up VPN** for download clients
4. **Enable fail2ban** for service protection
5. **Configure regular backups**

---

## 📱 MOBILE ACCESS

### Recommended Apps:
- **Jellyfin:** Official Jellyfin mobile app
- **Overseerr:** Web interface is mobile-friendly
- **Portainer:** Official mobile app available
- **Remote Management:** Use VPN or reverse proxy

---

## 🔧 MAINTENANCE TASKS

### Weekly:
- Check service logs for errors
- Monitor disk space usage
- Review download client activity
- Update container images (if desired)

### Monthly:
- Review Grafana metrics
- Clean up old downloads
- Backup configuration files
- Update indexer credentials

---

## 🆘 TROUBLESHOOTING

### Common Issues:

**Service won't start:**
```bash
docker logs [service_name]
docker restart [service_name]
```

**API connection issues:**
- Verify API keys are correct
- Check service URLs (use container names, not localhost)
- Ensure services are on same network

**Download issues:**
- Check indexer status in Prowlarr
- Verify download client configuration
- Check available disk space

**Media not appearing:**
- Force library scan in Jellyfin
- Check file permissions (PUID/PGID)
- Verify media paths are correct

---

## 📞 GETTING HELP

### Documentation:
- **Jellyfin:** https://jellyfin.org/docs/
- **Sonarr:** https://wiki.servarr.com/sonarr
- **Radarr:** https://wiki.servarr.com/radarr
- **Prowlarr:** https://wiki.servarr.com/prowlarr
- **Overseerr:** https://docs.overseerr.dev/

### Community Support:
- Reddit: r/jellyfin, r/sonarr, r/radarr
- Discord servers for each application
- GitHub issues for bug reports

---

**🎉 Your media server is ready! Start with Jellyfin setup and work your way through the automation stack.**