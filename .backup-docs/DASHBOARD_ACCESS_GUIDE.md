# 🎬 Media Server Dashboard Access Guide 2025

## Quick Start - 3 Easy Ways to Access Your Dashboards

### 🚀 Method 1: Simple Dashboard Server (RECOMMENDED FOR BEGINNERS)

**This is the easiest way to access your media services!**

1. **Start the Dashboard Server:**
   ```bash
   cd /Users/morlock/fun/newmedia
   python3 dashboard-server.py
   ```

2. **Open Your Browser:**
   - Main Dashboard: http://127.0.0.1:8888
   - Service Status: http://127.0.0.1:8888/status

3. **What You Get:**
   - ✅ Beautiful, modern interface
   - ✅ Live service status checking
   - ✅ Direct links to all services
   - ✅ API endpoints for advanced users
   - ✅ Works on any device on your network

---

### 🏠 Method 2: Homepage Dashboard (PROFESSIONAL)

**Advanced dashboard with widgets and real-time monitoring**

1. **Access Homepage:**
   - URL: http://localhost:3001
   - Features: Service widgets, real-time stats, Docker integration

2. **Homepage Benefits:**
   - Real-time service monitoring
   - API integration with all services
   - Customizable layout and themes
   - Docker container status

---

### 📝 Method 3: Direct Service Access (SIMPLE)

**Direct links when you know exactly what you want**

| Service | URL | Purpose |
|---------|-----|---------|
| 🎬 **Jellyfin** | http://localhost:8096 | Stream movies, TV shows, music |
| 📚 **AudioBookshelf** | http://localhost:13378 | Audiobooks and podcasts |
| 🎵 **Navidrome** | http://localhost:4533 | Music streaming |
| 📸 **Immich** | http://localhost:2283 | Photo management |
| 📥 **qBittorrent** | http://localhost:8080 | Download torrents |
| 📰 **SABnzbd** | http://localhost:8081 | Download from Usenet |
| 🎭 **Radarr** | http://localhost:7878 | Manage movies |
| 📺 **Sonarr** | http://localhost:8989 | Manage TV shows |
| 🔍 **Prowlarr** | http://localhost:9696 | Search indexers |
| 📊 **Grafana** | http://localhost:3000 | System monitoring |
| 🐳 **Portainer** | http://localhost:9000 | Docker management |

---

## 🔧 Troubleshooting Dashboard Issues

### Problem: "Homepage is unhealthy"
**Solution:** Restart the Homepage service
```bash
docker restart homepage
```

### Problem: "Can't access static HTML dashboards"
**Solution:** Use the dashboard server
```bash
cd /Users/morlock/fun/newmedia
python3 dashboard-server.py
```

### Problem: "Service shows as down but is actually running"
**Solution:** Check the actual service directly:
```bash
docker ps | grep service-name
curl http://localhost:PORT
```

### Problem: "Widget not loading in Homepage"
**Cause:** API keys may be missing or incorrect
**Solution:** Check service configuration or disable widgets temporarily

---

## 🌟 Best Practices for Beginners

### 1. **Start with Direct Links**
- Use direct service URLs first to ensure everything works
- Bookmark your most-used services
- Test each service individually before using dashboards

### 2. **Use the Simple Dashboard Server**
- Run `python3 dashboard-server.py` when you want a clean overview
- Access via http://127.0.0.1:8888 for the best experience
- Check service status at http://127.0.0.1:8888/status

### 3. **Homepage for Advanced Users**
- Configure API keys for full widget functionality
- Customize the layout in `/Users/morlock/fun/newmedia/homepage-config/`
- Use for monitoring multiple services at once

### 4. **Bookmark These URLs**
```
Main Media:     http://localhost:8096  (Jellyfin)
Downloads:      http://localhost:8080  (qBittorrent)
Management:     http://localhost:9000  (Portainer)
Dashboard:      http://127.0.0.1:8888  (Simple Server)
```

---

## 🔒 Security Notes

- **All services run on localhost only** - not accessible from internet
- **Default credentials** are documented in your deployment scripts
- **Change default passwords** after initial setup
- **Use strong passwords** for any externally exposed services

---

## 📱 Mobile Access

### From Same Network:
Replace `localhost` with your computer's IP address:
- Find IP: `ipconfig getifaddr en0` (macOS)
- Example: http://192.168.1.100:8096

### Dashboard Server Mobile Access:
- Start server: `python3 dashboard-server.py`
- Access from phone: http://YOUR-IP:8888

---

## 🆘 Emergency Access

If all dashboards fail, services are still accessible:

1. **Check what's running:**
   ```bash
   docker ps
   ```

2. **Access key services directly:**
   - Jellyfin: http://localhost:8096
   - Portainer: http://localhost:9000
   - qBittorrent: http://localhost:8080

3. **Restart all services:**
   ```bash
   cd /Users/morlock/fun/newmedia
   docker-compose -f docker-compose-optimized.yml restart
   ```

---

## 🎯 Quick Commands Reference

```bash
# Start dashboard server
python3 /Users/morlock/fun/newmedia/dashboard-server.py

# Check service status
docker ps

# Restart Homepage
docker restart homepage

# View service logs
docker logs service-name

# Stop all services
docker-compose -f docker-compose-optimized.yml down

# Start all services
docker-compose -f docker-compose-optimized.yml up -d
```

---

## 🎉 Success! You're All Set

Your media server stack is running with multiple dashboard options:

1. **For Beginners:** Use the Simple Dashboard Server
2. **For Power Users:** Use Homepage Dashboard  
3. **For Direct Access:** Use individual service URLs

Choose the method that works best for you, and enjoy your media server!