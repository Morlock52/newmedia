# 🎯 Media Server Dashboard Access Guide

## 🚀 Quick Access

### Main Dashboard (NEW!)
**Open in your browser:** 
```
file:///Users/morlock/fun/newmedia/ultimate-dashboard-live.html
```

Or click: [Open Dashboard](file:///Users/morlock/fun/newmedia/ultimate-dashboard-live.html)

---

## 📊 Available Dashboards

### 1. **Ultimate Dashboard (Recommended)**
- **Location:** `/Users/morlock/fun/newmedia/ultimate-dashboard-live.html`
- **Features:** 
  - Live service status monitoring
  - Auto-refresh every 30 seconds
  - Quick access buttons to all services
  - Visual status indicators
  - System health metrics

### 2. **Uptime Kuma (Service Monitoring)**
- **URL:** http://localhost:3001
- **Purpose:** Professional uptime monitoring
- **Setup:** Create admin account on first visit

### 3. **Portainer (Container Management)**
- **URL:** http://localhost:9000
- **Purpose:** Docker container management
- **Setup:** Create admin account on first visit

---

## 🎬 Direct Service Access

### Media Servers
- **Jellyfin:** http://localhost:8096
  - Default: No authentication (set up admin on first visit)

### Media Management
- **Sonarr:** http://localhost:8989 (TV Shows)
- **Radarr:** http://localhost:7878 (Movies)
- **Prowlarr:** http://localhost:9696 (Indexers)

### Download Clients
- **qBittorrent:** http://localhost:8080
  - Default login: `admin` / `adminadmin`

---

## 🛠️ Dashboard Features

### Ultimate Dashboard Live
The main dashboard provides:

1. **Real-time Status Monitoring**
   - Green = Online ✅
   - Yellow = Starting 🔄
   - Red = Offline ❌

2. **Quick Statistics**
   - Total services count
   - Online services count
   - System health percentage

3. **One-Click Access**
   - Direct links to all services
   - No need to remember ports

4. **Auto-Refresh**
   - Updates every 30 seconds
   - Manual refresh button available

5. **Responsive Design**
   - Works on desktop, tablet, and mobile
   - Modern, clean interface

---

## 🔧 Troubleshooting

### If Dashboard Shows Services as "Starting":
- **Normal behavior** for ARR services (Sonarr, Radarr, Prowlarr)
- First boot takes 2-5 minutes
- They will turn green when ready

### If Dashboard Won't Load:
1. Make sure file path is correct
2. Try opening in different browser
3. Check if JavaScript is enabled

### To Customize Dashboard:
- Edit `/Users/morlock/fun/newmedia/ultimate-dashboard-live.html`
- Add new services to the `services` array in the script section
- Modify colors and styles in the CSS section

---

## 📱 Mobile Access

To access from other devices on your network:

1. Find your Mac's IP address:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

2. Replace `localhost` with your IP address:
   - Example: `http://192.168.1.100:8096` for Jellyfin

3. Make sure firewall allows connections

---

## 🎯 Quick Start Sequence

1. **Open Dashboard:** [ultimate-dashboard-live.html](file:///Users/morlock/fun/newmedia/ultimate-dashboard-live.html)
2. **Set up Jellyfin:** Click "Open Jellyfin" → Create admin account
3. **Configure qBittorrent:** Click "Open qBittorrent" → Change default password
4. **Wait for ARR services:** They'll be ready in 2-5 minutes
5. **Set up monitoring:** Click "Open Monitoring" → Configure Uptime Kuma

---

## 💡 Pro Tips

- **Bookmark the dashboard** for quick access
- **Pin services** you use frequently to browser bookmarks
- **Use Portainer** to restart services if needed
- **Check Uptime Kuma** for detailed service metrics

---

*Dashboard created: August 9, 2025*
*Auto-refresh enabled for live status monitoring*