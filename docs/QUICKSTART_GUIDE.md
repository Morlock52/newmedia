# Ultimate Media Server 2025 - Quick Start Guide

Get your media server running in under 5 minutes! 🚀

---

## 🎯 One-Line Installation

### Linux/macOS
```bash
curl -sSL https://get.mediaserver.com | bash
```

### Windows (PowerShell as Admin)
```powershell
iwr -useb https://get.mediaserver.com/windows | iex
```

---

## 🚀 Manual Quick Start (3 Steps)

### Step 1: Clone & Navigate
```bash
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server
```

### Step 2: Configure (Optional)
```bash
# Use defaults or customize
cp .env.example .env
# Edit .env if needed
```

### Step 3: Deploy
```bash
# Linux/macOS
./deploy-ultimate-2025.sh

# Windows
.\deploy-ultimate-2025.ps1

# Or use Docker Compose directly
docker-compose up -d
```

---

## 📋 Quick Configuration

### Essential Settings Only

Create a `.env` file with just these essentials:

```env
# Timezone (find yours at https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
TZ=America/New_York

# User/Group IDs (Linux/macOS: use 'id' command)
PUID=1000
PGID=1000

# Media Storage Location
MEDIA_PATH=/path/to/your/media
DOWNLOADS_PATH=/path/to/downloads
```

### Default Credentials

| Service | Username | Password | First Login |
|---------|----------|----------|-------------|
| **Jellyfin** | - | - | Create on first visit |
| **Sonarr/Radarr** | - | - | No auth by default |
| **qBittorrent** | admin | adminadmin | Change immediately |
| **Grafana** | admin | admin | Change on first login |
| **Portainer** | - | - | Create on first visit |

---

## 🎬 5-Minute Setup Workflow

### 1️⃣ Access Main Dashboard (30 seconds)
```
http://localhost:3001
```
All your services in one place!

### 2️⃣ Configure Jellyfin (2 minutes)
1. Visit: http://localhost:8096
2. Create admin account
3. Add media folders:
   - Movies → `/media/movies`
   - TV Shows → `/media/tv`
   - Music → `/media/music`
4. Skip metadata providers (configure later)
5. Finish setup

### 3️⃣ Setup Download Client (1 minute)
1. Visit qBittorrent: http://localhost:8080
2. Login: `admin` / `adminadmin`
3. Go to Settings → Web UI → Change password
4. Settings → Downloads → Default Save Path: `/downloads`

### 4️⃣ Connect the Apps (1.5 minutes)

**Quick Prowlarr Setup:**
1. Visit: http://localhost:9696
2. Add an indexer (e.g., 1337x for testing)
3. Settings → Apps → Add:
   - Sonarr: `http://sonarr:8989`
   - Radarr: `http://radarr:7878`

**Quick Sonarr Setup:**
1. Visit: http://localhost:8989
2. Settings → Download Clients → Add → qBittorrent
3. Host: `qbittorrent`, Port: `8080`
4. Add root folder: `/media/tv`

**Quick Radarr Setup:**
1. Visit: http://localhost:7878
2. Same as Sonarr but use `/media/movies`

---

## 🔥 Super Quick Commands

### Service Management
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart a service
docker-compose restart jellyfin

# View logs
docker-compose logs -f jellyfin

# Update all services
docker-compose pull && docker-compose up -d
```

### Quick Health Check
```bash
# See what's running
docker ps

# Check service health
curl http://localhost:3011  # Uptime Kuma
```

---

## 📱 Mobile Access

### Local Network
- Find your server IP: `ip addr` (Linux) or `ipconfig` (Windows)
- Access services at: `http://YOUR-IP:PORT`

### Remote Access (Quick & Secure)
```bash
# Using Tailscale (recommended)
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up

# Now access from anywhere using Tailscale IP
```

---

## 🎯 Quick Wins

### Add Your First Media

1. **Copy media to folders:**
   ```bash
   cp -r /your/movies/* ./media-data/movies/
   cp -r /your/shows/* ./media-data/tv/
   ```

2. **Scan in Jellyfin:**
   - Dashboard → Libraries → Scan All Libraries

### Request Your First Movie

1. Visit Overseerr: http://localhost:5056
2. Sign in with Plex/Jellyfin
3. Search for a movie
4. Click "Request"
5. Watch it download automatically!

### Monitor Everything

- **Service Status**: http://localhost:3011 (Uptime Kuma)
- **System Metrics**: http://localhost:3000 (Grafana)
- **Container Stats**: https://localhost:9443 (Portainer)

---

## 🚨 Quick Troubleshooting

### Nothing is starting?
```bash
# Check Docker is running
docker version

# Check for port conflicts
sudo lsof -i :8096  # Example for Jellyfin
```

### Can't access services?
```bash
# Check firewall
sudo ufw status  # Ubuntu
sudo firewall-cmd --list-all  # RHEL/CentOS

# Restart Docker
sudo systemctl restart docker
```

### Out of space?
```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a
```

---

## 📚 Quick Links

### Service URLs
- **Main Dashboard**: http://localhost:3001
- **Media Server**: http://localhost:8096
- **Movie Downloads**: http://localhost:7878
- **TV Downloads**: http://localhost:8989
- **Requests**: http://localhost:5056
- **Monitoring**: http://localhost:3011

### Mobile Apps
- **Jellyfin**: [iOS](https://apps.apple.com/app/jellyfin-mobile/id1480192618) | [Android](https://play.google.com/store/apps/details?id=org.jellyfin.mobile)
- **Overseerr**: Use web browser
- **qBittorrent**: Web UI works great on mobile

---

## 🎉 You're Done!

Your media server is now running! Here's what to do next:

1. **Add Media**: Copy your files to the media folders
2. **Configure Quality**: Set up profiles in Sonarr/Radarr
3. **Add Users**: Create accounts for family/friends
4. **Setup Requests**: Configure Overseerr for easy requests
5. **Enjoy**: Start streaming!

Need more help? Check the [full installation guide](./INSTALLATION_GUIDE.md) or [troubleshooting guide](./TROUBLESHOOTING_GUIDE.md).

---

## 🆘 Quick Support

- **Discord**: [Join our community](https://discord.gg/mediaserver)
- **GitHub Issues**: [Report problems](https://github.com/yourusername/ultimate-media-server/issues)
- **Reddit**: [r/selfhosted](https://reddit.com/r/selfhosted)

---

*Happy Streaming! 🍿*