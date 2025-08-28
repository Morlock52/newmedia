# 🎬 How to Run Ultimate Media Server 2025

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated Installation (Easiest)
```bash
# One-line installer
curl -sSL https://raw.githubusercontent.com/yourusername/ultimate-media-server/main/install-media-server.sh | bash
```

### Option 2: Manual Quick Start
```bash
# 1. Clone the project
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server

# 2. Copy environment template
cp .env.example .env

# 3. Start everything
docker compose up -d

# 4. Access dashboard
open http://localhost:3001
```

---

## 🎯 What You're Running

This is a **complete media server and seedbox** that:

1. **Automatically downloads** movies/TV shows you request
2. **Organizes everything** with proper naming and metadata  
3. **Streams to any device** (phones, TVs, tablets)
4. **Manages requests** from family/friends
5. **Downloads safely** through VPN

### The Flow:
```
You want a movie → Request it in Overseerr → System finds it → 
Downloads via VPN → Organizes it → Ready to stream in Jellyfin
```

---

## 📋 Detailed Setup Instructions

### Step 1: System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 100GB SSD + 1TB media storage
- **OS**: Ubuntu 22.04, Debian 12, or any Linux with Docker
- **Network**: Stable internet, ideally static IP

### Step 2: Install Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### Step 3: Get the Project
```bash
# Clone repository
git clone https://github.com/yourusername/ultimate-media-server.git
cd ultimate-media-server

# Create required directories
mkdir -p config media/{movies,tv,music} downloads
```

### Step 4: Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit with your settings
nano .env

# Key settings to change:
# PUID=1000              # Your user ID (run: id -u)
# PGID=1000              # Your group ID (run: id -g)  
# TZ=America/New_York    # Your timezone
# DOMAIN=localhost       # Your domain or IP
# VPN_USER=              # Your VPN username
# VPN_PASS=              # Your VPN password
```

### Step 5: Choose Your Setup

#### 🟢 Beginner Setup (Just Streaming)
```bash
# Minimal setup - Jellyfin + basic features
docker compose -f docker-compose-minimal.yml up -d
```

#### 🟡 Standard Setup (Recommended)
```bash
# Full automation - All *arr apps + downloads
docker compose up -d
```

#### 🔴 Advanced Setup (Everything)
```bash
# All 60+ services including experimental features
docker compose -f docker-compose-full.yml up -d
```

---

## 🖥️ Accessing Your Services

Once running, access these URLs:

| Service | URL | What It Does |
|---------|-----|--------------|
| **Dashboard** | http://localhost:3001 | Main control panel |
| **Jellyfin** | http://localhost:8096 | Watch your media |
| **Overseerr** | http://localhost:5055 | Request new content |
| **Sonarr** | http://localhost:8989 | TV show automation |
| **Radarr** | http://localhost:7878 | Movie automation |

---

## 🔧 First-Time Configuration

### 1. Set Up Jellyfin (Media Player)
1. Go to http://localhost:8096
2. Create admin account
3. Add media folders:
   - Movies: `/media/movies`
   - TV: `/media/tv`
4. Finish setup wizard

### 2. Configure Download Automation
1. **Prowlarr** (http://localhost:9696)
   - Add an indexer (like 1337x)
   - Add apps (Sonarr, Radarr)

2. **Sonarr** (http://localhost:8989)
   - Settings → Download Clients → Add → qBittorrent
   - Host: `qbittorrent`, Port: `8080`

3. **qBittorrent** (http://localhost:8080)
   - Login: `admin` / `adminadmin` (CHANGE THIS!)
   - Set download location

### 3. Enable Requests
1. **Overseerr** (http://localhost:5055)
   - Sign in with Jellyfin
   - Connect Sonarr/Radarr
   - Invite users

---

## 🎮 Using Your Media Server

### To Watch Something:
1. Open Jellyfin on any device
2. Browse your library
3. Click play!

### To Add New Content:
1. Open Overseerr
2. Search for movie/show
3. Click "Request"
4. Wait 5-30 minutes
5. It appears in Jellyfin!

---

## 🚨 Important Security Steps

### Change Default Passwords!
```bash
# qBittorrent
# Go to http://localhost:8080
# Options → Web UI → Change password

# Grafana (if using monitoring)
# Go to http://localhost:3000
# Change admin password on first login
```

### Enable VPN (Required for Downloads)
```bash
# Edit .env file
VPN_PROVIDER=nordvpn    # or your provider
VPN_USER=your-username
VPN_PASS=your-password

# Restart download container
docker compose restart gluetun qbittorrent
```

---

## 📱 Mobile & Remote Access

### Option 1: Local Network Only
- Find your server IP: `ip addr | grep inet`
- Access via: `http://YOUR-IP:8096`

### Option 2: Secure Remote Access
```bash
# Install Tailscale (recommended)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Now access from anywhere using Tailscale IP
```

---

## 🛠️ Troubleshooting

### Nothing Starting?
```bash
# Check what's running
docker ps

# View logs
docker compose logs -f

# Restart everything
docker compose down
docker compose up -d
```

### Can't Access Web UI?
```bash
# Check if service is running
docker ps | grep jellyfin

# Check firewall
sudo ufw allow 8096

# Test locally
curl http://localhost:8096
```

### Downloads Not Working?
```bash
# Check VPN is connected
docker exec gluetun curl -s https://ipinfo.io

# Should show VPN location, not your real location
```

---

## 📊 Monitoring Your System

### Check Service Health
```bash
# Run health check script
./scripts/health-check.sh

# Or check manually
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f jellyfin
```

---

## 🔄 Maintenance

### Update Services
```bash
# Pull latest images
docker compose pull

# Restart with updates
docker compose up -d
```

### Backup Configuration
```bash
# Run backup script
./scripts/backup.sh

# Or manually
tar -czf backup-$(date +%Y%m%d).tar.gz config/
```

---

## 🎉 You're Done!

Your media server is now running! Here's what to do next:

1. **Add some media** to test
2. **Configure quality** settings in Radarr/Sonarr
3. **Invite family/friends** via Overseerr
4. **Install mobile apps** (Jellyfin has apps for all platforms)
5. **Enjoy** your personal Netflix!

---

## 📚 More Help

- **Full Guide**: [ULTIMATE_DEPLOYMENT_GUIDE_2025.md](ULTIMATE_DEPLOYMENT_GUIDE_2025.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING_GUIDE.md](docs/TROUBLESHOOTING_GUIDE.md)
- **Security**: [docs/SECURITY_HARDENING.md](docs/SECURITY_HARDENING.md)

---

*Remember: This is YOUR media server. Start simple, add features as you learn!*