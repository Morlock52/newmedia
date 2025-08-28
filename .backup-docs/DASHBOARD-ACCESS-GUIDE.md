# 🎬 BEGINNER'S GUIDE: Access Your Media Server Dashboard

**Last Updated: July 2025** | **Difficulty: Beginner** | **Time: 2 minutes**

---

## 🚀 QUICK START (For Absolute Beginners)

### Step 1: Open Your Dashboard
**Choose ONE of these methods:**

#### ✅ Method A: One-Click Launcher (RECOMMENDED)
```bash
./START-DASHBOARD.sh
```
**What it does:** Automatically finds and opens the best available dashboard in your browser.

#### ✅ Method B: Direct Browser Access
Open your web browser and go to: **http://localhost:3001**

#### ✅ Method C: Double-Click HTML File
Find and double-click: `smart-dashboard.html` in your project folder

---

## 🔍 What You'll See

### Homepage Dashboard (Best Option)
- **URL:** http://localhost:3001
- **Features:** Real-time monitoring, professional widgets, auto-refresh
- **Best for:** Daily use, system monitoring

### Smart Dashboard (Backup Option)  
- **File:** `smart-dashboard.html`
- **Features:** Health checking, mobile-friendly, works offline
- **Best for:** When Homepage service is down

---

## 🛠 Troubleshooting

### Problem: "Can't connect" or "Page not found"
**Solution:**
1. **Check if services are running:**
   ```bash
   docker ps
   ```
   Look for: `homepage`, `jellyfin`, `qbittorrent`

2. **Start services if needed:**
   ```bash
   docker-compose up -d
   ```

3. **Wait 2-5 minutes** for services to fully start

4. **Try again:** Run `./START-DASHBOARD.sh`

### Problem: "Some services show as offline"
**This is normal!** Services take time to start. Wait 2-3 minutes and refresh.

### Problem: "Script won't run"
**Make it executable:**
```bash
chmod +x START-DASHBOARD.sh
```

---

## 📱 Mobile Access

### On Your Phone/Tablet:
1. **Connect to same WiFi** as your server
2. **Open browser** and go to: `http://[YOUR-SERVER-IP]:3001`
3. **Find your server IP:**
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

---

## 🎯 Quick Service Access

### Main Entertainment
- **🎬 Jellyfin (Movies/TV):** http://localhost:8096
- **📚 AudioBookshelf (Audiobooks):** http://localhost:13378  
- **🎵 Navidrome (Music):** http://localhost:4533
- **📸 Immich (Photos):** http://localhost:2283

### Downloads & Management
- **📥 qBittorrent (Downloads):** http://localhost:8080
- **🎭 Radarr (Movies):** http://localhost:7878
- **📺 Sonarr (TV Shows):** http://localhost:8989
- **🔍 Prowlarr (Search):** http://localhost:9696

### System Management
- **🐳 Portainer (Docker):** http://localhost:9000
- **📊 Grafana (Monitoring):** http://localhost:3000
- **🏠 Homepage (Dashboard):** http://localhost:3001

---

## 🔐 Login Information

### Default Credentials
- **Username:** `admin` (for most services)
- **Passwords:** Check the `.generated_passwords.txt` file in your project folder

### First-Time Setup
1. **Jellyfin:** Create your admin account on first visit
2. **qBittorrent:** Default username is `admin`, check password file
3. **Others:** Most services will guide you through setup

---

## 💡 Pro Tips

### Bookmark Your Dashboard
Add http://localhost:3001 to your browser bookmarks for instant access

### Desktop Shortcut
Create a desktop shortcut to `START-DASHBOARD.sh` for one-click access

### Mobile Bookmark
Bookmark `http://[your-ip]:3001` on your phone for remote access

### Health Monitoring
The smart dashboard shows real-time service health - green means working!

---

## 🆘 Still Need Help?

### Check These Files:
- `README.md` - Full project documentation
- `WORKING_STATUS.md` - Current service status
- `.generated_passwords.txt` - Login credentials

### Common Solutions:
1. **Restart everything:** `docker-compose down && docker-compose up -d`
2. **Check logs:** `docker logs homepage`
3. **Verify Docker:** `docker --version`

### Emergency Access:
If nothing works, you can always access services directly:
- Main dashboard: http://localhost:3001
- Jellyfin: http://localhost:8096
- Downloads: http://localhost:8080

---

## 🎉 You're All Set!

Your media server dashboard should now be accessible and working perfectly. Enjoy your personal entertainment center!

**Questions?** Check the troubleshooting section above or look for additional documentation in your project folder.