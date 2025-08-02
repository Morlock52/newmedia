# Omega Media Server - Quick Start Guide

## 🚀 One-Line Installation

### Option 1: Pre-built Image (Recommended)
```bash
docker run -d \
  --name omega-media-server \
  -p 80:80 \
  -v ~/omega/config:/config \
  -v ~/omega/media:/media \
  omegaserver/omega:latest
```

### Option 2: Build from Source
```bash
git clone https://github.com/omega-server/omega-media-server.git
cd omega-media-server
docker-compose up -d
```

### Option 3: Simplified Version (For Testing)
```bash
# Uses the simplified Dockerfile for quick testing
docker-compose -f docker-compose-simple.yml up -d
```

## 🎯 First Steps

1. **Access the Web UI**
   - Open http://localhost in your browser
   - You'll see the setup wizard

2. **Complete Setup Wizard**
   - Create admin account
   - Configure media paths
   - Select features to enable
   - Configure AI settings (optional)

3. **Add Media**
   - Copy your media files to the configured directories:
     - Movies: `~/omega/media/movies`
     - TV Shows: `~/omega/media/tv`
     - Music: `~/omega/media/music`
     - Photos: `~/omega/media/photos`
     - Books: `~/omega/media/books`

4. **Access Media Servers**
   - Jellyfin: http://localhost/jellyfin
   - Omega Dashboard: http://localhost
   - API: http://localhost/api

## 📱 Mobile & TV Apps

### Jellyfin Apps
- **iOS**: [App Store](https://apps.apple.com/app/jellyfin/id1480192618)
- **Android**: [Play Store](https://play.google.com/store/apps/details?id=org.jellyfin.mobile)
- **Android TV**: [Play Store](https://play.google.com/store/apps/details?id=org.jellyfin.androidtv)
- **Roku**: Search "Jellyfin" in Channel Store
- **Fire TV**: Search "Jellyfin" in App Store

### Omega Mobile App (Coming Soon)
- Native apps with AI features
- Offline sync support
- Voice control integration

## 🎨 Key Features

### Instant Setup
- Zero configuration required
- Automatic service discovery
- Smart defaults based on your system

### AI-Powered
- Smart recommendations
- Auto-tagging and organization
- Voice search and control
- Automatic subtitle generation

### All-in-One
- 30+ apps pre-installed
- Single sign-on for all services
- Unified search across all media
- Centralized management

### Performance
- Hardware acceleration support
- 8K streaming ready
- Intelligent transcoding
- Predictive caching

## 🔧 Common Tasks

### Enable Hardware Acceleration
```bash
# For Intel GPUs
docker exec omega-media-server omega-config gpu enable intel

# For NVIDIA GPUs
docker exec omega-media-server omega-config gpu enable nvidia
```

### Add Users
1. Go to Settings → Users
2. Click "Add User"
3. Set permissions and media access
4. Share invite link

### Install Additional Apps
1. Go to Apps → App Store
2. Browse available apps
3. Click "Install" on desired apps
4. Apps auto-configure and integrate

### Configure Remote Access
1. Go to Settings → Network
2. Enable "Remote Access"
3. Set up port forwarding or use built-in VPN
4. Access from anywhere!

## 🚨 Troubleshooting

### Cannot Access Web UI
```bash
# Check if container is running
docker ps

# Check logs
docker logs omega-media-server

# Restart container
docker restart omega-media-server
```

### Media Not Showing Up
1. Check file permissions
2. Verify media paths in Settings
3. Trigger manual scan: Settings → Libraries → Scan

### Performance Issues
1. Enable hardware acceleration
2. Adjust transcoding settings
3. Check Settings → Dashboard for bottlenecks

## 📚 Advanced Configuration

### Environment Variables
See `.env.example` for all available options:
- `ENABLE_AI=true` - Enable AI features
- `ENABLE_8K=true` - Enable 8K streaming
- `TRANSCODE_THREADS=0` - Auto-detect CPU cores
- `AI_MODEL=advanced` - AI model selection

### Custom Domain with SSL
```bash
docker run -d \
  --name omega-media-server \
  -p 80:80 -p 443:443 \
  -e DOMAIN=media.yourdomain.com \
  -e ADMIN_EMAIL=you@email.com \
  -v ~/omega/config:/config \
  -v ~/omega/media:/media \
  omegaserver/omega:latest
```

### GPU Passthrough
```yaml
# In docker-compose.yml
devices:
  - /dev/dri:/dev/dri  # Intel
  # OR
  - /dev/nvidia0:/dev/nvidia0  # NVIDIA
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
```

## 🆘 Getting Help

- **Documentation**: [docs.omega-server.com](https://docs.omega-server.com)
- **Discord**: [Join our community](https://discord.gg/omega-media)
- **GitHub Issues**: [Report bugs](https://github.com/omega-server/omega/issues)
- **Reddit**: [r/OmegaMediaServer](https://reddit.com/r/omegamediaserver)

## 🎉 Next Steps

1. **Explore the Dashboard** - Discover AI recommendations
2. **Configure Automation** - Set up auto-downloading
3. **Try Voice Control** - "Play the latest episode of..."
4. **Share with Family** - Create user accounts
5. **Join Community** - Share your setup!

---

Welcome to the future of media servers! 🚀