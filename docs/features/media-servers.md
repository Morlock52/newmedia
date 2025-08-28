# 🎬 Media Servers - Complete Setup Guide

This guide covers the three media server options included in the Ultimate Media Server Stack: Jellyfin (primary), Plex (premium), and Emby (alternative).

## 🎯 Overview

Media servers are the heart of your home entertainment system. They organize, transcode, and stream your media collection to all your devices.

### Quick Comparison

| Feature | Jellyfin | Plex | Emby |
|---------|----------|------|------|
| **Cost** | Free & Open Source | Free + Premium | Free + Premium |
| **Transcoding** | ✅ Hardware & Software | ✅ Hardware & Software | ✅ Hardware & Software |
| **Mobile Apps** | ✅ Free | ✅ Premium only | ✅ Premium only |
| **Live TV** | ✅ Free | ✅ Premium only | ✅ Premium only |
| **4K Support** | ✅ Full | ✅ Premium only | ✅ Premium only |
| **Privacy** | ✅ Fully self-hosted | ⚠️ Requires account | ⚠️ Requires account |
| **Setup Difficulty** | Medium | Easy | Medium |

## 🚀 Jellyfin (Primary Recommendation)

Jellyfin is our primary recommendation as it's completely free, open source, and fully self-hosted.

### Key Features
- **100% Free**: No premium features or subscriptions
- **No Telemetry**: Complete privacy and control
- **Hardware Acceleration**: Intel QSV, NVIDIA NVENC, AMD VCE
- **Live TV & DVR**: Full IPTV and antenna support
- **Extensive Plugin System**: Anime, music, and metadata plugins

### Setup Instructions

#### 1. Access Jellyfin
```bash
# Jellyfin runs on port 8096
http://localhost:8096
```

#### 2. Initial Setup Wizard
1. **Create Admin Account**: Set your username and password
2. **Add Media Libraries**: 
   - Movies: `/data/media/movies`
   - TV Shows: `/data/media/tv`
   - Music: `/data/media/music`
3. **Configure Metadata**: Enable automatic metadata downloading
4. **Remote Access**: Configure for external access if needed

#### 3. Hardware Acceleration Setup
For GPU transcoding (Intel/AMD/NVIDIA):

```yaml
# In docker-compose.yml (already configured)
jellyfin:
  devices:
    - /dev/dri:/dev/dri  # Intel QSV
  # For NVIDIA (uncomment if you have NVIDIA GPU)
  # runtime: nvidia
  # environment:
  #   - NVIDIA_VISIBLE_DEVICES=all
```

#### 4. Essential Settings
Navigate to **Administration > Dashboard > Playback**:

- **Hardware acceleration**: Intel QSV (or your GPU type)
- **Enable hardware decoding**: All supported formats
- **Allow encoding in HEVC format**: Enabled
- **Throttle transcodes**: Enabled

#### 5. Recommended Plugins
Navigate to **Administration > Dashboard > Plugins > Catalog**:

- **Anime**: Enhanced anime metadata
- **AudioDB**: Music metadata
- **Open Subtitles**: Subtitle provider
- **Trakt**: Sync watch status
- **TMDb**: Enhanced movie metadata

### Mobile Apps
- **Official Apps**: Available for Android, iOS, Android TV, etc.
- **Third-party**: Swiftfin (iOS), Findroid (Android)
- **Web Interface**: Works on all devices

## 🎭 Plex (Premium Option)

Plex offers the most polished experience with excellent mobile apps and features.

### Key Features
- **Polished Interface**: Best-in-class user experience
- **Excellent Mobile Apps**: Industry-leading mobile experience
- **Plex Pass Features**: Hardware transcoding, mobile sync, live TV
- **Discovery**: Find new content across services
- **Sharing**: Easy sharing with friends and family

### Setup Instructions

#### 1. Access Plex
```bash
# Plex runs on port 32400
http://localhost:32400/web
```

#### 2. Initial Setup
1. **Sign in to Plex**: Create/use existing Plex account
2. **Claim Server**: Use the claim token from plex.tv/claim
3. **Add Libraries**:
   - Movies: `/data/media/movies`
   - TV Shows: `/data/media/tv`
   - Music: `/data/media/music`

#### 3. Configure Transcoding
Navigate to **Settings > Transcoder**:

- **Transcoder quality**: Automatic
- **Background transcoding**: Enabled (Plex Pass)
- **Use hardware acceleration**: Enabled (Plex Pass)

#### 4. Remote Access Setup
Navigate to **Settings > Remote Access**:

1. **Enable Remote Access**: Toggle on
2. **Manual Port**: 32400 (if automatic fails)
3. **Test Connection**: Ensure it's accessible

#### 5. Plex Pass Features
Consider Plex Pass for:
- **Mobile Apps**: Download and streaming
- **Hardware Transcoding**: GPU acceleration
- **Live TV**: Antenna and IPTV support
- **Skip Intros**: Automatic intro detection

### Advanced Configuration

#### GPU Transcoding (Plex Pass Required)
```yaml
# For Intel GPU
devices:
  - /dev/dri:/dev/dri

# For NVIDIA GPU (uncomment in docker-compose.yml)
# runtime: nvidia
# environment:
#   - NVIDIA_VISIBLE_DEVICES=all
```

## 📺 Emby (Alternative Option)

Emby provides a middle ground between Jellyfin and Plex.

### Key Features
- **Freemium Model**: Basic features free, premium features paid
- **Good Performance**: Efficient transcoding and streaming
- **Live TV Support**: Built-in TV tuner support
- **Plugins**: Extensible with plugins
- **Theater Mode**: Big screen interface

### Setup Instructions

#### 1. Access Emby
```bash
# Emby runs on port 8097 (to avoid conflicts)
http://localhost:8097
```

#### 2. Initial Setup
1. **Create Admin Account**: Set administrator credentials
2. **Add Media Libraries**:
   - Movies: `/data/media/movies`
   - TV Shows: `/data/media/tv`
   - Music: `/data/media/music`
3. **Configure Metadata**: Select preferred metadata sources

#### 3. Emby Premiere Features
Consider Emby Premiere for:
- **Mobile Apps**: iOS and Android apps
- **Cloud Sync**: Sync for offline viewing
- **Hardware Transcoding**: GPU acceleration
- **Cover Art**: Enhanced artwork and themes

## 🔧 Common Configuration

### Media Organization
Ensure your media follows this structure:
```
/data/media/
├── movies/
│   ├── Movie Name (2023)/
│   │   └── Movie Name (2023).mkv
│   └── Another Movie (2022)/
│       └── Another Movie (2022).mp4
├── tv/
│   ├── TV Show Name/
│   │   ├── Season 01/
│   │   │   ├── S01E01.mkv
│   │   │   └── S01E02.mkv
│   │   └── Season 02/
│   │       ├── S02E01.mkv
│   │       └── S02E02.mkv
└── music/
    ├── Artist Name/
    │   └── Album Name (2023)/
    │       ├── 01 - Track Name.flac
    │       └── 02 - Another Track.flac
```

### Hardware Acceleration Support

#### Intel Graphics (Recommended)
```yaml
devices:
  - /dev/dri:/dev/dri
```

#### NVIDIA Graphics
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

#### AMD Graphics
```yaml
devices:
  - /dev/dri:/dev/dri
```

### Network Configuration

All media servers are configured with:
- **Reverse Proxy**: Access via Nginx Proxy Manager
- **Health Checks**: Automatic service monitoring
- **Docker Networks**: Isolated media network

## 📱 Client Applications

### Jellyfin Clients
- **Web**: Built-in web interface
- **Android**: Jellyfin for Android
- **iOS**: Swiftfin (third-party, recommended)
- **Android TV**: Jellyfin for Android TV
- **Apple TV**: Swiftfin for tvOS
- **Roku**: Official Jellyfin channel
- **Desktop**: Jellyfin Media Player

### Plex Clients
- **Web**: plex.tv/web
- **Mobile**: Official iOS/Android apps (Plex Pass required)
- **TV**: Apple TV, Android TV, Roku, etc.
- **Desktop**: Plex Media Player
- **Gaming**: PlayStation, Xbox apps

### Emby Clients
- **Web**: Built-in web interface
- **Mobile**: Official iOS/Android apps (Premiere required)
- **TV**: Apple TV, Android TV, Roku
- **Desktop**: Emby Theater

## 🔒 Security & Access

### SSL/HTTPS Setup
Use Nginx Proxy Manager to add SSL certificates:

1. **Create Proxy Host**: Point to your media server
2. **Add SSL Certificate**: Use Let's Encrypt
3. **Configure Authentication**: Optional password protection

### Remote Access
For secure remote access:

1. **VPN**: Recommended for security (WireGuard/OpenVPN)
2. **Reverse Proxy**: Nginx Proxy Manager with SSL
3. **Dynamic DNS**: For changing IP addresses
4. **Port Forwarding**: Only if VPN isn't possible

## 📊 Performance Optimization

### Transcoding Settings
- **Use Hardware Acceleration**: Reduces CPU usage by 80%+
- **Optimize for Bandwidth**: Adjust quality based on connection
- **Pre-transcode**: Convert files in advance during off-hours

### Storage Optimization
- **SSD for OS**: Fast boot and app loading
- **HDD for Media**: Large, cost-effective storage
- **Cache Drive**: SSD cache for frequently accessed content

### Network Optimization
- **Wired Connections**: Ethernet for high-bitrate content
- **Quality of Service**: Prioritize media server traffic
- **Local Subnets**: Keep traffic local when possible

## 🔧 Troubleshooting

### Common Issues

#### Transcoding Problems
- **Check GPU drivers**: Ensure latest drivers installed
- **Verify hardware support**: Not all GPUs support all codecs
- **Monitor resources**: Check CPU/GPU usage during playback

#### Network Issues
- **Port conflicts**: Ensure ports aren't already in use
- **Firewall rules**: Check both host and container firewalls
- **DNS resolution**: Verify hostname resolution

#### Library Scanning
- **Permissions**: Ensure proper file permissions (PUID/PGID)
- **File naming**: Follow proper naming conventions
- **Metadata sources**: Check internet connectivity for metadata

### Performance Issues
- **Database maintenance**: Regular database cleanup
- **Log rotation**: Prevent logs from filling disk space
- **Resource monitoring**: Use included monitoring stack

## 🎯 Best Practices

1. **Start with Jellyfin**: Free and fully featured
2. **Organize Media Properly**: Follow naming conventions
3. **Use Hardware Acceleration**: Dramatically improves performance
4. **Regular Backups**: Backup configurations and databases
5. **Monitor Resources**: Use Grafana dashboards
6. **Secure Access**: Use VPN or proper SSL setup
7. **Update Regularly**: Keep servers and apps updated

## 🔗 Integration with Other Services

### Download Integration
- **Sonarr/Radarr**: Automatic episode/movie organization
- **Prowlarr**: Centralized indexer management
- **qBittorrent**: Automatic download handling

### Request Systems
- **Jellyseerr**: For Jellyfin users
- **Overseerr**: For Plex users
- **Ombi**: Universal request system

### Monitoring Integration
- **Prometheus**: Metrics collection
- **Grafana**: Performance dashboards
- **Uptime Kuma**: Service monitoring

---

## 📚 Additional Resources

- [Jellyfin Documentation](https://jellyfin.org/docs/)
- [Plex Support](https://support.plex.tv/)
- [Emby Documentation](https://emby.media/support.html)
- [Hardware Transcoding Guide](../operations/performance.md#hardware-acceleration)
- [Remote Access Setup](../operations/security.md#remote-access)

**Next Steps**: Set up [Download Management](download-management.md) to automatically add content to your media servers.