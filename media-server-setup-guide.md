# Media Server Auto-Configuration Guide

## Current Status

### 🟢 Jellyfin Status
- **URL**: http://localhost:8096
- **Container**: Running and healthy
- **Setup Required**: Initial wizard needs completion
- **Status Code**: 302 (Redirect to setup wizard)

### 🟡 Plex Status  
- **URL**: http://localhost:32400
- **Container**: Running
- **Setup Required**: Authentication required
- **Status Code**: 401 (Unauthorized - needs initial setup)

### 🟢 Media Storage
- **Mount Point**: /Volumes/Plex (21TB total, 3.3TB available)
- **Movies**: ✅ Accessible with content
- **TV Shows**: ✅ Accessible with content  
- **Music**: ✅ Accessible with content

## Container Status
```
JELLYFIN: Up About an hour (healthy)
PLEX:     Up About an hour
```

## Step-by-Step Setup Instructions

### 1. Complete Jellyfin Initial Setup

#### Access Jellyfin Web Interface
1. Open: http://localhost:8096
2. You'll be redirected to the setup wizard

#### Initial Configuration
1. **Language & Display**: Select your preferred language
2. **Create Admin User**:
   - Username: admin (or your choice)
   - Password: Create a strong password
   - Confirm password

#### Add Media Libraries
1. **Movies Library**:
   - Name: "Movies"
   - Content Type: "Movies"
   - Folder: `/media/Movies` (maps to /Volumes/Plex/data/Media/Movies)
   - Enable: "Automatically add to collection"

2. **TV Shows Library**:
   - Name: "TV Shows" 
   - Content Type: "TV Shows"
   - Folder: `/media/TV` (maps to /Volumes/Plex/data/Media/TV)
   - Enable: "Automatically add to collection"

3. **Music Library**:
   - Name: "Music"
   - Content Type: "Music"  
   - Folder: `/media/Music` (maps to /Volumes/Plex/data/Media/Music)
   - Enable: "Automatically add to collection"

#### Metadata & Remote Access
1. **Metadata Settings**: 
   - Enable internet metadata providers
   - Enable automatic metadata download
2. **Remote Access**: 
   - Enable if you want external access
   - Configure port forwarding if needed

### 2. Complete Plex Initial Setup

#### Access Plex Web Interface
1. Open: http://localhost:32400/web
2. You'll see a sign-in screen

#### Initial Configuration
1. **Create/Sign In to Plex Account**:
   - Create free Plex account or sign in
   - This enables remote access and mobile apps

2. **Server Setup**:
   - Server Name: "Your Media Server" (or custom name)
   - Enable "Allow me to access my media outside my home"

#### Add Media Libraries
1. **Movies Library**:
   - Click "Add Library" 
   - Type: "Movies"
   - Name: "Movies"
   - Add Folder: `/media/Movies`
   - Advanced: Enable "Scan my library automatically"

2. **TV Shows Library**:
   - Type: "TV Shows"
   - Name: "TV Shows" 
   - Add Folder: `/media/TV`
   - Advanced: Enable "Scan my library automatically"

3. **Music Library**:
   - Type: "Music"
   - Name: "Music"
   - Add Folder: `/media/Music`
   - Advanced: Enable "Scan my library automatically"

### 3. Verify Docker Volume Mapping

The containers are configured with these volume mappings:
```yaml
# Jellyfin
- media-data:/media

# Plex  
- media-data:/media
```

However, your actual media is at `/Volumes/Plex/data/Media/`. We need to update the volume mapping.

### 4. Fix Volume Mapping (Critical)

Current docker-compose.yml uses `media-data` volume, but your media is on `/Volumes/Plex/data/Media/`.

**Option A: Update docker-compose.yml (Recommended)**
```yaml
volumes:
  # For Jellyfin
  - /Volumes/Plex/data/Media:/media

  # For Plex
  - /Volumes/Plex/data/Media:/media
```

**Option B: Create Symbolic Link**
```bash
# Create link from media-data volume to actual location
ln -sf /Volumes/Plex/data/Media/* /var/lib/docker/volumes/newmedia_media-data/_data/
```

## Quick Fix Commands

### Update Container Volume Mapping
```bash
# Stop containers
docker-compose stop jellyfin plex

# Update the docker-compose.yml to use direct path mapping
# Then restart
docker-compose up -d jellyfin plex
```

### Test Media Access
```bash
# Verify containers can access media
docker exec jellyfin ls -la /media/
docker exec plex ls -la /media/
```

## Expected Directory Structure in Containers

After proper volume mapping, containers should see:
```
/media/
├── Movies/
│   ├── 12 Monkeys (1995)/
│   ├── 12 Strong (2018)/
│   └── ... (your movie folders)
├── TV/
│   ├── 1883/
│   ├── 1923/
│   └── ... (your TV show folders)  
└── Music/
    ├── (hed) P.E_/
    ├── 3 Doors Down/
    └── ... (your music folders)
```

## Troubleshooting

### Jellyfin Issues
- **Setup wizard loops**: Clear browser cache, try incognito mode
- **Libraries not scanning**: Check volume permissions, restart container
- **No media found**: Verify volume mapping points to correct paths

### Plex Issues  
- **Can't sign in**: Check internet connection, try different browser
- **Libraries empty**: Verify volume mapping, check folder permissions
- **Remote access fails**: Check router port forwarding (32400)

### General Issues
- **Containers won't start**: Check Docker logs: `docker logs jellyfin` / `docker logs plex`
- **Permission errors**: Ensure media files have proper permissions
- **Network issues**: Verify containers are on same network

## Next Steps After Setup

1. **Configure Transcoding** (if needed):
   - Jellyfin: Settings > Playback > Hardware acceleration
   - Plex: Settings > Transcoder > Use hardware acceleration

2. **Set up Mobile Apps**:
   - Download Jellyfin mobile app
   - Download Plex mobile app  

3. **Configure User Accounts**:
   - Create additional user accounts for family members
   - Set up parental controls if needed

4. **Enable External Access**:
   - Configure router port forwarding
   - Set up dynamic DNS if desired

## Security Recommendations

1. **Change Default Passwords**: Use strong, unique passwords
2. **Enable HTTPS**: Configure SSL certificates
3. **Limit External Access**: Only enable if needed
4. **Regular Updates**: Keep containers updated with Watchtower
5. **Backup Configuration**: Backup config volumes regularly

## Support Resources

- **Jellyfin Documentation**: https://jellyfin.org/docs/
- **Plex Support**: https://support.plex.tv/
- **Docker Compose Reference**: https://docs.docker.com/compose/