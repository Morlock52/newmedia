# Optimal Media Server Folder Structure for 2025

## Executive Summary

Based on comprehensive research from TRaSH Guides, Reddit communities (r/selfhosted, r/sonarr, r/radarr), and current best practices, this document outlines the optimal folder structure for a shared volume media server setup with hardlink support and proper Docker integration.

## Core Folder Structure (TRaSH Guide Standard)

```
/Volumes/Plex/data/
├── torrents/
│   ├── movies/
│   ├── tv/
│   ├── music/
│   └── books/
├── usenet/
│   ├── incomplete/
│   └── complete/
│       ├── movies/
│       ├── tv/
│       ├── music/
│       └── books/
└── media/
    ├── movies/
    ├── tv/
    ├── music/
    └── books/
```

## Detailed Media Library Structure

### Movies
```
/Volumes/Plex/data/media/movies/
├── Avatar (2009)/
│   ├── Avatar (2009).mkv
│   ├── Avatar (2009) - Deleted Scenes.mkv
│   └── extras/
│       └── behind-the-scenes.mkv
├── Blade Runner 2049 (2017)/
│   ├── Blade Runner 2049 (2017).mkv
│   └── Blade Runner 2049 (2017) {edition-Director's Cut}.mkv
└── The Matrix (1999)/
    └── The Matrix (1999).mkv
```

### TV Shows
```
/Volumes/Plex/data/media/tv/
├── Breaking Bad (2008)/
│   ├── Season 01/
│   │   ├── Breaking Bad - S01E01 - Pilot.mkv
│   │   ├── Breaking Bad - S01E02 - Cat's in the Bag.mkv
│   │   └── ...
│   ├── Season 02/
│   │   └── ...
│   └── extras/
│       └── behind-the-scenes/
├── The Office (US) (2005)/
│   ├── Season 01/
│   ├── Season 02/
│   └── ...
```

### Music
```
/Volumes/Plex/data/media/music/
├── Artist Name/
│   ├── Album Name (Year)/
│   │   ├── 01 - Track Name.flac
│   │   ├── 02 - Track Name.flac
│   │   └── folder.jpg
│   └── Another Album (Year)/
│       └── ...
```

### Audiobooks
```
/Volumes/Plex/data/media/books/
├── Author Name/
│   ├── Book Title (Year)/
│   │   ├── Book Title - Part 01.m4b
│   │   ├── Book Title - Part 02.m4b
│   │   └── cover.jpg
│   └── Another Book (Year)/
│       └── ...
```

## Docker Volume Mapping Configuration

### Recommended Docker Compose Structure

```yaml
version: '3.8'

services:
  plex:
    image: ghcr.io/linuxserver/plex:latest
    container_name: plex
    environment:
      - PUID=1001
      - PGID=1001
      - TZ=America/New_York
      - VERSION=docker
    volumes:
      - /Volumes/Plex/config/plex:/config
      - /Volumes/Plex/data/media:/data/media
    ports:
      - 32400:32400
    restart: unless-stopped

  sonarr:
    image: ghcr.io/linuxserver/sonarr:latest
    container_name: sonarr
    environment:
      - PUID=1001
      - PGID=1001
      - TZ=America/New_York
      - UMASK=002
    volumes:
      - /Volumes/Plex/config/sonarr:/config
      - /Volumes/Plex/data:/data
    ports:
      - 8989:8989
    restart: unless-stopped

  radarr:
    image: ghcr.io/linuxserver/radarr:latest
    container_name: radarr
    environment:
      - PUID=1001
      - PGID=1001
      - TZ=America/New_York
      - UMASK=002
    volumes:
      - /Volumes/Plex/config/radarr:/config
      - /Volumes/Plex/data:/data
    ports:
      - 7878:7878
    restart: unless-stopped

  qbittorrent:
    image: ghcr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=1001
      - PGID=1001
      - TZ=America/New_York
      - WEBUI_PORT=8080
      - UMASK=002
    volumes:
      - /Volumes/Plex/config/qbittorrent:/config
      - /Volumes/Plex/data/torrents:/data/torrents
    ports:
      - 8080:8080
      - 6881:6881
      - 6881:6881/udp
    restart: unless-stopped
```

## PUID/PGID Configuration

### Step 1: Create Media User and Group
```bash
# Create a dedicated media group
sudo groupadd -g 1001 media

# Create a media user
sudo useradd -u 1001 -g 1001 -m -s /bin/bash mediauser

# Add your user to the media group
sudo usermod -a -G media $USER
```

### Step 2: Set Proper Permissions
```bash
# Set ownership
sudo chown -R mediauser:media /Volumes/Plex/data

# Set permissions for folders (775) and files (664)
sudo chmod -R 775 /Volumes/Plex/data
find /Volumes/Plex/data -type f -exec chmod 664 {} \;
```

### Step 3: Configure UMASK
Set `UMASK=002` in all containers to ensure new files are created with group write permissions.

## Hardlink Requirements and Limitations

### ✅ Requirements for Hardlinks to Work
1. **Same File System**: All folders must be on the same physical drive/partition
2. **Supported File System**: Use NTFS, ext4, btrfs, ZFS (avoid exFAT, FAT32)
3. **Consistent Docker Mapping**: Map the root `/data` folder to all containers
4. **Proper Permissions**: Ensure all apps can read/write to shared folders

### ❌ Hardlink Limitations
1. Cannot hardlink across different file systems or partitions
2. Cannot hardlink directories (only files)
3. Some file systems (exFAT, FAT32) don't support hardlinks
4. Network drives may not support hardlinks properly

### Verification Commands
```bash
# Check if hardlinks are working (link count > 1)
ls -la /Volumes/Plex/data/torrents/movies/
ls -la /Volumes/Plex/data/media/movies/

# Alternative check using stat
stat /path/to/file/in/torrents
stat /path/to/file/in/media
```

## External Volume Considerations

### macOS External Drive Limitations
- **Performance**: External USB drives can cause poor performance
- **Reliability**: May disconnect unexpectedly, breaking container access
- **Hardlinks**: Limited or unreliable hardlink support
- **Permissions**: Complex permission management across USB connections

### Recommended Solutions
1. **Internal Drive**: Use internal SSD/HDD when possible
2. **Thunderbolt/USB-C**: If external is necessary, use fastest connection
3. **Always Mount**: Configure automatic mounting at startup
4. **Format Properly**: Use APFS or NTFS, avoid exFAT

## Naming Conventions and Best Practices

### File Naming Rules
- Avoid special characters: `< > : " / \ | ? *`
- Use parentheses for years: `Movie Name (2023)`
- Season padding: `Season 01`, not `Season 1`
- Episode format: `S01E01`
- Edition info: `{edition-Director's Cut}`

### Metadata Enhancement
```
# Example with metadata IDs
Avatar [imdbid-tt0499549] [tmdbid-19995] (2009)/
  Avatar [imdbid-tt0499549] [tmdbid-19995] (2009).mkv
```

### Multi-Edition Support
```
Blade Runner (1982)/
  Blade Runner (1982) {edition-Theatrical Cut}.mkv
  Blade Runner (1982) {edition-Director's Cut}.mkv
  Blade Runner (1982) {edition-Final Cut}.mkv
```

## Application Path Configuration

### Sonarr/Radarr Settings
- **Root Folder**: `/data/media/tv` (Sonarr) or `/data/media/movies` (Radarr)
- **Download Client Path**: `/data/torrents/tv` or `/data/torrents/movies`
- **Enable**: "Use Hardlinks instead of Copy" in Media Management

### Download Client Settings
- **Default Category**: Set categories for movies/tv
- **Download Path**: `/data/torrents/{category}/`
- **Completed Path**: Same as download path (for hardlinks)

### Plex Library Settings
- **Movies**: `/data/media/movies`
- **TV Shows**: `/data/media/tv`  
- **Music**: `/data/media/music`
- **Audiobooks**: `/data/media/books`

## Monitoring and Maintenance

### Health Checks
```bash
# Check disk usage
df -h /Volumes/Plex

# Monitor hardlink usage
find /Volumes/Plex/data -type f -links +1 | wc -l

# Check permissions
ls -la /Volumes/Plex/data/
```

### Automated Cleanup Scripts
Consider implementing automated cleanup for:
- Failed downloads
- Orphaned files
- Old logs
- Broken hardlinks

## Migration from Existing Setup

### Step 1: Backup Current Structure
```bash
# Create backup of current library
rsync -av --progress /current/media/path/ /backup/location/
```

### Step 2: Gradual Migration
1. Set up new structure alongside existing
2. Test with small subset of media
3. Verify hardlinks are working
4. Update application configs
5. Complete migration in phases

### Step 3: Validation
- Verify all media is accessible in Plex/Jellyfin
- Confirm download automation is working
- Test hardlink functionality
- Check disk space savings

## Performance Optimization Tips

1. **SSD for Databases**: Keep Plex/Sonarr/Radarr databases on SSD
2. **Separate Config**: Keep app configs separate from media
3. **Regular Maintenance**: Schedule periodic cleanup and optimization
4. **Monitor Resources**: Use tools like htop, iotop to monitor performance
5. **Backup Strategy**: Regular backups of configurations and important metadata

## Security Considerations

1. **Least Privilege**: Applications only access necessary folders
2. **Regular Updates**: Keep container images updated
3. **Network Security**: Use VPN for download clients if needed
4. **Access Control**: Limit Plex remote access if not needed
5. **File Permissions**: Regular audit of file permissions and ownership

This structure ensures optimal performance, efficient storage usage through hardlinks, proper permissions management, and compatibility with all major media server applications while following current best practices for 2025.