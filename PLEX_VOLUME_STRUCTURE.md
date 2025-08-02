# Plex Volume Structure Documentation

## Created Structure: `/Volumes/Plex/data/`

Following TRaSH guides, the optimal media folder structure has been created:

```
/Volumes/Plex/data/
├── torrents/           # Download client staging area
│   ├── movies/         # Movie torrents/downloads
│   ├── tv/            # TV show torrents/downloads  
│   ├── music/         # Music torrents/downloads
│   └── books/         # Book torrents/downloads
├── media/             # Final organized media for Plex
│   ├── movies/        # Movies for Plex library
│   ├── tv/           # TV shows for Plex library
│   ├── music/        # Music for Plex library
│   └── books/        # Books/audiobooks for Plex library
└── downloads/         # Download client working directory
    ├── complete/      # Completed downloads
    └── incomplete/    # In-progress downloads
```

## Volume Information

- **Mount Point**: `/Volumes/Plex`
- **Type**: SMB network share (//GUEST:@Morlocks-nas._smb._tcp.local/Plex)
- **Total Size**: 21TB
- **Available**: 3.3TB (15% free)
- **Permissions**: rwxrwxrwx (777) - SMB mount with full access

## Docker Compose Updates

Updated `docker-compose-demo.yml` with new volume mappings:

### Media Servers
- **Jellyfin**: `/Volumes/Plex/data/media:/media:ro`
- **Plex**: `/Volumes/Plex/data/media:/media:ro`

### Arr Services  
- **Sonarr**: 
  - TV: `/Volumes/Plex/data/media/tv:/tv`
  - Downloads: `/Volumes/Plex/data/downloads:/downloads`
  - Torrents: `/Volumes/Plex/data/torrents:/torrents`
  
- **Radarr**:
  - Movies: `/Volumes/Plex/data/media/movies:/movies`
  - Downloads: `/Volumes/Plex/data/downloads:/downloads` 
  - Torrents: `/Volumes/Plex/data/torrents:/torrents`
  
- **Lidarr**:
  - Music: `/Volumes/Plex/data/media/music:/music`
  - Downloads: `/Volumes/Plex/data/downloads:/downloads`
  - Torrents: `/Volumes/Plex/data/torrents:/torrents`

### Download Clients
- **qBittorrent**: 
  - Downloads: `/Volumes/Plex/data/downloads:/downloads`
  - Torrents: `/Volumes/Plex/data/torrents:/torrents`
  
- **Transmission**:
  - Downloads: `/Volumes/Plex/data/downloads:/downloads`
  - Torrents: `/Volumes/Plex/data/torrents:/torrents`
  
- **SABnzbd**:
  - Downloads: `/Volumes/Plex/data/downloads:/downloads`

## Existing Data Structure

The volume contains existing data that should be migrated:

### Legacy Folders
- `/Volumes/Plex/Torrents/` - Old torrent structure
- `/Volumes/Plex/Media/` - Old media structure  
- `/Volumes/Plex/Data/Media/` - Another media location

### Migration Recommendations

1. **Keep existing structure** for now to avoid data loss
2. **Gradually migrate** data to new TRaSH-compliant structure
3. **Update arr services** to use new paths progressively
4. **Test thoroughly** before removing old structures

## TRaSH Guide Benefits

This structure follows TRaSH (TRaSH-Guides) recommendations:

- **Hardlinks**: Enables instant moves instead of copying
- **Atomic moves**: Files appear instantly when complete
- **No duplicate storage**: Same file accessible from multiple paths
- **Consistent naming**: Standardized across all services
- **Optimal performance**: Reduces I/O operations

## Current User Context

- **User**: morlock (UID: 501)
- **Group**: staff (GID: 20)
- **Docker containers**: Using PUID=1000, PGID=1000

## Backup Information

- Original docker-compose backed up to: `docker-compose-demo.yml.backup`
- Created: $(date)
- Volume write test: ✅ Passed

## Next Steps

1. Test docker-compose up with new volume mappings
2. Configure arr services to use new folder structure
3. Verify hardlink functionality
4. Plan data migration from legacy folders
5. Monitor disk space during migration