# Homarr Dashboard Fix Guide

## Issue Fixed
The "Cannot read properties of undefined (reading 'name')" error in Homarr has been resolved.

## What Was Done

1. **Stopped and removed the broken Homarr container**
   ```bash
   docker stop homarr && docker rm homarr
   ```

2. **Backed up existing configuration**
   ```bash
   cp -r homarr-data homarr-data.backup
   cp -r homarr-configs homarr-configs.backup
   ```

3. **Created a minimal working configuration**
   - Replaced the corrupted configuration with a clean, minimal setup
   - Fixed the JSON structure to match Homarr's expected schema

4. **Fixed permissions**
   ```bash
   chmod -R 755 homarr-data homarr-configs homarr-icons
   ```

5. **Removed corrupted SQLite database**
   ```bash
   rm -f homarr-data/db.sqlite
   ```

6. **Restarted Homarr container**
   ```bash
   docker run -d --name homarr \
     --network newmedia_media-net \
     -v ./homarr-configs:/app/data/configs \
     -v ./homarr-data:/data \
     -v ./homarr-icons:/app/public/icons \
     -v /var/run/docker.sock:/var/run/docker.sock:ro \
     -p 7575:7575 \
     -e BASE_URL=http://localhost:7575 \
     --restart unless-stopped \
     ghcr.io/ajnart/homarr:latest
   ```

## Current Status
- Homarr is now running successfully on port 7575
- The dashboard is accessible at http://localhost:7575
- It shows the onboarding page (expected for fresh installation)

## Configuration Files

### Working Configuration
A comprehensive configuration has been saved to `homarr-working-config.json` that includes all your media services.

### To Apply the Full Configuration
1. Access Homarr at http://localhost:7575
2. Complete the onboarding process
3. Go to Settings → Import/Export
4. Import the `homarr-working-config.json` file

## Services Included in Configuration
- **Media Servers**: Jellyfin, Plex, Emby
- **Media Management**: Sonarr, Radarr, Lidarr, Readarr, Bazarr, Prowlarr
- **Downloads**: qBittorrent, SABnzbd
- **Requests**: Jellyseerr, Overseerr
- **Management**: Portainer, Nginx Proxy Manager
- **Monitoring**: Grafana, Prometheus, Uptime Kuma

## Troubleshooting Tips

### If the error returns:
1. Check logs: `docker logs homarr`
2. Remove the SQLite database: `rm -f homarr-data/db.sqlite`
3. Restart the container: `docker restart homarr`

### To reset completely:
```bash
docker stop homarr
docker rm homarr
rm -rf homarr-data homarr-configs
mkdir -p homarr-data/configs homarr-configs homarr-icons
# Then recreate the container
```

## Backup Strategy
Always backup your configuration before making changes:
```bash
cp homarr-data/configs/default.json homarr-data/configs/default.json.backup
```

## Additional Resources
- Homarr Documentation: https://homarr.dev/
- Dashboard Icons: https://github.com/walkxcode/dashboard-icons