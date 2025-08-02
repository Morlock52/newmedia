# ARR Services Configuration Guide

This guide explains how to configure Sonarr, Radarr, and Lidarr to work with Prowlarr as the indexer manager and qBittorrent as the download client.

## Prerequisites

Ensure all services are running:
```bash
docker-compose up -d prowlarr sonarr radarr lidarr qbittorrent gluetun
```

## Service Information

### API Keys
- **Prowlarr**: `b7ef1468932940b2a4cf27ad980f1076`
- **Sonarr**: `6e6bfac6e15d4f9a9d0e0d35ec0b8e23`
- **Radarr**: `7b74da952069425f9568ea361b001a12`
- **Lidarr**: `e8262da767e34a6b8ca7ca1e92384d96`

### Service URLs
- **Prowlarr**: http://localhost:9696
- **Sonarr**: http://localhost:8989
- **Radarr**: http://localhost:7878
- **Lidarr**: http://localhost:8686
- **qBittorrent**: http://localhost:8080

## Automatic Configuration

Run the configuration script:
```bash
cd /Users/morlock/fun/newmedia
./scripts/configure-arr-services.sh
```

This script will:
1. Add Sonarr, Radarr, and Lidarr to Prowlarr
2. Configure qBittorrent as the download client for each service
3. Set up root folders for media storage

## Manual Configuration Steps

### 1. Configure Prowlarr

#### Add Applications to Prowlarr:

1. Open Prowlarr: http://localhost:9696
2. Go to Settings → Apps
3. Click the "+" button to add each application:

**For Sonarr:**
- Name: Sonarr
- Sync Level: Full Sync
- Prowlarr Server: http://localhost:9696
- Sonarr Server: http://sonarr:8989
- API Key: `6e6bfac6e15d4f9a9d0e0d35ec0b8e23`

**For Radarr:**
- Name: Radarr
- Sync Level: Full Sync
- Prowlarr Server: http://localhost:9696
- Radarr Server: http://radarr:7878
- API Key: `7b74da952069425f9568ea361b001a12`

**For Lidarr:**
- Name: Lidarr
- Sync Level: Full Sync
- Prowlarr Server: http://localhost:9696
- Lidarr Server: http://lidarr:8686
- API Key: `e8262da767e34a6b8ca7ca1e92384d96`

#### Add Indexers to Prowlarr:

1. Go to Indexers → Add Indexer
2. Search for and add your preferred indexers
3. Configure each indexer with appropriate settings
4. Test the indexer to ensure it's working

### 2. Configure Download Clients in ARR Services

For each ARR service (Sonarr, Radarr, Lidarr):

1. Open the service web interface
2. Go to Settings → Download Clients
3. Click "+" to add a new download client
4. Select "qBittorrent"
5. Configure with these settings:
   - Name: qBittorrent
   - Host: gluetun (or localhost if not using Docker)
   - Port: 8080
   - Username: admin
   - Password: adminadmin
   - Category: [service-name] (e.g., "sonarr", "radarr", "lidarr")

### 3. Configure Root Folders

For each ARR service:

1. Go to Settings → Media Management
2. Add Root Folder:
   - **Sonarr**: `/media/tv`
   - **Radarr**: `/media/movies`
   - **Lidarr**: `/media/music`

### 4. Configure Quality Profiles

For each service, review and configure quality profiles:

1. Go to Settings → Profiles
2. Edit or create quality profiles based on your preferences
3. Common profiles:
   - HD-1080p for movies/TV
   - FLAC/MP3-320 for music

## Testing the Configuration

### 1. Test Prowlarr Integration
1. In Prowlarr, go to System → Tasks
2. Run "Application Sync" task
3. Check that indexers appear in each ARR service

### 2. Test Download Client
1. In any ARR service, go to Settings → Download Clients
2. Click "Test" on the qBittorrent entry
3. Ensure the test passes

### 3. Test Full Workflow
1. Search for media in any ARR service
2. Add something to download
3. Monitor the download in qBittorrent
4. Verify the file is imported after completion

## Troubleshooting

### Common Issues:

1. **Indexers not appearing in ARR services:**
   - Ensure Prowlarr sync is enabled
   - Check API keys are correct
   - Run manual sync in Prowlarr

2. **Download client connection failed:**
   - Verify qBittorrent is running
   - Check if using correct hostname (gluetun vs localhost)
   - Ensure qBittorrent WebUI is enabled

3. **Media not importing:**
   - Check root folder permissions
   - Verify download categories match
   - Ensure completed download handling is enabled

### API Testing Commands:

Test Prowlarr:
```bash
curl -H "X-Api-Key: b7ef1468932940b2a4cf27ad980f1076" http://localhost:9696/api/v1/system/status
```

Test Sonarr:
```bash
curl -H "X-Api-Key: 6e6bfac6e15d4f9a9d0e0d35ec0b8e23" http://localhost:8989/api/v3/system/status
```

Test Radarr:
```bash
curl -H "X-Api-Key: 7b74da952069425f9568ea361b001a12" http://localhost:7878/api/v3/system/status
```

Test Lidarr:
```bash
curl -H "X-Api-Key: e8262da767e34a6b8ca7ca1e92384d96" http://localhost:8686/api/v1/system/status
```

## Security Considerations

1. **Change default passwords:**
   - qBittorrent default password should be changed
   - Consider enabling authentication for ARR services

2. **API Key Security:**
   - Keep API keys secure
   - Don't expose services directly to the internet
   - Use reverse proxy with authentication if remote access is needed

3. **Network Security:**
   - Services communicate over Docker internal network
   - VPN protection for torrent traffic via Gluetun

## Next Steps

1. **Add Indexers:** Configure your preferred torrent/usenet indexers in Prowlarr
2. **Configure Profiles:** Set up quality profiles in each ARR service
3. **Add Media:** Start searching and adding content to your library
4. **Configure Notifications:** Set up Discord, Telegram, or email notifications
5. **Setup Backup:** Configure regular backups of configuration directories