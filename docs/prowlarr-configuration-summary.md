# Prowlarr Configuration Summary

## Current Status

✅ **Prowlarr is running** at http://localhost:9696  
✅ **API Key found**: `aad98e12668341e6a11630c125ab846e`  
✅ **Updated .env file** with the API key  
✅ **Created configuration script** at `/scripts/configure-prowlarr-indexers.sh`  
✅ **Created manual guide** at `/docs/prowlarr-manual-configuration.md`  

## API Key Information

- **Location**: `/Users/morlock/fun/newmedia/media-data-fixed/config/prowlarr/config.xml`
- **Value**: `aad98e12668341e6a11630c125ab846e`
- **Authentication**: Currently set to "Forms" with "DisabledForLocalAddresses"

## Free Indexers to Configure

The following free indexers are recommended for 2025:

1. **1337x** - General torrents (movies, TV, games, software)
2. **The Pirate Bay** - Largest general torrent site
3. **YTS** - High-quality movie torrents with small file sizes
4. **EZTV** - TV show specialist
5. **LimeTorrents** - General torrents with good availability
6. **TorrentGalaxy** - General torrents with active community

**Note**: RARBG shut down in 2023, so it's no longer available.

## Configuration Methods

### Option 1: Web UI (Recommended)
1. Navigate to http://localhost:9696
2. Go to Indexers → Add (+)
3. Search and add each indexer manually
4. Test each indexer after adding

### Option 2: API Script
```bash
cd /Users/morlock/fun/newmedia
./scripts/configure-prowlarr-indexers.sh
```

### Option 3: Manual API Calls
Use the curl commands in the script as examples

## Troubleshooting

If you encounter authentication issues:
1. The authentication is set to "DisabledForLocalAddresses"
2. This means local connections should work without authentication
3. If accessing remotely, you'll need to log in first

## Next Steps

After configuring indexers:
1. Test each indexer using the "Test" button
2. Configure apps (Sonarr/Radarr) in Settings → Apps
3. Set up download clients in Settings → Download Clients
4. Create sync profiles for better control

## Important Notes

- Free indexers may have availability issues
- Some ISPs block torrent sites - use DNS over HTTPS or VPN if needed
- Always test indexers after adding them
- Keep multiple indexers for redundancy