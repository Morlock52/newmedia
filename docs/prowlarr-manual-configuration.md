# Prowlarr Manual Configuration Guide

## Accessing Prowlarr

1. Open your web browser and navigate to: http://localhost:9696
2. If prompted for authentication, check the authentication settings in the config

## Getting Your API Key

The API key for your Prowlarr instance is: `aad98e12668341e6a11630c125ab846e`

You can also find it in:
- **Web UI**: Settings → General → Security → API Key
- **Config File**: `/Users/morlock/fun/newmedia/media-data-fixed/config/prowlarr/config.xml`

## Adding Free Indexers Manually

### Method 1: Through the Web UI

1. **Navigate to Indexers**
   - Click on "Indexers" in the left sidebar
   - Click the "+" button to add a new indexer

2. **Add 1337x**
   - Search for "1337x" in the indexer list
   - Click on it to add
   - Configure:
     - Name: 1337x
     - Enable: ✓
     - URL: https://1337x.to (or current mirror)
   - Test and Save

3. **Add The Pirate Bay**
   - Search for "The Pirate Bay" or "TPB"
   - Configure:
     - Name: The Pirate Bay
     - Enable: ✓
     - URL: https://thepiratebay.org (or current mirror)
   - Test and Save

4. **Add YTS**
   - Search for "YTS"
   - Configure:
     - Name: YTS
     - Enable: ✓
     - URL: https://yts.mx
   - Test and Save

5. **Add EZTV**
   - Search for "EZTV"
   - Configure:
     - Name: EZTV
     - Enable: ✓
     - URL: https://eztv.re
   - Test and Save

6. **Add LimeTorrents**
   - Search for "LimeTorrents"
   - Configure:
     - Name: LimeTorrents
     - Enable: ✓
     - URL: https://www.limetorrents.lol
   - Test and Save

7. **Add TorrentGalaxy**
   - Search for "TorrentGalaxy"
   - Configure:
     - Name: TorrentGalaxy
     - Enable: ✓
     - URL: https://torrentgalaxy.to
   - Test and Save

### Method 2: Using the API Script

Once you've confirmed the API key is working:

```bash
cd /Users/morlock/fun/newmedia
./scripts/configure-prowlarr-indexers.sh
```

If the script fails with authentication errors, you may need to:
1. Disable authentication for local addresses in Prowlarr settings
2. Or update the API key in the script

### Method 3: Direct API Calls

You can also add indexers using curl commands:

```bash
# Set your API key
API_KEY="aad98e12668341e6a11630c125ab846e"

# Example: Add 1337x
curl -X POST "http://localhost:9696/api/v1/indexer" \
  -H "X-Api-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "1337x",
    "implementation": "Cardigann",
    "configContract": "CardigannIndexerSettings",
    "enable": true,
    "protocol": "torrent",
    "priority": 25,
    "fields": [
      {"name": "definitionFile", "value": "1337x"},
      {"name": "baseUrl", "value": "https://1337x.to"}
    ]
  }'
```

## Troubleshooting

### Authentication Issues

If you're getting 401 Unauthorized errors:

1. **Check Authentication Settings**
   - Go to Settings → General → Security
   - Set "Authentication Required" to "Disabled for Local Addresses"
   - Or set it to "None" if only accessing locally

2. **Verify API Key**
   - The API key in the config file: `aad98e12668341e6a11630c125ab846e`
   - Make sure you're using the correct key in your requests

3. **Check Bind Address**
   - Ensure Prowlarr is binding to all interfaces (*)
   - Or specifically to 127.0.0.1 if accessing locally

### Indexer Connection Issues

If indexers fail to connect:

1. **Check DNS Resolution**
   - Some indexer domains may be blocked by your ISP
   - Consider using alternative DNS servers (8.8.8.8, 1.1.1.1)

2. **Use Alternative Mirrors**
   - Many torrent sites have multiple mirrors
   - Search online for current working mirrors

3. **Enable Proxy/VPN**
   - Configure proxy settings in Prowlarr if needed
   - Some indexers may require VPN access

## Integrating with Other Services

Once indexers are configured, you can:

1. **Connect to Sonarr/Radarr**
   - Go to Settings → Apps
   - Add Sonarr/Radarr instances
   - Use their respective API keys

2. **Configure Download Clients**
   - Add your torrent client (qBittorrent, etc.)
   - Configure categories for automatic organization

3. **Set Up Sync Profiles**
   - Create profiles to control which indexers are used by which apps
   - Configure minimum seeders, quality preferences, etc.

## Free Indexer Limitations

Keep in mind:
- Free indexers may have rate limits
- Some may require solving CAPTCHAs periodically
- Availability can be inconsistent
- Consider using multiple indexers for redundancy

## Additional Free Indexers

You can also try adding:
- Nyaa (for anime)
- 1337x proxies
- Zooqle
- Magnetdl
- Torlock

These may be available in the Prowlarr indexer list depending on your version.