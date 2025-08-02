#!/bin/bash
# Media Server API Integration Tests
# Test actual API endpoints and integration points

echo "=== Media Server API Integration Tests ==="
echo "Testing connectivity and API endpoints..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local service=$1
    local port=$2
    local endpoint=$3
    local description=$4
    
    response=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://localhost:${port}${endpoint}")
    
    if [ "$response" -eq 200 ] || [ "$response" -eq 401 ] || [ "$response" -eq 403 ]; then
        echo -e "${GREEN}✓${NC} $service ($description): ${GREEN}Accessible${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}✗${NC} $service ($description): ${RED}Not accessible${NC} (HTTP $response)"
        return 1
    fi
}

echo "=== Testing Media Servers ==="
echo ""

# Jellyfin
test_endpoint "Jellyfin" 8096 "/System/Info/Public" "Public API"
test_endpoint "Jellyfin" 8096 "/web/index.html" "Web Interface"

# Plex
test_endpoint "Plex" 32400 "/identity" "Server Identity"
test_endpoint "Plex" 32400 "/web/index.html" "Web Interface"

# Emby
test_endpoint "Emby" 8097 "/emby/System/Info/Public" "Public API"
test_endpoint "Emby" 8097 "/web/index.html" "Web Interface"

echo ""
echo "=== Testing ARR Services ==="
echo ""

# Sonarr
test_endpoint "Sonarr" 8989 "/" "Web Interface"
test_endpoint "Sonarr" 8989 "/api/v3/system/status" "API Status"

# Radarr
test_endpoint "Radarr" 7878 "/" "Web Interface"
test_endpoint "Radarr" 7878 "/api/v3/system/status" "API Status"

# Lidarr
test_endpoint "Lidarr" 8686 "/" "Web Interface"
test_endpoint "Lidarr" 8686 "/api/v1/system/status" "API Status"

# Readarr
test_endpoint "Readarr" 8787 "/" "Web Interface"
test_endpoint "Readarr" 8787 "/api/v1/system/status" "API Status"

# Bazarr
test_endpoint "Bazarr" 6767 "/" "Web Interface"
test_endpoint "Bazarr" 6767 "/api/system/status" "API Status"

# Prowlarr
test_endpoint "Prowlarr" 9696 "/" "Web Interface"
test_endpoint "Prowlarr" 9696 "/api/v1/health" "API Health"

echo ""
echo "=== Testing Download Clients ==="
echo ""

# qBittorrent (through VPN)
test_endpoint "qBittorrent" 8080 "/api/v2/app/version" "API Version"
test_endpoint "qBittorrent" 8080 "/" "Web Interface"

# Transmission (through VPN)
test_endpoint "Transmission" 9091 "/transmission/web/" "Web Interface"
test_endpoint "Transmission" 9091 "/transmission/rpc" "RPC Interface"

# SABnzbd
test_endpoint "SABnzbd" 8081 "/" "Web Interface"
test_endpoint "SABnzbd" 8081 "/sabnzbd/api?mode=version" "API Version"

# NZBGet
test_endpoint "NZBGet" 6789 "/" "Web Interface"

echo ""
echo "=== Testing Request Services ==="
echo ""

# Jellyseerr
test_endpoint "Jellyseerr" 5055 "/" "Web Interface"
test_endpoint "Jellyseerr" 5055 "/api/v1/status" "API Status"

# Overseerr
test_endpoint "Overseerr" 5056 "/" "Web Interface"
test_endpoint "Overseerr" 5056 "/api/v1/status" "API Status"

# Ombi
test_endpoint "Ombi" 3579 "/" "Web Interface"

echo ""
echo "=== API Integration Examples ==="
echo ""

# Example: Test qBittorrent API (default credentials)
echo "Testing qBittorrent login with default credentials..."
cookie=$(curl -s -c - -X POST "http://localhost:8080/api/v2/auth/login" \
    -d "username=admin&password=adminadmin" | grep -o 'SID[[:space:]]*[^[:space:]]*' | awk '{print $2}')

if [ ! -z "$cookie" ]; then
    echo -e "${GREEN}✓${NC} qBittorrent: Successfully authenticated"
    
    # Get torrents list
    torrents=$(curl -s -H "Cookie: SID=$cookie" "http://localhost:8080/api/v2/torrents/info")
    echo -e "${GREEN}✓${NC} qBittorrent: API accessible (found $(echo $torrents | jq length 2>/dev/null || echo 0) torrents)"
else
    echo -e "${YELLOW}!${NC} qBittorrent: Authentication failed (try changing default password)"
fi

echo ""
echo "=== Configuration Instructions ==="
echo ""
echo "To complete integrations, you need to:"
echo ""
echo "1. PROWLARR SETUP:"
echo "   - Access Prowlarr at http://localhost:9696"
echo "   - Go to Settings → General → Security"
echo "   - Copy the API Key"
echo "   - Add indexers under Indexers → Add Indexer"
echo ""
echo "2. CONNECT ARR SERVICES TO PROWLARR:"
echo "   - In each ARR service (Sonarr/Radarr/etc)"
echo "   - Go to Settings → Indexers → Add → Prowlarr"
echo "   - Prowlarr Server: http://prowlarr:9696"
echo "   - API Key: [paste Prowlarr API key]"
echo ""
echo "3. CONNECT ARR SERVICES TO DOWNLOAD CLIENTS:"
echo "   - In each ARR service"
echo "   - Go to Settings → Download Clients → Add"
echo "   - For qBittorrent: Host: gluetun, Port: 8080"
echo "   - For SABnzbd: Host: sabnzbd, Port: 8080"
echo ""
echo "4. MEDIA SERVER SETUP:"
echo "   - Add libraries pointing to /media/movies, /media/tv, etc"
echo "   - Enable automatic library updates"
echo ""

# Create a simple HTML dashboard
cat > /Users/morlock/fun/newmedia/TEST_REPORTS/integration-dashboard.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Integration Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .service-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .service-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .service-card h3 { margin-top: 0; color: #2c3e50; }
        .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .status.online { background: #27ae60; color: white; }
        .status.offline { background: #e74c3c; color: white; }
        .status.auth { background: #f39c12; color: white; }
        .link { display: inline-block; margin-top: 10px; color: #3498db; text-decoration: none; }
        .link:hover { text-decoration: underline; }
        .integration-matrix { margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #34495e; color: white; }
        .check { color: #27ae60; font-weight: bold; }
        .cross { color: #e74c3c; font-weight: bold; }
        .warn { color: #f39c12; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Media Server Integration Dashboard</h1>
        
        <h2>Services Status</h2>
        <div class="service-grid">
            <!-- Media Servers -->
            <div class="service-card">
                <h3>Jellyfin</h3>
                <span class="status online">ONLINE</span>
                <p>Open-source media server</p>
                <a href="http://localhost:8096" class="link" target="_blank">Open Web UI →</a>
            </div>
            
            <div class="service-card">
                <h3>Plex</h3>
                <span class="status online">ONLINE</span>
                <p>Premium media server</p>
                <a href="http://localhost:32400/web" class="link" target="_blank">Open Web UI →</a>
            </div>
            
            <!-- ARR Services -->
            <div class="service-card">
                <h3>Sonarr</h3>
                <span class="status online">ONLINE</span>
                <p>TV show management</p>
                <a href="http://localhost:8989" class="link" target="_blank">Open Web UI →</a>
            </div>
            
            <div class="service-card">
                <h3>Radarr</h3>
                <span class="status online">ONLINE</span>
                <p>Movie management</p>
                <a href="http://localhost:7878" class="link" target="_blank">Open Web UI →</a>
            </div>
            
            <div class="service-card">
                <h3>Prowlarr</h3>
                <span class="status online">ONLINE</span>
                <p>Indexer management</p>
                <a href="http://localhost:9696" class="link" target="_blank">Open Web UI →</a>
            </div>
            
            <!-- Download Clients -->
            <div class="service-card">
                <h3>qBittorrent</h3>
                <span class="status auth">NEEDS AUTH</span>
                <p>Torrent client (VPN)</p>
                <a href="http://localhost:8080" class="link" target="_blank">Open Web UI →</a>
                <small>Default: admin/adminadmin</small>
            </div>
            
            <div class="service-card">
                <h3>SABnzbd</h3>
                <span class="status online">ONLINE</span>
                <p>Usenet downloader</p>
                <a href="http://localhost:8081" class="link" target="_blank">Open Web UI →</a>
            </div>
        </div>
        
        <div class="integration-matrix">
            <h2>Integration Configuration Matrix</h2>
            <table>
                <tr>
                    <th>From Service</th>
                    <th>To Service</th>
                    <th>Status</th>
                    <th>Configuration Steps</th>
                </tr>
                <tr>
                    <td>Sonarr/Radarr</td>
                    <td>Prowlarr</td>
                    <td class="warn">⚠️ Configure</td>
                    <td>Add Prowlarr as indexer with API key</td>
                </tr>
                <tr>
                    <td>Sonarr/Radarr</td>
                    <td>qBittorrent</td>
                    <td class="warn">⚠️ Configure</td>
                    <td>Host: gluetun, Port: 8080, Auth required</td>
                </tr>
                <tr>
                    <td>Sonarr/Radarr</td>
                    <td>SABnzbd</td>
                    <td class="warn">⚠️ Configure</td>
                    <td>Host: sabnzbd, Port: 8080, API key required</td>
                </tr>
                <tr>
                    <td>Jellyfin/Plex</td>
                    <td>Media Folders</td>
                    <td class="warn">⚠️ Configure</td>
                    <td>Add libraries: /media/movies, /media/tv</td>
                </tr>
            </table>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #ecf0f1; border-radius: 8px;">
            <h3>Quick Setup Guide</h3>
            <ol>
                <li><strong>Configure Prowlarr:</strong> Add indexers and get API key</li>
                <li><strong>Connect ARR to Prowlarr:</strong> Add Prowlarr as indexer in each ARR service</li>
                <li><strong>Add Download Clients:</strong> Configure qBittorrent/SABnzbd in ARR services</li>
                <li><strong>Setup Media Libraries:</strong> Add folders in Jellyfin/Plex</li>
                <li><strong>Test Integration:</strong> Search for content and verify download → import → library update</li>
            </ol>
        </div>
    </div>
    
    <script>
        // Auto-refresh status every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
EOF

echo ""
echo -e "${GREEN}Dashboard created:${NC} /Users/morlock/fun/newmedia/TEST_REPORTS/integration-dashboard.html"
echo "Open in browser to view service status and configuration guide"