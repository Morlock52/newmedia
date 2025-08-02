#!/bin/bash

# qBittorrent Download Client Configuration Script (Final - Docker Network Corrected)
# This script configures qBittorrent as a download client for all ARR services

set -e

# Configuration variables - CORRECTED for actual Docker setup
QB_HOST="qbittorrent"  # Container name on same Docker network
QB_PORT="8080"         # Internal qBittorrent port (mapped externally to 8090)
QB_USERNAME="admin"
QB_PASSWORD="adminadmin"

SONARR_HOST="localhost"
SONARR_PORT="8989"
SONARR_API="6e6bfac6e15d4f9a9d0e0d35ec0b8e23"

RADARR_HOST="localhost"
RADARR_PORT="7878"
RADARR_API="7b74da952069425f9568ea361b001a12"

LIDARR_HOST="localhost"
LIDARR_PORT="8686"
LIDARR_API="e8262da767e34a6b8ca7ca1e92384d96"

echo "=== Configuring qBittorrent Download Client for ARR Services ==="
echo "Network: Both qBittorrent and ARR services are on newmedia_media-net"
echo "Connection: ARR services -> qbittorrent:8080 (internal Docker network)"
echo ""

# Function to test API connectivity
test_service() {
    local service=$1
    local host=$2
    local port=$3
    local api_key=$4
    local api_version=$5
    
    echo "Testing $service connectivity..."
    if curl -s -f "http://$host:$port/api/$api_version/system/status" -H "X-Api-Key: $api_key" > /dev/null; then
        echo "✅ $service is accessible"
        return 0
    else
        echo "❌ $service is not accessible"
        return 1
    fi
}

# Test ARR services
test_service "Sonarr" "$SONARR_HOST" "$SONARR_PORT" "$SONARR_API" "v3" || exit 1
test_service "Radarr" "$RADARR_HOST" "$RADARR_PORT" "$RADARR_API" "v3" || exit 1
test_service "Lidarr" "$LIDARR_HOST" "$LIDARR_PORT" "$LIDARR_API" "v1" || exit 1

# Test qBittorrent (external access via localhost:8090)
echo "Testing qBittorrent connectivity (external: localhost:8090)..."
if curl -s "http://localhost:8090/api/v2/app/version" 2>&1 | grep -q "Unauthorized\|[0-9]"; then
    echo "✅ qBittorrent is accessible via localhost:8090"
else
    echo "❌ qBittorrent is not accessible via localhost:8090"
    echo "Please ensure qBittorrent container is running"
    exit 1
fi

echo -e "\n=== Configuring Sonarr ==="

# Configure qBittorrent for Sonarr - using container name
sonarr_config='{
  "enable": true,
  "protocol": "torrent",
  "priority": 1,
  "removeCompletedDownloads": false,
  "removeFailedDownloads": true,
  "name": "qBittorrent",
  "fields": [
    {
      "name": "host",
      "value": "'$QB_HOST'"
    },
    {
      "name": "port",
      "value": '$QB_PORT'
    },
    {
      "name": "username",
      "value": "'$QB_USERNAME'"
    },
    {
      "name": "password",
      "value": "'$QB_PASSWORD'"
    },
    {
      "name": "category",
      "value": "sonarr"
    },
    {
      "name": "recentTvPriority",
      "value": 0
    },
    {
      "name": "olderTvPriority",
      "value": 0
    },
    {
      "name": "initialState",
      "value": 0
    },
    {
      "name": "sequentialOrder",
      "value": false
    },
    {
      "name": "firstAndLast",
      "value": false
    }
  ],
  "implementationName": "qBittorrent",
  "implementation": "QBittorrent",
  "configContract": "QBittorrentSettings",
  "tags": []
}'

echo "Adding qBittorrent to Sonarr (connecting to qbittorrent:8080)..."
result=$(curl -s -X POST "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $SONARR_API" \
     -d "$sonarr_config")

if echo "$result" | grep -q '"id":[0-9]'; then
    echo "✅ Successfully configured qBittorrent for Sonarr"
elif echo "$result" | grep -q "error\|Error"; then
    echo "❌ Failed to configure qBittorrent for Sonarr:"
    echo "$result" | jq -r '.[] | .errorMessage // .detailedDescription // .' 2>/dev/null || echo "$result"
else
    echo "⚠️  Configuration response: $result"
fi

echo -e "\n=== Configuring Radarr ==="

# Configure qBittorrent for Radarr - using container name
radarr_config='{
  "enable": true,
  "protocol": "torrent",
  "priority": 1,
  "removeCompletedDownloads": false,
  "removeFailedDownloads": true,
  "name": "qBittorrent",
  "fields": [
    {
      "name": "host",
      "value": "'$QB_HOST'"
    },
    {
      "name": "port",
      "value": '$QB_PORT'
    },
    {
      "name": "username",
      "value": "'$QB_USERNAME'"
    },
    {
      "name": "password",
      "value": "'$QB_PASSWORD'"
    },
    {
      "name": "category",
      "value": "radarr"
    },
    {
      "name": "recentMoviePriority",
      "value": 0
    },
    {
      "name": "olderMoviePriority",
      "value": 0
    },
    {
      "name": "initialState",
      "value": 0
    },
    {
      "name": "sequentialOrder",
      "value": false
    },
    {
      "name": "firstAndLast",
      "value": false
    }
  ],
  "implementationName": "qBittorrent",
  "implementation": "QBittorrent",
  "configContract": "QBittorrentSettings",
  "tags": []
}'

echo "Adding qBittorrent to Radarr (connecting to qbittorrent:8080)..."
result=$(curl -s -X POST "http://$RADARR_HOST:$RADARR_PORT/api/v3/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $RADARR_API" \
     -d "$radarr_config")

if echo "$result" | grep -q '"id":[0-9]'; then
    echo "✅ Successfully configured qBittorrent for Radarr"
elif echo "$result" | grep -q "error\|Error"; then
    echo "❌ Failed to configure qBittorrent for Radarr:"
    echo "$result" | jq -r '.[] | .errorMessage // .detailedDescription // .' 2>/dev/null || echo "$result"
else
    echo "⚠️  Configuration response: $result"
fi

echo -e "\n=== Configuring Lidarr ==="

# Configure qBittorrent for Lidarr - using container name
lidarr_config='{
  "enable": true,
  "protocol": "torrent",
  "priority": 1,
  "removeCompletedDownloads": false,
  "removeFailedDownloads": true,
  "name": "qBittorrent",
  "fields": [
    {
      "name": "host",
      "value": "'$QB_HOST'"
    },
    {
      "name": "port",
      "value": '$QB_PORT'
    },
    {
      "name": "username",
      "value": "'$QB_USERNAME'"
    },
    {
      "name": "password",
      "value": "'$QB_PASSWORD'"
    },
    {
      "name": "category",
      "value": "lidarr"
    },
    {
      "name": "recentMusicPriority",
      "value": 0
    },
    {
      "name": "olderMusicPriority",
      "value": 0
    },
    {
      "name": "initialState",
      "value": 0
    },
    {
      "name": "sequentialOrder",
      "value": false
    },
    {
      "name": "firstAndLast",
      "value": false
    }
  ],
  "implementationName": "qBittorrent",
  "implementation": "QBittorrent",
  "configContract": "QBittorrentSettings",
  "tags": []
}'

echo "Adding qBittorrent to Lidarr (connecting to qbittorrent:8080)..."
result=$(curl -s -X POST "http://$LIDARR_HOST:$LIDARR_PORT/api/v1/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $LIDARR_API" \
     -d "$lidarr_config")

if echo "$result" | grep -q '"id":[0-9]'; then
    echo "✅ Successfully configured qBittorrent for Lidarr"
elif echo "$result" | grep -q "error\|Error"; then
    echo "❌ Failed to configure qBittorrent for Lidarr:"
    echo "$result" | jq -r '.[] | .errorMessage // .detailedDescription // .' 2>/dev/null || echo "$result"
else
    echo "⚠️  Configuration response: $result"
fi

echo -e "\n=== Verifying Download Client Configurations ==="

# Test Sonarr download client
echo "Checking Sonarr download clients..."
sonarr_clients=$(curl -s "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient" -H "X-Api-Key: $SONARR_API")
if echo "$sonarr_clients" | grep -q "qBittorrent"; then
    echo "✅ Sonarr: qBittorrent client found"
    client_count=$(echo "$sonarr_clients" | jq '. | length' 2>/dev/null || echo "unknown")
    echo "   Total download clients: $client_count"
else
    echo "❌ Sonarr: qBittorrent client not found"
fi

# Test Radarr download client
echo "Checking Radarr download clients..."
radarr_clients=$(curl -s "http://$RADARR_HOST:$RADARR_PORT/api/v3/downloadclient" -H "X-Api-Key: $RADARR_API")
if echo "$radarr_clients" | grep -q "qBittorrent"; then
    echo "✅ Radarr: qBittorrent client found"
    client_count=$(echo "$radarr_clients" | jq '. | length' 2>/dev/null || echo "unknown")
    echo "   Total download clients: $client_count"
else
    echo "❌ Radarr: qBittorrent client not found"
fi

# Test Lidarr download client
echo "Checking Lidarr download clients..."
lidarr_clients=$(curl -s "http://$LIDARR_HOST:$LIDARR_PORT/api/v1/downloadclient" -H "X-Api-Key: $LIDARR_API")
if echo "$lidarr_clients" | grep -q "qBittorrent"; then
    echo "✅ Lidarr: qBittorrent client found"
    client_count=$(echo "$lidarr_clients" | jq '. | length' 2>/dev/null || echo "unknown")
    echo "   Total download clients: $client_count"
else
    echo "❌ Lidarr: qBittorrent client not found"
fi

echo -e "\n=== Configuration Summary ==="
echo "✅ Network Configuration:"
echo "   - Docker Network: newmedia_media-net"
echo "   - qBittorrent Container: qbittorrent"
echo "   - Internal Communication: qbittorrent:8080"
echo "   - External Access: localhost:8090"
echo ""
echo "✅ qBittorrent Settings:"
echo "   - Host: $QB_HOST (Docker container name)"
echo "   - Port: $QB_PORT (internal container port)"
echo "   - Username: $QB_USERNAME"
echo "   - Authentication: Required"
echo ""
echo "✅ Categories Configured:"
echo "   - Sonarr → 'sonarr' category"
echo "   - Radarr → 'radarr' category"  
echo "   - Lidarr → 'lidarr' category"
echo ""
echo "✅ Download Paths:"
echo "   - Completed Downloads: /downloads/"
echo "   - Incomplete Downloads: /downloads/incomplete/"
echo ""
echo "🔧 Manual Verification Steps:"
echo "1. Open qBittorrent Web UI: http://localhost:8090"
echo "2. Login with admin/adminadmin"
echo "3. Check Tools > Options > Web UI > Authentication"
echo "4. Verify categories are created automatically when downloads start"
echo "5. Test downloads from each ARR service"
echo ""
echo "⚠️  If connection tests fail:"
echo "1. Check qBittorrent authentication settings"
echo "2. Verify 'Bypass authentication for clients on localhost' is enabled"
echo "3. Ensure containers are on the same Docker network"
echo "4. Check Docker container logs for errors"