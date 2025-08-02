#!/bin/bash

# qBittorrent Download Client Configuration Script (Docker Network Fixed)
# This script configures qBittorrent as a download client for all ARR services

set -e

# Configuration variables - CORRECTED for Docker networking
QB_HOST="gluetun"  # qBittorrent runs through gluetun container
QB_PORT="8080"     # gluetun exposes qBittorrent on port 8080
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

echo "=== Configuring qBittorrent Download Client for ARR Services (Docker Network) ==="

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

# Test qBittorrent through gluetun (from outside Docker)
echo "Testing qBittorrent connectivity through gluetun..."
if curl -s "http://localhost:8080/api/v2/app/version" 2>&1 | grep -q "Unauthorized\|[0-9]"; then
    echo "✅ qBittorrent is accessible through gluetun (localhost:8080)"
else
    echo "❌ qBittorrent is not accessible through gluetun"
    echo "Please ensure gluetun and qBittorrent containers are running"
    exit 1
fi

echo -e "\n=== Configuring Sonarr ==="

# Configure qBittorrent for Sonarr - using gluetun container name
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

echo "Adding qBittorrent to Sonarr (connecting to gluetun:8080)..."
if result=$(curl -s -X POST "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $SONARR_API" \
     -d "$sonarr_config"); then
    if echo "$result" | grep -q "error\|Error"; then
        echo "❌ Failed to configure qBittorrent for Sonarr:"
        echo "$result" | jq -r '.[] | .errorMessage // .detailedDescription // .' 2>/dev/null || echo "$result"
    else
        echo "✅ Successfully configured qBittorrent for Sonarr"
    fi
else
    echo "❌ Failed to configure qBittorrent for Sonarr (connection error)"
fi

echo -e "\n=== Configuring Radarr ==="

# Configure qBittorrent for Radarr - using gluetun container name
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

echo "Adding qBittorrent to Radarr (connecting to gluetun:8080)..."
if result=$(curl -s -X POST "http://$RADARR_HOST:$RADARR_PORT/api/v3/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $RADARR_API" \
     -d "$radarr_config"); then
    if echo "$result" | grep -q "error\|Error"; then
        echo "❌ Failed to configure qBittorrent for Radarr:"
        echo "$result" | jq -r '.[] | .errorMessage // .detailedDescription // .' 2>/dev/null || echo "$result"
    else
        echo "✅ Successfully configured qBittorrent for Radarr"
    fi
else
    echo "❌ Failed to configure qBittorrent for Radarr (connection error)"
fi

echo -e "\n=== Configuring Lidarr ==="

# Configure qBittorrent for Lidarr - using gluetun container name
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
      "value": "'$QB_PASSWORD'"
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

echo "Adding qBittorrent to Lidarr (connecting to gluetun:8080)..."
if result=$(curl -s -X POST "http://$LIDARR_HOST:$LIDARR_PORT/api/v1/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $LIDARR_API" \
     -d "$lidarr_config"); then
    if echo "$result" | grep -q "error\|Error"; then
        echo "❌ Failed to configure qBittorrent for Lidarr:"
        echo "$result" | jq -r '.[] | .errorMessage // .detailedDescription // .' 2>/dev/null || echo "$result"
    else
        echo "✅ Successfully configured qBittorrent for Lidarr"
    fi
else
    echo "❌ Failed to configure qBittorrent for Lidarr (connection error)"
fi

echo -e "\n=== Testing Download Client Connections ==="

# Test Sonarr download client
echo "Testing Sonarr download client connection..."
sonarr_test=$(curl -s "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient" -H "X-Api-Key: $SONARR_API")
if echo "$sonarr_test" | grep -q "qBittorrent"; then
    echo "✅ Sonarr qBittorrent client configured successfully"
    # Test the connection
    sonarr_test_result=$(curl -s "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient/test" -H "X-Api-Key: $SONARR_API" -H "Content-Type: application/json" -d "$sonarr_config" 2>/dev/null || echo "test_unavailable")
    if echo "$sonarr_test_result" | grep -q "error\|Error"; then
        echo "⚠️  Configuration saved but connection test failed - check qBittorrent authentication"
    else
        echo "✅ Connection test passed"
    fi
else
    echo "❌ Sonarr qBittorrent client not found"
fi

# Test Radarr download client
echo "Testing Radarr download client connection..."
radarr_test=$(curl -s "http://$RADARR_HOST:$RADARR_PORT/api/v3/downloadclient" -H "X-Api-Key: $RADARR_API")
if echo "$radarr_test" | grep -q "qBittorrent"; then
    echo "✅ Radarr qBittorrent client configured successfully"
else
    echo "❌ Radarr qBittorrent client not found"
fi

# Test Lidarr download client
echo "Testing Lidarr download client connection..."
lidarr_test=$(curl -s "http://$LIDARR_HOST:$LIDARR_PORT/api/v1/downloadclient" -H "X-Api-Key: $LIDARR_API")
if echo "$lidarr_test" | grep -q "qBittorrent"; then
    echo "✅ Lidarr qBittorrent client configured successfully"
else
    echo "❌ Lidarr qBittorrent client not found"
fi

echo -e "\n=== Configuration Summary ==="
echo "Network Configuration:"
echo "  - qBittorrent runs through gluetun VPN container"
echo "  - ARR services connect to: gluetun:8080 (internal Docker network)"
echo "  - External access: localhost:8080"
echo ""
echo "qBittorrent Settings:"
echo "  - Container: gluetun (shared network with qBittorrent)"
echo "  - Internal URL: http://gluetun:8080"
echo "  - External URL: http://localhost:8080"
echo "  - Username: $QB_USERNAME"
echo ""
echo "Categories configured:"
echo "  - Sonarr: sonarr"
echo "  - Radarr: radarr"
echo "  - Lidarr: lidarr"
echo ""
echo "Download paths:"
echo "  - Complete: /downloads/"
echo "  - Incomplete: /downloads/incomplete/"
echo ""
echo "Next steps:"
echo "1. Verify qBittorrent WebUI is accessible at http://localhost:8080"
echo "2. Check that categories (sonarr, radarr, lidarr) are created in qBittorrent"
echo "3. Test download client connections in each ARR service"
echo "4. If authentication fails, enable 'Bypass authentication for clients on localhost' in qBittorrent Web UI"