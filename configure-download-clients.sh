#!/bin/bash

# qBittorrent Download Client Configuration Script
# This script configures qBittorrent as a download client for all ARR services

set -e

# Configuration variables
QB_HOST="localhost"
QB_PORT="8090"
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

# Test qBittorrent connectivity
echo "Testing qBittorrent connectivity..."
if curl -s -f "http://$QB_HOST:$QB_PORT/api/v2/app/version" > /dev/null 2>&1; then
    echo "✅ qBittorrent is accessible"
elif curl -s "http://$QB_HOST:$QB_PORT/api/v2/app/version" 2>&1 | grep -q "Unauthorized"; then
    echo "⚠️  qBittorrent requires authentication (this is expected)"
else
    echo "❌ qBittorrent is not accessible"
    exit 1
fi

# Test ARR services
test_service "Sonarr" "$SONARR_HOST" "$SONARR_PORT" "$SONARR_API" "v3" || exit 1
test_service "Radarr" "$RADARR_HOST" "$RADARR_PORT" "$RADARR_API" "v3" || exit 1
test_service "Lidarr" "$LIDARR_HOST" "$LIDARR_PORT" "$LIDARR_API" "v1" || exit 1

echo -e "\n=== Configuring Sonarr ==="

# Configure qBittorrent for Sonarr
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

echo "Adding qBittorrent to Sonarr..."
if curl -s -X POST "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $SONARR_API" \
     -d "$sonarr_config" > /dev/null; then
    echo "✅ Successfully configured qBittorrent for Sonarr"
else
    echo "❌ Failed to configure qBittorrent for Sonarr"
fi

echo -e "\n=== Configuring Radarr ==="

# Configure qBittorrent for Radarr
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

echo "Adding qBittorrent to Radarr..."
if curl -s -X POST "http://$RADARR_HOST:$RADARR_PORT/api/v3/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $RADARR_API" \
     -d "$radarr_config" > /dev/null; then
    echo "✅ Successfully configured qBittorrent for Radarr"
else
    echo "❌ Failed to configure qBittorrent for Radarr"
fi

echo -e "\n=== Configuring Lidarr ==="

# Configure qBittorrent for Lidarr
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

echo "Adding qBittorrent to Lidarr..."
if curl -s -X POST "http://$LIDARR_HOST:$LIDARR_PORT/api/v1/downloadclient" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $LIDARR_API" \
     -d "$lidarr_config" > /dev/null; then
    echo "✅ Successfully configured qBittorrent for Lidarr"
else
    echo "❌ Failed to configure qBittorrent for Lidarr"
fi

echo -e "\n=== Testing Download Client Connections ==="

# Test Sonarr download client
echo "Testing Sonarr download client connection..."
sonarr_test=$(curl -s "http://$SONARR_HOST:$SONARR_PORT/api/v3/downloadclient" -H "X-Api-Key: $SONARR_API")
if echo "$sonarr_test" | grep -q "qBittorrent"; then
    echo "✅ Sonarr qBittorrent client configured successfully"
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
echo "qBittorrent URL: http://$QB_HOST:$QB_PORT"
echo "Username: $QB_USERNAME"
echo "Categories:"
echo "  - Sonarr: sonarr"
echo "  - Radarr: radarr"
echo "  - Lidarr: lidarr"
echo ""
echo "Download paths:"
echo "  - Complete: /downloads/"
echo "  - Incomplete: /downloads/incomplete/"
echo ""
echo "Configuration complete! Check the ARR services web interfaces to verify the download clients are working."
echo ""
echo "If qBittorrent authentication fails, you may need to:"
echo "1. Enable 'Bypass authentication for clients on localhost' in qBittorrent settings"
echo "2. Or configure the correct username/password in qBittorrent Web UI settings"