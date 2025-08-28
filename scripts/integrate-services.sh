#!/bin/bash

# Automated Service Integration Script
set -euo pipefail

# Load configuration
CONFIG_FILE="/Users/morlock/fun/newmedia/.media-server-config"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Configuration file not found. Run configure-services.sh first."
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Automated Service Integration${NC}"
echo -e "${BLUE}================================================${NC}"

# Configure Sonarr
echo -e "\n${BLUE}Configuring Sonarr...${NC}"

# Add qBittorrent to Sonarr
echo "Adding qBittorrent to Sonarr..."
curl -X POST "${SONARR_URL}/api/v3/downloadclient" \
  -H "X-Api-Key: ${SONARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "enable": true,
    "protocol": "torrent",
    "priority": 1,
    "removeCompletedDownloads": false,
    "removeFailedDownloads": true,
    "name": "qBittorrent",
    "fields": [
      {"name": "host", "value": "qbittorrent"},
      {"name": "port", "value": 8080},
      {"name": "username", "value": "admin"},
      {"name": "password", "value": "adminadmin"},
      {"name": "category", "value": "tv-sonarr"},
      {"name": "recentPriority", "value": 0},
      {"name": "olderPriority", "value": 0},
      {"name": "initialState", "value": 0}
    ],
    "implementation": "QBittorrent",
    "configContract": "QBittorrentSettings"
  }' 2>/dev/null | jq '.' || echo "qBittorrent may already be configured in Sonarr"

# Add root folder to Sonarr
echo "Adding root folder to Sonarr..."
curl -X POST "${SONARR_URL}/api/v3/rootFolder" \
  -H "X-Api-Key: ${SONARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/media/tv",
    "accessible": true,
    "freeSpace": 0,
    "unmappedFolders": []
  }' 2>/dev/null | jq '.' || echo "Root folder may already exist in Sonarr"

echo -e "${GREEN}✅ Sonarr configured${NC}"

# Configure Radarr
echo -e "\n${BLUE}Configuring Radarr...${NC}"

# Add qBittorrent to Radarr
echo "Adding qBittorrent to Radarr..."
curl -X POST "${RADARR_URL}/api/v3/downloadclient" \
  -H "X-Api-Key: ${RADARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "enable": true,
    "protocol": "torrent",
    "priority": 1,
    "removeCompletedDownloads": false,
    "removeFailedDownloads": true,
    "name": "qBittorrent",
    "fields": [
      {"name": "host", "value": "qbittorrent"},
      {"name": "port", "value": 8080},
      {"name": "username", "value": "admin"},
      {"name": "password", "value": "adminadmin"},
      {"name": "category", "value": "movies-radarr"},
      {"name": "recentPriority", "value": 0},
      {"name": "olderPriority", "value": 0},
      {"name": "initialState", "value": 0}
    ],
    "implementation": "QBittorrent",
    "configContract": "QBittorrentSettings"
  }' 2>/dev/null | jq '.' || echo "qBittorrent may already be configured in Radarr"

# Add root folder to Radarr
echo "Adding root folder to Radarr..."
curl -X POST "${RADARR_URL}/api/v3/rootFolder" \
  -H "X-Api-Key: ${RADARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/media/movies",
    "accessible": true,
    "freeSpace": 0,
    "unmappedFolders": []
  }' 2>/dev/null | jq '.' || echo "Root folder may already exist in Radarr"

echo -e "${GREEN}✅ Radarr configured${NC}"

# Configure Prowlarr
echo -e "\n${BLUE}Configuring Prowlarr...${NC}"

# Add Sonarr to Prowlarr
echo "Adding Sonarr application to Prowlarr..."
curl -X POST "${PROWLARR_URL}/api/v1/applications" \
  -H "X-Api-Key: ${PROWLARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sonarr",
    "syncLevel": "fullSync",
    "implementation": "Sonarr",
    "configContract": "SonarrSettings",
    "fields": [
      {"name": "baseUrl", "value": "http://sonarr:8989"},
      {"name": "apiKey", "value": "'"${SONARR_API_KEY}"'"},
      {"name": "syncCategories", "value": [5000, 5030, 5040]}
    ]
  }' 2>/dev/null | jq '.' || echo "Sonarr may already be configured in Prowlarr"

# Add Radarr to Prowlarr
echo "Adding Radarr application to Prowlarr..."
curl -X POST "${PROWLARR_URL}/api/v1/applications" \
  -H "X-Api-Key: ${PROWLARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Radarr",
    "syncLevel": "fullSync",
    "implementation": "Radarr",
    "configContract": "RadarrSettings",
    "fields": [
      {"name": "baseUrl", "value": "http://radarr:7878"},
      {"name": "apiKey", "value": "'"${RADARR_API_KEY}"'"},
      {"name": "syncCategories", "value": [2000, 2010, 2020, 2030, 2040, 2050]}
    ]
  }' 2>/dev/null | jq '.' || echo "Radarr may already be configured in Prowlarr"

# Add a public indexer to Prowlarr (1337x as example)
echo "Adding 1337x indexer to Prowlarr..."
curl -X POST "${PROWLARR_URL}/api/v1/indexer" \
  -H "X-Api-Key: ${PROWLARR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "1337x",
    "enable": true,
    "redirect": false,
    "priority": 25,
    "appProfileId": 1,
    "protocol": "torrent",
    "privacy": "public",
    "configContract": "CardigannSettings",
    "implementation": "Cardigann",
    "fields": [
      {"name": "definitionFile", "value": "/definitions/1337x.yml"}
    ]
  }' 2>/dev/null | jq '.' || echo "1337x indexer may already exist"

echo -e "${GREEN}✅ Prowlarr configured${NC}"

# Test connections
echo -e "\n${BLUE}Testing connections...${NC}"

# Test Sonarr
if curl -s "${SONARR_URL}/api/v3/system/status" -H "X-Api-Key: ${SONARR_API_KEY}" | jq -e '.version' > /dev/null; then
    echo -e "${GREEN}✅ Sonarr API working${NC}"
else
    echo -e "${YELLOW}⚠️  Sonarr API not responding${NC}"
fi

# Test Radarr
if curl -s "${RADARR_URL}/api/v3/system/status" -H "X-Api-Key: ${RADARR_API_KEY}" | jq -e '.version' > /dev/null; then
    echo -e "${GREEN}✅ Radarr API working${NC}"
else
    echo -e "${YELLOW}⚠️  Radarr API not responding${NC}"
fi

# Test Prowlarr
if curl -s "${PROWLARR_URL}/api/v1/system/status" -H "X-Api-Key: ${PROWLARR_API_KEY}" | jq -e '.version' > /dev/null; then
    echo -e "${GREEN}✅ Prowlarr API working${NC}"
else
    echo -e "${YELLOW}⚠️  Prowlarr API not responding${NC}"
fi

echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}   Integration Complete!${NC}"
echo -e "${BLUE}================================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Visit Prowlarr (${PROWLARR_URL}) and add more indexers"
echo -e "2. Search for a TV show in Sonarr (${SONARR_URL})"
echo -e "3. Search for a movie in Radarr (${RADARR_URL})"
echo -e "4. Monitor downloads in qBittorrent (${QBITTORRENT_URL})"
echo -e "5. Complete Jellyfin setup (${JELLYFIN_URL})"

echo -e "\n${GREEN}The media server stack is now integrated and ready to use!${NC}"