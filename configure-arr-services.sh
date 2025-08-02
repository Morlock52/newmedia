#!/bin/bash

# API Keys
PROWLARR_API_KEY="c272eec1ce1447f1b119daade4b0e268"
SONARR_API_KEY="becf19261b1a4ed0b4f92de90eaf3015"
RADARR_API_KEY="f0acaf0200034ee69215482f212cda5a"
LIDARR_API_KEY="47b87e7aec5646c898d60b728f6126e0"

# Base URLs
PROWLARR_URL="http://localhost:9696"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"
LIDARR_URL="http://localhost:8686"

echo "Configuring Prowlarr connections to other *arr services..."

# Add Sonarr to Prowlarr
echo "Adding Sonarr to Prowlarr..."
curl -X POST "$PROWLARR_URL/api/v1/applications" \
  -H "X-Api-Key: $PROWLARR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sonarr",
    "syncLevel": "fullSync",
    "implementation": "Sonarr",
    "configContract": "SonarrSettings",
    "implementationName": "Sonarr",
    "fields": [
      {
        "name": "prowlarrUrl",
        "value": "http://prowlarr:9696"
      },
      {
        "name": "baseUrl",
        "value": "http://sonarr:8989"
      },
      {
        "name": "apiKey",
        "value": "'$SONARR_API_KEY'"
      },
      {
        "name": "syncCategories",
        "value": [5000, 5010, 5020, 5030, 5040, 5045, 5050]
      }
    ],
    "tags": []
  }' 2>/dev/null || echo "Sonarr might already be configured"

# Add Radarr to Prowlarr
echo "Adding Radarr to Prowlarr..."
curl -X POST "$PROWLARR_URL/api/v1/applications" \
  -H "X-Api-Key: $PROWLARR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Radarr",
    "syncLevel": "fullSync",
    "implementation": "Radarr",
    "configContract": "RadarrSettings",
    "implementationName": "Radarr",
    "fields": [
      {
        "name": "prowlarrUrl",
        "value": "http://prowlarr:9696"
      },
      {
        "name": "baseUrl",
        "value": "http://radarr:7878"
      },
      {
        "name": "apiKey",
        "value": "'$RADARR_API_KEY'"
      },
      {
        "name": "syncCategories",
        "value": [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060]
      }
    ],
    "tags": []
  }' 2>/dev/null || echo "Radarr might already be configured"

# Add Lidarr to Prowlarr
echo "Adding Lidarr to Prowlarr..."
curl -X POST "$PROWLARR_URL/api/v1/applications" \
  -H "X-Api-Key: $PROWLARR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Lidarr",
    "syncLevel": "fullSync",
    "implementation": "Lidarr",
    "configContract": "LidarrSettings",
    "implementationName": "Lidarr",
    "fields": [
      {
        "name": "prowlarrUrl",
        "value": "http://prowlarr:9696"
      },
      {
        "name": "baseUrl",
        "value": "http://lidarr:8686"
      },
      {
        "name": "apiKey",
        "value": "'$LIDARR_API_KEY'"
      },
      {
        "name": "syncCategories",
        "value": [3000, 3010, 3020, 3030, 3040]
      }
    ],
    "tags": []
  }' 2>/dev/null || echo "Lidarr might already be configured"

echo ""
echo "Configuration complete!"
echo ""
echo "Service URLs and API Keys:"
echo "=========================="
echo "Prowlarr: $PROWLARR_URL (API: $PROWLARR_API_KEY)"
echo "Sonarr:   $SONARR_URL (API: $SONARR_API_KEY)"
echo "Radarr:   $RADARR_URL (API: $RADARR_API_KEY)"
echo "Lidarr:   $LIDARR_URL (API: $LIDARR_API_KEY)"
echo ""
echo "You can now access the services and complete any additional setup through their web interfaces."