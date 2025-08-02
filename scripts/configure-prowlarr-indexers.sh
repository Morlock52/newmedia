#!/bin/bash

# Prowlarr Indexer Configuration Script
# This script adds popular free indexers to Prowlarr

# Configuration
PROWLARR_URL="http://localhost:9696"
API_KEY="${PROWLARR_API_KEY:-aad98e12668341e6a11630c125ab846e}"

echo "Prowlarr Indexer Configuration Script"
echo "====================================="
echo "Using API Key: $API_KEY"
echo ""

# Function to add an indexer
add_indexer() {
    local name="$1"
    local implementation="$2"
    local settings="$3"
    
    echo "Adding indexer: $name"
    
    curl -s -X POST "$PROWLARR_URL/api/v1/indexer" \
        -H "X-Api-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$name\",
            \"implementation\": \"$implementation\",
            \"configContract\": \"${implementation}IndexerSettings\",
            \"enable\": true,
            \"protocol\": \"torrent\",
            \"priority\": 25,
            \"fields\": $settings
        }"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully added $name"
    else
        echo "✗ Failed to add $name"
    fi
    echo ""
}

# Check API connectivity
echo "Checking Prowlarr API connectivity..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Api-Key: $API_KEY" "$PROWLARR_URL/api/v1/system/status")

if [ "$STATUS" != "200" ]; then
    echo "❌ Cannot connect to Prowlarr API. Status code: $STATUS"
    echo ""
    echo "Please ensure:"
    echo "1. Prowlarr is running at $PROWLARR_URL"
    echo "2. The API key is correct: $API_KEY"
    echo "3. Authentication is properly configured"
    exit 1
fi

echo "✓ API connection successful"
echo ""

# Get available indexer schemas
echo "Fetching available indexer schemas..."
SCHEMAS=$(curl -s -H "X-Api-Key: $API_KEY" "$PROWLARR_URL/api/v1/indexer/schema")

if [ -z "$SCHEMAS" ]; then
    echo "❌ Could not fetch indexer schemas"
    exit 1
fi

# Add indexers based on available schemas
echo "Adding free indexers..."
echo ""

# Note: The exact implementation names and field requirements may vary by Prowlarr version
# These are examples based on common configurations

# 1337x
add_indexer "1337x" "Cardigann" '[
    {"name": "definitionFile", "value": "1337x"},
    {"name": "baseUrl", "value": "https://1337x.to"}
]'

# The Pirate Bay
add_indexer "The Pirate Bay" "Cardigann" '[
    {"name": "definitionFile", "value": "thepiratebay"},
    {"name": "baseUrl", "value": "https://thepiratebay.org"}
]'

# YTS
add_indexer "YTS" "Cardigann" '[
    {"name": "definitionFile", "value": "yts"},
    {"name": "baseUrl", "value": "https://yts.mx"}
]'

# EZTV
add_indexer "EZTV" "Cardigann" '[
    {"name": "definitionFile", "value": "eztv"},
    {"name": "baseUrl", "value": "https://eztv.re"}
]'

# LimeTorrents
add_indexer "LimeTorrents" "Cardigann" '[
    {"name": "definitionFile", "value": "limetorrents"},
    {"name": "baseUrl", "value": "https://www.limetorrents.lol"}
]'

# TorrentGalaxy
add_indexer "TorrentGalaxy" "Cardigann" '[
    {"name": "definitionFile", "value": "torrentgalaxy"},
    {"name": "baseUrl", "value": "https://torrentgalaxy.to"}
]'

echo ""
echo "Configuration complete!"
echo ""
echo "To view configured indexers:"
echo "curl -s -H \"X-Api-Key: $API_KEY\" \"$PROWLARR_URL/api/v1/indexer\" | jq '.[] | {name: .name, enable: .enable}'"