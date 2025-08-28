#!/bin/bash
# Jellyfin Authentication Verification Script

JELLYFIN_URL="http://localhost:8096"

echo "🔍 Verifying Jellyfin Authentication Setup..."

# Test basic connectivity
if curl -s --connect-timeout 5 "$JELLYFIN_URL/health" > /dev/null; then
    echo "✅ Jellyfin is accessible"
else
    echo "❌ Jellyfin is not accessible"
    exit 1
fi

# Test public endpoints
if curl -s --connect-timeout 5 "$JELLYFIN_URL/System/Info/Public" > /dev/null; then
    echo "✅ Public API endpoints working"
else
    echo "❌ Public API endpoints not working"
    exit 1
fi

# Check if API key file exists
if [ -f "./scripts/jellyfin-api-key.txt" ]; then
    echo "✅ API key file found"
else
    echo "⚠️  API key file not found - may need to create one manually"
fi

# Check if config file exists
if [ -f "./scripts/jellyfin-api-config.json" ]; then
    echo "✅ API configuration file found"
else
    echo "⚠️  API configuration file not found"
fi

echo "🎉 Jellyfin authentication verification completed!"
