#!/bin/bash

# Plex Volume Structure Test Script
# Tests accessibility, permissions, and structure compliance

echo "=== PLEX VOLUME STRUCTURE TEST ==="
echo "Date: $(date)"
echo "User: $(whoami) ($(id))"
echo ""

# Test volume accessibility
echo "1. Testing volume accessibility..."
if mountpoint -q /Volumes/Plex 2>/dev/null || df /Volumes/Plex >/dev/null 2>&1; then
    echo "✅ Plex volume is mounted and accessible"
    df -h /Volumes/Plex | tail -1
else
    echo "❌ Plex volume is not accessible"
    exit 1
fi

echo ""

# Test folder structure
echo "2. Testing TRaSH folder structure..."
REQUIRED_FOLDERS=(
    "/Volumes/Plex/data"
    "/Volumes/Plex/data/torrents"
    "/Volumes/Plex/data/torrents/movies"
    "/Volumes/Plex/data/torrents/tv"
    "/Volumes/Plex/data/torrents/music"
    "/Volumes/Plex/data/torrents/books"
    "/Volumes/Plex/data/media"
    "/Volumes/Plex/data/media/movies"
    "/Volumes/Plex/data/media/tv"
    "/Volumes/Plex/data/media/music"
    "/Volumes/Plex/data/media/books"
    "/Volumes/Plex/data/downloads"
)

MISSING_FOLDERS=()
for folder in "${REQUIRED_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        echo "✅ $folder"
    else
        echo "❌ $folder (MISSING)"
        MISSING_FOLDERS+=("$folder")
    fi
done

if [ ${#MISSING_FOLDERS[@]} -eq 0 ]; then
    echo "✅ All required folders exist"
else
    echo "❌ Missing ${#MISSING_FOLDERS[@]} folders"
fi

echo ""

# Test write permissions
echo "3. Testing write permissions..."
TEST_FILE="/Volumes/Plex/data/test_permissions_$(date +%s).txt"
if touch "$TEST_FILE" 2>/dev/null; then
    echo "✅ Write permission test passed"
    rm "$TEST_FILE" 2>/dev/null
else
    echo "❌ Write permission test failed"
fi

echo ""

# Test docker-compose backup
echo "4. Checking docker-compose backup..."
if [ -f "/Users/morlock/fun/newmedia/docker-compose-demo.yml.backup" ]; then
    echo "✅ docker-compose-demo.yml.backup exists"
    echo "   Original size: $(wc -c < /Users/morlock/fun/newmedia/docker-compose-demo.yml.backup) bytes"
    echo "   Updated size:  $(wc -c < /Users/morlock/fun/newmedia/docker-compose-demo.yml) bytes"
else
    echo "❌ docker-compose backup not found"
fi

echo ""

# Show volume usage
echo "5. Volume usage summary..."
echo "Total space: $(df -h /Volumes/Plex | awk 'NR==2 {print $2}')"
echo "Used space:  $(df -h /Volumes/Plex | awk 'NR==2 {print $3}')"
echo "Free space:  $(df -h /Volumes/Plex | awk 'NR==2 {print $4}')"
echo "Usage:       $(df -h /Volumes/Plex | awk 'NR==2 {print $5}')"

echo ""

# Show existing data
echo "6. Existing data summary..."
echo "Legacy Torrents folder:"
if [ -d "/Volumes/Plex/Torrents" ]; then
    find /Volumes/Plex/Torrents -maxdepth 2 -type d | head -10
else
    echo "  Not found"
fi

echo ""
echo "Legacy Media folder:"
if [ -d "/Volumes/Plex/Media" ]; then
    find /Volumes/Plex/Media -maxdepth 2 -type d | head -10
else
    echo "  Not found"
fi

echo ""
echo "=== TEST COMPLETE ==="

if [ ${#MISSING_FOLDERS[@]} -eq 0 ] && mountpoint -q /Volumes/Plex 2>/dev/null; then
    echo "✅ All tests passed - Ready for docker-compose deployment"
    exit 0
else
    echo "❌ Some tests failed - Review issues before deployment"
    exit 1
fi