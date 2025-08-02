#!/bin/bash

# Media Server Auto-Configuration Script
# This script helps complete the setup of Jellyfin and Plex

set -e

echo "🚀 Media Server Auto-Configuration"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if containers are running
echo "Checking container status..."

if ! docker ps | grep -q "jellyfin"; then
    print_error "Jellyfin container is not running"
    exit 1
else
    print_status "Jellyfin container is running"
fi

if ! docker ps | grep -q "plex"; then
    print_error "Plex container is not running"
    exit 1
else
    print_status "Plex container is running"
fi

# Check media access
echo -e "\nChecking media access..."

JELLYFIN_MOVIES=$(docker exec jellyfin find /media/Movies -maxdepth 1 -type d | wc -l)
JELLYFIN_TV=$(docker exec jellyfin find /media/TV -maxdepth 1 -type d | wc -l)
JELLYFIN_MUSIC=$(docker exec jellyfin find /media/Music -maxdepth 1 -type d | wc -l)

print_status "Jellyfin can access:"
print_info "  Movies: $((JELLYFIN_MOVIES-1)) folders"
print_info "  TV Shows: $((JELLYFIN_TV-1)) folders" 
print_info "  Music: $((JELLYFIN_MUSIC-1)) folders"

PLEX_MOVIES=$(docker exec plex find /media/Movies -maxdepth 1 -type d | wc -l)
PLEX_TV=$(docker exec plex find /media/TV -maxdepth 1 -type d | wc -l)
PLEX_MUSIC=$(docker exec plex find /media/Music -maxdepth 1 -type d | wc -l)

print_status "Plex can access:"
print_info "  Movies: $((PLEX_MOVIES-1)) folders"
print_info "  TV Shows: $((PLEX_TV-1)) folders"
print_info "  Music: $((PLEX_MUSIC-1)) folders"

# Check service accessibility
echo -e "\nChecking service accessibility..."

JELLYFIN_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8096)
PLEX_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:32400)

if [ "$JELLYFIN_STATUS" = "302" ]; then
    print_warning "Jellyfin setup wizard needs completion"
    print_info "  Access: http://localhost:8096"
elif [ "$JELLYFIN_STATUS" = "200" ]; then
    print_status "Jellyfin is configured and ready"
else
    print_error "Jellyfin returned status code: $JELLYFIN_STATUS"
fi

if [ "$PLEX_STATUS" = "401" ]; then
    print_warning "Plex authentication required"
    print_info "  Access: http://localhost:32400/web"
elif [ "$PLEX_STATUS" = "200" ]; then
    print_status "Plex is configured and ready"
else
    print_error "Plex returned status code: $PLEX_STATUS"
fi

# Generate setup URLs
echo -e "\n🔧 Setup URLs:"
echo "==============="
echo "Jellyfin: http://localhost:8096"
echo "Plex:     http://localhost:32400/web"

# Media library configuration
echo -e "\n📁 Media Library Paths (for setup):"
echo "===================================="
echo "Movies:    /media/Movies"
echo "TV Shows:  /media/TV" 
echo "Music:     /media/Music"

# Sample media for testing
echo -e "\n🎬 Sample Media Available:"
echo "=========================="

echo "Movies (first 5):"
docker exec jellyfin find /media/Movies -maxdepth 1 -type d -name "*" | head -6 | tail -5 | sed 's|/media/Movies/||' | sed 's/^/  - /'

echo -e "\nTV Shows (first 5):"
docker exec plex find /media/TV -maxdepth 1 -type d -name "*" | head -6 | tail -5 | sed 's|/media/TV/||' | sed 's/^/  - /'

echo -e "\nMusic Artists (first 5):"
docker exec jellyfin find /media/Music -maxdepth 1 -type d -name "*" | head -6 | tail -5 | sed 's|/media/Music/||' | sed 's/^/  - /'

# Next steps
echo -e "\n📋 Next Steps:"
echo "=============="
echo "1. Open Jellyfin: http://localhost:8096"
echo "   - Complete setup wizard"
echo "   - Add media libraries using paths above"
echo ""
echo "2. Open Plex: http://localhost:32400/web"
echo "   - Sign in with Plex account"
echo "   - Add media libraries using paths above"
echo ""
echo "3. Wait for initial media scan to complete"
echo "4. Test playback on both platforms"

# Optional optimizations
echo -e "\n⚡ Optional Optimizations:"
echo "========================"
echo "- Enable hardware transcoding (Intel QuickSync available)"
echo "- Configure remote access (if desired)"
echo "- Install mobile apps (Jellyfin & Plex)"
echo "- Set up user accounts for family members"

# Storage information
TOTAL_SIZE=$(df -h /Volumes/Plex | awk 'NR==2 {print $2}')
AVAILABLE_SIZE=$(df -h /Volumes/Plex | awk 'NR==2 {print $4}')
USED_PERCENT=$(df -h /Volumes/Plex | awk 'NR==2 {print $5}')

echo -e "\n💾 Storage Information:"
echo "======================"
echo "Total Space:     $TOTAL_SIZE"
echo "Available Space: $AVAILABLE_SIZE"
echo "Used:           $USED_PERCENT"

if [ "${USED_PERCENT%?}" -gt 90 ]; then
    print_warning "Storage is over 90% full"
elif [ "${USED_PERCENT%?}" -gt 80 ]; then
    print_info "Storage is over 80% full - consider cleanup"
else
    print_status "Storage has adequate free space"
fi

echo -e "\n🎉 Setup Status: Ready for final configuration!"
echo "Visit the URLs above to complete setup."