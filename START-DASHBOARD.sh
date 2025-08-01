#!/bin/bash

# 🎬 BULLETPROOF MEDIA SERVER DASHBOARD LAUNCHER
# ==============================================
# This is your ONE-CLICK solution to access your media server
# Compatible with: macOS, Linux, Windows (Git Bash)

set -e

echo "🎬 MEDIA SERVER DASHBOARD LAUNCHER 2025"
echo "========================================"
echo ""

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${!1}%s${NC}\n" "$2"
}

# Function to check if a service is running
check_service() {
    local port=$1
    local name=$2
    
    if command -v curl >/dev/null 2>&1; then
        if curl -s --max-time 3 "http://localhost:${port}" >/dev/null 2>&1; then
            return 0
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -q --timeout=3 --tries=1 --spider "http://localhost:${port}" >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to open URL in browser
open_browser() {
    local url=$1
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$url"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url"
        elif command -v firefox >/dev/null 2>&1; then
            firefox "$url" &
        elif command -v google-chrome >/dev/null 2>&1; then
            google-chrome "$url" &
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash, Cygwin)
        start "$url"
    else
        print_color YELLOW "Please manually open: $url"
        return 1
    fi
    return 0
}

print_color CYAN "🔍 Checking for available dashboards..."
echo ""

# Method 1: Check for Docker Homepage service (preferred)
if check_service 3001 "Homepage"; then
    print_color GREEN "✅ Found Homepage Dashboard Service (port 3001)"
    print_color BLUE "   🚀 This is your main dashboard with real-time monitoring"
    
    if open_browser "http://localhost:3001"; then
        print_color GREEN "🎉 Homepage Dashboard opened successfully!"
        echo ""
        print_color CYAN "📋 What you'll see:"
        echo "   • Real-time service status"
        echo "   • One-click access to all services"
        echo "   • System monitoring widgets"
        echo "   • Professional interface"
        echo ""
        exit 0
    fi
fi

# Method 2: Check for static smart dashboard
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMART_DASHBOARD="$SCRIPT_DIR/smart-dashboard.html"

if [[ -f "$SMART_DASHBOARD" ]]; then
    print_color GREEN "✅ Found Smart Dashboard (HTML file)"
    print_color BLUE "   📱 This is your backup dashboard with health checking"
    
    if open_browser "file://$SMART_DASHBOARD"; then
        print_color GREEN "🎉 Smart Dashboard opened successfully!"
        echo ""
        print_color CYAN "📋 What you'll see:"
        echo "   • Service health indicators"
        echo "   • Mobile-friendly design"
        echo "   • Direct service links"
        echo "   • Offline capability"
        echo ""
        exit 0
    fi
fi

# Method 3: Check for optimized static dashboard
OPTIMIZED_DASHBOARD="$SCRIPT_DIR/service-access-optimized.html"

if [[ -f "$OPTIMIZED_DASHBOARD" ]]; then
    print_color GREEN "✅ Found Optimized Dashboard (HTML file)"
    print_color BLUE "   ⚡ This is your high-performance static dashboard"
    
    if open_browser "file://$OPTIMIZED_DASHBOARD"; then
        print_color GREEN "🎉 Optimized Dashboard opened successfully!"
        echo ""
        print_color CYAN "📋 What you'll see:"
        echo "   • Beautiful modern interface"
        echo "   • Fast loading times"
        echo "   • All service links"
        echo "   • 2025 optimized design"
        echo ""
        exit 0
    fi
fi

# Method 4: Check for basic dashboard
BASIC_DASHBOARD="$SCRIPT_DIR/service-access.html"

if [[ -f "$BASIC_DASHBOARD" ]]; then
    print_color YELLOW "⚠️ Found Basic Dashboard (HTML file)"
    print_color BLUE "   📄 This is your basic service listing"
    
    if open_browser "file://$BASIC_DASHBOARD"; then
        print_color GREEN "✅ Basic Dashboard opened!"
        echo ""
        print_color CYAN "📋 What you'll see:"
        echo "   • Simple service list"
        echo "   • Basic functionality"
        echo "   • All service links"
        echo ""
        exit 0
    fi
fi

# No dashboard found - provide helpful guidance
print_color RED "❌ No dashboard found!"
echo ""
print_color YELLOW "🔧 TROUBLESHOOTING STEPS:"
echo ""
echo "1. 📦 CHECK DOCKER SERVICES:"
echo "   Run: docker ps"
echo "   Look for: homepage, jellyfin, qbittorrent"
echo ""
echo "2. 🚀 START SERVICES:"
echo "   Run: docker-compose up -d"
echo "   Or:  ./deploy-optimized.sh"
echo ""
echo "3. 🌐 MANUAL ACCESS:"
echo "   • Jellyfin:     http://localhost:8096"
echo "   • qBittorrent:  http://localhost:8080"
echo "   • Portainer:    http://localhost:9000"
echo "   • Homepage:     http://localhost:3001"
echo ""
echo "4. ⏰ WAIT TIME:"
echo "   Services can take 2-5 minutes to fully start"
echo ""
echo "5. 📞 NEED HELP?"
echo "   Check the README.md file in this folder"
echo ""

print_color PURPLE "💡 TIP: Bookmark this script for easy access!"
print_color CYAN "🔄 Run this script again after starting services"

exit 1