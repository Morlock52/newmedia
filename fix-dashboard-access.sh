#!/bin/bash
# Dashboard Access Fix Script - Media Server 2025
# Fixes common dashboard accessibility issues for beginners

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Dashboard Access Fix Script 2025               ║"
echo "║                 Fixing Common Issues                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Function to check if a port is accessible
check_port() {
    local port=$1
    local service_name=$2
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -q "200\|302\|401"; then
        print_status "$service_name (port $port) is accessible"
        return 0
    else
        print_error "$service_name (port $port) is not accessible"
        return 1
    fi
}

# Function to test all key services
test_services() {
    echo "Testing service accessibility..."
    echo
    
    local all_good=true
    
    # Test core services
    if ! check_port 8096 "Jellyfin"; then all_good=false; fi
    if ! check_port 3001 "Homepage Dashboard"; then all_good=false; fi
    if ! check_port 9000 "Portainer"; then all_good=false; fi
    if ! check_port 8080 "qBittorrent"; then all_good=false; fi
    
    # Test additional services
    check_port 3000 "Grafana" || true
    check_port 7878 "Radarr" || true
    check_port 8989 "Sonarr" || true
    check_port 9696 "Prowlarr" || true
    check_port 8081 "SABnzbd" || true
    
    echo
    if $all_good; then
        print_status "All core services are accessible!"
    else
        print_warning "Some services may need attention"
    fi
}

# Function to fix Homepage issues
fix_homepage() {
    echo "Fixing Homepage Dashboard issues..."
    
    # Check if Homepage container is running
    if ! docker ps | grep -q "homepage"; then
        print_error "Homepage container is not running"
        print_info "Starting Homepage container..."
        docker-compose -f docker-compose-optimized.yml up -d homepage
        sleep 5
    fi
    
    # Restart Homepage to apply configuration changes
    print_info "Restarting Homepage with updated configuration..."
    docker restart homepage
    sleep 5
    
    # Test Homepage accessibility
    if check_port 3001 "Homepage Dashboard"; then
        print_status "Homepage Dashboard is now accessible at http://localhost:3001"
    else
        print_warning "Homepage may still have issues. Check logs with: docker logs homepage"
    fi
}

# Function to start the simple dashboard server
start_dashboard_server() {
    echo "Starting Simple Dashboard Server..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 to use the dashboard server."
        return 1
    fi
    
    # Check if the dashboard server script exists
    if [ ! -f "dashboard-server.py" ]; then
        print_error "Dashboard server script not found in current directory"
        return 1
    fi
    
    # Kill any existing dashboard server on port 8888
    if lsof -i :8888 &> /dev/null; then
        print_info "Stopping existing dashboard server on port 8888..."
        pkill -f "dashboard-server.py" || true
        sleep 2
    fi
    
    print_info "Starting dashboard server on http://127.0.0.1:8888"
    print_info "Press Ctrl+C to stop the server when done"
    echo
    python3 dashboard-server.py
}

# Function to show access options
show_access_options() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                     Dashboard Access Options                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo
    echo "Choose your preferred access method:"
    echo
    echo "1. 🚀 Simple Dashboard Server (RECOMMENDED FOR BEGINNERS)"
    echo "   - Beautiful interface with live status"
    echo "   - Access: http://127.0.0.1:8888"
    echo
    echo "2. 🏠 Homepage Dashboard (PROFESSIONAL)"
    echo "   - Advanced widgets and monitoring"
    echo "   - Access: http://localhost:3001"
    echo
    echo "3. 📝 Direct Service Access (QUICK)"
    echo "   - Jellyfin:     http://localhost:8096"
    echo "   - qBittorrent:  http://localhost:8080"
    echo "   - Portainer:    http://localhost:9000"
    echo "   - Grafana:      http://localhost:3000"
    echo
    echo "4. 🔧 Fix Homepage Issues"
    echo "   - Restart and reconfigure Homepage"
    echo
    echo "5. ❓ Show Detailed Access Guide"
    echo "   - Complete troubleshooting guide"
    echo
}

# Function to show detailed guide
show_detailed_guide() {
    if [ -f "DASHBOARD_ACCESS_GUIDE.md" ]; then
        print_info "Opening detailed access guide..."
        cat DASHBOARD_ACCESS_GUIDE.md
    else
        print_error "Detailed guide not found. Please check DASHBOARD_ACCESS_GUIDE.md"
    fi
}

# Main menu function
main_menu() {
    while true; do
        show_access_options
        echo -n "Enter your choice (1-5): "
        read -r choice
        echo
        
        case $choice in
            1)
                start_dashboard_server
                ;;
            2)
                fix_homepage
                echo
                print_info "Homepage should now be accessible at http://localhost:3001"
                echo "Press Enter to continue..."
                read -r
                ;;
            3)
                test_services
                echo
                print_info "Use the URLs above to access services directly"
                echo "Press Enter to continue..."
                read -r
                ;;
            4)
                fix_homepage
                echo
                echo "Press Enter to continue..."
                read -r
                ;;
            5)
                show_detailed_guide | less
                ;;
            *)
                print_error "Invalid choice. Please enter 1-5."
                echo
                ;;
        esac
    done
}

# Check if we're in the right directory
if [ ! -f "docker-compose-optimized.yml" ]; then
    print_error "Please run this script from the media server directory"
    print_info "Expected location: /Users/morlock/fun/newmedia"
    exit 1
fi

# Check if Docker is running
if ! docker ps &> /dev/null; then
    print_error "Docker is not running or not accessible"
    print_info "Please start Docker Desktop and try again"
    exit 1
fi

# Run initial service test
echo "Checking current service status..."
echo
test_services
echo

# If no arguments provided, show interactive menu
if [ $# -eq 0 ]; then
    main_menu
else
    # Handle command line arguments
    case $1 in
        --test)
            test_services
            ;;
        --fix-homepage)
            fix_homepage
            ;;
        --start-server)
            start_dashboard_server
            ;;
        --guide)
            show_detailed_guide
            ;;
        *)
            echo "Usage: $0 [--test|--fix-homepage|--start-server|--guide]"
            echo "  --test         Test all service accessibility"
            echo "  --fix-homepage Fix Homepage dashboard issues"
            echo "  --start-server Start simple dashboard server"
            echo "  --guide        Show detailed access guide"
            echo
            echo "Run without arguments for interactive menu"
            ;;
    esac
fi