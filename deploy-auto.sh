#!/bin/bash

# Alternative deployment without Docker Desktop requirement
# Uses system-level Docker if available, or provides manual instructions

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}
╔══════════════════════════════════════╗
║     Alternative Media Deployment    ║
║        No Docker Required            ║
╚══════════════════════════════════════╝${NC}"
}

# Check if we can install Homebrew first
install_homebrew() {
    print_status "Installing Homebrew package manager..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
}

# Install Docker via Homebrew
install_docker_brew() {
    print_status "Installing Docker via Homebrew..."
    brew install docker docker-compose colima
    
    print_status "Starting Docker environment with Colima..."
    colima start --cpu 2 --memory 4 --disk 10
}

# Deploy media server without Docker Desktop
deploy_alternative() {
    print_header
    
    # Check if we can use Docker
    if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
        print_status "Docker found and running ✓"
        deploy_with_docker
        return
    fi
    
    print_warning "Docker not available. Attempting installation..."
    
    # Try to install Homebrew and Docker
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew first..."
        install_homebrew
    fi
    
    # Install Docker
    install_docker_brew
    
    # Try deployment again
    if docker info &> /dev/null 2>&1; then
        deploy_with_docker
    else
        manual_instructions
    fi
}

# Deploy using Docker
deploy_with_docker() {
    print_status "Deploying containers..."
    
    # Create data directories
    mkdir -p ./media-data/{config,downloads,movies,tv,music,qbt-config}
    
    # Deploy Jellyfin
    print_status "Starting Jellyfin container..."
    docker run -d \
        --name simple-media-server \
        -p 8096:8096 \
        -v "$(pwd)/media-data/config:/config" \
        -v "$(pwd)/media-data:/media" \
        --restart unless-stopped \
        jellyfin/jellyfin:latest
    
    # Deploy qBittorrent  
    print_status "Starting qBittorrent container..."
    docker run -d \
        --name qbittorrent-simple \
        -p 8080:8080 \
        -v "$(pwd)/media-data/downloads:/downloads" \
        -v "$(pwd)/media-data/qbt-config:/config" \
        -e PUID=1000 \
        -e PGID=1000 \
        -e TZ=America/New_York \
        --restart unless-stopped \
        lscr.io/linuxserver/qbittorrent:latest
    
    test_deployment
}

# Test the deployment
test_deployment() {
    print_status "Testing deployment..."
    
    sleep 10
    
    # Check container status
    print_status "Container status:"
    docker ps --filter "name=simple-media-server" --filter "name=qbittorrent-simple"
    
    # Test service accessibility
    echo ""
    print_status "Testing service accessibility..."
    
    if curl -s --max-time 5 "http://localhost:8096" > /dev/null 2>&1; then
        print_status "✅ Jellyfin is accessible at http://localhost:8096"
    else
        print_warning "⏳ Jellyfin starting up... (may take a minute)"
    fi
    
    if curl -s --max-time 5 "http://localhost:8080" > /dev/null 2>&1; then
        print_status "✅ qBittorrent is accessible at http://localhost:8080"
    else
        print_warning "⏳ qBittorrent starting up... (may take a minute)"
    fi
    
    show_success
}

# Show success message
show_success() {
    echo -e "${GREEN}
🎉 Media Server Deployed Successfully!

📺 Jellyfin Media Server:  http://localhost:8096
⬇️  qBittorrent Downloads:  http://localhost:8080

📁 Data Location: $(pwd)/media-data/

🔧 Management:
├── Status:  docker ps
├── Logs:    docker logs simple-media-server
├── Stop:    docker stop simple-media-server qbittorrent-simple
└── Start:   docker start simple-media-server qbittorrent-simple

💡 Next Steps:
1. Visit http://localhost:8096 to set up Jellyfin
2. Visit http://localhost:8080 to configure qBittorrent
3. Add media to your libraries and enjoy!
${NC}"
}

# Manual installation instructions
manual_instructions() {
    print_error "Automatic installation failed."
    echo -e "${YELLOW}
📋 Manual Installation Steps:

1. Install Docker Desktop manually:
   - Download: https://www.docker.com/products/docker-desktop
   - Install the .dmg file
   - Start Docker Desktop (whale icon in menu bar)

2. Run deployment:
   cd /Users/morlock/fun/newmedia
   ./deploy-simple.sh

Alternative: Use Homebrew
   brew install --cask docker
   open /Applications/Docker.app

${NC}"
}

# Main execution
main() {
    deploy_alternative
}

# Run the deployment
main "$@"