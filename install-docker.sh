#!/bin/bash

# Docker Installation Script for macOS
# This will install Docker Desktop for Mac

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
║        Docker Installation          ║
║           macOS Setup                ║
╚══════════════════════════════════════╝${NC}"
}

# Check if Docker is already installed
check_existing_docker() {
    if command -v docker &> /dev/null; then
        print_status "Docker is already installed!"
        docker --version
        
        if docker compose version &> /dev/null; then
            print_status "Docker Compose is available!"
            docker compose version
            return 0
        elif command -v docker-compose &> /dev/null; then
            print_status "Docker Compose (standalone) is available!"
            docker-compose --version
            return 0
        else
            print_warning "Docker is installed but Compose is missing"
            return 1
        fi
    else
        return 1
    fi
}

# Install Docker Desktop using Homebrew
install_docker_homebrew() {
    print_status "Checking for Homebrew..."
    
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    print_status "Installing Docker Desktop via Homebrew..."
    brew install --cask docker
    
    print_status "Starting Docker Desktop..."
    open /Applications/Docker.app
    
    print_warning "Please wait for Docker Desktop to start (you'll see the whale icon in your menu bar)"
    print_warning "Then run this script again to verify installation"
}

# Manual installation instructions
manual_installation() {
    print_warning "Automatic installation failed. Please install manually:"
    echo ""
    echo "1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop"
    echo "2. Install the .dmg file"
    echo "3. Start Docker Desktop"
    echo "4. Run this script again to verify"
}

# Wait for Docker to be ready
wait_for_docker() {
    print_status "Waiting for Docker to be ready..."
    local timeout=60
    while [ $timeout -gt 0 ]; do
        if docker info &> /dev/null; then
            print_status "Docker is ready! ✓"
            return 0
        fi
        sleep 2
        ((timeout-=2))
        echo -n "."
    done
    
    print_error "Docker did not start within 60 seconds"
    return 1
}

# Main installation function
main() {
    print_header
    
    if check_existing_docker; then
        print_status "Docker installation verified! ✓"
        print_status "You can now run: ./deploy-media.sh"
        return 0
    fi
    
    print_status "Docker not found. Installing..."
    
    # Try Homebrew installation
    if install_docker_homebrew; then
        print_status "Docker Desktop installed via Homebrew"
        
        # Wait for user to start Docker Desktop
        read -p "Press Enter after Docker Desktop has started..."
        
        if wait_for_docker && check_existing_docker; then
            print_status "Installation successful! ✓"
            print_status "You can now run: ./deploy-media.sh"
        else
            print_error "Docker installation verification failed"
            manual_installation
        fi
    else
        print_error "Homebrew installation failed"
        manual_installation
    fi
}

# Run main function
main "$@"