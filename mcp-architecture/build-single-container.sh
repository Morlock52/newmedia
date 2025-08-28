#!/bin/bash

# Ultimate Media Server 2025 - Single Container Build Script
# Builds and deploys ALL 30 services in one Docker container

set -e

echo "🚀 Ultimate Media Server 2025 - Single Container Builder"
echo "================================================="
echo "Building ALL 30 media server services in ONE container"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then
        print_warning "Less than 10GB disk space available. Build may fail."
    fi
    
    # Check available RAM (minimum 4GB)
    available_ram=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_ram" -lt 4096 ]; then
        print_warning "Less than 4GB RAM available. Consider increasing memory."
    fi
    
    print_success "Prerequisites check completed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Create media directories
    mkdir -p media/{movies,tv,music,books,audiobooks,podcasts,documentaries,anime}
    mkdir -p downloads/{complete,incomplete,watch}
    mkdir -p backups
    mkdir -p docker/single-container/{configs,service-configs}
    
    print_success "Directory structure created"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.single-container ]; then
            cp .env.single-container .env
            print_success "Environment file created from template"
        else
            print_error "Environment template not found. Please ensure .env.single-container exists."
            exit 1
        fi
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    # Check if OpenAI API key is set
    if grep -q "sk-your-openai-api-key-here" .env; then
        print_warning "Please update your OpenAI API key in .env file"
        echo "Edit .env and set OPENAI_API_KEY=your-actual-key"
        read -p "Press Enter to continue after setting up .env file..."
    fi
}

# Create Docker configurations
create_docker_configs() {
    print_status "Creating Docker configuration files..."
    
    # Create startup script if it doesn't exist
    if [ ! -f docker/single-container/startup.sh ]; then
        print_error "startup.sh not found in docker/single-container/"
        exit 1
    fi
    
    # Create supervisord config if it doesn't exist
    if [ ! -f docker/single-container/supervisord.conf ]; then
        print_error "supervisord.conf not found in docker/single-container/"
        exit 1
    fi
    
    # Create nginx config if it doesn't exist
    if [ ! -f docker/single-container/nginx.conf ]; then
        print_error "nginx.conf not found in docker/single-container/"
        exit 1
    fi
    
    # Make startup script executable
    chmod +x docker/single-container/startup.sh
    
    print_success "Docker configurations ready"
}

# Build Docker image
build_image() {
    print_status "Building Ultimate Media Server 2025 Docker image..."
    print_status "This may take 15-30 minutes depending on your internet connection..."
    
    # Build with progress output
    docker-compose -f docker-compose.single-container.yml build --progress=plain
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Docker image build failed"
        exit 1
    fi
}

# Start container
start_container() {
    print_status "Starting Ultimate Media Server 2025 container..."
    
    # Stop any existing container
    docker-compose -f docker-compose.single-container.yml down 2>/dev/null || true
    
    # Start the container
    docker-compose -f docker-compose.single-container.yml up -d
    
    if [ $? -eq 0 ]; then
        print_success "Container started successfully"
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to start (this may take 2-5 minutes)..."
    
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
            print_success "Main dashboard is ready!"
            break
        fi
        
        echo -n "."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_warning "Services are taking longer than expected to start"
        print_status "You can check status with: docker-compose -f docker-compose.single-container.yml logs"
    fi
}

# Display service status
show_service_status() {
    print_status "Checking service status..."
    
    echo ""
    echo "🌟 Ultimate Media Server 2025 - Service Status"
    echo "=============================================="
    
    # Check main dashboard
    if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
        echo "✅ Main Dashboard: http://localhost:8090"
    else
        echo "❌ Main Dashboard: Not ready"
    fi
    
    # Check key services
    services=(
        "8096:Jellyfin"
        "8989:Sonarr" 
        "7878:Radarr"
        "9696:Prowlarr"
        "8080:qBittorrent"
        "5055:Overseerr"
        "8181:Tautulli"
        "3000:Homepage"
        "9000:Portainer"
        "19999:Netdata"
    )
    
    for service in "${services[@]}"; do
        port=$(echo $service | cut -d: -f1)
        name=$(echo $service | cut -d: -f2)
        
        if curl -sf http://localhost:$port >/dev/null 2>&1 || \
           curl -sf http://localhost:$port/web >/dev/null 2>&1 || \
           curl -sf http://localhost:$port/api/v1/status >/dev/null 2>&1; then
            echo "✅ $name: http://localhost:$port"
        else
            echo "⏳ $name: Starting... (http://localhost:$port)"
        fi
    done
    
    echo ""
    echo "📊 Container Status:"
    docker-compose -f docker-compose.single-container.yml ps
}

# Display final information
show_final_info() {
    echo ""
    echo "🎉 Ultimate Media Server 2025 Deployment Complete!"
    echo "=================================================="
    echo ""
    echo "🌐 Main Dashboard: http://localhost:8090"
    echo "🔧 MCP Server: http://localhost:3001"
    echo "📊 Health Check: http://localhost:8090/health"
    echo "📋 Service List: http://localhost:8090/services"
    echo ""
    echo "🚀 All 30 Services Available:"
    echo "   • 3 Media Servers (Jellyfin, Plex, Emby)"
    echo "   • 5 Content Management (*arr Suite)"  
    echo "   • 3 Indexers & Search"
    echo "   • 5 Download Clients"
    echo "   • 3 Request Management"
    echo "   • 2 Analytics & Monitoring"
    echo "   • 4 Dashboards"
    echo "   • 5 Infrastructure Services"
    echo ""
    echo "📖 Next Steps:"
    echo "   1. Access the dashboard: http://localhost:8090"
    echo "   2. Configure each service through web interfaces"
    echo "   3. Copy API keys to .env file"
    echo "   4. Restart container to apply API keys"
    echo ""
    echo "🔧 Management Commands:"
    echo "   • View logs: docker-compose -f docker-compose.single-container.yml logs"
    echo "   • Restart: docker-compose -f docker-compose.single-container.yml restart"
    echo "   • Stop: docker-compose -f docker-compose.single-container.yml down"
    echo "   • Update: git pull && ./build-single-container.sh"
    echo ""
    echo "📚 Documentation: README-SINGLE-CONTAINER.md"
    echo ""
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Build failed. Cleaning up..."
        docker-compose -f docker-compose.single-container.yml down 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo "Starting Ultimate Media Server 2025 single container build..."
    echo ""
    
    check_prerequisites
    create_directories  
    setup_environment
    create_docker_configs
    build_image
    start_container
    wait_for_services
    show_service_status
    show_final_info
    
    print_success "Ultimate Media Server 2025 is ready!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --no-start)
            SKIP_START=true
            shift
            ;;
        --force)
            FORCE_REBUILD=true
            shift
            ;;
        -h|--help)
            echo "Ultimate Media Server 2025 - Single Container Builder"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-build    Skip Docker image build"
            echo "  --no-start    Skip container startup"
            echo "  --force       Force rebuild even if image exists"
            echo "  -h, --help    Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main

# Remove trap
trap - EXIT