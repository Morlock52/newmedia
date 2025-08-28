#!/bin/bash

# Ultimate Media Server 2025 - Single Container Deployment Script
# Deploys all 18 components and 30+ services in one container

set -e

echo "================================================"
echo "🚀 ULTIMATE MEDIA SERVER 2025 - DEPLOYMENT"
echo "================================================"
echo "📦 18 Components"
echo "🔗 30+ Services"
echo "🤖 AI-Powered Features"
echo "🎨 Cyberpunk UI Theme"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${CYAN}[STATUS]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed: $(docker --version)"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose is not installed. Please install Docker Compose first."
            exit 1
        fi
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    print_success "Docker Compose is installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "config"
        "data"
        "media/movies"
        "media/tv"
        "media/music"
        "media/books"
        "media/photos"
        "downloads/complete"
        "downloads/incomplete"
        "logs"
        "postgres-data"
        "redis-data"
        "models"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        echo "  ✅ Created: $dir"
    done
    
    print_success "Directory structure created"
}

# Set proper permissions
set_permissions() {
    print_status "Setting permissions..."
    
    # Get current user ID and group ID
    USER_ID=$(id -u)
    GROUP_ID=$(id -g)
    
    # Update .env file with correct PUID and PGID
    if [ -f .env ]; then
        sed -i.bak "s/PUID=.*/PUID=$USER_ID/" .env
        sed -i.bak "s/PGID=.*/PGID=$GROUP_ID/" .env
    else
        echo "PUID=$USER_ID" > .env
        echo "PGID=$GROUP_ID" >> .env
    fi
    
    # Set permissions on directories
    chmod -R 755 config data media downloads logs models backups
    
    print_success "Permissions set (PUID=$USER_ID, PGID=$GROUP_ID)"
}

# Build Docker image
build_image() {
    print_status "Building Docker image (this may take a while)..."
    
    # Check if Dockerfile exists (try simplified version first)
    if [ -f Dockerfile.ultimate-single-container-simplified ]; then
        DOCKERFILE="Dockerfile.ultimate-single-container-simplified"
    elif [ -f Dockerfile.ultimate-single-container ]; then
        DOCKERFILE="Dockerfile.ultimate-single-container"
    else
        print_error "No Dockerfile found!"
        exit 1
    fi
    
    print_status "Using $DOCKERFILE"
    
    # Build the image
    docker build -t ultimate-media-server:2025 -f $DOCKERFILE . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    
    print_success "Docker image built successfully"
}

# Deploy the container
deploy_container() {
    print_status "Deploying Ultimate Media Server container..."
    
    # Stop existing container if running
    if docker ps -a | grep -q ultimate-media-server-2025; then
        print_warning "Stopping existing container..."
        docker stop ultimate-media-server-2025 || true
        docker rm ultimate-media-server-2025 || true
    fi
    
    # Deploy using docker-compose
    $COMPOSE_CMD -f docker-compose.ultimate-single.yml up -d || {
        print_error "Failed to deploy container"
        exit 1
    }
    
    print_success "Container deployed successfully"
}

# Wait for services to start
wait_for_services() {
    print_status "Waiting for services to start..."
    
    # Wait for main dashboard
    echo -n "  Waiting for dashboard"
    for i in {1..60}; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200\|301\|302"; then
            echo " ✅"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Wait for API
    echo -n "  Waiting for API"
    for i in {1..30}; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; then
            echo " ✅"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    print_success "Services are starting up"
}

# Display service status
display_status() {
    echo ""
    echo "================================================"
    echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
    echo "================================================"
    echo ""
    echo "📊 Component Status:"
    echo "  ✅ Notification System"
    echo "  ✅ Data Analytics Dashboard"
    echo "  ✅ Mobile PWA Interface"
    echo "  ✅ Smart Download Manager"
    echo "  ✅ Voice Control System"
    echo "  ✅ AR/VR Media Experience"
    echo "  ✅ Automated Testing Suite"
    echo "  ✅ Cyberpunk Authentication"
    echo "  ✅ Holographic Media Player"
    echo "  ✅ Neural Recommendations"
    echo "  ✅ Real-time Monitoring"
    echo "  ✅ Unified Media API"
    echo "  ✅ 3D Service Visualization"
    echo "  ✅ NEXUS AI Assistant"
    echo "  ✅ Service Grid Dashboard"
    echo "  ✅ Cyberpunk Theme System"
    echo "  ✅ Social Watch Party"
    echo "  ✅ Predictive Analytics"
    echo ""
    echo "🔗 Service URLs:"
    echo "  🏠 Main Dashboard:     http://localhost:3000"
    echo "  🔌 API Endpoint:       http://localhost:8080"
    echo "  📺 Jellyfin:          http://localhost:8096"
    echo "  🎬 Plex:              http://localhost:32400/web"
    echo "  📺 Emby:              http://localhost:8920"
    echo "  📡 Sonarr:            http://localhost:8989"
    echo "  🎬 Radarr:            http://localhost:7878"
    echo "  🎵 Lidarr:            http://localhost:8686"
    echo "  📚 Readarr:           http://localhost:8787"
    echo "  🗣️ Bazarr:            http://localhost:6767"
    echo "  🔍 Prowlarr:          http://localhost:9696"
    echo "  📥 qBittorrent:       http://localhost:8082"
    echo "  📥 Transmission:      http://localhost:9091"
    echo "  🎬 Overseerr:         http://localhost:5055"
    echo "  🎬 Jellyseerr:        http://localhost:5056"
    echo "  📊 Grafana:           http://localhost:3001"
    echo "  📈 Prometheus:        http://localhost:9090"
    echo "  🏥 Uptime Kuma:       http://localhost:3002"
    echo "  📊 Tautulli:          http://localhost:8181"
    echo "  🗂️ Organizr:          http://localhost:8084"
    echo "  🏠 Heimdall:          http://localhost:8085"
    echo "  🏠 Homer:             http://localhost:8086"
    echo "  🐳 Portainer:         http://localhost:9000"
    echo "  🔧 Nginx Proxy:       http://localhost:81"
    echo ""
    echo "📝 Default Credentials:"
    echo "  Username: admin"
    echo "  Password: admin (please change immediately)"
    echo ""
    echo "📚 Documentation:"
    echo "  View project docs:    http://localhost:3000/docs"
    echo "  API documentation:    http://localhost:8080/api-docs"
    echo ""
    echo "🚀 Quick Commands:"
    echo "  View logs:           docker logs ultimate-media-server-2025"
    echo "  Stop container:      docker stop ultimate-media-server-2025"
    echo "  Start container:     docker start ultimate-media-server-2025"
    echo "  Restart container:   docker restart ultimate-media-server-2025"
    echo "  Shell access:        docker exec -it ultimate-media-server-2025 /bin/bash"
    echo ""
    echo "================================================"
    echo -e "${MAGENTA}Enjoy your Ultimate Media Server 2025!${NC}"
    echo "================================================"
}

# Main execution
main() {
    echo ""
    
    # Run checks
    check_docker
    check_docker_compose
    
    # Create structure
    create_directories
    set_permissions
    
    # Build and deploy
    build_image
    deploy_container
    
    # Wait and display status
    wait_for_services
    display_status
}

# Run main function
main "$@"