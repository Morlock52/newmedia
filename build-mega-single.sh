#!/bin/bash

# ========================================
# MEGA SINGLE CONTAINER BUILD SCRIPT
# Builds and deploys the complete 30+ service stack
# ========================================

set -e

echo "🚀 Building Mega Single Container with 30+ Services"
echo "=================================================="

# Color codes for output
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

# ========================================
# CONFIGURATION
# ========================================

IMAGE_NAME="mega-media-server"
IMAGE_TAG="latest"
CONTAINER_NAME="mega-media-server"
COMPOSE_FILE="docker-compose.mega-single.yml"

# ========================================
# PRE-BUILD CHECKS
# ========================================

print_status "Performing pre-build checks..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Check available disk space (need at least 5GB)
available_space=$(df . | awk 'NR==2 {print $4}')
if [ "$available_space" -lt 5242880 ]; then  # 5GB in KB
    print_warning "Low disk space detected. Build may fail."
fi

# Check if required files exist
required_files=("Dockerfile.mega-single" "supervisord.conf" "entrypoint.sh" "$COMPOSE_FILE")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file $file not found!"
        exit 1
    fi
done

print_success "All required files found"

# ========================================
# DIRECTORY PREPARATION
# ========================================

print_status "Preparing directories..."

# Create necessary directories
mkdir -p config/{sonarr,radarr,lidarr,readarr,prowlarr,bazarr}
mkdir -p config/{qbittorrent,jellyfin,tautulli,portainer,uptime-kuma,heimdall}
mkdir -p config/{postgresql,redis,rabbitmq,traefik,authelia}
mkdir -p config/{prometheus,grafana,loki,promtail}
mkdir -p config/api-keys
mkdir -p config/scripts

mkdir -p data/{media,downloads,databases,backups}
mkdir -p data/media/{movies,tv,music,books}
mkdir -p data/downloads/{complete,incomplete,watch}

mkdir -p logs/{apps,system,access,databases}

# Set permissions
chmod +x entrypoint.sh
chmod +x service-interconnection-config.py

print_success "Directory structure created"

# ========================================
# BUILD CONFIGURATION
# ========================================

print_status "Configuring build environment..."

# Create build arguments file
cat > .build-args << EOF
PUID=1000
PGID=1000
TZ=America/New_York
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=1.0.0
EOF

print_success "Build configuration created"

# ========================================
# DOCKER IMAGE BUILD
# ========================================

print_status "Building Docker image..."
print_status "This may take 15-30 minutes depending on your system..."

# Build the image with build args
docker build \
    --file Dockerfile.mega-single \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --build-arg PUID=1000 \
    --build-arg PGID=1000 \
    --build-arg TZ=America/New_York \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    --build-arg VERSION="1.0.0" \
    .

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Docker image build failed"
    exit 1
fi

# ========================================
# IMAGE VERIFICATION
# ========================================

print_status "Verifying image..."

# Check image size
image_size=$(docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "{{.Size}}")
print_status "Image size: $image_size"

# Check image layers
layer_count=$(docker history "${IMAGE_NAME}:${IMAGE_TAG}" --format "{{.ID}}" | wc -l)
print_status "Image layers: $layer_count"

print_success "Image verification completed"

# ========================================
# COMPOSE PREPARATION
# ========================================

print_status "Preparing Docker Compose deployment..."

# Create environment file for compose
cat > .env << EOF
# Mega Container Environment Configuration
COMPOSE_PROJECT_NAME=megamedia
IMAGE_NAME=${IMAGE_NAME}
IMAGE_TAG=${IMAGE_TAG}
CONTAINER_NAME=${CONTAINER_NAME}

# System Configuration
PUID=1000
PGID=1000
TZ=America/New_York

# Security (Change these in production!)
AUTHELIA_JWT_SECRET=your-jwt-secret-change-this-in-production
AUTHELIA_SESSION_SECRET=your-session-secret-change-this-in-production
GRAFANA_SECRET_KEY=your-grafana-secret-key-change-this

# Resource Configuration
MAX_DOWNLOAD_SPEED=0
MAX_UPLOAD_SPEED=0
POSTGRES_MAX_CONNECTIONS=200
REDIS_MAX_MEMORY=512mb
EOF

print_success "Environment configuration created"

# ========================================
# DEPLOYMENT
# ========================================

print_status "Deploying the mega container..."

# Stop and remove existing container if it exists
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    print_status "Stopping existing container..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
fi

# Start the mega container
docker-compose -f "$COMPOSE_FILE" up -d

if [ $? -eq 0 ]; then
    print_success "Mega container deployed successfully"
else
    print_error "Deployment failed"
    exit 1
fi

# ========================================
# POST-DEPLOYMENT VERIFICATION
# ========================================

print_status "Verifying deployment..."

# Wait for container to be running
sleep 10

# Check container status
if docker ps | grep -q "$CONTAINER_NAME"; then
    print_success "Container is running"
else
    print_error "Container is not running"
    print_error "Container logs:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
fi

# ========================================
# HEALTH CHECK
# ========================================

print_status "Performing health checks..."

# Wait for services to start
print_status "Waiting for services to initialize (this may take 2-3 minutes)..."
sleep 60

# Check health endpoint
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    print_status "Health check attempt $attempt/$max_attempts..."
    
    if curl -f -s http://localhost:8888/health >/dev/null 2>&1; then
        print_success "Health check passed"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Health check failed after $max_attempts attempts"
        print_status "Container may still be starting up. Check logs for details."
        break
    fi
    
    sleep 30
    attempt=$((attempt + 1))
done

# ========================================
# SERVICE INTERCONNECTION
# ========================================

print_status "Running service interconnection configuration..."

# Copy service interconnection script into container and run it
docker exec "$CONTAINER_NAME" python3 /opt/scripts/service_interconnector.py &

print_success "Service interconnection started in background"

# ========================================
# COMPLETION SUMMARY
# ========================================

echo ""
echo "=========================================="
echo "🎉 MEGA CONTAINER DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""

print_success "Container Status: Running"
print_success "Container Name: $CONTAINER_NAME"
print_success "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "🌐 SERVICE ACCESS POINTS:"
echo "=========================================="
echo "Main Dashboard:       http://localhost/"
echo "Traefik Dashboard:    http://localhost:8080/"
echo "Health Monitor:       http://localhost:8888/health"
echo "Supervisor:           http://localhost:9001/"
echo ""
echo "🎬 MEDIA MANAGEMENT:"
echo "Sonarr (TV):          http://localhost:8989/"
echo "Radarr (Movies):      http://localhost:7878/"
echo "Lidarr (Music):       http://localhost:8686/"
echo "Readarr (Books):      http://localhost:8787/"
echo "Prowlarr (Indexers):  http://localhost:9696/"
echo "Bazarr (Subtitles):   http://localhost:6767/"
echo ""
echo "⬇️ DOWNLOAD CLIENTS:"
echo "qBittorrent:          http://localhost:8090/"
echo ""
echo "📺 MEDIA SERVERS:"
echo "Jellyfin:             http://localhost:8096/"
echo "Tautulli:             http://localhost:8181/"
echo ""
echo "📊 MONITORING:"
echo "Grafana:              http://localhost:3000/"
echo "Prometheus:           http://localhost:9090/"
echo "Loki:                 http://localhost:3100/"
echo ""
echo "🔐 AUTHENTICATION:"
echo "Authelia:             http://localhost:9091/"
echo ""
echo "🛠️ MANAGEMENT:"
echo "Portainer:            http://localhost:9000/"
echo "Uptime Kuma:          http://localhost:3001/"
echo ""

echo "📋 NEXT STEPS:"
echo "=========================================="
echo "1. Wait 5-10 minutes for all services to fully initialize"
echo "2. Access the main dashboard at http://localhost/"
echo "3. Configure authentication in Authelia (admin/admin123)"
echo "4. Set up your media libraries in the *arr applications"
echo "5. Configure indexers in Prowlarr"
echo "6. Set up your media server libraries in Jellyfin"
echo "7. Monitor system health at http://localhost:8888/health"
echo ""

echo "📄 USEFUL COMMANDS:"
echo "=========================================="
echo "View logs:           docker-compose -f $COMPOSE_FILE logs -f"
echo "Restart container:   docker-compose -f $COMPOSE_FILE restart"
echo "Stop container:      docker-compose -f $COMPOSE_FILE down"
echo "Enter container:     docker exec -it $CONTAINER_NAME bash"
echo "Health check:        curl http://localhost:8888/health"
echo ""

print_success "Mega Single Container deployment completed successfully!"
print_status "All 30+ services are now running in a single container with full interconnection."

# Clean up
rm -f .build-args

echo "🚀 Enjoy your complete media server stack!"