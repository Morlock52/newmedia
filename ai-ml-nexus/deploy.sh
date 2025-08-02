#!/bin/bash

# NEXUS AI/ML System Deployment Script
# This script deploys the complete AI/ML system for NEXUS Media Server

set -e

echo "🚀 NEXUS AI/ML System Deployment"
echo "================================"

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Node.js (for local development)
    if ! command -v node &> /dev/null; then
        echo "⚠️  Node.js is not installed. Required for local development only."
    fi
    
    # Check GPU support
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        nvidia-smi
    else
        echo "⚠️  No NVIDIA GPU detected. AI/ML will run on CPU (slower performance)"
    fi
    
    echo "✅ Prerequisites check complete"
}

# Create necessary directories
setup_directories() {
    echo "📁 Setting up directories..."
    
    mkdir -p models/{face,whisper,encoder,decoder}
    mkdir -p cache
    mkdir -p logs
    mkdir -p data/{media,compressed,training}
    mkdir -p infrastructure/ssl
    
    echo "✅ Directories created"
}

# Download AI models
download_models() {
    echo "🤖 Downloading AI models..."
    
    # Check if models already exist
    if [ -d "models/face/ssd_mobilenetv1_model.weights" ]; then
        echo "✅ Face detection models already downloaded"
    else
        echo "📥 Downloading face detection models..."
        # Models are downloaded during Docker build
    fi
    
    echo "✅ Model download complete"
}

# Generate SSL certificates
generate_ssl_certs() {
    echo "🔒 Generating SSL certificates..."
    
    if [ -f "infrastructure/ssl/cert.pem" ]; then
        echo "✅ SSL certificates already exist"
    else
        openssl req -x509 -newkey rsa:4096 -keyout infrastructure/ssl/key.pem \
            -out infrastructure/ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=NEXUS/CN=localhost"
        echo "✅ SSL certificates generated"
    fi
}

# Build Docker images
build_images() {
    echo "🔨 Building Docker images..."
    
    # Build main AI/ML image
    docker build -t nexus-ai-ml:latest .
    
    # Build training image if Dockerfile.training exists
    if [ -f "Dockerfile.training" ]; then
        docker build -f Dockerfile.training -t nexus-ai-ml-training:latest .
    fi
    
    echo "✅ Docker images built"
}

# Deploy with Docker Compose
deploy_services() {
    echo "🚀 Deploying AI/ML services..."
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Start services
    if [ "$1" == "production" ]; then
        docker-compose -f docker-compose.yml up -d
    else
        docker-compose up -d
    fi
    
    echo "✅ Services deployed"
}

# Wait for services to be healthy
wait_for_services() {
    echo "⏳ Waiting for services to be healthy..."
    
    services=("ai-orchestrator" "recommendation-engine" "content-analysis" "voice-processor" "neural-compression" "emotion-detection")
    
    for service in "${services[@]}"; do
        echo -n "   Waiting for $service..."
        while ! docker-compose ps | grep $service | grep -q "healthy"; do
            sleep 2
            echo -n "."
        done
        echo " ✅"
    done
    
    echo "✅ All services are healthy"
}

# Run post-deployment tests
run_tests() {
    echo "🧪 Running post-deployment tests..."
    
    # Test health endpoint
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        exit 1
    fi
    
    # Test each service endpoint
    endpoints=(
        "http://localhost:8081/health"
        "http://localhost:8082/health"
        "http://localhost:8083/health"
        "http://localhost:8084/health"
        "http://localhost:8085/health"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f $endpoint > /dev/null 2>&1; then
            echo "✅ $endpoint is responsive"
        else
            echo "⚠️  $endpoint is not responding"
        fi
    done
}

# Display deployment information
show_deployment_info() {
    echo ""
    echo "🎉 NEXUS AI/ML System Deployed Successfully!"
    echo "==========================================="
    echo ""
    echo "📍 Service Endpoints:"
    echo "   - Main API: http://localhost:8080"
    echo "   - Dashboard: http://localhost:8080/frontend/index.html"
    echo "   - Grafana: http://localhost:3000 (admin/admin)"
    echo "   - Prometheus: http://localhost:9090"
    echo ""
    echo "🔌 AI/ML Services:"
    echo "   - Recommendation Engine: http://localhost:8081"
    echo "   - Content Analysis: http://localhost:8082"
    echo "   - Voice Processing: http://localhost:8083"
    echo "   - Neural Compression: http://localhost:8084"
    echo "   - Emotion Detection: http://localhost:8085"
    echo ""
    echo "📝 Quick Start:"
    echo "   1. Open the dashboard: http://localhost:8080/frontend/index.html"
    echo "   2. Try voice commands: 'Hey Nexus, play something relaxing'"
    echo "   3. Upload media for AI analysis"
    echo "   4. Check emotion-based recommendations"
    echo ""
    echo "📊 Monitoring:"
    echo "   - View logs: docker-compose logs -f"
    echo "   - Stop services: docker-compose down"
    echo "   - Restart services: docker-compose restart"
    echo ""
}

# Main deployment flow
main() {
    echo "Starting deployment at $(date)"
    
    # Parse arguments
    MODE=${1:-development}
    
    # Run deployment steps
    check_prerequisites
    setup_directories
    download_models
    generate_ssl_certs
    build_images
    deploy_services $MODE
    wait_for_services
    run_tests
    show_deployment_info
    
    echo "Deployment completed at $(date)"
}

# Run main function
main "$@"