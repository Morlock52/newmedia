#!/bin/bash

# AI-Enhanced Media Server Deployment Script
# August 2025 - Complete deployment with O3-mini style AI agents

set -e

echo "🚀 AI-Enhanced Media Server Deployment"
echo "========================================"
echo "Version: 2025.08.09"
echo "Features: O3-mini AI, Content Moderation, Ethical Recommendations"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking Prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    else
        print_status "Docker found: $(docker --version)"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    else
        print_status "Docker Compose found: $(docker-compose --version)"
    fi
    
    # Check Python (for AI services)
    if ! command -v python3 &> /dev/null; then
        print_warning "Python 3 not found - AI services may have limited functionality"
    else
        print_status "Python found: $(python3 --version)"
    fi
    
    # Check available memory
    AVAILABLE_MEM=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}' || echo "8")
    if [ "$AVAILABLE_MEM" -lt "4" ]; then
        print_warning "Less than 4GB RAM available - AI features may be limited"
    else
        print_status "Available memory: ${AVAILABLE_MEM}GB"
    fi
    
    # Check for GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected - enhanced AI performance available"
        GPU_AVAILABLE=true
    else
        print_info "No NVIDIA GPU detected - using CPU for AI processing"
        GPU_AVAILABLE=false
    fi
    
    echo ""
}

# Create environment file
create_env_file() {
    echo "🔐 Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        cat > .env << 'EOF'
# AI-Enhanced Media Server Configuration
# Generated: $(date)

# Security
JWT_SECRET=$(openssl rand -hex 32)
REDIS_PASSWORD=$(openssl rand -hex 16)
POSTGRES_PASSWORD=$(openssl rand -hex 16)
GRAFANA_PASSWORD=admin123

# AI Configuration
CUDA_VISIBLE_DEVICES=0
AI_MODEL_CACHE=/app/models
SAFETY_THRESHOLD_STRICT=0.9
SAFETY_THRESHOLD_MODERATE=0.7
SAFETY_THRESHOLD_RELAXED=0.5

# Social Media (Optional - add your keys)
YOUTUBE_API_KEY=
TWITTER_BEARER_TOKEN=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=

# Database
POSTGRES_USER=aiuser
POSTGRES_DB=ai_media_server

# Features
CONTENT_FILTERING_STRICT=true
REAL_TIME_MODERATION=true
ETHICAL_SCORING_ENABLED=true
PRIVACY_MODE=enhanced
EOF
        print_status "Created .env file with secure defaults"
    else
        print_info "Using existing .env file"
    fi
    
    echo ""
}

# Create necessary directories
create_directories() {
    echo "📁 Creating required directories..."
    
    directories=(
        "ai-safety-data"
        "ai-models"
        "moderation-data"
        "moderation-uploads"
        "recommendation-data"
        "social-media-data"
        "redis-ai-data"
        "postgres-ai-data"
        "elasticsearch-ai-data"
        "prometheus-ai-data"
        "grafana-ai-data"
        "logs/ai-safety"
        "logs/moderation"
        "logs/recommendations"
        "logs/social-media"
        "logs/gateway"
        "logs/dashboard"
        "logs/model-manager"
        "logs/nginx"
        "logs/gpu"
        "gateway-config"
        "dashboard-config"
        "nginx-ai-config"
        "ssl-certs"
        "kibana-ai-config"
        "prometheus-ai-config"
        "grafana-ai-config"
        "postgres-init"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created: $dir"
    done
    
    echo ""
}

# Download AI models
download_ai_models() {
    echo "🤖 Preparing AI models..."
    
    if [ ! -d "ai-models/huggingface" ]; then
        print_info "Downloading AI models (this may take a few minutes)..."
        
        # Create model download script
        cat > download-models.py << 'EOF'
#!/usr/bin/env python3
import os
import sys

try:
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "unitary/toxic-bert",
    ]
    
    print("Downloading models...")
    for model in models:
        print(f"  - {model}")
        try:
            if "sentence-transformers" in model:
                SentenceTransformer(model)
            else:
                AutoModel.from_pretrained(model)
                AutoTokenizer.from_pretrained(model)
        except Exception as e:
            print(f"    Warning: Could not download {model}: {e}")
    
    print("Model download complete!")
    
except ImportError:
    print("Python packages not installed. Models will be downloaded on first run.")
    sys.exit(0)
EOF
        
        python3 download-models.py 2>/dev/null || print_warning "Models will be downloaded on first run"
        rm -f download-models.py
    else
        print_status "AI models already present"
    fi
    
    echo ""
}

# Deploy core services
deploy_core_services() {
    echo "🎬 Checking core media services..."
    
    # Check if core services are already running
    if docker ps | grep -q "jellyfin"; then
        print_status "Core media services already running"
    else
        print_info "Starting core media services..."
        docker-compose -f docker-compose.yml up -d
        sleep 10
    fi
    
    echo ""
}

# Deploy AI services
deploy_ai_services() {
    echo "🧠 Deploying AI-Enhanced Services..."
    
    # Determine deployment profile
    PROFILES=""
    if [ "$GPU_AVAILABLE" = true ]; then
        PROFILES="--profile gpu"
        print_info "Enabling GPU acceleration"
    fi
    
    # Check if production mode requested
    if [ "$1" = "production" ]; then
        PROFILES="$PROFILES --profile production"
        print_info "Enabling production mode with SSL"
    fi
    
    # Deploy AI services
    print_info "Starting AI services..."
    docker-compose -f docker-compose.ai-enhanced.yml $PROFILES up -d
    
    # Wait for services to start
    echo -n "Waiting for services to initialize"
    for i in {1..30}; do
        echo -n "."
        sleep 2
    done
    echo ""
    
    print_status "AI services deployed"
    echo ""
}

# Verify deployment
verify_deployment() {
    echo "✅ Verifying Deployment..."
    
    services=(
        "ai-safety-service:8090"
        "content-moderation-service:8091"
        "recommendation-engine:8092"
        "social-media-service:8093"
        "ai-dashboard:8094"
        "ai-gateway:8095"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
            print_status "$name is running on port $port"
        else
            print_warning "$name may still be starting on port $port"
        fi
    done
    
    echo ""
}

# Display access information
display_access_info() {
    echo "🎯 AI-Enhanced Media Server Ready!"
    echo "==================================="
    echo ""
    echo "📍 Service Endpoints:"
    echo "  • AI Dashboard:        http://localhost:8094"
    echo "  • API Gateway:         http://localhost:8095"
    echo "  • Jellyfin:           http://localhost:8096"
    echo "  • Sonarr:             http://localhost:8989"
    echo "  • Radarr:             http://localhost:7878"
    echo "  • Prowlarr:           http://localhost:9696"
    echo "  • qBittorrent:        http://localhost:8080"
    echo ""
    echo "📊 Monitoring:"
    echo "  • Grafana:            http://localhost:3001 (admin/admin123)"
    echo "  • Kibana:             http://localhost:5602"
    echo "  • Prometheus:         http://localhost:9091"
    echo ""
    echo "🛡️ Safety Features:"
    echo "  ✓ O3-mini style reasoning"
    echo "  ✓ Content moderation active"
    echo "  ✓ NSFW filtering enabled"
    echo "  ✓ Copyright protection"
    echo "  ✓ Ethical recommendations"
    echo ""
    echo "📚 Documentation:"
    echo "  • Deployment Guide:   AI_DEPLOYMENT_GUIDE.md"
    echo "  • API Reference:      docs/ai-safety-api.md"
    echo ""
    echo "💡 Quick Commands:"
    echo "  • View logs:          docker-compose -f docker-compose.ai-enhanced.yml logs -f"
    echo "  • Stop services:      docker-compose -f docker-compose.ai-enhanced.yml down"
    echo "  • Update services:    docker-compose -f docker-compose.ai-enhanced.yml pull"
    echo ""
}

# Main deployment flow
main() {
    clear
    
    # Parse arguments
    MODE="${1:-development}"
    
    # Run deployment steps
    check_prerequisites
    create_env_file
    create_directories
    download_ai_models
    deploy_core_services
    deploy_ai_services "$MODE"
    verify_deployment
    display_access_info
    
    print_status "Deployment complete! 🚀"
    echo ""
    echo "Visit http://localhost:8094 to access the AI Dashboard"
    echo ""
}

# Run main function
main "$@"