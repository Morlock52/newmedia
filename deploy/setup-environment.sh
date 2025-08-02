#!/bin/bash

#############################################
# ENVIRONMENT SETUP AUTOMATION
# Automated environment configuration
# for NewMedia deployment
#############################################

set -euo pipefail

# Color codes
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Environment variables
ENV_TYPE="${1:-production}"
PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"

echo -e "${BLUE}Setting up environment for ${ENV_TYPE}...${NC}"

# Create directory structure
create_directories() {
    echo -e "${BLUE}Creating directory structure...${NC}"
    
    local dirs=(
        "deploy/secrets"
        "deploy/configs"
        "deploy/scripts"
        "deploy/kubernetes"
        "deploy/monitoring"
        "deploy/backups"
        "data/mongodb"
        "data/redis"
        "data/elasticsearch"
        "data/rabbitmq"
        "logs/deployment"
        "logs/applications"
        "logs/monitoring"
        "media/uploads"
        "media/thumbnails"
        "media/transcoded"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${dir}"
        echo -e "  Created: ${dir}"
    done
}

# Install system dependencies
install_dependencies() {
    echo -e "${BLUE}Installing system dependencies...${NC}"
    
    case "$PLATFORM" in
        darwin)
            # macOS
            if command -v brew &> /dev/null; then
                echo "Installing dependencies via Homebrew..."
                brew install jq openssl wget curl git
                
                # Install Docker Desktop if not present
                if ! command -v docker &> /dev/null; then
                    echo "Docker not found. Please install Docker Desktop from:"
                    echo "https://www.docker.com/products/docker-desktop"
                fi
            else
                echo -e "${YELLOW}Homebrew not found. Please install Homebrew first.${NC}"
            fi
            ;;
        linux)
            # Linux
            if command -v apt-get &> /dev/null; then
                echo "Installing dependencies via apt..."
                sudo apt-get update
                sudo apt-get install -y \
                    jq openssl wget curl git \
                    docker.io docker-compose \
                    build-essential python3-pip
                
                # Add user to docker group
                sudo usermod -aG docker $USER
            elif command -v yum &> /dev/null; then
                echo "Installing dependencies via yum..."
                sudo yum install -y \
                    jq openssl wget curl git \
                    docker docker-compose \
                    gcc make python3-pip
                
                # Start Docker service
                sudo systemctl start docker
                sudo systemctl enable docker
                sudo usermod -aG docker $USER
            fi
            ;;
    esac
}

# Generate secure secrets
generate_secrets() {
    echo -e "${BLUE}Generating secure secrets...${NC}"
    
    local secrets_dir="${PROJECT_ROOT}/deploy/secrets"
    
    # Define secrets with descriptions
    declare -A secrets=(
        ["jwt_secret"]="JWT signing secret"
        ["session_secret"]="Session encryption secret"
        ["mongo_root_password"]="MongoDB root password"
        ["mongo_user_password"]="MongoDB user password"
        ["redis_password"]="Redis password"
        ["rabbitmq_password"]="RabbitMQ password"
        ["admin_password"]="Admin dashboard password"
        ["api_key"]="API authentication key"
        ["encryption_key"]="Data encryption key"
        ["oauth_client_secret"]="OAuth client secret"
        ["smtp_password"]="SMTP password"
        ["s3_secret_key"]="S3 secret access key"
    )
    
    for secret_name in "${!secrets[@]}"; do
        local secret_file="${secrets_dir}/${secret_name}"
        if [[ ! -f "$secret_file" ]]; then
            # Generate secure random secret
            openssl rand -base64 32 > "$secret_file"
            chmod 600 "$secret_file"
            echo -e "  Generated: ${secret_name} - ${secrets[$secret_name]}"
        fi
    done
}

# Create environment files
create_env_files() {
    echo -e "${BLUE}Creating environment configuration files...${NC}"
    
    # Development environment
    cat > "${PROJECT_ROOT}/.env.dev" << 'EOF'
# Development Environment Configuration
NODE_ENV=development
LOG_LEVEL=debug
DEBUG=true

# Application
APP_NAME=newmedia
APP_URL=http://localhost:3000
CLIENT_URL=http://localhost:3001

# Database
MONGODB_URI=mongodb://localhost:27017/newmedia_dev
REDIS_URL=redis://localhost:6379/0

# Message Queue
AMQP_URL=amqp://guest:guest@localhost:5672

# Storage
UPLOAD_PATH=./media/uploads
THUMBNAIL_PATH=./media/thumbnails
TRANSCODE_PATH=./media/transcoded

# Development tools
HOT_RELOAD=true
MOCK_EXTERNAL_SERVICES=true
EOF

    # Production environment
    local secrets_dir="${PROJECT_ROOT}/deploy/secrets"
    cat > "${PROJECT_ROOT}/.env.production" << EOF
# Production Environment Configuration
NODE_ENV=production
LOG_LEVEL=info
DEBUG=false

# Application
APP_NAME=newmedia
APP_URL=https://api.newmedia.com
CLIENT_URL=https://app.newmedia.com

# Secrets (loaded from files)
JWT_SECRET=\$(cat ${secrets_dir}/jwt_secret)
SESSION_SECRET=\$(cat ${secrets_dir}/session_secret)

# Database
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=\$(cat ${secrets_dir}/mongo_root_password)
MONGODB_URI=mongodb://admin:\$(cat ${secrets_dir}/mongo_root_password)@mongodb:27017/newmedia?authSource=admin
REDIS_URL=redis://:\$(cat ${secrets_dir}/redis_password)@redis:6379/0

# Message Queue
RABBITMQ_USER=admin
RABBITMQ_PASS=\$(cat ${secrets_dir}/rabbitmq_password)
AMQP_URL=amqp://admin:\$(cat ${secrets_dir}/rabbitmq_password)@rabbitmq:5672

# Storage
UPLOAD_PATH=/data/uploads
THUMBNAIL_PATH=/data/thumbnails
TRANSCODE_PATH=/data/transcoded

# Security
CORS_ORIGINS=https://app.newmedia.com
RATE_LIMIT_WINDOW=15m
RATE_LIMIT_MAX=100

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
EOF

    chmod 600 "${PROJECT_ROOT}/.env.dev"
    chmod 600 "${PROJECT_ROOT}/.env.production"
}

# Setup SSL certificates
setup_ssl() {
    echo -e "${BLUE}Setting up SSL certificates...${NC}"
    
    local ssl_dir="${PROJECT_ROOT}/deploy/configs/ssl"
    mkdir -p "$ssl_dir"
    
    # Generate self-signed certificate for development
    if [[ ! -f "${ssl_dir}/cert.pem" ]]; then
        openssl req -x509 -newkey rsa:4096 \
            -keyout "${ssl_dir}/key.pem" \
            -out "${ssl_dir}/cert.pem" \
            -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=NewMedia/CN=localhost"
        
        chmod 600 "${ssl_dir}/key.pem"
        echo -e "  Generated self-signed SSL certificate"
    fi
}

# Configure firewall
configure_firewall() {
    if [[ "$ENV_TYPE" == "production" ]] && [[ "$PLATFORM" == "linux" ]]; then
        echo -e "${BLUE}Configuring firewall...${NC}"
        
        # Check if ufw is available
        if command -v ufw &> /dev/null; then
            sudo ufw allow 22/tcp    # SSH
            sudo ufw allow 80/tcp    # HTTP
            sudo ufw allow 443/tcp   # HTTPS
            sudo ufw allow 3000/tcp  # API Gateway
            
            # Block direct access to internal services
            sudo ufw deny 27017/tcp  # MongoDB
            sudo ufw deny 6379/tcp   # Redis
            sudo ufw deny 5672/tcp   # RabbitMQ
            
            echo -e "  Firewall rules configured"
        fi
    fi
}

# Setup system limits
setup_system_limits() {
    if [[ "$PLATFORM" == "linux" ]]; then
        echo -e "${BLUE}Configuring system limits...${NC}"
        
        # Increase file descriptor limits
        cat > /tmp/newmedia-limits.conf << EOF
# NewMedia system limits
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
        
        sudo mv /tmp/newmedia-limits.conf /etc/security/limits.d/
        
        # Configure sysctl for production
        if [[ "$ENV_TYPE" == "production" ]]; then
            cat > /tmp/99-newmedia.conf << EOF
# NewMedia sysctl configuration
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
vm.max_map_count = 262144
EOF
            
            sudo mv /tmp/99-newmedia.conf /etc/sysctl.d/
            sudo sysctl -p /etc/sysctl.d/99-newmedia.conf
        fi
        
        echo -e "  System limits configured"
    fi
}

# Setup Docker networks
setup_docker_networks() {
    echo -e "${BLUE}Setting up Docker networks...${NC}"
    
    # Create custom networks
    docker network create newmedia-frontend 2>/dev/null || true
    docker network create newmedia-backend 2>/dev/null || true
    docker network create newmedia-data 2>/dev/null || true
    
    echo -e "  Docker networks created"
}

# Validate environment
validate_environment() {
    echo -e "${BLUE}Validating environment setup...${NC}"
    
    local issues=()
    
    # Check Docker
    if ! docker info &> /dev/null; then
        issues+=("Docker daemon is not running")
    fi
    
    # Check required directories
    local required_dirs=("deploy/secrets" "data" "logs")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "${PROJECT_ROOT}/${dir}" ]]; then
            issues+=("Missing directory: ${dir}")
        fi
    done
    
    # Check secrets
    local required_secrets=("jwt_secret" "mongo_root_password" "redis_password")
    for secret in "${required_secrets[@]}"; do
        if [[ ! -f "${PROJECT_ROOT}/deploy/secrets/${secret}" ]]; then
            issues+=("Missing secret: ${secret}")
        fi
    done
    
    # Report issues
    if [[ ${#issues[@]} -gt 0 ]]; then
        echo -e "${RED}Environment validation failed:${NC}"
        for issue in "${issues[@]}"; do
            echo -e "  - ${issue}"
        done
        exit 1
    else
        echo -e "${GREEN}Environment validation passed!${NC}"
    fi
}

# Main setup function
main() {
    echo -e "${GREEN}=== NewMedia Environment Setup ===${NC}"
    echo -e "Environment Type: ${ENV_TYPE}"
    echo -e "Platform: ${PLATFORM}"
    echo ""
    
    # Run setup steps
    create_directories
    install_dependencies
    generate_secrets
    create_env_files
    setup_ssl
    configure_firewall
    setup_system_limits
    setup_docker_networks
    validate_environment
    
    echo ""
    echo -e "${GREEN}Environment setup completed successfully!${NC}"
    echo -e "Next steps:"
    echo -e "  1. Review generated configuration in: ${BLUE}${PROJECT_ROOT}/.env.${ENV_TYPE}${NC}"
    echo -e "  2. Run deployment: ${BLUE}${PROJECT_ROOT}/deploy/deploy.sh${NC}"
    echo ""
}

# Run main function
main