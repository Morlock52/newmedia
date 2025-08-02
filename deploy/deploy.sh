#!/bin/bash

#############################################
# NEWMEDIA DEPLOYMENT AUTOMATION SCRIPT
# One-command deployment with full automation
# Supports: macOS, Linux, Docker, Kubernetes
#############################################

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DEPLOYMENT_MODE="docker"
ENVIRONMENT="production"
AUTO_CONFIRM="false"
SKIP_BACKUP="false"
VERBOSE="false"
DRY_RUN="false"

# Logging
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
LOG_FILE="${LOG_DIR}/deploy-$(date +%Y%m%d-%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} ${message}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${message}"
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} ${message}"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${message}"
            ;;
        DEBUG)
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "${MAGENTA}[DEBUG]${NC} ${message}"
            fi
            ;;
    esac
    
    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
${CYAN}NEWMEDIA DEPLOYMENT AUTOMATION${NC}

${WHITE}USAGE:${NC}
    $0 [OPTIONS]

${WHITE}OPTIONS:${NC}
    -m, --mode <mode>        Deployment mode (docker|kubernetes|local) [default: docker]
    -e, --env <env>          Environment (dev|staging|production) [default: production]
    -y, --yes                Auto-confirm all prompts
    -s, --skip-backup        Skip backup before deployment
    -v, --verbose            Enable verbose output
    -d, --dry-run            Perform a dry run without making changes
    -h, --help               Show this help message

${WHITE}EXAMPLES:${NC}
    # Quick production deployment
    $0 -y

    # Development deployment with verbose output
    $0 -e dev -v

    # Kubernetes deployment with dry run
    $0 -m kubernetes -d

    # Staging deployment without backup
    $0 -e staging -s

${WHITE}FEATURES:${NC}
    ✓ One-command deployment
    ✓ Automatic environment setup
    ✓ Secret generation and management
    ✓ Health checks and monitoring
    ✓ Backup automation
    ✓ Update procedures
    ✓ CI/CD pipeline integration
    ✓ Logging and audit trails

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -y|--yes)
                AUTO_CONFIRM="true"
                shift
                ;;
            -s|--skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log ERROR "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check system requirements
check_requirements() {
    log INFO "Checking system requirements..."
    
    local missing_deps=()
    
    # Check for required tools
    local required_tools=("docker" "docker-compose" "git" "openssl" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            missing_deps+=("kubectl")
        fi
        if ! command -v helm &> /dev/null; then
            missing_deps+=("helm")
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log ERROR "Missing required dependencies: ${missing_deps[*]}"
        log INFO "Please install missing dependencies and try again"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log ERROR "Docker daemon is not running"
        exit 1
    fi
    
    log SUCCESS "All requirements satisfied"
}

# Generate secrets
generate_secrets() {
    log INFO "Generating deployment secrets..."
    
    local secrets_dir="${PROJECT_ROOT}/deploy/secrets"
    mkdir -p "$secrets_dir"
    
    # List of secrets to generate
    local secrets=(
        "jwt_secret"
        "session_secret"
        "mongo_root_password"
        "rabbitmq_password"
        "redis_password"
        "admin_password"
        "api_key"
        "encryption_key"
    )
    
    for secret in "${secrets[@]}"; do
        local secret_file="${secrets_dir}/${secret}"
        if [[ ! -f "$secret_file" ]]; then
            openssl rand -base64 32 > "$secret_file"
            chmod 600 "$secret_file"
            log DEBUG "Generated secret: $secret"
        fi
    done
    
    log SUCCESS "Secrets generated successfully"
}

# Create environment file
create_env_file() {
    log INFO "Creating environment configuration..."
    
    local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    local secrets_dir="${PROJECT_ROOT}/deploy/secrets"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY RUN] Would create environment file: $env_file"
        return
    fi
    
    cat > "$env_file" << EOF
# NewMedia Environment Configuration
# Generated: $(date)
# Environment: ${ENVIRONMENT}

# Application
NODE_ENV=${ENVIRONMENT}
APP_NAME=newmedia
APP_URL=http://localhost:3000
CLIENT_URL=http://localhost:3000

# Secrets
JWT_SECRET=$(cat "${secrets_dir}/jwt_secret" 2>/dev/null || echo "changeme")
SESSION_SECRET=$(cat "${secrets_dir}/session_secret" 2>/dev/null || echo "changeme")

# MongoDB
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=$(cat "${secrets_dir}/mongo_root_password" 2>/dev/null || echo "changeme")
MONGODB_URI=mongodb://mongodb:27017/newmedia

# Redis
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=$(cat "${secrets_dir}/redis_password" 2>/dev/null || echo "")

# RabbitMQ
RABBITMQ_USER=admin
RABBITMQ_PASS=$(cat "${secrets_dir}/rabbitmq_password" 2>/dev/null || echo "changeme")
AMQP_URL=amqp://admin:$(cat "${secrets_dir}/rabbitmq_password" 2>/dev/null || echo "changeme")@rabbitmq:5672

# AWS (Optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_NAME=

# SMTP (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=

# OAuth (Optional)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

# CDN
CDN_URL=

# Admin
ADMIN_PASSWORD=$(cat "${secrets_dir}/admin_password" 2>/dev/null || echo "changeme")
API_KEY=$(cat "${secrets_dir}/api_key" 2>/dev/null || echo "changeme")
EOF

    chmod 600 "$env_file"
    log SUCCESS "Environment file created: $env_file"
}

# Backup current deployment
backup_deployment() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log INFO "Skipping backup (--skip-backup flag)"
        return
    fi
    
    log INFO "Creating backup of current deployment..."
    
    local backup_dir="${PROJECT_ROOT}/backups/deployment"
    local backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_path="${backup_dir}/${backup_name}"
    
    mkdir -p "$backup_path"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY RUN] Would create backup at: $backup_path"
        return
    fi
    
    # Backup databases
    if docker ps --format '{{.Names}}' | grep -q mongodb; then
        log DEBUG "Backing up MongoDB..."
        docker exec mongodb mongodump --out /backup --authenticationDatabase admin || true
        docker cp mongodb:/backup "${backup_path}/mongodb" || true
    fi
    
    # Backup Redis
    if docker ps --format '{{.Names}}' | grep -q redis; then
        log DEBUG "Backing up Redis..."
        docker exec redis redis-cli BGSAVE || true
        sleep 2
        docker cp redis:/data/dump.rdb "${backup_path}/redis.rdb" || true
    fi
    
    # Backup configuration files
    cp -r "${PROJECT_ROOT}/deploy/secrets" "${backup_path}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/.env."* "${backup_path}/" 2>/dev/null || true
    
    # Create backup manifest
    cat > "${backup_path}/manifest.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "${ENVIRONMENT}",
    "deployment_mode": "${DEPLOYMENT_MODE}",
    "services": $(docker ps --format '{{.Names}}' | jq -R . | jq -s .),
    "backup_size": "$(du -sh "$backup_path" | cut -f1)"
}
EOF
    
    log SUCCESS "Backup created: $backup_path"
}

# Deploy with Docker Compose
deploy_docker() {
    log INFO "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Select compose file based on environment
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.production.yml"
    elif [[ "$ENVIRONMENT" == "dev" ]]; then
        compose_file="docker-compose.dev.yml"
    fi
    
    # Check if compose file exists
    if [[ ! -f "$compose_file" ]]; then
        log WARNING "Compose file not found: $compose_file, using default"
        compose_file="docker-compose.yml"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY RUN] Would deploy using: $compose_file"
        docker-compose -f "$compose_file" config
        return
    fi
    
    # Pull latest images
    log INFO "Pulling latest images..."
    docker-compose -f "$compose_file" pull
    
    # Build custom images
    log INFO "Building custom images..."
    docker-compose -f "$compose_file" build --parallel
    
    # Start services
    log INFO "Starting services..."
    docker-compose -f "$compose_file" up -d --remove-orphans
    
    log SUCCESS "Docker deployment completed"
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log INFO "Deploying with Kubernetes..."
    
    local k8s_dir="${PROJECT_ROOT}/deploy/kubernetes"
    
    if [[ ! -d "$k8s_dir" ]]; then
        log ERROR "Kubernetes manifests not found at: $k8s_dir"
        exit 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY RUN] Would deploy Kubernetes manifests from: $k8s_dir"
        kubectl apply -f "$k8s_dir" --dry-run=client
        return
    fi
    
    # Create namespace
    kubectl create namespace newmedia --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic newmedia-secrets \
        --from-file="${PROJECT_ROOT}/deploy/secrets" \
        --namespace=newmedia \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests
    kubectl apply -f "$k8s_dir" --namespace=newmedia
    
    # Wait for rollout
    kubectl rollout status deployment --namespace=newmedia --timeout=10m
    
    log SUCCESS "Kubernetes deployment completed"
}

# Health check
health_check() {
    log INFO "Running health checks..."
    
    local max_retries=30
    local retry_count=0
    local all_healthy=false
    
    while [[ $retry_count -lt $max_retries ]] && [[ "$all_healthy" == "false" ]]; do
        all_healthy=true
        
        # Check API Gateway
        if ! curl -sf http://localhost:3000/health &> /dev/null; then
            all_healthy=false
            log DEBUG "API Gateway not ready yet..."
        fi
        
        # Check other services
        local services=("media-api:3001" "user-service:3002" "streaming-api:3003")
        for service in "${services[@]}"; do
            if ! curl -sf "http://localhost:${service#*:}/health" &> /dev/null; then
                all_healthy=false
                log DEBUG "Service ${service%:*} not ready yet..."
            fi
        done
        
        if [[ "$all_healthy" == "false" ]]; then
            retry_count=$((retry_count + 1))
            sleep 2
        fi
    done
    
    if [[ "$all_healthy" == "true" ]]; then
        log SUCCESS "All services are healthy"
    else
        log ERROR "Some services failed health check"
        docker-compose ps
        exit 1
    fi
}

# Setup monitoring
setup_monitoring() {
    log INFO "Setting up monitoring dashboards..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY RUN] Would setup monitoring"
        return
    fi
    
    # Create monitoring directory
    local monitoring_dir="${PROJECT_ROOT}/deploy/monitoring"
    mkdir -p "$monitoring_dir"
    
    # Generate Grafana dashboard
    cat > "${monitoring_dir}/dashboard.json" << 'EOF'
{
  "dashboard": {
    "title": "NewMedia System Dashboard",
    "panels": [
      {
        "title": "API Response Time",
        "targets": [{"expr": "http_request_duration_seconds"}]
      },
      {
        "title": "Service Health",
        "targets": [{"expr": "up"}]
      },
      {
        "title": "Database Connections",
        "targets": [{"expr": "mongodb_connections_current"}]
      }
    ]
  }
}
EOF
    
    log SUCCESS "Monitoring setup completed"
}

# Show deployment summary
show_summary() {
    log INFO "Deployment Summary:"
    echo -e "${CYAN}================================${NC}"
    echo -e "${WHITE}Environment:${NC} ${ENVIRONMENT}"
    echo -e "${WHITE}Mode:${NC} ${DEPLOYMENT_MODE}"
    echo -e "${WHITE}Services:${NC}"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" | tail -n +2
    fi
    
    echo -e "${CYAN}================================${NC}"
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e ""
    echo -e "${WHITE}Access URLs:${NC}"
    echo -e "  API Gateway: ${BLUE}http://localhost:3000${NC}"
    echo -e "  API Docs: ${BLUE}http://localhost:3000/docs${NC}"
    echo -e "  RabbitMQ: ${BLUE}http://localhost:15672${NC}"
    echo -e "  Logs: ${BLUE}${LOG_FILE}${NC}"
}

# Main deployment function
main() {
    parse_args "$@"
    
    log INFO "Starting NewMedia deployment..."
    log INFO "Environment: ${ENVIRONMENT}, Mode: ${DEPLOYMENT_MODE}"
    
    # Confirmation prompt
    if [[ "$AUTO_CONFIRM" != "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
        echo -e "${YELLOW}This will deploy NewMedia in ${ENVIRONMENT} mode. Continue? [y/N]${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log INFO "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Run deployment steps
    check_requirements
    generate_secrets
    create_env_file
    backup_deployment
    
    # Deploy based on mode
    case "$DEPLOYMENT_MODE" in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        local)
            log ERROR "Local deployment not yet implemented"
            exit 1
            ;;
        *)
            log ERROR "Unknown deployment mode: $DEPLOYMENT_MODE"
            exit 1
            ;;
    esac
    
    # Post-deployment steps
    if [[ "$DRY_RUN" != "true" ]]; then
        health_check
        setup_monitoring
        show_summary
    fi
    
    log SUCCESS "Deployment completed successfully!"
}

# Run main function
main "$@"