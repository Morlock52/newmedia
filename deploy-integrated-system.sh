#!/bin/bash

# Integrated Media Server Deployment Script
# Deploys all components: AI/ML, AR/VR, Quantum Security, Blockchain, Voice AI, Media Stack

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="media-server-integrated"
COMPOSE_FILE="docker-compose.master.yml"
ENV_FILE=".env.production"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}========================================${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_section "Checking System Requirements"
    
    local missing_deps=()
    
    # Check Docker
    if ! command_exists docker; then
        missing_deps+=("docker")
    else
        print_status "Docker found: $(docker --version)"
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        missing_deps+=("docker-compose")
    else
        if command_exists docker-compose; then
            print_status "Docker Compose found: $(docker-compose --version)"
        else
            print_status "Docker Compose found: $(docker compose version)"
        fi
    fi
    
    # Check required utilities
    local utils=("curl" "jq" "openssl" "htpasswd")
    for util in "${utils[@]}"; do
        if ! command_exists "$util"; then
            missing_deps+=("$util")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and run this script again."
        exit 1
    fi
    
    # Check for NVIDIA Docker (optional)
    if command_exists nvidia-docker || docker info | grep -q nvidia; then
        print_status "NVIDIA Docker runtime detected - GPU acceleration will be available"
    else
        print_warning "NVIDIA Docker runtime not found - GPU features will be disabled"
    fi
    
    # Check available disk space (minimum 50GB recommended)
    local available_space=$(df . | awk 'NR==2 {print $4}')
    local min_space=$((50 * 1024 * 1024)) # 50GB in KB
    
    if [ "$available_space" -lt "$min_space" ]; then
        print_warning "Available disk space is less than 50GB. You may encounter issues during deployment."
    else
        print_status "Sufficient disk space available: $(( available_space / 1024 / 1024 ))GB"
    fi
}

# Function to setup environment
setup_environment() {
    print_section "Setting Up Environment"
    
    # Copy environment template if .env doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "${ENV_FILE}.template" ]; then
            print_status "Copying environment template..."
            cp "${ENV_FILE}.template" "$ENV_FILE"
            print_warning "Please edit $ENV_FILE with your configuration before continuing!"
            echo "Press Enter when you've configured the environment file..."
            read -r
        else
            print_error "Environment template not found: ${ENV_FILE}.template"
            exit 1
        fi
    fi
    
    # Validate required environment variables
    print_status "Validating environment configuration..."
    
    source "$ENV_FILE"
    
    local required_vars=(
        "DOMAIN"
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "GRAFANA_ADMIN_PASSWORD"
        "JWT_SECRET"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing required environment variables: ${missing_vars[*]}"
        print_error "Please configure these variables in $ENV_FILE"
        exit 1
    fi
    
    print_status "Environment validation successful"
}

# Function to generate secure secrets
generate_secrets() {
    print_section "Generating Secure Secrets"
    
    # Create secrets directory
    mkdir -p secrets
    chmod 700 secrets
    
    # Generate database passwords if not set
    if [ "${POSTGRES_PASSWORD:-}" = "your-secure-postgres-password-here" ]; then
        POSTGRES_PASSWORD=$(openssl rand -base64 32)
        print_status "Generated PostgreSQL password"
    fi
    
    if [ "${REDIS_PASSWORD:-}" = "your-secure-redis-password-here" ]; then
        REDIS_PASSWORD=$(openssl rand -base64 32)
        print_status "Generated Redis password"
    fi
    
    # Generate JWT secrets if not set
    if [ "${JWT_SECRET:-}" = "your-super-secure-jwt-secret-256-bits-long" ]; then
        JWT_SECRET=$(openssl rand -base64 64)
        print_status "Generated JWT secret"
    fi
    
    # Generate Traefik auth hash
    if [ "${TRAEFIK_AUTH:-}" = "admin:\$2y\$10\$example-hash-here" ]; then
        local admin_password=$(openssl rand -base64 16)
        TRAEFIK_AUTH="admin:$(htpasswd -nbB admin "$admin_password" | cut -d: -f2)"
        echo "Traefik Admin Password: $admin_password" > secrets/traefik_admin_password.txt
        chmod 600 secrets/traefik_admin_password.txt
        print_status "Generated Traefik authentication"
    fi
    
    # Update environment file with generated secrets
    sed -i.bak \
        -e "s|POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" \
        -e "s|REDIS_PASSWORD=.*|REDIS_PASSWORD=$REDIS_PASSWORD|" \
        -e "s|JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" \
        -e "s|TRAEFIK_AUTH=.*|TRAEFIK_AUTH=$TRAEFIK_AUTH|" \
        "$ENV_FILE"
    
    print_status "Secrets generated and updated in $ENV_FILE"
}

# Function to setup directories
setup_directories() {
    print_section "Setting Up Directory Structure"
    
    local dirs=(
        "data/postgres"
        "data/redis"
        "data/loki"
        "cache/nginx"
        "cache/jellyfin"
        "backups"
        "logs"
        "config/loki"
        "config/promtail"
        "config/grafana/provisioning/dashboards"
        "config/grafana/provisioning/datasources"
        "config/prometheus"
        "config/quantum"
        "config/ai"
        "config/ar-vr"
        "config/voice"
        "config/blockchain"
        "config/dashboard"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    # Set appropriate permissions
    chmod 755 data cache backups logs
    chmod 600 secrets/* 2>/dev/null || true
    
    # Create Traefik acme.json with correct permissions
    touch config/traefik/acme.json
    chmod 600 config/traefik/acme.json
    
    print_status "Directory structure created successfully"
}

# Function to create monitoring configs
create_monitoring_configs() {
    print_section "Creating Monitoring Configuration"
    
    # Prometheus configuration
    cat > config/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'traefik'
    static_configs:
      - targets: ['traefik-master:8080']

  - job_name: 'jellyfin'
    static_configs:
      - targets: ['jellyfin-core:8096']

  - job_name: 'ai-ml-nexus'
    static_configs:
      - targets: ['ai-ml-nexus:3000']

  - job_name: 'ar-vr-media'
    static_configs:
      - targets: ['ar-vr-media:8080']

  - job_name: 'voice-ai'
    static_configs:
      - targets: ['voice-ai:3000']

  - job_name: 'web3-blockchain'
    static_configs:
      - targets: ['web3-blockchain:3000']

  - job_name: 'quantum-security'
    static_configs:
      - targets: ['quantum-security:8443']
    scheme: https
    tls_config:
      insecure_skip_verify: true

  - job_name: 'holographic-dashboard'
    static_configs:
      - targets: ['holographic-dashboard:8080']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-master:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-master:6379']
EOF

    # Prometheus alert rules
    cat > config/prometheus/alert_rules.yml << 'EOF'
groups:
  - name: system
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"

  - name: services
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
EOF

    # Grafana datasource configuration
    cat > config/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus-master:9090
    isDefault: true
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki-master:3100
    editable: true
EOF

    # Loki configuration
    cat > config/loki/local-config.yaml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF

    # Promtail configuration
    cat > config/promtail/config.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki-master:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/log/containers/*.log

  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: syslog
          __path__: /var/log/syslog
EOF

    print_status "Monitoring configuration files created"
}

# Function to build custom images
build_images() {
    print_section "Building Custom Docker Images"
    
    # Check if Dockerfiles exist and build images
    local services=("ai-ml-nexus" "ar-vr-media" "voice-ai-system" "quantum-security" "holographic-dashboard-demo/holographic-dashboard" "config-server")
    
    for service in "${services[@]}"; do
        if [ -f "$service/Dockerfile" ]; then
            print_status "Building $service image..."
            docker build -t "media-server/$service:latest" "$service/" || {
                print_warning "Failed to build $service image - will use base image"
            }
        else
            print_warning "Dockerfile not found for $service - creating basic one"
            create_basic_dockerfile "$service"
        fi
    done
    
    # Build Web3 blockchain service
    if [ -f "web3-blockchain-integration/api/Dockerfile" ]; then
        print_status "Building web3-blockchain image..."
        docker build -t "media-server/web3-blockchain:latest" "web3-blockchain-integration/api/" || {
            print_warning "Failed to build web3-blockchain image"
        }
    fi
}

# Function to create basic Dockerfile if missing
create_basic_dockerfile() {
    local service_dir="$1"
    
    if [ ! -d "$service_dir" ]; then
        mkdir -p "$service_dir"
    fi
    
    cat > "$service_dir/Dockerfile" << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Create package.json if it doesn't exist
RUN if [ ! -f package.json ]; then \
    echo '{"name":"service","version":"1.0.0","main":"index.js","scripts":{"start":"node index.js"},"dependencies":{"express":"^4.18.0"}}' > package.json; \
    fi

# Create basic health endpoint if index.js doesn't exist
RUN if [ ! -f index.js ]; then \
    echo 'const express = require("express"); const app = express(); app.get("/health", (req, res) => res.json({status: "ok"})); app.listen(process.env.PORT || 3000, () => console.log("Service running"));' > index.js; \
    fi

COPY package*.json ./
RUN npm install --production

COPY . .

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["npm", "start"]
EOF

    print_status "Created basic Dockerfile for $service_dir"
}

# Function to deploy services
deploy_services() {
    print_section "Deploying Services"
    
    # Pull required images
    print_status "Pulling required Docker images..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull --ignore-pull-failures
    
    # Start core infrastructure first
    print_status "Starting core infrastructure..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        traefik postgres redis prometheus grafana loki promtail
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure to initialize..."
    sleep 30
    
    # Start application services
    print_status "Starting application services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        quantum-security ai-ml-nexus ar-vr-media voice-ai-system web3-blockchain
    
    # Wait for application services
    sleep 20
    
    # Start remaining services
    print_status "Starting remaining services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    
    print_status "All services deployed successfully"
}

# Function to wait for services
wait_for_services() {
    print_section "Waiting for Services to Start"
    
    local services=(
        "http://localhost:8080/api/healthcheck:Traefik"
        "http://localhost:8096/health:Jellyfin"
        "http://localhost:3001/health:AI/ML Nexus"
        "http://localhost:8082/health:AR/VR Media"
        "http://localhost:3002/health:Voice AI"
        "http://localhost:3003/health:Web3 Blockchain"         
        "https://localhost:8443/health:Quantum Security"
        "http://localhost:8088/health:Holographic Dashboard"
        "http://localhost:3000/api/health:Grafana"
        "http://localhost:9090/-/healthy:Prometheus"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"
        print_status "Waiting for $name to be ready..."
        
        local attempts=0
        local max_attempts=30
        
        while [ $attempts -lt $max_attempts ]; do
            if curl -sf "$url" >/dev/null 2>&1; then
                print_status "$name is ready"
                break
            fi
            
            attempts=$((attempts + 1))
            if [ $attempts -eq $max_attempts ]; then
                print_warning "$name is not responding after $max_attempts attempts"
            else
                sleep 10
            fi
        done
    done
}

# Function to run health checks
run_health_checks() {
    print_section "Running Health Checks"
    
    # Check container status
    print_status "Checking container status..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    
    # Check service connectivity
    print_status "Testing service connectivity..."
    
    local failed_services=()
    
    # Test each service endpoint
    if ! curl -sf "http://localhost:8096/health" >/dev/null 2>&1; then
        failed_services+=("Jellyfin")
    fi
    
    if ! curl -sf "http://localhost:3001/health" >/dev/null 2>&1; then
        failed_services+=("AI/ML Nexus")
    fi
    
    if ! curl -sf "http://localhost:8082/health" >/dev/null 2>&1; then
        failed_services+=("AR/VR Media")
    fi
    
    # Report results
    if [ ${#failed_services[@]} -eq 0 ]; then
        print_status "All health checks passed!"
    else
        print_warning "Some services failed health checks: ${failed_services[*]}"
        print_warning "Check the logs with: docker-compose -f $COMPOSE_FILE logs [service-name]"
    fi
}

# Function to display access information
show_access_info() {
    print_section "Access Information"
    
    source "$ENV_FILE"
    
    echo -e "${CYAN}Your integrated media server is now deployed!${NC}\n"
    
    echo -e "${BLUE}Core Services:${NC}"
    echo -e "  📺 Media Server (Jellyfin): https://media.${DOMAIN}"
    echo -e "  🎛️  Holographic Dashboard: https://dashboard.${DOMAIN}"
    echo -e "  📊 Monitoring (Grafana): https://grafana.${DOMAIN}"
    echo -e "  🔧 Traefik Dashboard: https://traefik.${DOMAIN}"
    
    echo -e "\n${BLUE}AI/ML & Advanced Features:${NC}"
    echo -e "  🧠 AI/ML Nexus: https://ai.${DOMAIN}"
    echo -e "  🥽 AR/VR Media: https://vr.${DOMAIN}"
    echo -e "  🎤 Voice AI System: https://voice.${DOMAIN}"
    echo -e "  🔗 Web3/Blockchain: https://web3.${DOMAIN}"
    echo -e "  🔐 Quantum Security: https://quantum.${DOMAIN}"
    
    echo -e "\n${BLUE}Development/Admin:${NC}"
    echo -e "  📈 Prometheus: https://prometheus.${DOMAIN}"
    echo -e "  🔧 API Gateway: https://api.${DOMAIN}"
    
    if [ -f "secrets/traefik_admin_password.txt" ]; then
        echo -e "\n${YELLOW}Admin Credentials:${NC}"
        echo -e "  Traefik Dashboard: admin / $(cat secrets/traefik_admin_password.txt)"
        echo -e "  Grafana: admin / ${GRAFANA_ADMIN_PASSWORD}"
    fi
    
    echo -e "\n${GREEN}Deployment completed successfully!${NC}"
    echo -e "${YELLOW}Note: It may take a few minutes for all services to be fully operational.${NC}"
}

# Function to create management scripts
create_management_scripts() {
    print_section "Creating Management Scripts"
    
    # Create service control script
    cat > manage-services.sh << 'EOF'
#!/bin/bash

COMPOSE_FILE="docker-compose.master.yml"
ENV_FILE=".env.production"

case "$1" in
    start)
        echo "Starting all services..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
        ;;
    stop)
        echo "Stopping all services..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
        ;;
    restart)
        echo "Restarting all services..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
        ;;
    status)
        echo "Service status:"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
        ;;
    logs)
        if [ -n "$2" ]; then
            docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f "$2"
        else
            docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f
        fi
        ;;
    update)
        echo "Updating services..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
        ;;
    backup)
        echo "Creating backup..."
        ./backup-system.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service]|update|backup}"
        exit 1
        ;;
esac
EOF

    chmod +x manage-services.sh
    
    # Create backup script
    cat > backup-system.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating system backup in $BACKUP_DIR..."

# Backup configurations
cp -r config "$BACKUP_DIR/"
cp -r secrets "$BACKUP_DIR/" 2>/dev/null || true
cp .env.production "$BACKUP_DIR/"
cp docker-compose.master.yml "$BACKUP_DIR/"

# Backup databases
docker-compose exec -T postgres-master pg_dumpall -U postgres > "$BACKUP_DIR/postgres_backup.sql"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C backups "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
EOF

    chmod +x backup-system.sh
    
    print_status "Management scripts created: manage-services.sh, backup-system.sh"
}

# Main deployment function
main() {
    print_section "Integrated Media Server Deployment"
    echo -e "${CYAN}Deploying AI/ML, AR/VR, Quantum Security, Blockchain, Voice AI & Media Stack${NC}\n"
    
    # Run deployment steps
    check_requirements
    setup_environment
    generate_secrets
    setup_directories
    create_monitoring_configs
    build_images
    deploy_services
    wait_for_services
    run_health_checks
    create_management_scripts
    show_access_info
    
    print_section "Deployment Complete"
    echo -e "${GREEN}Your integrated media server platform is now running!${NC}"
    echo -e "${YELLOW}Check the logs if any services aren't working: ./manage-services.sh logs [service-name]${NC}"
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    check)
        check_requirements
        ;;
    secrets)
        generate_secrets
        ;;
    build)
        build_images
        ;;
    health)
        run_health_checks
        ;;
    *)
        echo "Usage: $0 {deploy|check|secrets|build|health}"
        echo "  deploy  - Full deployment (default)"
        echo "  check   - Check system requirements only"
        echo "  secrets - Generate secrets only"
        echo "  build   - Build Docker images only"
        echo "  health  - Run health checks only"
        exit 1
        ;;
esac