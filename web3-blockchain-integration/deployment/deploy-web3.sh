#!/bin/bash

# Web3 Media Platform Deployment Script
# Deploys the complete Web3 blockchain integration with existing media server

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
WEB3_DIR="$PROJECT_ROOT/web3-blockchain-integration"
ENV_FILE="$PROJECT_ROOT/.env"
WEB3_ENV_FILE="$WEB3_DIR/.env.web3"

# Default values
NETWORK="mainnet"
DEPLOY_CONTRACTS=false
SKIP_FRONTEND=false
PRODUCTION_MODE=false
BACKUP_CONFIG=true

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

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Check Node.js (for smart contract deployment)
    if ! command -v node &> /dev/null; then
        missing_deps+=("nodejs")
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        missing_deps+=("npm")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up Web3 environment..."
    
    # Create Web3 environment file if it doesn't exist
    if [ ! -f "$WEB3_ENV_FILE" ]; then
        print_status "Creating Web3 environment file..."
        cat > "$WEB3_ENV_FILE" << EOF
# Web3 Media Platform Configuration

# Blockchain Network Configuration
NETWORK=$NETWORK
ETHEREUM_RPC_URL=https://rpc.ankr.com/eth
POLYGON_RPC_URL=https://rpc.ankr.com/polygon
BSC_RPC_URL=https://rpc.ankr.com/bsc
AVALANCHE_RPC_URL=https://rpc.ankr.com/avalanche
ARBITRUM_RPC_URL=https://rpc.ankr.com/arbitrum
OPTIMISM_RPC_URL=https://rpc.ankr.com/optimism

# Smart Contract Addresses (will be populated after deployment)
CONTENT_OWNERSHIP_ADDRESS=
MEDIA_DAO_ADDRESS=
MARKETPLACE_ADDRESS=

# IPFS Configuration
IPFS_SWARM_KEY=$(openssl rand -hex 32)
CLUSTER_SECRET=$(openssl rand -hex 32)

# Database Configuration
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Security Configuration
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -hex 32)

# API Keys (fill in with your own)
ETHERSCAN_API_KEY=
POLYGONSCAN_API_KEY=
BSCSCAN_API_KEY=
INFURA_PROJECT_ID=
ALCHEMY_API_KEY=

# Backup Configuration (for production)
BACKUP_S3_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

# Monitoring
SECURITY_ALERT_WEBHOOK=

# Development/Testing
DEPLOYER_PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000001
EOF
        print_success "Web3 environment file created at $WEB3_ENV_FILE"
        print_warning "Please edit $WEB3_ENV_FILE and fill in your API keys and configuration"
    fi
    
    # Source both environment files
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    set -a
    source "$WEB3_ENV_FILE"
    set +a
    
    # Create necessary directories
    mkdir -p "$WEB3_DIR"/{ipfs-integration,smart-contracts,web3-frontend,integration}
    mkdir -p "$PROJECT_ROOT"/{backups/ipfs,logs/web3}
    
    print_success "Environment setup complete"
}

# Function to build Docker images
build_images() {
    print_status "Building Web3 Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build Web3 API image
    print_status "Building Web3 API image..."
    docker build -t web3-media-api:latest \
        -f "$WEB3_DIR/api/Dockerfile" \
        "$WEB3_DIR/api/"
    
    # Build Web3 Frontend image
    if [ "$SKIP_FRONTEND" = false ]; then
        print_status "Building Web3 Frontend image..."
        docker build -t web3-media-frontend:latest \
            --build-arg REACT_APP_CONTENT_OWNERSHIP_ADDRESS="$CONTENT_OWNERSHIP_ADDRESS" \
            --build-arg REACT_APP_MEDIA_DAO_ADDRESS="$MEDIA_DAO_ADDRESS" \
            --build-arg REACT_APP_MARKETPLACE_ADDRESS="$MARKETPLACE_ADDRESS" \
            -f "$WEB3_DIR/web3-frontend/Dockerfile" \
            "$WEB3_DIR/web3-frontend/"
    fi
    
    # Build other supporting images
    docker build -t web3-analytics:latest \
        -f "$WEB3_DIR/analytics/Dockerfile" \
        "$WEB3_DIR/analytics/"
    
    docker build -t event-indexer:latest \
        -f "$WEB3_DIR/indexer/Dockerfile" \
        "$WEB3_DIR/indexer/"
    
    if [ "$PRODUCTION_MODE" = true ]; then
        docker build -t ipfs-backup:latest \
            -f "$WEB3_DIR/backup/Dockerfile" \
            "$WEB3_DIR/backup/"
        
        docker build -t security-scanner:latest \
            -f "$WEB3_DIR/security/Dockerfile" \
            "$WEB3_DIR/security/"
    fi
    
    print_success "Docker images built successfully"
}

# Function to deploy smart contracts
deploy_contracts() {
    if [ "$DEPLOY_CONTRACTS" = false ]; then
        print_status "Skipping smart contract deployment"
        return
    fi
    
    print_status "Deploying smart contracts to $NETWORK..."
    
    cd "$WEB3_DIR/smart-contracts"
    
    # Install dependencies
    if [ ! -d "node_modules" ]; then
        print_status "Installing smart contract dependencies..."
        npm install
    fi
    
    # Compile contracts
    print_status "Compiling smart contracts..."
    npx hardhat compile
    
    # Deploy contracts
    print_status "Deploying contracts to $NETWORK..."
    if [ "$NETWORK" = "localhost" ]; then
        # Start local blockchain first
        print_status "Starting local blockchain..."
        npx hardhat node --hostname 0.0.0.0 &
        HARDHAT_PID=$!
        sleep 10
    fi
    
    # Deploy contracts and capture addresses
    DEPLOYMENT_OUTPUT=$(npx hardhat run scripts/deploy.js --network "$NETWORK")
    
    # Extract contract addresses (assuming deploy script outputs them)
    CONTENT_OWNERSHIP_ADDRESS=$(echo "$DEPLOYMENT_OUTPUT" | grep "ContentOwnership deployed to:" | awk '{print $4}')
    MEDIA_DAO_ADDRESS=$(echo "$DEPLOYMENT_OUTPUT" | grep "MediaDAO deployed to:" | awk '{print $4}')
    MARKETPLACE_ADDRESS=$(echo "$DEPLOYMENT_OUTPUT" | grep "CrossChainMarketplace deployed to:" | awk '{print $4}')
    
    # Update environment file with contract addresses
    if [ -n "$CONTENT_OWNERSHIP_ADDRESS" ] && [ -n "$MEDIA_DAO_ADDRESS" ] && [ -n "$MARKETPLACE_ADDRESS" ]; then
        print_status "Updating environment with contract addresses..."
        sed -i.bak "s/CONTENT_OWNERSHIP_ADDRESS=.*/CONTENT_OWNERSHIP_ADDRESS=$CONTENT_OWNERSHIP_ADDRESS/" "$WEB3_ENV_FILE"
        sed -i.bak "s/MEDIA_DAO_ADDRESS=.*/MEDIA_DAO_ADDRESS=$MEDIA_DAO_ADDRESS/" "$WEB3_ENV_FILE"
        sed -i.bak "s/MARKETPLACE_ADDRESS=.*/MARKETPLACE_ADDRESS=$MARKETPLACE_ADDRESS/" "$WEB3_ENV_FILE"
        rm "$WEB3_ENV_FILE.bak"
        
        print_success "Smart contracts deployed successfully:"
        print_success "  ContentOwnership: $CONTENT_OWNERSHIP_ADDRESS"
        print_success "  MediaDAO: $MEDIA_DAO_ADDRESS"
        print_success "  CrossChainMarketplace: $MARKETPLACE_ADDRESS"
    else
        print_error "Failed to extract contract addresses from deployment"
        exit 1
    fi
    
    # Kill local blockchain if we started it
    if [ "$NETWORK" = "localhost" ] && [ -n "${HARDHAT_PID:-}" ]; then
        kill $HARDHAT_PID || true
    fi
    
    cd "$PROJECT_ROOT"
}

# Function to setup IPFS configuration
setup_ipfs() {
    print_status "Setting up IPFS configuration..."
    
    # Create IPFS configuration directory
    mkdir -p "$WEB3_DIR/ipfs-integration/config"
    
    # Create IPFS swarm key for private network (optional)
    if [ -n "${IPFS_SWARM_KEY:-}" ]; then
        cat > "$WEB3_DIR/ipfs-integration/config/swarm.key" << EOF
/key/swarm/psk/1.0.0/
/base16/
$IPFS_SWARM_KEY
EOF
    fi
    
    # Create IPFS cluster configuration
    mkdir -p "$WEB3_DIR/ipfs-integration/cluster"
    
    print_success "IPFS configuration complete"
}

# Function to initialize database
init_database() {
    print_status "Initializing Web3 database..."
    
    # Create database initialization script
    mkdir -p "$WEB3_DIR/database/init"
    
    cat > "$WEB3_DIR/database/init/01-create-tables.sql" << EOF
-- Web3 Media Platform Database Schema

-- Content mappings table
CREATE TABLE IF NOT EXISTS content_mappings (
    id SERIAL PRIMARY KEY,
    jellyfin_item_id VARCHAR(255) UNIQUE NOT NULL,
    token_id VARCHAR(255) NOT NULL,
    ipfs_hash VARCHAR(255) NOT NULL,
    blockchain VARCHAR(50) NOT NULL DEFAULT 'ethereum',
    requires_license BOOLEAN DEFAULT false,
    requires_nft BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    wallet_address VARCHAR(42) NOT NULL,
    jellyfin_user_id VARCHAR(255),
    authenticated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    active BOOLEAN DEFAULT true
);

-- License cache table
CREATE TABLE IF NOT EXISTS license_cache (
    id SERIAL PRIMARY KEY,
    token_id VARCHAR(255) NOT NULL,
    wallet_address VARCHAR(42) NOT NULL,
    has_license BOOLEAN NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(token_id, wallet_address)
);

-- Blockchain events table
CREATE TABLE IF NOT EXISTS blockchain_events (
    id SERIAL PRIMARY KEY,
    contract_address VARCHAR(42) NOT NULL,
    event_name VARCHAR(100) NOT NULL,
    block_number BIGINT NOT NULL,
    transaction_hash VARCHAR(66) NOT NULL,
    log_index INTEGER NOT NULL,
    event_data JSONB,
    processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(transaction_hash, log_index)
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    metadata JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_content_mappings_token_id ON content_mappings(token_id);
CREATE INDEX IF NOT EXISTS idx_content_mappings_ipfs_hash ON content_mappings(ipfs_hash);
CREATE INDEX IF NOT EXISTS idx_user_sessions_wallet ON user_sessions(wallet_address);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(active, expires_at);
CREATE INDEX IF NOT EXISTS idx_license_cache_lookup ON license_cache(token_id, wallet_address, expires_at);
CREATE INDEX IF NOT EXISTS idx_blockchain_events_contract ON blockchain_events(contract_address, event_name);
CREATE INDEX IF NOT EXISTS idx_blockchain_events_block ON blockchain_events(block_number);
CREATE INDEX IF NOT EXISTS idx_blockchain_events_processed ON blockchain_events(processed);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_time ON performance_metrics(metric_name, recorded_at);
EOF
    
    print_success "Database initialization script created"
}

# Function to start Web3 services
start_services() {
    print_status "Starting Web3 services..."
    
    cd "$PROJECT_ROOT"
    
    # Create the docker-compose command
    COMPOSE_CMD="docker-compose -f docker-compose.yml -f web3-blockchain-integration/docker-compose.web3.yml"
    
    # Add production profile if in production mode
    if [ "$PRODUCTION_MODE" = true ]; then
        COMPOSE_CMD="$COMPOSE_CMD --profile production"
    fi
    
    # Start the services
    print_status "Starting infrastructure services..."
    $COMPOSE_CMD up -d postgres redis ipfs-node
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure to be ready..."
    sleep 30
    
    # Start IPFS cluster
    print_status "Starting IPFS cluster..."
    $COMPOSE_CMD up -d ipfs-cluster
    sleep 15
    
    # Start Web3 API
    print_status "Starting Web3 API..."
    $COMPOSE_CMD up -d web3-api
    sleep 10
    
    # Start frontend if not skipped
    if [ "$SKIP_FRONTEND" = false ]; then
        print_status "Starting Web3 frontend..."
        $COMPOSE_CMD up -d web3-frontend
    fi
    
    # Start analytics and indexer
    print_status "Starting analytics and event indexer..."
    $COMPOSE_CMD up -d web3-analytics event-indexer
    
    # Start backup services in production
    if [ "$PRODUCTION_MODE" = true ]; then
        print_status "Starting production services..."
        $COMPOSE_CMD up -d ipfs-backup security-scanner
    fi
    
    print_success "Web3 services started successfully"
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying Web3 deployment..."
    
    local services_healthy=true
    
    # Check IPFS
    if ! curl -f "http://localhost:5001/api/v0/version" &> /dev/null; then
        print_error "IPFS node is not responding"
        services_healthy=false
    else
        print_success "IPFS node is healthy"
    fi
    
    # Check Web3 API
    if ! curl -f "http://localhost:3030/health" &> /dev/null; then
        print_error "Web3 API is not responding"
        services_healthy=false
    else
        print_success "Web3 API is healthy"
    fi
    
    # Check frontend if not skipped
    if [ "$SKIP_FRONTEND" = false ]; then
        if ! curl -f "http://localhost:3031" &> /dev/null; then
            print_error "Web3 frontend is not responding"
            services_healthy=false
        else
            print_success "Web3 frontend is healthy"
        fi
    fi
    
    # Check database connection
    if ! docker exec web3-postgres pg_isready -U postgres &> /dev/null; then
        print_error "PostgreSQL database is not ready"
        services_healthy=false
    else
        print_success "PostgreSQL database is healthy"
    fi
    
    if [ "$services_healthy" = false ]; then
        print_error "Some services are not healthy. Check the logs with:"
        print_error "docker-compose -f docker-compose.yml -f web3-blockchain-integration/docker-compose.web3.yml logs"
        exit 1
    fi
    
    print_success "All Web3 services are healthy and running"
}

# Function to show deployment summary
show_summary() {
    print_success "🎉 Web3 Media Platform deployment completed successfully!"
    echo
    echo "📊 Service URLs:"
    echo "  • Web3 Frontend: http://localhost:3031"
    echo "  • Web3 API: http://localhost:3030"
    echo "  • IPFS Gateway: http://localhost:8080"
    echo "  • IPFS API: http://localhost:5001"
    echo "  • Analytics Dashboard: http://localhost:3032"
    echo "  • Existing Jellyfin: http://localhost:8096"
    echo "  • Existing Homepage: http://localhost:3001"
    echo
    echo "📄 Important Files:"
    echo "  • Web3 Environment: $WEB3_ENV_FILE"
    echo "  • Docker Compose: $WEB3_DIR/docker-compose.web3.yml"
    echo "  • Logs Directory: $PROJECT_ROOT/logs/web3"
    echo
    if [ -n "${CONTENT_OWNERSHIP_ADDRESS:-}" ]; then
        echo "📜 Smart Contract Addresses:"
        echo "  • Content Ownership: $CONTENT_OWNERSHIP_ADDRESS"
        echo "  • Media DAO: $MEDIA_DAO_ADDRESS"
        echo "  • Cross-Chain Marketplace: $MARKETPLACE_ADDRESS"
        echo
    fi
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose -f docker-compose.yml -f web3-blockchain-integration/docker-compose.web3.yml logs -f"
    echo "  • Stop services: docker-compose -f docker-compose.yml -f web3-blockchain-integration/docker-compose.web3.yml down"
    echo "  • Restart services: docker-compose -f docker-compose.yml -f web3-blockchain-integration/docker-compose.web3.yml restart"
    echo
    echo "⚠️  Next Steps:"
    echo "  1. Review and update $WEB3_ENV_FILE with your API keys"
    echo "  2. Configure your Web3 wallet to connect to the platform"
    echo "  3. Upload your first content as an NFT"
    echo "  4. Set up DAO governance proposals"
    echo
    if [ "$NETWORK" = "mainnet" ]; then
        print_warning "You are running on mainnet - ensure all configurations are secure!"
    fi
}

# Function to backup existing configuration
backup_config() {
    if [ "$BACKUP_CONFIG" = true ] && [ -f "$ENV_FILE" ]; then
        print_status "Backing up existing configuration..."
        cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        print_success "Configuration backed up"
    fi
}

# Function to cleanup on error
cleanup_on_error() {
    print_error "Deployment failed. Cleaning up..."
    
    # Stop any running services
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.yml -f web3-blockchain-integration/docker-compose.web3.yml down || true
    
    # Kill any background processes
    if [ -n "${HARDHAT_PID:-}" ]; then
        kill $HARDHAT_PID || true
    fi
    
    exit 1
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -n, --network NETWORK     Blockchain network (mainnet, testnet, localhost) [default: mainnet]"
    echo "  -c, --deploy-contracts    Deploy smart contracts"
    echo "  -s, --skip-frontend       Skip frontend deployment"
    echo "  -p, --production          Enable production mode"
    echo "  --no-backup              Skip configuration backup"
    echo "  -h, --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0                        # Basic deployment with existing contracts"
    echo "  $0 -c -n localhost        # Deploy contracts on localhost"
    echo "  $0 -p -n mainnet          # Production deployment on mainnet"
    echo "  $0 -s                     # Deploy without frontend"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--network)
            NETWORK="$2"
            shift 2
            ;;
        -c|--deploy-contracts)
            DEPLOY_CONTRACTS=true
            shift
            ;;
        -s|--skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        -p|--production)
            PRODUCTION_MODE=true
            shift
            ;;
        --no-backup)
            BACKUP_CONFIG=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Trap for cleanup on error
trap cleanup_on_error ERR

# Main deployment flow
main() {
    print_status "🚀 Starting Web3 Media Platform deployment..."
    echo
    
    # Pre-deployment checks and setup
    check_dependencies
    backup_config
    setup_environment
    setup_ipfs
    init_database
    
    # Smart contract deployment
    if [ "$DEPLOY_CONTRACTS" = true ]; then
        deploy_contracts
    fi
    
    # Build and deploy services
    build_images
    start_services
    
    # Post-deployment verification
    sleep 30  # Give services time to fully start
    verify_deployment
    
    # Show summary
    show_summary
}

# Run main function
main "$@"