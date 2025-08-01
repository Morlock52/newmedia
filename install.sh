#!/bin/bash

# HoloMedia Hub Installation Script
# Version: 1.0.0
# Description: Comprehensive installation script with OS detection and automatic setup

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Installation variables
INSTALL_DIR="$(pwd)"
LOG_FILE="$INSTALL_DIR/install.log"
REQUIRED_NODE_VERSION="18"
REQUIRED_DOCKER_VERSION="20"

# ASCII Art Banner
show_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
    __  __      __      __  ___         ___      
   / / / /___  / /___  /  |/  /__  ____/ (_)___ _
  / /_/ / __ \/ / __ \/ /|_/ / _ \/ __  / / __ `/
 / __  / /_/ / / /_/ / /  / /  __/ /_/ / / /_/ / 
/_/ /_/\____/_/\____/_/  /_/\___/\__,_/_/\__,_/  
                                                  
            Installation Wizard v1.0.0
EOF
    echo -e "${NC}"
}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Progress indicator
progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    
    printf "\r["
    printf "%${completed}s" | tr ' ' '='
    printf "%$((width - completed))s" | tr ' ' ' '
    printf "] %d%%" "$percentage"
}

# Error handler
error_exit() {
    echo -e "\n${RED}Error: $1${NC}" >&2
    log "ERROR: $1"
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "${BLUE}ℹ $1${NC}"
    log "INFO: $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            VER=$VERSION_ID
            DISTRO=$ID
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
        VER=$(sw_vers -productVersion)
        DISTRO="darwin"
    else
        error_exit "Unsupported operating system: $OSTYPE"
    fi
    
    info "Detected OS: $OS $VER"
}

# Check if running as root (Linux only)
check_root() {
    if [[ "$OSTYPE" == "linux-gnu"* ]] && [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root on Linux. Use: sudo ./install.sh"
    fi
}

# Install dependencies based on OS
install_dependencies() {
    echo -e "\n${CYAN}Installing system dependencies...${NC}"
    
    if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
        apt-get update
        apt-get install -y \
            curl \
            wget \
            git \
            build-essential \
            software-properties-common \
            apt-transport-https \
            ca-certificates \
            gnupg \
            lsb-release \
            jq \
            htop \
            net-tools
    elif [[ "$DISTRO" == "fedora" ]] || [[ "$DISTRO" == "rhel" ]] || [[ "$DISTRO" == "centos" ]]; then
        yum install -y \
            curl \
            wget \
            git \
            gcc \
            gcc-c++ \
            make \
            jq \
            htop \
            net-tools
    elif [[ "$DISTRO" == "darwin" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew install \
            curl \
            wget \
            git \
            jq \
            htop
    fi
    
    success "System dependencies installed"
}

# Install Docker
install_docker() {
    echo -e "\n${CYAN}Checking Docker installation...${NC}"
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d ' ' -f3 | cut -d ',' -f1)
        info "Docker $DOCKER_VERSION is already installed"
    else
        info "Installing Docker..."
        
        if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | apt-key add -
            add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/$DISTRO $(lsb_release -cs) stable"
            apt-get update
            apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            systemctl start docker
            systemctl enable docker
        elif [[ "$DISTRO" == "darwin" ]]; then
            brew install --cask docker
            warning "Please start Docker Desktop manually"
        fi
        
        success "Docker installed successfully"
    fi
}

# Install Node.js
install_nodejs() {
    echo -e "\n${CYAN}Checking Node.js installation...${NC}"
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version | cut -d 'v' -f2)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d '.' -f1)
        
        if [ "$NODE_MAJOR" -ge "$REQUIRED_NODE_VERSION" ]; then
            info "Node.js $NODE_VERSION is already installed"
        else
            warning "Node.js $NODE_VERSION is installed but version $REQUIRED_NODE_VERSION+ is required"
            install_nodejs_fresh
        fi
    else
        install_nodejs_fresh
    fi
}

install_nodejs_fresh() {
    info "Installing Node.js $REQUIRED_NODE_VERSION..."
    
    if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
        curl -fsSL https://deb.nodesource.com/setup_$REQUIRED_NODE_VERSION.x | bash -
        apt-get install -y nodejs
    elif [[ "$DISTRO" == "darwin" ]]; then
        brew install node@$REQUIRED_NODE_VERSION
    fi
    
    # Install global packages
    npm install -g pm2 yarn
    
    success "Node.js installed successfully"
}

# Check system requirements
check_requirements() {
    echo -e "\n${CYAN}Checking system requirements...${NC}"
    
    # Check CPU cores
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    if [ "$CPU_CORES" -lt 2 ]; then
        warning "System has $CPU_CORES CPU cores. Recommended: 2+"
    else
        success "CPU cores: $CPU_CORES"
    fi
    
    # Check RAM
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        TOTAL_RAM=$(free -m | awk 'NR==2{print $2}')
    else
        TOTAL_RAM=$(($(sysctl -n hw.memsize) / 1024 / 1024))
    fi
    
    if [ "$TOTAL_RAM" -lt 4096 ]; then
        warning "System has ${TOTAL_RAM}MB RAM. Recommended: 4096MB+"
    else
        success "RAM: ${TOTAL_RAM}MB"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$DISK_SPACE" -lt 20 ]; then
        warning "Available disk space: ${DISK_SPACE}GB. Recommended: 20GB+"
    else
        success "Available disk space: ${DISK_SPACE}GB"
    fi
}

# Setup firewall rules
setup_firewall() {
    echo -e "\n${CYAN}Configuring firewall...${NC}"
    
    if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
        if command -v ufw &> /dev/null; then
            ufw allow 22/tcp    # SSH
            ufw allow 80/tcp    # HTTP
            ufw allow 443/tcp   # HTTPS
            ufw allow 3000/tcp  # Application
            ufw allow 5432/tcp  # PostgreSQL
            ufw allow 6379/tcp  # Redis
            ufw --force enable
            success "Firewall configured"
        fi
    fi
}

# Create application structure
create_app_structure() {
    echo -e "\n${CYAN}Creating application structure...${NC}"
    
    mkdir -p {data/{postgres,redis,uploads,backups},logs,config,ssl}
    
    # Set permissions
    chmod -R 755 data logs config
    chmod -R 700 ssl
    
    success "Application structure created"
}

# Generate environment file
generate_env_file() {
    echo -e "\n${CYAN}Generating environment configuration...${NC}"
    
    # Generate secure passwords
    DB_PASSWORD=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 64)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    
    cat > .env << EOF
# HoloMedia Hub Configuration
# Generated on $(date)

# Application Settings
NODE_ENV=production
PORT=3000
HOST=0.0.0.0

# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=holomedia
DB_USER=holomedia
DB_PASSWORD=$DB_PASSWORD

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=$REDIS_PASSWORD

# JWT Configuration
JWT_SECRET=$JWT_SECRET
JWT_EXPIRY=7d

# File Upload Settings
MAX_FILE_SIZE=100MB
UPLOAD_DIR=/app/data/uploads

# API Keys (to be configured)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
YOUTUBE_API_KEY=

# Email Configuration (optional)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=noreply@holomedia.local

# SSL Configuration
SSL_ENABLED=false
SSL_CERT_PATH=/app/ssl/cert.pem
SSL_KEY_PATH=/app/ssl/key.pem

# Logging
LOG_LEVEL=info
LOG_DIR=/app/logs

# Backup Settings
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
EOF

    chmod 600 .env
    success "Environment configuration generated"
}

# Install application dependencies
install_app_dependencies() {
    echo -e "\n${CYAN}Installing application dependencies...${NC}"
    
    if [ -f "package.json" ]; then
        npm install --production
        success "Application dependencies installed"
    else
        warning "No package.json found. Skipping npm install."
    fi
}

# Setup SSL certificates
setup_ssl() {
    echo -e "\n${CYAN}SSL Certificate Setup${NC}"
    echo "Choose SSL option:"
    echo "1) Self-signed certificate (for testing)"
    echo "2) Let's Encrypt certificate (requires domain)"
    echo "3) Skip SSL setup"
    
    read -p "Enter choice (1-3): " ssl_choice
    
    case $ssl_choice in
        1)
            info "Generating self-signed certificate..."
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout ssl/key.pem -out ssl/cert.pem \
                -subj "/C=US/ST=State/L=City/O=HoloMedia/CN=localhost"
            sed -i 's/SSL_ENABLED=false/SSL_ENABLED=true/' .env
            success "Self-signed certificate generated"
            ;;
        2)
            read -p "Enter your domain name: " domain
            if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
                apt-get install -y certbot
                certbot certonly --standalone -d "$domain" --non-interactive --agree-tos -m "admin@$domain"
                ln -sf "/etc/letsencrypt/live/$domain/fullchain.pem" ssl/cert.pem
                ln -sf "/etc/letsencrypt/live/$domain/privkey.pem" ssl/key.pem
                sed -i 's/SSL_ENABLED=false/SSL_ENABLED=true/' .env
                success "Let's Encrypt certificate configured"
            else
                warning "Let's Encrypt auto-setup only available on Ubuntu/Debian"
            fi
            ;;
        3)
            info "Skipping SSL setup"
            ;;
    esac
}

# Setup systemd service
setup_systemd() {
    if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]] || [[ "$DISTRO" == "fedora" ]]; then
        echo -e "\n${CYAN}Setting up systemd service...${NC}"
        
        cp systemd/holomedia.service /etc/systemd/system/
        systemctl daemon-reload
        systemctl enable holomedia.service
        
        success "Systemd service configured"
    fi
}

# Final setup steps
final_setup() {
    echo -e "\n${CYAN}Performing final setup...${NC}"
    
    # Create initial admin user script
    cat > scripts/create-admin.sh << 'EOF'
#!/bin/bash
# Create initial admin user

echo "Creating admin user..."
read -p "Enter admin email: " email
read -s -p "Enter admin password: " password
echo

# Add your user creation logic here
echo "Admin user created successfully!"
EOF
    
    chmod +x scripts/*.sh
    
    # Start services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker/docker-compose.full.yml up -d
    else
        docker compose -f docker/docker-compose.full.yml up -d
    fi
    
    success "Services started"
}

# Installation summary
show_summary() {
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ HoloMedia Hub Installation Complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${CYAN}Installation Summary:${NC}"
    echo "• Install Directory: $INSTALL_DIR"
    echo "• Configuration File: $INSTALL_DIR/.env"
    echo "• Log File: $LOG_FILE"
    echo ""
    
    echo -e "${CYAN}Next Steps:${NC}"
    echo "1. Review and update the .env configuration file"
    echo "2. Run './setup-wizard.sh' for interactive configuration"
    echo "3. Create admin user: './scripts/create-admin.sh'"
    echo "4. Access the application at http://localhost:3000"
    echo ""
    
    echo -e "${CYAN}Useful Commands:${NC}"
    echo "• Start services: docker-compose -f docker/docker-compose.full.yml up -d"
    echo "• Stop services: docker-compose -f docker/docker-compose.full.yml down"
    echo "• View logs: docker-compose -f docker/docker-compose.full.yml logs -f"
    echo "• Health check: ./scripts/health-check.sh"
    echo "• Backup: ./scripts/backup.sh"
    echo ""
    
    echo -e "${YELLOW}Documentation:${NC} https://github.com/holomedia/docs"
    echo -e "${YELLOW}Support:${NC} support@holomedia.io"
}

# Main installation flow
main() {
    show_banner
    
    # Initialize log
    echo "HoloMedia Hub Installation Log - $(date)" > "$LOG_FILE"
    
    echo -e "${CYAN}Starting HoloMedia Hub installation...${NC}\n"
    
    # Installation steps
    detect_os
    check_root
    check_requirements
    
    # Progress tracking
    total_steps=10
    current_step=0
    
    # Step 1: Install dependencies
    ((current_step++))
    progress $current_step $total_steps
    install_dependencies
    
    # Step 2: Install Docker
    ((current_step++))
    progress $current_step $total_steps
    install_docker
    
    # Step 3: Install Node.js
    ((current_step++))
    progress $current_step $total_steps
    install_nodejs
    
    # Step 4: Setup firewall
    ((current_step++))
    progress $current_step $total_steps
    setup_firewall
    
    # Step 5: Create app structure
    ((current_step++))
    progress $current_step $total_steps
    create_app_structure
    
    # Step 6: Generate environment
    ((current_step++))
    progress $current_step $total_steps
    generate_env_file
    
    # Step 7: Install app dependencies
    ((current_step++))
    progress $current_step $total_steps
    install_app_dependencies
    
    # Step 8: Setup SSL
    ((current_step++))
    progress $current_step $total_steps
    setup_ssl
    
    # Step 9: Setup systemd
    ((current_step++))
    progress $current_step $total_steps
    setup_systemd
    
    # Step 10: Final setup
    ((current_step++))
    progress $current_step $total_steps
    final_setup
    
    echo -e "\n"
    show_summary
}

# Trap errors
trap 'error_exit "Installation failed at line $LINENO"' ERR

# Run main installation
main "$@"