#!/bin/bash
# Media Server Quick Start Script
# This script provides an automated setup for the enhanced media server stack

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/mediaserver"
DOMAIN="media.example.com"
GITHUB_REPO="https://github.com/yourusername/media-server-architecture.git"

# Functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_requirements() {
    print_status "Checking system requirements..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root!"
        exit 1
    fi
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        print_error "Cannot detect OS version"
        exit 1
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG /opt | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 100 ]]; then
        print_warning "Less than 100GB available in /opt. Recommended: 500GB+"
    fi
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk 'NR==2 {print $2}')
    if [[ $TOTAL_RAM -lt 16 ]]; then
        print_warning "Less than 16GB RAM detected. Recommended: 32GB+"
    fi
    
    print_status "System requirements check completed"
}

install_dependencies() {
    print_status "Installing dependencies..."
    
    sudo apt update
    sudo apt install -y \
        curl \
        git \
        htop \
        iotop \
        jq \
        ncdu \
        net-tools \
        software-properties-common \
        unzip \
        wget
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        print_status "Installing Docker..."
        curl -fsSL https://get.docker.com | sudo bash
        sudo usermod -aG docker $USER
        print_warning "You need to log out and back in for Docker group changes to take effect"
    fi
    
    # Install Docker Compose if not present
    if ! command -v docker-compose &> /dev/null; then
        print_status "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    print_status "Dependencies installed successfully"
}

setup_directory_structure() {
    print_status "Setting up directory structure..."
    
    sudo mkdir -p $INSTALL_DIR/{config,data,backups,cache,logs,architecture}
    sudo mkdir -p $INSTALL_DIR/config/{traefik,authelia,jellyfin,navidrome,immich,grafana,prometheus,homepage}
    sudo mkdir -p $INSTALL_DIR/data/{media,torrents,elasticsearch,postgres,redis}
    sudo mkdir -p $INSTALL_DIR/data/media/{movies,tv,music,audiobooks,podcasts,books,comics,photos}
    sudo mkdir -p $INSTALL_DIR/logs/{traefik,authelia,services}
    
    # Set ownership
    sudo chown -R $USER:$USER $INSTALL_DIR
    
    print_status "Directory structure created"
}

create_docker_networks() {
    print_status "Creating Docker networks..."
    
    docker network create --driver=bridge --subnet=10.10.0.0/24 proxy_network 2>/dev/null || true
    docker network create --driver=bridge --subnet=10.10.1.0/24 media_network 2>/dev/null || true
    docker network create --driver=bridge --subnet=10.10.2.0/24 admin_network 2>/dev/null || true
    docker network create --driver=bridge --subnet=10.10.3.0/24 data_network 2>/dev/null || true
    
    print_status "Docker networks created"
}

generate_secrets() {
    print_status "Generating secrets..."
    
    if [[ ! -f $INSTALL_DIR/.env ]]; then
        cat > $INSTALL_DIR/.env << EOF
# Auto-generated secrets - $(date)
DOMAIN=$DOMAIN
TZ=$(timedatectl show -p Timezone --value)

# Authentication
AUTHELIA_JWT_SECRET=$(openssl rand -hex 32)
AUTHELIA_SESSION_SECRET=$(openssl rand -hex 32)
AUTHELIA_STORAGE_ENCRYPTION_KEY=$(openssl rand -hex 32)
OIDC_HMAC_SECRET=$(openssl rand -hex 32)

# Databases
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
POSTGRES_AUTHELIA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
IMMICH_DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Services
GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-12)
MEILI_MASTER_KEY=$(openssl rand -hex 32)

# Email (update these!)
EMAIL_FROM=notifications@$DOMAIN
EMAIL_TO=admin@$DOMAIN
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# API Keys (update these!)
JELLYFIN_API_KEY=generate_after_setup
SONARR_API_KEY=generate_after_setup
RADARR_API_KEY=generate_after_setup
PROWLARR_API_KEY=generate_after_setup
EOF
        print_status "Secrets generated in $INSTALL_DIR/.env"
        print_warning "Please update email settings and API keys in .env file"
    else
        print_warning ".env file already exists, skipping secret generation"
    fi
}

setup_configurations() {
    print_status "Setting up configurations..."
    
    # Copy architecture files
    cp -r architecture/* $INSTALL_DIR/architecture/
    
    # Setup Traefik
    cp $INSTALL_DIR/architecture/configs/traefik.yml $INSTALL_DIR/config/traefik/
    mkdir -p $INSTALL_DIR/config/traefik/dynamic
    cp $INSTALL_DIR/architecture/configs/traefik-dynamic.yml $INSTALL_DIR/config/traefik/dynamic/
    
    # Setup Authelia
    cp $INSTALL_DIR/architecture/configs/authelia-configuration.yml $INSTALL_DIR/config/authelia/
    
    # Create basic users database
    cat > $INSTALL_DIR/config/authelia/users_database.yml << EOF
users:
  admin:
    displayname: "Admin User"
    password: "\$argon2id\$v=19\$m=65536,t=3,p=4\$BpLnfgDsc2WD8F2q\$zqU0pJayJa6DWeJvJqPg5qF4v1kPvF5gPzfRc7lWUkI"  # password: admin (CHANGE THIS!)
    email: admin@$DOMAIN
    groups:
      - admins
      - users
EOF
    
    # Setup Prometheus
    cp $INSTALL_DIR/architecture/configs/prometheus.yml $INSTALL_DIR/config/prometheus/
    mkdir -p $INSTALL_DIR/config/prometheus/rules
    cp $INSTALL_DIR/architecture/configs/prometheus-alerts.yml $INSTALL_DIR/config/prometheus/rules/
    
    print_status "Configurations set up"
}

create_systemd_service() {
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/mediaserver.service > /dev/null << EOF
[Unit]
Description=Media Server Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/docker-compose -f $INSTALL_DIR/architecture/docker-compose-enhanced.yml up -d
ExecStop=/usr/local/bin/docker-compose -f $INSTALL_DIR/architecture/docker-compose-enhanced.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable mediaserver.service
    
    print_status "Systemd service created"
}

optimize_system() {
    print_status "Optimizing system settings..."
    
    # Increase file limits
    if ! grep -q "fs.file-max" /etc/sysctl.conf; then
        echo "fs.file-max = 65536" | sudo tee -a /etc/sysctl.conf
    fi
    
    # Increase map count for Elasticsearch
    if ! grep -q "vm.max_map_count" /etc/sysctl.conf; then
        echo "vm.max_map_count = 262144" | sudo tee -a /etc/sysctl.conf
    fi
    
    # Apply settings
    sudo sysctl -p
    
    # Configure Docker daemon
    sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF
    
    sudo systemctl restart docker
    
    print_status "System optimized"
}

setup_firewall() {
    print_status "Setting up firewall..."
    
    # Check if ufw is installed
    if command -v ufw &> /dev/null; then
        sudo ufw --force enable
        sudo ufw allow 22/tcp    # SSH
        sudo ufw allow 80/tcp    # HTTP
        sudo ufw allow 443/tcp   # HTTPS
        sudo ufw allow 51820/udp # WireGuard
        
        print_status "Firewall configured"
    else
        print_warning "UFW not installed, skipping firewall setup"
    fi
}

start_services() {
    print_status "Starting services..."
    
    cd $INSTALL_DIR
    
    # Start core infrastructure first
    print_status "Starting infrastructure services..."
    docker-compose -f architecture/docker-compose-enhanced.yml up -d \
        traefik authelia postgres redis elasticsearch
    
    sleep 20
    
    # Start media services
    print_status "Starting media services..."
    docker-compose -f architecture/docker-compose-enhanced.yml up -d \
        jellyfin navidrome audiobookshelf kavita
    
    # Start support services
    print_status "Starting support services..."
    docker-compose -f architecture/docker-compose-enhanced.yml up -d \
        sonarr radarr lidarr readarr prowlarr overseerr
    
    # Start monitoring
    print_status "Starting monitoring services..."
    docker-compose -f architecture/docker-compose-enhanced.yml up -d \
        prometheus grafana loki portainer
    
    print_status "All services started"
}

display_summary() {
    echo
    echo "============================================="
    echo "       Media Server Setup Complete!"
    echo "============================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo "Domain: $DOMAIN"
    echo
    echo "Access your services at:"
    echo "  Main Dashboard:     https://$DOMAIN"
    echo "  Jellyfin:          https://jellyfin.$DOMAIN"
    echo "  Music:             https://music.$DOMAIN"
    echo "  Audiobooks:        https://audiobooks.$DOMAIN"
    echo "  Photos:            https://photos.$DOMAIN"
    echo "  Books:             https://books.$DOMAIN"
    echo "  Monitoring:        https://grafana.$DOMAIN"
    echo
    echo "Default credentials:"
    echo "  Username: admin"
    echo "  Password: admin (CHANGE THIS!)"
    echo
    echo "Next steps:"
    echo "1. Update .env file with your email settings"
    echo "2. Change default passwords"
    echo "3. Configure each service through web UI"
    echo "4. Set up automated backups"
    echo
    echo "Useful commands:"
    echo "  View logs:         docker-compose -f $INSTALL_DIR/architecture/docker-compose-enhanced.yml logs -f [service]"
    echo "  Restart services:  sudo systemctl restart mediaserver"
    echo "  Stop all:          docker-compose -f $INSTALL_DIR/architecture/docker-compose-enhanced.yml down"
    echo
    echo "Documentation: $INSTALL_DIR/architecture/media-server-architecture.md"
    echo "============================================="
}

# Main installation flow
main() {
    echo "============================================="
    echo "    Media Server Automated Setup Script"
    echo "============================================="
    echo
    
    # Get domain from user
    read -p "Enter your domain (e.g., media.example.com): " user_domain
    if [[ ! -z "$user_domain" ]]; then
        DOMAIN=$user_domain
    fi
    
    # Confirm installation
    echo
    echo "This will install the media server stack to: $INSTALL_DIR"
    echo "Domain: $DOMAIN"
    read -p "Continue? (y/N): " confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        print_error "Installation cancelled"
        exit 1
    fi
    
    # Run installation steps
    check_requirements
    install_dependencies
    setup_directory_structure
    create_docker_networks
    generate_secrets
    setup_configurations
    create_systemd_service
    optimize_system
    setup_firewall
    
    # Ask to start services
    echo
    read -p "Start all services now? (y/N): " start_now
    if [[ "$start_now" =~ ^[Yy]$ ]]; then
        start_services
    fi
    
    display_summary
}

# Run main function
main "$@"