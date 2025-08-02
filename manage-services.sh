#!/bin/bash
# Ultimate Media Server 2025 - Service Management Script
# =====================================================
# Enable/Disable services by profile with ease

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-unified-2025.yml"
ENV_FILE=".env"

# Available profiles
PROFILES=(
    "core:Essential infrastructure (Traefik, Authelia, Redis, PostgreSQL)"
    "media:Media servers (Jellyfin, Plex, Emby)"
    "music:Music services (Navidrome, Lidarr)"
    "books:E-book and audiobook services (Calibre-Web, AudioBookshelf, Kavita)"
    "photos:Photo management (Immich, PhotoPrism)"
    "automation:Media automation (*arr stack)"
    "downloads:Download clients (qBittorrent, SABnzbd, NZBGet)"
    "requests:Request management (Overseerr, Ombi)"
    "monitoring:System monitoring (Prometheus, Grafana, Loki)"
    "management:Container management (Homepage, Portainer, Yacht)"
    "backup:Backup solutions (Duplicati, Restic)"
    "advanced:Advanced/experimental features"
)

# Display banner
show_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║       Ultimate Media Server 2025 - Service Manager       ║"
    echo "║                 Enable/Disable with Ease!                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Display help
show_help() {
    echo -e "${CYAN}Usage:${NC} $0 [command] [options]"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo "  enable <profile>    Enable services in the specified profile"
    echo "  disable <profile>   Disable services in the specified profile"
    echo "  list                List all available profiles"
    echo "  status              Show status of all services"
    echo "  preset <name>       Apply a preset configuration"
    echo "  wizard              Interactive setup wizard"
    echo ""
    echo -e "${CYAN}Available Profiles:${NC}"
    for profile in "${PROFILES[@]}"; do
        IFS=':' read -r name desc <<< "$profile"
        printf "  ${GREEN}%-15s${NC} %s\n" "$name" "$desc"
    done
    echo ""
    echo -e "${CYAN}Presets:${NC}"
    echo "  ${GREEN}minimal${NC}         Core + Media only"
    echo "  ${GREEN}basic${NC}           Core + Media + Downloads + Requests"
    echo "  ${GREEN}standard${NC}        Basic + Automation + Monitoring"
    echo "  ${GREEN}power-user${NC}      Everything except advanced"
    echo "  ${GREEN}everything${NC}      All profiles enabled"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  $0 enable media"
    echo "  $0 disable downloads"
    echo "  $0 preset standard"
    echo "  $0 wizard"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running!${NC}"
        echo "Please start Docker and try again."
        exit 1
    fi
}

# Check if compose file exists
check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${RED}Error: $COMPOSE_FILE not found!${NC}"
        echo "Please ensure you're in the correct directory."
        exit 1
    fi
}

# Enable profile
enable_profile() {
    local profile=$1
    echo -e "${YELLOW}Enabling profile: ${GREEN}$profile${NC}"
    
    # Start services with the specified profile
    docker compose -f "$COMPOSE_FILE" --profile "$profile" up -d
    
    echo -e "${GREEN}✓ Profile '$profile' enabled successfully!${NC}"
}

# Disable profile
disable_profile() {
    local profile=$1
    echo -e "${YELLOW}Disabling profile: ${RED}$profile${NC}"
    
    # Get list of services in the profile
    services=$(docker compose -f "$COMPOSE_FILE" config --profile "$profile" --services 2>/dev/null || true)
    
    if [ -n "$services" ]; then
        echo "Stopping services: $services"
        # Stop and remove services
        docker compose -f "$COMPOSE_FILE" --profile "$profile" stop
        docker compose -f "$COMPOSE_FILE" --profile "$profile" rm -f
    fi
    
    echo -e "${GREEN}✓ Profile '$profile' disabled successfully!${NC}"
}

# List profiles
list_profiles() {
    echo -e "${CYAN}Available Service Profiles:${NC}"
    echo ""
    for profile in "${PROFILES[@]}"; do
        IFS=':' read -r name desc <<< "$profile"
        printf "  ${GREEN}%-15s${NC} %s\n" "$name" "$desc"
    done
}

# Show service status
show_status() {
    echo -e "${CYAN}Service Status:${NC}"
    echo ""
    
    # Get all running containers
    running=$(docker compose -f "$COMPOSE_FILE" ps --format "table {{.Service}}\t{{.Status}}" 2>/dev/null || true)
    
    if [ -n "$running" ]; then
        echo "$running"
    else
        echo "No services are currently running."
    fi
}

# Apply preset
apply_preset() {
    local preset=$1
    
    case $preset in
        minimal)
            echo -e "${YELLOW}Applying minimal preset...${NC}"
            enable_profile core
            enable_profile media
            ;;
        basic)
            echo -e "${YELLOW}Applying basic preset...${NC}"
            enable_profile core
            enable_profile media
            enable_profile downloads
            enable_profile requests
            ;;
        standard)
            echo -e "${YELLOW}Applying standard preset...${NC}"
            enable_profile core
            enable_profile media
            enable_profile downloads
            enable_profile requests
            enable_profile automation
            enable_profile monitoring
            ;;
        power-user)
            echo -e "${YELLOW}Applying power-user preset...${NC}"
            for profile in "${PROFILES[@]}"; do
                IFS=':' read -r name desc <<< "$profile"
                if [ "$name" != "advanced" ]; then
                    enable_profile "$name"
                fi
            done
            ;;
        everything)
            echo -e "${YELLOW}Applying everything preset...${NC}"
            for profile in "${PROFILES[@]}"; do
                IFS=':' read -r name desc <<< "$profile"
                enable_profile "$name"
            done
            ;;
        *)
            echo -e "${RED}Unknown preset: $preset${NC}"
            echo "Available presets: minimal, basic, standard, power-user, everything"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}✓ Preset '$preset' applied successfully!${NC}"
}

# Interactive wizard
wizard() {
    show_banner
    echo -e "${CYAN}Welcome to the Interactive Setup Wizard!${NC}"
    echo ""
    echo "I'll help you configure your media server step by step."
    echo ""
    
    # Check for env file
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${GREEN}✓ Created .env file from template${NC}"
        else
            echo -e "${RED}Warning: No .env.example found. Please configure .env manually.${NC}"
        fi
    fi
    
    # Ask for user type
    echo ""
    echo "What type of user are you?"
    echo "1) Beginner - I want something simple that just works"
    echo "2) Intermediate - I want control over my media automation"
    echo "3) Advanced - I want all the features and customization"
    echo ""
    read -p "Select (1-3): " user_type
    
    case $user_type in
        1)
            echo -e "${GREEN}Great! Setting up a simple configuration...${NC}"
            apply_preset minimal
            ;;
        2)
            echo -e "${GREEN}Perfect! Setting up standard media automation...${NC}"
            apply_preset standard
            ;;
        3)
            echo -e "${GREEN}Excellent! Let's customize your setup...${NC}"
            echo ""
            echo "Which profiles would you like to enable?"
            for profile in "${PROFILES[@]}"; do
                IFS=':' read -r name desc <<< "$profile"
                read -p "Enable $name ($desc)? [y/N]: " enable
                if [[ $enable =~ ^[Yy]$ ]]; then
                    enable_profile "$name"
                fi
            done
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo ""
    echo "Your services are starting up. Here's what to do next:"
    echo ""
    echo "1. Wait a few minutes for services to initialize"
    echo "2. Access your dashboard at: http://localhost:3001"
    echo "3. Configure your media paths in the .env file"
    echo "4. Set up your indexers in Prowlarr"
    echo ""
    echo -e "${CYAN}Need help? Check the documentation or run: $0 help${NC}"
}

# Generate systemd service
generate_systemd() {
    echo -e "${YELLOW}Generating systemd service...${NC}"
    
    cat > media-server.service << EOF
[Unit]
Description=Ultimate Media Server 2025
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/local/bin/docker-compose -f $COMPOSE_FILE up -d
ExecStop=/usr/local/bin/docker-compose -f $COMPOSE_FILE down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${GREEN}✓ Systemd service file created: media-server.service${NC}"
    echo ""
    echo "To install:"
    echo "  sudo cp media-server.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable media-server"
    echo "  sudo systemctl start media-server"
}

# Main script
main() {
    show_banner
    check_docker
    check_compose_file
    
    case ${1:-help} in
        enable)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: Please specify a profile to enable${NC}"
                echo "Usage: $0 enable <profile>"
                list_profiles
                exit 1
            fi
            enable_profile "$2"
            ;;
        disable)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: Please specify a profile to disable${NC}"
                echo "Usage: $0 disable <profile>"
                list_profiles
                exit 1
            fi
            disable_profile "$2"
            ;;
        list)
            list_profiles
            ;;
        status)
            show_status
            ;;
        preset)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: Please specify a preset${NC}"
                echo "Available presets: minimal, basic, standard, power-user, everything"
                exit 1
            fi
            apply_preset "$2"
            ;;
        wizard)
            wizard
            ;;
        systemd)
            generate_systemd
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"