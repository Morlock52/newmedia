#!/bin/bash

# Media Manager - Main control script for Docker services
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to display header
show_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${WHITE}             MEDIA SERVICES MANAGER                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${WHITE}                  Version 1.0                           ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Function to check Docker status
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed!${NC}"
        echo -e "${YELLOW}Please install Docker Desktop from https://docker.com${NC}"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}Docker daemon is not running!${NC}"
        echo -e "${YELLOW}Please start Docker Desktop${NC}"
        exit 1
    fi
}

# Function to check docker-compose
check_compose() {
    if ! command -v docker-compose &> /dev/null; then
        # Try docker compose (newer syntax)
        if docker compose version &> /dev/null; then
            alias docker-compose="docker compose"
        else
            echo -e "${RED}docker-compose is not installed!${NC}"
            exit 1
        fi
    fi
}

# Function to display quick status
quick_status() {
    cd "$PROJECT_ROOT"
    
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}No docker-compose.yml found${NC}"
        return
    fi
    
    local services=($(docker-compose ps --services 2>/dev/null))
    local running=0
    local stopped=0
    
    for service in "${services[@]}"; do
        local container_id=$(docker-compose ps -q $service 2>/dev/null)
        if [ -n "$container_id" ]; then
            local is_running=$(docker inspect -f '{{.State.Running}}' $container_id 2>/dev/null)
            if [ "$is_running" = "true" ]; then
                ((running++))
            else
                ((stopped++))
            fi
        else
            ((stopped++))
        fi
    done
    
    echo -e "${BLUE}Quick Status:${NC} ${GREEN}$running running${NC}, ${YELLOW}$stopped stopped${NC} (${#services[@]} total)"
}

# Function to display main menu
main_menu() {
    show_header
    
    # Display quick status
    quick_status
    echo
    
    echo -e "${BLUE}Main Menu:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "  ${CYAN}1.${NC} ${WHITE}Service Control${NC} - Start/Stop/Restart services"
    echo -e "  ${CYAN}2.${NC} ${WHITE}Service Status${NC}  - Check health and resource usage"
    echo -e "  ${CYAN}3.${NC} ${WHITE}View Logs${NC}       - View and search service logs"
    echo -e "  ${CYAN}4.${NC} ${WHITE}Update Services${NC} - Update containers and images"
    echo
    echo -e "  ${PURPLE}5.${NC} ${WHITE}Quick Actions${NC}   - Common tasks menu"
    echo -e "  ${PURPLE}6.${NC} ${WHITE}System Info${NC}     - Docker system information"
    echo
    echo -e "  ${RED}0.${NC} ${WHITE}Exit${NC}"
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    read -p "Select option (0-6): " choice
    
    case $choice in
        1)
            cd "$PROJECT_ROOT" && "$SCRIPT_DIR/service-control.sh"
            ;;
        2)
            cd "$PROJECT_ROOT" && "$SCRIPT_DIR/service-status.sh"
            ;;
        3)
            cd "$PROJECT_ROOT" && "$SCRIPT_DIR/service-logs.sh"
            ;;
        4)
            cd "$PROJECT_ROOT" && "$SCRIPT_DIR/update-services.sh"
            ;;
        5)
            quick_actions_menu
            ;;
        6)
            system_info
            ;;
        0)
            echo -e "${GREEN}Thank you for using Media Manager!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 2
            ;;
    esac
    
    main_menu
}

# Function for quick actions menu
quick_actions_menu() {
    show_header
    
    echo -e "${BLUE}Quick Actions:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "  ${CYAN}1.${NC} Start all services"
    echo -e "  ${CYAN}2.${NC} Stop all services"
    echo -e "  ${CYAN}3.${NC} Restart all services"
    echo -e "  ${CYAN}4.${NC} View all logs (follow)"
    echo -e "  ${CYAN}5.${NC} Clean up Docker resources"
    echo -e "  ${CYAN}6.${NC} Backup service data"
    echo
    echo -e "  ${RED}0.${NC} Back to main menu"
    echo
    
    read -p "Select option (0-6): " choice
    
    cd "$PROJECT_ROOT"
    
    case $choice in
        1)
            echo -e "${YELLOW}Starting all services...${NC}"
            docker-compose up -d
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ All services started${NC}"
            else
                echo -e "${RED}✗ Failed to start services${NC}"
            fi
            read -p "Press Enter to continue..."
            ;;
        2)
            echo -e "${YELLOW}Stopping all services...${NC}"
            docker-compose stop
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ All services stopped${NC}"
            else
                echo -e "${RED}✗ Failed to stop services${NC}"
            fi
            read -p "Press Enter to continue..."
            ;;
        3)
            echo -e "${YELLOW}Restarting all services...${NC}"
            docker-compose restart
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ All services restarted${NC}"
            else
                echo -e "${RED}✗ Failed to restart services${NC}"
            fi
            read -p "Press Enter to continue..."
            ;;
        4)
            echo -e "${YELLOW}Following all logs (Ctrl+C to stop)...${NC}"
            docker-compose logs -f --tail=50
            ;;
        5)
            echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
            docker system prune -a --volumes
            echo -e "${GREEN}✓ Cleanup completed${NC}"
            read -p "Press Enter to continue..."
            ;;
        6)
            backup_data
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 2
            quick_actions_menu
            ;;
    esac
    
    quick_actions_menu
}

# Function to backup data
backup_data() {
    echo -e "${BLUE}Backing up service data...${NC}"
    echo
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Find all volumes
    local volumes=$(docker-compose config --volumes 2>/dev/null)
    
    if [ -z "$volumes" ]; then
        echo -e "${YELLOW}No volumes found to backup${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    
    echo -e "${YELLOW}Found volumes: $volumes${NC}"
    echo -e "${YELLOW}Backing up to: $backup_dir${NC}"
    echo
    
    for volume in $volumes; do
        echo -e "Backing up volume: ${CYAN}$volume${NC}"
        docker run --rm -v ${volume}:/data -v ${backup_dir}:/backup alpine tar czf /backup/${volume}.tar.gz -C /data .
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Backed up $volume${NC}"
        else
            echo -e "${RED}✗ Failed to backup $volume${NC}"
        fi
    done
    
    echo
    echo -e "${GREEN}Backup completed!${NC}"
    echo -e "Location: ${CYAN}$backup_dir${NC}"
    read -p "Press Enter to continue..."
}

# Function to display system info
system_info() {
    show_header
    
    echo -e "${BLUE}Docker System Information:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Docker version
    echo -e "${CYAN}Docker Version:${NC}"
    docker version --format 'Client: {{.Client.Version}}\nServer: {{.Server.Version}}'
    echo
    
    # System df
    echo -e "${CYAN}Disk Usage:${NC}"
    docker system df
    echo
    
    # Running containers
    echo -e "${CYAN}Running Containers:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    
    read -p "Press Enter to continue..."
}

# Function to check and create scripts
check_scripts() {
    local scripts=("service-control.sh" "service-status.sh" "service-logs.sh" "update-services.sh")
    
    for script in "${scripts[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$script" ]; then
            echo -e "${RED}Missing script: $script${NC}"
            echo -e "${YELLOW}Please ensure all scripts are in the same directory${NC}"
            exit 1
        fi
        
        # Make executable
        chmod +x "$SCRIPT_DIR/$script"
    done
}

# Main execution
check_docker
check_compose
check_scripts

# Make this script executable
chmod +x "$SCRIPT_DIR/media-manager.sh"

# Start main menu
main_menu