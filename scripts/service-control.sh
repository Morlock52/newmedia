#!/bin/bash

# Service Control Script - Start/Stop/Restart Docker Services
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display header
show_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       Docker Service Control           ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
    echo
}

# Function to get all services
get_services() {
    docker-compose ps --services 2>/dev/null
}

# Function to control service
control_service() {
    local service=$1
    local action=$2
    
    echo -e "${YELLOW}${action^}ing $service...${NC}"
    
    case $action in
        start)
            docker-compose up -d $service
            ;;
        stop)
            docker-compose stop $service
            ;;
        restart)
            docker-compose restart $service
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $service ${action}ed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to $action $service${NC}"
    fi
}

# Main menu
main_menu() {
    show_header
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found in current directory${NC}"
        echo -e "${YELLOW}Please run this script from the project root directory${NC}"
        exit 1
    fi
    
    # Get services
    services=($(get_services))
    if [ ${#services[@]} -eq 0 ]; then
        echo -e "${RED}No services found in docker-compose.yml${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Available services:${NC}"
    for i in "${!services[@]}"; do
        status=$(docker-compose ps -q ${services[$i]} 2>/dev/null)
        if [ -n "$status" ]; then
            running=$(docker inspect -f '{{.State.Running}}' $status 2>/dev/null)
            if [ "$running" == "true" ]; then
                echo -e "  $((i+1)). ${GREEN}${services[$i]} (running)${NC}"
            else
                echo -e "  $((i+1)). ${YELLOW}${services[$i]} (stopped)${NC}"
            fi
        else
            echo -e "  $((i+1)). ${RED}${services[$i]} (not created)${NC}"
        fi
    done
    echo -e "  $((${#services[@]}+1)). ${PURPLE}All services${NC}"
    echo -e "  0. ${RED}Exit${NC}"
    echo
    
    read -p "Select service (1-$((${#services[@]}+1))): " service_choice
    
    if [ "$service_choice" = "0" ]; then
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
    elif [ "$service_choice" -gt 0 ] && [ "$service_choice" -le "${#services[@]}" ]; then
        selected_service=${services[$((service_choice-1))]}
    elif [ "$service_choice" = "$((${#services[@]}+1))" ]; then
        selected_service="all"
    else
        echo -e "${RED}Invalid selection${NC}"
        sleep 2
        main_menu
        return
    fi
    
    echo
    echo -e "${BLUE}Select action:${NC}"
    echo "  1. Start"
    echo "  2. Stop"
    echo "  3. Restart"
    echo "  0. Back"
    echo
    
    read -p "Select action (0-3): " action_choice
    
    case $action_choice in
        1)
            if [ "$selected_service" = "all" ]; then
                echo -e "${YELLOW}Starting all services...${NC}"
                docker-compose up -d
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ All services started successfully${NC}"
                else
                    echo -e "${RED}✗ Failed to start services${NC}"
                fi
            else
                control_service "$selected_service" "start"
            fi
            ;;
        2)
            if [ "$selected_service" = "all" ]; then
                echo -e "${YELLOW}Stopping all services...${NC}"
                docker-compose stop
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ All services stopped successfully${NC}"
                else
                    echo -e "${RED}✗ Failed to stop services${NC}"
                fi
            else
                control_service "$selected_service" "stop"
            fi
            ;;
        3)
            if [ "$selected_service" = "all" ]; then
                echo -e "${YELLOW}Restarting all services...${NC}"
                docker-compose restart
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ All services restarted successfully${NC}"
                else
                    echo -e "${RED}✗ Failed to restart services${NC}"
                fi
            else
                control_service "$selected_service" "restart"
            fi
            ;;
        0)
            main_menu
            return
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
    main_menu
}

# Start the script
main_menu