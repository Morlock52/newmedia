#!/bin/bash

# Service Logs Script - View logs for Docker services
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
    echo -e "${CYAN}║       Docker Service Logs              ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
    echo
}

# Function to get all services
get_services() {
    docker-compose ps --services 2>/dev/null
}

# Function to view logs
view_logs() {
    local service=$1
    local lines=$2
    local follow=$3
    
    show_header
    echo -e "${BLUE}Viewing logs for: ${PURPLE}$service${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ "$follow" = "true" ]; then
        echo -e "${YELLOW}Following logs (Ctrl+C to stop)...${NC}"
        echo
        docker-compose logs -f --tail=$lines $service
    else
        docker-compose logs --tail=$lines $service
        echo
        echo -e "${GREEN}End of logs${NC}"
    fi
}

# Function to search logs
search_logs() {
    local service=$1
    local pattern=$2
    
    show_header
    echo -e "${BLUE}Searching logs for: ${PURPLE}$service${NC}"
    echo -e "${BLUE}Pattern: ${YELLOW}$pattern${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Get logs and search
    docker-compose logs --no-color $service 2>&1 | grep -i "$pattern" | tail -100
    
    local count=$(docker-compose logs --no-color $service 2>&1 | grep -i "$pattern" | wc -l)
    echo
    echo -e "${GREEN}Found $count matches (showing last 100)${NC}"
}

# Function to export logs
export_logs() {
    local service=$1
    local filename="logs_${service}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${YELLOW}Exporting logs for $service...${NC}"
    docker-compose logs --no-color --timestamps $service > "$filename" 2>&1
    
    if [ $? -eq 0 ]; then
        local size=$(ls -lh "$filename" | awk '{print $5}')
        echo -e "${GREEN}✓ Logs exported to: $filename (size: $size)${NC}"
    else
        echo -e "${RED}✗ Failed to export logs${NC}"
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
        local container_id=$(docker-compose ps -q ${services[$i]} 2>/dev/null)
        if [ -n "$container_id" ]; then
            echo -e "  $((i+1)). ${GREEN}${services[$i]}${NC}"
        else
            echo -e "  $((i+1)). ${YELLOW}${services[$i]} (not created)${NC}"
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
        selected_service=""  # Empty means all services for docker-compose
    else
        echo -e "${RED}Invalid selection${NC}"
        sleep 2
        main_menu
        return
    fi
    
    # Log viewing options
    echo
    echo -e "${BLUE}Log viewing options:${NC}"
    echo "  1. View last 50 lines"
    echo "  2. View last 100 lines"
    echo "  3. View last 500 lines"
    echo "  4. Follow logs (real-time)"
    echo "  5. Search logs"
    echo "  6. Export logs to file"
    echo "  0. Back"
    echo
    
    read -p "Select option (0-6): " log_option
    
    case $log_option in
        1)
            view_logs "$selected_service" 50 false
            ;;
        2)
            view_logs "$selected_service" 100 false
            ;;
        3)
            view_logs "$selected_service" 500 false
            ;;
        4)
            view_logs "$selected_service" 50 true
            ;;
        5)
            echo
            read -p "Enter search pattern: " pattern
            if [ -n "$pattern" ]; then
                search_logs "$selected_service" "$pattern"
            else
                echo -e "${RED}No pattern provided${NC}"
            fi
            ;;
        6)
            export_logs "$selected_service"
            ;;
        0)
            main_menu
            return
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 2
            main_menu
            return
            ;;
    esac
    
    if [ "$log_option" != "4" ] && [ "$log_option" != "0" ]; then
        echo
        read -p "Press Enter to continue..."
    fi
    
    main_menu
}

# Start the script
main_menu