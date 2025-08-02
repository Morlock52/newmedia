#!/bin/bash

# Service Status Script - Check health and status of Docker services
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
    echo -e "${CYAN}║       Docker Service Status            ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
    echo
}

# Function to format bytes
format_bytes() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt 1048576 ]; then
        echo "$((bytes/1024))KB"
    elif [ $bytes -lt 1073741824 ]; then
        echo "$((bytes/1048576))MB"
    else
        echo "$((bytes/1073741824))GB"
    fi
}

# Function to get container stats
get_container_stats() {
    local container_id=$1
    local stats=$(docker stats --no-stream --format "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}" $container_id 2>/dev/null)
    echo "$stats"
}

# Function to check service health
check_service_health() {
    local service=$1
    local container_id=$(docker-compose ps -q $service 2>/dev/null)
    
    if [ -z "$container_id" ]; then
        echo -e "${RED}✗ Not created${NC}"
        return
    fi
    
    local running=$(docker inspect -f '{{.State.Running}}' $container_id 2>/dev/null)
    local health=$(docker inspect -f '{{.State.Health.Status}}' $container_id 2>/dev/null)
    
    if [ "$running" != "true" ]; then
        echo -e "${YELLOW}✗ Stopped${NC}"
        return
    fi
    
    if [ "$health" = "healthy" ]; then
        echo -e "${GREEN}✓ Healthy${NC}"
    elif [ "$health" = "unhealthy" ]; then
        echo -e "${RED}✗ Unhealthy${NC}"
    elif [ "$health" = "starting" ]; then
        echo -e "${YELLOW}⟳ Starting${NC}"
    else
        echo -e "${GREEN}✓ Running${NC}"
    fi
}

# Function to show detailed service info
show_service_details() {
    local service=$1
    local container_id=$(docker-compose ps -q $service 2>/dev/null)
    
    echo -e "${BLUE}Service: ${PURPLE}$service${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -z "$container_id" ]; then
        echo -e "${RED}Service not created${NC}"
        return
    fi
    
    # Get container info
    local info=$(docker inspect $container_id 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to get container info${NC}"
        return
    fi
    
    # Parse container info
    local state=$(echo "$info" | jq -r '.[0].State.Status')
    local started=$(echo "$info" | jq -r '.[0].State.StartedAt' | cut -d'T' -f1,2 | sed 's/T/ /')
    local image=$(echo "$info" | jq -r '.[0].Config.Image')
    local ports=$(docker port $container_id 2>/dev/null)
    
    echo -e "Status: $(check_service_health $service)"
    echo -e "Image: ${CYAN}$image${NC}"
    
    if [ "$state" = "running" ]; then
        echo -e "Started: ${GREEN}$started${NC}"
        
        # Get resource usage
        local stats=$(get_container_stats $container_id)
        if [ -n "$stats" ]; then
            local cpu=$(echo "$stats" | cut -d'|' -f1)
            local mem=$(echo "$stats" | cut -d'|' -f2)
            local net=$(echo "$stats" | cut -d'|' -f3)
            local disk=$(echo "$stats" | cut -d'|' -f4)
            
            echo -e "CPU: ${YELLOW}$cpu${NC}"
            echo -e "Memory: ${YELLOW}$mem${NC}"
            echo -e "Network: ${YELLOW}$net${NC}"
            echo -e "Disk: ${YELLOW}$disk${NC}"
        fi
        
        if [ -n "$ports" ]; then
            echo -e "Ports:"
            echo "$ports" | while read line; do
                echo -e "  ${CYAN}$line${NC}"
            done
        fi
    else
        echo -e "State: ${YELLOW}$state${NC}"
    fi
    
    echo
}

# Function to show all services status
show_all_status() {
    show_header
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found in current directory${NC}"
        echo -e "${YELLOW}Please run this script from the project root directory${NC}"
        exit 1
    fi
    
    # Get all services
    services=($(docker-compose ps --services 2>/dev/null))
    if [ ${#services[@]} -eq 0 ]; then
        echo -e "${RED}No services found in docker-compose.yml${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Service Status Overview:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Display summary
    local running=0
    local stopped=0
    local unhealthy=0
    
    for service in "${services[@]}"; do
        local container_id=$(docker-compose ps -q $service 2>/dev/null)
        if [ -n "$container_id" ]; then
            local is_running=$(docker inspect -f '{{.State.Running}}' $container_id 2>/dev/null)
            local health=$(docker inspect -f '{{.State.Health.Status}}' $container_id 2>/dev/null)
            
            if [ "$is_running" = "true" ]; then
                ((running++))
                if [ "$health" = "unhealthy" ]; then
                    ((unhealthy++))
                fi
            else
                ((stopped++))
            fi
        else
            ((stopped++))
        fi
    done
    
    echo -e "Total Services: ${PURPLE}${#services[@]}${NC}"
    echo -e "Running: ${GREEN}$running${NC}"
    echo -e "Stopped: ${YELLOW}$stopped${NC}"
    if [ $unhealthy -gt 0 ]; then
        echo -e "Unhealthy: ${RED}$unhealthy${NC}"
    fi
    echo
    
    # Display individual service status
    echo -e "${BLUE}Individual Service Status:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    for service in "${services[@]}"; do
        printf "%-20s " "$service"
        check_service_health "$service"
    done
    
    echo
    echo -e "${BLUE}Options:${NC}"
    echo "  1. Show detailed status for a service"
    echo "  2. Refresh status"
    echo "  0. Exit"
    echo
    
    read -p "Select option (0-2): " choice
    
    case $choice in
        1)
            echo
            echo -e "${BLUE}Select service for details:${NC}"
            for i in "${!services[@]}"; do
                echo "  $((i+1)). ${services[$i]}"
            done
            echo
            read -p "Select service (1-${#services[@]}): " service_choice
            
            if [ "$service_choice" -gt 0 ] && [ "$service_choice" -le "${#services[@]}" ]; then
                echo
                show_service_details "${services[$((service_choice-1))]}"
                read -p "Press Enter to continue..."
            else
                echo -e "${RED}Invalid selection${NC}"
                sleep 2
            fi
            show_all_status
            ;;
        2)
            show_all_status
            ;;
        0)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 2
            show_all_status
            ;;
    esac
}

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq is not installed. Some features may not work properly.${NC}"
    echo -e "${YELLOW}Install with: brew install jq${NC}"
    echo
fi

# Start the script
show_all_status