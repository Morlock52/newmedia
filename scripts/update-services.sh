#!/bin/bash

# Update Services Script - Update Docker containers and images
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
    echo -e "${CYAN}║       Docker Services Update           ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
    echo
}

# Function to get all services
get_services() {
    docker-compose ps --services 2>/dev/null
}

# Function to check for updates
check_updates() {
    local service=$1
    local image=$(docker-compose config | grep -A 1 "^\s*$service:" | grep "image:" | awk '{print $2}')
    
    if [ -z "$image" ]; then
        echo -e "${YELLOW}No image specified (using build)${NC}"
        return
    fi
    
    echo -e "${BLUE}Checking updates for: ${CYAN}$image${NC}"
    
    # Pull the latest image
    docker pull $image 2>&1 | grep -E "(Status:|Pull complete|Already exists)" | tail -1
}

# Function to update single service
update_service() {
    local service=$1
    
    echo -e "${YELLOW}Updating $service...${NC}"
    echo
    
    # Check if service uses build or image
    local has_build=$(docker-compose config | grep -A 5 "^\s*$service:" | grep "build:")
    
    if [ -n "$has_build" ]; then
        echo -e "${BLUE}Service uses build configuration${NC}"
        echo -e "${YELLOW}Rebuilding image...${NC}"
        docker-compose build --no-cache $service
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Image rebuilt successfully${NC}"
        else
            echo -e "${RED}✗ Failed to rebuild image${NC}"
            return 1
        fi
    else
        check_updates $service
    fi
    
    # Stop the service
    echo -e "${YELLOW}Stopping $service...${NC}"
    docker-compose stop $service
    
    # Remove the container
    echo -e "${YELLOW}Removing old container...${NC}"
    docker-compose rm -f $service
    
    # Start with new image
    echo -e "${YELLOW}Starting $service with updated image...${NC}"
    docker-compose up -d $service
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $service updated successfully${NC}"
        
        # Wait a moment for service to start
        sleep 3
        
        # Check if service is running
        local container_id=$(docker-compose ps -q $service 2>/dev/null)
        if [ -n "$container_id" ]; then
            local running=$(docker inspect -f '{{.State.Running}}' $container_id 2>/dev/null)
            if [ "$running" = "true" ]; then
                echo -e "${GREEN}✓ $service is running${NC}"
            else
                echo -e "${RED}✗ $service failed to start${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ Failed to update $service${NC}"
        return 1
    fi
}

# Function to update all services
update_all_services() {
    local services=($(get_services))
    local failed=0
    local success=0
    
    echo -e "${BLUE}Updating all services (${#services[@]} total)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Pull all images first
    echo -e "${YELLOW}Pulling latest images...${NC}"
    docker-compose pull --ignore-pull-failures
    echo
    
    # Update each service
    for service in "${services[@]}"; do
        echo -e "${BLUE}[$((++success + failed))/${#services[@]}] Processing $service${NC}"
        echo -e "${BLUE}────────────────────────────────────────${NC}"
        
        update_service $service
        
        if [ $? -eq 0 ]; then
            ((success++))
        else
            ((failed++))
        fi
        echo
    done
    
    # Summary
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Update Summary:${NC}"
    echo -e "  ${GREEN}Successful: $success${NC}"
    echo -e "  ${RED}Failed: $failed${NC}"
    echo -e "  ${BLUE}Total: ${#services[@]}${NC}"
}

# Function to clean up old images
cleanup_images() {
    show_header
    echo -e "${BLUE}Cleaning up Docker resources...${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Remove unused containers
    echo -e "${YELLOW}Removing stopped containers...${NC}"
    docker container prune -f
    echo
    
    # Remove unused images
    echo -e "${YELLOW}Removing unused images...${NC}"
    docker image prune -f
    echo
    
    # Remove dangling images
    echo -e "${YELLOW}Removing dangling images...${NC}"
    docker images -f "dangling=true" -q | xargs -r docker rmi
    echo
    
    # Show disk usage
    echo -e "${BLUE}Docker disk usage:${NC}"
    docker system df
    echo
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
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
    
    echo -e "${BLUE}Update Options:${NC}"
    echo "  1. Update single service"
    echo "  2. Update all services"
    echo "  3. Check for updates (dry run)"
    echo "  4. Clean up old images"
    echo "  5. Full update (update all + cleanup)"
    echo "  0. Exit"
    echo
    
    read -p "Select option (0-5): " choice
    
    case $choice in
        1)
            echo
            echo -e "${BLUE}Available services:${NC}"
            for i in "${!services[@]}"; do
                echo "  $((i+1)). ${services[$i]}"
            done
            echo
            read -p "Select service (1-${#services[@]}): " service_choice
            
            if [ "$service_choice" -gt 0 ] && [ "$service_choice" -le "${#services[@]}" ]; then
                echo
                update_service "${services[$((service_choice-1))]}"
            else
                echo -e "${RED}Invalid selection${NC}"
            fi
            ;;
        2)
            echo
            echo -e "${YELLOW}This will update all services with minimal downtime.${NC}"
            read -p "Continue? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                echo
                update_all_services
            else
                echo -e "${YELLOW}Update cancelled${NC}"
            fi
            ;;
        3)
            echo
            echo -e "${BLUE}Checking for available updates...${NC}"
            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            for service in "${services[@]}"; do
                echo
                echo -e "${PURPLE}Service: $service${NC}"
                check_updates $service
            done
            ;;
        4)
            cleanup_images
            ;;
        5)
            echo
            echo -e "${YELLOW}This will update all services and clean up old images.${NC}"
            read -p "Continue? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                echo
                update_all_services
                echo
                read -p "Press Enter to continue with cleanup..."
                cleanup_images
            else
                echo -e "${YELLOW}Update cancelled${NC}"
            fi
            ;;
        0)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 2
            main_menu
            return
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
    main_menu
}

# Start the script
main_menu