#!/bin/bash

# ============================================================================
# Smart Auto-Configuration Script v2025
# ============================================================================
# This script intelligently handles existing containers and configurations
# ============================================================================

set -e

echo "🧠 Smart Media Server Auto-Configuration v2025"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${1:-$PROJECT_ROOT/docker-compose-demo.yml}"

# Logging function
log() {
    echo -e "$1"
}

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "   Waiting for $service"
    while ! nc -z localhost $port 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e " ${RED}[TIMEOUT]${NC}"
            return 1
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e " ${GREEN}[OK]${NC}"
}

# ============================================================================
# STEP 1: Handle Existing Containers
# ============================================================================

handle_existing_containers() {
    log "${PURPLE}📦 Step 1: Checking existing containers...${NC}"
    
    # Get all services from docker-compose file
    if [ -f "$COMPOSE_FILE" ]; then
        COMPOSE_SERVICES=$(docker-compose -f "$COMPOSE_FILE" config --services 2>/dev/null || echo "")
    else
        log "${RED}❌ Compose file not found: $COMPOSE_FILE${NC}"
        exit 1
    fi
    
    # Check each service
    for service in $COMPOSE_SERVICES; do
        if container_exists "$service"; then
            if container_running "$service"; then
                log "${GREEN}✅ $service - Already running${NC}"
            else
                log "${YELLOW}🔄 $service - Exists but stopped, starting...${NC}"
                docker start "$service" 2>/dev/null || {
                    log "${RED}   Failed to start, will recreate${NC}"
                    docker rm "$service" 2>/dev/null || true
                }
            fi
        else
            log "${BLUE}📦 $service - Will be created${NC}"
        fi
    done
    
    echo ""
}

# ============================================================================
# STEP 2: Start Only Non-Running Services
# ============================================================================

start_required_services() {
    log "${PURPLE}🚀 Step 2: Starting required services...${NC}"
    
    # Essential services that must be running
    ESSENTIAL_SERVICES=(prowlarr sonarr radarr jellyfin qbittorrent)
    RUNNING_SERVICES=$(docker ps --format "{{.Names}}")
    SERVICES_TO_START=()
    
    for service in "${ESSENTIAL_SERVICES[@]}"; do
        if ! echo "$RUNNING_SERVICES" | grep -q "^${service}$"; then
            SERVICES_TO_START+=("$service")
        fi
    done
    
    if [ ${#SERVICES_TO_START[@]} -gt 0 ]; then
        log "${BLUE}Starting: ${SERVICES_TO_START[*]}...${NC}"
        
        # Use docker-compose to start only non-running services
        for service in "${SERVICES_TO_START[@]}"; do
            docker-compose -f "$COMPOSE_FILE" up -d "$service" 2>&1 | grep -v "is already in use" || true
        done
    else
        log "${GREEN}✅ All essential services are already running${NC}"
    fi
    
    # Special handling for homarr
    if echo "$COMPOSE_SERVICES" | grep -q "homarr"; then
        if container_exists "homarr"; then
            if ! container_running "homarr"; then
                log "${YELLOW}🔄 Starting existing homarr container...${NC}"
                docker start homarr 2>/dev/null || {
                    log "${YELLOW}   Recreating homarr...${NC}"
                    docker rm homarr 2>/dev/null || true
                    docker-compose -f "$COMPOSE_FILE" up -d homarr 2>&1 | grep -v "is already in use" || true
                }
            fi
        else
            log "${BLUE}📦 Creating homarr...${NC}"
            docker-compose -f "$COMPOSE_FILE" up -d homarr 2>&1 | grep -v "is already in use" || true
        fi
    fi
    
    echo ""
}

# ============================================================================
# STEP 3: Wait for Services
# ============================================================================

wait_for_services() {
    log "${PURPLE}⏳ Step 3: Waiting for services to be ready...${NC}"
    
    # Service ports
    declare -A service_ports=(
        ["prowlarr"]=9696
        ["sonarr"]=8989
        ["radarr"]=7878
        ["jellyfin"]=8096
        ["qbittorrent"]=8090
        ["homarr"]=7575
    )
    
    for service in "${!service_ports[@]}"; do
        if container_running "$service"; then
            wait_for_service "$service" "${service_ports[$service]}"
        fi
    done
    
    echo ""
}

# ============================================================================
# STEP 4: Configure Services
# ============================================================================

configure_services() {
    log "${PURPLE}⚙️  Step 4: Configuring services...${NC}"
    
    # Run the full auto-configuration
    if [ -f "$SCRIPT_DIR/auto-configure-all-services.sh" ]; then
        log "${BLUE}Running full auto-configuration...${NC}"
        # Pass the compose file to the configuration script
        "$SCRIPT_DIR/auto-configure-all-services.sh" "$COMPOSE_FILE"
    else
        log "${YELLOW}⚠️  auto-configure-all-services.sh not found${NC}"
        log "${YELLOW}   Please run it manually later${NC}"
    fi
    
    echo ""
}

# ============================================================================
# STEP 5: Final Status Report
# ============================================================================

final_status_report() {
    log "${PURPLE}📊 Step 5: Final Status Report${NC}"
    log "${CYAN}================================${NC}"
    
    # Check all running containers
    log "${BLUE}Running Services:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(prowlarr|sonarr|radarr|jellyfin|qbittorrent|homarr|homepage)" | head -20
    
    echo ""
    log "${CYAN}🌐 Service URLs:${NC}"
    log "  ${BLUE}Homarr Dashboard:${NC}    http://localhost:7575"
    log "  ${BLUE}Homepage Dashboard:${NC}  http://localhost:3001"
    log "  ${BLUE}Jellyfin:${NC}           http://localhost:8096"
    log "  ${BLUE}Prowlarr:${NC}           http://localhost:9696"
    log "  ${BLUE}Sonarr:${NC}             http://localhost:8989"
    log "  ${BLUE}Radarr:${NC}             http://localhost:7878"
    log "  ${BLUE}qBittorrent:${NC}        http://localhost:8090 (admin/adminadmin)"
    
    echo ""
    log "${YELLOW}📝 Next Steps:${NC}"
    log "  1. Access Homarr at http://localhost:7575 for your main dashboard"
    log "  2. Complete Jellyfin setup at http://localhost:8096"
    log "  3. Prowlarr should have indexers configured automatically"
    log "  4. Check that all *arr apps are connected to Prowlarr"
    
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    # Check Docker
    if ! docker info > /dev/null 2>&1; then
        log "${RED}❌ Docker is not running! Please start Docker first.${NC}"
        exit 1
    fi
    
    # Execute steps
    handle_existing_containers
    start_required_services
    wait_for_services
    configure_services
    final_status_report
    
    log "${GREEN}✅ Smart auto-configuration complete!${NC}"
    log ""
    log "${CYAN}🎉 Your media server is ready to use!${NC}"
}

# Run main function
main "$@"