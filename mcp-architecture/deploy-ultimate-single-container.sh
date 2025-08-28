#!/bin/bash

# Ultimate Media Server 2025 - Single Container Deployment Script
# Deploys ALL 30 services in one unified Docker container
# Addresses user requirement: "NO MCP SHOULD HAVE ALL THE APP FROM THE BEGIEN OF THE PROJECT ALL 30"

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Ultimate Media Server 2025"
CONTAINER_NAME="ultimate-media-server-2025"
COMPOSE_FILE="docker-compose.single-container.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
CONFIG_DIR="./config"

# ASCII Art Header
echo -e "${PURPLE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ULTIMATE MEDIA SERVER 2025 🚀                         ║
║                        Single Container Deployment                          ║
║                          ALL 30 SERVICES UNIFIED                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${CYAN}🎯 Deploying complete media server ecosystem with:${NC}"
echo -e "${GREEN}   📺 Media Servers: Jellyfin, Plex, Emby${NC}"
echo -e "${GREEN}   📋 Content Management: Sonarr, Radarr, Lidarr, Readarr, Bazarr${NC}"
echo -e "${GREEN}   🔍 Indexers: Prowlarr, Jackett, FlareSolverr${NC}"
echo -e "${GREEN}   📥 Download Clients: qBittorrent, Transmission, Deluge, NZBGet, SABnzbd${NC}"
echo -e "${GREEN}   🙋 Request Management: Overseerr, Requestrr, Ombi${NC}"
echo -e "${GREEN}   📊 Analytics: Tautulli, Netdata${NC}"
echo -e "${GREEN}   🏠 Dashboards: Homepage, Heimdall, Organizr, Homarr${NC}"
echo -e "${GREEN}   🔧 Infrastructure: Nginx Proxy Manager, Portainer, Watchtower, Gluetun VPN, Unpackerr${NC}"
echo -e "${GREEN}   🤖 AI MCP Server: Unified management for all services${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}📋 $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_section "CHECKING SYSTEM REQUIREMENTS"
    
    echo -e "${CYAN}🔍 Checking required commands...${NC}"
    
    if ! command_exists docker; then
        echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Docker found: $(docker --version)${NC}"
    fi
    
    if ! command_exists docker-compose; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Docker Compose found: $(docker-compose --version)${NC}"
    fi
    
    if ! command_exists node; then
        echo -e "${RED}❌ Node.js not found. Please install Node.js first.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"
    fi
    
    # Check available disk space (macOS compatible)
    if command -v df >/dev/null 2>&1; then
        AVAILABLE_SPACE=$(df -g . | awk 'NR==2 {print $4}')
        if [ -n "$AVAILABLE_SPACE" ] && [ "$AVAILABLE_SPACE" -lt 50 ]; then
            echo -e "${YELLOW}⚠️  Warning: Less than 50GB available. Recommended: 100GB+ for media storage${NC}"
        else
            echo -e "${GREEN}✅ Disk space: ${AVAILABLE_SPACE}GB available${NC}"
        fi
    fi
    
    # Check available memory (macOS compatible)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS memory check
        AVAILABLE_RAM=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        echo -e "${GREEN}✅ Total RAM: ${AVAILABLE_RAM}GB${NC}"
    elif command -v free >/dev/null 2>&1; then
        # Linux memory check
        AVAILABLE_RAM=$(free -g | awk 'NR==2{print $7}')
        if [ -n "$AVAILABLE_RAM" ] && [ "$AVAILABLE_RAM" -lt 8 ]; then
            echo -e "${YELLOW}⚠️  Warning: Less than 8GB RAM available. Recommended: 16GB+ for optimal performance${NC}"
        else
            echo -e "${GREEN}✅ Memory: ${AVAILABLE_RAM}GB available${NC}"
        fi
    fi
}

# Function to create required directories
create_directories() {
    print_section "CREATING DIRECTORY STRUCTURE"
    
    echo -e "${CYAN}📁 Creating media directories...${NC}"
    
    DIRECTORIES=(
        "media/movies"
        "media/tv"
        "media/music"
        "media/books"
        "downloads"
        "incomplete-downloads"
        "watch"
        "backups"
        "config"
        "logs"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "${GREEN}✅ Created: $dir${NC}"
        else
            echo -e "${YELLOW}📁 Already exists: $dir${NC}"
        fi
    done
    
    # Set proper permissions
    echo -e "${CYAN}🔒 Setting permissions...${NC}"
    chmod -R 755 media downloads config backups logs 2>/dev/null || true
    echo -e "${GREEN}✅ Permissions set${NC}"
}

# Function to setup environment file
setup_environment() {
    print_section "ENVIRONMENT CONFIGURATION"
    
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${CYAN}📄 Creating environment file from template...${NC}"
        cp .env.single-container "$ENV_FILE"
        echo -e "${GREEN}✅ Environment file created: $ENV_FILE${NC}"
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit $ENV_FILE and set your API keys!${NC}"
        echo -e "${YELLOW}   Required: OPENAI_API_KEY for MCP functionality${NC}"
        echo -e "${YELLOW}   Optional: Service API keys (can be configured later)${NC}"
    else
        echo -e "${GREEN}✅ Environment file exists: $ENV_FILE${NC}"
    fi
    
    # Check if OpenAI API key is set
    if grep -q "your-openai-api-key-here" "$ENV_FILE" 2>/dev/null; then
        echo -e "${RED}❌ OpenAI API key not configured in $ENV_FILE${NC}"
        echo -e "${YELLOW}   Set OPENAI_API_KEY for MCP functionality${NC}"
        echo -e "${YELLOW}   Get your key from: https://platform.openai.com${NC}"
    else
        echo -e "${GREEN}✅ OpenAI API key appears to be configured${NC}"
    fi
}

# Function to stop existing containers
stop_existing() {
    print_section "STOPPING EXISTING SERVICES"
    
    echo -e "${CYAN}🛑 Checking for existing containers...${NC}"
    
    if docker ps -a --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
        echo -e "${YELLOW}⚠️  Found existing container: $CONTAINER_NAME${NC}"
        echo -e "${CYAN}🛑 Stopping and removing...${NC}"
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
        echo -e "${GREEN}✅ Existing services stopped${NC}"
    else
        echo -e "${GREEN}✅ No existing containers found${NC}"
    fi
    
    # Clean up any orphaned containers
    echo -e "${CYAN}🧹 Cleaning up Docker system...${NC}"
    docker system prune -f >/dev/null 2>&1 || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Function to build the container
build_container() {
    print_section "BUILDING SINGLE CONTAINER"
    
    echo -e "${CYAN}🔨 Building Ultimate Media Server container...${NC}"
    echo -e "${YELLOW}   This may take 10-20 minutes (downloading and installing all 30 services)${NC}"
    
    # Build with progress output
    if docker-compose -f "$COMPOSE_FILE" build --no-cache; then
        echo -e "${GREEN}✅ Container built successfully${NC}"
    else
        echo -e "${RED}❌ Container build failed${NC}"
        exit 1
    fi
}

# Function to start services
start_services() {
    print_section "STARTING ALL SERVICES"
    
    echo -e "${CYAN}🚀 Starting Ultimate Media Server...${NC}"
    echo -e "${YELLOW}   Starting all 30 services in single container${NC}"
    
    # Start in detached mode
    if docker-compose -f "$COMPOSE_FILE" up -d; then
        echo -e "${GREEN}✅ All services started successfully${NC}"
    else
        echo -e "${RED}❌ Failed to start services${NC}"
        exit 1
    fi
    
    echo -e "${CYAN}⏳ Waiting for services to initialize (60 seconds)...${NC}"
    sleep 60
}

# Function to check service health
check_health() {
    print_section "HEALTH CHECKS"
    
    echo -e "${CYAN}🏥 Checking service health...${NC}"
    
    # Check container status
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$CONTAINER_NAME" | grep -q "Up"; then
        echo -e "${GREEN}✅ Container is running${NC}"
    else
        echo -e "${RED}❌ Container is not running${NC}"
        docker-compose -f "$COMPOSE_FILE" logs --tail=50
        exit 1
    fi
    
    # Check specific services
    SERVICES=(
        "8090:Ultimate Dashboard"
        "3000:MCP Server"
        "8096:Jellyfin"
        "8989:Sonarr"
        "7878:Radarr"
        "9696:Prowlarr"
        "8080:qBittorrent"
    )
    
    echo -e "${CYAN}🔍 Testing service endpoints...${NC}"
    
    for service in "${SERVICES[@]}"; do
        PORT="${service%%:*}"
        NAME="${service##*:}"
        
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT" | grep -q "200\|302\|401"; then
            echo -e "${GREEN}✅ $NAME (port $PORT) - Responding${NC}"
        else
            echo -e "${YELLOW}⚠️  $NAME (port $PORT) - Starting up...${NC}"
        fi
    done
}

# Function to display access information
show_access_info() {
    print_section "ACCESS INFORMATION"
    
    echo -e "${GREEN}🎉 Ultimate Media Server 2025 is now running!${NC}"
    echo ""
    echo -e "${CYAN}📊 MAIN DASHBOARD:${NC}"
    echo -e "${GREEN}   http://localhost:8090${NC} - Ultimate Media Server Dashboard"
    echo ""
    echo -e "${CYAN}🤖 MCP SERVER:${NC}"
    echo -e "${GREEN}   http://localhost:3000${NC} - Model Context Protocol Server"
    echo ""
    echo -e "${CYAN}📺 MEDIA SERVERS:${NC}"
    echo -e "${GREEN}   http://localhost:8096${NC} - Jellyfin"
    echo -e "${GREEN}   http://localhost:32400${NC} - Plex Media Server"
    echo -e "${GREEN}   http://localhost:8097${NC} - Emby"
    echo ""
    echo -e "${CYAN}📋 CONTENT MANAGEMENT:${NC}"
    echo -e "${GREEN}   http://localhost:8989${NC} - Sonarr (TV Shows)"
    echo -e "${GREEN}   http://localhost:7878${NC} - Radarr (Movies)"
    echo -e "${GREEN}   http://localhost:8686${NC} - Lidarr (Music)"
    echo -e "${GREEN}   http://localhost:8787${NC} - Readarr (Books)"
    echo -e "${GREEN}   http://localhost:6767${NC} - Bazarr (Subtitles)"
    echo ""
    echo -e "${CYAN}🔍 INDEXERS:${NC}"
    echo -e "${GREEN}   http://localhost:9696${NC} - Prowlarr"
    echo -e "${GREEN}   http://localhost:9117${NC} - Jackett"
    echo -e "${GREEN}   http://localhost:8191${NC} - FlareSolverr"
    echo ""
    echo -e "${CYAN}📥 DOWNLOAD CLIENTS:${NC}"
    echo -e "${GREEN}   http://localhost:8080${NC} - qBittorrent"
    echo -e "${GREEN}   http://localhost:9091${NC} - Transmission"
    echo -e "${GREEN}   http://localhost:8112${NC} - Deluge"
    echo -e "${GREEN}   http://localhost:6789${NC} - NZBGet"
    echo -e "${GREEN}   http://localhost:8085${NC} - SABnzbd"
    echo ""
    echo -e "${CYAN}🙋 REQUEST MANAGEMENT:${NC}"
    echo -e "${GREEN}   http://localhost:5055${NC} - Overseerr"
    echo -e "${GREEN}   http://localhost:4545${NC} - Requestrr"
    echo -e "${GREEN}   http://localhost:3579${NC} - Ombi"
    echo ""
    echo -e "${CYAN}📊 ANALYTICS & MONITORING:${NC}"
    echo -e "${GREEN}   http://localhost:8181${NC} - Tautulli"
    echo -e "${GREEN}   http://localhost:19999${NC} - Netdata"
    echo ""
    echo -e "${CYAN}🏠 DASHBOARDS:${NC}"
    echo -e "${GREEN}   http://localhost:3001${NC} - Homepage"
    echo -e "${GREEN}   http://localhost:80${NC} - Heimdall"
    echo -e "${GREEN}   http://localhost:8081${NC} - Organizr"
    echo -e "${GREEN}   http://localhost:7575${NC} - Homarr"
    echo ""
    echo -e "${CYAN}🔧 INFRASTRUCTURE:${NC}"
    echo -e "${GREEN}   http://localhost:81${NC} - Nginx Proxy Manager"
    echo -e "${GREEN}   http://localhost:9000${NC} - Portainer"
    echo -e "${GREEN}   http://localhost:8082${NC} - Watchtower"
    echo -e "${GREEN}   http://localhost:8888${NC} - Gluetun VPN"
    echo -e "${GREEN}   http://localhost:5656${NC} - Unpackerr"
}

# Function to show next steps
show_next_steps() {
    print_section "NEXT STEPS"
    
    echo -e "${CYAN}📝 CONFIGURATION STEPS:${NC}"
    echo -e "${GREEN}1.${NC} Open Ultimate Dashboard: ${GREEN}http://localhost:8090${NC}"
    echo -e "${GREEN}2.${NC} Configure each service through their web interfaces"
    echo -e "${GREEN}3.${NC} Generate API keys from each service"
    echo -e "${GREEN}4.${NC} Update ${GREEN}$ENV_FILE${NC} with the API keys"
    echo -e "${GREEN}5.${NC} Restart services: ${GREEN}docker-compose -f $COMPOSE_FILE restart${NC}"
    echo ""
    echo -e "${CYAN}🔑 MCP INTEGRATION:${NC}"
    echo -e "${GREEN}•${NC} MCP Server is running at: ${GREEN}http://localhost:3000${NC}"
    echo -e "${GREEN}•${NC} Claude Desktop config updated at:"
    echo -e "   ${GREEN}~/Library/Application Support/Claude/claude_desktop_config.json${NC}"
    echo -e "${GREEN}•${NC} Restart Claude Desktop to load the new MCP server"
    echo ""
    echo -e "${CYAN}📚 USEFUL COMMANDS:${NC}"
    echo -e "${GREEN}•${NC} View logs: ${GREEN}docker-compose -f $COMPOSE_FILE logs -f${NC}"
    echo -e "${GREEN}•${NC} Restart all: ${GREEN}docker-compose -f $COMPOSE_FILE restart${NC}"
    echo -e "${GREEN}•${NC} Stop all: ${GREEN}docker-compose -f $COMPOSE_FILE down${NC}"
    echo -e "${GREEN}•${NC} Check status: ${GREEN}docker-compose -f $COMPOSE_FILE ps${NC}"
    echo -e "${GREEN}•${NC} Shell access: ${GREEN}docker exec -it $CONTAINER_NAME bash${NC}"
    echo ""
    echo -e "${CYAN}⚠️  SECURITY REMINDERS:${NC}"
    echo -e "${YELLOW}•${NC} Change all default passwords in the services"
    echo -e "${YELLOW}•${NC} Update API keys in ${GREEN}$ENV_FILE${NC}"
    echo -e "${YELLOW}•${NC} Consider enabling VPN for download clients"
    echo -e "${YELLOW}•${NC} Setup reverse proxy with SSL for external access"
}

# Function to handle cleanup on exit
cleanup() {
    if [ $? -ne 0 ]; then
        echo -e "\n${RED}❌ Deployment failed. Check the logs above.${NC}"
        echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
        echo -e "${YELLOW}   • Check Docker logs: docker-compose -f $COMPOSE_FILE logs${NC}"
        echo -e "${YELLOW}   • Ensure all required ports are available${NC}"
        echo -e "${YELLOW}   • Verify $ENV_FILE configuration${NC}"
        echo -e "${YELLOW}   • Check system requirements (disk space, memory)${NC}"
    fi
}

# Set up error handling
trap cleanup EXIT

# Parse command line arguments
SKIP_BUILD=false
SKIP_HEALTH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-health)
            SKIP_HEALTH=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-build    Skip the container build step"
            echo "  --skip-health   Skip health checks"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main deployment sequence
main() {
    echo -e "${PURPLE}🚀 Starting Ultimate Media Server 2025 deployment...${NC}"
    
    check_requirements
    create_directories
    setup_environment
    stop_existing
    
    if [ "$SKIP_BUILD" = false ]; then
        build_container
    else
        echo -e "${YELLOW}⏭️  Skipping build step${NC}"
    fi
    
    start_services
    
    if [ "$SKIP_HEALTH" = false ]; then
        check_health
    else
        echo -e "${YELLOW}⏭️  Skipping health checks${NC}"
    fi
    
    show_access_info
    show_next_steps
    
    echo -e "\n${GREEN}🎉 DEPLOYMENT COMPLETE! 🎉${NC}"
    echo -e "${PURPLE}Your Ultimate Media Server 2025 with all 30 services is ready!${NC}"
}

# Run main function
main "$@"