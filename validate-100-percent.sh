#!/bin/bash
# Ultimate Media Server 2025 - 100% Operational Validation Script

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Log functions
log_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
    ((TOTAL_CHECKS++))
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_CHECKS++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_CHECKS++))
}

# Banner
echo -e "${CYAN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║     ULTIMATE MEDIA SERVER 2025 - 100% VALIDATION CHECK         ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if service is running and accessible
check_service() {
    local service=$1
    local port=$2
    local name=$3
    
    log_check "Checking ${name} (${service})"
    
    # Check if container is running
    if docker ps | grep -q "$service"; then
        # Check if port is accessible
        if curl -f -s -o /dev/null "http://localhost:${port}" || curl -f -s -o /dev/null "http://localhost:${port}/login" || curl -f -s -o /dev/null "http://localhost:${port}/api"; then
            log_pass "${name} is running and accessible on port ${port}"
            return 0
        else
            # Try with timeout for services that might be initializing
            if timeout 10 curl -f -s -o /dev/null --retry 3 --retry-delay 2 "http://localhost:${port}"; then
                log_pass "${name} is running and accessible on port ${port}"
                return 0
            else
                log_fail "${name} container is running but not accessible on port ${port}"
                return 1
            fi
        fi
    else
        log_fail "${name} container is not running"
        return 1
    fi
}

# Check configuration files
check_config() {
    local service=$1
    local config_path=$2
    
    log_check "Checking ${service} configuration"
    
    if [ -d "$config_path" ]; then
        if [ "$(ls -A $config_path 2>/dev/null)" ]; then
            log_pass "${service} configuration exists"
            return 0
        else
            log_fail "${service} configuration directory is empty"
            return 1
        fi
    else
        log_fail "${service} configuration directory not found"
        return 1
    fi
}

# Check volumes
check_volume() {
    local volume=$1
    local description=$2
    
    log_check "Checking ${description}"
    
    if docker volume inspect "$volume" &>/dev/null || [ -d "$volume" ]; then
        log_pass "${description} exists"
        return 0
    else
        log_fail "${description} not found"
        return 1
    fi
}

# Check network connectivity
check_network() {
    local network=$1
    
    log_check "Checking Docker network: ${network}"
    
    if docker network inspect "$network" &>/dev/null; then
        log_pass "Docker network ${network} exists"
        return 0
    else
        log_fail "Docker network ${network} not found"
        return 1
    fi
}

# Main validation
echo -e "${CYAN}🔍 Starting comprehensive validation...${NC}\n"

# Check Docker
log_check "Checking Docker daemon"
if docker info &>/dev/null; then
    log_pass "Docker daemon is running"
else
    log_fail "Docker daemon is not running"
    exit 1
fi

# Check Docker Compose
log_check "Checking Docker Compose"
if docker-compose version &>/dev/null || docker compose version &>/dev/null; then
    log_pass "Docker Compose is available"
else
    log_fail "Docker Compose not found"
fi

echo -e "\n${CYAN}📦 Validating Core Services...${NC}"

# Core Media Services
check_service "jellyfin" "8096" "Jellyfin Media Server"

# Arr Suite
check_service "sonarr" "8989" "Sonarr (TV Shows)"
check_service "radarr" "7878" "Radarr (Movies)"
check_service "lidarr" "8686" "Lidarr (Music)"
check_service "readarr" "8787" "Readarr (Books)"
check_service "prowlarr" "9696" "Prowlarr (Indexers)"
check_service "bazarr" "6767" "Bazarr (Subtitles)"

# Request Management
check_service "overseerr" "5055" "Overseerr"
check_service "jellyseerr" "5056" "Jellyseerr"

# Download Clients
check_service "qbittorrent" "8080" "qBittorrent"
check_service "sabnzbd" "8081" "SABnzbd"

# Monitoring
check_service "tautulli" "8181" "Tautulli"
check_service "grafana" "3000" "Grafana"
check_service "prometheus" "9090" "Prometheus"

# Management
check_service "homepage" "3001" "Homepage Dashboard"
check_service "portainer" "9000" "Portainer"
check_service "traefik" "8082" "Traefik"

# Additional Services
check_service "authentik" "9091" "Authentik SSO"
check_service "immich" "2283" "Immich Photos"
check_service "navidrome" "4533" "Navidrome Music"
check_service "calibre-web" "8083" "Calibre Web"
check_service "audiobookshelf" "13378" "Audiobookshelf"
check_service "duplicati" "8200" "Duplicati Backup"

echo -e "\n${CYAN}📁 Validating Configurations...${NC}"

# Check configurations
check_config "Jellyfin" "./config/jellyfin"
check_config "Sonarr" "./config/sonarr"
check_config "Radarr" "./config/radarr"
check_config "Prowlarr" "./config/prowlarr"

echo -e "\n${CYAN}🌐 Validating Networks...${NC}"

# Check networks
check_network "media_network"
check_network "download_network"

echo -e "\n${CYAN}💾 Validating Storage...${NC}"

# Check directories
check_volume "./media-data" "Media directory"
check_volume "./config" "Configuration directory"
check_volume "./downloads" "Downloads directory"

echo -e "\n${CYAN}🔒 Validating Security...${NC}"

# Check for environment file
log_check "Checking environment configuration"
if [ -f ".env" ]; then
    log_pass "Environment file exists"
else
    log_fail "Environment file not found"
fi

# Check for passwords file
log_check "Checking passwords file"
if [ -f "./config/.passwords" ]; then
    log_pass "Passwords file exists"
else
    log_fail "Passwords file not found"
fi

echo -e "\n${CYAN}🚀 Validating Performance Features...${NC}"

# Check for AI recommendations (might not have the image yet)
log_check "Checking AI recommendations service"
if docker ps | grep -q "ai-recommendations"; then
    log_pass "AI recommendations service is running"
else
    log_fail "AI recommendations service not running (optional)"
fi

# Check monitoring stack
log_check "Checking monitoring stack"
if docker ps | grep -q "prometheus" && docker ps | grep -q "grafana"; then
    log_pass "Monitoring stack is operational"
else
    log_fail "Monitoring stack incomplete"
fi

# Final Summary
echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 VALIDATION SUMMARY${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"

PERCENTAGE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

echo -e "Total Checks: ${TOTAL_CHECKS}"
echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"
echo -e "\nOperational Status: ${PERCENTAGE}%"

# Progress bar
printf "\n["
FILLED=$((PERCENTAGE / 2))
EMPTY=$((50 - FILLED))
printf "%${FILLED}s" | tr ' ' '█'
printf "%${EMPTY}s" | tr ' ' '▒'
printf "] ${PERCENTAGE}%%\n\n"

if [ $PERCENTAGE -eq 100 ]; then
    echo -e "${GREEN}✅ CONGRATULATIONS! Your Ultimate Media Server 2025 is 100% OPERATIONAL!${NC}"
    echo -e "${GREEN}   All services are running perfectly! 🎉${NC}"
elif [ $PERCENTAGE -ge 90 ]; then
    echo -e "${GREEN}✅ Your media server is ${PERCENTAGE}% operational - Excellent!${NC}"
    echo -e "${YELLOW}   Minor issues detected but core functionality is intact.${NC}"
elif [ $PERCENTAGE -ge 75 ]; then
    echo -e "${YELLOW}⚠️  Your media server is ${PERCENTAGE}% operational - Good${NC}"
    echo -e "${YELLOW}   Some services need attention.${NC}"
else
    echo -e "${RED}❌ Your media server is only ${PERCENTAGE}% operational${NC}"
    echo -e "${RED}   Critical issues detected. Please run the installer or check logs.${NC}"
fi

# Provide fix suggestions if not 100%
if [ $FAILED_CHECKS -gt 0 ]; then
    echo -e "\n${CYAN}🔧 Suggested Fixes:${NC}"
    echo -e "1. Run: ${BLUE}docker-compose up -d${NC} to start missing containers"
    echo -e "2. Check logs: ${BLUE}docker-compose logs [service-name]${NC}"
    echo -e "3. Verify .env file has correct values"
    echo -e "4. Ensure ports are not already in use"
    echo -e "5. Run: ${BLUE}./health-check.sh${NC} for detailed diagnostics"
fi

echo -e "\n${CYAN}📝 For detailed logs, check:${NC}"
echo -e "   ${BLUE}./logs/install.log${NC} - Installation log"
echo -e "   ${BLUE}docker-compose logs${NC} - Container logs"

exit $FAILED_CHECKS