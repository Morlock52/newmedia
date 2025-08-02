#!/bin/bash

# Ultimate Media Server 2025 - Deployment Verification Script
# Comprehensive testing of volume mappings, service health, and workflow

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🔍 Ultimate Media Server 2025 - Deployment Verification${NC}"
echo "========================================================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

PLEX_VOLUME="${MEDIA_PATH:-/Volumes/Plex}"
DOWNLOADS_PATH="${DOWNLOADS_PATH:-$PLEX_VOLUME/downloads}"

echo -e "${BLUE}📋 Configuration:${NC}"
echo "   Media Path: $PLEX_VOLUME"
echo "   Downloads Path: $DOWNLOADS_PATH"
echo ""

# Test Results Array
declare -a TEST_RESULTS=()

# Function to add test result
add_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    TEST_RESULTS+=("$status|$test_name|$details")
}

# Function to check service
check_service() {
    local service_name=$1
    local port=$2
    local timeout=${3:-10}
    
    echo -n "   Testing $service_name (port $port)..."
    
    for i in $(seq 1 $timeout); do
        if nc -z localhost $port 2>/dev/null; then
            echo -e " ${GREEN}[ONLINE]${NC}"
            add_result "$service_name Service" "✅" "Port $port responding"
            return 0
        fi
        sleep 1
    done
    
    echo -e " ${RED}[OFFLINE]${NC}"
    add_result "$service_name Service" "❌" "Port $port not responding"
    return 1
}

# Test 1: Volume Accessibility
echo -e "${YELLOW}🔍 Test 1: Volume Accessibility${NC}"
echo ""

if [ -d "$PLEX_VOLUME" ]; then
    echo -e "${GREEN}✅ $PLEX_VOLUME exists${NC}"
    add_result "Volume Exists" "✅" "$PLEX_VOLUME accessible"
else
    echo -e "${RED}❌ $PLEX_VOLUME does not exist${NC}"
    add_result "Volume Exists" "❌" "$PLEX_VOLUME not found"
fi

# Test write permissions
TEST_FILE="$PLEX_VOLUME/test_permissions_$(date +%s).txt"
if echo "Permission test" > "$TEST_FILE" 2>/dev/null; then
    echo -e "${GREEN}✅ Write permissions OK${NC}"
    rm -f "$TEST_FILE"
    add_result "Write Permissions" "✅" "Volume writable"
else
    echo -e "${RED}❌ No write permissions${NC}"
    add_result "Write Permissions" "❌" "Volume not writable"
fi

# Test 2: Directory Structure
echo ""
echo -e "${YELLOW}🔍 Test 2: Directory Structure${NC}"
echo ""

REQUIRED_DIRS=("movies" "tv" "music" "audiobooks" "books" "comics" "photos" "podcasts")
MISSING_DIRS=()

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$PLEX_VOLUME/$dir" ]; then
        echo -e "${GREEN}✅ $dir folder exists${NC}"
    else
        echo -e "${RED}❌ $dir folder missing${NC}"
        MISSING_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    add_result "Directory Structure" "✅" "All media directories present"
else
    add_result "Directory Structure" "⚠️" "Missing: ${MISSING_DIRS[*]}"
fi

# Test download directories
DOWNLOAD_DIRS=("complete" "incomplete" "torrents" "watch")
MISSING_DOWNLOAD_DIRS=()

for dir in "${DOWNLOAD_DIRS[@]}"; do
    if [ -d "$DOWNLOADS_PATH/$dir" ]; then
        echo -e "${GREEN}✅ downloads/$dir exists${NC}"
    else
        echo -e "${RED}❌ downloads/$dir missing${NC}"
        MISSING_DOWNLOAD_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DOWNLOAD_DIRS[@]} -eq 0 ]; then
    add_result "Download Structure" "✅" "All download directories present"
else
    add_result "Download Structure" "⚠️" "Missing: ${MISSING_DOWNLOAD_DIRS[*]}"
fi

# Test 3: Container Status
echo ""
echo -e "${YELLOW}🔍 Test 3: Container Status${NC}"
echo ""

EXPECTED_CONTAINERS=("jellyfin" "sonarr" "radarr" "prowlarr" "overseerr" "qbittorrent")
RUNNING_COUNT=0

for container in "${EXPECTED_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^$container$"; then
        status=$(docker ps --format '{{.Status}}' --filter "name=$container")
        echo -e "${GREEN}✅ $container: $status${NC}"
        ((RUNNING_COUNT++))
    else
        if docker ps -a --format '{{.Names}}' | grep -q "^$container$"; then
            status=$(docker ps -a --format '{{.Status}}' --filter "name=$container")
            echo -e "${YELLOW}⚠️  $container: $status${NC}"
        else
            echo -e "${RED}❌ $container: Not found${NC}"
        fi
    fi
done

add_result "Container Status" "ℹ️" "$RUNNING_COUNT/${#EXPECTED_CONTAINERS[@]} containers running"

# Test 4: Service Connectivity
echo ""
echo -e "${YELLOW}🔍 Test 4: Service Connectivity${NC}"
echo ""

SERVICES_TO_CHECK=(
    "Jellyfin:8096"
    "Sonarr:8989"
    "Radarr:7878"
    "Prowlarr:9696"
    "Overseerr:5055"
    "qBittorrent:8080"
)

ONLINE_SERVICES=0
for service_info in "${SERVICES_TO_CHECK[@]}"; do
    IFS=':' read -r service_name port <<< "$service_info"
    if check_service "$service_name" "$port" 3; then
        ((ONLINE_SERVICES++))
    fi
done

add_result "Service Connectivity" "ℹ️" "$ONLINE_SERVICES/${#SERVICES_TO_CHECK[@]} services online"

# Test 5: Volume Mounts in Containers
echo ""
echo -e "${YELLOW}🔍 Test 5: Container Volume Mounts${NC}"
echo ""

MOUNT_TESTS=(
    "jellyfin:/media"
    "sonarr:/media"
    "radarr:/media"
    "prowlarr:/config"
)

SUCCESSFUL_MOUNTS=0
for mount_test in "${MOUNT_TESTS[@]}"; do
    IFS=':' read -r container mount_point <<< "$mount_test"
    echo -n "   Testing $container mount $mount_point..."
    
    if docker exec "$container" ls "$mount_point" >/dev/null 2>&1; then
        echo -e " ${GREEN}[OK]${NC}"
        ((SUCCESSFUL_MOUNTS++))
    else
        echo -e " ${RED}[FAILED]${NC}"
    fi
done

add_result "Volume Mounts" "ℹ️" "$SUCCESSFUL_MOUNTS/${#MOUNT_TESTS[@]} mounts accessible"

# Test 6: API Health Checks
echo ""
echo -e "${YELLOW}🔍 Test 6: API Health Checks${NC}"
echo ""

# Jellyfin health check
echo -n "   Testing Jellyfin API..."
if curl -s "http://localhost:8096/health" | grep -q "Healthy" 2>/dev/null; then
    echo -e " ${GREEN}[HEALTHY]${NC}"
    add_result "Jellyfin API" "✅" "Health endpoint responding"
else
    echo -e " ${RED}[UNHEALTHY]${NC}"
    add_result "Jellyfin API" "❌" "Health endpoint not responding"
fi

# Sonarr system status
echo -n "   Testing Sonarr API..."
if curl -s "http://localhost:8989/api/v3/system/status" >/dev/null 2>&1; then
    echo -e " ${GREEN}[RESPONDING]${NC}"
    add_result "Sonarr API" "✅" "API responding"
else
    echo -e " ${YELLOW}[NEEDS CONFIG]${NC}"
    add_result "Sonarr API" "⚠️" "API needs configuration"
fi

# Radarr system status  
echo -n "   Testing Radarr API..."
if curl -s "http://localhost:7878/api/v3/system/status" >/dev/null 2>&1; then
    echo -e " ${GREEN}[RESPONDING]${NC}"
    add_result "Radarr API" "✅" "API responding"
else
    echo -e " ${YELLOW}[NEEDS CONFIG]${NC}"
    add_result "Radarr API" "⚠️" "API needs configuration" 
fi

# Test 7: File System Operations
echo ""
echo -e "${YELLOW}🔍 Test 7: File System Operations${NC}"
echo ""

# Create test files in each media directory
TEST_FILES_CREATED=0
for dir in "${REQUIRED_DIRS[@]}"; do
    test_file="$PLEX_VOLUME/$dir/test_file_$(date +%s).txt"
    if echo "Test content for $dir" > "$test_file" 2>/dev/null; then
        echo -e "${GREEN}✅ Can write to $dir${NC}"
        rm -f "$test_file"
        ((TEST_FILES_CREATED++))
    else
        echo -e "${RED}❌ Cannot write to $dir${NC}"
    fi
done

if [ $TEST_FILES_CREATED -eq ${#REQUIRED_DIRS[@]} ]; then
    add_result "File Operations" "✅" "All directories writable"
else
    add_result "File Operations" "⚠️" "$TEST_FILES_CREATED/${#REQUIRED_DIRS[@]} directories writable"
fi

# Test 8: Resource Usage
echo ""
echo -e "${YELLOW}🔍 Test 8: Resource Usage${NC}"
echo ""

# Docker system info
DOCKER_CONTAINERS=$(docker ps -q | wc -l | tr -d ' ')
DOCKER_IMAGES=$(docker images -q | wc -l | tr -d ' ')
DOCKER_VOLUMES=$(docker volume ls -q | wc -l | tr -d ' ')

echo -e "${BLUE}📊 Docker Resources:${NC}"
echo "   Containers: $DOCKER_CONTAINERS running"
echo "   Images: $DOCKER_IMAGES total"
echo "   Volumes: $DOCKER_VOLUMES total"

add_result "Resource Usage" "ℹ️" "$DOCKER_CONTAINERS containers, $DOCKER_IMAGES images"

# Disk usage
if command -v df >/dev/null 2>&1; then
    DISK_USAGE=$(df -h "$PLEX_VOLUME" 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -lt 90 ]; then
        echo -e "${GREEN}✅ Disk usage: ${DISK_USAGE}%${NC}"
        add_result "Disk Usage" "✅" "${DISK_USAGE}% used"
    else
        echo -e "${YELLOW}⚠️  Disk usage: ${DISK_USAGE}%${NC}"
        add_result "Disk Usage" "⚠️" "${DISK_USAGE}% used (high)"
    fi
fi

# Generate Test Results Summary
echo ""
echo -e "${CYAN}📊 Test Results Summary${NC}"
echo "======================="
echo ""

SUCCESS_COUNT=0
WARNING_COUNT=0
ERROR_COUNT=0

for result in "${TEST_RESULTS[@]}"; do
    IFS='|' read -r status test_name details <<< "$result"
    case "$status" in
        "✅") printf "%-20s %s %s\n" "$status" "$test_name" "$details"; ((SUCCESS_COUNT++)) ;;
        "⚠️") printf "%-20s %s %s\n" "$status" "$test_name" "$details"; ((WARNING_COUNT++)) ;;
        "❌") printf "%-20s %s %s\n" "$status" "$test_name" "$details"; ((ERROR_COUNT++)) ;;
        "ℹ️") printf "%-20s %s %s\n" "$status" "$test_name" "$details" ;;
    esac
done

echo ""
echo -e "${CYAN}📋 Summary Statistics:${NC}"
echo "   ✅ Successful: $SUCCESS_COUNT"
echo "   ⚠️  Warnings: $WARNING_COUNT" 
echo "   ❌ Errors: $ERROR_COUNT"

# Overall Health Assessment
echo ""
if [ $ERROR_COUNT -eq 0 ] && [ $WARNING_COUNT -le 2 ]; then
    echo -e "${GREEN}🎉 Overall Status: HEALTHY${NC}"
    echo "   Your media server deployment is working correctly!"
elif [ $ERROR_COUNT -le 2 ]; then
    echo -e "${YELLOW}⚠️  Overall Status: NEEDS ATTENTION${NC}"
    echo "   Some issues detected but server should be functional."
else
    echo -e "${RED}❌ Overall Status: CRITICAL ISSUES${NC}"
    echo "   Multiple problems detected. Review configuration."
fi

# Generate Deployment Report
cat > deployment-verification-report.md << EOF
# Ultimate Media Server 2025 - Verification Report

**Verification Date:** $(date)
**Media Volume:** $PLEX_VOLUME
**Downloads Path:** $DOWNLOADS_PATH

## Test Results Summary

| Status | Test | Details |
|--------|------|---------|
$(for result in "${TEST_RESULTS[@]}"; do
    IFS='|' read -r status test_name details <<< "$result"
    echo "| $status | $test_name | $details |"
done)

## Statistics

- ✅ Successful Tests: $SUCCESS_COUNT
- ⚠️  Warning Tests: $WARNING_COUNT
- ❌ Failed Tests: $ERROR_COUNT

## Service URLs

- **Jellyfin:** http://localhost:8096
- **Sonarr:** http://localhost:8989
- **Radarr:** http://localhost:7878
- **Prowlarr:** http://localhost:9696
- **Overseerr:** http://localhost:5055
- **qBittorrent:** http://localhost:8080

## Recommendations

$(if [ $ERROR_COUNT -eq 0 ]; then
    echo "✅ No critical issues detected. Your media server is ready for use."
else
    echo "❌ Critical issues detected:"
    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r status test_name details <<< "$result"
        if [ "$status" = "❌" ]; then
            echo "- Fix $test_name: $details"
        fi
    done
fi)

$(if [ $WARNING_COUNT -gt 0 ]; then
    echo ""
    echo "⚠️  Warnings that should be addressed:"
    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r status test_name details <<< "$result"
        if [ "$status" = "⚠️" ]; then
            echo "- $test_name: $details"
        fi
    done
fi)

---
*Report generated by verify-deployment.sh on $(date)*
EOF

echo ""
echo -e "${GREEN}📄 Detailed report saved: deployment-verification-report.md${NC}"
echo ""
echo -e "${BLUE}🔗 Quick Access Links:${NC}"
echo "   • View report: cat deployment-verification-report.md"
echo "   • Jellyfin: http://localhost:8096"
echo "   • Media management: http://localhost:8989 (Sonarr)"
echo ""
echo -e "${PURPLE}🚀 Verification complete!${NC}"