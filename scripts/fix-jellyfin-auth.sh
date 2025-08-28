#!/bin/bash

# Jellyfin Authentication Fix Script
# Comprehensive solution for Jellyfin authentication issues

set -euo pipefail

# Configuration
JELLYFIN_URL="http://localhost:8096"
JELLYFIN_CONTAINER="jellyfin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/jellyfin-auth-fix.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$LOG_FILE"
}

# Check if Jellyfin container is running
check_jellyfin_status() {
    log "Checking Jellyfin container status..."
    if ! docker ps | grep -q "$JELLYFIN_CONTAINER"; then
        error "Jellyfin container is not running!"
        return 1
    fi
    log "Jellyfin container is running ✓"
}

# Wait for Jellyfin to be ready
wait_for_jellyfin() {
    log "Waiting for Jellyfin to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$JELLYFIN_URL/health" > /dev/null 2>&1; then
            log "Jellyfin is ready ✓"
            return 0
        fi
        
        info "Attempt $attempt/$max_attempts - Jellyfin not ready yet, waiting..."
        sleep 10
        ((attempt++))
    done
    
    error "Jellyfin failed to become ready after $max_attempts attempts"
    return 1
}

# Reset Jellyfin authentication
reset_jellyfin_auth() {
    log "Resetting Jellyfin authentication..."
    
    # Stop Jellyfin container
    log "Stopping Jellyfin container..."
    docker stop "$JELLYFIN_CONTAINER" || true
    
    # Clear authentication data
    log "Clearing authentication data..."
    docker exec "$JELLYFIN_CONTAINER" rm -f /config/data/jellyfin.db-wal 2>/dev/null || true
    docker exec "$JELLYFIN_CONTAINER" rm -f /config/data/jellyfin.db-shm 2>/dev/null || true
    
    # Reset system configuration to complete startup wizard
    log "Updating system configuration..."
    docker exec "$JELLYFIN_CONTAINER" bash -c '
        if [ -f "/config/config/system.xml" ]; then
            sed -i "s/<IsStartupWizardCompleted>false<\/IsStartupWizardCompleted>/<IsStartupWizardCompleted>true<\/IsStartupWizardCompleted>/g" /config/config/system.xml
        fi
    ' 2>/dev/null || true
    
    # Start Jellyfin container
    log "Starting Jellyfin container..."
    docker start "$JELLYFIN_CONTAINER"
    
    # Wait for startup
    sleep 20
    wait_for_jellyfin
}

# Configure CORS settings
configure_cors() {
    log "Configuring CORS settings..."
    
    # Create network.xml if it doesn't exist
    docker exec "$JELLYFIN_CONTAINER" bash -c '
        mkdir -p /config/config
        cat > /config/config/network.xml << EOF
<?xml version="1.0" encoding="utf-8"?>
<NetworkConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <EnableHttps>false</EnableHttps>
  <RequireHttps>false</RequireHttps>
  <HttpServerPortNumber>8096</HttpServerPortNumber>
  <HttpsPortNumber>8920</HttpsPortNumber>
  <EnableRemoteAccess>true</EnableRemoteAccess>
  <EnableAutomaticPortForwarding>false</EnableAutomaticPortForwarding>
  <KnownProxies />
  <LocalNetworkSubnets>
    <string>10.0.0.0/8</string>
    <string>172.16.0.0/12</string>
    <string>192.168.0.0/16</string>
    <string>127.0.0.1/32</string>
  </LocalNetworkSubnets>
  <LocalNetworkAddresses />
  <EnableIPV6>false</EnableIPV6>
  <EnableIPV4>true</EnableIPV4>
  <EnablePublishedServerUriByRequest>false</EnablePublishedServerUriByRequest>
  <PublishedServerUriBySubnet />
  <RemoteIPFilter />
  <IsRemoteIPFilterBlacklist>false</IsRemoteIPFilterBlacklist>
</NetworkConfiguration>
EOF
    '
    
    log "CORS settings configured ✓"
}

# Create default admin user
create_admin_user() {
    log "Creating default admin user..."
    
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        # Try to create user via API
        local response=$(curl -s -X POST "$JELLYFIN_URL/Startup/User" \
            -H "Content-Type: application/json" \
            -d '{
                "Name": "admin",
                "Password": "admin123"
            }' 2>/dev/null || echo "failed")
        
        if [[ "$response" != "failed" && "$response" != *"error"* ]]; then
            log "Admin user created successfully ✓"
            return 0
        fi
        
        warning "Attempt $attempt/$max_attempts failed to create user, retrying..."
        sleep 5
        ((attempt++))
    done
    
    warning "Could not create admin user via API, user may already exist"
}

# Complete startup wizard
complete_startup_wizard() {
    log "Completing startup wizard..."
    
    # Complete startup configuration
    curl -s -X POST "$JELLYFIN_URL/Startup/Complete" \
        -H "Content-Type: application/json" \
        -d '{}' >/dev/null 2>&1 || true
    
    log "Startup wizard completed ✓"
}

# Generate API key
generate_api_key() {
    log "Generating API key..."
    
    # First, try to authenticate
    local auth_response=$(curl -s -X POST "$JELLYFIN_URL/Users/AuthenticateByName" \
        -H "Content-Type: application/json" \
        -d '{
            "Username": "admin",
            "Pw": "admin123"
        }' 2>/dev/null || echo "")
    
    if [[ -n "$auth_response" ]]; then
        local access_token=$(echo "$auth_response" | jq -r '.AccessToken' 2>/dev/null || echo "")
        
        if [[ -n "$access_token" && "$access_token" != "null" ]]; then
            log "Authentication successful ✓"
            
            # Create API key
            local api_key_response=$(curl -s -X POST "$JELLYFIN_URL/Auth/Keys" \
                -H "Authorization: MediaBrowser Token=$access_token" \
                -H "Content-Type: application/json" \
                -d '{
                    "App": "Dashboard"
                }' 2>/dev/null || echo "")
            
            if [[ -n "$api_key_response" ]]; then
                local api_key=$(echo "$api_key_response" | jq -r '.AccessToken' 2>/dev/null || echo "")
                if [[ -n "$api_key" && "$api_key" != "null" ]]; then
                    log "API Key generated: $api_key"
                    echo "$api_key" > "$SCRIPT_DIR/jellyfin-api-key.txt"
                    log "API key saved to jellyfin-api-key.txt ✓"
                fi
            fi
        fi
    fi
}

# Test API endpoints
test_api_endpoints() {
    log "Testing API endpoints..."
    
    # Test basic endpoints
    local endpoints=(
        "/System/Info"
        "/System/Configuration"
        "/Users"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s --connect-timeout 5 "$JELLYFIN_URL$endpoint" > /dev/null 2>&1; then
            log "✓ $endpoint - OK"
        else
            warning "✗ $endpoint - Failed"
        fi
    done
}

# Fix permissions
fix_permissions() {
    log "Fixing Jellyfin permissions..."
    
    docker exec "$JELLYFIN_CONTAINER" bash -c '
        chown -R jellyfin:jellyfin /config
        chmod -R 755 /config
    ' 2>/dev/null || true
    
    log "Permissions fixed ✓"
}

# Update system configuration with proper settings
update_system_config() {
    log "Updating system configuration..."
    
    docker exec "$JELLYFIN_CONTAINER" bash -c '
        cat > /config/config/system.xml << EOF
<?xml version="1.0" encoding="utf-8"?>
<ServerConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <LogFileRetentionDays>3</LogFileRetentionDays>
  <IsStartupWizardCompleted>true</IsStartupWizardCompleted>
  <EnableMetrics>false</EnableMetrics>
  <EnableNormalizedItemByNameIds>true</EnableNormalizedItemByNameIds>
  <IsPortAuthorized>true</IsPortAuthorized>
  <QuickConnectAvailable>true</QuickConnectAvailable>
  <EnableCaseSensitiveItemIds>true</EnableCaseSensitiveItemIds>
  <DisableLiveTvChannelUserDataName>true</DisableLiveTvChannelUserDataName>
  <MetadataPath />
  <PreferredMetadataLanguage>en</PreferredMetadataLanguage>
  <MetadataCountryCode>US</MetadataCountryCode>
  <RemoteClientBitrateLimit>0</RemoteClientBitrateLimit>
  <EnableFolderView>false</EnableFolderView>
  <EnableGroupingIntoCollections>false</EnableGroupingIntoCollections>
  <DisplaySpecialsWithinSeasons>true</DisplaySpecialsWithinSeasons>
  <LocalNetworkSubnets>
    <string>10.0.0.0/8</string>
    <string>172.16.0.0/12</string>
    <string>192.168.0.0/16</string>
    <string>127.0.0.1/32</string>
  </LocalNetworkSubnets>
  <EnableExternalContentInSuggestions>true</EnableExternalContentInSuggestions>
  <ImageExtractionTimeoutMs>0</ImageExtractionTimeoutMs>
  <PathSubstitutions />
  <UninstalledPlugins />
  <CollapseVideoFolders>false</CollapseVideoFolders>
  <EnablePeoplePrefixSubFolders>true</EnablePeoplePrefixSubFolders>
  <UICulture>en-US</UICulture>
  <SaveMetadataHidden>false</SaveMetadataHidden>
  <ContentTypes />
  <RemoteClientBitrateLimit>0</RemoteClientBitrateLimit>
  <EnableDashboard>true</EnableDashboard>
  <EnableThumbnailsForRemoteItems>true</EnableThumbnailsForRemoteItems>
</ServerConfiguration>
EOF
    '
    
    log "System configuration updated ✓"
}

# Restart Jellyfin with clean start
restart_jellyfin_clean() {
    log "Performing clean restart of Jellyfin..."
    
    docker stop "$JELLYFIN_CONTAINER"
    sleep 5
    docker start "$JELLYFIN_CONTAINER"
    sleep 10
    
    wait_for_jellyfin
}

# Main execution
main() {
    log "========================================="
    log "Jellyfin Authentication Fix Script"
    log "========================================="
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check prerequisites
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        error "curl is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        warning "jq is not installed - some features may not work properly"
    fi
    
    # Execute fix steps
    check_jellyfin_status
    
    log "Step 1: Resetting authentication..."
    reset_jellyfin_auth
    
    log "Step 2: Configuring CORS..."
    configure_cors
    
    log "Step 3: Updating system configuration..."
    update_system_config
    
    log "Step 4: Fixing permissions..."
    fix_permissions
    
    log "Step 5: Restarting Jellyfin..."
    restart_jellyfin_clean
    
    log "Step 6: Completing startup wizard..."
    complete_startup_wizard
    
    log "Step 7: Creating admin user..."
    create_admin_user
    
    log "Step 8: Generating API key..."
    generate_api_key
    
    log "Step 9: Testing API endpoints..."
    test_api_endpoints
    
    log "========================================="
    log "Jellyfin Authentication Fix Complete!"
    log "========================================="
    log ""
    log "Access Jellyfin at: $JELLYFIN_URL"
    log "Default credentials: admin / admin123"
    log "Log file: $LOG_FILE"
    
    if [ -f "$SCRIPT_DIR/jellyfin-api-key.txt" ]; then
        log "API Key saved to: $SCRIPT_DIR/jellyfin-api-key.txt"
    fi
}

# Run main function
main "$@"