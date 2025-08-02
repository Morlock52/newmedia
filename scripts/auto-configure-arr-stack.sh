#!/bin/bash

# Auto-Configure ARR Stack Script
# Based on 2025 community best practices and automation tools

set -e

echo "🚀 ARR Stack Auto-Configuration Script v2025"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

# Create necessary directories
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p "$CONFIG_DIR"/{prowlarr,sonarr,radarr,lidarr,readarr,bazarr,recyclarr,qbittorrent}
mkdir -p "$PROJECT_ROOT"/media/{movies,tv,music,books,downloads}
mkdir -p "$PROJECT_ROOT"/media/downloads/{complete,incomplete,torrents,usenet}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}Waiting for $service_name to be ready...${NC}"
    while ! curl -s "http://localhost:$port" > /dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}Failed to connect to $service_name after $max_attempts attempts${NC}"
            return 1
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo -e "${GREEN}$service_name is ready!${NC}"
    return 0
}

# Function to get API key from service
get_api_key() {
    local service=$1
    local config_path=$2
    local api_key=""
    
    # Extract API key from config.xml
    if [ -f "$config_path/config.xml" ]; then
        api_key=$(grep -oP '(?<=<ApiKey>)[^<]+' "$config_path/config.xml" 2>/dev/null || echo "")
    fi
    
    echo "$api_key"
}

# Start the stack
echo -e "${BLUE}Starting Docker containers...${NC}"
docker-compose -f "$COMPOSE_FILE" up -d prowlarr sonarr radarr lidarr bazarr qbittorrent

# Wait for all services to be ready
sleep 10
wait_for_service "Prowlarr" 9696
wait_for_service "Sonarr" 8989
wait_for_service "Radarr" 7878
wait_for_service "Lidarr" 8686

# Auto-configure Prowlarr indexers
echo -e "${BLUE}Configuring Prowlarr with free indexers...${NC}"

# Create Prowlarr configuration script
cat > "$CONFIG_DIR/prowlarr/configure-indexers.sh" << 'EOF'
#!/bin/bash

# Wait for API key to be generated
sleep 5

API_KEY=$(grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
if [ -z "$API_KEY" ]; then
    echo "Failed to get Prowlarr API key"
    exit 1
fi

# Function to add indexer
add_indexer() {
    local name=$1
    local protocol=$2
    local privacy=$3
    local implementation=$4
    
    echo "Adding indexer: $name"
    
    curl -X POST "http://localhost:9696/api/v1/indexer" \
        -H "X-Api-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"enable\": true,
            \"redirect\": false,
            \"supportsRss\": true,
            \"supportsSearch\": true,
            \"protocol\": \"$protocol\",
            \"privacy\": \"$privacy\",
            \"name\": \"$name\",
            \"fields\": [],
            \"implementationName\": \"$implementation\",
            \"implementation\": \"$implementation\",
            \"configContract\": \"${implementation}Settings\",
            \"infoLink\": \"https://wiki.servarr.com/prowlarr/supported-indexers\",
            \"tags\": []
        }" 2>/dev/null || echo "Failed to add $name"
}

# Add popular free public indexers
add_indexer "1337x" "torrent" "public" "Cardigann"
add_indexer "The Pirate Bay" "torrent" "public" "Cardigann"
add_indexer "RARBG" "torrent" "public" "Cardigann"
add_indexer "YTS" "torrent" "public" "Cardigann"
add_indexer "EZTV" "torrent" "public" "Cardigann"
add_indexer "Nyaa" "torrent" "public" "Cardigann"
add_indexer "LimeTorrents" "torrent" "public" "Cardigann"

echo "Indexer configuration complete!"
EOF

chmod +x "$CONFIG_DIR/prowlarr/configure-indexers.sh"

# Configure ARR applications in Prowlarr
echo -e "${BLUE}Configuring ARR applications in Prowlarr...${NC}"

cat > "$CONFIG_DIR/prowlarr/configure-apps.sh" << 'EOF'
#!/bin/bash

API_KEY=$(grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
if [ -z "$API_KEY" ]; then
    echo "Failed to get Prowlarr API key"
    exit 1
fi

# Function to add application
add_application() {
    local name=$1
    local implementation=$2
    local port=$3
    local apiPath=$4
    
    # Get the application's API key
    local app_api_key=$(docker exec $name grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
    
    if [ -z "$app_api_key" ]; then
        echo "Failed to get $name API key"
        return
    fi
    
    echo "Adding $name to Prowlarr..."
    
    curl -X POST "http://localhost:9696/api/v1/applications" \
        -H "X-Api-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"syncLevel\": \"fullSync\",
            \"name\": \"$name\",
            \"fields\": [
                {\"name\": \"baseUrl\", \"value\": \"http://$name:$port\"},
                {\"name\": \"apiKey\", \"value\": \"$app_api_key\"},
                {\"name\": \"syncCategories\", \"value\": []}
            ],
            \"implementationName\": \"$implementation\",
            \"implementation\": \"$implementation\",
            \"configContract\": \"${implementation}Settings\",
            \"infoLink\": \"https://wiki.servarr.com/prowlarr/supported-applications\",
            \"tags\": []
        }" 2>/dev/null || echo "Failed to add $name"
}

# Add all ARR applications
sleep 10
add_application "sonarr" "Sonarr" "8989" "/api"
add_application "radarr" "Radarr" "7878" "/api"
add_application "lidarr" "Lidarr" "8686" "/api"

echo "Application configuration complete!"
EOF

chmod +x "$CONFIG_DIR/prowlarr/configure-apps.sh"

# Create Recyclarr configuration
echo -e "${BLUE}Creating Recyclarr configuration...${NC}"

cat > "$CONFIG_DIR/recyclarr/recyclarr.yml" << 'EOF'
# Recyclarr Configuration
# Updated for 2025 - Using TRaSH Guides recommendations

sonarr:
  sonarr-main:
    base_url: http://sonarr:8989
    api_key: !env_var SONARR_API_KEY
    
    quality_definition:
      type: series
      
    quality_profiles:
      - name: HD-1080p
        min_format_score: 0
        
    custom_formats:
      - trash_ids:
          - 0f12c086e289cf966fa5948eac571f44  # Hybrid
          - 570bc9ebecd92723d2d21500f4be314c  # Remaster
          - eca37840c13c6ef2dd0262b141a5482f  # 4K Remaster
          - e0c07d59beb37348e975a930d5e50319  # Criterion Collection
          - 9d27d9d2181838f76dee150882bdc58c  # Masters of Cinema
        quality_profiles:
          - name: HD-1080p
            score: 10

radarr:
  radarr-main:
    base_url: http://radarr:7878
    api_key: !env_var RADARR_API_KEY
    
    quality_definition:
      type: movie
      
    quality_profiles:
      - name: HD-1080p
        min_format_score: 0
        
    custom_formats:
      - trash_ids:
          - 3a3ff47579026e76d6504ebea39390de  # Remux Tier 01
          - 9f98181fe5a3fbeb0cc29340da2a468a  # Remux Tier 02
          - 4d74ac4c4db0b64bff6ce0cffef99bf0  # UHD Bluray Tier 01
          - a58f517a70193f8e578056642178419d  # UHD Bluray Tier 02
        quality_profiles:
          - name: HD-1080p
            score: 100
EOF

# Create auto-update script
echo -e "${BLUE}Creating auto-update script...${NC}"

cat > "$SCRIPT_DIR/auto-update-configs.sh" << 'EOF'
#!/bin/bash

# Auto-update configurations using Recyclarr
echo "Running Recyclarr sync..."
docker run --rm \
    -v $(pwd)/config/recyclarr:/config \
    -e SONARR_API_KEY=$(docker exec sonarr grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml) \
    -e RADARR_API_KEY=$(docker exec radarr grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml) \
    ghcr.io/recyclarr/recyclarr:latest sync

echo "Configuration sync complete!"
EOF

chmod +x "$SCRIPT_DIR/auto-update-configs.sh"

# Create download client auto-configuration
echo -e "${BLUE}Configuring qBittorrent...${NC}"

cat > "$CONFIG_DIR/qbittorrent/auto-configure.sh" << 'EOF'
#!/bin/bash

# Wait for qBittorrent to initialize
sleep 10

# Default qBittorrent settings for optimal performance
cat > /config/qBittorrent/qBittorrent.conf << 'EOFCONF'
[Preferences]
WebUI\Username=admin
WebUI\Password_PBKDF2="@ByteArray(ARCwPWE7RbUSOoJ4n8o+jw==:KQ6n9oxPtFMJlHdqnJ9Vc6wDKdCkPvDZrhzXRvQrPrs6OedFMrKfH3G5h5sD8A9ib2LkCst7u7OpnwQJmLDK7g==)"
WebUI\Port=8080
WebUI\LocalHostAuth=false
Downloads\SavePath=/downloads/complete/
Downloads\TempPath=/downloads/incomplete/
Downloads\PreAllocation=true
Downloads\UseIncompleteExtension=true
Connection\GlobalDLLimit=0
Connection\GlobalUPLimit=0
Connection\PortRangeMin=6881
Queueing\QueueingEnabled=false
EOFCONF

echo "qBittorrent configuration complete!"
EOF

chmod +x "$CONFIG_DIR/qbittorrent/auto-configure.sh"

# Execute configurations
echo -e "${BLUE}Executing auto-configuration...${NC}"

# Run Prowlarr configurations
docker exec prowlarr /config/configure-indexers.sh
sleep 5
docker exec prowlarr /config/configure-apps.sh

# Configure download paths in ARR apps
echo -e "${BLUE}Configuring download paths in ARR applications...${NC}"

# Function to configure download client in ARR app
configure_download_client() {
    local app_name=$1
    local port=$2
    
    local api_key=$(docker exec $app_name grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
    
    if [ -z "$api_key" ]; then
        echo "Failed to get $app_name API key"
        return
    fi
    
    echo "Configuring qBittorrent in $app_name..."
    
    curl -X POST "http://localhost:$port/api/v3/downloadclient" \
        -H "X-Api-Key: $api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "enable": true,
            "protocol": "torrent",
            "priority": 1,
            "name": "qBittorrent",
            "fields": [
                {"name": "host", "value": "qbittorrent"},
                {"name": "port", "value": 8080},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "category", "value": "'$app_name'"},
                {"name": "recentTvPriority", "value": 0},
                {"name": "olderTvPriority", "value": 0},
                {"name": "initialState", "value": 0}
            ],
            "implementationName": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "infoLink": "https://wiki.servarr.com/sonarr/supported#qbittorrent",
            "tags": []
        }' 2>/dev/null || echo "Failed to configure download client in $app_name"
}

sleep 10
configure_download_client "sonarr" 8989
configure_download_client "radarr" 7878
configure_download_client "lidarr" 8686

echo -e "${GREEN}✅ ARR Stack auto-configuration complete!${NC}"
echo -e "${GREEN}Access your services at:${NC}"
echo -e "  ${BLUE}Prowlarr:${NC} http://localhost:9696"
echo -e "  ${BLUE}Sonarr:${NC} http://localhost:8989"
echo -e "  ${BLUE}Radarr:${NC} http://localhost:7878"
echo -e "  ${BLUE}Lidarr:${NC} http://localhost:8686"
echo -e "  ${BLUE}Bazarr:${NC} http://localhost:6767"
echo -e "  ${BLUE}qBittorrent:${NC} http://localhost:8080 (admin/adminadmin)"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Access each service and complete initial setup"
echo "2. Run ./scripts/auto-update-configs.sh to sync TRaSH Guides settings"
echo "3. Configure your media folders and quality preferences"
echo "4. Add any premium indexers you have access to"