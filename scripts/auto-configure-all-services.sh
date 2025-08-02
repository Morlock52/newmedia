#!/bin/bash

# ============================================================================
# Ultimate Media Server Auto-Configuration Script v2025
# ============================================================================
# This script automatically detects and configures all running services
# including ARR apps, media servers, download clients, and monitoring
# ============================================================================

set -e

echo "🚀 Ultimate Media Server Auto-Configuration Script v2025"
echo "======================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
LOG_DIR="$PROJECT_ROOT/logs/auto-config"
COMPOSE_FILE="${1:-$PROJECT_ROOT/docker-compose-demo.yml}"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"/{prowlarr,sonarr,radarr,lidarr,bazarr,jellyfin,plex,jellyseerr,overseerr}
mkdir -p "$PROJECT_ROOT"/media/{movies,tv,music,books,downloads}
mkdir -p "$PROJECT_ROOT"/media/downloads/{complete,incomplete,torrents,usenet}

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_DIR/$(date +%Y%m%d)-configuration.log"
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=60
    local attempt=0
    
    log "${YELLOW}⏳ Waiting for $service_name to be ready on port $port...${NC}"
    while ! curl -s "http://localhost:$port" > /dev/null 2>&1; do
        if [ $attempt -eq $max_attempts ]; then
            log "${RED}❌ Failed to connect to $service_name after $max_attempts attempts${NC}"
            return 1
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    echo ""
    log "${GREEN}✅ $service_name is ready!${NC}"
    return 0
}

# Function to get API key from service config
get_api_key() {
    local service=$1
    local config_path=$2
    local api_key=""
    
    # Try multiple methods to get API key
    if docker ps --format "{{.Names}}" | grep -q "^${service}$"; then
        # Method 1: Extract from config.xml
        api_key=$(docker exec "$service" grep -oP '(?<=<ApiKey>)[^<]+' /config/config.xml 2>/dev/null || echo "")
        
        # Method 2: Extract from app.db (for some services)
        if [ -z "$api_key" ]; then
            api_key=$(docker exec "$service" sqlite3 /config/app.db "SELECT ApiKey FROM Config LIMIT 1;" 2>/dev/null || echo "")
        fi
        
        # Method 3: Generate if needed
        if [ -z "$api_key" ]; then
            api_key=$(openssl rand -hex 16)
            log "${YELLOW}⚠️  Generated API key for $service: $api_key${NC}"
        fi
    fi
    
    echo "$api_key"
}

# Function to detect running services
detect_running_services() {
    log "${BLUE}🔍 Detecting running services...${NC}"
    
    local services=()
    
    # Check for each service type
    declare -A service_ports=(
        ["prowlarr"]=9696
        ["sonarr"]=8989
        ["radarr"]=7878
        ["lidarr"]=8686
        ["bazarr"]=6767
        ["jellyfin"]=8096
        ["plex"]=32400
        ["jellyseerr"]=5055
        ["overseerr"]=5056
        ["qbittorrent"]=8090
        ["transmission"]=9091
        ["sabnzbd"]=8085
        ["homarr"]=7575
        ["homepage"]=3001
        ["prometheus"]=9090
        ["grafana"]=3000
        ["uptime-kuma"]=3004
        ["portainer"]=9000
        ["nginx-proxy-manager"]=8181
    )
    
    for service in "${!service_ports[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${service}$"; then
            services+=("$service:${service_ports[$service]}")
            log "${GREEN}✅ Found: $service on port ${service_ports[$service]}${NC}"
        fi
    done
    
    echo "${services[@]}"
}

# ============================================================================
# PROWLARR CONFIGURATION
# ============================================================================

configure_prowlarr() {
    log "${PURPLE}⚙️  Configuring Prowlarr...${NC}"
    
    local api_key=$(get_api_key "prowlarr" "/config")
    if [ -z "$api_key" ]; then
        log "${RED}❌ Failed to get Prowlarr API key${NC}"
        return 1
    fi
    
    # Add free indexers
    log "${BLUE}📚 Adding free indexers to Prowlarr...${NC}"
    
    # List of free public indexers with their configurations
    declare -A indexers=(
        ["1337x"]="torrent|public|Cardigann"
        ["The Pirate Bay"]="torrent|public|Cardigann"
        ["RARBG"]="torrent|public|Cardigann"
        ["YTS"]="torrent|public|Cardigann"
        ["EZTV"]="torrent|public|Cardigann"
        ["Nyaa"]="torrent|public|Cardigann"
        ["LimeTorrents"]="torrent|public|Cardigann"
        ["TorrentGalaxy"]="torrent|public|Cardigann"
    )
    
    for indexer_name in "${!indexers[@]}"; do
        IFS='|' read -r protocol privacy implementation <<< "${indexers[$indexer_name]}"
        
        curl -X POST "http://localhost:9696/api/v1/indexer" \
            -H "X-Api-Key: $api_key" \
            -H "Content-Type: application/json" \
            -d "{
                \"enable\": true,
                \"redirect\": false,
                \"supportsRss\": true,
                \"supportsSearch\": true,
                \"protocol\": \"$protocol\",
                \"privacy\": \"$privacy\",
                \"name\": \"$indexer_name\",
                \"fields\": [],
                \"implementationName\": \"$implementation\",
                \"implementation\": \"$implementation\",
                \"configContract\": \"${implementation}Settings\",
                \"tags\": []
            }" 2>/dev/null && log "${GREEN}✅ Added indexer: $indexer_name${NC}" || log "${YELLOW}⚠️  Skipped: $indexer_name (may already exist)${NC}"
    done
}

# ============================================================================
# ARR APPS CONNECTION TO PROWLARR
# ============================================================================

connect_arr_to_prowlarr() {
    log "${PURPLE}⚙️  Connecting ARR apps to Prowlarr...${NC}"
    
    local prowlarr_api_key=$(get_api_key "prowlarr" "/config")
    if [ -z "$prowlarr_api_key" ]; then
        log "${RED}❌ Failed to get Prowlarr API key${NC}"
        return 1
    fi
    
    # ARR applications to connect
    declare -A arr_apps=(
        ["sonarr"]="8989|Sonarr"
        ["radarr"]="7878|Radarr"
        ["lidarr"]="8686|Lidarr"
    )
    
    for app_name in "${!arr_apps[@]}"; do
        IFS='|' read -r port implementation <<< "${arr_apps[$app_name]}"
        
        if docker ps --format "{{.Names}}" | grep -q "^${app_name}$"; then
            local app_api_key=$(get_api_key "$app_name" "/config")
            
            if [ -n "$app_api_key" ]; then
                curl -X POST "http://localhost:9696/api/v1/applications" \
                    -H "X-Api-Key: $prowlarr_api_key" \
                    -H "Content-Type: application/json" \
                    -d "{
                        \"syncLevel\": \"fullSync\",
                        \"name\": \"$app_name\",
                        \"fields\": [
                            {\"name\": \"baseUrl\", \"value\": \"http://$app_name:$port\"},
                            {\"name\": \"apiKey\", \"value\": \"$app_api_key\"},
                            {\"name\": \"syncCategories\", \"value\": []}
                        ],
                        \"implementationName\": \"$implementation\",
                        \"implementation\": \"$implementation\",
                        \"configContract\": \"${implementation}Settings\",
                        \"tags\": []
                    }" 2>/dev/null && log "${GREEN}✅ Connected $app_name to Prowlarr${NC}" || log "${YELLOW}⚠️  $app_name may already be connected${NC}"
            fi
        fi
    done
}

# ============================================================================
# DOWNLOAD CLIENT CONFIGURATION
# ============================================================================

configure_download_clients() {
    log "${PURPLE}⚙️  Configuring download clients...${NC}"
    
    # Configure qBittorrent
    if docker ps --format "{{.Names}}" | grep -q "^qbittorrent$"; then
        log "${BLUE}🔧 Configuring qBittorrent...${NC}"
        
        # Create optimal qBittorrent configuration
        docker exec qbittorrent mkdir -p /config/qBittorrent
        docker exec qbittorrent bash -c 'cat > /config/qBittorrent/qBittorrent.conf << "EOF"
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
AutoRun\enabled=false
EOF'
        
        # Restart qBittorrent to apply settings
        docker restart qbittorrent
        sleep 5
        
        log "${GREEN}✅ qBittorrent configured (admin/adminadmin)${NC}"
    fi
    
    # Configure Transmission
    if docker ps --format "{{.Names}}" | grep -q "^transmission$"; then
        log "${BLUE}🔧 Configuring Transmission...${NC}"
        
        # Stop transmission to modify settings
        docker stop transmission
        
        # Create settings
        docker exec transmission bash -c 'cat > /config/settings.json << "EOF"
{
    "download-dir": "/downloads/complete",
    "incomplete-dir": "/downloads/incomplete",
    "incomplete-dir-enabled": true,
    "rpc-authentication-required": false,
    "rpc-bind-address": "0.0.0.0",
    "rpc-enabled": true,
    "rpc-host-whitelist": "",
    "rpc-host-whitelist-enabled": false,
    "rpc-port": 9091,
    "rpc-url": "/transmission/",
    "rpc-whitelist": "127.0.0.1,::1",
    "rpc-whitelist-enabled": false
}
EOF'
        
        # Start transmission
        docker start transmission
        sleep 5
        
        log "${GREEN}✅ Transmission configured${NC}"
    fi
}

# ============================================================================
# CONNECT ARR APPS TO DOWNLOAD CLIENTS
# ============================================================================

connect_arr_to_download_clients() {
    log "${PURPLE}⚙️  Connecting ARR apps to download clients...${NC}"
    
    # ARR applications
    declare -A arr_apps=(
        ["sonarr"]=8989
        ["radarr"]=7878
        ["lidarr"]=8686
    )
    
    for app_name in "${!arr_apps[@]}"; do
        port=${arr_apps[$app_name]}
        
        if docker ps --format "{{.Names}}" | grep -q "^${app_name}$"; then
            local api_key=$(get_api_key "$app_name" "/config")
            
            if [ -n "$api_key" ]; then
                # Add qBittorrent if running
                if docker ps --format "{{.Names}}" | grep -q "^qbittorrent$"; then
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
                            "tags": []
                        }' 2>/dev/null && log "${GREEN}✅ Connected $app_name to qBittorrent${NC}" || log "${YELLOW}⚠️  qBittorrent connection exists${NC}"
                fi
                
                # Add Transmission if running
                if docker ps --format "{{.Names}}" | grep -q "^transmission$"; then
                    curl -X POST "http://localhost:$port/api/v3/downloadclient" \
                        -H "X-Api-Key: $api_key" \
                        -H "Content-Type: application/json" \
                        -d '{
                            "enable": true,
                            "protocol": "torrent",
                            "priority": 2,
                            "name": "Transmission",
                            "fields": [
                                {"name": "host", "value": "transmission"},
                                {"name": "port", "value": 9091},
                                {"name": "urlBase", "value": "/transmission/"},
                                {"name": "username", "value": ""},
                                {"name": "password", "value": ""},
                                {"name": "category", "value": "'$app_name'"},
                                {"name": "recentTvPriority", "value": 0},
                                {"name": "olderTvPriority", "value": 0}
                            ],
                            "implementationName": "Transmission",
                            "implementation": "Transmission",
                            "configContract": "TransmissionSettings",
                            "tags": []
                        }' 2>/dev/null && log "${GREEN}✅ Connected $app_name to Transmission${NC}" || log "${YELLOW}⚠️  Transmission connection exists${NC}"
                fi
                
                # Add SABnzbd if running
                if docker ps --format "{{.Names}}" | grep -q "^sabnzbd$"; then
                    curl -X POST "http://localhost:$port/api/v3/downloadclient" \
                        -H "X-Api-Key: $api_key" \
                        -H "Content-Type: application/json" \
                        -d '{
                            "enable": true,
                            "protocol": "usenet",
                            "priority": 1,
                            "name": "SABnzbd",
                            "fields": [
                                {"name": "host", "value": "sabnzbd"},
                                {"name": "port", "value": 8080},
                                {"name": "apiKey", "value": ""},
                                {"name": "username", "value": ""},
                                {"name": "password", "value": ""},
                                {"name": "category", "value": "'$app_name'"},
                                {"name": "recentTvPriority", "value": 0},
                                {"name": "olderTvPriority", "value": 0}
                            ],
                            "implementationName": "Sabnzbd",
                            "implementation": "Sabnzbd",
                            "configContract": "SabnzbdSettings",
                            "tags": []
                        }' 2>/dev/null && log "${GREEN}✅ Connected $app_name to SABnzbd${NC}" || log "${YELLOW}⚠️  SABnzbd connection exists${NC}"
                fi
            fi
        fi
    done
}

# ============================================================================
# MEDIA SERVER CONFIGURATION
# ============================================================================

configure_media_servers() {
    log "${PURPLE}⚙️  Configuring media servers...${NC}"
    
    # Configure Jellyfin
    if docker ps --format "{{.Names}}" | grep -q "^jellyfin$"; then
        log "${BLUE}🎬 Configuring Jellyfin libraries...${NC}"
        
        # Wait for Jellyfin to be fully initialized
        wait_for_service "Jellyfin" 8096
        
        # Note: Jellyfin requires initial setup through web UI
        # We'll create the directory structure for libraries
        docker exec jellyfin mkdir -p /media/{movies,tv,music}
        
        log "${YELLOW}ℹ️  Jellyfin requires initial setup at http://localhost:8096${NC}"
        log "${YELLOW}   Add libraries: Movies (/media/movies), TV (/media/tv), Music (/media/music)${NC}"
    fi
    
    # Configure Plex
    if docker ps --format "{{.Names}}" | grep -q "^plex$"; then
        log "${BLUE}🎬 Configuring Plex libraries...${NC}"
        
        # Note: Plex requires claim token and initial setup
        log "${YELLOW}ℹ️  Plex requires initial setup at http://localhost:32400/web${NC}"
        log "${YELLOW}   Get claim token from https://www.plex.tv/claim${NC}"
        log "${YELLOW}   Add libraries: Movies (/media/movies), TV (/media/tv), Music (/media/music)${NC}"
    fi
}

# ============================================================================
# REQUEST SERVICE CONFIGURATION (JELLYSEERR/OVERSEERR)
# ============================================================================

configure_request_services() {
    log "${PURPLE}⚙️  Configuring request services...${NC}"
    
    # Configure Jellyseerr
    if docker ps --format "{{.Names}}" | grep -q "^jellyseerr$"; then
        log "${BLUE}📺 Setting up Jellyseerr...${NC}"
        
        wait_for_service "Jellyseerr" 5055
        
        # Jellyseerr auto-configuration would require API access after initial setup
        log "${YELLOW}ℹ️  Complete Jellyseerr setup at http://localhost:5055${NC}"
        log "${YELLOW}   1. Connect to Jellyfin (http://jellyfin:8096)${NC}"
        log "${YELLOW}   2. Connect to Radarr (http://radarr:7878) and Sonarr (http://sonarr:8989)${NC}"
    fi
    
    # Configure Overseerr
    if docker ps --format "{{.Names}}" | grep -q "^overseerr$"; then
        log "${BLUE}📺 Setting up Overseerr...${NC}"
        
        wait_for_service "Overseerr" 5056
        
        log "${YELLOW}ℹ️  Complete Overseerr setup at http://localhost:5056${NC}"
        log "${YELLOW}   1. Connect to Plex (http://plex:32400)${NC}"
        log "${YELLOW}   2. Connect to Radarr (http://radarr:7878) and Sonarr (http://sonarr:8989)${NC}"
    fi
}

# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

configure_monitoring() {
    log "${PURPLE}⚙️  Configuring monitoring services...${NC}"
    
    # Configure Prometheus
    if docker ps --format "{{.Names}}" | grep -q "^prometheus$"; then
        log "${BLUE}📊 Creating Prometheus configuration...${NC}"
        
        # Create Prometheus config if it doesn't exist
        if [ ! -f "$PROJECT_ROOT/prometheus.yml" ]; then
            cat > "$PROJECT_ROOT/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'docker'
    static_configs:
      - targets: ['host.docker.internal:9323']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # ARR services metrics
  - job_name: 'sonarr'
    static_configs:
      - targets: ['sonarr:8989']
  
  - job_name: 'radarr'
    static_configs:
      - targets: ['radarr:7878']
  
  - job_name: 'prowlarr'
    static_configs:
      - targets: ['prowlarr:9696']

  # Media servers
  - job_name: 'jellyfin'
    static_configs:
      - targets: ['jellyfin:8096']
EOF
            log "${GREEN}✅ Created Prometheus configuration${NC}"
        fi
    fi
    
    # Configure Grafana
    if docker ps --format "{{.Names}}" | grep -q "^grafana$"; then
        log "${BLUE}📊 Setting up Grafana dashboards...${NC}"
        
        wait_for_service "Grafana" 3000
        
        # Get Grafana credentials from environment or use defaults
        GRAFANA_USER="${GRAFANA_USER:-admin}"
        GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
        
        # Add Prometheus as data source
        curl -X POST "http://$GRAFANA_USER:$GRAFANA_PASSWORD@localhost:3000/api/datasources" \
            -H "Content-Type: application/json" \
            -d '{
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy",
                "isDefault": true,
                "jsonData": {
                    "httpMethod": "POST"
                }
            }' 2>/dev/null && log "${GREEN}✅ Added Prometheus data source to Grafana${NC}" || log "${YELLOW}⚠️  Prometheus data source may already exist${NC}"
        
        # Import media server dashboard
        curl -X POST "http://$GRAFANA_USER:$GRAFANA_PASSWORD@localhost:3000/api/dashboards/import" \
            -H "Content-Type: application/json" \
            -d '{
                "dashboard": {
                    "title": "Media Server Overview",
                    "panels": [],
                    "schemaVersion": 16,
                    "version": 0
                },
                "overwrite": true
            }' 2>/dev/null && log "${GREEN}✅ Created Media Server dashboard${NC}" || log "${YELLOW}⚠️  Dashboard creation skipped${NC}"
        
        log "${GREEN}✅ Grafana configured (http://localhost:3000 - $GRAFANA_USER/$GRAFANA_PASSWORD)${NC}"
    fi
    
    # Configure Uptime Kuma
    if docker ps --format "{{.Names}}" | grep -q "^uptime-kuma$"; then
        log "${BLUE}📊 Setting up Uptime Kuma monitors...${NC}"
        
        wait_for_service "Uptime Kuma" 3004
        
        log "${YELLOW}ℹ️  Complete Uptime Kuma setup at http://localhost:3004${NC}"
        log "${YELLOW}   Add monitors for all your services${NC}"
    fi
}

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

configure_dashboards() {
    log "${PURPLE}⚙️  Configuring dashboards...${NC}"
    
    # Configure Homarr
    if docker ps --format "{{.Names}}" | grep -q "^homarr$"; then
        log "${BLUE}🏠 Setting up Homarr dashboard...${NC}"
        
        wait_for_service "Homarr" 7575
        
        log "${GREEN}✅ Homarr is ready at http://localhost:7575${NC}"
        log "${YELLOW}   It will auto-detect Docker services${NC}"
    fi
    
    # Configure Homepage
    if docker ps --format "{{.Names}}" | grep -q "^homepage$"; then
        log "${BLUE}🏠 Configuring Homepage dashboard...${NC}"
        
        # Create Homepage configuration
        mkdir -p "$PROJECT_ROOT/homepage-configs"
        
        cat > "$PROJECT_ROOT/homepage-configs/services.yaml" << 'EOF'
---
# Media Management
- Media Management:
    - Sonarr:
        href: http://localhost:8989
        icon: sonarr.png
        description: TV Series Management
        widget:
          type: sonarr
          url: http://sonarr:8989
          key: {{SONARR_API_KEY}}
    - Radarr:
        href: http://localhost:7878
        icon: radarr.png
        description: Movie Management
        widget:
          type: radarr
          url: http://radarr:7878
          key: {{RADARR_API_KEY}}
    - Prowlarr:
        href: http://localhost:9696
        icon: prowlarr.png
        description: Indexer Management

# Media Servers
- Media Servers:
    - Jellyfin:
        href: http://localhost:8096
        icon: jellyfin.png
        description: Media Server
    - Plex:
        href: http://localhost:32400/web
        icon: plex.png
        description: Media Server

# Downloads
- Downloads:
    - qBittorrent:
        href: http://localhost:8090
        icon: qbittorrent.png
        description: Torrent Client
    - Transmission:
        href: http://localhost:9091
        icon: transmission.png
        description: Torrent Client

# Monitoring
- Monitoring:
    - Grafana:
        href: http://localhost:3000
        icon: grafana.png
        description: Metrics Dashboard
    - Uptime Kuma:
        href: http://localhost:3004
        icon: uptime-kuma.png
        description: Uptime Monitoring
EOF
        
        log "${GREEN}✅ Homepage configured at http://localhost:3001${NC}"
    fi
}

# ============================================================================
# CREATE MANAGEMENT SCRIPTS
# ============================================================================

create_management_scripts() {
    log "${PURPLE}📝 Creating management scripts...${NC}"
    
    # Create backup script
    cat > "$SCRIPT_DIR/backup-configs.sh" << 'EOF'
#!/bin/bash
# Backup all service configurations

BACKUP_DIR="./backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Backing up configurations to $BACKUP_DIR..."

# Backup all config directories
for service in sonarr radarr lidarr prowlarr bazarr jellyfin plex qbittorrent; do
    if [ -d "./${service}-config" ]; then
        echo "  - Backing up $service..."
        tar -czf "$BACKUP_DIR/${service}-config.tar.gz" "./${service}-config"
    fi
done

echo "✅ Backup complete!"
EOF
    chmod +x "$SCRIPT_DIR/backup-configs.sh"
    
    # Create health check script
    cat > "$SCRIPT_DIR/health-check.sh" << 'EOF'
#!/bin/bash
# Check health of all services

echo "🏥 Health Check Report"
echo "====================="

# Service health checks
declare -A services=(
    ["Sonarr"]=8989
    ["Radarr"]=7878
    ["Lidarr"]=8686
    ["Prowlarr"]=9696
    ["Jellyfin"]=8096
    ["Plex"]=32400
    ["qBittorrent"]=8090
    ["Transmission"]=9091
    ["Grafana"]=3000
    ["Prometheus"]=9090
)

for service in "${!services[@]}"; do
    port=${services[$service]}
    if curl -s "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ $service: HEALTHY"
    else
        echo "❌ $service: UNHEALTHY"
    fi
done

# Docker container status
echo ""
echo "🐳 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(sonarr|radarr|prowlarr|jellyfin|plex|qbittorrent)"
EOF
    chmod +x "$SCRIPT_DIR/health-check.sh"
    
    # Create update script
    cat > "$SCRIPT_DIR/update-all-services.sh" << 'EOF'
#!/bin/bash
# Update all services to latest versions

echo "🔄 Updating all services..."

# Pull latest images
docker-compose pull

# Recreate containers with new images
docker-compose up -d --force-recreate

# Clean up old images
docker image prune -f

echo "✅ Update complete!"
EOF
    chmod +x "$SCRIPT_DIR/update-all-services.sh"
    
    log "${GREEN}✅ Management scripts created${NC}"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log "${CYAN}🚀 Starting auto-configuration process...${NC}"
    log "${CYAN}Using compose file: $COMPOSE_FILE${NC}"
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log "${RED}❌ Docker is not running! Please start Docker first.${NC}"
        exit 1
    fi
    
    # Start services if not running
    log "${BLUE}🐳 Checking and starting services if needed...${NC}"
    
    # Get list of services from compose file
    COMPOSE_SERVICES=$(docker-compose -f "$COMPOSE_FILE" config --services 2>/dev/null || echo "")
    RUNNING_CONTAINERS=$(docker ps --format "{{.Names}}")
    SERVICES_TO_START=()
    
    # Check each service
    for service in $COMPOSE_SERVICES; do
        if echo "$RUNNING_CONTAINERS" | grep -q "^${service}$"; then
            log "${GREEN}✅ $service is already running${NC}"
        else
            log "${YELLOW}🚀 Will start $service${NC}"
            SERVICES_TO_START+=("$service")
        fi
    done
    
    # Only start services that aren't running
    if [ ${#SERVICES_TO_START[@]} -gt 0 ]; then
        log "${BLUE}📦 Starting services: ${SERVICES_TO_START[*]}...${NC}"
        docker-compose -f "$COMPOSE_FILE" up -d ${SERVICES_TO_START[*]} 2>&1 | grep -v "is already in use" || true
    else
        log "${GREEN}✅ All services are already running${NC}"
    fi
    
    # Wait for services to initialize
    log "${YELLOW}⏳ Waiting for services to initialize (30 seconds)...${NC}"
    sleep 30
    
    # Detect running services
    RUNNING_SERVICES=$(detect_running_services)
    
    # Execute configuration steps
    if docker ps --format "{{.Names}}" | grep -q "^prowlarr$"; then
        configure_prowlarr
        sleep 5
        connect_arr_to_prowlarr
    fi
    
    configure_download_clients
    sleep 5
    
    connect_arr_to_download_clients
    sleep 5
    
    configure_media_servers
    configure_request_services
    configure_monitoring
    configure_dashboards
    
    # Create management scripts
    create_management_scripts
    
    # Final report
    log ""
    log "${GREEN}✅ Auto-configuration complete!${NC}"
    log ""
    log "${CYAN}📋 Service Access URLs:${NC}"
    log "${CYAN}========================${NC}"
    
    # Display all service URLs
    declare -A service_urls=(
        ["Homarr Dashboard"]="http://localhost:7575"
        ["Homepage Dashboard"]="http://localhost:3001"
        ["Prowlarr"]="http://localhost:9696"
        ["Sonarr"]="http://localhost:8989"
        ["Radarr"]="http://localhost:7878"
        ["Lidarr"]="http://localhost:8686"
        ["Bazarr"]="http://localhost:6767"
        ["Jellyfin"]="http://localhost:8096"
        ["Plex"]="http://localhost:32400/web"
        ["Jellyseerr"]="http://localhost:5055"
        ["Overseerr"]="http://localhost:5056"
        ["qBittorrent"]="http://localhost:8090 (admin/adminadmin)"
        ["Transmission"]="http://localhost:9091"
        ["SABnzbd"]="http://localhost:8085"
        ["Grafana"]="http://localhost:3000 (admin/admin)"
        ["Prometheus"]="http://localhost:9090"
        ["Uptime Kuma"]="http://localhost:3004"
        ["Portainer"]="http://localhost:9000"
        ["Nginx Proxy Manager"]="http://localhost:8181"
    )
    
    for service in "${!service_urls[@]}"; do
        log "  ${BLUE}$service:${NC} ${service_urls[$service]}"
    done
    
    log ""
    log "${YELLOW}📝 Next Steps:${NC}"
    log "  1. Complete initial setup for media servers (Jellyfin/Plex)"
    log "  2. Configure request services (Jellyseerr/Overseerr)"
    log "  3. Add any premium indexers to Prowlarr"
    log "  4. Set up media library paths in ARR apps"
    log "  5. Configure quality profiles to your preference"
    log ""
    log "${CYAN}🛠️  Management Scripts:${NC}"
    log "  - ${GREEN}./scripts/health-check.sh${NC} - Check service health"
    log "  - ${GREEN}./scripts/backup-configs.sh${NC} - Backup configurations"
    log "  - ${GREEN}./scripts/update-all-services.sh${NC} - Update all services"
    log ""
    log "${PURPLE}📊 Monitoring:${NC}"
    log "  - View metrics at ${BLUE}http://localhost:3000${NC} (Grafana)"
    log "  - Check uptime at ${BLUE}http://localhost:3004${NC} (Uptime Kuma)"
    log "  - Manage containers at ${BLUE}http://localhost:9000${NC} (Portainer)"
    log ""
    log "${GREEN}🎉 Happy streaming!${NC}"
}

# Run main function
main "$@"