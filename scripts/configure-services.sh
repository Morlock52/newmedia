#!/bin/bash

# Media Server Automated Configuration Script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Media Server Configuration Assistant${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}" | grep -q "200\|301\|302"; then
        echo -e "${GREEN}✅ ${service} is accessible at http://localhost:${port}${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  ${service} at port ${port} needs attention${NC}"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${BLUE}Waiting for ${service} to be ready...${NC}"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}" | grep -q "200\|301\|302"; then
            echo -e "${GREEN}✅ ${service} is ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e "${RED}❌ ${service} failed to start${NC}"
    return 1
}

# Check all services
echo -e "\n${BLUE}Checking Service Status...${NC}"
check_service "Jellyfin" 8096
check_service "Sonarr" 8989
check_service "Radarr" 7878
check_service "Prowlarr" 9696
check_service "qBittorrent" 8080

# Create necessary directories
echo -e "\n${BLUE}Creating media directories...${NC}"
mkdir -p "${PROJECT_ROOT}/media"/{tv,movies,music,books}
mkdir -p "${PROJECT_ROOT}/downloads"/{complete,incomplete,torrents}
echo -e "${GREEN}✅ Directories created${NC}"

# Extract API keys if available
echo -e "\n${BLUE}Checking for API Keys...${NC}"

# Try to get Sonarr API key
if docker exec sonarr test -f /config/config.xml 2>/dev/null; then
    SONARR_API=$(docker exec sonarr sed -n 's/.*<ApiKey>\(.*\)<\/ApiKey>.*/\1/p' /config/config.xml 2>/dev/null || echo "")
    if [ -n "$SONARR_API" ]; then
        echo -e "${GREEN}✅ Sonarr API Key: ${SONARR_API}${NC}"
    else
        echo -e "${YELLOW}⚠️  Sonarr API key not found - complete setup at http://localhost:8989${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Sonarr not configured yet - visit http://localhost:8989${NC}"
fi

# Try to get Radarr API key
if docker exec radarr test -f /config/config.xml 2>/dev/null; then
    RADARR_API=$(docker exec radarr sed -n 's/.*<ApiKey>\(.*\)<\/ApiKey>.*/\1/p' /config/config.xml 2>/dev/null || echo "")
    if [ -n "$RADARR_API" ]; then
        echo -e "${GREEN}✅ Radarr API Key: ${RADARR_API}${NC}"
    else
        echo -e "${YELLOW}⚠️  Radarr API key not found - complete setup at http://localhost:7878${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Radarr not configured yet - visit http://localhost:7878${NC}"
fi

# Try to get Prowlarr API key
if docker exec prowlarr test -f /config/config.xml 2>/dev/null; then
    PROWLARR_API=$(docker exec prowlarr sed -n 's/.*<ApiKey>\(.*\)<\/ApiKey>.*/\1/p' /config/config.xml 2>/dev/null || echo "")
    if [ -n "$PROWLARR_API" ]; then
        echo -e "${GREEN}✅ Prowlarr API Key: ${PROWLARR_API}${NC}"
    else
        echo -e "${YELLOW}⚠️  Prowlarr API key not found - complete setup at http://localhost:9696${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Prowlarr not configured yet - visit http://localhost:9696${NC}"
fi

# Save configuration to file
CONFIG_FILE="${PROJECT_ROOT}/.media-server-config"
echo -e "\n${BLUE}Saving configuration...${NC}"
cat > "$CONFIG_FILE" << EOF
# Media Server Configuration
# Generated: $(date)

# Service URLs
JELLYFIN_URL=http://localhost:8096
SONARR_URL=http://localhost:8989
RADARR_URL=http://localhost:7878
PROWLARR_URL=http://localhost:9696
QBITTORRENT_URL=http://localhost:8080

# API Keys (update after initial setup)
SONARR_API_KEY=${SONARR_API:-"PENDING_SETUP"}
RADARR_API_KEY=${RADARR_API:-"PENDING_SETUP"}
PROWLARR_API_KEY=${PROWLARR_API:-"PENDING_SETUP"}
JELLYFIN_API_KEY=PENDING_SETUP

# Paths
MEDIA_ROOT=${PROJECT_ROOT}/media
DOWNLOADS_ROOT=${PROJECT_ROOT}/downloads

# Docker Network
DOCKER_NETWORK=media-net
EOF

echo -e "${GREEN}✅ Configuration saved to ${CONFIG_FILE}${NC}"

# Generate quick access HTML dashboard
echo -e "\n${BLUE}Generating Quick Access Dashboard...${NC}"
cat > "${PROJECT_ROOT}/media-dashboard.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Media Server Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .services-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        .service-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .service-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .service-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .service-icon {
            width: 40px;
            height: 40px;
            margin-right: 1rem;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        .media-icon { background: #ff6b6b; }
        .arr-icon { background: #4ecdc4; }
        .download-icon { background: #45b7d1; }
        .monitor-icon { background: #96ceb4; }
        
        .service-name {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
        }
        .service-status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }
        .status-ready { background: #d4edda; color: #155724; }
        .status-setup { background: #fff3cd; color: #856404; }
        
        .service-link {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: opacity 0.3s;
        }
        .service-link:hover {
            opacity: 0.9;
        }
        .setup-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .setup-step {
            margin: 1rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .step-number {
            display: inline-block;
            width: 30px;
            height: 30px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 30px;
            margin-right: 1rem;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Media Server Dashboard</h1>
        
        <div class="services-grid">
            <!-- Media Servers -->
            <div class="service-card">
                <div class="service-header">
                    <div class="service-icon media-icon">🎬</div>
                    <div class="service-name">Jellyfin</div>
                </div>
                <div class="service-status status-setup">Needs Setup</div>
                <a href="http://localhost:8096" target="_blank" class="service-link">Open Jellyfin</a>
            </div>
            
            <!-- Arr Services -->
            <div class="service-card">
                <div class="service-header">
                    <div class="service-icon arr-icon">📺</div>
                    <div class="service-name">Sonarr</div>
                </div>
                <div class="service-status status-ready">Ready</div>
                <a href="http://localhost:8989" target="_blank" class="service-link">Open Sonarr</a>
            </div>
            
            <div class="service-card">
                <div class="service-header">
                    <div class="service-icon arr-icon">🎥</div>
                    <div class="service-name">Radarr</div>
                </div>
                <div class="service-status status-ready">Ready</div>
                <a href="http://localhost:7878" target="_blank" class="service-link">Open Radarr</a>
            </div>
            
            <div class="service-card">
                <div class="service-header">
                    <div class="service-icon arr-icon">🔍</div>
                    <div class="service-name">Prowlarr</div>
                </div>
                <div class="service-status status-ready">Ready</div>
                <a href="http://localhost:9696" target="_blank" class="service-link">Open Prowlarr</a>
            </div>
            
            <!-- Download Clients -->
            <div class="service-card">
                <div class="service-header">
                    <div class="service-icon download-icon">⬇️</div>
                    <div class="service-name">qBittorrent</div>
                </div>
                <div class="service-status status-ready">Ready</div>
                <a href="http://localhost:8080" target="_blank" class="service-link">Open qBittorrent</a>
            </div>
        </div>
        
        <div class="setup-section">
            <h2 style="margin-bottom: 1.5rem; color: #333;">⚡ Quick Setup Guide</h2>
            
            <div class="setup-step">
                <span class="step-number">1</span>
                <strong>Jellyfin:</strong> Complete setup wizard, create admin user (admin/admin123)
            </div>
            
            <div class="setup-step">
                <span class="step-number">2</span>
                <strong>Prowlarr:</strong> Add indexers (1337x, TPB, etc.), then sync to Sonarr/Radarr
            </div>
            
            <div class="setup-step">
                <span class="step-number">3</span>
                <strong>Sonarr:</strong> Set root folder to /media/tv, add qBittorrent as download client
            </div>
            
            <div class="setup-step">
                <span class="step-number">4</span>
                <strong>Radarr:</strong> Set root folder to /media/movies, add qBittorrent as download client
            </div>
            
            <div class="setup-step">
                <span class="step-number">5</span>
                <strong>qBittorrent:</strong> Login with admin/adminadmin, configure download paths
            </div>
            
            <div class="setup-step" style="background: #d4edda; border-left-color: #28a745;">
                <span class="step-number" style="background: #28a745;">✓</span>
                <strong>Test:</strong> Search for content in Sonarr/Radarr and verify download workflow
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh status every 30 seconds
        setInterval(() => {
            console.log('Checking service status...');
            // In production, this would check actual service status
        }, 30000);
    </script>
</body>
</html>
EOF

echo -e "${GREEN}✅ Dashboard created: ${PROJECT_ROOT}/media-dashboard.html${NC}"

echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}      Configuration Helper Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Open the dashboard: ${BLUE}open ${PROJECT_ROOT}/media-dashboard.html${NC}"
echo -e "2. Complete initial setup for each service"
echo -e "3. Run this script again to update API keys"
echo -e "\n${GREEN}Configuration saved to: ${CONFIG_FILE}${NC}"