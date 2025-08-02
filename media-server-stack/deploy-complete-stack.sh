#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Complete Media Server Stack Deployment"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    echo -e "${RED}❌ .env file not found${NC}"
    exit 1
fi

echo -e "${BLUE}Domain:${NC} $DOMAIN"
echo -e "${BLUE}Deployment:${NC} Complete Media Stack + Monitoring"
echo ""

# Function to show progress
show_progress() {
    local step="$1"
    local total="$2"
    local description="$3"
    
    echo -e "${BLUE}[$step/$total] $description${NC}"
}

# Step 1: Prerequisites
show_progress "1" "8" "Checking prerequisites..."

if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker not installed${NC}"
    exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose not installed${NC}"
    exit 1
fi

echo "✅ Docker and Docker Compose available"

# Step 2: Setup directories
show_progress "2" "8" "Setting up directory structure..."
chmod +x setup-directories.sh
./setup-directories.sh

# Step 3: Generate secrets
show_progress "3" "8" "Generating secrets..."
chmod +x generate-secrets.sh
./generate-secrets.sh

# Step 4: Create networks
show_progress "4" "8" "Creating Docker networks..."
docker network create traefik_network 2>/dev/null || echo "traefik_network already exists"
docker network create monitoring_network 2>/dev/null || echo "monitoring_network already exists"

# Step 5: Stop existing containers
show_progress "5" "8" "Stopping existing containers..."
docker-compose down 2>/dev/null || echo "No existing containers to stop"

# Step 6: Deploy core stack
show_progress "6" "8" "Deploying complete media stack..."
echo "Starting all services (this may take a few minutes)..."

# Use the complete compose file
docker-compose -f docker-compose-complete.yml up -d

echo "Waiting for containers to start..."
sleep 30

# Step 7: Deploy monitoring
show_progress "7" "8" "Deploying monitoring stack..."
if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
    echo "Starting monitoring services..."
    docker-compose -f compose/compose.monitoring.yml up -d
    echo "Monitoring stack deployed"
else
    echo "Monitoring deployment skipped (set DEPLOY_MONITORING=true to enable)"
fi

# Step 8: Verify deployment
show_progress "8" "8" "Verifying deployment..."

echo ""
echo "🔍 Checking service status..."

# Define all services
core_services=("jellyfin" "sonarr" "radarr" "prowlarr" "overseerr" "traefik" "cloudflared")
additional_services=("lidarr" "readarr" "bazarr" "tautulli" "mylar" "podgrab" "youtube-dl-material" "photoprism")
download_services=("gluetun" "qbittorrent")

check_service() {
    local service="$1"
    if docker ps | grep -q "$service"; then
        echo "  ✅ $service"
        return 0
    else
        echo "  ❌ $service"
        return 1
    fi
}

echo ""
echo "Core Services:"
core_working=0
for service in "${core_services[@]}"; do
    if check_service "$service"; then
        ((core_working++))
    fi
done

echo ""
echo "Additional Media Services:"
additional_working=0
for service in "${additional_services[@]}"; do
    if check_service "$service"; then
        ((additional_working++))
    fi
done

echo ""
echo "Download Services:"
download_working=0
for service in "${download_services[@]}"; do
    if check_service "$service"; then
        ((download_working++))
    fi
done

# Check monitoring if enabled
monitoring_working=0
if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
    echo ""
    echo "Monitoring Services:"
    monitoring_services=("prometheus" "grafana" "alertmanager" "node-exporter" "cadvisor")
    for service in "${monitoring_services[@]}"; do
        if check_service "$service"; then
            ((monitoring_working++))
        fi
    done
fi

echo ""
echo "🎯 Deployment Summary:"
echo "======================"
echo "Core Services: $core_working/${#core_services[@]} running"
echo "Additional Services: $additional_working/${#additional_services[@]} running"
echo "Download Services: $download_working/${#download_services[@]} running"

if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
    echo "Monitoring: $monitoring_working/5 running"
fi

# Show service URLs
echo ""
echo -e "${GREEN}🌐 Your Media Server Services:${NC}"
echo "============================="

echo ""
echo -e "${BLUE}📺 Core Media Management:${NC}"
echo "• Jellyfin (Media Server): https://jellyfin.$DOMAIN"
echo "• Overseerr (Requests): https://overseerr.$DOMAIN"
echo "• Sonarr (TV): https://sonarr.$DOMAIN"
echo "• Radarr (Movies): https://radarr.$DOMAIN"
echo "• Prowlarr (Indexers): https://prowlarr.$DOMAIN"

echo ""
echo -e "${BLUE}📚 Additional Media:${NC}"
echo "• Lidarr (Music): https://lidarr.$DOMAIN"
echo "• Readarr (Books): https://readarr.$DOMAIN"
echo "• Bazarr (Subtitles): https://bazarr.$DOMAIN"
echo "• Mylar (Comics): https://mylar.$DOMAIN"
echo "• Podgrab (Podcasts): https://podgrab.$DOMAIN"
echo "• YouTube-DL: https://youtube-dl.$DOMAIN"

echo ""
echo -e "${BLUE}📊 Analytics & Photos:${NC}"
echo "• Tautulli (Jellyfin Stats): https://tautulli.$DOMAIN"
echo "• PhotoPrism (Photos): https://photoprism.$DOMAIN"

if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
    echo ""
    echo -e "${BLUE}📈 Monitoring:${NC}"
    echo "• Grafana (Dashboards): https://grafana.$DOMAIN"
    echo "• Prometheus (Metrics): https://prometheus.$DOMAIN"
    echo "• AlertManager (Alerts): https://alertmanager.$DOMAIN"
fi

echo ""
echo -e "${BLUE}🔧 Management:${NC}"
echo "• Traefik (Reverse Proxy): https://traefik.$DOMAIN"

# VPN Status
echo ""
echo -e "${BLUE}🔒 Security Status:${NC}"
if docker ps | grep -q gluetun; then
    echo "• VPN (Gluetun): ✅ Running"
    if docker logs gluetun --tail 5 | grep -q "You are protected" 2>/dev/null; then
        echo "• VPN Connection: ✅ Connected and protecting traffic"
    else
        echo "• VPN Connection: ⚠️  Check connection (run: docker logs gluetun)"
    fi
else
    echo "• VPN (Gluetun): ❌ Not running"
fi

if docker ps | grep -q cloudflared; then
    echo "• Cloudflare Tunnel: ✅ Running"
else
    echo "• Cloudflare Tunnel: ❌ Not running"
fi

# Check for any failed containers
echo ""
failed_containers=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | grep -E "(jellyfin|sonarr|radarr|lidarr|readarr|bazarr|prowlarr|overseerr|tautulli|mylar|podgrab|youtube-dl|photoprism|gluetun|qbittorrent|traefik|cloudflared)" || true)

if [[ -n "$failed_containers" ]]; then
    echo -e "${YELLOW}⚠️  Some containers failed to start:${NC}"
    echo "$failed_containers"
    echo ""
    echo "Check logs with: docker logs [container-name]"
    echo "Restart with: docker-compose restart [service-name]"
fi

# Show next steps
echo ""
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "======================="

echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Configure Cloudflare Zero Trust authentication"
echo "2. Set up indexers in Prowlarr"
echo "3. Configure download clients in Sonarr/Radarr"
echo "4. Add media libraries in Jellyfin"
echo "5. Set up monitoring dashboards in Grafana"

echo ""
echo -e "${BLUE}🛠️  Quick Commands:${NC}"
echo "• View all containers: docker ps"
echo "• Check logs: docker logs [container-name]"
echo "• Restart service: docker-compose restart [service-name]"
echo "• Stop everything: docker-compose -f docker-compose-complete.yml down"
echo "• Update containers: docker-compose pull && docker-compose up -d"

echo ""
echo -e "${BLUE}🔐 Authentication:${NC}"
echo "All services are protected by Cloudflare Zero Trust"
echo "Login with: $EMAIL"
echo "You'll receive PIN codes via email for each service"

echo ""
echo -e "${BLUE}📱 Test Your Setup:${NC}"
echo "Run: ./test-complete-stack.sh"

if [[ $core_working -eq ${#core_services[@]} ]] && docker ps | grep -q cloudflared; then
    echo ""
    echo -e "${GREEN}🚀 Your complete media server is ready!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Some services need attention. Check the status above.${NC}"
fi
