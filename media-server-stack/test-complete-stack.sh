#!/usr/bin/env bash
set -euo pipefail

echo "🎬 Complete Media Server Stack Testing Suite"
echo "============================================"
echo "Testing ALL applications in the morloksmaze.com media server"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    echo -e "${RED}❌ .env file not found${NC}"
    exit 1
fi

echo -e "${BLUE}Domain:${NC} $DOMAIN"
echo -e "${BLUE}Email:${NC} $EMAIL"
echo ""

# Define ALL services in the media server stack
declare -A core_services=(
    ["jellyfin"]="8096:Media Server:Stream movies, TV shows, music"
    ["sonarr"]="8989:TV Management:Automatically download TV shows"
    ["radarr"]="7878:Movie Management:Automatically download movies"
    ["prowlarr"]="9696:Indexer Management:Manage torrent/usenet indexers"
    ["overseerr"]="5055:Request Management:User-friendly request system"
    ["traefik"]="8080:Reverse Proxy:Route and secure all services"
)

declare -A download_services=(
    ["qbittorrent"]="8080:Torrent Client:Download torrents via VPN"
    ["gluetun"]="8000:VPN Gateway:Secure VPN tunnel for downloads"
)

declare -A additional_media_services=(
    ["lidarr"]="8686:Music Management:Automatically download music"
    ["readarr"]="8787:Book Management:Automatically download ebooks"
    ["bazarr"]="6767:Subtitle Management:Download subtitles for movies/TV"
    ["tautulli"]="8181:Jellyfin Analytics:Monitor Jellyfin usage and stats"
    ["mylar"]="8090:Comic Management:Automatically download comics"
    ["podgrab"]="8080:Podcast Manager:Download and manage podcasts"
    ["youtube-dl-material"]="17442:YouTube Downloader:Download videos from YouTube and other sites"
    ["photoprism"]="2342:Photo Management:AI-powered photo organization"
)

declare -A monitoring_services=(
    ["prometheus"]="9090:Metrics Collection:Collect system and service metrics"
    ["grafana"]="3000:Dashboards:Visualize metrics and create dashboards"
    ["alertmanager"]="9093:Alert Management:Send notifications for issues"
    ["node-exporter"]="9100:System Metrics:Export system metrics to Prometheus"
    ["cadvisor"]="8080:Container Metrics:Monitor Docker container performance"
    ["watchtower"]="8080:Auto Updates:Automatically update containers"
)

declare -A utility_services=(
    ["cloudflared"]="N/A:Cloudflare Tunnel:Secure external access"
    ["test-web"]="80:Test Service:Simple test web service"
    ["security-headers"]="N/A:Security Middleware:Add security headers"
)

# Utility functions
print_section() {
    local title="$1"
    local color="$2"
    echo ""
    echo -e "${color}${title}${NC}"
    printf '=%.0s' {1..50}
    echo ""
}

check_container_status() {
    local service="$1"
    
    if docker ps -q -f name="^${service}$" | grep -q .; then
        echo "✅ Running"
        return 0
    elif docker ps -a -q -f name="^${service}$" | grep -q .; then
        echo "🔶 Stopped"
        return 1
    else
        echo "❌ Not created"
        return 2
    fi
}

check_service_health() {
    local service="$1"
    local port="$2"
    
    if [[ "$port" == "N/A" ]]; then
        echo "N/A"
        return 0
    fi
    
    # For qBittorrent, check via gluetun network
    if [[ "$service" == "qbittorrent" ]]; then
        local gluetun_ip
        gluetun_ip=$(docker inspect gluetun 2>/dev/null | jq -r '.[0].NetworkSettings.Networks.download_network.IPAddress // "not_found"' 2>/dev/null || echo "not_found")
        if [[ "$gluetun_ip" != "not_found" && "$gluetun_ip" != "null" ]]; then
            local http_code
            http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://$gluetun_ip:$port" 2>/dev/null || echo "000")
            case $http_code in
                200|302|401) echo "✅ Healthy ($http_code)" ;;
                000) echo "❌ No response" ;;
                *) echo "⚠️  Response: $http_code" ;;
            esac
        else
            echo "❌ Gluetun network issue"
        fi
        return
    fi
    
    # Standard localhost check for other services
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" 2>/dev/null || echo "000")
    
    case $http_code in
        200|302|401) echo "✅ Healthy ($http_code)" ;;
        000) echo "❌ No response" ;;
        *) echo "⚠️  Response: $http_code" ;;
    esac
}

check_external_access() {
    local service="$1"
    local url="https://$service.$DOMAIN"
    
    echo -n "  External: "
    
    local response
    response=$(curl -s -I "$url" 2>/dev/null || echo "FAILED")
    
    if echo "$response" | grep -qi "cloudflare"; then
        echo "🔐 Protected"
    elif echo "$response" | grep -q "HTTP/[12].[01] [23]"; then
        echo "📡 Accessible"
    else
        echo "❌ No response"
    fi
}

test_service_group() {
    local title="$1"
    local color="$2"
    local -n services_ref=$3
    
    print_section "$title" "$color"
    
    for service_key in "${!services_ref[@]}"; do
        IFS=':' read -r port description details <<< "${services_ref[$service_key]}"
        
        echo -e "${CYAN}${service_key}${NC} - $description"
        echo "  $details"
        
        echo -n "  Container: "
        check_container_status "$service_key"
        
        echo -n "  Health: "
        check_service_health "$service_key" "$port"
        
        # Only check external access for services that should have web interfaces
        if [[ "$port" != "N/A" && "$service_key" != "node-exporter" && "$service_key" != "cadvisor" ]]; then
            check_external_access "$service_key"
        fi
        
        echo ""
    done
}

# Main testing sequence
echo -e "${GREEN}🔍 System Overview${NC}"
echo "=================="

echo -n "Docker daemon: "
if docker info >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not running"
    exit 1
fi

echo -n "Total containers: "
total_containers=$(docker ps -a | grep -c "media-server-stack" 2>/dev/null || echo "0")
running_containers=$(docker ps | grep -c "media-server-stack" 2>/dev/null || echo "0")
echo "$running_containers/$total_containers running"

echo -n "Internet connectivity: "
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo "✅ Online"
else
    echo "❌ Offline"
fi

# Test service groups
test_service_group "🎬 Core Media Services" "$GREEN" core_services
test_service_group "⬇️  Download Services" "$YELLOW" download_services  
test_service_group "📚 Additional Media Services" "$BLUE" additional_media_services
test_service_group "📊 Monitoring & Analytics" "$PURPLE" monitoring_services
test_service_group "🔧 Utility Services" "$CYAN" utility_services

# VPN Status Check
print_section "🔒 VPN & Security Status" "$RED"

echo -n "Gluetun VPN: "
if docker ps | grep -q gluetun; then
    echo "✅ Running"
    
    # Check VPN connection
    if docker logs gluetun --tail 10 2>/dev/null | grep -q "You are protected"; then
        echo "  🛡️  VPN connected and protecting traffic"
    else
        echo "  ⚠️  VPN may not be connected - check logs"
    fi
    
    # Get public IP through VPN
    echo -n "  Public IP: "
    vpn_ip=$(docker exec gluetun wget -qO- http://checkip.amazonaws.com 2>/dev/null || echo "Unable to check")
    echo "$vpn_ip"
else
    echo "❌ Not running"
fi

echo -n "Cloudflare Tunnel: "
if docker ps | grep -q cloudflared; then
    echo "✅ Running"
    
    if docker logs cloudflared --tail 10 | grep -q "Registered tunnel connection"; then
        echo "  🌐 Tunnel connected to Cloudflare"
    else
        echo "  ⚠️  Tunnel connection issue"
    fi
else
    echo "❌ Not running"
fi

# DNS Records Check
print_section "🌐 DNS Configuration" "$BLUE"

all_services=(
    "${!core_services[@]}"
    "${!download_services[@]}" 
    "${!additional_media_services[@]}"
    "${!monitoring_services[@]}"
)

# Remove qbittorrent since it doesn't have its own subdomain
all_services=("${all_services[@]/qbittorrent}")

echo "Checking DNS records for all services..."
dns_working=0
dns_total=${#all_services[@]}

for service in "${all_services[@]}"; do
    echo -n "$service.$DOMAIN: "
    if nslookup "$service.$DOMAIN" >/dev/null 2>&1; then
        echo "✅ Resolves"
        ((dns_working++))
    else
        echo "❌ No DNS record"
    fi
done

echo ""
echo "DNS Status: $dns_working/$dns_total records configured"

# Storage Check
print_section "💾 Storage & Data" "$YELLOW"

echo "Checking data directories..."
data_dirs=("data/media/movies" "data/media/tv" "data/media/music" "data/torrents" "data/usenet")

for dir in "${data_dirs[@]}"; do
    echo -n "$dir: "
    if [[ -d "$dir" ]]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "Unknown")
        files=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ' || echo "0")
        echo "✅ Exists ($size, $files files)"
    else
        echo "❌ Missing"
    fi
done

# Port Conflicts
print_section "🚨 Port Conflict Analysis" "$RED"

critical_ports=("80" "443" "8080" "8096" "8989" "7878" "9696" "5055")
conflicts_found=0

for port in "${critical_ports[@]}"; do
    if lsof -i ":$port" >/dev/null 2>&1; then
        process=$(lsof -i ":$port" 2>/dev/null | tail -n +2 | head -n 1 | awk '{print $1}')
        if [[ "$process" =~ ^(docker|com\.docker) ]]; then
            echo "Port $port: ✅ Used by Docker"
        else
            echo "Port $port: ⚠️  Conflict with $process"
            ((conflicts_found++))
        fi
    else
        echo "Port $port: 🔶 Available"
    fi
done

if [[ $conflicts_found -gt 0 ]]; then
    echo ""
    echo "⚠️  $conflicts_found port conflicts found - some services may not be accessible"
fi

# Interactive Testing
print_section "🧪 Interactive Authentication Testing" "$GREEN"

read -p "Test Cloudflare authentication for core services? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Opening each core service for authentication testing..."
    echo "You should see Cloudflare Access login → enter $EMAIL → get PIN → access service"
    echo ""
    
    for service in "${!core_services[@]}"; do
        IFS=':' read -r port description details <<< "${core_services[$service]}"
        url="https://$service.$DOMAIN"
        
        echo -e "${PURPLE}Testing: $description${NC}"
        echo "URL: $url"
        
        osascript -e "tell application \"Google Chrome\" to open location \"$url\""
        
        read -p "Does $description load correctly with authentication? (y/n): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}✅ $description working${NC}"
        else
            echo -e "${RED}❌ $description needs attention${NC}"
        fi
        echo ""
    done
fi

# Generate Summary Report
print_section "📋 Final Summary Report" "$GREEN"

# Count working services
working_core=0
total_core=${#core_services[@]}

for service in "${!core_services[@]}"; do
    if docker ps | grep -q "$service"; then
        ((working_core++))
    fi
done

working_additional=0
total_additional=${#additional_media_services[@]}

for service in "${!additional_media_services[@]}"; do
    if docker ps | grep -q "$service"; then
        ((working_additional++))
    fi
done

working_monitoring=0
total_monitoring=${#monitoring_services[@]}

for service in "${!monitoring_services[@]}"; do
    if docker ps | grep -q "$service"; then
        ((working_monitoring++))
    fi
done

echo "🎬 Core Media Services: $working_core/$total_core running"
echo "📚 Additional Services: $working_additional/$total_additional running"  
echo "📊 Monitoring Services: $working_monitoring/$total_monitoring running"
echo "🌐 DNS Records: $dns_working/$dns_total configured"

if docker ps | grep -q cloudflared; then
    echo "🔗 Cloudflare Tunnel: ✅ Active"
else
    echo "🔗 Cloudflare Tunnel: ❌ Inactive"
fi

if docker ps | grep -q gluetun; then
    echo "🔒 VPN Protection: ✅ Active"
else
    echo "🔒 VPN Protection: ❌ Inactive"
fi

echo ""
echo -e "${BLUE}📱 Your Complete Media Server:${NC}"
echo "============================="

echo ""
echo -e "${GREEN}Core Services:${NC}"
for service in "${!core_services[@]}"; do
    IFS=':' read -r port description details <<< "${core_services[$service]}"
    status="❌"
    if docker ps | grep -q "$service"; then status="✅"; fi
    echo "  $status $description: https://$service.$DOMAIN"
done

echo ""
echo -e "${BLUE}Additional Media:${NC}"
for service in "${!additional_media_services[@]}"; do
    IFS=':' read -r port description details <<< "${additional_media_services[$service]}"
    status="❌"
    if docker ps | grep -q "$service"; then status="✅"; fi
    echo "  $status $description: https://$service.$DOMAIN"
done

echo ""
echo -e "${PURPLE}Monitoring:${NC}"
for service in "${!monitoring_services[@]}"; do
    IFS=':' read -r port description details <<< "${monitoring_services[$service]}"
    status="❌"
    if docker ps | grep -q "$service"; then status="✅"; fi
    if [[ "$service" != "node-exporter" && "$service" != "cadvisor" && "$service" != "watchtower" ]]; then
        echo "  $status $description: https://$service.$DOMAIN"
    fi
done

echo ""
echo -e "${BLUE}🛠️  Management Commands:${NC}"
echo "• Start all: docker-compose up -d"
echo "• Start with monitoring: docker-compose -f docker-compose.yml -f compose/compose.monitoring.yml up -d"
echo "• View logs: docker logs [service-name]"
echo "• Restart service: docker-compose restart [service-name]"
echo "• Stop all: docker-compose down"

echo ""
echo -e "${BLUE}📚 Available Services to Deploy:${NC}"
echo "The following services are configured but may not be running:"

not_running=()
for service in "${!additional_media_services[@]}"; do
    if ! docker ps | grep -q "$service"; then
        not_running+=("$service")
    fi
done

if [[ ${#not_running[@]} -gt 0 ]]; then
    echo ""
    for service in "${not_running[@]}"; do
        IFS=':' read -r port description details <<< "${additional_media_services[$service]}"
        echo "• $description ($service) - $details"
    done
    echo ""
    echo "To add these services, they need to be added to your docker-compose.yml"
else
    echo "All configured services are running!"
fi

echo ""
if [[ $working_core -eq $total_core ]] && docker ps | grep -q cloudflared; then
    echo -e "${GREEN}🎉 Your media server is fully operational!${NC}"
else
    echo -e "${YELLOW}⚠️  Some core services need attention. Check the details above.${NC}"
fi

echo ""
echo "🔐 Remember: All services are protected by Cloudflare Zero Trust authentication!"
echo "📧 You'll receive PIN codes at: $EMAIL"
